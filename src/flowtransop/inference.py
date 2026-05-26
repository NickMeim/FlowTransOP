from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .models import ElementWiseLinear, Flow, SimpleEncoder, VarDecoder, activation_from_name


def _load_checkpoint(path: str | Path, device: str | torch.device) -> dict:
    return torch.load(Path(path), map_location=device, weights_only=False)


def _input_dim(state_dict: dict[str, torch.Tensor], elementwise_key: str = "0.weight") -> int:
    if elementwise_key in state_dict:
        return int(state_dict[elementwise_key].numel())
    first_linear = next(k for k in state_dict if k.endswith("linear_layers.0.weight"))
    return int(state_dict[first_linear].shape[1])


def _args(ckpt: dict) -> dict:
    args = ckpt.get("args", {})
    if not isinstance(args, dict):
        args = vars(args)
    return args


def _build_components(ckpt: dict, device: torch.device):
    args = _args(ckpt)
    dtype = torch.float
    human_dim = _input_dim(ckpt["encoder_human"])
    mouse_dim = _input_dim(ckpt["encoder_mouse"])
    latent_dim = int(args.get("latent_dim", ckpt["flow_h2m"]["net.6.weight"].shape[0]))
    enc_act = activation_from_name(args.get("encoder_activation", "ELU"))
    dec_act = activation_from_name(args.get("decoder_activation", "ELU"))

    enc_h = torch.nn.Sequential(
        ElementWiseLinear(human_dim),
        SimpleEncoder(
            human_dim,
            list(args.get("encoder_1_hiddens", [4096, 2048, 1024, 512])),
            latent_dim,
            dropRate=float(args.get("dropout_encoder", 0.2)),
            bn=float(args.get("bn_encoder", 0.6)),
            activation=enc_act,
            dropIn=float(args.get("dropout_input_encoder", 0.5)),
            dtype=dtype,
        ),
    ).to(device)
    enc_m = torch.nn.Sequential(
        ElementWiseLinear(mouse_dim),
        SimpleEncoder(
            mouse_dim,
            list(args.get("encoder_2_hiddens", [4096, 2048, 1024, 512])),
            latent_dim,
            dropRate=float(args.get("dropout_encoder", 0.2)),
            bn=float(args.get("bn_encoder", 0.6)),
            activation=enc_act,
            dropIn=float(args.get("dropout_input_encoder", 0.5)),
            dtype=dtype,
        ),
    ).to(device)
    dec_h = VarDecoder(
        latent_dim,
        list(args.get("decoder_1_hiddens", [512, 1024, 2048, 4096])),
        human_dim,
        dropRate=float(args.get("dropout_decoder", 0.2)),
        bn=float(args.get("bn_decoder", 0.6)),
        activation=dec_act,
        dropIn=float(args.get("dropout_input_decoder", 0.2)),
        loss="gauss",
        dtype=dtype,
    ).to(device)
    dec_m = VarDecoder(
        latent_dim,
        list(args.get("decoder_2_hiddens", [512, 1024, 2048, 4096])),
        mouse_dim,
        dropRate=float(args.get("dropout_decoder", 0.2)),
        bn=float(args.get("bn_decoder", 0.6)),
        activation=dec_act,
        dropIn=float(args.get("dropout_input_decoder", 0.2)),
        loss="gauss",
        dtype=dtype,
    ).to(device)
    flow_h2m = Flow(latent_dim, latent_dim // 2, dtype=dtype).to(device)
    flow_m2h = Flow(latent_dim, latent_dim // 2, dtype=dtype).to(device)
    enc_h.load_state_dict(ckpt["encoder_human"])
    enc_m.load_state_dict(ckpt["encoder_mouse"])
    dec_h.load_state_dict(ckpt["decoder_human"])
    dec_m.load_state_dict(ckpt["decoder_mouse"])
    flow_h2m.load_state_dict(ckpt["flow_h2m"])
    if "flow_m2h" in ckpt:
        flow_m2h.load_state_dict(ckpt["flow_m2h"])
    for module in [enc_h, enc_m, dec_h, dec_m, flow_h2m, flow_m2h]:
        module.eval()
    return enc_h, enc_m, dec_h, dec_m, flow_h2m, flow_m2h, args


def flow_step_n(flow: Flow, z: torch.Tensor, n_steps: int = 10) -> torch.Tensor:
    device = z.device
    time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=z.dtype)
    out = z.clone()
    for step in range(n_steps):
        out = flow.step(out, time_steps[step], time_steps[step + 1])
    return out


@dataclass
class FlowTransOPTranslator:
    encoder_human: torch.nn.Module
    encoder_mouse: torch.nn.Module
    decoder_human: torch.nn.Module
    decoder_mouse: torch.nn.Module
    flow_h2m: Flow
    flow_m2h: Flow
    args: dict
    device: torch.device

    @torch.no_grad()
    def translate(
        self,
        x: np.ndarray | torch.Tensor,
        direction: str,
        batch_size: int = 256,
        n_steps: int = 10,
        return_variance: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        direction = direction.lower()
        if direction not in {"h2m", "human-to-mouse", "m2h", "mouse-to-human"}:
            raise ValueError("direction must be h2m/human-to-mouse or m2h/mouse-to-human")
        arr = torch.as_tensor(x, dtype=torch.float32)
        means, variances = [], []
        for start in range(0, arr.shape[0], batch_size):
            block = arr[start : start + batch_size].to(self.device)
            if direction in {"h2m", "human-to-mouse"}:
                z = self.encoder_human(block)
                z_t = flow_step_n(self.flow_h2m, z, n_steps=n_steps)
                mu, var = self.decoder_mouse(z_t)
            else:
                z = self.encoder_mouse(block)
                z_t = flow_step_n(self.flow_m2h, z, n_steps=n_steps)
                mu, var = self.decoder_human(z_t)
            means.append(mu.cpu())
            variances.append(var.cpu())
        mean_np = torch.cat(means, dim=0).numpy()
        if not return_variance:
            return mean_np
        return mean_np, torch.cat(variances, dim=0).numpy()


def load_archs4_translator(
    normal_checkpoint: str | Path,
    m2h_checkpoint: str | Path | None = None,
    device: str | torch.device | None = None,
) -> FlowTransOPTranslator:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = _load_checkpoint(normal_checkpoint, device)
    enc_h, enc_m, dec_h, dec_m, flow_h2m, flow_m2h, args = _build_components(ckpt, device)
    if m2h_checkpoint is not None and "flow_m2h" not in ckpt:
        m2h = _load_checkpoint(m2h_checkpoint, device)
        flow_m2h.load_state_dict(m2h["flow_m2h"])
        flow_m2h.eval()
    elif "flow_m2h" not in ckpt:
        raise ValueError("The normal checkpoint does not contain flow_m2h; provide m2h_checkpoint.")
    return FlowTransOPTranslator(enc_h, enc_m, dec_h, dec_m, flow_h2m, flow_m2h, args, device)


def translate_array(
    input_npy: str | Path,
    output_npy: str | Path,
    normal_checkpoint: str | Path,
    direction: str,
    m2h_checkpoint: str | Path | None = None,
    device: str | None = None,
    batch_size: int = 256,
    n_steps: int = 10,
) -> Path:
    translator = load_archs4_translator(normal_checkpoint, m2h_checkpoint=m2h_checkpoint, device=device)
    x = np.load(input_npy)
    y = translator.translate(x, direction=direction, batch_size=batch_size, n_steps=n_steps)
    output_npy = Path(output_npy)
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, y)
    return output_npy
