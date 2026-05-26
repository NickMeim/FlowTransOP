from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch

from .inference import FlowTransOPTranslator, load_archs4_translator


def parse_ensemble_ids(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        ids: list[int] = []
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, stop = part.split("-", 1)
                ids.extend(range(int(start), int(stop) + 1))
            else:
                ids.append(int(part))
        return ids
    return [int(v) for v in value]


@dataclass
class ARCHS4EnsembleTranslator:
    translators: list[FlowTransOPTranslator]
    ensemble_ids: list[int]
    archs4_dir: Path

    @torch.no_grad()
    def translate(
        self,
        x: np.ndarray | torch.Tensor,
        direction: str,
        batch_size: int = 256,
        n_steps: int = 10,
        return_members: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        member_outputs = [
            translator.translate(
                x,
                direction=direction,
                batch_size=batch_size,
                n_steps=n_steps,
                return_variance=False,
            )
            for translator in self.translators
        ]
        stack = np.stack(member_outputs, axis=0)
        mean = stack.mean(axis=0)
        if return_members:
            return mean, stack
        return mean

    def translate_file(
        self,
        input_npy: str | Path,
        output_npy: str | Path,
        direction: str,
        batch_size: int = 256,
        n_steps: int = 10,
        return_members: bool = False,
        members_output_npy: str | Path | None = None,
    ) -> Path:
        x = np.load(input_npy)
        y = self.translate(
            x,
            direction=direction,
            batch_size=batch_size,
            n_steps=n_steps,
            return_members=return_members or members_output_npy is not None,
        )
        output_npy = Path(output_npy)
        output_npy.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(y, tuple):
            mean, members = y
            np.save(output_npy, mean)
            if members_output_npy is not None:
                members_output_npy = Path(members_output_npy)
                members_output_npy.parent.mkdir(parents=True, exist_ok=True)
                np.save(members_output_npy, members)
        else:
            np.save(output_npy, y)
        return output_npy


def load_archs4_ensemble(
    archs4_dir: str | Path = "archs4",
    ensemble_ids: str | Iterable[int] = "0-9",
    model_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    require_all: bool = True,
) -> ARCHS4EnsembleTranslator:
    archs4_dir = Path(archs4_dir)
    model_dir = Path(model_dir) if model_dir is not None else archs4_dir / "models"
    ids = parse_ensemble_ids(ensemble_ids)
    translators: list[FlowTransOPTranslator] = []
    missing: list[Path] = []

    for ensemble_id in ids:
        normal = model_dir / f"full_ensemble_{ensemble_id}_normal.pt"
        m2h = model_dir / f"full_ensemble_{ensemble_id}_normal_m2h.pt"
        if not normal.exists():
            missing.append(normal)
            continue
        translators.append(
            load_archs4_translator(
                normal,
                m2h_checkpoint=m2h if m2h.exists() else None,
                device=device,
            )
        )

    if missing and require_all:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing ARCHS4 ensemble checkpoints:\n{missing_text}")
    if not translators:
        raise FileNotFoundError(f"No ARCHS4 ensemble checkpoints found in {model_dir}")

    return ARCHS4EnsembleTranslator(translators=translators, ensemble_ids=ids, archs4_dir=archs4_dir)


def translate_archs4_ensemble_array(
    input_npy: str | Path,
    output_npy: str | Path,
    direction: str,
    archs4_dir: str | Path = "archs4",
    ensemble_ids: str | Iterable[int] = "0-9",
    model_dir: str | Path | None = None,
    device: str | None = None,
    batch_size: int = 256,
    n_steps: int = 10,
    members_output_npy: str | Path | None = None,
) -> Path:
    ensemble = load_archs4_ensemble(
        archs4_dir=archs4_dir,
        ensemble_ids=ensemble_ids,
        model_dir=model_dir,
        device=device,
    )
    return ensemble.translate_file(
        input_npy=input_npy,
        output_npy=output_npy,
        direction=direction,
        batch_size=batch_size,
        n_steps=n_steps,
        members_output_npy=members_output_npy,
    )


def _add_learning_path(repo_root: str | Path) -> Path:
    root = Path(repo_root).resolve()
    learning = root / "learning"
    if str(learning) not in sys.path:
        sys.path.insert(0, str(learning))
    return root


def _activation(train_full: Any, args: dict[str, Any], key: str, default: str):
    value = args.get(key, default)
    if not isinstance(value, str):
        value = default
    return train_full.activation(value)


def _model_params(train_full: Any, ckpt_args: dict[str, Any], epochs: int, batch_size: int) -> dict[str, Any]:
    return {
        "encoder_1_hiddens": list(ckpt_args.get("encoder_1_hiddens", [4096, 2048, 1024, 512])),
        "encoder_2_hiddens": list(ckpt_args.get("encoder_2_hiddens", [4096, 2048, 1024, 512])),
        "latent_dim": int(ckpt_args.get("latent_dim", 512)),
        "decoder_1_hiddens": list(ckpt_args.get("decoder_1_hiddens", [512, 1024, 2048, 4096])),
        "decoder_2_hiddens": list(ckpt_args.get("decoder_2_hiddens", [512, 1024, 2048, 4096])),
        "dropout_decoder": float(ckpt_args.get("dropout_decoder", 0.2)),
        "dropout_encoder": float(ckpt_args.get("dropout_encoder", 0.2)),
        "encoder_activation": _activation(train_full, ckpt_args, "encoder_activation", "ELU"),
        "decoder_activation": _activation(train_full, ckpt_args, "decoder_activation", "ELU"),
        "bn_encoder": float(ckpt_args.get("bn_encoder", 0.6)),
        "bn_decoder": float(ckpt_args.get("bn_decoder", 0.6)),
        "dropout_input_encoder": float(ckpt_args.get("dropout_input_encoder", 0.5)),
        "dropout_input_decoder": float(ckpt_args.get("dropout_input_decoder", 0.2)),
        "encoding_lr": float(ckpt_args.get("encoding_lr", 0.001)),
        "schedule_step_enc": int(ckpt_args.get("schedule_step_enc", 20)),
        "gamma_enc": float(ckpt_args.get("gamma_enc", 0.8)),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "enc_l2_reg": float(ckpt_args.get("enc_l2_reg", 0.001)),
        "dec_l2_reg": float(ckpt_args.get("dec_l2_reg", 0.001)),
        "autoencoder_wd": float(ckpt_args.get("autoencoder_wd", 0.0)),
        "flow_lambda": float(ckpt_args.get("flow_lambda", 1.0)),
        "conditional_flow_lambda": float(ckpt_args.get("conditional_flow_lambda", 1e-3)),
    }


def finetune_archs4_ensemble_member(
    repo_root: str | Path,
    archs4_dir: str | Path = "archs4",
    ensemble_id: int = 0,
    fold: int = 0,
    epochs: int = 5,
    batch_size: int = 4096,
    device: str | None = None,
    include_liver_test: bool = True,
    output_model_dir: str | Path | None = None,
    base_checkpoint: str | Path | None = None,
) -> tuple[Path, Path]:
    """Fine-tune one pretrained full ARCHS4 ensemble member.

    This expects the repository's `learning/` folder plus the gitignored
    `archs4/` folder with `preprocessed/` and pretrained `models/` files.
    """

    root = _add_learning_path(repo_root)
    train_full = importlib.import_module("train_ARCHS4_full_ensemble")

    archs4_dir = Path(archs4_dir)
    if not archs4_dir.is_absolute():
        archs4_dir = root / archs4_dir
    preproc_dir = archs4_dir / "preprocessed"
    input_model_dir = archs4_dir / "models"
    output_model_dir = Path(output_model_dir) if output_model_dir is not None else archs4_dir / "models_finetuned"
    if not output_model_dir.is_absolute():
        output_model_dir = root / output_model_dir
    output_model_dir.mkdir(parents=True, exist_ok=True)

    normal = Path(base_checkpoint) if base_checkpoint is not None else input_model_dir / f"full_ensemble_{ensemble_id}_normal.pt"
    if not normal.is_absolute():
        normal = root / normal
    translator = load_archs4_translator(normal, device=device)
    device_obj = translator.device
    model_params = _model_params(train_full, translator.args, epochs=epochs, batch_size=batch_size)

    x_human = train_full.load_full_species("human", fold, preproc_dir, include_liver_test)
    x_mouse = train_full.load_full_species("mouse", fold, preproc_dir, include_liver_test)

    _, dec_h, enc_h = train_full.train_RNAseq_AE_fold_gauss(
        model_params,
        device_obj,
        x_human,
        translator.decoder_human,
        translator.encoder_human,
        batch_size,
        epochs,
        evaluate=False,
        plot_label=f"finetune_ens{ensemble_id}_human",
    )
    _, dec_m, enc_m = train_full.train_RNAseq_AE_fold_gauss(
        model_params,
        device_obj,
        x_mouse,
        translator.decoder_mouse,
        translator.encoder_mouse,
        batch_size,
        epochs,
        evaluate=False,
        plot_label=f"finetune_ens{ensemble_id}_mouse",
    )

    z_human = train_full.encode_all(enc_h, x_human, device_obj, bs=batch_size).to(device_obj)
    z_mouse = train_full.encode_all(enc_m, x_mouse, device_obj, bs=batch_size).to(device_obj)

    _, flow_h2m = train_full.train_RNAseq_flowMatch_fold(
        model_params,
        device_obj,
        x_human,
        x_mouse,
        z_human,
        z_mouse,
        translator.flow_h2m,
        batch_size,
        batch_size,
        epochs,
        translation_direction="1 to 2",
        plot_label=f"finetune_ens{ensemble_id}_h2m",
    )
    _, flow_m2h = train_full.train_RNAseq_flowMatch_fold(
        model_params,
        device_obj,
        x_human,
        x_mouse,
        z_human,
        z_mouse,
        translator.flow_m2h,
        batch_size,
        batch_size,
        epochs,
        translation_direction="2 to 1",
        plot_label=f"finetune_ens{ensemble_id}_m2h",
    )

    ckpt_args = dict(translator.args)
    ckpt_args.update(
        {
            "fine_tuned_from": str(normal),
            "fine_tune_epochs": int(epochs),
            "fine_tune_batch_size": int(batch_size),
            "fine_tune_fold": int(fold),
        }
    )
    normal_out = output_model_dir / f"full_ensemble_{ensemble_id}_finetuned_normal.pt"
    m2h_out = output_model_dir / f"full_ensemble_{ensemble_id}_finetuned_normal_m2h.pt"
    torch.save(
        {
            "encoder_human": enc_h.state_dict(),
            "encoder_mouse": enc_m.state_dict(),
            "decoder_human": dec_h.state_dict(),
            "decoder_mouse": dec_m.state_dict(),
            "flow_h2m": flow_h2m.state_dict(),
            "flow_m2h": flow_m2h.state_dict(),
            "args": ckpt_args,
        },
        normal_out,
    )
    torch.save({"flow_m2h": flow_m2h.state_dict(), "args": ckpt_args}, m2h_out)
    return normal_out, m2h_out
