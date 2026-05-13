#!/usr/bin/env python
"""
Train/evaluate an AutoTransOP/CPA-style model on the same extremely-low-paired
A375_HT29 splits used by FlowTransOP in cases with extemely few pairs.

This script keeps the few-pairs experimental design from the FlowTransOP script:
  - use the full A375_HT29 5-fold split folder under CellPairs/
  - find the largest sample_len subfolder
  - for each n_pairs, fold, and repeat, choose disjoint subsets of train_paired
  - train with the selected paired samples plus all unpaired samples
  - evaluate translation on all validation pairs

The model/training objective follows CPA_PerformaceAnalysis_lands.py:
  - two encoders/decoders
  - shared base latent space + SpeciesCovariate
  - local MI discriminator
  - prior discriminator
  - species classifier and adversarial classifier
  - paired latent similarity + cosine alignment penalties

Run from the OmicTranslationBenchmark/learning directory, or adjust paths.
"""

from __future__ import absolute_import, division

import argparse
import json
import logging
import math
import os
import random
import warnings
from logging import FileHandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix

from evaluationUtils import pearson_r, pseudoAccuracy, r_square
from models_autotransop import (
    Classifier,
    Decoder,
    LocalDiscriminator,
    PriorDiscriminator,
    SimpleEncoder,
    SpeciesCovariate,
)
from utility import find_largest_sample_len

warnings.filterwarnings("ignore", message=".*ks_2samp.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def getSamples(N: int, batchSize: int) -> List[np.ndarray]:
    """Original CPA mini-batch sampler, with guards for tiny N."""
    if N <= 0:
        return []
    batchSize = max(1, min(int(batchSize), int(N)))
    order = np.random.permutation(N)
    outList = []
    while len(order) > 0:
        outList.append(order[0:batchSize])
        order = order[batchSize:]
    return outList


def extend_loader_to_length(loader: List[np.ndarray], N: int, batchSize: int, target_len: int) -> List[np.ndarray]:
    """Repeat freshly sampled loaders until target_len is reached.

    This fixes the original CPA pattern for extreme few-pair runs, where
    maxLen - len_paired can be much larger than one extra sampled loader.
    """
    if target_len <= 0:
        return []
    if len(loader) == 0:
        raise ValueError("Cannot extend an empty loader; check that all train sets are non-empty.")
    out = list(loader)
    while len(out) < target_len:
        out.extend(getSamples(N, batchSize))
    return out[:target_len]


def compute_gradients(output: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
    grads = torch.autograd.grad(output, input_tensor, create_graph=True)[0]
    return grads.pow(2).mean()


def safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps, max=1.0 - eps))


def to_float(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def activation_from_name(name: str) -> torch.nn.Module:
    if name == "LeakyReLU":
        return torch.nn.LeakyReLU(0.01)
    if name == "ReLU":
        return torch.nn.ReLU()
    if name == "ELU":
        return torch.nn.ELU()
    if name == "Sigmoid":
        return torch.nn.Sigmoid()
    raise ValueError(f"Unknown activation: {name}")


def _mkdir_for_file(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        Path(parent).mkdir(parents=True, exist_ok=True)


def _load_or_empty_csv(path: str) -> pd.DataFrame:
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def _atomic_to_csv(df: pd.DataFrame, path: str, **to_csv_kwargs) -> None:
    _mkdir_for_file(path)
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False, **to_csv_kwargs)
    os.replace(tmp, path)


def _append_and_write_safely(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
    out_path: str,
    dedup_subset: Optional[List[str]] = None,
) -> pd.DataFrame:
    if new_df is None or new_df.empty:
        if not existing_df.empty and not os.path.exists(out_path):
            _atomic_to_csv(existing_df, out_path)
        return existing_df
    merged = pd.concat([existing_df, new_df], ignore_index=True)
    if dedup_subset is not None and all(c in merged.columns for c in dedup_subset):
        merged = merged.drop_duplicates(subset=dedup_subset, keep="last")
    else:
        merged = merged.drop_duplicates(keep="last")
    _atomic_to_csv(merged, out_path)
    return merged


def save_progress(path: str, n_pairs_idx: int, fold_id: int, repeat: int) -> None:
    state = {"last_completed": {"n_pairs_idx": n_pairs_idx, "fold_id": fold_id, "repeat": repeat}}
    _mkdir_for_file(path)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)


def load_progress(path: str) -> Optional[dict]:
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                return json.load(f)
            except Exception:
                return None
    return None


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / max(1, y_pred.size)
    denom = 2 * tp + fp + fn
    f1 = np.nan if denom == 0 else (2 * tp / denom)
    return float(f1), float(acc)


def infer_batch_size_paired(n_pairs: int) -> int:
    """Keep the FlowTransOP low-pair batching convention."""
    if n_pairs <= 1:
        return 1
    if n_pairs <= 5:
        return 2
    if n_pairs <= 15:
        return 5
    if n_pairs <= 30:
        return 10
    if n_pairs <= 100:
        return 50
    return 90


def condition_values(df: pd.DataFrame) -> np.ndarray:
    for col in ("conditionId", "condition_id", "condition", "pert_idose", "sig_id"):
        if col in df.columns:
            return df[col].values
    raise ValueError(
        "Could not find a condition column. Expected one of: "
        "conditionId, condition_id, condition, pert_idose, sig_id."
    )


def make_species_onehot(n: int, species: int, device: torch.device) -> torch.Tensor:
    if species == 1:
        return torch.cat((torch.ones(n, 1), torch.zeros(n, 1)), dim=1).to(device)
    if species == 2:
        return torch.cat((torch.zeros(n, 1), torch.ones(n, 1)), dim=1).to(device)
    raise ValueError("species must be 1 or 2")


# ---------------------------------------------------------------------------
# Metrics/evaluation helpers
# ---------------------------------------------------------------------------

def mean_spearman(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    true_np = y_true.detach().cpu().numpy()
    pred_np = y_pred.detach().cpu().numpy()
    rhos = []
    for i in range(true_np.shape[0]):
        rho, _ = spearmanr(true_np[i, :], pred_np[i, :])
        rhos.append(rho)
    return float(np.nanmean(rhos))


def metric_bundle(y_true: torch.Tensor, y_pred: torch.Tensor, prefix: str = "") -> Dict[str, float]:
    p = f"{prefix}_" if prefix else ""
    out = {
        f"{p}r2": to_float(r_square(y_pred.detach().flatten(), y_true.detach().flatten())),
        f"{p}pearson": to_float(pearson_r(y_pred.detach().flatten(), y_true.detach().flatten())),
        f"{p}mse": to_float(torch.mean(torch.mean((y_pred.detach() - y_true.detach()) ** 2, dim=1))),
        f"{p}spearman": mean_spearman(y_true.detach(), y_pred.detach()),
        f"{p}pseudo_acc": float(np.nanmean(pseudoAccuracy(y_true.detach().cpu(), y_pred.detach().cpu(), eps=1e-6))),
    }
    return out


def evaluate_reconstruction(
    x_1: torch.Tensor,
    x_2: torch.Tensor,
    encoder_1: SimpleEncoder,
    encoder_2: SimpleEncoder,
    decoder_1: Decoder,
    decoder_2: Decoder,
    Vsp: SpeciesCovariate,
    device: torch.device,
) -> Dict[str, float]:
    with torch.no_grad():
        z_species_1 = make_species_onehot(x_1.shape[0], 1, device)
        z_species_2 = make_species_onehot(x_2.shape[0], 2, device)
        z_1 = Vsp(encoder_1(x_1), z_species_1)
        z_2 = Vsp(encoder_2(x_2), z_species_2)
        xhat_1 = decoder_1(z_1)
        xhat_2 = decoder_2(z_2)

    m1 = metric_bundle(x_1, xhat_1, prefix="A375")
    m2 = metric_bundle(x_2, xhat_2, prefix="HT29")
    return {
        **m1,
        **m2,
        "mean_pearson": 0.5 * (m1["A375_pearson"] + m2["HT29_pearson"]),
        "mean_spearman": 0.5 * (m1["A375_spearman"] + m2["HT29_spearman"]),
        "mean_pseudo_acc": 0.5 * (m1["A375_pseudo_acc"] + m2["HT29_pseudo_acc"]),
    }


def evaluate_translation_pairs(
    x_1_pairs: torch.Tensor,
    x_2_pairs: torch.Tensor,
    encoder_1: SimpleEncoder,
    encoder_2: SimpleEncoder,
    decoder_1: Decoder,
    decoder_2: Decoder,
    Vsp: SpeciesCovariate,
    device: torch.device,
) -> Dict[str, float]:
    """CPA translation: encode in native context, add opposite species covariate, decode in target."""
    with torch.no_grad():
        z_species_1 = make_species_onehot(x_1_pairs.shape[0], 1, device)
        z_species_2 = make_species_onehot(x_2_pairs.shape[0], 2, device)

        z_latent_1_as_2 = Vsp(encoder_1(x_1_pairs), 1.0 - z_species_1)
        xhat_2_from_1 = decoder_2(z_latent_1_as_2)

        z_latent_2_as_1 = Vsp(encoder_2(x_2_pairs), 1.0 - z_species_2)
        xhat_1_from_2 = decoder_1(z_latent_2_as_1)

    m12 = metric_bundle(x_2_pairs, xhat_2_from_1, prefix="A375_to_HT29")
    m21 = metric_bundle(x_1_pairs, xhat_1_from_2, prefix="HT29_to_A375")

    direct = metric_bundle(x_2_pairs, x_1_pairs, prefix="direct_A375_as_HT29")
    direct_rev = metric_bundle(x_1_pairs, x_2_pairs, prefix="direct_HT29_as_A375")

    return {
        **m12,
        **m21,
        **direct,
        **direct_rev,
        "mean_pearson": 0.5 * (m12["A375_to_HT29_pearson"] + m21["HT29_to_A375_pearson"]),
        "mean_spearman": 0.5 * (m12["A375_to_HT29_spearman"] + m21["HT29_to_A375_spearman"]),
        "mean_pseudo_acc": 0.5 * (m12["A375_to_HT29_pseudo_acc"] + m21["HT29_to_A375_pseudo_acc"]),
        "direct_mean_pearson": 0.5 * (
            direct["direct_A375_as_HT29_pearson"] + direct_rev["direct_HT29_as_A375_pearson"]
        ),
        "direct_mean_spearman": 0.5 * (
            direct["direct_A375_as_HT29_spearman"] + direct_rev["direct_HT29_as_A375_spearman"]
        ),
        "direct_mean_pseudo_acc": 0.5 * (
            direct["direct_A375_as_HT29_pseudo_acc"] + direct_rev["direct_HT29_as_A375_pseudo_acc"]
        ),
    }


def evaluate_classifier(
    x_1: torch.Tensor,
    x_2: torch.Tensor,
    encoder_1: SimpleEncoder,
    encoder_2: SimpleEncoder,
    classifier: Classifier,
    Vsp: SpeciesCovariate,
    device: torch.device,
) -> Dict[str, float]:
    with torch.no_grad():
        z_species_1 = make_species_onehot(x_1.shape[0], 1, device)
        z_species_2 = make_species_onehot(x_2.shape[0], 2, device)
        z_1 = Vsp(encoder_1(x_1), z_species_1)
        z_2 = Vsp(encoder_2(x_2), z_species_2)
        labels = classifier(torch.cat((z_1, z_2), dim=0))
        y_true = torch.cat(
            (torch.ones(z_1.shape[0]), torch.zeros(z_2.shape[0])), dim=0
        ).long().detach().cpu().numpy()
        y_pred = torch.argmax(labels, dim=1).detach().cpu().numpy()
    f1, acc = binary_metrics(y_true, y_pred)
    return {"class_f1": f1, "class_acc": acc}


# ---------------------------------------------------------------------------
# Model construction and one-run training
# ---------------------------------------------------------------------------

def initialize_models(
    model_params: dict,
    gene_size: int,
    device: torch.device,
    pretrained_adv_state: Optional[dict] = None,
):
    decoder_1 = Decoder(
        model_params["latent_dim"],
        model_params["decoder_1_hiddens"],
        gene_size,
        dropRate=model_params["dropout_decoder"],
        activation=model_params["decoder_activation"],
    ).to(device)
    decoder_2 = Decoder(
        model_params["latent_dim"],
        model_params["decoder_2_hiddens"],
        gene_size,
        dropRate=model_params["dropout_decoder"],
        activation=model_params["decoder_activation"],
    ).to(device)
    encoder_1 = SimpleEncoder(
        gene_size,
        model_params["encoder_1_hiddens"],
        model_params["latent_dim"],
        dropRate=model_params["dropout_encoder"],
        activation=model_params["encoder_activation"],
    ).to(device)
    encoder_2 = SimpleEncoder(
        gene_size,
        model_params["encoder_2_hiddens"],
        model_params["latent_dim"],
        dropRate=model_params["dropout_encoder"],
        activation=model_params["encoder_activation"],
    ).to(device)
    prior_d = PriorDiscriminator(model_params["latent_dim"]).to(device)
    local_d = LocalDiscriminator(model_params["latent_dim"], model_params["latent_dim"]).to(device)

    classifier = Classifier(
        in_channel=model_params["latent_dim"],
        hidden_layers=model_params["state_class_hidden"],
        num_classes=model_params["no_states"],
        drop_in=model_params["state_class_drop_in"],
        drop=model_params["state_class_drop"],
    ).to(device)
    adverse_classifier = Classifier(
        in_channel=model_params["latent_dim"],
        hidden_layers=model_params["adv_class_hidden"],
        num_classes=model_params["no_adv_class"],
        drop_in=model_params["adv_class_drop_in"],
        drop=model_params["adv_class_drop"],
    ).to(device)

    if pretrained_adv_state is not None:
        adverse_classifier.load_state_dict(pretrained_adv_state)

    Vsp = SpeciesCovariate(2, model_params["latent_dim"], dropRate=model_params["V_dropout"]).to(device)
    return decoder_1, decoder_2, encoder_1, encoder_2, prior_d, local_d, classifier, adverse_classifier, Vsp


def load_pretrained_adv_state(path: str, device: torch.device, print2log) -> Optional[dict]:
    if not path:
        print2log("[Info] No pretrained adversarial classifier path was provided; using random initialization.")
        return None
    if not os.path.exists(path):
        print2log(f"[Warning] Pretrained adversarial classifier not found: {path}. Using random initialization.")
        return None
    try:
        try:
            obj = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            obj = torch.load(path, map_location=device)
        if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
            # Either a state_dict or a checkpoint containing one.
            if "state_dict" in obj and isinstance(obj["state_dict"], dict):
                state = obj["state_dict"]
            else:
                state = obj
        else:
            state = obj.state_dict()
        print2log(f"[Info] Loaded pretrained adversarial classifier from {path}")
        return state
    except Exception as exc:
        print2log(f"[Warning] Could not load pretrained adversarial classifier: {exc}. Using random initialization.")
        return None


def train_autotransop_one_run(
    model_params: dict,
    device: torch.device,
    cmap: pd.DataFrame,
    trainInfo_paired: pd.DataFrame,
    trainInfo_1: pd.DataFrame,
    trainInfo_2: pd.DataFrame,
    pretrained_adv_state: Optional[dict],
    print2log,
    fold_id: int,
    n_pairs: int,
    rep: int,
):
    gene_size = len(cmap.columns)
    class_criterion = torch.nn.CrossEntropyLoss()
    log_2 = math.log(2.0)

    (
        decoder_1,
        decoder_2,
        encoder_1,
        encoder_2,
        prior_d,
        local_d,
        classifier,
        adverse_classifier,
        Vsp,
    ) = initialize_models(model_params, gene_size, device, pretrained_adv_state)

    allParams = list(decoder_1.parameters()) + list(decoder_2.parameters())
    allParams += list(encoder_1.parameters()) + list(encoder_2.parameters())
    allParams += list(prior_d.parameters()) + list(local_d.parameters())
    allParams += list(classifier.parameters()) + list(Vsp.parameters())

    optimizer = torch.optim.Adam(
        allParams,
        lr=model_params["encoding_lr"],
        weight_decay=model_params.get("autoencoder_wd", 0.0),
    )
    optimizer_adv = torch.optim.Adam(
        adverse_classifier.parameters(),
        lr=model_params["adv_lr"],
        weight_decay=model_params.get("adversary_wd", 0.0),
    )

    scheduler_adv = None
    if model_params["schedule_step_adv"] is not None and model_params["schedule_step_adv"] > 0:
        scheduler_adv = torch.optim.lr_scheduler.StepLR(
            optimizer_adv,
            step_size=model_params["schedule_step_adv"],
            gamma=model_params["gamma_adv"],
        )
    scheduler = None
    if model_params["schedule_step_enc"] is not None and model_params["schedule_step_enc"] > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=model_params["schedule_step_enc"],
            gamma=model_params["gamma_enc"],
        )

    N_paired = len(trainInfo_paired)
    N_1 = len(trainInfo_1)
    N_2 = len(trainInfo_2)
    if N_paired <= 0 or N_1 <= 0 or N_2 <= 0:
        raise ValueError(f"Empty training set: N_paired={N_paired}, N_1={N_1}, N_2={N_2}")

    last_metrics = {}
    for e in range(model_params["epochs"]):
        decoder_1.train(); decoder_2.train()
        encoder_1.train(); encoder_2.train()
        prior_d.train(); local_d.train()
        classifier.train(); adverse_classifier.train(); Vsp.train()

        trainloader_1 = getSamples(N_1, model_params["batch_size_1"])
        trainloader_2 = getSamples(N_2, model_params["batch_size_2"])
        trainloader_paired = getSamples(N_paired, model_params["batch_size_paired"])

        maxLen = max(len(trainloader_1), len(trainloader_2), len(trainloader_paired))
        trainloader_1 = extend_loader_to_length(trainloader_1, N_1, model_params["batch_size_1"], maxLen)
        trainloader_2 = extend_loader_to_length(trainloader_2, N_2, model_params["batch_size_2"], maxLen)
        trainloader_paired = extend_loader_to_length(
            trainloader_paired, N_paired, model_params["batch_size_paired"], maxLen
        )

        for j in range(maxLen):
            df_pairs = trainInfo_paired.iloc[trainloader_paired[j], :]
            df_1 = trainInfo_1.iloc[trainloader_1[j], :]
            df_2 = trainInfo_2.iloc[trainloader_2[j], :]
            paired_inds = len(df_pairs)

            X_1 = torch.tensor(
                np.concatenate((cmap.loc[df_pairs["sig_id.x"]].values, cmap.loc[df_1.sig_id].values)),
                dtype=torch.float32,
            ).to(device)
            X_2 = torch.tensor(
                np.concatenate((cmap.loc[df_pairs["sig_id.y"]].values, cmap.loc[df_2.sig_id].values)),
                dtype=torch.float32,
            ).to(device)

            z_species_1 = make_species_onehot(X_1.shape[0], 1, device)
            z_species_2 = make_species_onehot(X_2.shape[0], 2, device)

            conditions = np.concatenate(
                (
                    condition_values(df_pairs),
                    condition_values(df_1),
                    condition_values(df_pairs),
                    condition_values(df_2),
                )
            )
            conditions = conditions.reshape(conditions.size, 1)
            mask = torch.tensor((conditions == conditions.T).astype(np.float32), device=device).detach()
            pos_mask = mask
            neg_mask = 1.0 - mask

            # --------------------------- adversary step ---------------------------
            optimizer.zero_grad(set_to_none=True)
            optimizer_adv.zero_grad(set_to_none=True)

            z_base_1 = encoder_1(X_1)
            z_base_2 = encoder_2(X_2)
            latent_base_vectors = torch.cat((z_base_1, z_base_2), 0)
            labels_adv = adverse_classifier(latent_base_vectors)
            true_labels_adv = torch.cat(
                (torch.ones(z_base_1.shape[0]), torch.zeros(z_base_2.shape[0])), dim=0
            ).long().to(device)
            adv_entropy_for_adv = class_criterion(labels_adv, true_labels_adv)
            adversary_drugs_penalty = compute_gradients(labels_adv.sum(), latent_base_vectors)
            loss_adv = adv_entropy_for_adv + model_params["adv_penalnty"] * adversary_drugs_penalty
            loss_adv.backward()
            optimizer_adv.step()

            adv_pred = torch.argmax(labels_adv.detach(), dim=1).cpu().numpy()
            f1_basal_trained, _ = binary_metrics(true_labels_adv.detach().cpu().numpy(), adv_pred)

            # ----------------------------- encoder step ---------------------------
            optimizer.zero_grad(set_to_none=True)

            z_base_1 = encoder_1(X_1)
            z_base_2 = encoder_2(X_2)
            latent_base_vectors = torch.cat((z_base_1, z_base_2), 0)

            z_un = local_d(latent_base_vectors)
            res_un = torch.matmul(z_un, z_un.t())

            z_1 = Vsp(z_base_1, z_species_1)
            z_2 = Vsp(z_base_2, z_species_2)
            latent_vectors = torch.cat((z_1, z_2), 0)

            y_pred_1 = decoder_1(z_1)
            fitLoss_1 = torch.mean(torch.sum((y_pred_1 - X_1) ** 2, dim=1))
            L2Loss_1 = decoder_1.L2Regularization(model_params["dec_l2_reg"]) + encoder_1.L2Regularization(
                model_params["enc_l2_reg"]
            )
            loss_1 = fitLoss_1 + L2Loss_1

            y_pred_2 = decoder_2(z_2)
            fitLoss_2 = torch.mean(torch.sum((y_pred_2 - X_2) ** 2, dim=1))
            L2Loss_2 = decoder_2.L2Regularization(model_params["dec_l2_reg"]) + encoder_2.L2Regularization(
                model_params["enc_l2_reg"]
            )
            loss_2 = fitLoss_2 + L2Loss_2

            silimalityLoss = torch.mean(torch.sum((z_base_1[:paired_inds, :] - z_base_2[:paired_inds, :]) ** 2, dim=-1))
            cosineLoss = F.cosine_similarity(z_base_1[:paired_inds, :], z_base_2[:paired_inds, :], dim=-1).mean()

            p_samples = res_un * pos_mask.float()
            q_samples = res_un * neg_mask.float()
            Ep = log_2 - F.softplus(-p_samples)
            Eq = F.softplus(-q_samples) + q_samples - log_2
            Ep = (Ep * pos_mask.float()).sum() / torch.clamp(pos_mask.float().sum(), min=1.0)
            Eq = (Eq * neg_mask.float()).sum() / torch.clamp(neg_mask.float().sum(), min=1.0)
            mi_loss = Eq - Ep

            prior = torch.rand_like(latent_base_vectors)
            term_a = safe_log(prior_d(prior)).mean()
            term_b = safe_log(1.0 - prior_d(latent_base_vectors)).mean()
            prior_loss = -(term_a + term_b) * model_params["prior_beta"]

            labels = classifier(latent_vectors)
            true_labels = torch.cat((torch.ones(z_1.shape[0]), torch.zeros(z_2.shape[0])), dim=0).long().to(device)
            entropy = class_criterion(labels, true_labels)
            pred_latent = torch.argmax(labels.detach(), dim=1).cpu().numpy()
            f1_latent, _ = binary_metrics(true_labels.detach().cpu().numpy(), pred_latent)

            labels_adv_for_enc = adverse_classifier(latent_base_vectors)
            adv_entropy = class_criterion(labels_adv_for_enc, true_labels_adv)
            pred_basal = torch.argmax(labels_adv_for_enc.detach(), dim=1).cpu().numpy()
            f1_basal, _ = binary_metrics(true_labels_adv.detach().cpu().numpy(), pred_basal)

            loss = (
                loss_1
                + loss_2
                + model_params["similarity_reg"] * silimalityLoss
                + model_params["lambda_mi_loss"] * mi_loss
                + prior_loss
                + model_params["reg_classifier"] * entropy
                - model_params["reg_adv"] * adv_entropy
                + classifier.L2Regularization(model_params["state_class_reg"])
                + Vsp.Regularization(model_params["v_reg"])
                - model_params["cosine_loss"] * cosineLoss
            )

            loss.backward()
            optimizer.step()

            last_metrics = {
                "r2_1": to_float(r_square(y_pred_1.detach().flatten(), X_1.detach().flatten())),
                "pearson_1": to_float(pearson_r(y_pred_1.detach().flatten(), X_1.detach().flatten())),
                "mse_1": to_float(torch.mean(torch.mean((y_pred_1.detach() - X_1.detach()) ** 2, dim=1))),
                "r2_2": to_float(r_square(y_pred_2.detach().flatten(), X_2.detach().flatten())),
                "pearson_2": to_float(pearson_r(y_pred_2.detach().flatten(), X_2.detach().flatten())),
                "mse_2": to_float(torch.mean(torch.mean((y_pred_2.detach() - X_2.detach()) ** 2, dim=1))),
                "mi_loss": to_float(mi_loss),
                "prior_loss": to_float(prior_loss),
                "entropy_loss": to_float(entropy),
                "adv_entropy": to_float(adv_entropy),
                "cosine_loss": to_float(cosineLoss),
                "loss": to_float(loss),
                "f1_latent": float(f1_latent),
                "f1_basal": float(f1_basal),
                "f1_basal_trained": float(f1_basal_trained),
            }

        if scheduler_adv is not None:
            scheduler_adv.step()
        if scheduler is not None:
            scheduler.step()

        if e == 0 or ((e + 1) % model_params["print_every"] == 0) or (e + 1 == model_params["epochs"]):
            print2log(
                "n_pairs={n_pairs} fold={fold_id} rep={rep} epoch={epoch}/{epochs}, "
                "r1={r1:.4f}, r2={r2:.4f}, mi={mi:.4f}, prior={prior:.4f}, "
                "entropy={entropy:.4f}, adv_entropy={adv:.4f}, loss={loss:.4f}, "
                "F1 latent={f1_latent:.4f}, F1 basal={f1_basal:.4f}, F1 basal trained={f1_bt:.4f}".format(
                    n_pairs=n_pairs,
                    fold_id=fold_id,
                    rep=rep,
                    epoch=e + 1,
                    epochs=model_params["epochs"],
                    r1=last_metrics.get("pearson_1", np.nan),
                    r2=last_metrics.get("pearson_2", np.nan),
                    mi=last_metrics.get("mi_loss", np.nan),
                    prior=last_metrics.get("prior_loss", np.nan),
                    entropy=last_metrics.get("entropy_loss", np.nan),
                    adv=last_metrics.get("adv_entropy", np.nan),
                    loss=last_metrics.get("loss", np.nan),
                    f1_latent=last_metrics.get("f1_latent", np.nan),
                    f1_basal=last_metrics.get("f1_basal", np.nan),
                    f1_bt=last_metrics.get("f1_basal_trained", np.nan),
                )
            )

    return decoder_1, decoder_2, encoder_1, encoder_2, prior_d, local_d, classifier, adverse_classifier, Vsp


def save_models_at_rep_end(
    ckpt_dir: str,
    tag: str,
    fold_id: int,
    encoder_1,
    decoder_1,
    encoder_2,
    decoder_2,
    prior_d,
    local_d,
    classifier,
    adverse_classifier,
    Vsp,
    extra: Optional[dict] = None,
) -> None:
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(ckpt_dir, f"fold{fold_id}__{tag}.pt")
    payload = {
        "encoder_1": encoder_1.state_dict() if encoder_1 is not None else None,
        "decoder_1": decoder_1.state_dict() if decoder_1 is not None else None,
        "encoder_2": encoder_2.state_dict() if encoder_2 is not None else None,
        "decoder_2": decoder_2.state_dict() if decoder_2 is not None else None,
        "prior_d": prior_d.state_dict() if prior_d is not None else None,
        "local_d": local_d.state_dict() if local_d is not None else None,
        "classifier": classifier.state_dict() if classifier is not None else None,
        "adverse_classifier": adverse_classifier.state_dict() if adverse_classifier is not None else None,
        "Vsp": Vsp.state_dict() if Vsp is not None else None,
        "extra": extra or {},
    }
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "AutoTransOP/CPA training on A375_HT29 using the full 5-fold split "
            "but an extremely small number of paired samples per repeat."
        )
    )

    # Checkpointing / logging
    parser.add_argument("--checkpoint_dir", type=str, default="./chkpts_AutoTransOP_lowPairsPercentage_extreme")
    parser.add_argument("--log_file", type=str, default="logs/AutoTransOP_lowPairsPercentage_extreme.log")
    parser.add_argument("--resume", type=int, default=1, help="1 to resume if checkpoint exists")
    parser.add_argument("--save_models", type=int, default=1, help="1 to save model checkpoints per run")

    # Few-pairs experimental design
    parser.add_argument("--n_pairs_list", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--n_repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)

    # Data and output paths
    parser.add_argument("--data_root", type=str, default="../preprocessing/preprocessed_data/CellPairs/")
    parser.add_argument("--cmap_file", type=str, default="../preprocessing/preprocessed_data/CellPairs/drug_landmarks.csv")
    parser.add_argument("--output_dir", type=str, default="../results/AutoTransOP_extremely_fewPairs_A375_HT29/")
    parser.add_argument("--dataset1", type=str, default="A375")
    parser.add_argument("--dataset2", type=str, default="HT29")
    parser.add_argument("--no_folds", type=int, default=5, help="A375_HT29 full split has 5 folds")
    parser.add_argument(
        "--pretrained_adv_class",
        type=str,
        default="../../TranslationalModels/OmicTranslationBenchmark/preprocessing/preprocessed_data/sampledDatasetes/A375_HT29/pre_trained_classifier_adverse_1_lands.pt",
        help="Optional pretrained adverse classifier used by CPA_PerformaceAnalysis_lands.py.",
    )

    # CPA_PerformaceAnalysis_lands.py hyperparameters
    parser.add_argument("--encoder_1_hiddens", type=int, nargs="+", default=[640, 384])
    parser.add_argument("--encoder_2_hiddens", type=int, nargs="+", default=[640, 384])
    parser.add_argument("--latent_dim", type=int, default=292)
    parser.add_argument("--decoder_1_hiddens", type=int, nargs="+", default=[384, 640])
    parser.add_argument("--decoder_2_hiddens", type=int, nargs="+", default=[384, 640])
    parser.add_argument("--dropout_decoder", type=float, default=0.2)
    parser.add_argument("--dropout_encoder", type=float, default=0.1)
    parser.add_argument("--encoder_activation", type=str, choices=["LeakyReLU", "ReLU", "ELU", "Sigmoid"], default="ELU")
    parser.add_argument("--decoder_activation", type=str, choices=["LeakyReLU", "ReLU", "ELU", "Sigmoid"], default="ELU")
    parser.add_argument("--V_dropout", type=float, default=0.25)
    parser.add_argument("--state_class_hidden", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--state_class_drop_in", type=float, default=0.5)
    parser.add_argument("--state_class_drop", type=float, default=0.25)
    parser.add_argument("--no_states", type=int, default=2)
    parser.add_argument("--adv_class_hidden", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--adv_class_drop_in", type=float, default=0.3)
    parser.add_argument("--adv_class_drop", type=float, default=0.1)
    parser.add_argument("--no_adv_class", type=int, default=2)
    parser.add_argument("--encoding_lr", type=float, default=0.001)
    parser.add_argument("--adv_lr", type=float, default=0.001)
    parser.add_argument("--schedule_step_adv", type=int, default=200)
    parser.add_argument("--gamma_adv", type=float, default=0.5)
    parser.add_argument("--schedule_step_enc", type=int, default=200)
    parser.add_argument("--gamma_enc", type=float, default=0.8)
    parser.add_argument("--batch_size_1", type=int, default=178)
    parser.add_argument("--batch_size_2", type=int, default=154)
    parser.add_argument("--batch_size_paired", type=int, default=0, help="0 = infer from n_pairs using low-pair convention")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--prior_beta", type=float, default=1.0)
    parser.add_argument("--v_reg", type=float, default=1e-4)
    parser.add_argument("--state_class_reg", type=float, default=1e-2)
    parser.add_argument("--enc_l2_reg", type=float, default=0.01)
    parser.add_argument("--dec_l2_reg", type=float, default=0.01)
    parser.add_argument("--lambda_mi_loss", type=float, default=100)
    parser.add_argument("--effsize_reg", type=float, default=100)
    parser.add_argument("--cosine_loss", type=float, default=10)
    parser.add_argument("--adv_penalnty", type=float, default=100)
    parser.add_argument("--reg_adv", type=float, default=1000)
    parser.add_argument("--reg_classifier", type=float, default=1000)
    parser.add_argument("--similarity_reg", type=float, default=10)
    parser.add_argument("--adversary_steps", type=int, default=4)
    parser.add_argument("--autoencoder_wd", type=float, default=0.0)
    parser.add_argument("--adversary_wd", type=float, default=0.0)
    parser.add_argument("--print_every", type=int, default=250)

    return parser


def main() -> None:
    args = build_argparser().parse_args()

    _mkdir_for_file(args.log_file)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)
    fh = FileHandler(args.log_file, mode="a")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    print2log = logger.info

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    progress_path = os.path.join(args.checkpoint_dir, "progress.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print2log(f"Using device: {device}")
    seed_everything(args.seed)

    cmap = pd.read_csv(args.cmap_file, index_col=0)
    gene_size = len(cmap.columns)
    print2log(f"Loaded cmap: {cmap.shape[0]} signatures x {gene_size} genes from {args.cmap_file}")

    model_params = {
        "encoder_1_hiddens": args.encoder_1_hiddens,
        "encoder_2_hiddens": args.encoder_2_hiddens,
        "latent_dim": args.latent_dim,
        "decoder_1_hiddens": args.decoder_1_hiddens,
        "decoder_2_hiddens": args.decoder_2_hiddens,
        "dropout_decoder": args.dropout_decoder,
        "dropout_encoder": args.dropout_encoder,
        "encoder_activation": activation_from_name(args.encoder_activation),
        "decoder_activation": activation_from_name(args.decoder_activation),
        "V_dropout": args.V_dropout,
        "state_class_hidden": args.state_class_hidden,
        "state_class_drop_in": args.state_class_drop_in,
        "state_class_drop": args.state_class_drop,
        "no_states": args.no_states,
        "adv_class_hidden": args.adv_class_hidden,
        "adv_class_drop_in": args.adv_class_drop_in,
        "adv_class_drop": args.adv_class_drop,
        "no_adv_class": args.no_adv_class,
        "encoding_lr": args.encoding_lr,
        "adv_lr": args.adv_lr,
        "schedule_step_adv": args.schedule_step_adv,
        "gamma_adv": args.gamma_adv,
        "schedule_step_enc": args.schedule_step_enc,
        "gamma_enc": args.gamma_enc,
        "batch_size_1": args.batch_size_1,
        "batch_size_2": args.batch_size_2,
        "batch_size_paired": args.batch_size_paired,
        "epochs": args.epochs,
        "prior_beta": args.prior_beta,
        "no_folds": args.no_folds,
        "v_reg": args.v_reg,
        "state_class_reg": args.state_class_reg,
        "enc_l2_reg": args.enc_l2_reg,
        "dec_l2_reg": args.dec_l2_reg,
        "lambda_mi_loss": args.lambda_mi_loss,
        "effsize_reg": args.effsize_reg,
        "cosine_loss": args.cosine_loss,
        "adv_penalnty": args.adv_penalnty,
        "reg_adv": args.reg_adv,
        "reg_classifier": args.reg_classifier,
        "similarity_reg": args.similarity_reg,
        "adversary_steps": args.adversary_steps,
        "autoencoder_wd": args.autoencoder_wd,
        "adversary_wd": args.adversary_wd,
        "print_every": args.print_every,
    }

    pretrained_adv_state = load_pretrained_adv_state(args.pretrained_adv_class, device, print2log)

    recon_csv_path = os.path.join(args.output_dir, "A375_HT29_ExtremelyfewPairs_AutoTransOP_reconstruction.csv")
    trans_csv_path = os.path.join(args.output_dir, "A375_HT29_ExtremelyfewPairs_AutoTransOP_translation.csv")
    details_csv_path = os.path.join(args.output_dir, "A375_HT29_ExtremelyfewPairs_AutoTransOP_details.csv")

    df_recon_all = _load_or_empty_csv(recon_csv_path)
    df_trans_all = _load_or_empty_csv(trans_csv_path)
    df_details_all = _load_or_empty_csv(details_csv_path)

    resume_ptr = {"n_pairs_idx": -1, "fold_id": -1, "repeat": -1}
    if args.resume:
        p = load_progress(progress_path)
        if p and "last_completed" in p:
            resume_ptr = p["last_completed"]
            print2log(
                f"[Resume] Last completed: n_pairs_idx={resume_ptr['n_pairs_idx']} "
                f"fold_id={resume_ptr['fold_id']} repeat={resume_ptr['repeat']}"
            )

    folder = f"{args.dataset1}_{args.dataset2}"
    folder_path = os.path.join(args.data_root, folder)
    largest_sample_len = find_largest_sample_len(folder_path)
    print2log(f"Processing folder: {folder} with sample_len: {largest_sample_len}")

    for n_pairs_idx, n_pairs in enumerate(args.n_pairs_list):
        if n_pairs_idx < resume_ptr["n_pairs_idx"]:
            continue

        print2log("============================================================")
        print2log(f"n_pairs = {n_pairs}")
        model_params["batch_size_paired"] = (
            infer_batch_size_paired(n_pairs) if args.batch_size_paired == 0 else min(args.batch_size_paired, n_pairs)
        )
        print2log(f"batch_size_paired = {model_params['batch_size_paired']}")

        for fold_id in range(model_params["no_folds"]):
            if n_pairs_idx == resume_ptr["n_pairs_idx"] and fold_id < resume_ptr["fold_id"]:
                continue

            print2log(f"=== Start fold {fold_id} for n_pairs {n_pairs} ===")

            trainInfo_paired_full = pd.read_csv(
                os.path.join(folder_path, largest_sample_len, f"train_paired_{fold_id + 1}.csv"),
                index_col=None,
            )
            trainInfo_1 = pd.read_csv(
                os.path.join(folder_path, largest_sample_len, f"train_{args.dataset1}_{fold_id + 1}.csv"),
                index_col=None,
            )
            trainInfo_2 = pd.read_csv(
                os.path.join(folder_path, largest_sample_len, f"train_{args.dataset2}_{fold_id + 1}.csv"),
                index_col=None,
            )
            valInfo_paired = pd.read_csv(
                os.path.join(folder_path, largest_sample_len, f"val_paired_{fold_id + 1}.csv"),
                index_col=None,
            )
            valInfo_1 = pd.read_csv(
                os.path.join(folder_path, largest_sample_len, f"val_{args.dataset1}_{fold_id + 1}.csv"),
                index_col=None,
            )
            valInfo_2 = pd.read_csv(
                os.path.join(folder_path, largest_sample_len, f"val_{args.dataset2}_{fold_id + 1}.csv"),
                index_col=None,
            )

            all_pairs_train = np.arange(len(trainInfo_paired_full))
            if len(all_pairs_train) == 0:
                print2log(f"[Warning] No paired samples in training for fold {fold_id}; skipping.")
                continue

            rng = np.random.default_rng(args.seed + 10000 * n_pairs_idx + 100 * fold_id)
            perm = rng.permutation(all_pairs_train)
            max_repeats_possible = len(perm) // n_pairs
            if max_repeats_possible <= 0:
                print2log(
                    f"[Warning] Not enough pairs (N={len(perm)}) for n_pairs={n_pairs} in fold {fold_id}; skipping."
                )
                continue
            effective_repeats = min(args.n_repeats, max_repeats_possible)
            if effective_repeats < args.n_repeats:
                print2log(
                    f"[Info] Only {effective_repeats} non-overlapping repeats possible for n_pairs={n_pairs} "
                    f"in fold {fold_id} (requested {args.n_repeats})."
                )

            for rep in range(effective_repeats):
                if (
                    n_pairs_idx == resume_ptr["n_pairs_idx"]
                    and fold_id == resume_ptr["fold_id"]
                    and rep <= resume_ptr["repeat"]
                ):
                    continue

                start = rep * n_pairs
                end = start + n_pairs
                selected_pairs_train = np.sort(perm[start:end])
                print2log(
                    f"  [Repeat {rep + 1}/{effective_repeats}] "
                    f"Using paired indices in full trainInfo_paired: {selected_pairs_train.tolist()}"
                )

                trainInfo_paired = trainInfo_paired_full.iloc[selected_pairs_train, :].reset_index(drop=True)

                # Construct full train tensors for post-training reconstruction and selected train pairs for translation.
                X_1_train = torch.tensor(
                    np.concatenate((cmap.loc[trainInfo_paired["sig_id.x"]].values, cmap.loc[trainInfo_1.sig_id].values)),
                    dtype=torch.float32,
                ).to(device)
                X_2_train = torch.tensor(
                    np.concatenate((cmap.loc[trainInfo_paired["sig_id.y"]].values, cmap.loc[trainInfo_2.sig_id].values)),
                    dtype=torch.float32,
                ).to(device)
                X_1_val = torch.tensor(
                    np.concatenate((cmap.loc[valInfo_paired["sig_id.x"]].values, cmap.loc[valInfo_1.sig_id].values)),
                    dtype=torch.float32,
                ).to(device)
                X_2_val = torch.tensor(
                    np.concatenate((cmap.loc[valInfo_paired["sig_id.y"]].values, cmap.loc[valInfo_2.sig_id].values)),
                    dtype=torch.float32,
                ).to(device)

                train_pair_n = len(trainInfo_paired)
                val_pair_n = len(valInfo_paired)

                (
                    decoder_1,
                    decoder_2,
                    encoder_1,
                    encoder_2,
                    prior_d,
                    local_d,
                    classifier,
                    adverse_classifier,
                    Vsp,
                ) = train_autotransop_one_run(
                    model_params=model_params,
                    device=device,
                    cmap=cmap,
                    trainInfo_paired=trainInfo_paired,
                    trainInfo_1=trainInfo_1,
                    trainInfo_2=trainInfo_2,
                    pretrained_adv_state=pretrained_adv_state,
                    print2log=print2log,
                    fold_id=fold_id,
                    n_pairs=n_pairs,
                    rep=rep,
                )

                decoder_1.eval(); decoder_2.eval()
                encoder_1.eval(); encoder_2.eval()
                prior_d.eval(); local_d.eval()
                classifier.eval(); adverse_classifier.eval(); Vsp.eval()

                recon_train = evaluate_reconstruction(
                    X_1_train, X_2_train, encoder_1, encoder_2, decoder_1, decoder_2, Vsp, device
                )
                recon_val = evaluate_reconstruction(
                    X_1_val, X_2_val, encoder_1, encoder_2, decoder_1, decoder_2, Vsp, device
                )
                trans_train = evaluate_translation_pairs(
                    X_1_train[:train_pair_n],
                    X_2_train[:train_pair_n],
                    encoder_1,
                    encoder_2,
                    decoder_1,
                    decoder_2,
                    Vsp,
                    device,
                )
                trans_val = evaluate_translation_pairs(
                    X_1_val[:val_pair_n],
                    X_2_val[:val_pair_n],
                    encoder_1,
                    encoder_2,
                    decoder_1,
                    decoder_2,
                    Vsp,
                    device,
                )
                class_val = evaluate_classifier(X_1_val, X_2_val, encoder_1, encoder_2, classifier, Vsp, device)

                print2log(
                    f"    Fold {fold_id}, rep {rep}: "
                    f"recon_r train={recon_train['mean_pearson']:.4f}, val={recon_val['mean_pearson']:.4f}; "
                    f"translation_r train={trans_train['mean_pearson']:.4f}, val={trans_val['mean_pearson']:.4f}; "
                    f"direct_val={trans_val['direct_mean_pearson']:.4f}; "
                    f"class_acc={class_val['class_acc']:.4f}"
                )

                common = {
                    "fold": fold_id,
                    "n_pairs": n_pairs,
                    "repeat": rep,
                    "selected_pair_indices": ";".join(map(str, selected_pairs_train.tolist())),
                    "train_pairs": train_pair_n,
                    "val_pairs": val_pair_n,
                    "dataset1": args.dataset1,
                    "dataset2": args.dataset2,
                    "sample_len": largest_sample_len,
                }

                tmp_recon = pd.DataFrame(
                    {
                        **common,
                        "train": recon_train["mean_pearson"],
                        "test": recon_val["mean_pearson"],
                        "train_spearman": recon_train["mean_spearman"],
                        "test_spearman": recon_val["mean_spearman"],
                        "train_pseudo_acc": recon_train["mean_pseudo_acc"],
                        "test_pseudo_acc": recon_val["mean_pseudo_acc"],
                        "train_A375_pearson": recon_train["A375_pearson"],
                        "train_HT29_pearson": recon_train["HT29_pearson"],
                        "test_A375_pearson": recon_val["A375_pearson"],
                        "test_HT29_pearson": recon_val["HT29_pearson"],
                    },
                    index=[0],
                )

                tmp_trans = pd.DataFrame(
                    {
                        **common,
                        "train": trans_train["mean_pearson"],
                        "test": trans_val["mean_pearson"],
                        "train_spearman": trans_train["mean_spearman"],
                        "test_spearman": trans_val["mean_spearman"],
                        "train_pseudo_acc": trans_train["mean_pseudo_acc"],
                        "test_pseudo_acc": trans_val["mean_pseudo_acc"],
                        "train_A375_to_HT29_pearson": trans_train["A375_to_HT29_pearson"],
                        "train_HT29_to_A375_pearson": trans_train["HT29_to_A375_pearson"],
                        "test_A375_to_HT29_pearson": trans_val["A375_to_HT29_pearson"],
                        "test_HT29_to_A375_pearson": trans_val["HT29_to_A375_pearson"],
                        "direct_train": trans_train["direct_mean_pearson"],
                        "direct_test": trans_val["direct_mean_pearson"],
                    },
                    index=[0],
                )

                tmp_details = pd.DataFrame(
                    {
                        **common,
                        **{f"recon_train_{k}": v for k, v in recon_train.items()},
                        **{f"recon_test_{k}": v for k, v in recon_val.items()},
                        **{f"translation_train_{k}": v for k, v in trans_train.items()},
                        **{f"translation_test_{k}": v for k, v in trans_val.items()},
                        **class_val,
                    },
                    index=[0],
                )

                keys = ["n_pairs", "fold", "repeat"]
                df_recon_all = _append_and_write_safely(df_recon_all, tmp_recon, recon_csv_path, dedup_subset=keys)
                df_trans_all = _append_and_write_safely(df_trans_all, tmp_trans, trans_csv_path, dedup_subset=keys)
                df_details_all = _append_and_write_safely(df_details_all, tmp_details, details_csv_path, dedup_subset=keys)

                if args.save_models:
                    tag = f"nPairs{n_pairs}_rep{rep}"
                    # Store JSON-friendly model params in metadata; actual activation modules are not serialized here.
                    model_params_meta = dict(vars(args))
                    save_models_at_rep_end(
                        args.checkpoint_dir,
                        tag=tag,
                        fold_id=fold_id,
                        encoder_1=encoder_1,
                        decoder_1=decoder_1,
                        encoder_2=encoder_2,
                        decoder_2=decoder_2,
                        prior_d=prior_d,
                        local_d=local_d,
                        classifier=classifier,
                        adverse_classifier=adverse_classifier,
                        Vsp=Vsp,
                        extra={
                            "n_pairs": n_pairs,
                            "fold": fold_id,
                            "repeat": rep,
                            "selected_pair_indices": selected_pairs_train.tolist(),
                            "model_params": model_params_meta,
                        },
                    )

                save_progress(progress_path, n_pairs_idx=n_pairs_idx, fold_id=fold_id, repeat=rep)
                print2log(f"=== END n_pairs={n_pairs}, fold={fold_id}, repeat={rep} ===")

    print2log("All AutoTransOP low-pair experiments completed.")


if __name__ == "__main__":
    main()
