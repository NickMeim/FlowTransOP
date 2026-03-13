import os
import json
import argparse
import torch
from models import Decoder, SimpleEncoder, Flow
from trainingUtils import train_GeneralFM_fold, validate_flowMatch_fold, train_AE_fold
from utility import *
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from logging import FileHandler
import warnings

warnings.filterwarnings("ignore", message=".*ks_2samp.*")

# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description=(
        "Flow-matching on A375_HT29 using the full 5-fold splits but with an "
        "extremely small number of paired samples (3/5/10) per fold. "
        "For each n_pairs we repeat the experiment multiple times, "
        "using disjoint subsets of pairs within each fold."
    )
)

# Checkpointing / logging
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="./chkpts_PairAndSimilarity_lowPairsPercentage_extreme",
    help="Directory to store checkpoints and progress.json",
)
parser.add_argument(
    "--log_file",
    type=str,
    default="logs/PairAndSimilarity_lowPairsPercentage_extreme.log",
    help="Location of the log file for print outputs",
)
parser.add_argument(
    "--resume",
    type=int,
    default=1,
    help="1 to resume if checkpoint exists",
)

# Few-pairs experimental design
parser.add_argument(
    "--n_pairs_list",
    type=int,
    nargs="+",
    default=[1, 2, 3],
    help="List with numbers of paired samples to use in each training run.",
)
parser.add_argument(
    "--n_repeats",
    type=int,
    default=30,
    help="Maximum number of random repeats per n_pairs setting (per fold).",
)

# Data and output paths
parser.add_argument(
    "--data_root",
    type=str,
    default="../preprocessing/preprocessed_data/CellPairs/",
    help="Root directory for preprocessed data (A375_HT29 full 5-fold dataset).",
)
parser.add_argument(
    "--cmap_file",
    type=str,
    default="../preprocessing/preprocessed_data/CellPairs/drug_landmarks.csv",
    help="Path to the CMAP CSV file.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="../results/FlowMatch_extremely_fewPairs_A375_HT29_PairAndSimilarity/",
    help="Directory to save output results (CSV metrics).",
)

# Training parameters (mirroring the full A375_HT29 setup)
parser.add_argument("--batch_size_1", type=int, default=120, help="Batch size for dataset 1.")
parser.add_argument("--batch_size_2", type=int, default=120, help="Batch size for dataset 2.")
parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs for training.")
parser.add_argument("--seed", type=int, default=42, help="Base random seed for reproducibility.")

# Model parameters (you can override via CLI)
parser.add_argument("--encoder_1_hiddens", type=int, nargs="+", default=[640, 384], help="Hidden layer sizes for encoder 1.")
parser.add_argument("--encoder_2_hiddens", type=int, nargs="+", default=[640, 384], help="Hidden layer sizes for encoder 2.")
parser.add_argument("--latent_dim", type=int, default=292, help="Dimension of the latent space.")
parser.add_argument("--decoder_1_hiddens", type=int, nargs="+", default=[384, 640], help="Hidden layer sizes for decoder 1.")
parser.add_argument("--decoder_2_hiddens", type=int, nargs="+", default=[384, 640], help="Hidden layer sizes for decoder 2.")
parser.add_argument("--dropout_decoder", type=float, default=0.2, help="Dropout rate for the decoder.")
parser.add_argument("--dropout_encoder", type=float, default=0.1, help="Dropout rate for the encoder.")
parser.add_argument("--bn_decoder", type=float, default=0.6, help="Use batch normalization in the decoder.")
parser.add_argument("--bn_encoder", type=float, default=0.6, help="Use batch normalization in the encoder.")
parser.add_argument("--dropout_input_encoder", type=float, default=0.5, help="Dropout rate for the input of the encoder.")
parser.add_argument("--dropout_input_decoder", type=float, default=0.0, help="Dropout rate for the input of the decoder.")
parser.add_argument(
    "--encoder_activation",
    type=str,
    choices=["LeakyReLU", "ReLU", "ELU", "Sigmoid"],
    default="ELU",
    help="Activation function used between layers of the encoder",
)
parser.add_argument(
    "--decoder_activation",
    type=str,
    choices=["LeakyReLU", "ReLU", "ELU", "Sigmoid"],
    default="ELU",
    help="Activation function used between layers of the decoder",
)
parser.add_argument("--encoding_lr", type=float, default=0.001, help="Learning rate for the encoder.")
parser.add_argument("--schedule_step_enc", type=int, default=200, help="Step size for encoder LR scheduler.")
parser.add_argument("--gamma_enc", type=float, default=0.8, help="Gamma for encoder LR scheduler.")
parser.add_argument("--no_folds", type=int, default=5, help="Number of cross-validation folds (A375_HT29 has 5).")
parser.add_argument("--enc_l2_reg", type=float, default=0.01, help="L2 regularization for the encoder.")
parser.add_argument("--dec_l2_reg", type=float, default=0.01, help="L2 regularization for the decoder.")
parser.add_argument("--autoencoder_wd", type=float, default=0.0, help="Weight decay for the autoencoder.")

# Flow-related regularization
parser.add_argument("--flow_lambda", type=float, default=1.0, help="Flow matrix regularization parameter.")
parser.add_argument("--conditional_flow_lambda", type=float, default=1e-3, help="Flow matching regularization parameter.")

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Logging & device
# ---------------------------------------------------------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = FileHandler(args.log_file, mode="a")
fh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(fh)
print2log = logger.info

output_dir = args.output_dir
Path(output_dir).mkdir(parents=True, exist_ok=True)

CKPT_DIR = args.checkpoint_dir
os.makedirs(CKPT_DIR, exist_ok=True)
PROGRESS = os.path.join(CKPT_DIR, "progress.json")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print2log(f"Using device: {device}")

# Initialize environment and seeds for reproducibility
seed_everything(args.seed)

# ---------------------------------------------------------------------------
# Checkpoint helper functions (adapted from lowPairsPercentage script)
# ---------------------------------------------------------------------------

def _load_or_empty_csv(path: str) -> pd.DataFrame:
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def _atomic_to_csv(df: pd.DataFrame, path: str, **to_csv_kwargs):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False, **to_csv_kwargs)
    os.replace(tmp, path)


def _append_and_write_safely(existing_df: pd.DataFrame,
                             new_df: pd.DataFrame,
                             out_path: str,
                             dedup_subset=None):
    if new_df is None or (isinstance(new_df, pd.DataFrame) and new_df.empty):
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


# progress pointer (resume after last COMPLETED triple: n_pairs_idx, fold_id, repeat)

def save_progress(path: str, n_pairs_idx: int, fold_id: int, repeat: int):
    state = {"last_completed": {"n_pairs_idx": n_pairs_idx, "fold_id": fold_id, "repeat": repeat}}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)


def load_progress(path: str):
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                return json.load(f)
            except Exception:
                return None
    return None


# Save models at fold end (per n_pairs and repeat)

def save_models_at_rep_end(
    ckpt_dir: str,
    tag: str,
    fold_id: int,
    encoder_1,
    decoder_1,
    encoder_2,
    decoder_2,
    flow_12,
    flow_21,
    extra=None,
):
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(ckpt_dir, f"fold{fold_id}__{tag}.pt")
    payload = {
        "encoder_1": encoder_1.state_dict() if encoder_1 is not None else None,
        "decoder_1": decoder_1.state_dict() if decoder_1 is not None else None,
        "encoder_2": encoder_2.state_dict() if encoder_2 is not None else None,
        "decoder_2": decoder_2.state_dict() if decoder_2 is not None else None,
        "flow_12": flow_12.state_dict() if flow_12 is not None else None,
        "flow_21": flow_21.state_dict() if flow_21 is not None else None,
        "extra": extra or {},
    }
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Read data & activation functions
# ---------------------------------------------------------------------------

cmap = pd.read_csv(args.cmap_file, index_col=0)
genes = cmap.columns.values
gene_size = len(cmap.columns)

if args.decoder_activation == "LeakyReLU":
    decoder_activation = torch.nn.LeakyReLU(0.01)
elif args.decoder_activation == "ReLU":
    decoder_activation = torch.nn.ReLU()
elif args.decoder_activation == "ELU":
    decoder_activation = torch.nn.ELU()
elif args.decoder_activation == "Sigmoid":
    decoder_activation = torch.nn.Sigmoid()
else:
    raise ValueError(f"Unknown decoder activation: {args.decoder_activation}")

if args.encoder_activation == "LeakyReLU":
    encoder_activation = torch.nn.LeakyReLU(0.01)
elif args.encoder_activation == "ReLU":
    encoder_activation = torch.nn.ReLU()
elif args.encoder_activation == "ELU":
    encoder_activation = torch.nn.ELU()
elif args.encoder_activation == "Sigmoid":
    encoder_activation = torch.nn.Sigmoid()
else:
    raise ValueError(f"Unknown encoder activation: {args.encoder_activation}")

# Model parameters dict passed to train_* helpers
model_params = {
    "encoder_1_hiddens": args.encoder_1_hiddens,
    "encoder_2_hiddens": args.encoder_2_hiddens,
    "latent_dim": args.latent_dim,
    "decoder_1_hiddens": args.decoder_1_hiddens,
    "decoder_2_hiddens": args.decoder_2_hiddens,
    "dropout_decoder": args.dropout_decoder,
    "dropout_encoder": args.dropout_encoder,
    "encoder_activation": encoder_activation,
    "decoder_activation": decoder_activation,
    "bn_encoder": args.bn_encoder,
    "bn_decoder": args.bn_decoder,
    "dropout_input_encoder": args.dropout_input_encoder,
    "dropout_input_decoder": args.dropout_input_decoder,
    "encoding_lr": args.encoding_lr,
    "schedule_step_enc": args.schedule_step_enc,
    "gamma_enc": args.gamma_enc,
    "batch_size_1": args.batch_size_1,
    "batch_size_2": args.batch_size_2,
    "batch_size_paired": 10, # placeholder
    "epochs": args.epochs,
    "no_folds": args.no_folds,
    "enc_l2_reg": args.enc_l2_reg,
    "dec_l2_reg": args.dec_l2_reg,
    "autoencoder_wd": args.autoencoder_wd,
    "flow_lambda": args.flow_lambda,
    "conditional_flow_lambda": args.conditional_flow_lambda,
}

# ---------------------------------------------------------------------------
# Prepare global CSVs for reconstruction and translation metrics
# ---------------------------------------------------------------------------

recon_csv_path = os.path.join(output_dir, "A375_HT29_ExtremelyfewPairs_reconstruction.csv")
trans_csv_path = os.path.join(output_dir, "A375_HT29_ExtremelyfewPairs_translation.csv")

df_recon_all = _load_or_empty_csv(recon_csv_path)
df_trans_all = _load_or_empty_csv(trans_csv_path)

progress_path = PROGRESS
Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

resume_ptr = {"n_pairs_idx": -1, "fold_id": -1, "repeat": -1}
if args.resume:
    p = load_progress(progress_path)
    if p and "last_completed" in p:
        resume_ptr = p["last_completed"]
        print2log(
            f"[Resume] Last completed: n_pairs_idx={resume_ptr['n_pairs_idx']} "
            f"fold_id={resume_ptr['fold_id']} repeat={resume_ptr['repeat']}"
        )

# ---------------------------------------------------------------------------
# Main experimental loops
# ---------------------------------------------------------------------------

# Fixed to A375_HT29, using the same folder structure as in the full script
DATASET1, DATASET2 = "A375", "HT29"
folder = f"{DATASET1}_{DATASET2}"
folder_path = os.path.join(args.data_root, folder)

# Find folder that corresponds to the full paired dataset (largest sample length)
largest_sample_len = find_largest_sample_len(folder_path)
print2log(f"Processing folder: {folder} with sample_len: {largest_sample_len}")

for i, n_pairs in enumerate(args.n_pairs_list):
    if i < resume_ptr["n_pairs_idx"]:
        continue

    print2log("============================================================")
    print2log(f"n_pairs = {n_pairs}")
    if n_pairs ==  1:
        model_params['batch_size_paired'] = 1
    elif n_pairs <= 5:
        model_params['batch_size_paired'] = 2
    elif n_pairs <= 15:
        model_params['batch_size_paired'] = 5
    elif n_pairs <= 30:
        model_params['batch_size_paired'] = 10
    elif n_pairs <= 100:
        model_params['batch_size_paired'] = 50
    else:
        model_params['batch_size_paired'] = 90


    for fold_id in range(model_params["no_folds"]):
        if i == resume_ptr["n_pairs_idx"] and fold_id < resume_ptr["fold_id"]:
            continue

        print2log(f"=== Start fold {fold_id} for n_pairs {n_pairs} ===")

        # ------------------ Load full metadata for this fold ------------------
        trainInfo_paired_full = pd.read_csv(
            os.path.join(folder_path, largest_sample_len, f"train_paired_{fold_id + 1}.csv"),
            index_col=None,
        )
        trainInfo_1 = pd.read_csv(
            os.path.join(folder_path, largest_sample_len, f"train_{DATASET1}_{fold_id + 1}.csv"),
            index_col=None,
        )
        trainInfo_2 = pd.read_csv(
            os.path.join(folder_path, largest_sample_len, f"train_{DATASET2}_{fold_id + 1}.csv"),
            index_col=None,
        )

        valInfo_paired = pd.read_csv(
            os.path.join(folder_path, largest_sample_len, f"val_paired_{fold_id + 1}.csv"),
            index_col=None,
        )
        valInfo_1 = pd.read_csv(
            os.path.join(folder_path, largest_sample_len, f"val_{DATASET1}_{fold_id + 1}.csv"),
            index_col=None,
        )
        valInfo_2 = pd.read_csv(
            os.path.join(folder_path, largest_sample_len, f"val_{DATASET2}_{fold_id + 1}.csv"),
            index_col=None,
        )

        # All training paired indices
        all_pairs_train = np.arange(len(trainInfo_paired_full))
        if len(all_pairs_train) == 0:
            print2log(f"[Warning] No paired samples in training for fold {fold_id}")
            continue

        # We want disjoint subsets of size n_pairs for each repeat
        rng = np.random.default_rng(args.seed + 10000 * i + 100 * fold_id)
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
                i == resume_ptr["n_pairs_idx"]
                and fold_id == resume_ptr["fold_id"]
                and rep <= resume_ptr["repeat"]
            ):
                continue

            start = rep * n_pairs
            end = start + n_pairs
            selected_pairs_train = np.sort(perm[start:end])

            print2log(
                f"  [Repeat {rep + 1}/{effective_repeats}] "
                f"Using paired indices (in full trainInfo_paired): {selected_pairs_train.tolist()}"
            )

            # ------------------ Subset trainInfo_paired to ONLY these pairs ------------------
            trainInfo_paired = trainInfo_paired_full.iloc[selected_pairs_train, :].reset_index(drop=True)

            # ------------------ Build X_1 / X_2 for TRAIN ------------------
            # Only the selected pairs + all unpaired samples (trainInfo_1 / trainInfo_2)
            X_1 = torch.tensor(
                np.concatenate(
                    (
                        cmap.loc[trainInfo_paired["sig_id.x"]].values,
                        cmap.loc[trainInfo_1.sig_id].values,
                    )
                )
            ).float().to(device)

            X_2 = torch.tensor(
                np.concatenate(
                    (
                        cmap.loc[trainInfo_paired["sig_id.y"]].values,
                        cmap.loc[trainInfo_2.sig_id].values,
                    )
                )
            ).float().to(device)

            # ------------------ Build X_1_val / X_2_val (FULL validation pairs) ------------------
            X_1_val = torch.tensor(
                np.concatenate(
                    (
                        cmap.loc[valInfo_paired["sig_id.x"]].values,
                        cmap.loc[valInfo_1.sig_id].values,
                    )
                )
            ).float().to(device)

            X_2_val = torch.tensor(
                np.concatenate(
                    (
                        cmap.loc[valInfo_paired["sig_id.y"]].values,
                        cmap.loc[valInfo_2.sig_id].values,
                    )
                )
            ).float().to(device)

            # Pair indices relative to the constructed tensors:
            #   - For training, pairs occupy the first len(trainInfo_paired) positions.
            #   - For validation, pairs occupy the first len(valInfo_paired) positions.
            pairs_train = np.arange(len(trainInfo_paired))
            pairs_val = np.arange(len(valInfo_paired))

            # ------------------ Initialize models for this (n_pairs, fold, rep) ------------------
            decoder_1 = Decoder(
                model_params["latent_dim"],
                model_params["decoder_1_hiddens"],
                gene_size,
                dropRate=model_params["dropout_decoder"],
                bn=model_params["bn_decoder"],
                activation=model_params["decoder_activation"],
                dropIn=model_params["dropout_input_decoder"],
            ).to(device)

            decoder_2 = Decoder(
                model_params["latent_dim"],
                model_params["decoder_2_hiddens"],
                gene_size,
                dropRate=model_params["dropout_decoder"],
                bn=model_params["bn_decoder"],
                activation=model_params["decoder_activation"],
                dropIn=model_params["dropout_input_decoder"],
            ).to(device)

            encoder_1 = SimpleEncoder(
                gene_size,
                model_params["encoder_1_hiddens"],
                model_params["latent_dim"],
                dropRate=model_params["dropout_encoder"],
                bn=model_params["bn_encoder"],
                activation=model_params["encoder_activation"],
                dropIn=model_params["dropout_input_encoder"],
            ).to(device)

            encoder_2 = SimpleEncoder(
                gene_size,
                model_params["encoder_2_hiddens"],
                model_params["latent_dim"],
                dropRate=model_params["dropout_encoder"],
                bn=model_params["bn_encoder"],
                activation=model_params["encoder_activation"],
                dropIn=model_params["dropout_input_encoder"],
            ).to(device)

            flow_12 = Flow(model_params["latent_dim"], int(model_params["latent_dim"] / 2)).to(device)
            flow_21 = Flow(model_params["latent_dim"], int(model_params["latent_dim"] / 2)).to(device)

            # ------------------ Autoencoder pretraining (on this tiny-paired + full-unpaired train set) ------------------
            (r1, decoder_1, encoder_1) = train_AE_fold(
                model_params,
                device,
                X_1,
                decoder_1,
                encoder_1,
                model_params["batch_size_1"],
                model_params["epochs"],
            )
            (r2, decoder_2, encoder_2) = train_AE_fold(
                model_params,
                device,
                X_2,
                decoder_2,
                encoder_2,
                model_params["batch_size_2"],
                model_params["epochs"],
            )
            print2log(
                f"    AE train (n_pairs={n_pairs}, fold={fold_id}, rep={rep}): "
                f"r1={np.nanmean(r1):.4f}, r2={np.nanmean(r2):.4f}"
            )

            encoder_1.eval()
            encoder_2.eval()
            decoder_1.eval()
            decoder_2.eval()

            Z_1 = encoder_1(X_1.double())
            Z_2 = encoder_2(X_2.double())

            ## put all embeddings in a DataFrame
            all_emb1 = pd.DataFrame(encoder_1(torch.tensor(cmap.values,dtype=torch.double).to(device)).detach().cpu().numpy(),
                                    index= cmap.index)
            all_emb2 = pd.DataFrame(encoder_2(torch.tensor(cmap.values,dtype=torch.double).to(device)).detach().cpu().numpy(),
                                    index= cmap.index)

            # ------------------ Train flow using ONLY the selected training pairs ------------------
            print2log(f"    Flow train (n_pairs={n_pairs}, fold={fold_id}, rep={rep})...")
            (pearson_1_to_2, _, flow_12) = train_GeneralFM_fold(
                model_params,
                device,
                X_1,
                X_2,
                Z_1,
                Z_2,
                cmap,
                all_emb1, all_emb2,
                trainInfo_1, trainInfo_2,trainInfo_paired,
                decoder_1,
                decoder_2,
                flow_12,
                model_params["batch_size_1"],
                model_params["batch_size_2"],
                model_params['batch_size_paired'],
                model_params["epochs"],
                pairs_train=pairs_train,
                tanslation_direction="1 to 2",
                similarity_agregation = 'mean'
            )

            # ------------------ Validate on ALL validation pairs ------------------
            (r_1_to_2, pearson1, pearson2, _) = validate_flowMatch_fold(
                device,
                X_1_val,
                X_2_val,
                decoder_1,
                decoder_2,
                encoder_1,
                encoder_2,
                flow_12,
                pairs_val,
                "1 to 2",
            )

            print2log('Repeat flow training for the oppo direction...')
            (pearson_2_to_1,_,flow_21) = train_GeneralFM_fold(model_params, device,
                                                          X_1, X_2,
                                                          Z_1, Z_2,
                                                          cmap,
                                                          all_emb1, all_emb2,
                                                          trainInfo_1, trainInfo_2,trainInfo_paired,
                                                          decoder_1, decoder_2,
                                                          flow_21,
                                                          model_params['batch_size_1'], model_params['batch_size_2'],model_params['batch_size_paired'], model_params['epochs'],
                                                          pairs_train=pairs_train,
                                                          tanslation_direction = '2 to 1',
                                                          similarity_agregation = 'mean')
            r_2_to_1,_,pearson2,_ = validate_flowMatch_fold(device, X_1_val, X_2_val, decoder_1, decoder_2, encoder_1, encoder_2,flow_21,pairs_val,'2 to 1')
            ## aggregate
            mu_r = 0.5*(np.nanmean(pearson1) + np.nanmean(pearson2))
            mu_r_translation_val = 0.5*(np.nanmean(r_1_to_2) + np.nanmean(r_2_to_1))
            mu_r_translation_train = 0.5*(np.nanmean(pearson_1_to_2) + np.nanmean(pearson_2_to_1))

            print2log(
                f"    Fold {fold_id}, rep {rep}: "
                f"recon_r={mu_r:.4f}, "
                f"train_translation_r={mu_r_translation_train:.4f}, "
                f"val_translation_r={mu_r_translation_val:.4f}"
            )

            # -------------------- PER-EXPERIMENT DATAFRAMES --------------------
            # Reconstruction: train (AE) vs test (reconstruction on val)
            tmp_recon = pd.DataFrame(
                {
                    "train": [0.5 * (np.nanmean(r1) + np.nanmean(r2))],
                    "test": [mu_r],
                    "fold": [fold_id],
                    "n_pairs": [n_pairs],
                    "repeat": [rep],
                }
            )

            # Translation: train (flow on paired train) vs test (flow on val pairs)
            tmp_trans = pd.DataFrame(
                {
                    "train": [mu_r_translation_train],
                    "test": [mu_r_translation_val],
                    "fold": [fold_id],
                    "n_pairs": [n_pairs],
                    "repeat": [rep],
                }
            )

            # -------------------- APPEND, DEDUP, ATOMIC WRITE -----------------
            recon_keys = ["n_pairs", "fold", "repeat"]
            trans_keys = ["n_pairs", "fold", "repeat"]
            df_recon_all = _append_and_write_safely(
                df_recon_all, tmp_recon, recon_csv_path, dedup_subset=recon_keys
            )
            df_trans_all = _append_and_write_safely(
                df_trans_all, tmp_trans, trans_csv_path, dedup_subset=trans_keys
            )

            print2log(
                f"[Saved] n_pairs={n_pairs} fold={fold_id} repeat={rep} "
                f"reconstruction r (mean) train={tmp_recon['train'].iloc[0]:.4f} "
                f"test={tmp_recon['test'].iloc[0]:.4f}"
            )
            print2log(
                f"[Saved] n_pairs={n_pairs} fold={fold_id} repeat={rep} "
                f"translation r (mean) train={tmp_trans['train'].iloc[0]:.4f} "
                f"test={tmp_trans['test'].iloc[0]:.4f}"
            )

            # -------------------- SAVE MODELS & PROGRESS ----------------------
            tag = f"nPairs{n_pairs}_rep{rep}"
            save_models_at_rep_end(
                args.checkpoint_dir,
                tag=tag,
                fold_id=fold_id,
                encoder_1=encoder_1,
                decoder_1=decoder_1,
                encoder_2=encoder_2,
                decoder_2=decoder_2,
                flow_12=flow_12,
                flow_21=flow_21,
                extra={"n_pairs": n_pairs, "fold": fold_id, "repeat": rep},
            )
            save_progress(progress_path, n_pairs_idx=i, fold_id=fold_id, repeat=rep)

            print2log(
                f"=== END n_pairs={n_pairs}, fold={fold_id}, repeat={rep} "
                f"(effective_repeats={effective_repeats}) ==="
            )

print2log("All experiments completed.")
