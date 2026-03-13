import os
import sys
import json
import argparse
import torch
from models import Decoder
from trainingUtils import train_decoder_fold
from evaluationUtils import pearson_r
from utility import *
from transact_utility_gpu import *
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from logging import FileHandler

# Console logging by default; file handler is added after parsing args
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()
print2log = logger.info

# Initialize argparse
parser = argparse.ArgumentParser(description="Run cross-fold validation for comparing with AutoTransOP.")
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="./chkpts_decoder_diffenetInputs",
    help="Directory to store checkpoints and progress.json",
)
parser.add_argument(
    "--log_file",
    type=str,
    default="logs/DecodeFromConsensus_diffenetInputs.log",
    help="Location of the log file that will capture print outputs",
)
parser.add_argument(
    "--resume",
    type=int,
    default=1,
    help="1 to resume if checkpoint exists, 0 to start from scratch",
)

# Data and output paths
parser.add_argument(
    "--cell_lines",
    metavar="N",
    type=str,
    nargs="*",
    help="cell lines to artificially create different inputs",
    default=[
        "PC3",
        "HT29",
        "MCF7",
        "A549",
        "NPC",
        "HEPG2",
        "A375",
        "YAPC",
        "U2OS",
        "MCF10A",
        "HA1E",
        "HCC515",
        "ASC",
        "VCAP",
        "HUVEC",
        "HELA",
    ],
)
parser.add_argument(
    "--random_iterations",
    type=int,
    default=5,
    help="Number of random iterations for imputing different inputs.",
)
parser.add_argument(
    "--data_root",
    type=str,
    help="Root directory for preprocessed data.",
    default="../preprocessing/preprocessed_data/SameCellimputationModel/",
)
parser.add_argument(
    "--cmap_file",
    type=str,
    help="Path to the CMAP CSV file.",
    default="../preprocessing/preprocessed_data/CellPairs/drug_landmarks.csv",
)
parser.add_argument(
    "--output_dir",
    type=str,
    help="Directory to save output results.",
    default="../results/Decoders_only_diffenetInputs/",
)

# Training parameters
parser.add_argument(
    "--batch_size", type=int, default=512, help="Batch size for training"
)
parser.add_argument(
    "--epochs", type=int, default=1000, help="Number of epochs for training."
)
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for reproducibility."
)

# Model parameters
parser.add_argument(
    "--decoder_hiddens",
    type=int,
    nargs="+",
    default=[256, 384],
    help="Hidden layer sizes for decoders.",
)
parser.add_argument(
    "--dropout_decoder", type=float, default=0.2, help="Dropout rate for the decoder."
)
parser.add_argument(
    "--bn_decoder",
    type=float,
    default=0.6,
    help="Use batch normalization in the decoder.",
)
parser.add_argument(
    "--dropout_input_decoder",
    type=float,
    default=0.0,
    help="Dropout rate for the imput of the decoder.",
)
parser.add_argument(
    "--decoder_activation",
    type=str,
    choices=["LeakyReLU", "ReLU", "ELU", "Sigmoid"],
    help="Activation function used between layers of the decoder",
    default="ELU",
)
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
parser.add_argument(
    "--schedule_step_enc",
    type=int,
    default=200,
    help="Step size for the encoder learning rate scheduler.",
)
parser.add_argument(
    "--gamma_enc",
    type=float,
    default=0.8,
    help="Gamma for the encoder learning rate scheduler.",
)
parser.add_argument(
    "--dec_l2_reg",
    type=float,
    default=0.01,
    help="L2 regularization for the decoder.",
)
parser.add_argument(
    "--autoencoder_wd",
    type=float,
    default=0.0,
    help="Weight decay for the autoencoder.",
)

args = parser.parse_args()

# Reconfigure logger to also log to file
logger.setLevel(logging.INFO)
Path(os.path.dirname(args.log_file)).mkdir(parents=True, exist_ok=True)
fh = FileHandler(args.log_file, mode="a")
fh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(fh)
print2log = logger.info

cell_lines = args.cell_lines
output_dir = args.output_dir
random_iterations = args.random_iterations
data_root = args.data_root

Path(output_dir).mkdir(parents=True, exist_ok=True)
CKPT_DIR = args.checkpoint_dir
os.makedirs(CKPT_DIR, exist_ok=True)
progress_path = os.path.join(CKPT_DIR, "progress.json")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print2log(f"Using device: {device}")

# Initialize environment and seeds for reproducibility
seed_everything(args.seed)

# ----------------------------------------------------------------------
# Checkpoint / CSV helper utilities (mirroring AutoTransOP script style)
# ----------------------------------------------------------------------
def _load_or_empty_csv(path: str) -> pd.DataFrame:
    """Load CSV if it exists and is non-empty, otherwise return empty DataFrame."""
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def _atomic_to_csv(df: pd.DataFrame, path: str, **to_csv_kwargs):
    """Atomically write DataFrame to CSV to avoid partial writes on preemption."""
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False, **to_csv_kwargs)
    os.replace(tmp, path)


def _append_and_write_safely(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
    out_path: str,
    dedup_subset=None,
):
    """Append new_df to existing_df, drop duplicates, and atomically write to CSV."""
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


def save_progress(path: str, cell_idx: int, fold_id: int, iteration_idx: int):
    """Save last COMPLETED (cell_idx, iteration_idx, fold_id) triple."""
    state = {
        "last_completed": {
            "cell_idx": cell_idx,
            "fold_id": fold_id,
            "iteration_id": iteration_idx,
        }
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)


def load_progress(path: str):
    """Load progress pointer dictionary if it exists; otherwise None."""
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                return json.load(f)
            except Exception:
                return None
    return None


# ------------------------- Read CMAP data --------------------------------
cmap = pd.read_csv(args.cmap_file, index_col=0)
genes = cmap.columns.values
gene_size = len(cmap.columns)
samples = cmap.index.values

# decoder activation
if args.decoder_activation == "LeakyReLU":
    decoder_activation = torch.nn.LeakyReLU(0.01)
elif args.decoder_activation == "ReLU":
    decoder_activation = torch.nn.ReLU()
elif args.decoder_activation == "ELU":
    decoder_activation = torch.nn.ELU()
elif args.decoder_activation == "Sigmoid":
    decoder_activation = torch.nn.Sigmoid()
else:
    raise ValueError(f"Unknown decoder activation {args.decoder_activation}")

# Model parameters
model_params = {
    "latent_dim": 0,  # placeholder, will be set after TRANSACT alignment
    "decoder_hiddens": args.decoder_hiddens,
    "dropout_decoder": args.dropout_decoder,
    "decoder_activation": decoder_activation,
    "bn_decoder": args.bn_decoder,
    "dropout_input_decoder": args.dropout_input_decoder,
    "lr": args.lr,
    "schedule_step_enc": args.schedule_step_enc,
    "gamma_enc": args.gamma_enc,
    "batch_size": args.batch_size,
    "epochs": args.epochs,
    "dec_l2_reg": args.dec_l2_reg,
    "autoencoder_wd": args.autoencoder_wd,
}

# Set up resume pointer
resume_ptr = {"cell_idx": -1, "fold_id": -1, "iteration_id": -1}
if args.resume:
    p = load_progress(progress_path)
    if p and "last_completed" in p:
        resume_ptr = p["last_completed"]
        print2log(
            f"[Resume] Last completed: cell_idx={resume_ptr['cell_idx']} "
            f"fold_id={resume_ptr['fold_id']} iteration_id={resume_ptr['iteration_id']}"
        )

# ----------------------------------------------------------------------
# Main loops over cell lines / iterations / CV folds
# ----------------------------------------------------------------------
for ci, cell in enumerate(cell_lines):
    if ci < resume_ptr["cell_idx"]:
        continue

    print2log(f"Cell line to create artificially different inputs: {cell}")

    # Per-cell CSVs (load existing or start empty)
    recon_csv_path = os.path.join(
        output_dir,
        f"{cell}_Decoder_TransActPaired_reconstruction_differentInputs.csv",
    )
    trans_csv_path = os.path.join(
        output_dir,
        f"{cell}_Decoder_TransActPaired_translation_differentInputs.csv",
    )
    df_recon_all = _load_or_empty_csv(recon_csv_path)
    df_trans_all = _load_or_empty_csv(trans_csv_path)

    for j in range(random_iterations):
        if (ci == resume_ptr["cell_idx"]) and (j < resume_ptr["iteration_id"]):
            continue

        genes_1 = np.load(
            os.path.join(data_root, f"genes_1{cell}_iter{j}.npy"),
            allow_pickle=True,
        )
        genes_2 = np.setdiff1d(cmap.columns.values, genes_1)

        for fold_id in range(5):
            if (
                ci == resume_ptr["cell_idx"]
                and j == resume_ptr["iteration_id"]
                and fold_id <= resume_ptr["fold_id"]
            ):
                continue

            print2log(
                f"=== Start fold {fold_id} in iteration {j}/{random_iterations} for cell {cell} ==="
            )

            trainInfo = pd.read_csv(
                os.path.join(data_root, cell, f"train_{fold_id}.csv"), index_col=0
            )
            valInfo = pd.read_csv(
                os.path.join(data_root, cell, f"val_{fold_id}.csv"), index_col=0
            )

            if len(trainInfo) < 950:
                bs = 256
            else:
                bs = model_params["batch_size"]

            cmap_train = cmap.loc[trainInfo.sig_id, :]
            cmap_val = cmap.loc[valInfo.sig_id, :]

            X_1 = torch.tensor(cmap_train.loc[:, genes_1].values).float().to(device)
            X_2 = torch.tensor(cmap_train.loc[:, genes_2].values).float().to(device)
            X_1_val = torch.tensor(cmap_val.loc[:, genes_1].values).float().to(device)
            X_2_val = torch.tensor(cmap_val.loc[:, genes_2].values).float().to(device)

            # ------------------------------------------------------------------
            # Align the samples in the consensus space using GPU TRANSACT
            # ------------------------------------------------------------------
            align_device = "cuda" if device.type == "cuda" else "cpu"
            _, _, _, initial_alignment_model = transact_align_gpu(
                X_2,  # source → will become Z_source
                X_1,  # target → will become Z_target
                n_src_pcs=75,
                n_tgt_pcs=75,
                n_pv=30,
                kernel="rbf",
                gamma=5e-4,
                device=align_device,
            )

            # Training data in consensus space
            X_2_aligned = (
                transact_transform_gpu(X_2, initial_alignment_model, space="source")
                .detach()
                .cpu()
                .numpy()
            )
            X_1_aligned = (
                transact_transform_gpu(X_1, initial_alignment_model, space="target")
                .detach()
                .cpu()
                .numpy()
            )

            # Latent dimension equals consensus dimension
            model_params["latent_dim"] = X_2_aligned.shape[1]

            # Validation data in consensus space
            X_2_val_aligned = (
                transact_transform_gpu(X_2_val, initial_alignment_model, space="source")
                .detach()
                .cpu()
                .numpy()
            )
            X_1_val_aligned = (
                transact_transform_gpu(X_1_val, initial_alignment_model, space="target")
                .detach()
                .cpu()
                .numpy()
            )

            # Initialize models for the fold
            decoder_1 = Decoder(
                model_params["latent_dim"],
                model_params["decoder_hiddens"],
                X_1.shape[1],
                dropRate=model_params["dropout_decoder"],
                bn=model_params["bn_decoder"],
                activation=model_params["decoder_activation"],
                dropIn=model_params["dropout_input_decoder"],
            ).to(device)
            decoder_2 = Decoder(
                model_params["latent_dim"],
                model_params["decoder_hiddens"],
                X_2.shape[1],
                dropRate=model_params["dropout_decoder"],
                bn=model_params["bn_decoder"],
                activation=model_params["decoder_activation"],
                dropIn=model_params["dropout_input_decoder"],
            ).to(device)

            # First pretrain autoencoders for each biological context
            r1, decoder_1 = train_decoder_fold(
                model_params,
                device,
                X_1,
                torch.tensor(X_1_aligned, dtype=torch.double).to(device),
                decoder_1,
                bs,
                model_params["epochs"],
            )
            r2, decoder_2 = train_decoder_fold(
                model_params,
                device,
                X_2,
                torch.tensor(X_2_aligned, dtype=torch.double).to(device),
                decoder_2,
                bs,
                model_params["epochs"],
            )
            print2log("Autoencoders training performance:")
            print2log(f"Fold {fold_id}: {r1}, {r2}")

            # Validation sets
            x_1_equivalent_val = torch.tensor(
                X_1_val_aligned, dtype=torch.double
            ).to(device)
            x_2_equivalent_val = torch.tensor(
                X_2_val_aligned, dtype=torch.double
            ).to(device)
            x_1_equivalent_train = torch.tensor(
                X_1_aligned, dtype=torch.double
            ).to(device)
            x_2_equivalent_train = torch.tensor(
                X_2_aligned, dtype=torch.double
            ).to(device)

            decoder_1.eval()
            decoder_2.eval()
            with torch.no_grad():
                # Training translation performance
                x_hat_2_equivalent_train = decoder_2(x_1_equivalent_train)
                x_hat_1_equivalent_train = decoder_1(x_2_equivalent_train)
                pearson_1_to_2 = pearson_r(
                    x_hat_2_equivalent_train.flatten(), X_2.flatten()
                ).detach().cpu().numpy()
                pearson_2_to_1 = pearson_r(
                    x_hat_1_equivalent_train.flatten(), X_1.flatten()
                ).detach().cpu().numpy()

                # Reconstruction on validation
                yhat2 = decoder_2(
                    torch.tensor(X_2_val_aligned, dtype=torch.double).to(device)
                )
                yhat1 = decoder_1(
                    torch.tensor(X_1_val_aligned, dtype=torch.double).to(device)
                )
                pearson1 = pearson_r(
                    yhat1.flatten(), X_1_val.flatten()
                ).detach().cpu().numpy()
                pearson2 = pearson_r(
                    yhat2.flatten(), X_2_val.flatten()
                ).detach().cpu().numpy()

                # Translation on validation
                x_hat_2_equivalent_val = decoder_2(x_1_equivalent_val)
                x_hat_1_equivalent_val = decoder_1(x_2_equivalent_val)
                r_1_to_2 = pearson_r(
                    x_hat_2_equivalent_val.flatten(), X_2_val.flatten()
                ).detach().cpu().numpy()
                r_2_to_1 = pearson_r(
                    x_hat_1_equivalent_val.flatten(), X_1_val.flatten()
                ).detach().cpu().numpy()

                mu_r = 0.5 * (np.nanmean(pearson1) + np.nanmean(pearson2))
                mu_r_translation = 0.5 * (np.nanmean(r_1_to_2) + np.nanmean(r_2_to_1))

            print2log(
                f"Fold {fold_id}: r = {mu_r:.4f}, r_translation = {mu_r_translation:.4f}"
            )

            # -------------------- PER-FOLD DATAFRAMES (THIS FOLD ONLY) --------------------
            # Reconstruction metrics
            tmp_recon = pd.DataFrame(
                {"train": 0.5 * (r1 + r2), "test": 0.5 * (pearson1 + pearson2)},
                index=[fold_id],
            )
            tmp_recon["fold"] = fold_id
            tmp_recon["cell"] = cell
            tmp_recon["iteration"] = j

            # Translation metrics
            tmp_trans = pd.DataFrame(
                {
                    "train": 0.5 * (pearson_1_to_2 + pearson_2_to_1),
                    "test": 0.5 * (r_1_to_2 + r_2_to_1),
                },
                index=[fold_id],
            )
            tmp_trans["fold"] = fold_id
            tmp_trans["cell"] = cell
            tmp_trans["iteration"] = j

            # Append, deduplicate, and atomically write
            recon_keys = ["cell", "fold", "iteration"]
            trans_keys = ["cell", "fold", "iteration"]
            df_recon_all = _append_and_write_safely(
                df_recon_all, tmp_recon, recon_csv_path, dedup_subset=recon_keys
            )
            df_trans_all = _append_and_write_safely(
                df_trans_all, tmp_trans, trans_csv_path, dedup_subset=trans_keys
            )

            print2log(
                f"[Saved] {cell} fold={fold_id} iteration={j}/{random_iterations} "
                f"reconstruction r (mean) train={np.nanmean(0.5*(r1+r2)):.4f} "
                f"test={np.nanmean(0.5*(pearson1+pearson2)):.4f}"
            )
            print2log(
                f"[Saved] {cell} fold={fold_id} iteration={j}/{random_iterations} "
                f"translation r (mean) train={np.nanmean(pearson_1_to_2):.4f} "
                f"test={np.nanmean(r_1_to_2):.4f}"
            )

            # Save progress pointer at end of fold
            save_progress(progress_path, cell_idx=ci, fold_id=fold_id, iteration_idx=j)
            print2log(
                f"=== END fold {fold_id} in iteration {j}/{random_iterations} for {cell} ==="
            )

    print2log("Completely finished " + cell)
