#!/bin/bash
#SBATCH -p pi_lauffen
#SBATCH --mail-user=meimetis@mit.edu
#SBATCH --mail-type=ALL
#SBATCH -c 28
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --mem=76G
#SBATCH --gres=gpu:h200:2
#SBATCH --time=48:00:00
#SBATCH -J flow_s1_no_guidance_rev
#SBATCH -o logs/slurm_%x_%j.out
#SBATCH -e logs/slurm_%x_%j.err

module load miniforge/24.3.0-0
mamba activate nikos

# Supplementary Figure S1 reverse-direction ablation:
# Run only the dataset2 -> dataset1 shared-feature L1000 FlowTransOP direction
# with the structural guidance term disabled. This adds flow21 outputs to the
# same no-structural-guidance result folder without rerunning flow12.
mkdir -p logs
LOG=logs/supplementary_figure_s1_no_structural_guidance_reverse.log
echo "[$(date)] Job $SLURM_JOB_ID starting on $HOSTNAME" >> "$LOG"

python3 ./AutoTransOP_Pretrain_FlowMatch_reverse.py \
  --output_dir ../results/FlowMatch_no_structural_guidance/ \
  --conditional_flow_lambda 0 \
  --epochs 1000 \
  >> "$LOG" 2>&1

echo "[$(date)] Job $SLURM_JOB_ID finished completely!" >> "$LOG"
