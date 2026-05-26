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
#SBATCH -J flow_s1_no_guidance
#SBATCH -o logs/slurm_%x_%j.out
#SBATCH -e logs/slurm_%x_%j.err

module load miniforge/24.3.0-0
mamba activate nikos

# Supplementary Figure S1 ablation:
# Run the shared-feature L1000 FlowTransOP benchmark with the structural
# guidance term disabled. The only intentional model change from the main
# shared-feature FlowTransOP run is --conditional_flow_lambda 0.
mkdir -p logs
LOG=logs/supplementary_figure_s1_no_structural_guidance.log
echo "[$(date)] Job $SLURM_JOB_ID starting on $HOSTNAME" >> "$LOG"

python3 ./AutoTransOP_Pretrain_FlowMatch.py \
  --output_dir ../results/FlowMatch_no_structural_guidance/ \
  --conditional_flow_lambda 0 \
  --epochs 1000 \
  >> "$LOG" 2>&1

echo "[$(date)] Job $SLURM_JOB_ID finished completely!" >> "$LOG"
