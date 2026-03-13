#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --mail-user=meimetis@mit.edu
#SBATCH --mail-type=ALL
#SBATCH -c 32
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --mem=500G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH -J autotransop_flow

module load miniforge/24.3.0-0
mamba activate nikos

# Ensure logs append across requeues
mkdir -p logs
LOG=logs/DecodeFromConsensus_diffenetInputs_bracketed.log
echo "[$(date)] Job $SLURM_JOB_ID starting on $HOSTNAME" >> "$LOG"

python3 DecodeFromConsencusSpace_diffenetInputs_bracketed.py --checkpoint_dir chkpts_decoder_diffenetInputs_bracketed --log_file logs/DecodeFromConsensus_diffenetInputs_bracketed.log --resume 1 >> "$LOG" 2>&1

echo "[$(date)] Job $SLURM_JOB_ID finished completely!" >> "$LOG"

