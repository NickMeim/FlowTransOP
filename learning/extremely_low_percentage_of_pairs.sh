#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --mail-user=meimetis@mit.edu
#SBATCH --mail-type=ALL
#SBATCH -c 32
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --mem=500G
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH -J flow_extremely_low_pairs

module load miniforge/24.3.0-0
mamba activate nikos

# Ensure logs append across requeues
mkdir -p logs
LOG=logs/FlowMatch_lowPairsPercentage_extreme.log

echo "[$(date)] Job $SLURM_JOB_ID starting on $HOSTNAME" >> "$LOG"
python3 ./AutoTransOP_Pretrain_FlowMatch_lowPairsPercentageExtreme.py --checkpoint_dir chkpts_training_lowPairsPercentage_extreme --log_file logs/FlowMatch_lowPairsPercentage_extreme.log --resume 1 >> "$LOG" 2>&1
echo "[$(date)] Job $SLURM_JOB_ID finished completely!" >> "$LOG"

