#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --mail-user=meimetis@mit.edu
#SBATCH --mail-type=ALL
#SBATCH -c 32
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --mem=250G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH -J flow_low_withpairs

module load miniforge/24.3.0-0
mamba activate nikos

# Ensure logs append across requeues
mkdir -p logs
LOG=logs/Paired_FlowMatch_lowPairsPercentage.log

echo "[$(date)] Job $SLURM_JOB_ID starting on $HOSTNAME" >> "$LOG"
python3 ./AutoTransOP_Pretrain_FlowMatchPaired_lowPairsPercentage.py --checkpoint_dir paired_chkpts_training_lowPairsPercentage --log_file logs/Paired_FlowMatch_lowPairsPercentage.log --resume 1 >> "$LOG" 2>&1
echo "[$(date)] Job $SLURM_JOB_ID finished completely!" >> "$LOG"

