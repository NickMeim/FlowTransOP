#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --mail-user=meimetis@mit.edu
#SBATCH --mail-type=ALL
#SBATCH -c 32
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --mem=250G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH -J flow_extreme_withpairs

module load miniforge/24.3.0-0
mamba activate nikos

# Ensure logs append across requeues
mkdir -p logs
LOG=logs/FlowMatch_lowPairsPercentage_extreme_withPairs.log

echo "[$(date)] Job $SLURM_JOB_ID starting on $HOSTNAME" >> "$LOG"
python3 ./AutoTransOP_Pretrain_FlowMatch_lowPairsPercentageExtreme_withPairs.py --checkpoint_dir chkpts_training_lowPairsPercentage_extreme_withPairs --log_file logs/FlowMatch_lowPairsPercentage_extreme_withPairs.log --resume 1 >> "$LOG" 2>&1
echo "[$(date)] Job $SLURM_JOB_ID finished completely!" >> "$LOG"

