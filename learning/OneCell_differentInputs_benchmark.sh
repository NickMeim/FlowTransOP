#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --mail-user=meimetis@mit.edu
#SBATCH --mail-type=ALL
#SBATCH -c 32
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --mem=500G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH -J autotransop_flow

module load miniforge/24.3.0-0
mamba activate nikos

# Ensure logs append across requeues
mkdir -p logs
LOG=logs/AutoTransOP_Pretrain_FlowMatch_differentInputs_bracketed.log

python3 ./AutoTransOP_Pretrain_FlowMatch_differentInputs_bracketed.py --epochs 1000 --conditional_flow_lambda 1e-2 --resume 1 >> "$LOG" 2>&1 &
PY_PID=$!

# forward USR1 to python, then requeue
_graceful_requeue() {
  echo "[SLURM] USR1 received; forwarding to python (pid=$PY_PID) and waiting..."
  kill -USR1 $PY_PID 2>/dev/null || true
  wait $PY_PID 2>/dev/null
  echo "[SLURM] Requeuing job $SLURM_JOB_ID"
  scontrol requeue $SLURM_JOB_ID
}
trap _graceful_requeue USR1

wait $PY_PID
