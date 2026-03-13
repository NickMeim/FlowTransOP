#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --mail-user=meimetis@mit.edu
#SBATCH --mail-type=ALL
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --mem=250G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH -J autotransop_flow

module load miniforge/24.3.0-0
mamba activate nikos

# Ensure logs append across requeues
mkdir -p logs
LOG=logs/AutoTransOP_Pretrain_FlowMatch_withSTRUCTURE.log

#python3 ./AutoTransOP_Pretrain_FlowMatch.py #--latent_dim 30 --folders MCF7_HA1E MCF7_PC3 PC3_HA1E
#python3 ./AutoTransOP_Pretrain_FlowMatch_withPairs.py --folders MCF7_HA1E MCF7_PC3 PC3_HA1E
#python3 ./AutoTransOP_Comparison.py --distance_reg 10 --reg_adv 1000 --adv_penalnty 100 --reg_classifier 100 --adversary_steps 5
python3 ./AutoTransOP_Pretrain_FlowMatch_withSTRUCTURE.py --resume 1 >> "$LOG" 2>&1 &
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
