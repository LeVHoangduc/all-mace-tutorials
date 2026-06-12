#!/usr/bin/env bash
#SBATCH --job-name=train-scratch
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=400GB 
#SBATCH --time=23:30:00
#SBATCH --output=slurm_logs/log-%j.out
#SBATCH --error=slurm_logs/log-%j.err
#SBATCH --export=ALL

set -eo pipefail

echo "== JOB START: $SLURM_JOB_ID on $(hostname) =="

# nodelist=ruche-gpu19 can run with 32 bs
# -----------------------------
# Load conda from module
# -----------------------------
module purge
module load anaconda3/2023.09-0/none-none

CONDA_BASE="$(conda info --base)"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    eval "$("$CONDA_BASE/bin/conda" shell.bash hook)"
fi

# -----------------------------
# Activate environment
# -----------------------------
ENV_NAME="ENV"
conda activate "$ENV_NAME"

echo "Activated env: $CONDA_PREFIX"
echo "Python path: $(which python)"

echo "CUDA_VISIBLE_DEVICES set to $CUDA_VISIBLE_DEVICES"

# -----------------------------
# Run from scratch
# -----------------------------

echo "== Running finetuning from CLI =="
echo "Running from directory: $(pwd)"

START_TIME=$(date +%s)

echo "== JOB START: $(date) =="

echo "======================================================="

# --hidden_irreps='128x0e + 128x1o' -> num_channels = 128, max_L = 1
# It indicates the angular channels up to l = 1 only: scalar 0e and vector 1o
# --max_ell is a hyperparameter in the interaction/message construction

srun python -m mace.cli.run_train \
    --model="MACE" \
    --stress_weight=0.0 \
    --forces_weight=1000.0 \
    --energy_weight=50.0 \
    --lr=0.005 \
    --hidden_irreps='128x0e + 128x1o' \
    --num_interactions=2 \
    --correlation=3 \
    --max_ell=3 \
    --r_max=6.0 \
    --name="model-name" \
    --model_dir="evals" \
    --log_dir="logs" \
    --checkpoints_dir="checkpoints" \
    --results_dir="results" \
    --train_file="path-to-train-dts" \
    --valid_fraction=0.1 \
    --test_file="path-to-test-dts" \
    --energy_key="DFT_energy" \
    --forces_key="DFT_forces" \
    --E0s="average" \
    --weight_decay=5e-7 \
    --ema \
    --ema_decay=0.999 \
    --clip_grad=10.0 \
    --amsgrad \
    --device=cuda \
    --batch_size=32 \
    --max_num_epochs=303 \
    --distributed \
    --enable_cueq=True \
    --seed=123 \
# ---- job body here ----

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "== JOB END: $(date) =="
echo "Elapsed time: $((ELAPSED/3600))h $(((ELAPSED%3600)/60))m $((ELAPSED%60))s"

# -----------------------------
# Print error summary
# -----------------------------
echo "== Error summary (if any) =="

echo "== JOB DONE =="