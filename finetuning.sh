#!/usr/bin/env bash
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=400GB # 800GB for Spice
#SBATCH --time=23:30:00
#SBATCH --output=slurm_logs/log-%j.out
#SBATCH --error=slurm_logs/log-%j.err
#SBATCH --export=ALL

set -eo pipefail

echo "== JOB START: $SLURM_JOB_ID on $(hostname) =="

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
ENV_NAME="bioink"
conda activate "$ENV_NAME"

echo "Activated env: $CONDA_PREFIX"
echo "Python path: $(which python)"

echo "CUDA_VISIBLE_DEVICES set to $CUDA_VISIBLE_DEVICES"

# -----------------------------
# Run finetuning
# -----------------------------
echo "== Running finetuning from CLI =="
echo "Running from directory: $(pwd)"

START_TIME=$(date +%s)

echo "== JOB START: $(date) =="

echo "======================================================="

srun python -m mace.cli.run_train \
    --model="MACE" \
    --stress_weight=0.0 \
    --forces_weight=100.0 \
    --energy_weight=1.0 \
    --lr=0.001 \
    --scheduler="ReduceLROnPlateau" \
    --lr_factor=0.8 \
    --scheduler_patience=10 \
    --foundation_model="path-to-your-foundation-model" \
    --multiheads_finetuning=False \
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
    --ema \
    --ema_decay=0.999 \
    --amsgrad \
    --device=cuda \
    --batch_size=32 \
    --max_num_epochs=1000 \
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
