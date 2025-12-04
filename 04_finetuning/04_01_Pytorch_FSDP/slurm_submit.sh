#!/bin/bash -l

#SBATCH --job-name=FSDP_finetune      # Job name
#SBATCH --output=%x-%j.out            # Standard output log file (%x = job name, %j = job ID)
#SBATCH --error=%x-%j.err             # Standard error log file
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=2                    # Number of tasks (one per GPU)
#SBATCH --cpus-per-task=10            # Number of CPU cores per task
#SBATCH --gres=gpu:2                  # Number of GPUs allocated
#SBATCH --time=04:00:00               # Time limit (HH:MM:SS)
#SBATCH --partition=gpu               # Partition for GPU nodes
#SBATCH --reservation=SCA             # Reservation (if required)

# =====================
# Load Environment
# =====================

module purge
module load miniconda

conda activate tutorial                      # Activate the desired Conda environment

# =============================
# 🔹 Set Training Parameters
# =============================

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Total batch size (adjustable)
total_batch_size=${TOTAL_BATCH_SIZE:-8}

# Per-GPU batch size (adjustable based on memory)
batch_size_per_device=${BATCH_SIZE_PER_DEVICE:-4}

# Get the number of available GPUs
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')

# Compute gradient accumulation steps
gradient_accumulation_steps=$(( total_batch_size / batch_size_per_device / num_gpus ))

# =============================
# 🔹 Debugging Information
# =============================
echo "------------------------"
echo " Training Configuration:"
echo " Using $num_gpus GPUs"
echo " Total batch size: $total_batch_size"
echo " Batch size per device: $batch_size_per_device"
echo " Gradient accumulation steps: $gradient_accumulation_steps"
echo "------------------------"

# =============================
# Launch Distributed Training
# =============================
torchrun \
    --nnodes=1 \
    --nproc-per-node=$num_gpus \
    chess.py \
        --batch_size_per_device $batch_size_per_device \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        "$@"
