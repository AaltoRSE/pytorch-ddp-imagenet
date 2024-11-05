#!/bin/bash
#SBATCH --job-name=run_lightning
#SBATCH --output=run_lightning_%j.out
#SBATCH --error=run_lightning_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
# SBATCH --partition=gpu-debug
# Ensure the script stops on the first error
set -e

# Load necessary modules
module load scicomp-python-env
module load parallel

# Extract the ImageNet dataset to the local disk
./extract_images.sh

# Set the data directory
DATA_DIR="/tmp/ILSVRC2012_extracted"

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

BATCH_SIZE=512
EPOCHS=10
LEARNING_RATE=0.001

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
  echo "Error: Data directory $DATA_DIR does not exist."
  exit 1
fi

# Run the Python script with specified arguments
srun python3 lightning_imagenet.py \
    --data_dir "$DATA_DIR" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE
