#!/bin/bash
#SBATCH --job-name=ddp_imagenet              # Job name
#SBATCH --nodes=1                            # Number of nodes (can not be more than 1 on triton)
#SBATCH --ntasks=2                           # Number of tasks (usually equals number of GPUs)
#SBATCH --gres=gpu:2  
#SBATCH --cpus-per-task=8                    # Number of CPU cores per task
#SBATCH --mem=100G                            # Memory per node
#SBATCH --time=02:00:00                      # Maximum time for the job
#SBATCH --output=ddp_training_%j.out         # Standard output and error log (%j inserts the job ID)
#SBATCH --error=ddp_training_%j.err          # Error log (%j inserts the job ID)

# Load necessary modules 
module load scicomp-python-env

# Extract the ImageNet dataset to the local disk
./extract_images.sh

# Set the data directory
DATA_DIR="/tmp/ILSVRC2012_extracted"

## Launch the training using srun
srun python ddp_imagenet.py --data_dir $DATA_DIR --batch_size 512 --epochs 5
