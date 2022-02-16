#!/bin/bash
#SBATCH --job-name=mountain_car_ddqn
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 2

# Source the created env.sh file to get mujoco and python3.6
singularity exec --nv --bind $HOME/mujocopy-buildlock:/ext3/miniconda3/lib/python3.9/site-packages/mujoco_py/generated/mujocopy-buildlock --overlay /scratch/ig2283/tmp/overlay-5GB-200K.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "

source /ext3/env.sh
conda activate hw2_drq

python main.py

"