#!/bin/bash
#
#SBATCH --cpus-per-task=2
#SBATCH --time=00:01:00
#SBATCH --mem=10GB
#SBATCH --job-name=test_srgan
#SBATCH --mail-type=END
#SBATCH --mail-user=xz1757@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1


module load python3/intel/3.6.3
module load pytorch/python3.6/0.3.0_4
module load cuda/8.0.44
module load cudnn/8.0v5.1


time python3 test_image.py --image_name test_img.png
