#!/bin/bash
#SBATCH --job-name=LTMU
#SBATCH --output=/home/yans/LTMU-rgbd/logs/log-demo-dimp-ltmu-output.txt
#SBATCH --error=/home/yans/LTMU-rgbd/logs/log-demo-dimp-ltmu-error.txt
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --time=7-00:00:00
#SBATCH --mem=64000
#SBATCH --partition=gpu --gres=gpu:teslav100:1

module load CUDA/10.0
module load fgci-common
module load ninja/1.9.0
module load all/libjpeg-turbo/2.0.0-GCCcore-7.3.0

cd /home/yans/LTMU-rgbd/DiMP_LTMU/

source activate DiMP_LTMU

python Demo.py

conda deactivate
