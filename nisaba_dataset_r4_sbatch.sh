#!/bin/bash
# **************************
# MODIFY THESE OPTIONS - setup for nisaba.nist.gov

#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --job-name=base_trojai
#SBATCH -o log-%N.%j.out
#SBATCH --time=48:0:0

printf -v numStr "%08d" ${1}
echo "id-$numStr"

source /apps/anaconda3/etc/profile.d/conda.sh
conda activate r4venv

source ./evaluate_models_round4.sh

