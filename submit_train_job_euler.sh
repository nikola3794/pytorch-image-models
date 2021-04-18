#!/bin/bash
#BSUB -o /cluster/home/nipopovic/job_logs  # path to output file
#BSUB -W 71:59 # HH:MM runtime
#BSUB -n 32 # number of cpu cores
#BSUB -R "rusage[mem=4000]" # MB per CPU core
#BSUB -R "rusage[ngpus_excl_p=4]" # number of GPU cores
#BSUB -R "select[gpu_mtotal0>=10240]" # MB per GPU core

# TODO: How to choose which GPUs to use ?????

# Activate python environment
source /cluster/home/nipopovic/python_envs/cls_models/bin/activate

# Access to internet to download torch models
module load eth_proxy

# Print number of GPU cores available
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

echo "Number of CPU threads/core: $(nproc --all)"

# Set paths
PROJECT_ROOT_DIR=/cluster/project/cvl/nipopovic/code/pytorch-image-models
export PYTHONPATH=${PYTHONPATH}:${PROJECT_ROOT_DIR}
cd ${PROJECT_ROOT_DIR}
pwd

#export OMP_NUM_THREADS=8

# python -u run_experiment/imitation_learning/main.py "$@"
./distributed_train.sh 4
# CUDA_LAUNCH_BLOCKING=1 for debugging cuda errors
