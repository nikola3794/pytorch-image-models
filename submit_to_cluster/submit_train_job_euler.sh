#!/bin/bash
#BSUB -o /cluster/home/nipopovic/job_logs  # path to output file
#BSUB -W 71:59 # HH:MM runtime
#BSUB -n 16 # number of cpu cores
#BSUB -R "rusage[mem=8192]" # MB per CPU core
#BSUB -R "rusage[ngpus_excl_p=4]" # number of GPU cores
#BSUB -R "select[gpu_mtotal0>=10240]" # MB per GPU core

# Activate python environment
source /cluster/home/nipopovic/python_envs/cls_models/bin/activate

# Access to internet to download torch models
module load eth_proxy
# For parallel data unzipping
module load pigz

# Print number of GPU/CPU resources available
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Number of CPU threads/core: $(nproc --all)"

# Transfer ImageNet to scratch
#tar -I pigz -xf /cluster/work/cvl/yawli/data/ILSVRC2012.tar.gz -C ${TMPDIR}/
#cp -R /cluster/work/cvl/nipopovic/data/ImageNet/2012-1k/partitions ${TMPDIR}/ILSVRC2012/

# Set project paths
PROJECT_ROOT_DIR=/cluster/project/cvl/nipopovic/code/pytorch-image-models
export PYTHONPATH=${PYTHONPATH}:${PROJECT_ROOT_DIR}
cd ${PROJECT_ROOT_DIR}
pwd

#export OMP_NUM_THREADS=8

./distributed_train.sh 4 ${PROJECT_ROOT_DIR}/params_1.yaml
# CUDA_LAUNCH_BLOCKING=1 for debugging cuda errors
