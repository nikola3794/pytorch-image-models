#!/bin/bash
#BSUB -o /cluster/home/nipopovic/job_logs  # path to output file
#BSUB -W 23:59 # HH:MM runtime
#BSUB -n 8 # number of cpu cores
#BSUB -R "rusage[mem=4096]" # MB per CPU core
#BSUB -R "rusage[ngpus_excl_p=1]" # number of GPU cores
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
tar -I pigz -xf /cluster/work/cvl/yawli/data/ILSVRC2012.tar.gz -C ${TMPDIR}/
cp -R /cluster/work/cvl/nipopovic/data/ImageNet/2012-1k/partitions ${TMPDIR}/ILSVRC2012/
echo "Content of directory where data is transfered $(ls ${TMPDIR})"

# Set project paths
PROJECT_ROOT_DIR=/cluster/project/cvl/nipopovic/code/pytorch-image-models
export PYTHONPATH=${PYTHONPATH}:${PROJECT_ROOT_DIR}
cd ${PROJECT_ROOT_DIR}
pwd

#export OMP_NUM_THREADS=8

# CUDA_LAUNCH_BLOCKING=1 for debugging cuda errors
BATCH_SIZE=32
DATA_TMP_DIR=${TMPDIR}/ILSVRC2012

echo ${DATA_TMP_DIR}

EPS1=0.05
 
python3 validate_adversarial.py --model=resnet34_s32_trf_frac_just_v_1 --eps=${EPS1} --checkpoint=/cluster/work/cvl/nipopovic/experiments/ImageNet/SELECTED_EXPERIMENTS/fractal_transformers_classifier/20_%_tr/20210426-174809-resnet34_s32_trf_frac_just_v_1/last.pth.tar --data_dir=${DATA_TMP_DIR}  --batch-size=${BATCH_SIZE} 

python3 validate_adversarial.py --model=resnet50 --eps=${EPS1} --checkpoint=/cluster/work/cvl/nipopovic/experiments/ImageNet/SELECTED_EXPERIMENTS/resnet_backbones/20_%_tr/20210428-192038-resnet50/last.pth.tar --data_dir=${DATA_TMP_DIR}  --batch-size=${BATCH_SIZE} 
python3 validate_adversarial.py --model=resnet101 --eps=${EPS1} --checkpoint=/cluster/work/cvl/nipopovic/experiments/ImageNet/SELECTED_EXPERIMENTS/resnet_backbones/20_%_tr/20210429-040649-resnet101/last.pth.tar --data_dir=${DATA_TMP_DIR}  --batch-size=${BATCH_SIZE} 
