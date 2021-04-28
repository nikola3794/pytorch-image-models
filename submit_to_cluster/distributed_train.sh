#!/bin/bash
NUM_PROC=$1
# Shift is a builtin command in bash which after getting executed, shifts/move the command line arguments to one position left. The first argument is lost after using shift command. 
shift
echo "Executing:python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC train.py --config=$1 --data_dir=${TMPDIR}/ILSVRC2012"
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC train.py --config=$1 --data_dir=${TMPDIR}/ILSVRC2012 #"$@"
