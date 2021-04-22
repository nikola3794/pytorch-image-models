#!/bin/bash
NUM_PROC=$4
shift
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC train.py "$@"

