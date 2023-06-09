#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-38423}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}

# --load-from ./work_dirs/binsformer_swint_w7_2000C/pre_iter_16000.pth 