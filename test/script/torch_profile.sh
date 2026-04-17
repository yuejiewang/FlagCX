#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLAGCX_DEBUG=INFO
export FLAGCX_DEBUG_SUBSYS=INIT
export FLAGCX_USE_HETERO_COMM=1
export FLAGCX_REDUCE_FIFO_CAPACITY=1024
export FLAGCX_P2P_EVENT_POOL_SIZE=1024
export FLAGCX_UNIRUNNER_NSLICES=2
export FLAGCX_UNIRUNNER_NTHREADS=32
export FLAGCX_UNIRUNNER_NBLOCKS=24
export FLAGCX_UNIRUNNER_USE_SLICEDAR=1

CMD_BASE='torchrun --nproc_per_node 8 --nnodes=1 --node_rank=0 --master_addr="localhost"'
PY_SCRIPT='./plugin/torch/example/example.py'

echo "[INFO] Launching PyTorch API tests in heterogeneous mode"
while true; do
    PORT=$(shuf -i 20000-65535 -n 1)
    (echo >/dev/tcp/127.0.0.1/$PORT) &>/dev/null || break
done
CMD="$CMD_BASE --master_port=$PORT $PY_SCRIPT --op allreduce-profile"
echo "$CMD"
eval "$CMD"
echo "[INFO] Completed PyTorch API tests in heterogeneous mode"
echo "--------------------------------------------------------"
