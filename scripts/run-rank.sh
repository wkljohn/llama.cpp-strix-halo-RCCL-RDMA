#!/bin/bash
B=/tmp/rocm-runtime-bundle
export LD_LIBRARY_PATH=$B/lib
export ROCM_PATH=$B HIP_DEVICE_LIB_PATH=$B/amdgcn HSA_OVERRIDE_GFX_VERSION=11.5.1
export GGML_NCCL_RANK=$1 GGML_NCCL_WORLD=2 GGML_NCCL_MASTER=10.4.0.1:29500
export NCCL_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN NCCL_CUMEM_ENABLE=0
cd $B/bin && ./test-world-allreduce --layers 60 --iters 30 --ne 6144
