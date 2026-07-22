#!/bin/bash
set -e
cd /home/wkljohn/Desktop/cc
export ROCM_PATH=/opt/rocm-7.2.0
if [ ! -d rocm-systems ]; then
  git clone --depth 1 -b gfx1151-rccl https://github.com/kyuz0/rocm-systems.git 2>&1
fi
cd rocm-systems/projects/rccl
mkdir -p build_gfx1151 && cd build_gfx1151
CXX=$ROCM_PATH/bin/hipcc cmake .. \
  -DCMAKE_CXX_COMPILER=$ROCM_PATH/bin/hipcc \
  -DDEFAULT_GPUS="gfx1151" -DGPU_TARGETS="gfx1151" -DAMDGPU_TARGETS="gfx1151" \
  -DCMAKE_INSTALL_PREFIX=./install -DBUILD_TESTS=OFF -DGENERATE_SYM_KERNELS=OFF \
  -DENABLE_AMDSMI=OFF -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
make -j$(nproc) 2>&1 | tail -3
echo "RCCL_BUILD_DONE_$?"
ls -la librccl.so.1 2>/dev/null
