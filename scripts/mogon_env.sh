#!/usr/bin/env bash

module purge
module load tools/Meson/1.4.0-GCCcore-13.3.0
module use /apps/easybuild/2025/cuda/modules/all
module load tools/Ninja
module load system/CUDA

export PKG_CONFIG_PATH=/apps/easybuild/cuda/software/NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0/lib/pkgconfig:$PKG_CONFIG_PATH

export http_proxy=http://webproxy.zdv.uni-mainz.de:8888
export https_proxy=$http_proxy