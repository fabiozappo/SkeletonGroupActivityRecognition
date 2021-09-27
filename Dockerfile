FROM nvidia/cuda:10.0-cudnn7-devel
# Ubuntu 18.04.5 LTS
# CUDA 10.0.130
# cuDNN 7.6.05

# install required Ubuntu packages
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y \
   # OpenPose
   lsb-release sudo git cmake libopencv-dev \
   # Skeleton Group Activity Recognition
   libgl1-mesa-glx python3-dev python3-pip


#
# OpenPose
#
# compile OpenPose from source
WORKDIR /openpose/
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose . && \
    git submodule update --init --recursive --remote && \
    bash ./scripts/ubuntu/install_deps.sh

WORKDIR /openpose/build
ARG DOWNLOAD_MODELS=ON
RUN cmake \
      -DBUILD_PYTHON=ON \
      # in case you run into 'file DOWNLOAD HASH mismatch' 'status: [7;"Couldn't connect to server"]' use 'docker build . --build-arg DOWNLOAD_MODELS=OFF'
      -DDOWNLOAD_BODY_25_MODEL=$DOWNLOAD_MODELS -DDOWNLOAD_BODY_MPI_MODEL=$DOWNLOAD_MODELS -DDOWNLOAD_HAND_MODEL=$DOWNLOAD_MODELS -DDOWNLOAD_FACE_MODEL=$DOWNLOAD_MODELS \
      ..
# fix 'nvcc fatal: Unsupported gpu architecture'
RUN sed -ie 's/set(AMPERE "80 86")/#&/g'  ../cmake/Cuda.cmake && \
    sed -ie 's/set(AMPERE "80 86")/#&/g'  ../3rdparty/caffe/cmake/Cuda.cmake
# fix 'recipe for target 'caffe/src/openpose_lib-stamp/openpose_lib-configure' failed'
RUN sed -i '5 i set(CUDA_cublas_device_LIBRARY "/usr/local/cuda-10.0/targets/x86_64-linux/lib/libcublas.so")' ../3rdparty/caffe/cmake/Cuda.cmake

RUN make -j$(nproc)
RUN make install
ENV PYTHONPATH "${PYTHONPATH}:/openpose/build/python/openpose"


#
# Learning Group Activities from Skeletons without Individual Action Labels
#
WORKDIR /work/sk-gar/
RUN git clone https://github.com/fabiozappo/SkeletonGroupActivityRecognition.git .

# install required python packages
RUN pip3 install -r requirements.txt

# download pretrained-3d-cnn weights
RUN pip3 install gdown && \
    gdown https://drive.google.com/uc?id=16aR8hNbinzk7nxj6LEmGqyopUt7GiLG8 -O Weights/p3d_flow_199.checkpoint.pth.tar && \
    gdown https://drive.google.com/uc?id=1slkxHCUCReJaVo8X2xOkef8ARPKiKx2B -O Weights/p3d_rgb_199.checkpoint.pth.tar