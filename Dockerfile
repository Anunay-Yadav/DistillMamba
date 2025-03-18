FROM nvcr.io/nvidia/pytorch:23.10-py3
MAINTAINER Anunay Yadav<anunay.yadav@epfl.ch>
ARG DEBIAN_FRONTEND=noninteractive
# package install
RUN apt-get update &&  apt-get install -y \
    curl vim htop\
    ca-certificates \
    openssh-server \
    cmake \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    zip \
    unzip ssh \
    tmux \
 && rm -rf /var/lib/apt/lists/*
RUN wget https://github.com/acollins3/triton/releases/download/triton-2.1.0-arm64/triton-2.1.0-cp310-cp310-linux_aarch64.whl
RUN pip install triton-2.1.0-cp310-cp310-linux_aarch64.whl
RUN git clone https://github.com/Dao-AILab/causal-conv1d.git /causal-conv1d && \
    cd /causal-conv1d && \
    python setup.py install
RUN pip --no-cache-dir install \
    easydict \
    h5py \
    pyyaml \
    tqdm \
    flake8 \
    pillow \
    protobuf \
    seaborn \
    scipy \
    scikit-learn \
    wandb \
    hydra-core \
    transformers \
    datasets \
    evaluate \
    accelerate \
    sentencepiece \
    mamba-ssm[causal_conv1d]==2.2.2 \
    torchmetrics
RUN pip install --upgrade protobuf==3.20.0

RUN wget https://github.com/state-spaces/mamba/archive/refs/tags/v2.2.2.tar.gz && \
    tar -xvzf v2.2.2.tar.gz && \
    cd mamba-2.2.2 && \
    python setup.py install

RUN MAX_JOBS=4 pip install flash-attn==2.6.3 peft==0.12.0 huggingface-hub==0.24.5 deepspeed==0.12.2 trl==0.8.6 transformers==4.43.1 --no-build-isolation
RUN pip install transformers -U