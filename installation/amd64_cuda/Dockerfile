# Adapted from:
# Pytorch Dockerfile: https://github.com/pytorch/pytorch/blob/master/Dockerfile

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04


# Install system/build dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev && \
    rm -rf /var/lib/apt/lists/*

# Create user.
ARG USER
ARG UID
RUN useradd -u $UID --create-home $USER
USER $USER
WORKDIR /home/$USER
ENV PATH="$PATH:/home/$USER/.local/bin"

# Install Miniforge.
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O miniforge.sh \
    && chmod +x miniforge.sh \
    && ./miniforge.sh -b -p $HOME/miniforge3 \
    && rm miniforge.sh \
    && $HOME/miniforge/bin/conda init bash

RUN conda create -n behavior-transformer && conda activate behavior-transformer

# Install Pytorch build dependencies
RUN conda install python=3.8 astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses

# Install pytorch from source



