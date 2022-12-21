# syntax = docker/dockerfile:1.4
# The top line is used by BuildKit. _**DO NOT ERASE IT**_.
ARG BUILD_MODE
ARG USE_CUDA
ARG USE_PRECOMPILED_HEADERS=1
ARG MKL_MODE
ARG CUDA_VERSION=11.7.1
ARG CUDNN_VERSION=8
ARG PYTHON_VERSION=3.8
ARG LINUX_DISTRO=ubuntu
ARG DISTRO_VERSION=20.04
ARG TORCH_CUDA_ARCH_LIST
ARG BUILD_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-${LINUX_DISTRO}${DISTRO_VERSION}
ARG TRAIN_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-${LINUX_DISTRO}${DISTRO_VERSION}

########################################################################
FROM ${BUILD_IMAGE} AS install-ubuntu
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git && \
    rm -rf /var/lib/apt/lists/*

########################################################################
FROM install-${LINUX_DISTRO} AS install-base

LABEL maintainer=veritas9872@gmail.com
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ARG MKL_MODE
ARG PYTHON_VERSION
ENV PATH=/opt/conda/bin:${PATH}
# Available Miniconda installations: https://docs.conda.io/en/latest/miniconda.html
ARG CONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN curl -fsSL -v -o /tmp/miniconda.sh -O ${CONDA_URL} && \
    chmod +x /tmp/miniconda.sh && \
    /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    conda config --remove channels defaults && \
    conda config --append channels conda-forge && \
    conda install -y python=${PYTHON_VERSION} && \
    conda clean -ya

########################################################################
FROM install-base AS install-conda

# Get build requirements. Set package versions manually if compatibility issues arise.
COPY --link reqs/conda-build.requirements.txt /tmp/conda/build-requirements.txt
ENV conda=/opt/conda/bin/conda

########################################################################
FROM install-conda AS install-include-mkl

ARG MKL_VERSION=2022.2.1
ARG CUDA_VERSION
RUN $conda install -y \
        --file /tmp/conda/build-requirements.txt \
        pytorch::magma-cuda$(echo ${CUDA_VERSION} | sed 's/\.//; s/\..*//') \
        mkl-include==${MKL_VERSION} \
        mkl==${MKL_VERSION} && \
    conda clean -ya

# Enable Intel MKL optimizations on AMD CPUs.
# https://danieldk.eu/Posts/2020-08-31-MKL-Zen.html
ENV MKL_DEBUG_CPU_TYPE=5
RUN echo 'int mkl_serv_intel_cpu_true() {return 1;}' > /opt/conda/fakeintel.c && \
    gcc -shared -fPIC -o /opt/conda/libfakeintel.so /opt/conda/fakeintel.c
ENV LD_PRELOAD=/opt/conda/libfakeintel.so:${LD_PRELOAD}

# Use Intel OpenMP with optimizations enabled.
# Some compilers can use OpenMP for faster builds.
ENV KMP_BLOCKTIME=0
ENV LD_PRELOAD=/opt/conda/lib/libiomp5.so:${LD_PRELOAD}

########################################################################
FROM install-conda AS install-exclude-mkl

ARG CUDA_VERSION
RUN $conda install -y \
        --file /tmp/conda/build-requirements.txt \
        pytorch::magma-cuda$(echo ${CUDA_VERSION} | sed 's/\.//; s/\..*//') \
        nomkl && \
    conda clean -ya

########################################################################
FROM install-${MKL_MODE}-mkl AS build-base
# `build-base` is the base stage for all builds in the Dockerfile.

ENV LD_PRELOAD=/opt/conda/lib/libjemalloc.so:$LD_PRELOAD
# See the documentation for an explanation of the following configuration.
# https://android.googlesource.com/platform/external/jemalloc_new/+/6e6a93170475c05ebddbaf3f0df6add65ba19f01/TUNING.md
ENV MALLOC_CONF=background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000

WORKDIR /opt/ccache
ENV PATH=/opt/conda/bin/ccache:$PATH
# Enable `ccache` with unlimited memory size for faster builds.
RUN ccache --set-config=cache_dir=/opt/ccache && ccache --max-size 0

# Use LLD as the default linker for faster linking.
RUN ln -sf /opt/conda/bin/ld.lld /usr/bin/ld

# Use `ldconfig` to update link directories and include `conda` in dynamic linking.
# Setting `LD_LIBRARY_PATH` directly is bad practice.
RUN echo /opt/conda/lib >> /etc/ld.so.conf.d/conda.conf && ldconfig

########################################################################
FROM build-base AS build-torch

# Minimize downloads by only cloning shallow branches and not the full `git` history.
# If the build fails during `git clone`, just try again.
# Updating git submodules is not fail-safe.
# The reason for failure is likely due to networking issues during installation.
# See https://stackoverflow.com/a/8573310/9289275
WORKDIR /opt/pytorch
ARG PYTORCH_VERSION_TAG
ARG TORCH_URL=https://github.com/pytorch/pytorch.git
RUN git clone --jobs 0 --depth 1 --single-branch --shallow-submodules \
        --recurse-submodules --branch ${PYTORCH_VERSION_TAG} \
        ${TORCH_URL} /opt/pytorch

# Read `setup.py` and `CMakeLists.txt` to find build flags.
# Different flags are available for different versions of PyTorch.
# Variables without default values here recieve defaults from the top of the Dockerfile.
# Disabling Caffe2, NNPack, and QNNPack as they are legacy and most users do not need them.
ARG USE_CUDA
ARG USE_CUDNN=${USE_CUDA}
ARG USE_NNPACK=0
ARG USE_QNNPACK=0
ARG BUILD_TEST=0
ARG BUILD_CAFFE2=0
ARG BUILD_CAFFE2_OPS=0
ARG USE_PRECOMPILED_HEADERS
ARG TORCH_CUDA_ARCH_LIST
ARG CMAKE_PREFIX_PATH=/opt/conda
ARG TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
# Build wheel for installation in later stages.
# Install PyTorch for subsidiary libraries (e.g., TorchVision).
RUN --mount=type=cache,target=/opt/ccache \
    python setup.py bdist_wheel -d /tmp/dist && \
    python setup.py install

########################################################################
FROM build-torch AS build-vision

WORKDIR /opt/vision
ARG TORCHVISION_VERSION_TAG
ARG VISION_URL=https://github.com/pytorch/vision.git
RUN git clone --jobs 0 --depth 1 --single-branch --shallow-submodules \
        --recurse-submodules --branch ${TORCHVISION_VERSION_TAG} \
        ${VISION_URL} /opt/vision

ARG USE_CUDA
ARG USE_FFMPEG=1
ARG USE_PRECOMPILED_HEADERS
ARG FORCE_CUDA=${USE_CUDA}
ARG TORCH_CUDA_ARCH_LIST
RUN --mount=type=cache,target=/opt/ccache \
    python setup.py bdist_wheel -d /tmp/dist

########################################################################
FROM install-base AS fetch-pure

# Z-Shell related libraries.
ARG PURE_URL=https://github.com/sindresorhus/pure.git
ARG ZSHA_URL=https://github.com/zsh-users/zsh-autosuggestions
ARG ZSHS_URL=https://github.com/zsh-users/zsh-syntax-highlighting.git

RUN git clone --depth 1 ${PURE_URL} /opt/zsh/pure
RUN git clone --depth 1 ${ZSHA_URL} /opt/zsh/zsh-autosuggestions
RUN git clone --depth 1 ${ZSHS_URL} /opt/zsh/zsh-syntax-highlighting

########################################################################
FROM install-base AS fetch-torch

# For users who wish to download wheels instead of building them.
ARG PYTORCH_INDEX_URL
ARG PYTORCH_VERSION
RUN python -m pip wheel --no-deps --wheel-dir /tmp/dist \
        --index-url ${PYTORCH_INDEX_URL} \
        torch==${PYTORCH_VERSION}

########################################################################
FROM install-base AS fetch-vision

ARG PYTORCH_INDEX_URL
ARG TORCHVISION_VERSION
RUN python -m pip wheel --no-deps --wheel-dir /tmp/dist \
        --index-url ${PYTORCH_INDEX_URL} \
        torchvision==${TORCHVISION_VERSION}

########################################################################
FROM install-base AS fetch-mujoco

# Install Mujoco
ARG MUJOCO_URL=https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
WORKDIR /tmp/tar/mujoco
RUN curl ${MUJOCO_URL} --output /tmp/tar/mujoco.tar.gz && \
    tar -xzf /tmp/tar/mujoco.tar.gz -C /tmp/tar/mujoco && \
    rm /tmp/tar/mujoco.tar.gz

########################################################################
FROM ${BUILD_IMAGE} AS train-builds-include

COPY --link --from=install-base /opt/conda /opt/conda
COPY --link --from=build-vision /tmp/dist  /tmp/dist
COPY --link --from=fetch-mujoco /tmp/tar   /tmp/tar
COPY --link --from=fetch-pure   /opt/zsh   /opt/zsh

########################################################################
FROM ${BUILD_IMAGE} AS train-builds-exclude

COPY --link --from=install-base /opt/conda /opt/conda
COPY --link --from=fetch-torch  /tmp/dist  /tmp/dist
COPY --link --from=fetch-vision /tmp/dist  /tmp/dist
COPY --link --from=fetch-mujoco /tmp/tar   /tmp/tar
COPY --link --from=fetch-pure   /opt/zsh   /opt/zsh

########################################################################
FROM train-builds-${BUILD_MODE} AS train-builds
# Gather Python packages built in previous stages and
# install into a conda virtual environment using pip.
# Using a separate stage allows for build modularity
# and and parallel installation with system packages.

ARG PATH=/opt/conda/bin:${PATH}
# Using `PIP_CACHE_DIR` to cache previous installations.
ARG PIP_CACHE_DIR=/tmp/.cache/pip
COPY --link reqs/requirements.txt /tmp/pip/requirements.txt
# The `/tmp/dist/*.whl` files are the wheels built or installed in previous stages.
# `--find-links` gives higher priority to the wheels in `/tmp/dist`.
# Installing all packages in one command allows `pip` to resolve dependencies correctly.
# Using multiple `pip` installs may break the dependencies of all but the last installation.
RUN --mount=type=cache,target=${PIP_CACHE_DIR} \
    python -m pip install --find-links /tmp/dist \
        -r /tmp/pip/requirements.txt \
        /tmp/dist/*.whl

# Enable Intel MKL optimizations on AMD CPUs.
# https://danieldk.eu/Posts/2020-08-31-MKL-Zen.html
RUN echo 'int mkl_serv_intel_cpu_true() {return 1;}' > /opt/conda/fakeintel.c && \
    gcc -shared -fPIC -o /opt/conda/libfakeintel.so /opt/conda/fakeintel.c

########################################################################
FROM ${TRAIN_IMAGE} AS train
# Example training image for Ubuntu 20.04+ on Intel x86_64 CPUs.
# Edit this section if necessary but use `docker-compose.yaml` if possible.

LABEL maintainer=veritas9872@gmail.com
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ARG DEBIAN_FRONTEND=noninteractive

# Using `sed` and `xargs` to imitate the behavior of a requirements file.
# The `--mount=type=bind` temporarily mounts a directory from another stage.
# See the `deploy` stage below to see how to add other apt reporitories.
COPY --link reqs/apt-train.requirements.txt /tmp/apt/requirements.txt

# Using caching to speed up apt installation.
ARG DEB_OLD
ARG DEB_NEW
RUN rm -f /etc/apt/apt.conf.d/docker-clean; \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > \
    /etc/apt/apt.conf.d/keep-cache

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    if [ ${DEB_NEW} ]; then sed -i "s%${DEB_OLD}%${DEB_NEW}%g" /etc/apt/sources.list; fi && \
    apt-get update && sed 's/#.*//g; s/\r//g' /tmp/apt/requirements.txt | \
    xargs apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

ARG GID
ARG UID
ARG GRP=user
ARG USR=user
ARG PASSWD=ubuntu
# The `zsh` shell is used due to its convenience and popularity.
# Creating user with password-free sudo permissions. This may cause security issues.
RUN groupadd -f -g ${GID} ${GRP} && \
    useradd --shell /bin/zsh --create-home -u ${UID} -g ${GRP} \
        -p $(openssl passwd -1 ${PASSWD}) ${USR} && \
    echo "${USR} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Get conda with the directory ownership given to the user.
# Using conda for the virtual environment but not package installation.
COPY --link --from=train-builds --chown=${UID}:${GID} /opt/conda /opt/conda
RUN echo /opt/conda/lib >> /etc/ld.so.conf.d/conda.conf && ldconfig

# Enable Intel MKL optimizations on AMD CPUs.
# https://danieldk.eu/Posts/2020-08-31-MKL-Zen.html
ENV MKL_DEBUG_CPU_TYPE=5
ENV LD_PRELOAD=/opt/conda/libfakeintel.so:${LD_PRELOAD}

# Use Intel OpenMP with optimizations. See documentation for details.
# https://intel.github.io/intel-extension-for-pytorch/tutorials/performance_tuning/tuning_guide.html
ENV KMP_BLOCKTIME=0
ENV LD_PRELOAD=/opt/conda/lib/libiomp5.so:$LD_PRELOAD

# Use Jemalloc for efficient memory management.
# This configuration only works for Ubuntu 20.04+.
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
ENV MALLOC_CONF=background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000

# Docker must use absolute paths in `COPY` and cannot find `${HOME}`.
# Setting ${HOME} to its default value explicitly as a fix.
ARG HOME=/home/${USR}

# Get mujoco.
COPY --link --from=train-builds --chown=${UID}:${GID} /tmp/tar/mujoco ${HOME}/.mujoco
RUN echo ${HOME}/.mujoco >> /etc/ld.so.conf.d/mujoco.conf && ldconfig

USER ${USR}

# `PROJECT_ROOT` is where the project code will reside.
# The conda root path must be placed at the end of the
# PATH variable to prevent system program search errors.
# This is the opposite of the build stage.
ARG PROJECT_ROOT=/opt/project
ENV PATH=${PROJECT_ROOT}:/opt/conda/bin:${PATH}
ENV PYTHONPATH=${PROJECT_ROOT}

# Conda configurations are not carried with the directory.
# Resetting configurations in case conda packages are needed.
RUN conda config --append channels conda-forge && \
    conda config --remove channels defaults && \
    conda config --set channel_priority strict

# Setting the prompt to `pure`.
ARG PURE_PATH=${HOME}/.zsh/pure
COPY --link --from=train-builds --chown=${UID}:${GID} /opt/zsh/pure ${PURE_PATH}
RUN {   echo "fpath+=${PURE_PATH}"; \
        echo "autoload -Uz promptinit; promptinit"; \
        echo "prompt pure"; \
    } >> ${HOME}/.zshrc

# Add syntax highlighting. This must be activated after auto-suggestions.
ARG ZSHS_PATH=${HOME}/.zsh/zsh-syntax-highlighting
COPY --link --from=train-builds --chown=${UID}:${GID} \
    /opt/zsh/zsh-syntax-highlighting ${ZSHS_PATH}
RUN echo "source ${ZSHS_PATH}/zsh-syntax-highlighting.zsh" >> ${HOME}/.zshrc

# Enable mouse scrolling for tmux.
RUN echo 'set -g mouse on' >> ${HOME}/.tmux.conf

# `PROJECT_ROOT` belongs to `USR` if created after `USER` has been set.
# Not so for pre-existing directories, which will still belong to root.
WORKDIR ${PROJECT_ROOT}

# MuJoCo.
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
# Kitchen Environment
ENV PYTHONPATH=$PYTHONPATH:${PROJECT_ROOT}/relay-policy-learning/adept_envs
# BlockPush
ENV ASSET_PATH=${PROJECT_ROOT}
