# A Reproducibility study of [_Behavior Transformers: Cloning k modes with one stone_](https://github.com/notmahi/bet)

## Overview

This repository extends the original code repository of [_Behavior Transformers: Cloning k modes with one
stone_](https://github.com/notmahi/bet) to serve as an in-depth reproducibility assessment of the paper.

It contains scripts to reproduce each of the figures and tables in the paper in addition to scripts to run additional
experiments.

It is structures as follows:

## Table of Contents

## Getting Started

### Installation and Setup

#### Cloning

Clone the dataset with its submodules.
We track [Relay Policy Learning](https://github.com/google-research/relay-policy-learning) repo as a submodule for the
Franka kitchen environment.

```bash
git clone --recurse-submodules
```

If you didn't clone the repo with `--recurse-submodules`, you can clone the submodules with:

```bash
git submodule update --init
```

#### Datasets

The datasets are stored in the `data` folder and are not tracked by `git`.

1. Download the datasets [here](https://osf.io/download/4g53p/).
2. Extract the datasets into the `data` folder with

```bash
tar -xvf bet_data_release.tar -C this_repo/data
```

The contents of the `data` folder should look like this:

* `data/bet_data_release`: contains the datasets released by the paper authors.

### Environment

We provide installation methods to meet different systems.
The methods aim to insure easiness of use, portability, and reproducibility thanks to Docker.
It is hard to cover all systems, so we focused on the main ones.

1. **amd64 with CUDA:** for machines with Nvidia GPUs with Intel CPUs.
2. **amd 64 CPU-only:** for machines with Intel CPUs.
3. **arm64 with MPS:** to leverage the M1 GPU of Apple machines.

#### amd64 (CUDA and CPU-only)

This installation method is adapted from the [Cresset initiative](https://github.com/cresset-template/cresset).
Refer to the Cresset repository for more details.

Steps prefixed with [CUDA] are only required for the CUDA option.

**Prerequisites:**
To check if you have each of them run `<command-name> --version` or `<command-name> version` in the terminal.

* [`make`](https://cmake.org/install/).
* [`docker`](https://docs.docker.com/get-docker/). (v20.10+)
* [`docker compose`](https://docs.docker.com/compose/install/) (V2)
* [CUDA] [Nvidia CUDA Driver](https://www.nvidia.com/download/index.aspx) (Only the driver. No CUDA toolkit, etc)
* [CUDA] [`nvidia-docker`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

**Installation**

```bash
cd installation/amd64
```

In `Makefile`, change the first line`SERVICE` to `cuda` or `cpu`. Then run

```bash
make env
```

A `.env` file will be created in the installation directory. You need to edit it according to the following needs:

[CUDA] If you are using an old Nvidia GPU (i.e. [capability](https://developer.nvidia.com/cuda-gpus#compute)) < 3.7) you
need to compile PyTorch from source.
Find the compute capability for your GPU and edit it below.

```bash
BUILD_MODE=include               # Whether to build PyTorch from source.
CCA=3.5                          # Compute capability.
```

[CUDA] If your Nvidia drivers are also old you may need to change the CUDA Toolkit version.
See the [compatibility matrix](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions)
for compatible versions of the CUDA driver and CUDA Toolkit

```bash
CUDA_VERSION=11.3.1                # Must be compatible with hardware and CUDA driver.
CUDNN_VERSION=8                    # Only major version specifications are available.
```

#### MPS

As the MPS backend isn't supported on PyTorch on Docker, this methods relies on a local installation of `conda`, thus
unfortunately limiting portability and reproducibility.
We provide an `environment.yml` file adapted from the BeT's author's repo to be compatible with the M1 system.

**Prerequisites:**
* `conda`: which we recommend installing with [miniforge](https://github.com/conda-forge/miniforge).

**Installation**
```bash
conda env create --file=installation/arm64/environment.yml
conda activate behavior-transformer
```

### Logging
### Rendering

### Testing 

Test your setup by running the default training and evaluation scripts in each of the environments.

TODO: make the default config do something minimal
i.e. num workers 1, take the last training config.
and the scripts below fo 1 step, 1 episode ....

#### Point-Mass

#### Block-Push
Training

```bash
python train.py --config-name=train_blockpush num_prior_epochs=1
```

Evaluation

```bash
python run_on_env.py --config-name=eval_blockpush num_eval_steps=10 num_eval_eps=1 enable_render=False

```

#### Kitchen (Franka)
Training

```bash
python train.py --config-name=train_kitchen num_prior_epochs=1
```

Evaluation

```bash
xvfb-run -a -s "-screen 0 1400x900x24" \
python run_on_env.py --config-name=eval_kitchen \
num_eval_steps=10 num_eval_eps=1 enable_render=False
```

## Reproducing The Figures

## More Experiments

## Experiment with Different Configurations




