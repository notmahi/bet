# A Reproducibility study of [_Behavior Transformers: Cloning k modes with one stone_](https://github.com/notmahi/bet)

## Overview

This repository extends the original code repository of [_Behavior Transformers: Cloning k modes with one
stone_](https://github.com/notmahi/bet) to serve as an in-depth reproducibility assessment of the paper.

It contains scripts to reproduce each of the figures and tables in the paper in addition to scripts to run additional
experiments.

It is structures as follows:

## Table of Contents

TODO: Add a table of contents.

## Getting Started

### Installation and Setup

#### Cloning

Clone the dataset with its submodules.
We track [Relay Policy Learning](https://github.com/google-research/relay-policy-learning) repo as a submodule for the
Franka kitchen environment.
It uses [`git-lfs`](https://git-lfs.github.com/). Make sure you have it installed.

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
   ```bash
   wget https://osf.io/download/4g53p/ -O ./data/bet_data_release.tar.gz
   ```
2. Extract the datasets into the `data/` folder.

   ```bash
   tar -xvf data/bet_data_release.tar.gz -C data
   ```

The contents of the `data` folder should look like this:

* `data/bet_data_release.tar.gz`: The archive just downloaded.
* `data/bet_data_release`: contains the datasets released by the paper authors.
* `data/README.md`: A placeholder.

### Environment

We provide installation methods to meet different systems.
The methods aim to insure easiness of use, portability, and reproducibility thanks to Docker.
It is hard to cover all systems, so we focused on the main ones.

- A: **amd64 with CUDA:** for machines with Nvidia GPUs with Intel CPUs.
- B: **amd 64 CPU-only:** for machines with Intel CPUs.
- C: **arm64 with MPS:** to leverage the M1 GPU of Apple machines.

#### A&B: amd64 (CUDA and CPU-only)

This installation method is adapted from the [Cresset initiative](https://github.com/cresset-template/cresset).
Refer to the Cresset repository for more details.

Steps prefixed with [CUDA] are only required for the CUDA option.

**Prerequisites:**
To check if you have each of them run `<command-name> --version` or `<command-name> version` in the terminal.

* [`make`](https://cmake.org/install/).
* [`docker`](https://docs.docker.com/engine/). (v20.10+)
* [`docker compose`](https://docs.docker.com/compose/install/) (V2)
* [CUDA] [Nvidia CUDA Driver](https://www.nvidia.com/download/index.aspx) (Only the driver. No CUDA toolkit, etc)
* [CUDA] [`nvidia-docker`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) (the NVIDIA Container Toolkit).

**Installation**

```bash
cd installation/amd64
```

Run

```bash
make env
```

The `.env` file will be created in the installation directory. You need to edit it according to the following needs:

[CUDA] If you are using an old Nvidia GPU (i.e. [capability](https://developer.nvidia.com/cuda-gpus#compute)) < 3.7) you
need to compile PyTorch from source.
Find the compute capability for your GPU and edit it below.

```bash
BUILD_MODE=include               # Whether to build PyTorch from source.
CCA=3.5                          # Compute capability.
```

[CUDA] If your Nvidia drivers are also old you may need to change the CUDA Toolkit version.
See
the [compatibility matrix](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions)
for compatible versions of the CUDA driver and CUDA Toolkit

```bash
CUDA_VERSION=11.3.1                # Must be compatible with hardware and CUDA driver.
CUDNN_VERSION=8                    # Only major version specifications are available.
```

Build the docker image by running the following.
Setting `SERVICE=cuda` or `SERVICE=cpu` for your desired option.

```bash
make build SERVICE=<cpu|cuda>
````

Then to use the environment, run

```bash
make exec SERVICE=<cpu|cuda>
```

and you'll be inside the container.

To run multiple instances of the container you can use

```bash
make run SERVICE=<cpu|cuda>
```

#### C: MPS (Apple silicon)

As the MPS backend isn't supported on PyTorch on Docker, this methods relies on a local installation of `conda`, thus
unfortunately limiting portability and reproducibility.
We provide an `environment.yml` file adapted from the BeT's author's repo to be compatible with the M1 system.

**Prerequisites:**

* `conda`: which we recommend installing with [miniforge](https://github.com/conda-forge/miniforge).

**Installation**

```bash
conda env create --file=installation/osx-arm64/environment.yml
conda activate behavior-transformer
```

Set environment variables.

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/relay-policy-learning/adept_envs
export ASSET_PATH=$(pwd)
export PYTHONDONTWRITEBYTECODE=1
export HYDRA_FULL_ERROR=1
```

### Logging

We track our experiments with [Weights and Biases](https://wandb.ai/site).
To use it, either

1. [Docker] Add your `wandb` [API key](https://wandb.ai/authorize) to the `.env` file
    ```bash
    WANDB_MODE=online
    WANDB_API_KEY=<your_key>
    ```
   then `make up SERVICE=<cpu|cuda>`.

2. Or, run

    ```bash
    export WANDB_MODE=online && wandb login
    ```
   in the docker container, or in your custom environment.

### Testing

Test your setup by running the default training and evaluation scripts in each of the environments.

Environment values (`<env>` below) can be `blockpush` or `kitchen`.

Training.

```bash
python train.py env=<env> experiment.num_prior_epochs=1
```

Evaluation.
Find the model you just trained in `train_runs/train_<env>/<date>/<job_id>`.
Plug it in the command below.

```bash
python run_on_env.py env=<env> experiment.num_eval_eps=1 \
model.load_dir=$(pwd)/train_runs/train_<env>/<date>/<job_id>
```

### Recording Videos

You can record videos of rollouts by adding `expriment.record_video=True` in the commandline.

## Reproducing The Figures

We provide model weights and their rollouts for all the experiments we ran.
You can use these to reproduce our results.
The scripts used to generate the models and rollouts and to get our figures can be found in `reproducibility_scripts/`

Obtain the model weights with

```bash
wget https://www.dropbox.com/s/eoc40tx7bh1nql9/train_runs.tar.gz
tar -xvf train_runs.tar.gz
```

TODO: Obtain the rollouts with

```bash
wget 
tar -xvf 
```

## More Experiments

## Experiment with Different Configurations




