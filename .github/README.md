# A Reproducibility study of [_Behavior Transformers: Cloning k modes with one stone_](https://github.com/notmahi/bet)

## Overview

This repository extends the original code repository of [_Behavior Transformers: Cloning k modes with one
stone_](https://github.com/notmahi/bet) to serve as an in-depth reproducibility assessment of the paper.

It contains **TODO**.

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

The datasets are stored in the `data` folder, not tracked by `git`.

1. Download the datasets [here](https://osf.io/download/4g53p/).
2. Extract the datasets into the `data` folder. `tar -xvf bet_data_release.tar.gz -C this/repo/data`

The contents of the `data` folder should look like this:

* `data/something_todo`: the downloaded archive.
* `data/bet_data_release`: contains the datasets released by the paper authors.




