# InfluencerLoss for End-to-end Point Cloud Segmentation
    
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Introduction

This repository contains the implementation of the InfluencerLoss for end-to-end tracking. More generally, the InfluencerLoss is a novel loss function that can be used to train a graph neural network (GNN), transformer, or any other backbone to perform end-to-end point cloud segmentation. The implementation here is based on the original presentation at [CHEP2023](https://indico.jlab.org/event/459/contributions/11743/), and proceedings are under preparation.

## Installation

Installation is done via conda. The following commands will clone the repository and install the required packages.

```bash
git clone git@github.com:murnanedaniel/InfluencerNet.git
cd InfluencerNet
conda env create -f environment.yml
conda activate influencer
```

Note that this assumes compatibility with CUDA 11.7. Using other versions of CUDA is possible by tweaking the `environment.yml` and `requirements.txt` files at
```
- pytorch-cuda==11.7
...
--find-links https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

## Quickstart

The fastest way to see the InfluencerLoss in action is through the example scripts that reproduce the results presented at CHEP2023. See the [CHEP example subfolder](examples/CHEP2023) for more details.

## Usage

**TODO**

...

## Citation

**TODO**

If you use this code in your research, please cite the following:

```
@article{murnane2023influencer,
  title={InfluencerLoss: End-to-End Geometric Representation Learning for Track Reconstruction},
  author={Murnane, Daniel},
  year={2023}
}
```