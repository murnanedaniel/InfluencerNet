# CHEP 2023 Reproducible Example

This repository contains the code to reproduce the [results presented at CHEP 2023](https://indico.jlab.org/event/459/contributions/11743/).

## Installation

Follow the instructions on the [main README](../../README.md) to install the required packages.

## Usage

### Physics Performance of the InfluencerLoss

The physics performance of the InfluencerLoss can be reproduced by running the following commands.


#### 1. Training

```bash
python train.py --config influencer_config.yml
```

#### 2. Produce the Physics Performance Plots

```bash
python eval.py --config influencer_config.yml
```

### Physics Performance of the Baseline ("NaiveLoss")

The physics performance of the baseline ("NaiveLoss") can be reproduced by running the following commands.

#### 1. Training

```bash
python train.py --config naive_config.yml
```

#### 2. Produce the Physics Performance Plots

```bash
python eval.py --config naive_config.yml
```
<!-- 
**n.b.** Assuming you have produced both sets of track candidates from InfluencerLoss and NaiveLoss, you can subsequently produce the comparison plots found in the CHEP proceedings by running the following command:

```bash
python plot.py --config influencer_config.yml naive_config.yml
``` -->

#### Computational Performance

The computational performance of the InfluencerLoss vs. NaiveLoss can be reproduced by running the following commands.

**TODO**