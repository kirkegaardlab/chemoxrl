# Reinforcement learning on a chemoatic cell

**On the integration of spatial and temporal information**

[![arXiv](https://img.shields.io/badge/arXiv-2310.10531-b31b1b.svg?style=flat)](https://arxiv.org/abs/2310.10531)

<p align="center">
  <img src="https://github.com/kirkegaardlab/chemoxrl/assets/38870744/d31bccfe-06f6-4bbc-820d-4cd4f3fe4bc1" height="236" />
</p>

This repository contains the code used on the paper "Learning optimal integration of spatial and temporal information in noisy chemotaxis"


## File Structure

The project is organized as follows:

```
├── README.md 
├── chemoxrl/ # Folder containing the code related to training the agents and env.
├── chemoxrl_aux/ # Utils functions used in analysis but not needed for training
├── models/ # Trained agents weights.
├── pyproject.toml  # Package setup for both chemoxrl and chemoxrl[aux]
├── requirements.txt # Project dependencies
└── train.py # Script to train the agent.
```

## Setup

Create a virtual environment and install the requiered packages (Note: we use `jaxlib` for GPU)

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Likewise, we recommend to install this library locally

```
pip install -e .
```

## Usage

One can train the agent by using 

```
python3 train.py
```
 The available options can be seen by using

```
python3 train.py --help
```
