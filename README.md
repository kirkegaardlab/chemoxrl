# Chemotaxis & Reinforcement learning: spatial-temporal information optimal integration

This is the repository for the project of studying chemotaxis optimal strategies
when the agent uses both temporal and spatial information.

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
