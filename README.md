# An ML based network digital twin for QoE estimation in Software Defined Network
## Overview

This project provides implementations of several state-of-the-art GNN architectures for predicting network performance metrics such as RTT (Round-Trip Time) and packet loss based on network topology and features.

This code is part of the thesis work by Oussama Ben Taarit titled "An ML based network digital twin for QoE estimation in Software Defined Network".

Model training was performed on Google Colab to leverage their GPU resources for faster training of the graph neural network models.

## Getting Started

Just clone the repo and install the requirements in your python env to get everything installed:

```bash
# Run the setup script - might take a while depending on your internet
pip install -r requirements.txt
```

The setup script will handle all the dependencies and installation requirements. If you run into any GPU issues, you might need to install CUDA separately - check the PyTorch website for that.

## Usage

To train a model you can use the main script with various command line options:

```bash
# Train a single model (default: SAGE)
python main.py --model SAGE --epochs 200 --patience 20 --use-gpu

# Compare all implemented architectures
python main.py --compare-all --trials 3 --use-gpu

# Available model options: SAGE, GATv2, GIN, GraphTransformer, ResGatedGCN, ChebNet, GENConv
```

Key command line arguments:
- `--model`: Architecture to use (default: SAGE)
- `--epochs`: Maximum number of training epochs (default: 200)
- `--patience`: Early stopping patience (default: 20)
- `--compare-all`: Compare all implemented architectures
- `--trials`: Number of trials per architecture when comparing (default: 3)
- `--use-gpu`: Use GPU for training if available
- `--output-dir`: Directory to save results (default: results)

## Data

The project can use RIPE Atlas data for training and evaluation. RIPE Atlas is a global network of probes that measure internet connectivity and reachability.

## Author

Oussama Ben Taarit