# EnergyNet

EnergyNet is a package for smart grid simulation, designed to model and simulate energy dynamics in a network of interconnected entities. It provides tools for creating, managing, and analyzing energy networks, making it suitable for research and development in the field of smart grids.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Specialized Guides](#specialized-guides)


## Features

- **Network Simulation**: Simulate energy dynamics in a network of interconnected entities.
- **Multi-Agent Support**: Utilize the PettingZoo API for multi-agent reinforcement learning environments.
- **Customizable Entities**: Define custom network entities with specific energy dynamics.
- **Experiment Tracking**: Track and visualize experiments using Weights & Biases (wandb) and TensorBoard.
- **Reinforcement Learning**: Train and evaluate RL agents for power control systems and ISO.

## Installation

There are two ways to install the dependencies for this project:

### Option 1: Development Installation

This is recommended if you're working on the codebase:

```sh
pip install -e .
```

This will install the package in development mode using setup.cfg, allowing you to modify the code without reinstalling.

### Option 2: Using Requirements File

For exact dependency versions (recommended for reproducibility):

```sh
pip install -r requirements.txt
```

This installs all specific dependency versions listed in requirements.txt.

## Requirements

- **Python**: Version 3.8+ 
  - If using Python 3.8, ensure all type annotations use the `typing` module (e.g., `List`, `Dict`, `Tuple`) instead of built-in types
- **Operating System**: Compatible with Linux, macOS, and Windows

## Usage

For detailed usage instructions, please refer to the specialized guides listed below.

## Project Structure

```
energy-net/
├── energy_net/           # Core package code
│   ├── components/       # Network components (PCS units, etc.)
│   ├── env/              # Environment definitions
│   ├── model/            # Data models
│   └── utils/            # Utility functions
├── models/               # Saved models
├── hyperparams/          # Hyperparameters for RL algorithms
├── logs/                 # Training and evaluation logs
├── eval_results/         # Evaluation results and visualizations
├── scripts/              # Helper scripts
└── README*.md            # Documentation files
```

## Dependencies

The project has the following dependencies:

- **Python 3.8+**
  - **Description:** The programming language used for this project.
  - **Note:** When using Python 3.8, use the typing module for type hints.

- **numpy**
  - **Description:** A fundamental package for scientific computing with Python.
  - **Usage:** Used for numerical operations and handling large datasets.

- **pettingzoo**
  - **Description:** A library for multi-agent reinforcement learning environments.
  - **Usage:** Provides environments for training and evaluating multi-agent systems.

- **scipy**
  - **Description:** A Python library used for scientific and technical computing.
  - **Usage:** Used for advanced mathematical functions and algorithms.

- **stable-baselines3**
  - **Description:** A set of improved implementations of reinforcement learning algorithms.
  - **Usage:** Used for training reinforcement learning models.

- **wandb**
  - **Description:** Weights & Biases (wandb) is a tool for experiment tracking.
  - **Usage:** Used for tracking and visualizing machine learning experiments.

- **tensorboard**
  - **Description:** A suite of visualization tools for TensorFlow.
  - **Usage:** Used for visualizing training metrics and model graphs.

- **rl_zoo3**
  - **Description:** A collection of pre-trained reinforcement learning agents.
  - **Usage:** Provides pre-trained models and training scripts.

## Specialized Guides

For more detailed instructions on specific components of EnergyNet:

- [**PCS RL Zoo Guide**](README_RL_ZOO.md) - Complete guide for training and evaluating RL agents for PCS
- [**ISO RL Zoo Guide**](README_ISO_RL_ZOO.md) - Complete guide for training and evaluating RL agents for Independent System Operators
