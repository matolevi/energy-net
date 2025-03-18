# Energy-Net RL Zoo: A Complete Guide

This guide provides detailed instructions for training and evaluating reinforcement learning (RL) agents for Power Consumption and Storage (PCS) units using our RL Zoo integration.

## Overview

The Energy-Net RL Zoo provides a complete pipeline for:
1. **Hyperparameter Optimization**: Find optimal hyperparameters for your RL algorithms
2. **Training**: Train agents using optimized hyperparameters 
3. **Evaluation**: Evaluate trained agents and generate performance visualizations

## Available Scripts

- `optimize_pcs_zoo.sh`: Runs hyperparameter optimization using Optuna
- `run_pcs_zoo.sh`: Trains an agent with specified or optimized hyperparameters
- `eval_pcs_zoo.py`: Evaluates a trained agent

## Complete Pipeline Workflow

### Step 1: Hyperparameter Optimization

The first step is to optimize hyperparameters for your chosen algorithm:

```bash
./optimize_pcs_zoo.sh --algo PPO --n-trials 100 --n-timesteps 10000 --eval-episodes 5
```

This command:
- Uses Optuna to find optimal hyperparameters for PPO
- Runs 100 optimization trials
- Each trial trains for 10,000 timesteps
- Uses 5 episodes for evaluation during optimization

The optimized hyperparameters are saved to `hyperparams/optimized/ppo_best.yml`.

### Step 2: Training

Once hyperparameters are optimized, train a full agent:

```bash
./run_pcs_zoo.sh --algo PPO --demand-pattern SINUSOIDAL --cost-type CONSTANT --total-timesteps 100000 --eval-episodes 5
```

This command:
- Trains a PPO agent for a Power Consumption and Storage (PCS) unit
- Uses SINUSOIDAL demand pattern and CONSTANT cost type
- Trains for 100,000 timesteps
- Evaluates the agent every 10,000 timesteps using 5 episodes
- Automatically uses optimized hyperparameters if available

The trained model is saved to `models/pcs_zoo/ppo/final_model_ppo.zip` and the normalizer to `models/pcs_zoo/ppo/final_model_normalizer.pkl`.

### Step 3: Evaluation

After training, evaluate your agent:

```bash
python3 eval_pcs_zoo.py --algo ppo --env PCSUnitEnv-v0 --model-path models/pcs_zoo/ppo/final_model_ppo.zip --normalizer-path models/pcs_zoo/ppo/final_model_normalizer.pkl --demand-pattern SINUSOIDAL --cost-type CONSTANT --n-eval-episodes 1 --deterministic --output-dir eval_results/ppo_SINUSOIDAL_CONSTANT
```

This generates comprehensive evaluation metrics and visualizations in the specified output directory.

Alternatively, evaluation happens automatically at the end of training if you run `run_pcs_zoo.sh`.

## Detailed Script Options

### 1. Hyperparameter Optimization (`optimize_pcs_zoo.sh`)

```bash
./optimize_pcs_zoo.sh [OPTIONS]
```

**Required Options:**
- `--algo ALGO`: Algorithm to optimize (PPO, A2C, SAC, TD3)

**Common Options:**
- `--n-trials N`: Number of optimization trials (default: 100)
- `--n-timesteps N`: Timesteps per trial (default: 100000)
- `--eval-episodes N`: Number of episodes for evaluation (default: 5)
- `--demand-pattern PATTERN`: Demand pattern (SINUSOIDAL, RANDOM, PERIODIC, SPIKES)
- `--cost-type TYPE`: Cost structure (CONSTANT, VARIABLE, TIME_OF_USE)

**Advanced Options:**
- `--study-name NAME`: Custom name for the Optuna study
- `--storage URL`: Database URL for storing results (default: local SQLite)

### 2. Training (`run_pcs_zoo.sh`)

```bash
./run_pcs_zoo.sh [OPTIONS]
```

**Required Options:**
- `--algo ALGO`: Algorithm to train (PPO, A2C, SAC, TD3)

**Common Options:**
- `--demand-pattern PATTERN`: Demand pattern (SINUSOIDAL, RANDOM, PERIODIC, SPIKES)
- `--cost-type TYPE`: Cost structure (CONSTANT, VARIABLE, TIME_OF_USE)
- `--total-timesteps N`: Total timesteps for training (default: 1000000)
- `--eval-episodes N`: Number of evaluation episodes (default: 5)
- `--seed N`: Random seed (default: 42)

**Advanced Options:**
- `--use-zoo-defaults`: Use RL Zoo's default hyperparameters instead of optimized ones
- `--conf-file PATH`: Custom configuration file path
- `--eval-freq N`: Evaluation frequency in timesteps (default: 10000)
- `--record-video`: Enable video recording (default: false)
- `--record-video-freq N`: Video recording frequency (default: 10000)
- `--tensorboard-log PATH`: TensorBoard log directory

### 3. Evaluation (`eval_pcs_zoo.py`)

```bash
python3 eval_pcs_zoo.py [OPTIONS]
```

**Required Options:**
- `--algo ALGO`: Algorithm to evaluate (ppo, a2c, sac, td3)
- `--env ENV`: Environment ID (typically PCSUnitEnv-v0)
- `--model-path PATH`: Path to trained model file
- `--normalizer-path PATH`: Path to normalizer file

**Common Options:**
- `--demand-pattern PATTERN`: Demand pattern (SINUSOIDAL, RANDOM, PERIODIC, SPIKES)
- `--cost-type TYPE`: Cost structure (CONSTANT, VARIABLE, TIME_OF_USE)
- `--n-eval-episodes N`: Number of evaluation episodes (default: 5)
- `--deterministic`: Use deterministic actions (recommended)
- `--output-dir DIR`: Output directory for evaluation results

## Extended Environment Options

For the Independent System Operator (ISO) environment (using `run_iso_zoo.sh`, `optimize_iso_zoo.sh`, and `eval_iso_zoo.py`), the following additional options are available:

**Dispatch Control Options:**
- `--use-dispatch-action`: Enable dispatch control in the agent's action space, allowing the ISO to control both pricing and dispatch decisions simultaneously
- `--dispatch-strategy STRATEGY`: Strategy for dispatch when not controlled by agent actions (choices: "predicted_demand", "fixed", "scaled", "manual_profile", "daily_pattern")
  - `predicted_demand`: Match dispatch to predicted demand (default)
  - `fixed`: Use a constant dispatch value
  - `scaled`: Scale predicted demand by a factor
  - `manual_profile`: Use a predefined dispatch profile
  - `daily_pattern`: Use a time-of-day based pattern

**Agent Simulation Options:**
- `--trained-pcs-model-path PATH`: Path to a trained PCS model to use for simulating consumer/prosumer behavior
- `--num-pcs-agents N`: Number of Power Consumption and Storage units to simulate in the environment (default: 1)

These additional options provide more control over the dispatch mechanism and allow for testing with simulated consumer behavior using pre-trained PCS models.

## Examples

### Quick Test Run (5 minutes)

```bash
# Run quick optimization
./optimize_pcs_zoo.sh --algo PPO --n-trials 1 --n-timesteps 1 --eval-episodes 1

# Train briefly with optimized hyperparameters
./run_pcs_zoo.sh --algo PPO --demand-pattern SINUSOIDAL --cost-type CONSTANT --total-timesteps 1 --eval-episodes 1 
```

### Full Training Run (1-2 hours)

```bash
# Run comprehensive optimization
./optimize_pcs_zoo.sh --algo PPO --n-trials 100 --n-timesteps 10000 --eval-episodes 5

# Train for longer with optimized hyperparameters
./run_pcs_zoo.sh --algo PPO --demand-pattern SINUSOIDAL --cost-type CONSTANT --total-timesteps 100000 --eval-episodes 5
```

### Experiment with Different Conditions

```bash
# Train with different demand patterns
./run_pcs_zoo.sh --algo PPO --demand-pattern RANDOM --cost-type CONSTANT --total-timesteps 50000

# Train with different cost types
./run_pcs_zoo.sh --algo PPO --demand-pattern SINUSOIDAL --cost-type VARIABLE --total-timesteps 50000

# Try different algorithms
./run_pcs_zoo.sh --algo SAC --demand-pattern SINUSOIDAL --cost-type CONSTANT --total-timesteps 50000
```

## Output Structure

```
hyperparams/optimized/
└── ppo_best.yml              # Optimized hyperparameters

logs/pcs_zoo/
├── <timestamp>_PPO/          # Experiment run
│   ├── evaluations/          # Evaluation results
│   ├── videos/               # Recorded videos (if enabled)
│   ├── train_monitor/        # Training metrics
│   ├── eval_monitor/         # Evaluation metrics
│   └── experiment_config.yml # Experiment configuration

models/pcs_zoo/ppo/
├── final_model_ppo.zip       # Trained model
└── final_model_normalizer.pkl # Environment normalizer

eval_results/ppo_SINUSOIDAL_CONSTANT/
├── episode_X_flows_prices.png      # Energy flow and price visualizations
├── episode_X_cost_components.png   # Cost breakdown visualizations
├── episode_X_final_cost_distribution.png # Cost distribution
├── evaluation_results.csv          # Summary metrics
└── eval_monitor.monitor.csv        # Detailed evaluation metrics
```

## Monitoring Training

1. **TensorBoard**:
```bash
tensorboard --logdir logs/pcs_zoo
```

2. **Training Metrics**:
The training process outputs detailed information including:
- Episode rewards
- Policy losses
- Value losses
- Entropy losses
- Mean episode lengths

3. **Evaluation Results**:
After training, check the `eval_results/` directory for detailed visualizations and metrics.

## Troubleshooting

**Issue**: "No such file" error for normalizer during evaluation.
**Solution**: Make sure training completed successfully and saved the normalizer file. Check `models/pcs_zoo/ppo/` for `final_model_normalizer.pkl`.

**Issue**: Training gets stuck or performs poorly.
**Solution**: Try different hyperparameters or run the optimization process again with more trials.

**Issue**: Out of memory errors.
**Solution**: Reduce batch size or network size in the hyperparameters.