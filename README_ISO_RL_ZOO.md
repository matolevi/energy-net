# Energy-Net ISO RL Zoo: A Complete Guide

This guide provides detailed instructions for training and evaluating reinforcement learning (RL) agents for Independent System Operators (ISO) using our RL Zoo integration.

## Overview

The Energy-Net ISO RL Zoo provides a complete pipeline for:
1. **Hyperparameter Optimization**: Find optimal hyperparameters for your RL algorithms
2. **Training**: Train ISO agents using optimized hyperparameters 
3. **Evaluation**: Evaluate trained agents and generate performance visualizations

## Available Scripts

- `optimize_iso_zoo.sh`: Runs hyperparameter optimization using Optuna
- `run_iso_zoo.sh`: Trains an agent with specified or optimized hyperparameters
- `eval_iso_zoo.py`: Evaluates a trained agent

## Complete Pipeline Workflow

### Step 1: Hyperparameter Optimization

The first step is to optimize hyperparameters for your chosen algorithm:

```bash
./optimize_iso_zoo.sh --algo PPO --n-trials 100 --n-timesteps 10000 --eval-episodes 5 --pricing-policy ONLINE
```

This command:
- Uses Optuna to find optimal hyperparameters for PPO
- Runs 100 optimization trials
- Each trial trains for 10,000 timesteps
- Uses 5 episodes for evaluation during optimization
- Sets the pricing policy to ONLINE

The optimized hyperparameters are saved to `hyperparams/optimized/ppo_iso_best.yml`.

### Step 2: Training

Once hyperparameters are optimized, train a full agent:

```bash
./run_iso_zoo.sh --algo PPO --demand-pattern SINUSOIDAL --cost-type CONSTANT --pricing-policy ONLINE --total-timesteps 100000 --eval-episodes 5
```

This command:
- Trains a PPO agent for the ISO environment
- Uses SINUSOIDAL demand pattern and CONSTANT cost type
- Sets the pricing policy to ONLINE
- Trains for 100,000 timesteps
- Evaluates the agent every 10,000 timesteps using 5 episodes
- Automatically uses optimized hyperparameters if available

The trained model is saved to `models/iso_zoo/ppo/final_model_iso.zip` and the normalizer to `models/iso_zoo/ppo/final_model_normalizer.pkl`.

### Step 3: Evaluation

After training, evaluate your agent:

```bash
python eval_iso_zoo.py --algo ppo --env ISOEnv-v0 --model-path models/iso_zoo/ppo/final_model_iso.zip --normalizer-path models/iso_zoo/ppo/final_model_normalizer.pkl --demand-pattern SINUSOIDAL --cost-type CONSTANT --pricing-policy ONLINE --num-pcs-agents 1 --n-eval-episodes 10 --deterministic --output-dir eval_results/PPO_iso_SINUSOIDAL_CONSTANT
```

This generates comprehensive evaluation metrics and visualizations in the specified output directory.

Alternatively, evaluation happens automatically at the end of training if you run `run_iso_zoo.sh`.

## Detailed Script Options

### 1. Hyperparameter Optimization (`optimize_iso_zoo.sh`)

```bash
./optimize_iso_zoo.sh [OPTIONS]
```

**Required Options:**
- `--algo ALGO`: Algorithm to optimize (PPO, A2C, SAC, TD3)
- `--pricing-policy POLICY`: Pricing policy to use (ONLINE, QUADRATIC, CONSTANT)

**Common Options:**
- `--n-trials N`: Number of optimization trials (default: 100)
- `--n-timesteps N`: Timesteps per trial (default: 100000)
- `--eval-episodes N`: Number of episodes for evaluation (default: 5)
- `--demand-pattern PATTERN`: Demand pattern (SINUSOIDAL, RANDOM, PERIODIC, SPIKES)
- `--cost-type TYPE`: Cost structure (CONSTANT, VARIABLE, TIME_OF_USE)
- `--num-pcs-agents N`: Number of PCS agents to simulate (default: 1)

**Advanced Options:**
- `--study-name NAME`: Custom name for the Optuna study
- `--storage URL`: Database URL for storing results (default: local SQLite)
- `--seed N`: Random seed (default: 42)

### 2. Training (`run_iso_zoo.sh`)

```bash
./run_iso_zoo.sh [OPTIONS]
```

**Required Options:**
- `--algo ALGO`: Algorithm to train (PPO, A2C, SAC, TD3)
- `--pricing-policy POLICY`: Pricing policy to use (ONLINE, QUADRATIC, CONSTANT)

**Common Options:**
- `--demand-pattern PATTERN`: Demand pattern (SINUSOIDAL, RANDOM, PERIODIC, SPIKES)
- `--cost-type TYPE`: Cost structure (CONSTANT, VARIABLE, TIME_OF_USE)
- `--num-pcs-agents N`: Number of PCS agents to simulate (default: 1)
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

### 3. Evaluation (`eval_iso_zoo.py`)

```bash
python eval_iso_zoo.py [OPTIONS]
```

**Required Options:**
- `--algo ALGO`: Algorithm to evaluate (ppo, a2c, sac, td3)
- `--env ENV`: Environment ID (typically ISOEnv-v0)
- `--model-path PATH`: Path to trained model file
- `--normalizer-path PATH`: Path to normalizer file
- `--pricing-policy POLICY`: Pricing policy to use (ONLINE, QUADRATIC, CONSTANT)

**Common Options:**
- `--demand-pattern PATTERN`: Demand pattern (SINUSOIDAL, RANDOM, PERIODIC, SPIKES)
- `--cost-type TYPE`: Cost structure (CONSTANT, VARIABLE, TIME_OF_USE)
- `--num-pcs-agents N`: Number of PCS agents to simulate (default: 1)
- `--n-eval-episodes N`: Number of evaluation episodes (default: 5)
- `--deterministic`: Use deterministic actions (recommended)
- `--output-dir DIR`: Output directory for evaluation results

## Examples

### Quick Test Run (5 minutes)

```bash
# Run quick optimization
./optimize_iso_zoo.sh --algo PPO --n-trials 1 --n-timesteps 1 --eval-episodes 1 --pricing-policy QUADRATIC

# Train briefly with optimized hyperparameters
./run_iso_zoo.sh --algo PPO --demand-pattern SINUSOIDAL --cost-type CONSTANT --pricing-policy QUADRATIC --total-timesteps 4800 --eval-episodes 5 --dispatch-strategy predicted_demand --use-dispatch-action
```

### Full Training Run (1-2 hours)

```bash
# Run comprehensive optimization
./optimize_iso_zoo.sh --algo PPO --n-trials 100 --n-timesteps 10000 --eval-episodes 5 --pricing-policy QUADRATIC

# Train for longer with optimized hyperparameters
./run_iso_zoo.sh --algo PPO --demand-pattern SINUSOIDAL --cost-type CONSTANT --pricing-policy QUADRATIC --total-timesteps 100000 --eval-episodes 10
```

### Experiment with Different Conditions

```bash
# Train with different demand patterns
./run_iso_zoo.sh --algo PPO --demand-pattern RANDOM --cost-type CONSTANT --pricing-policy ONLINE --total-timesteps 50000

# Train with different cost types
./run_iso_zoo.sh --algo PPO --demand-pattern SINUSOIDAL --cost-type VARIABLE --pricing-policy ONLINE --total-timesteps 50000

# Try different pricing policies
./run_iso_zoo.sh --algo PPO --demand-pattern SINUSOIDAL --cost-type CONSTANT --pricing-policy QUADRATIC --total-timesteps 50000

# Try different algorithms
./run_iso_zoo.sh --algo SAC --demand-pattern SINUSOIDAL --cost-type CONSTANT --pricing-policy ONLINE --total-timesteps 50000
```

## Output Structure

```
hyperparams/optimized/
└── ppo_iso_best.yml           # Optimized hyperparameters

logs/iso_zoo/
├── <timestamp>_PPO/          # Experiment run
│   ├── evaluations/          # Evaluation results
│   ├── videos/               # Recorded videos (if enabled)
│   ├── train_monitor/        # Training metrics
│   ├── eval_monitor/         # Evaluation metrics
│   └── experiment_config.yml # Experiment configuration

models/iso_zoo/ppo/
├── final_model_iso.zip       # Trained model
└── final_model_normalizer.pkl # Environment normalizer

eval_results/PPO_iso_SINUSOIDAL_CONSTANT/
├── episode_X_flows_prices.png      # Energy flow and price visualizations
├── evaluation_results.csv          # Summary metrics
└── eval_monitor.monitor.csv        # Detailed evaluation metrics
```

## Monitoring Training

1. **TensorBoard**:
```bash
tensorboard --logdir logs/iso_zoo
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

## ISO-Specific Features

The Independent System Operator (ISO) agent represents a grid operator with specific parameters and features:

1. **Pricing Policies**:
   - `ONLINE`: Direct price setting for buy and sell prices
   - `QUADRATIC`: Uses polynomial coefficients for price setting
   - `CONSTANT`: Uses fixed prices

2. **PCS Agent Simulation**:
   The `--num-pcs-agents` parameter allows simulating multiple Power Consumption and Storage (PCS) units in the environment, representing consumers/prosumers with battery storage capability.

3. **Dispatch Control**:
   The ISO can control electricity dispatch in addition to pricing using the `--use-dispatch-action` parameter, allowing for more comprehensive grid management.

## Troubleshooting

**Issue**: "No such file" error for normalizer during evaluation.
**Solution**: Make sure training completed successfully and saved the normalizer file. Check `models/iso_zoo/ppo/` for `final_model_normalizer.pkl`.

**Issue**: Training gets stuck or performs poorly.
**Solution**: Try different hyperparameters or run the optimization process again with more trials.

**Issue**: Error with pricing policy.
**Solution**: Ensure you're using a valid pricing policy (ONLINE, QUADRATIC, CONSTANT) and that it's properly capitalized.

**Issue**: Out of memory errors.
**Solution**: Reduce batch size or network size in the hyperparameters.