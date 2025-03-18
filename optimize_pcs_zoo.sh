#!/bin/bash
# ===============================================================================
# Power Consumption and Storage (PCS) Hyperparameter Optimization Script
# ===============================================================================
#
# This script runs Optuna-based hyperparameter optimization for PCS agents.
# It systematically searches for optimal hyperparameter configurations for 
# the specified RL algorithm to maximize the performance of PCS agents in 
# battery management.
#
# The optimization process creates multiple trials, each training an agent with
# different hyperparameter settings, and tracks which configuration performs best.
#
# Key hyperparameters optimized include:
# - Learning rates and batch sizes
# - Network architecture (layer sizes)
# - Algorithm-specific parameters (e.g., entropy coefficient)
# - Training parameters (GAE lambda, discount factor)
#
# Usage:
#   ./optimize_pcs_zoo.sh --algo PPO --n-trials 100
# ===============================================================================

# Enable debug output and error handling
set -x
set -e

# Default values
ALGO="ppo"
ENV="PCSUnitEnv-v0"
DEMAND_PATTERN="SINUSOIDAL"
COST_TYPE="CONSTANT"
N_TRIALS=100
N_TIMESTEPS=100000
EVAL_EPISODES=5
SEED=42

# Parse command line arguments
while [ $# -gt 0 ]; do
    case $1 in
        --algo)
            # Algorithm to optimize hyperparameters for
            ALGO=$(echo "$2" | tr '[:upper:]' '[:lower:]')
            shift 2
            ;;
        --n-trials)
            # Number of optimization trials to run
            N_TRIALS=$2
            shift 2
            ;;
        --n-timesteps)
            # Number of timesteps to train each trial
            N_TIMESTEPS=$2
            shift 2
            ;;
        --eval-episodes)
            # Number of episodes for evaluating each trial
            EVAL_EPISODES=$2
            shift 2
            ;;
        --demand-pattern)
            # Demand pattern used for training environment
            DEMAND_PATTERN=$2
            shift 2
            ;;
        --cost-type)
            # Cost structure used for training environment
            COST_TYPE=$2
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Create necessary directories
mkdir -p logs/pcs_zoo_optimize
mkdir -p hyperparams/optimized

# Start optimization
echo "Starting optimization with $ALGO..."
echo "Number of trials: $N_TRIALS"
echo "Timesteps per trial: $N_TIMESTEPS"
echo "Evaluation episodes: $EVAL_EPISODES"

python3 optimize_pcs_zoo.py \
    --algo $ALGO \
    --env $ENV \
    --n-trials $N_TRIALS \
    --n-timesteps $N_TIMESTEPS \
    --n-eval-episodes $EVAL_EPISODES \
    --demand-pattern $DEMAND_PATTERN \
    --cost-type $COST_TYPE

OPTIMIZE_EXIT_CODE=$?

if [ $OPTIMIZE_EXIT_CODE -eq 0 ]; then
    echo "Optimization completed successfully"
    echo ""
    echo "Best hyperparameters saved to: hyperparams/optimized/${ALGO}_best.yml"
    echo ""
    echo "Next steps:"
    echo "1. Train a model with optimized hyperparameters:"
    echo "   ./run_pcs_zoo.sh --algo $ALGO --demand-pattern $DEMAND_PATTERN --cost-type $COST_TYPE"
else
    echo "Optimization failed with exit code $OPTIMIZE_EXIT_CODE"
    exit $OPTIMIZE_EXIT_CODE
fi