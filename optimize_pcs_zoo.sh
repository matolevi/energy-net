#!/bin/bash

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
            ALGO=$(echo "$2" | tr '[:upper:]' '[:lower:]')
            shift 2
            ;;
        --n-trials)
            N_TRIALS=$2
            shift 2
            ;;
        --n-timesteps)
            N_TIMESTEPS=$2
            shift 2
            ;;
        --eval-episodes)
            EVAL_EPISODES=$2
            shift 2
            ;;
        --demand-pattern)
            DEMAND_PATTERN=$2
            shift 2
            ;;
        --cost-type)
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
else
    echo "Optimization failed with exit code $OPTIMIZE_EXIT_CODE"
    exit $OPTIMIZE_EXIT_CODE
fi 