#!/bin/bash

# Default parameters
ALGO="ppo"
ENV="ISOEnv-v0"
DEMAND_PATTERN="SINUSOIDAL"
COST_TYPE="CONSTANT"
PRICING_POLICY="ONLINE"
NUM_PCS_AGENTS=1
N_TRIALS=100
N_TIMESTEPS=100000
EVAL_EPISODES=5
SEED=42

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --algo)
            ALGO="$2"
            shift 2
            ;;
        --n-trials)
            N_TRIALS="$2"
            shift 2
            ;;
        --n-timesteps)
            N_TIMESTEPS="$2"
            shift 2
            ;;
        --eval-episodes)
            EVAL_EPISODES="$2"
            shift 2
            ;;
        --study-name)
            STUDY_NAME="$2"
            shift 2
            ;;
        --storage)
            STORAGE="$2"
            shift 2
            ;;
        --demand-pattern)
            DEMAND_PATTERN="$2"
            shift 2
            ;;
        --cost-type)
            COST_TYPE="$2"
            shift 2
            ;;
        --pricing-policy)
            PRICING_POLICY="$2"
            shift 2
            ;;
        --num-pcs-agents)
            NUM_PCS_AGENTS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create directories if they don't exist
mkdir -p logs/iso_zoo_optimize
mkdir -p hyperparams/optimized

# Print optimization parameters
echo "Starting ISO optimization with:"
echo "  Algorithm: $ALGO"
echo "  Environment: $ENV"
echo "  Demand Pattern: $DEMAND_PATTERN"
echo "  Cost Type: $COST_TYPE"
echo "  Pricing Policy: $PRICING_POLICY"
echo "  Number of PCS Agents: $NUM_PCS_AGENTS"
echo "  Number of Trials: $N_TRIALS"
echo "  Number of Timesteps: $N_TIMESTEPS"
echo "  Evaluation Episodes: $EVAL_EPISODES"
echo "  Seed: $SEED"

# Run the optimization
echo "Logs will be created in logs/iso_zoo_optimize"
echo "Optimized hyperparameters will be saved in hyperparams/optimized"

STUDY_ARGS=""
if [ ! -z "$STUDY_NAME" ]; then
    STUDY_ARGS="--study-name $STUDY_NAME"
fi

STORAGE_ARGS=""
if [ ! -z "$STORAGE" ]; then
    STORAGE_ARGS="--storage $STORAGE"
fi

python optimize_iso_zoo.py \
    --algo $ALGO \
    --env $ENV \
    --n-trials $N_TRIALS \
    --n-timesteps $N_TIMESTEPS \
    --eval-episodes $EVAL_EPISODES \
    --demand-pattern $DEMAND_PATTERN \
    --cost-type $COST_TYPE \
    --pricing-policy $PRICING_POLICY \
    --num-pcs-agents $NUM_PCS_AGENTS \
    --seed $SEED \
    $STUDY_ARGS $STORAGE_ARGS

# Check if optimization completed successfully
exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "Optimization completed successfully!"
else
    echo "Optimization failed with exit code $exit_code"
    exit $exit_code
fi 