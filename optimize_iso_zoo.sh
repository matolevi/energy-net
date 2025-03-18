#!/bin/bash
# ===============================================================================
# Independent System Operator (ISO) Hyperparameter Optimization Script
# ===============================================================================
#
# This script runs Optuna-based hyperparameter optimization for ISO agents.
# It systematically searches for optimal hyperparameter configurations for 
# the specified RL algorithm to maximize the performance of ISO agents in grid
# management.
#
# The optimization process creates multiple trials, each training an agent with
# different hyperparameter settings, and tracks which configuration performs best.
#
# Usage:
#   ./optimize_iso_zoo.sh --algo PPO --pricing-policy ONLINE --n-trials 100
# ===============================================================================

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
USE_DISPATCH_ACTION=false
DISPATCH_STRATEGY="predicted_demand"
TRAINED_PCS_MODEL_PATH=NULL

# Parse command line arguments with enhanced documentation
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --algo)
            # RL algorithm to optimize (PPO, A2C, SAC, TD3)
            ALGO="$2"
            shift 2
            ;;
        --n-trials)
            # Number of optimization trials to run
            N_TRIALS="$2"
            shift 2
            ;;
        --n-timesteps)
            # Number of timesteps per trial
            N_TIMESTEPS="$2"
            shift 2
            ;;
        --eval-episodes)
            # Number of evaluation episodes per trial
            EVAL_EPISODES="$2"
            shift 2
            ;;
        --study-name)
            # Custom name for the Optuna study
            STUDY_NAME="$2"
            shift 2
            ;;
        --storage)
            # Database URL for storing optimization results
            STORAGE="$2"
            shift 2
            ;;
        --demand-pattern)
            # Demand pattern for the environment (SINUSOIDAL, RANDOM, PERIODIC, SPIKES)
            DEMAND_PATTERN="$2"
            shift 2
            ;;
        --cost-type)
            # Cost structure (CONSTANT, VARIABLE, TIME_OF_USE)
            COST_TYPE="$2"
            shift 2
            ;;
        --pricing-policy)
            # Pricing policy for ISO (ONLINE, QUADRATIC, CONSTANT)
            PRICING_POLICY="$2"
            shift 2
            ;;
        --num-pcs-agents)
            # Number of PCS agents to simulate in environment
            NUM_PCS_AGENTS="$2"
            shift 2
            ;;
        --seed)
            # Random seed for reproducibility
            SEED="$2"
            shift 2
            ;;
        --use-dispatch-action)
            # Enable dispatch control in agent's action space
            USE_DISPATCH_ACTION=true
            shift 1
            ;;
        --dispatch-strategy)
            # Strategy for dispatch when not controlled by agent
            DISPATCH_STRATEGY="$2"
            shift 2
            ;;
        --trained-pcs-model-path)
            # Path to trained PCS model for PCS agent simulation
            TRAINED_PCS_MODEL_PATH="$2"
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

# Print optimization parameters with clear documentation
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
echo "  Use Dispatch Action: $USE_DISPATCH_ACTION  # Whether to optimize with dispatch control"
echo "  Dispatch Strategy: $DISPATCH_STRATEGY  # Strategy used when not agent-controlled"
if [ "$TRAINED_PCS_MODEL_PATH" != "NULL" ]; then
    echo "  Trained PCS Model Path: $TRAINED_PCS_MODEL_PATH"
fi

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

# Execute the Python optimization script with all parameters
python3 optimize_iso_zoo.py \
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
    $([ "$USE_DISPATCH_ACTION" = true ] && echo "--use-dispatch-action") \
    --dispatch-strategy $DISPATCH_STRATEGY \
    $([ "$TRAINED_PCS_MODEL_PATH" != "NULL" ] && echo "--trained-pcs-model-path $TRAINED_PCS_MODEL_PATH") \
    $STUDY_ARGS $STORAGE_ARGS

# Check if optimization completed successfully
exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "Optimization completed successfully!"
    echo "Best hyperparameters saved to: hyperparams/optimized/${ALGO}_iso_best.yml"
    echo ""
    echo "Next steps:"
    echo "1. Train a full model with these hyperparameters using:"
    echo "   ./run_iso_zoo.sh --algo $ALGO --pricing-policy $PRICING_POLICY --demand-pattern $DEMAND_PATTERN --cost-type $COST_TYPE"
    echo "2. Evaluate the trained model using:"
    echo "   python3 eval_iso_zoo.py --model-path models/iso_zoo/${ALGO}/final_model_iso.zip"
else
    echo "Optimization failed with exit code $exit_code"
    exit $exit_code
fi