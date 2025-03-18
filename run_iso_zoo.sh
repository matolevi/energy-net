#!/bin/bash
# ===============================================================================
# Independent System Operator (ISO) Training Script
# ===============================================================================
#
# This script provides a convenient command-line interface for training ISO agents.
# It handles parameter parsing, environment setup, and initiates both training
# and evaluation of ISO agents.
#
# The ISO agent learns to:
# - Set electricity buy/sell prices
# - Optionally control dispatch amounts
# - Balance the grid between supply and demand
# - Minimize operation costs
#
# This script supports multiple algorithms (PPO, A2C, SAC, TD3) and
# configuration options for environment settings.
#
# Usage:
#   ./run_iso_zoo.sh --algo PPO --pricing-policy ONLINE --demand-pattern SINUSOIDAL
# ===============================================================================

# Default parameters
ALGO="ppo"
ENV="ISOEnv-v0"
DEMAND_PATTERN="SINUSOIDAL"
COST_TYPE="CONSTANT"
PRICING_POLICY="ONLINE"
NUM_PCS_AGENTS=1
TOTAL_TIMESTEPS=1000000
EVAL_FREQ=10000
EVAL_EPISODES=5
SEED=42
LOG_FOLDER="logs/iso_zoo"
SAVE_FREQ=50000
USE_OPTIMIZED=true
RECORD_VIDEO=false
RECORD_VIDEO_FREQ=10000
TENSORBOARD_LOG=NULL
USE_DISPATCH_ACTION=false
DISPATCH_STRATEGY="predicted_demand"
TRAINED_PCS_MODEL_PATH=NULL

# Parse command line arguments with enhanced documentation
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --algo)
            # RL algorithm to train (PPO, A2C, SAC, TD3)
            ALGO="$2"
            shift 2
            ;;
        --conf-file)
            # Custom configuration file path
            CONF_FILE="$2"
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
        --total-timesteps)
            # Total timesteps for training
            TOTAL_TIMESTEPS="$2"
            shift 2
            ;;
        --eval-freq)
            # Evaluation frequency in timesteps
            EVAL_FREQ="$2"
            shift 2
            ;;
        --eval-episodes)
            # Number of evaluation episodes
            EVAL_EPISODES="$2"
            shift 2
            ;;
        --seed)
            # Random seed for reproducibility
            SEED="$2"
            shift 2
            ;;
        --log-folder)
            # Log folder path
            LOG_FOLDER="$2"
            shift 2
            ;;
        --save-freq)
            # Model save frequency
            SAVE_FREQ="$2"
            shift 2
            ;;
        --use-zoo-defaults)
            # Use RL Zoo's default hyperparameters instead of optimized ones
            USE_OPTIMIZED=false
            shift 1
            ;;
        --record-video)
            # Enable video recording
            RECORD_VIDEO=true
            shift 1
            ;;
        --record-video-freq)
            # Video recording frequency
            RECORD_VIDEO_FREQ="$2"
            shift 2
            ;;
        --tensorboard-log)
            # TensorBoard log directory
            TENSORBOARD_LOG="$2"
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

# Create necessary directories
mkdir -p $LOG_FOLDER
mkdir -p models/iso_zoo/$ALGO

# Print training parameters with clear documentation
echo "Starting ISO agent training with:"
echo "  Algorithm: $ALGO"
echo "  Environment: $ENV"
echo "  Demand Pattern: $DEMAND_PATTERN"
echo "  Cost Type: $COST_TYPE"
echo "  Pricing Policy: $PRICING_POLICY"
echo "  Number of PCS Agents: $NUM_PCS_AGENTS"
echo "  Total Timesteps: $TOTAL_TIMESTEPS"
echo "  Evaluation Frequency: $EVAL_FREQ"
echo "  Evaluation Episodes: $EVAL_EPISODES"
echo "  Seed: $SEED"
echo "  Use Dispatch Action: $USE_DISPATCH_ACTION  # Whether agent controls dispatch"
echo "  Dispatch Strategy: $DISPATCH_STRATEGY  # Strategy used when agent doesn't control dispatch"
if [ "$TRAINED_PCS_MODEL_PATH" != "NULL" ]; then
    echo "  Trained PCS Model Path: $TRAINED_PCS_MODEL_PATH"
fi

# Construct training command with comprehensive documentation
COMMAND="python3 train_iso_zoo.py --algo $ALGO --env $ENV"
COMMAND="$COMMAND --demand-pattern $DEMAND_PATTERN --cost-type $COST_TYPE"
COMMAND="$COMMAND --pricing-policy $PRICING_POLICY --num-pcs-agents $NUM_PCS_AGENTS"
COMMAND="$COMMAND --total-timesteps $TOTAL_TIMESTEPS --eval-freq $EVAL_FREQ"
COMMAND="$COMMAND --eval-episodes $EVAL_EPISODES --seed $SEED --log-folder $LOG_FOLDER"
COMMAND="$COMMAND --save-freq $SAVE_FREQ"

# Add optional configurations
if [ -n "$CONF_FILE" ]; then
    COMMAND="$COMMAND --hyperparams $CONF_FILE"
fi

if [ "$USE_OPTIMIZED" = false ]; then
    COMMAND="$COMMAND --use-optimized false"
fi

if [ "$RECORD_VIDEO" = true ]; then
    COMMAND="$COMMAND --record-video --record-video-freq $RECORD_VIDEO_FREQ"
fi

if [ "$TENSORBOARD_LOG" != "NULL" ]; then
    COMMAND="$COMMAND --tensorboard-log $TENSORBOARD_LOG"
fi

if [ "$USE_DISPATCH_ACTION" = true ]; then
    COMMAND="$COMMAND --use-dispatch-action"
fi

if [ "$TRAINED_PCS_MODEL_PATH" != "NULL" ]; then
    COMMAND="$COMMAND --trained-pcs-model-path $TRAINED_PCS_MODEL_PATH"
fi

COMMAND="$COMMAND --dispatch-strategy $DISPATCH_STRATEGY"

# Run training
echo "Running command: $COMMAND"
eval $COMMAND

# Check training exit code
TRAIN_EXIT_CODE=$?
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

echo "Training completed successfully!"
echo "Model saved to: models/iso_zoo/${ALGO}/final_model_iso.zip"
echo "Normalizer saved to: models/iso_zoo/${ALGO}/final_model_normalizer.pkl"

# Run evaluation automatically after training
# Convert algorithm name to lowercase for directory paths
ALGO_LOWER=$(echo "$ALGO" | tr '[:upper:]' '[:lower:]')

# Construct evaluation command with detailed options
EVAL_COMMAND="python3 eval_iso_zoo.py --algo $ALGO --env $ENV"
EVAL_COMMAND="$EVAL_COMMAND --model-path models/iso_zoo/$ALGO_LOWER/final_model_iso.zip"
EVAL_COMMAND="$EVAL_COMMAND --normalizer-path models/iso_zoo/$ALGO_LOWER/final_model_normalizer.pkl"
EVAL_COMMAND="$EVAL_COMMAND --demand-pattern $DEMAND_PATTERN --cost-type $COST_TYPE"
EVAL_COMMAND="$EVAL_COMMAND --pricing-policy $PRICING_POLICY --num-pcs-agents $NUM_PCS_AGENTS"

if [ "$USE_DISPATCH_ACTION" = true ]; then
    EVAL_COMMAND="$EVAL_COMMAND --use-dispatch-action"
fi

if [ "$TRAINED_PCS_MODEL_PATH" != "NULL" ]; then
    EVAL_COMMAND="$EVAL_COMMAND --trained-pcs-model-path $TRAINED_PCS_MODEL_PATH"
fi

EVAL_COMMAND="$EVAL_COMMAND --dispatch-strategy $DISPATCH_STRATEGY"
EVAL_COMMAND="$EVAL_COMMAND --n-eval-episodes $EVAL_EPISODES --deterministic"
EVAL_COMMAND="$EVAL_COMMAND --output-dir eval_results/${ALGO}_iso_${DEMAND_PATTERN}_${COST_TYPE}"

echo "Running evaluation: $EVAL_COMMAND"
eval $EVAL_COMMAND

# Check evaluation exit code
EVAL_EXIT_CODE=$?
if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo "Evaluation failed with exit code $EVAL_EXIT_CODE"
    exit $EVAL_EXIT_CODE
fi

echo "Evaluation completed successfully!"
echo "Evaluation results saved to: eval_results/${ALGO}_iso_${DEMAND_PATTERN}_${COST_TYPE}/"
echo ""
echo "You can visualize training progress with:"
echo "  tensorboard --logdir $LOG_FOLDER"