#!/bin/bash

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

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --algo)
            ALGO="$2"
            shift 2
            ;;
        --conf-file)
            CONF_FILE="$2"
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
        --total-timesteps)
            TOTAL_TIMESTEPS="$2"
            shift 2
            ;;
        --eval-freq)
            EVAL_FREQ="$2"
            shift 2
            ;;
        --eval-episodes)
            EVAL_EPISODES="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --log-folder)
            LOG_FOLDER="$2"
            shift 2
            ;;
        --save-freq)
            SAVE_FREQ="$2"
            shift 2
            ;;
        --use-zoo-defaults)
            USE_OPTIMIZED=false
            shift 1
            ;;
        --record-video)
            RECORD_VIDEO=true
            shift 1
            ;;
        --record-video-freq)
            RECORD_VIDEO_FREQ="$2"
            shift 2
            ;;
        --tensorboard-log)
            TENSORBOARD_LOG="$2"
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

# Print training parameters
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

# Construct command
COMMAND="python3 train_iso_zoo.py --algo $ALGO --env $ENV"
COMMAND="$COMMAND --demand-pattern $DEMAND_PATTERN --cost-type $COST_TYPE"
COMMAND="$COMMAND --pricing-policy $PRICING_POLICY --num-pcs-agents $NUM_PCS_AGENTS"
COMMAND="$COMMAND --total-timesteps $TOTAL_TIMESTEPS --eval-freq $EVAL_FREQ"
COMMAND="$COMMAND --eval-episodes $EVAL_EPISODES --seed $SEED --log-folder $LOG_FOLDER"
COMMAND="$COMMAND --save-freq $SAVE_FREQ"

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

# Run evaluation automatically
# Convert algorithm name to lowercase for directory paths
ALGO_LOWER=$(echo "$ALGO" | tr '[:upper:]' '[:lower:]')

# Construct evaluation command
EVAL_COMMAND="python3 eval_iso_zoo.py --algo $ALGO --env $ENV"
EVAL_COMMAND="$EVAL_COMMAND --model-path models/iso_zoo/$ALGO_LOWER/final_model_iso.zip"
EVAL_COMMAND="$EVAL_COMMAND --normalizer-path models/iso_zoo/$ALGO_LOWER/final_model_normalizer.pkl"
EVAL_COMMAND="$EVAL_COMMAND --demand-pattern $DEMAND_PATTERN --cost-type $COST_TYPE"
EVAL_COMMAND="$EVAL_COMMAND --pricing-policy $PRICING_POLICY --num-pcs-agents $NUM_PCS_AGENTS"
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