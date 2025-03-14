#!/bin/bash

# Enable debug output and error handling
set -x
set -e

# Default values
ALGO="ppo"  # Default to lowercase
ENV="PCSUnitEnv-v0"
DEMAND_PATTERN="SINUSOIDAL"
COST_TYPE="CONSTANT"
TOTAL_TIMESTEPS=1000000
EVAL_FREQ=100
EVAL_EPISODES=5
SEED=42

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --algo)
            ALGO=$(echo "$2" | tr '[:upper:]' '[:lower:]')
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
        --total-timesteps)
            TOTAL_TIMESTEPS="$2"
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
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Create required directories
mkdir -p logs/pcs_zoo
mkdir -p models/pcs_zoo

# Run training
echo "Starting training with ${ALGO}..."
python train_pcs_zoo.py \
    --algo "${ALGO}" \
    --env "${ENV}" \
    --demand-pattern "${DEMAND_PATTERN}" \
    --cost-type "${COST_TYPE}" \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --eval-freq "${EVAL_FREQ}" \
    --eval-episodes "${EVAL_EPISODES}" \
    --seed "${SEED}"

TRAIN_EXIT_CODE=$?

# If training succeeded, run evaluation
if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
    echo "Training completed. Starting evaluation..."
    
    MODEL_PATH="models/pcs_zoo/${ALGO}/final_model_${ALGO}.zip"
    NORMALIZER_PATH="models/pcs_zoo/${ALGO}/final_model_normalizer.pkl"
    
    # Check if training outputs exist
    if [ ! -f "${MODEL_PATH}" ]; then
        echo "Error: Trained model not found at ${MODEL_PATH}"
        exit 1
    fi
    
    python eval_pcs_zoo.py \
        --algo "${ALGO}" \
        --env "${ENV}" \
        --model-path "${MODEL_PATH}" \
        --normalizer-path "${NORMALIZER_PATH}" \
        --demand-pattern "${DEMAND_PATTERN}" \
        --cost-type "${COST_TYPE}" \
        --n-eval-episodes "${EVAL_EPISODES}" \
        --deterministic \
        --output-dir "eval_results/${ALGO}_${DEMAND_PATTERN}_${COST_TYPE}"
    
    EVAL_EXIT_CODE=$?
    if [ ${EVAL_EXIT_CODE} -eq 0 ]; then
        echo "Evaluation completed successfully"
    else
        echo "Evaluation failed with code ${EVAL_EXIT_CODE}"
        exit 1
    fi
else
    echo "Training failed with code ${TRAIN_EXIT_CODE}"
    exit 1
fi 