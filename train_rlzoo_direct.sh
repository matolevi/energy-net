#!/bin/bash
# Training script that uses RL-Zoo3 with direct callback

# Create output directories
mkdir -p logs/iso/ppo/run_1
mkdir -p logs/pcs/ppo/run_1
mkdir -p rl-baselines3-zoo/hyperparams/ppo

# Ensure our environment is on PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create or update the hyperparameter files
cat > rl-baselines3-zoo/hyperparams/ppo/ISO-RLZoo-v0.yml << EOF
ISO-RLZoo-v0:
  env_wrapper:
    - gymnasium.wrappers.RescaleAction:
        min_action: -1.0
        max_action: 1.0
  
  normalize: "{'norm_obs': True, 'norm_reward': True}"
  n_envs: 1
  n_timesteps: !!float 1e6
  policy: 'MlpPolicy'
  n_steps: 512
  batch_size: 64
  gae_lambda: 0.95
  gamma: 0.99
  n_epochs: 10
  ent_coef: 0.3
  learning_rate: !!float 3e-4
  clip_range: 0.2
  max_grad_norm: 0.5
  vf_coef: 0.5
  
  policy_kwargs:
    net_arch:
      pi: [64, 64]
      vf: [64, 64]
    
  # Use our direct callback class
  callback: plot_callback.PlotCallback
EOF

cat > rl-baselines3-zoo/hyperparams/ppo/PCS-RLZoo-v0.yml << EOF
PCS-RLZoo-v0:
  env_wrapper:
    - gymnasium.wrappers.RescaleAction:
        min_action: -1.0
        max_action: 1.0
  
  normalize: "{'norm_obs': True, 'norm_reward': True}"
  n_envs: 1
  n_timesteps: !!float 1e6
  policy: 'MlpPolicy'
  n_steps: 512
  batch_size: 64
  gae_lambda: 0.95
  gamma: 0.99
  n_epochs: 10
  ent_coef: 0.3
  learning_rate: !!float 3e-4
  clip_range: 0.2
  max_grad_norm: 0.5
  vf_coef: 0.5
  
  policy_kwargs:
    net_arch:
      pi: [64, 64]
      vf: [64, 64]
    
  # Use our direct callback class
  callback: plot_callback.PlotCallback
EOF

# Define environment kwargs
BASE_ENV_KWARGS=(
  "cost_type:'CONSTANT'"
  "pricing_policy:'Online'"
  "demand_pattern:'CONSTANT'"
  "use_dispatch_action:True"
)

# Set number of iterations for alternating training
ITERATIONS=5
# Set training steps per iteration
TIMESTEPS=50
# Set random seed
SEED=422

echo "Starting alternating training of ISO and PCS agents with $ITERATIONS iterations..."

# First iteration: Train ISO agent alone
ITERATION=1
echo "Iteration $ITERATION: Training ISO agent..."

ISO_ENV_KWARGS=("${BASE_ENV_KWARGS[@]}")

python -m rl_zoo3.train \
  --algo ppo \
  --env ISO-RLZoo-v0 \
  --gym-packages energy_net.env.register_envs \
  --eval-freq 50 \
  --eval-episodes 10 \
  --save-freq 50 \
  --log-folder logs/iso/ppo/run_1 \
  --tensorboard-log logs/iso/tensorboard/run_1 \
  --env-kwargs "${ISO_ENV_KWARGS[@]}" \
  --n-timesteps $TIMESTEPS \
  --seed $SEED \
  -conf rl-baselines3-zoo/hyperparams/ppo/ISO-RLZoo-v0.yml

# Define initial model paths
ISO_MODEL_PATH="logs/iso/ppo/run_1/ppo/ISO-RLZoo-v0_1/ISO-RLZoo-v0.zip"

# Check if ISO training succeeded
if [ ! -f "$ISO_MODEL_PATH" ]; then
  ISO_MODEL_PATH="logs/iso/ppo/run_1/ISO-RLZoo-v0.zip"
  if [ ! -f "$ISO_MODEL_PATH" ]; then
    echo "ERROR: ISO training failed. Model not found."
    exit 1
  fi
fi

# Now alternate between training PCS and ISO
for ((ITERATION=1; ITERATION<=$ITERATIONS; ITERATION++)); do
  # Train PCS agent with fixed ISO policy
  echo "Iteration $ITERATION: Training PCS agent with fixed ISO policy..."
  
  PCS_ENV_KWARGS=("${BASE_ENV_KWARGS[@]}")
  PCS_ENV_KWARGS+=("iso_policy_path:'$ISO_MODEL_PATH'")
  
  python -m rl_zoo3.train \
    --algo ppo \
    --env PCS-RLZoo-v0 \
    --gym-packages energy_net.env.register_envs \
    --eval-freq 50 \
    --eval-episodes 10 \
    --save-freq 50 \
    --log-folder logs/pcs/ppo/run_1 \
    --tensorboard-log logs/pcs/tensorboard/run_1 \
    --env-kwargs "${PCS_ENV_KWARGS[@]}" \
    --n-timesteps $TIMESTEPS \
    --seed $SEED \
    -conf rl-baselines3-zoo/hyperparams/ppo/PCS-RLZoo-v0.yml
  
  # Define PCS model path
  PCS_MODEL_PATH="logs/pcs/ppo/run_1/ppo/PCS-RLZoo-v0_1/PCS-RLZoo-v0.zip"
  if [ ! -f "$PCS_MODEL_PATH" ]; then
    PCS_MODEL_PATH="logs/pcs/ppo/run_1/PCS-RLZoo-v0.zip"
    if [ ! -f "$PCS_MODEL_PATH" ]; then
      echo "ERROR: PCS training failed. Model not found."
      exit 1
    fi
  fi
  
  # Skip ISO training in the last iteration
  if [ $ITERATION -eq $ITERATIONS ]; then
    break
  fi
  
  # Train ISO agent with fixed PCS policy
  echo "Iteration $((ITERATION+1)): Training ISO agent with fixed PCS policy..."
  
  ISO_ENV_KWARGS=("${BASE_ENV_KWARGS[@]}")
  ISO_ENV_KWARGS+=("pcs_policy_path:'$PCS_MODEL_PATH'")
  
  python -m rl_zoo3.train \
    --algo ppo \
    --env ISO-RLZoo-v0 \
    --gym-packages energy_net.env.register_envs \
    --eval-freq 50 \
    --eval-episodes 10 \
    --save-freq 50 \
    --log-folder logs/iso/ppo/run_1 \
    --tensorboard-log logs/iso/tensorboard/run_1 \
    --env-kwargs "${ISO_ENV_KWARGS[@]}" \
    --n-timesteps $TIMESTEPS \
    --seed $SEED \
    -conf rl-baselines3-zoo/hyperparams/ppo/ISO-RLZoo-v0.yml
  
  # Update ISO model path for next iteration
  ISO_MODEL_PATH="logs/iso/ppo/run_1/ppo/ISO-RLZoo-v0_1/ISO-RLZoo-v0.zip"
  if [ ! -f "$ISO_MODEL_PATH" ]; then
    ISO_MODEL_PATH="logs/iso/ppo/run_1/ISO-RLZoo-v0.zip"
    if [ ! -f "$ISO_MODEL_PATH" ]; then
      echo "ERROR: ISO training failed. Model not found."
      exit 1
    fi
  fi
done

echo "Alternating training completed!"
echo "Plots should be saved in the respective model directories." 