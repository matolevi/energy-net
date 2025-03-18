"""
Power Consumption and Storage (PCS) Training Script

This script implements the core training pipeline for PCS agents using RL Zoo3 integration.
It enables training of agents that learn optimal battery charging and discharging strategies
in response to electricity prices set by the ISO.

The PCS agent makes decisions that affect:
1. When to charge/discharge batteries
2. How much energy to buy/sell from the grid
3. How to balance self-produced energy with consumption needs

The script supports multiple algorithms (PPO, A2C, SAC, TD3) and environment configurations,
and includes functionality for:
- Loading and processing hyperparameters
- Creating and configuring training environments
- Setting up evaluation callbacks
- Saving trained models and normalization statistics

Usage:
    python train_pcs_zoo.py --algo PPO --demand-pattern SINUSOIDAL --cost-type CONSTANT

Author: Energy-Net Team
Date: 2023-2024
"""

import os
import argparse
import yaml
import time
from typing import Any, Dict, Optional
from stable_baselines3.common.utils import set_random_seed, get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import gymnasium as gym
from gymnasium.wrappers import RescaleAction  # Add this import
from rl_zoo3.utils import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.exp_manager import ExperimentManager
from energy_net.utils.callbacks import ActionTrackingCallback
import torch  # Make sure this is at the top

from energy_net.market.iso.demand_patterns import DemandPattern
from energy_net.market.iso.cost_types import CostType

def parse_args():
    """
    Parse command line arguments for PCS agent training.
    
    This function defines all the available command-line options for configuring
    PCS agent training, including algorithm selection, environment settings,
    evaluation parameters, and logging options.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing algorithm selection,
        environment settings, and training parameters.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo", help="RL Algorithm", choices=list(ALGOS.keys()))
    parser.add_argument("--env", type=str, default="PCSUnitEnv-v0", help="environment ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--total-timesteps", type=int, default=1000000, help="Total timesteps")
    parser.add_argument("--demand-pattern", type=str, default="SINUSOIDAL", choices=[p.name for p in DemandPattern])
    parser.add_argument("--cost-type", type=str, default="CONSTANT", choices=[c.name for c in CostType])
    parser.add_argument("--tensorboard-log", type=str, default="logs/pcs_zoo/")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluate every n steps")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--save-freq", type=int, default=10000, help="Save model every n steps")
    parser.add_argument("--log-folder", type=str, default="logs/pcs_zoo/", help="Log folder")
    parser.add_argument("--conf-file", type=str, default=None, help="Path to experiment config file")
    parser.add_argument("--use-zoo-defaults", action="store_true", help="Use RL Zoo default hyperparameters")
    parser.add_argument("--record-video", action="store_true", help="Record videos of evaluation episodes")
    parser.add_argument("--record-video-freq", type=int, default=50000, help="Record video every n steps")
    return parser.parse_args()

def load_experiment_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load experiment configuration from YAML file.
    
    This function loads custom experiment configurations from a specified YAML file,
    which can include model architecture, hyperparameters, and environment settings.
    
    Args:
        config_path: Path to the configuration YAML file
        
    Returns:
        Dict[str, Any]: Configuration dictionary, empty if file not found
    """
    if config_path is None or not os.path.exists(config_path):
        return {}
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def process_hyperparameters(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process hyperparameters to convert string representations to actual Python objects.
    
    This function handles conversion of string representations of PyTorch activation functions
    to their actual class objects, which is required because YAML can't directly store class objects.
    
    Args:
        hyperparams: Dictionary containing hyperparameters loaded from YAML
        
    Returns:
        Dict[str, Any]: Processed hyperparameters with string references converted to objects
    """
    if "policy_kwargs" in hyperparams and "activation_fn" in hyperparams["policy_kwargs"]:
        if hyperparams["policy_kwargs"]["activation_fn"] == "torch.nn.Tanh":
            hyperparams["policy_kwargs"]["activation_fn"] = torch.nn.Tanh
    return hyperparams

def main():
    """
    Main function for training PCS agents.
    
    This is the core training pipeline that:
    1. Parses command line arguments
    2. Sets up the experiment and loads hyperparameters
    3. Creates and configures training and evaluation environments
    4. Initializes the RL model with the chosen algorithm
    5. Sets up callbacks for evaluation and checkpointing
    6. Trains the model and saves the results
    
    The function handles environment normalization, observation scaling, and hyperparameter
    loading from multiple possible sources (optimized params, RL Zoo defaults, or base params).
    """
    args = parse_args()
    
    # Load experiment config if provided
    exp_config = load_experiment_config(args.conf_file)
    
    # Set random seed
    set_random_seed(args.seed)

    # Create log dir
    os.makedirs(args.log_folder, exist_ok=True)

    # Load hyperparameters - try optimized first, then zoo defaults, then our base hyperparams
    hyperparams = {}
    optimized_params_path = os.path.join("hyperparams", "optimized", f"{args.algo}_best.yml")
    
    if os.path.exists(optimized_params_path):
        print(f"Loading optimized hyperparameters from {optimized_params_path}")
        with open(optimized_params_path, "r") as f:
            hyperparams = yaml.safe_load(f)[args.env]
            hyperparams = process_hyperparameters(hyperparams)
    elif args.use_zoo_defaults:
        print("Loading default hyperparameters from RL Zoo")
        try:
            hyperparams = get_saved_hyperparams(
                args.log_folder,
                args.algo,
                args.env,
                verbose=1,
                net_kwargs=exp_config.get("net_arch", None)
            )
        except ValueError:
            print("RL Zoo hyperparameters not found for this environment")
    
    # If no hyperparams loaded yet, try our base hyperparameters
    if not hyperparams:
        print("Loading base hyperparameters from pcs_unit.yml")
        base_params_path = os.path.join("hyperparams", "pcs_unit.yml")
        if os.path.exists(base_params_path):
            with open(base_params_path, "r") as f:
                config = yaml.safe_load(f)
                if args.env in config and args.algo in config[args.env]:
                    hyperparams = config[args.env][args.algo]
                    hyperparams = process_hyperparameters(hyperparams)
        
    if not hyperparams:
        print("No hyperparameters found, using default values")
        hyperparams = {
            "policy": "MlpPolicy",
            "n_steps": 2048,
            "batch_size": 64,
            "gamma": 0.99,
            "learning_rate": 3e-4,
            "ent_coef": 0.0,
            "clip_range": 0.2,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "max_grad_norm": 0.5,
            "vf_coef": 0.5,
            "normalize": True
        }

    # Create timestamp for unique experiment folder
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    exp_id = f"{timestamp}_{args.algo}"
    log_folder = os.path.join(args.log_folder, exp_id)
    os.makedirs(log_folder, exist_ok=True)
    print(f"Experiment output directory: {log_folder}")
    
    # Save hyperparameters to the log folder
    with open(os.path.join(log_folder, "hyperparameters.yml"), "w") as f:
        yaml.dump(hyperparams, f)
    
    # Create training environment directly instead of using ExperimentManager
    env = gym.make(
        args.env,
        demand_pattern=DemandPattern[args.demand_pattern],
        cost_type=CostType[args.cost_type],
        env_config_path='configs/environment_config.yaml',
        iso_config_path='configs/iso_config.yaml',
        pcs_unit_config_path='configs/pcs_unit_config.yaml'
    )
    
    # Add explicit action rescaling
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)
    
    # Wrap for monitoring with enhanced metrics
    env = Monitor(
        env,
        os.path.join(log_folder, 'train_monitor'),
        info_keywords=("battery_level", "net_exchange", "predicted_demand", "realized_demand")
    )
    
    # Create vectorized environment
    env = DummyVecEnv([lambda: env])
    
    # Add video recording if requested
    if args.record_video:
        env = VecVideoRecorder(
            env,
            log_folder,
            record_video_trigger=lambda x: x % args.record_video_freq == 0,
            video_length=2000
        )
    
    # Create evaluation environment directly
    eval_env = gym.make(
        args.env,
        demand_pattern=DemandPattern[args.demand_pattern],
        cost_type=CostType[args.cost_type],
        env_config_path='configs/environment_config.yaml',
        iso_config_path='configs/iso_config.yaml',
        pcs_unit_config_path='configs/pcs_unit_config.yaml'
    )
    
    # Add explicit action rescaling for evaluation environment too
    eval_env = RescaleAction(eval_env, min_action=-1.0, max_action=1.0)
    
    eval_env = Monitor(
        eval_env,
        os.path.join(log_folder, 'eval_monitor'),
        info_keywords=("battery_level", "net_exchange", "predicted_demand", "realized_demand")
    )
    
    eval_env = DummyVecEnv([lambda: eval_env])
    
    # Add video recording if requested
    if args.record_video:
        eval_env = VecVideoRecorder(
            eval_env,
            os.path.join(log_folder, "videos"),
            record_video_trigger=lambda x: x % args.record_video_freq == 0,
            video_length=2000
        )
    
    # Normalize both environments if specified
    if hyperparams.get('normalize', False):
        # Normalize training environment
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=1.0,
            clip_reward=1.0,
            gamma=hyperparams.get('gamma', 0.99),
            epsilon=1e-8
        )
        
        # Normalize evaluation environment with same settings
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=1.0,
            clip_reward=1.0,
            gamma=hyperparams.get('gamma', 0.99),
            epsilon=1e-8,
            training=False  # No updates during evaluation
        )
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_folder, "best_model"),
        log_path=log_folder,
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=args.eval_episodes
    )
    
    action_tracker = ActionTrackingCallback("PCS")
    callbacks = CallbackList([eval_callback, action_tracker])
    
    # Setup learning rate schedule if needed
    if "learning_rate" in hyperparams:
        hyperparams["learning_rate"] = get_schedule_fn(hyperparams["learning_rate"])
    
    # Create the model directly
    model = ALGOS[args.algo](
        policy=hyperparams.get('policy', 'MlpPolicy'),
        env=env,
        tensorboard_log=args.tensorboard_log,
        verbose=1,
        seed=args.seed,
        **{k: v for k, v in hyperparams.items() if k not in ['normalize', 'policy', 'n_envs', 'n_timesteps']}
    )

    try:
        # Train the model
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        # Save the final model
        model_dir = os.path.join("models", "pcs_zoo", args.algo)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"final_model_{args.algo}.zip")
        model.save(model_path)

        # Save the normalizer if it exists
        if isinstance(env, VecNormalize):
            normalizer_path = os.path.join(model_dir, "final_model_normalizer.pkl")
            env.save(normalizer_path)
            eval_env.save(os.path.join(model_dir, "final_model_eval_normalizer.pkl"))
        
        # Save experiment configuration
        config_path = os.path.join(log_folder, "experiment_config.yml")
        with open(config_path, "w") as f:
            yaml.dump({
                "args": vars(args),
                "hyperparams": hyperparams,
                "exp_config": exp_config
            }, f)
            
        print(f"Training completed. Final model and experiment data saved in {log_folder}")
        print(f"Experiment configuration saved to {config_path}")
        
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model...")
        model_dir = os.path.join("models", "pcs_zoo", args.algo)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "interrupted_model_ppo.zip")
        model.save(model_path)

if __name__ == "__main__":
    main()