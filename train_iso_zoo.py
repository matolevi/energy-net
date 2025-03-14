import os
import argparse
import gymnasium as gym
import numpy as np
import yaml
import json
from datetime import datetime
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from energy_net.utils.callbacks import ActionTrackingCallback
from rl_zoo3.utils import ALGOS
from typing import Any, Dict, Optional
from gymnasium.wrappers import RescaleAction
import torch  # Add this import at the top

# Add the missing imports
from energy_net.market.iso.demand_patterns import DemandPattern
from energy_net.market.iso.cost_types import CostType
from energy_net.env import PricingPolicy

def parse_args():
    """Parse command line arguments for ISO agent training"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo", help="RL algorithm to use")
    parser.add_argument("--env", type=str, default="ISOEnv-v0", help="Environment ID")
    parser.add_argument("--hyperparams", type=str, default=None, help="Hyperparameters file path")
    parser.add_argument("--log-folder", type=str, default="logs/iso_zoo", help="Log folder path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--total-timesteps", type=int, default=1000000, help="Total timesteps for training")
    parser.add_argument("--save-freq", type=int, default=50000, help="Model save frequency")
    parser.add_argument("--demand-pattern", type=str, default="SINUSOIDAL", 
                      choices=["SINUSOIDAL", "RANDOM", "PERIODIC", "SPIKES"],
                      help="Demand pattern to use")
    parser.add_argument("--cost-type", type=str, default="CONSTANT", 
                      choices=["CONSTANT", "VARIABLE", "TIME_OF_USE"],
                      help="Cost type to use")
    parser.add_argument("--pricing-policy", type=str, default="ONLINE",
                      choices=["QUADRATIC", "ONLINE", "CONSTANT"],
                      help="Pricing policy to use")
    parser.add_argument("--num-pcs-agents", type=int, default=1,
                      help="Number of PCS agents to simulate in the environment")
    parser.add_argument("--record-video", action="store_true", default=False, help="Record videos")
    parser.add_argument("--record-video-freq", type=int, default=10000, help="Video recording frequency")
    parser.add_argument("--tensorboard-log", type=str, default=None, help="TensorBoard log directory")
    parser.add_argument("--use-optimized", action="store_true", default=True, 
                      help="Use optimized hyperparameters")
    return parser.parse_args()

def process_hyperparameters(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    """Process hyperparameters to convert string representations to actual objects"""
    if "policy_kwargs" in hyperparams and "activation_fn" in hyperparams["policy_kwargs"]:
        if hyperparams["policy_kwargs"]["activation_fn"] == "torch.nn.Tanh":
            hyperparams["policy_kwargs"]["activation_fn"] = torch.nn.Tanh
        elif hyperparams["policy_kwargs"]["activation_fn"] == "torch.nn.ReLU":
            hyperparams["policy_kwargs"]["activation_fn"] = torch.nn.ReLU
    return hyperparams

def load_hyperparams(args):
    """Load hyperparameters from file or use optimized if available"""
    hyperparams = {}
    
    if args.hyperparams is not None:
        # Load from specified file
        with open(args.hyperparams, "r") as f:
            hyperparams = yaml.safe_load(f)
    elif args.use_optimized:
        # Try to load optimized hyperparameters
        optimized_path = os.path.join("hyperparams", "optimized", f"{args.algo.lower()}_iso_best.yml")
        if os.path.exists(optimized_path):
            with open(optimized_path, "r") as f:
                hyperparams = yaml.safe_load(f)
                # Process hyperparameters to convert string representations to objects
                hyperparams = process_hyperparameters(hyperparams)
            print(f"Loaded optimized hyperparameters from {optimized_path}")
        else:
            print(f"No optimized hyperparameters found at {optimized_path}, using defaults")
    
    return hyperparams

def create_env(args):
    """Create ISO environment with the specified settings"""
    # Convert pricing_policy string to enum
    pricing_policy_enum = PricingPolicy[args.pricing_policy]
    
    def _init():
        env = gym.make(
            args.env,
            render_mode="rgb_array",
            pricing_policy=pricing_policy_enum,
            demand_pattern=DemandPattern[args.demand_pattern],
            cost_type=CostType[args.cost_type],
            num_pcs_agents=args.num_pcs_agents
        )
        
        # Print the actual action space to understand its bounds
        print(f"ISO native action space: {env.action_space.low} to {env.action_space.high}")
        
        # If your policy uses tanh activation (outputs in [-1,1])
        # then use RescaleAction to map from [-1,1] to the environment's range
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        
        # Monitor the environment for logging
        log_path = os.path.join(args.log_folder, f"{args.algo}_{args.demand_pattern}_{args.cost_type}", "train_monitor")
        os.makedirs(log_path, exist_ok=True)
        env = Monitor(env, log_path)
        
        return env
    
    return _init

def create_test_env(args):
    """Create environment for evaluation"""
    # Convert pricing_policy string to enum
    pricing_policy_enum = PricingPolicy[args.pricing_policy]
    
    def _init_test_env():
        env = gym.make(
            args.env,
            render_mode="rgb_array",
            pricing_policy=pricing_policy_enum,
            demand_pattern=DemandPattern[args.demand_pattern],
            cost_type=CostType[args.cost_type],
            num_pcs_agents=args.num_pcs_agents
        )
        
        # Add action rescaling to match PCS pipeline
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        
        # Monitor the environment for logging
        log_path = os.path.join(args.log_folder, f"{args.algo}_{args.demand_pattern}_{args.cost_type}", "eval_monitor")
        os.makedirs(log_path, exist_ok=True)
        env = Monitor(env, log_path)
        
        return env
    
    return _init_test_env

def setup_experiment(args, hyperparams):
    """Set up the experiment with the given hyperparameters"""
    # Create experiment timestamp for unique folder names
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"{timestamp}_{args.algo}"
    
    # Create log folder
    log_folder = os.path.join(args.log_folder, experiment_name)
    os.makedirs(log_folder, exist_ok=True)
    
    # Save experiment configuration
    config = {
        "algorithm": args.algo,
        "environment": args.env,
        "demand_pattern": args.demand_pattern,
        "cost_type": args.cost_type,
        "total_timesteps": args.total_timesteps,
        "seed": args.seed,
        "hyperparameters": hyperparams,
        "timestamp": timestamp
    }
    
    with open(os.path.join(log_folder, "experiment_config.yml"), "w") as f:
        yaml.dump(config, f)
    
    # Set up tensorboard logs
    tensorboard_log = args.tensorboard_log
    if tensorboard_log is None:
        tensorboard_log = os.path.join(args.log_folder, "tensorboard_logs")
    
    return log_folder, tensorboard_log

def train(args):
    """Main training function for ISO agent"""
    hyperparams = load_hyperparams(args)
    log_folder, tensorboard_log = setup_experiment(args, hyperparams)
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Create environments
    env_fn = create_env(args)
    env = DummyVecEnv([env_fn])
    
    # Always normalize the environment with explicit parameters matching PCS pipeline
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=1.0,
        clip_reward=1.0,
        gamma=hyperparams.get('gamma', 0.99),
        epsilon=1e-8
    )
    
    # Create evaluation env
    eval_env_fn = create_test_env(args)
    eval_env = DummyVecEnv([eval_env_fn])
    
    # Apply normalization to evaluation env with the same parameters
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=1.0,
        clip_reward=1.0,
        gamma=hyperparams.get('gamma', 0.99),
        epsilon=1e-8,
        training=False  # No updates during evaluation - added to match PCS
    )
    
    # Copy statistics from training env
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms
    # Don't normalize rewards for evaluation
    eval_env.norm_reward = False
    
    # Create model
    algo = args.algo.lower()
    model_class = ALGOS[algo]
    
    # Set up model kwargs
    model_kwargs = {"verbose": 1, "seed": args.seed, "policy": "MlpPolicy"}
    
    # Add tensorboard logging if specified
    if tensorboard_log:
        model_kwargs["tensorboard_log"] = tensorboard_log
    
    # Combine with hyperparameters
    model_kwargs.update(hyperparams)
    
    # Create model
    model = model_class(env=env, **model_kwargs)
    
    # Set up callbacks
    callbacks = []
    
    # Checkpoint callback
    save_path = os.path.join(log_folder, "checkpoints")
    os.makedirs(save_path, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=save_path,
        name_prefix="iso_model",
        save_replay_buffer=True,
        save_vecnormalize=True
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_path = os.path.join(log_folder, "evaluations")
    os.makedirs(eval_path, exist_ok=True)
    
    # Action tracking callback
    action_track_callback = ActionTrackingCallback(
        agent_name="ISO"
    )
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        callback_after_eval=action_track_callback,
        best_model_save_path=os.path.join(log_folder, "best_model"),
        log_path=eval_path,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    # Create callback list
    callback = CallbackList(callbacks)
    
    # Train the model
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        progress_bar=True  # Add progress bar for better training visualization
    )
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join("models", "iso_zoo", args.algo.lower())
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the final model
    model_path = os.path.join(models_dir, "final_model_iso.zip")
    model.save(model_path)
    print(f"Final model saved to {model_path}")
    
    # Save the VecNormalize statistics if used
    if isinstance(env, VecNormalize):
        norm_path = os.path.join(models_dir, "final_model_normalizer.pkl")
        env.save(norm_path)
        print(f"Saved normalizer to {norm_path}")
        
        # Also save it under a different name for evaluation
        eval_norm_path = os.path.join(models_dir, "final_model_eval_normalizer.pkl")
        env.save(eval_norm_path)
        # Set normalization of rewards to False for evaluation
        env.norm_reward = False
        print(f"Saved evaluation normalizer to {eval_norm_path}")
    
    # Close environments
    env.close()
    eval_env.close()
    
    print("Training completed!")
    print(f"Model saved to {model_path}")
    return model_path

if __name__ == "__main__":
    args = parse_args()
    train(args)