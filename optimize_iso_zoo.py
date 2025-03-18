"""
Independent System Operator (ISO) Hyperparameter Optimization

This script implements hyperparameter optimization for ISO agents using Optuna.
It systematically searches for optimal hyperparameter configurations for the specified
RL algorithm to maximize the performance of ISO agents in grid management.

The optimization process:
1. Creates a search space for hyperparameters based on the selected algorithm
2. Trains multiple agents with different hyperparameter configurations
3. Evaluates each configuration's performance
4. Uses Bayesian optimization to guide the search toward promising regions
5. Saves the best hyperparameters for future use

Key hyperparameters optimized include:
- Learning rates
- Network architecture
- Training parameters (batch size, epochs, etc.)
- Algorithm-specific parameters

Usage:
    python optimize_iso_zoo.py --algo PPO --pricing-policy ONLINE --n-trials 100

"""

import os
import argparse
import optuna
import yaml
import numpy as np
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import gymnasium as gym
from gymnasium.wrappers import RescaleAction  # Add this import
from rl_zoo3.utils import ALGOS, create_test_env
from typing import Any, Dict
from energy_net.utils.callbacks import ActionTrackingCallback
from rl_zoo3.exp_manager import ExperimentManager
import torch  # Add this import

from energy_net.market.iso.demand_patterns import DemandPattern
from energy_net.market.iso.cost_types import CostType
from energy_net.env import PricingPolicy

def parse_args():
    """
    Parse command line arguments for ISO hyperparameter optimization.
    
    Returns:
        argparse.Namespace: Parsed command line arguments including algorithm selection,
        optimization settings, environment configuration, and dispatch settings.
    """
    parser = argparse.ArgumentParser(description="Optimize hyperparameters for ISO RL agents")
    parser.add_argument("--algo", type=str, default="ppo", help="RL algorithm to use")
    parser.add_argument("--env", type=str, default="ISOEnv-v0", help="Environment to use")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--n-timesteps", type=int, default=100000, help="Number of timesteps per trial")
    parser.add_argument("--study-name", type=str, default=None, help="Study name")
    parser.add_argument("--storage", type=str, default=None, help="Study storage location")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of episodes for evaluation")
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
    parser.add_argument("--use-dispatch-action", action="store_true", default=False,
                        help="Enable dispatch control in the agent's action space, expanding the "
                             "learning task to include both pricing and dispatch decisions")
    parser.add_argument("--dispatch-strategy", type=str, default="predicted_demand",
                        choices=["predicted_demand", "fixed", "scaled", "manual_profile", "daily_pattern"],
                        help="Default strategy for dispatch when not controlled by agent: "
                             "predicted_demand=match demand prediction, fixed=constant value, "
                             "scaled=apply scaling factor to demand, manual_profile=use defined profile")
    parser.add_argument("--trained-pcs-model-path", type=str, default=None,
                        help="Path to trained PCS model to use for PCS agent simulation")
    return parser.parse_args()

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sample hyperparameters for PPO algorithm using Optuna.
    
    This function defines the search space for Proximal Policy Optimization 
    hyperparameters, including optimization parameters, neural network architecture,
    and algorithm-specific settings.
    
    The search space is based on recommended ranges from literature and empirical testing.
    
    Args:
        trial: Current Optuna trial
        
    Returns:
        Dict[str, Any]: Dictionary of sampled hyperparameters
    """
    # Sample batch size from 32 to 256
    batch_size = trial.suggest_int("batch_size", 32, 256)
    
    # Sample n_steps (horizon length) from 16 to 2048
    n_steps = trial.suggest_int("n_steps", 16, 2048)
    
    # Sample discount factor (gamma) from 0.9 to 0.9999
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    
    # Sample learning rate on log scale
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    
    # Sample entropy coefficient on log scale
    # Controls exploration vs exploitation trade-off
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    
    # Sample clip range for PPO algorithm
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    
    # Sample number of PPO epochs
    n_epochs = trial.suggest_int("n_epochs", 3, 10)
    
    # Sample GAE lambda for advantage estimation
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.999)
    
    # Sample max gradient norm for gradient clipping
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0)
    
    # Sample value function coefficient
    vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
    
    # Sample neural network architecture
    pi_h1 = trial.suggest_int("pi_h1", 32, 128)  # Policy network first hidden layer
    pi_h2 = trial.suggest_int("pi_h2", 32, 128)  # Policy network second hidden layer
    vf_h1 = trial.suggest_int("vf_h1", 32, 128)  # Value function first hidden layer
    vf_h2 = trial.suggest_int("vf_h2", 32, 128)  # Value function second hidden layer
    
    # Return complete hyperparameter dictionary
    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        "policy_kwargs": {
            "net_arch": [
                {"pi": [pi_h1, pi_h2], "vf": [vf_h1, vf_h2]}
            ]
        },
        "policy": "MlpPolicy"  # Use MLP-based policy
    }

def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sample A2C hyperparameters
    """
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_int("n_steps", 4, 128)
    ent_coef = trial.suggest_float("ent_coef", 0.00001, 0.1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
    use_rms_prop = trial.suggest_categorical("use_rms_prop", [False, True])
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.999)
    normalize_advantage = trial.suggest_categorical("normalize_advantage", [False, True])
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0)
    
    # Network architecture
    pi_h1 = trial.suggest_int("pi_h1", 32, 128)
    pi_h2 = trial.suggest_int("pi_h2", 32, 128)
    vf_h1 = trial.suggest_int("vf_h1", 32, 128)
    vf_h2 = trial.suggest_int("vf_h2", 32, 128)
    
    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "use_rms_prop": use_rms_prop,
        "gae_lambda": gae_lambda,
        "normalize_advantage": normalize_advantage,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": {
            "net_arch": [
                {"pi": [pi_h1, pi_h2], "vf": [vf_h1, vf_h2]}
            ]
        },
        "policy": "MlpPolicy"
    }

def sample_sac_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sample SAC hyperparameters
    """
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 256)
    buffer_size = trial.suggest_int("buffer_size", 10000, 1000000)
    learning_starts = trial.suggest_int("learning_starts", 100, 10000)
    train_freq = trial.suggest_int("train_freq", 1, 10)
    tau = trial.suggest_float("tau", 0.001, 0.1)
    ent_coef = 'auto'
    gradient_steps = train_freq
    
    # Network architecture
    h1 = trial.suggest_int("h1", 32, 128)
    h2 = trial.suggest_int("h2", 32, 128)
    
    return {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "tau": tau,
        "ent_coef": ent_coef,
        "policy_kwargs": {
            "net_arch": [
                {"pi": [h1, h2], "vf": [h1, h2]}
            ]
        },
        "policy": "MlpPolicy"
    }

def sample_td3_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sample TD3 hyperparameters
    """
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 256)
    buffer_size = trial.suggest_int("buffer_size", 10000, 1000000)
    learning_starts = trial.suggest_int("learning_starts", 100, 10000)
    train_freq = trial.suggest_int("train_freq", 1, 10)
    tau = trial.suggest_float("tau", 0.001, 0.1)
    gradient_steps = train_freq
    policy_delay = trial.suggest_int("policy_delay", 1, 5)
    target_policy_noise = trial.suggest_float("target_policy_noise", 0.1, 0.5)
    
    # Network architecture
    h1 = trial.suggest_int("h1", 32, 128)
    h2 = trial.suggest_int("h2", 32, 128)
    
    return {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "tau": tau,
        "policy_delay": policy_delay,
        "target_policy_noise": target_policy_noise,
        "policy_kwargs": {
            "net_arch": [
                {"pi": [h1, h2], "vf": [h1, h2]}
            ]
        },
        "policy": "MlpPolicy"
    }

HYPERPARAMS_SAMPLER = {
    "ppo": sample_ppo_params,
    "a2c": sample_a2c_params,
    "sac": sample_sac_params,
    "td3": sample_td3_params,
}

def create_env(args):
    """
    Create the ISO environment with specified configurations for optimization.
    
    Sets up the environment with the appropriate pricing policy, demand pattern,
    cost type, and dispatch configuration.
    
    Args:
        args: Command line arguments containing environment settings
        
    Returns:
        callable: Function that creates and initializes an ISO environment for optimization
    """
    # Convert pricing_policy string to enum
    pricing_policy_enum = PricingPolicy[args.pricing_policy]
    
    def _init():
        # Create a dispatch configuration dict with comprehensive settings
        dispatch_config = {
            "use_dispatch_action": args.use_dispatch_action,  # Agent controls dispatch when True
            "default_strategy": args.dispatch_strategy        # Strategy used when not agent-controlled
        }
        
        env = gym.make(
            args.env,
            render_mode="rgb_array",
            pricing_policy=pricing_policy_enum,
            demand_pattern=DemandPattern[args.demand_pattern],
            cost_type=CostType[args.cost_type],
            num_pcs_agents=args.num_pcs_agents,
            dispatch_config=dispatch_config,  # Pass dispatch configuration
            trained_pcs_model_path=args.trained_pcs_model_path,  # Pass trained PCS model path
        )
        env = Monitor(env)
        return env
    
    return _init

def process_hyperparameters(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    """Process hyperparameters to convert string representations to actual objects"""
    if "policy_kwargs" in hyperparams and "activation_fn" in hyperparams["policy_kwargs"]:
        if isinstance(hyperparams["policy_kwargs"]["activation_fn"], str):
            if hyperparams["policy_kwargs"]["activation_fn"] == "torch.nn.Tanh":
                hyperparams["policy_kwargs"]["activation_fn"] = torch.nn.Tanh
            elif hyperparams["policy_kwargs"]["activation_fn"] == "torch.nn.ReLU":
                hyperparams["policy_kwargs"]["activation_fn"] = torch.nn.ReLU
    return hyperparams

def objective(trial: optuna.Trial, args) -> float:
    """
    Optimization objective function for Optuna trials.
    
    This function serves as the objective function for hyperparameter optimization:
    1. Samples hyperparameters from the appropriate algorithm-specific sampler
    2. Creates environments for training and evaluation
    3. Trains an agent with the sampled hyperparameters
    4. Evaluates performance and returns the mean reward as optimization objective
    
    Args:
        trial: Current Optuna trial that provides hyperparameter suggestions
        args: Command line arguments containing fixed experiment settings
        
    Returns:
        float: Mean evaluation reward (higher is better)
        
    Raises:
        ValueError: If the specified algorithm is not supported
    """
    # === SETUP PHASE ===
    # Get algorithm name and validate it's supported
    algo = args.algo.lower()
    if algo not in HYPERPARAMS_SAMPLER:
        raise ValueError(f"Algorithm {algo} not supported! Try: {list(HYPERPARAMS_SAMPLER.keys())}")
    
    # Sample hyperparameters for this trial
    hyperparams = HYPERPARAMS_SAMPLER[algo](trial)
    
    # Process hyperparameters (convert string representations to objects)
    hyperparams = process_hyperparameters(hyperparams)
    
    # Setup learning rate schedule if needed
    if "learning_rate" in hyperparams:
        schedule_fn = get_schedule_fn(hyperparams["learning_rate"])
        hyperparams["learning_rate"] = schedule_fn
    
    # Create log directory for this specific trial
    log_path = os.path.join("logs", "iso_zoo_optimize", args.algo, f"trial_{trial.number}")
    os.makedirs(log_path, exist_ok=True)
    
    # === ENVIRONMENT CREATION ===
    # Get pricing policy enum from string
    pricing_policy_enum = PricingPolicy[args.pricing_policy]
    
    # Create training environment with monitoring
    dispatch_config = {
        "use_dispatch_action": args.use_dispatch_action,
        "default_strategy": args.dispatch_strategy
    }
    
    env = gym.make(
        args.env, 
        render_mode="rgb_array",
        pricing_policy=pricing_policy_enum,
        demand_pattern=DemandPattern[args.demand_pattern],
        cost_type=CostType[args.cost_type],
        num_pcs_agents=args.num_pcs_agents,
        dispatch_config=dispatch_config,
        trained_pcs_model_path=args.trained_pcs_model_path,
    )
    
    # Add monitoring wrapper
    env = Monitor(
        env,
        os.path.join(log_path, 'train_monitor')
    )
    
    # Create vectorized environment (for compatibility)
    env = DummyVecEnv([lambda: env])
    
    # Apply observation and reward normalization
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.,
        clip_reward=10.,
        gamma=hyperparams.get('gamma', 0.99),
        epsilon=1e-8
    )
    
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    
    # Create separate evaluation environment
    eval_env = gym.make(
        args.env, 
        render_mode="rgb_array",
        pricing_policy=pricing_policy_enum,
        demand_pattern=DemandPattern[args.demand_pattern],
        cost_type=CostType[args.cost_type],
        num_pcs_agents=args.num_pcs_agents,
        dispatch_config=dispatch_config,
        trained_pcs_model_path=args.trained_pcs_model_path,
    )
    
    eval_env = Monitor(
        eval_env,
        os.path.join(log_path, 'eval_monitor')
    )
    
    eval_env = DummyVecEnv([lambda: eval_env])
    
    # Apply normalization to evaluation env
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.,
        clip_reward=10.,
        gamma=hyperparams.get('gamma', 0.99),
        epsilon=1e-8
    )
    
    # Share normalization stats between training and evaluation
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms
    eval_env.norm_reward = False  # Don't normalize rewards for evaluation
    
    # === CALLBACK SETUP ===
    # Create action tracking callback
    action_tracker = ActionTrackingCallback(agent_name="ISO")
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        callback_after_eval=action_tracker,
        best_model_save_path=log_path,
        log_path=log_path,
        eval_freq=min(2000, args.n_timesteps // 10),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False
    )
    
    # Combine callbacks
    callbacks = CallbackList([
        eval_callback,
        action_tracker
    ])
    
    # === TRAINING AND EVALUATION ===
    try:
        # Create model with tensorboard logging
        model = ALGOS[algo](
            env=env,
            tensorboard_log=log_path,
            verbose=0,  # Reduce console output during optimization
            seed=args.seed,
            **hyperparams
        )
        
        # Train the agent
        model.learn(total_timesteps=args.n_timesteps, callback=callbacks)
        
        # Clean up environments
        env.close()
        eval_env.close()
        
        # Return the best mean reward as optimization objective
        return eval_callback.best_mean_reward
        
    except Exception as e:
        print(f"Trial failed: {e}")
        # Return very negative value to indicate failed trial
        return float("-inf")

def main():
    args = parse_args()
    
    # Create required directories
    os.makedirs(os.path.join("logs", "iso_zoo_optimize"), exist_ok=True)
    os.makedirs(os.path.join("hyperparams", "optimized"), exist_ok=True)
    
    # Convert pricing_policy string to enum
    pricing_policy_enum = PricingPolicy[args.pricing_policy]
    
    # Register the environment
    try:
        gym.make(
            args.env,
            pricing_policy=pricing_policy_enum,
            demand_pattern=DemandPattern[args.demand_pattern],
            cost_type=CostType[args.cost_type],
            num_pcs_agents=args.num_pcs_agents
        )
    except gym.error.NameNotFound:
        print(f"Environment {args.env} not found. Make sure it's registered.")
        return
        
    # Default study name if not provided
    if args.study_name is None:
        args.study_name = f"{args.algo}_{args.env}"
    
    # Create Optuna study
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=int(args.n_timesteps * 0.1))
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True
    )
    
    try:
        study.optimize(
            lambda trial: objective(trial, args),
            n_trials=args.n_trials,
            timeout=None
        )
    except KeyboardInterrupt:
        print("Optimization stopped.")
    
    print("Number of finished trials: ", len(study.trials))
    
    print("Best trial:")
    trial = study.best_trial
    
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best hyperparameters to YAML file
    best_params = HYPERPARAMS_SAMPLER[args.algo.lower()](trial)
    
    # Save the best parameters
    output_path = os.path.join("hyperparams", "optimized", f"{args.algo.lower()}_iso_best.yml")
    
    # Convert torch objects to string representation before saving
    if "policy_kwargs" in best_params and "activation_fn" in best_params["policy_kwargs"]:
        if best_params["policy_kwargs"]["activation_fn"] == torch.nn.Tanh:
            best_params["policy_kwargs"]["activation_fn"] = "torch.nn.Tanh"
        elif best_params["policy_kwargs"]["activation_fn"] == "torch.nn.ReLU":
            best_params["policy_kwargs"]["activation_fn"] = "torch.nn.ReLU"
    
    with open(output_path, "w") as f:
        yaml.dump(best_params, f)
    
    print(f"Saved best hyperparameters to {output_path}")

if __name__ == "__main__":
    main()