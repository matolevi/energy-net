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
    parser = argparse.ArgumentParser()
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
    return parser.parse_args()

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sample PPO hyperparameters
    """
    batch_size = trial.suggest_int("batch_size", 32, 256)
    n_steps = trial.suggest_int("n_steps", 16, 2048)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    n_epochs = trial.suggest_int("n_epochs", 3, 10)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.999)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
    
    # Network architecture
    pi_h1 = trial.suggest_int("pi_h1", 32, 128)
    pi_h2 = trial.suggest_int("pi_h2", 32, 128)
    vf_h1 = trial.suggest_int("vf_h1", 32, 128)
    vf_h2 = trial.suggest_int("vf_h2", 32, 128)
    
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
        "policy": "MlpPolicy"
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
    """Create the ISO environment with specified configurations"""
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
    """Optimization objective for the ISO environment"""
    algo = args.algo.lower()
    if algo not in HYPERPARAMS_SAMPLER:
        raise ValueError(f"Algorithm {algo} not supported! Try: {list(HYPERPARAMS_SAMPLER.keys())}")
    
    # Sample hyperparameters
    hyperparams = HYPERPARAMS_SAMPLER[algo](trial)
    
    # Process hyperparameters
    hyperparams = process_hyperparameters(hyperparams)
    
    # Setup learning rate schedule if needed
    if "learning_rate" in hyperparams:
        schedule_fn = get_schedule_fn(hyperparams["learning_rate"])
        hyperparams["learning_rate"] = schedule_fn
    
    # Create log dir for this trial
    log_path = os.path.join("logs", "iso_zoo_optimize", args.algo, f"trial_{trial.number}")
    os.makedirs(log_path, exist_ok=True)
    
    # Create env with pricing policy converted to enum
    pricing_policy_enum = PricingPolicy[args.pricing_policy]
    
    # Training environment with monitoring
    env = gym.make(
        args.env, 
        render_mode="rgb_array",
        pricing_policy=pricing_policy_enum,
        demand_pattern=DemandPattern[args.demand_pattern],
        cost_type=CostType[args.cost_type],
        num_pcs_agents=args.num_pcs_agents
    )
    
    env = Monitor(
        env,
        os.path.join(log_path, 'train_monitor')
    )
    
    env = DummyVecEnv([lambda: env])
    
    # Apply VecNormalize - we'll handle this separately from the hyperparameters
    should_normalize = True  # You can make this a parameter if needed
    if should_normalize:
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.,
            clip_reward=10.,
            gamma=hyperparams.get('gamma', 0.99),
            epsilon=1e-8
        )
    
    # Set the random seed for reproducibility
    set_random_seed(args.seed)
    
    # Create evaluation env
    eval_env = gym.make(
        args.env, 
        render_mode="rgb_array",
        pricing_policy=pricing_policy_enum,
        demand_pattern=DemandPattern[args.demand_pattern],
        cost_type=CostType[args.cost_type],
        num_pcs_agents=args.num_pcs_agents
    )
    
    eval_env = Monitor(
        eval_env,
        os.path.join(log_path, 'eval_monitor')
    )
    
    eval_env = DummyVecEnv([lambda: eval_env])
    
    # Apply normalization to evaluation env if used during training
    if should_normalize:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.,
            clip_reward=10.,
            gamma=hyperparams.get('gamma', 0.99),
            epsilon=1e-8
        )
        eval_env.obs_rms = env.obs_rms
        eval_env.ret_rms = env.ret_rms
        # Don't normalize rewards for evaluation
        eval_env.norm_reward = False
    
    # Create the action tracking callback and evaluation callback
    action_tracker = ActionTrackingCallback(agent_name="ISO")
    
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
    
    # Combine all callbacks
    callbacks = CallbackList([
        eval_callback,
        action_tracker
    ])
    
    # Train the agent
    try:
        # Create and train the model with tensorboard logging
        model = ALGOS[algo](
            env=env,
            tensorboard_log=log_path,
            verbose=0,
            seed=args.seed,
            **hyperparams
        )
        
        model.learn(total_timesteps=args.n_timesteps, callback=callbacks)
        
        # Close environments
        env.close()
        eval_env.close()
        
        # Return the best evaluation mean reward
        return eval_callback.best_mean_reward
    except Exception as e:
        print(f"Trial failed: {e}")
        # Return a very negative value to indicate failure
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
        elif best_params["policy_kwargs"]["activation_fn"] == torch.nn.ReLU:
            best_params["policy_kwargs"]["activation_fn"] = "torch.nn.ReLU"
    
    with open(output_path, "w") as f:
        yaml.dump(best_params, f)
    
    print(f"Saved best hyperparameters to {output_path}")

if __name__ == "__main__":
    main()