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
import torch

from energy_net.market.iso.demand_patterns import DemandPattern
from energy_net.market.iso.cost_types import CostType

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo", help="RL Algorithm", choices=list(ALGOS.keys()))
    parser.add_argument("--env", type=str, default="PCSUnitEnv-v0", help="environment ID")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of trials for optimization")
    parser.add_argument("--n-timesteps", type=int, default=100000, help="Number of timesteps per trial")
    parser.add_argument("--n-eval-episodes", type=int, default=5, help="Number of episodes for evaluation")
    parser.add_argument("--study-name", type=str, default=None, help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage path (SQLite URL)")
    parser.add_argument("--demand-pattern", type=str, default="SINUSOIDAL", choices=[p.name for p in DemandPattern])
    parser.add_argument("--cost-type", type=str, default="CONSTANT", choices=[c.name for c in CostType])
    return parser.parse_args()

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample hyperparameters for PPO algorithm"""
    return {
        "n_steps": trial.suggest_int("n_steps", 64, 2048, step=64),
        "batch_size": trial.suggest_int("batch_size", 64, 512, step=64),
        "gamma": trial.suggest_float("gamma", 0.98, 0.999),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.02),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
        "n_epochs": trial.suggest_int("n_epochs", 5, 10),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.98),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.5, 1.0),
        "vf_coef": trial.suggest_float("vf_coef", 0.4, 0.6),
        "policy": "MlpPolicy",
        "policy_kwargs": {
            "net_arch": {
                "pi": [trial.suggest_int("pi_h1", 32, 256), trial.suggest_int("pi_h2", 32, 256)],
                "vf": [trial.suggest_int("vf_h1", 32, 256), trial.suggest_int("vf_h2", 32, 256)]
            },
            "activation_fn": "torch.nn.Tanh"
        },
        "normalize_advantage": True
    }

def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample hyperparameters for A2C algorithm"""
    return {
        "n_steps": trial.suggest_int("n_steps", 32, 128, step=32),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.01),
        "vf_coef": trial.suggest_float("vf_coef", 0.3, 0.7),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 0.9),
        "rms_prop_eps": trial.suggest_float("rms_prop_eps", 1e-6, 1e-4, log=True),
        "policy": "MlpPolicy",
        "policy_kwargs": {
            "net_arch": {
                "pi": [trial.suggest_int("pi_h1", 32, 128), trial.suggest_int("pi_h2", 32, 128)],
                "vf": [trial.suggest_int("vf_h1", 32, 128), trial.suggest_int("vf_h2", 32, 128)]
            }
        }
    }

def sample_sac_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample hyperparameters for SAC algorithm"""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True),
        "buffer_size": trial.suggest_int("buffer_size", 50000, 200000, step=50000),
        "batch_size": trial.suggest_int("batch_size", 64, 512, step=64),
        "gamma": trial.suggest_float("gamma", 0.98, 0.999),
        "tau": trial.suggest_float("tau", 0.001, 0.01),
        "train_freq": trial.suggest_int("train_freq", 1, 10),
        "gradient_steps": trial.suggest_int("gradient_steps", 1, 10),
        "learning_starts": trial.suggest_int("learning_starts", 500, 2000),
        "policy": "MlpPolicy",
        "policy_kwargs": {
            "net_arch": [
                trial.suggest_int("hidden_size_1", 32, 256),
                trial.suggest_int("hidden_size_2", 32, 256)
            ],
            "activation_fn": "torch.nn.Tanh"
        }
    }

def sample_td3_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample hyperparameters for TD3 algorithm"""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True),
        "buffer_size": trial.suggest_int("buffer_size", 50000, 200000, step=50000),
        "batch_size": trial.suggest_int("batch_size", 64, 512, step=64),
        "gamma": trial.suggest_float("gamma", 0.98, 0.999),
        "tau": trial.suggest_float("tau", 0.001, 0.01),
        "train_freq": trial.suggest_int("train_freq", 1, 10),
        "gradient_steps": trial.suggest_int("gradient_steps", 1, 10),
        "learning_starts": trial.suggest_int("learning_starts", 500, 2000),
        "policy": "MlpPolicy",
        "policy_kwargs": {
            "net_arch": {
                "pi": [trial.suggest_int("pi_h1", 32, 256), trial.suggest_int("pi_h2", 32, 256)],
                "qf": [trial.suggest_int("qf_h1", 32, 256), trial.suggest_int("qf_h2", 32, 256)]
            },
            "activation_fn": "torch.nn.Tanh"
        }
    }

HYPERPARAMS_SAMPLER = {
    "ppo": sample_ppo_params,
    "a2c": sample_a2c_params,
    "sac": sample_sac_params,
    "td3": sample_td3_params
}

def optimize_agent(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """Optimization target for Optuna"""
    
    # Sample hyperparameters
    hyperparams = HYPERPARAMS_SAMPLER[args.algo](trial)
    
    # Process activation function - convert string to actual torch class
    if "policy_kwargs" in hyperparams and "activation_fn" in hyperparams["policy_kwargs"]:
        if hyperparams["policy_kwargs"]["activation_fn"] == "torch.nn.Tanh":
            hyperparams["policy_kwargs"]["activation_fn"] = torch.nn.Tanh
    
    # Setup learning rate schedule
    if "learning_rate" in hyperparams:
        schedule_fn = get_schedule_fn(hyperparams["learning_rate"])
        hyperparams["learning_rate"] = schedule_fn
    
    # Create log dir
    log_path = os.path.join("logs", "pcs_zoo_optimize", args.algo, f"trial_{trial.number}")
    os.makedirs(log_path, exist_ok=True)
    
    # Create and wrap the environment
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
    
    env = Monitor(
        env,
        os.path.join(log_path, 'train_monitor'),
        info_keywords=("battery_level", "net_exchange", "predicted_demand", "realized_demand")
    )
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=1.0,
        clip_reward=1.0,
        gamma=hyperparams.get('gamma', 0.99),
        epsilon=1e-8
    )

    # Create evaluation environment
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
        os.path.join(log_path, 'eval_monitor'),
        info_keywords=("battery_level", "net_exchange", "predicted_demand", "realized_demand")
    )
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=1.0,
        clip_reward=1.0,
        gamma=hyperparams.get('gamma', 0.99),
        epsilon=1e-8
    )

    # Create the callback for evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_path,
        log_path=log_path,
        eval_freq=min(2000, args.n_timesteps // 10),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False
    )

    # Create custom action tracking callback
    action_tracker = ActionTrackingCallback("PCS")

    # Combine all callbacks
    callbacks = CallbackList([
        eval_callback,
        action_tracker
    ])

    try:
        # Create and train the model
        model = ALGOS[args.algo](
            env=env,
            tensorboard_log=log_path,
            verbose=0,
            **hyperparams
        )
        
        model.learn(
            total_timesteps=args.n_timesteps,
            callback=callbacks
        )

        # Get mean reward from the best evaluation
        mean_reward = eval_callback.best_mean_reward
        
        # Save hyperparameters and results
        with open(os.path.join(log_path, "hyperparameters.yml"), "w") as f:
            yaml.dump({
                "hyperparameters": {
                    "policy": "MlpPolicy",
                    **{k: v for k, v in hyperparams.items() if k != "policy"}
                },
                "mean_reward": float(mean_reward)
            }, f)
        
        return mean_reward

    except Exception as e:
        print(f"Trial failed: {e}")
        return float("-inf")

def main():
    args = parse_args()
    
    # Create or load the study
    study_name = args.study_name or f"pcs_optimize_{args.algo}"
    storage = args.storage or f"sqlite:///logs/pcs_zoo_optimize/{study_name}.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize"
    )
    
    # Run optimization
    try:
        study.optimize(
            lambda trial: optimize_agent(trial, args),
            n_trials=args.n_trials,
            show_progress_bar=True
        )
        
        # Print results
        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
            
        # Save best hyperparameters
        best_params_dir = os.path.join("hyperparams", "optimized")
        os.makedirs(best_params_dir, exist_ok=True)
        
        best_params = {
            args.env: {
                "n_steps": study.best_trial.params["n_steps"],
                "batch_size": study.best_trial.params["batch_size"],
                "gamma": study.best_trial.params["gamma"],
                "learning_rate": study.best_trial.params["learning_rate"],
                "ent_coef": study.best_trial.params["ent_coef"],
                "clip_range": study.best_trial.params["clip_range"],
                "n_epochs": study.best_trial.params["n_epochs"],
                "gae_lambda": study.best_trial.params["gae_lambda"],
                "max_grad_norm": study.best_trial.params["max_grad_norm"],
                "vf_coef": study.best_trial.params["vf_coef"],
                "n_envs": 1,
                "n_timesteps": args.n_timesteps,
                "policy": "MlpPolicy",
                "normalize": True,
                "policy_kwargs": dict(
                    net_arch=dict(
                        pi=[study.best_trial.params["pi_h1"], study.best_trial.params["pi_h2"]],
                        vf=[study.best_trial.params["vf_h1"], study.best_trial.params["vf_h2"]]
                    ),
                    activation_fn="torch.nn.Tanh"
                ),
                "normalize_advantage": True
            }
        }
        
        with open(os.path.join(best_params_dir, f"{args.algo}_best.yml"), "w") as f:
            yaml.dump(best_params, f)
            
    except KeyboardInterrupt:
        print("\nOptimization interrupted. Saving current best results...")
        
if __name__ == "__main__":
    main()