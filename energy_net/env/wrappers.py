"""
Environment wrappers for alternating training of ISO and PCS agents.

These wrappers convert the multi-agent EnergyNetV0 environment into
single-agent environments suitable for training with RL Zoo. They handle
the sequential nature of the ISO-PCS interaction and maintain compatibility
with standard RL algorithms.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union
from stable_baselines3.common.vec_env import VecEnv

from energy_net.env.energy_net_v0 import EnergyNetV0


class ISOEnvWrapper(gym.Wrapper):
    """
    Environment wrapper for ISO agent with fixed PCS policy.
    
    This wrapper converts the multi-agent EnergyNetV0 environment into a
    single-agent environment for training the ISO agent. It uses a fixed
    PCS policy to generate actions for the PCS agent.
    
    The wrapper ensures that the ISO agent receives properly formatted
    observations and rewards, and that the environment steps occur in the
    correct sequential order (ISO first, then PCS).
    """
    
    def __init__(self, env, pcs_policy=None):
        """
        Initialize the ISO environment wrapper.
        
        Args:
            env: The EnergyNetV0 environment to wrap
            pcs_policy: Optional fixed policy for the PCS agent
        """
        super().__init__(env)
        self.pcs_policy = pcs_policy
        
        # Use only ISO observation and action spaces
        self.observation_space = env.observation_space["iso"]
        self.action_space = env.action_space["iso"]
        
        # Store last observed state for PCS policy
        self.last_pcs_obs = None
        self.last_iso_action = None
        
    def reset(self, **kwargs):
        """
        Reset the environment and return the initial ISO observation.
        
        Returns:
            Initial observation for the ISO agent
            Info dictionary
        """
        obs_dict, info = self.env.reset(**kwargs)
        
        # Store PCS observation for future use
        self.last_pcs_obs = obs_dict["pcs"]
        
        # Reset last ISO action
        self.last_iso_action = None
        
        return obs_dict["iso"], info
    
    def step(self, action):
        """
        Execute ISO action and automatically handle PCS action.
        
        This method:
        1. Stores the ISO action
        2. Processes the ISO action to update prices
        3. Gets the updated PCS observation with new prices
        4. Gets PCS action from the fixed policy
        5. Steps the environment with both actions
        6. Returns ISO-specific results
        
        Args:
            action: Action from the ISO agent
            
        Returns:
            ISO observation, reward, terminated flag, truncated flag, info dict
        """
        # Store ISO action
        self.last_iso_action = action
        
        # Access controller and process ISO action if possible
        # This updates prices before PCS observes them
        if hasattr(self.unwrapped, "controller"):
            controller = self.unwrapped.controller
            
            # Set the ISO prices based on the action
            controller._process_iso_action(action)
            
            # Get updated PCS observation with new prices
            pcs_obs = controller._get_pcs_observation()
            self.last_pcs_obs = pcs_obs
        
        # Get PCS action from policy or use default action
        if self.pcs_policy is not None:
            # Convert to batch format for policy prediction
            pcs_obs_batch = np.array([self.last_pcs_obs])
            
            # Get action from policy
            pcs_action, _ = self.pcs_policy.predict(
                pcs_obs_batch, 
                deterministic=True
            )
            
            # Extract from batch
            pcs_action = pcs_action[0]
        else:
            # Default action (neutral battery action)
            pcs_action = np.zeros(self.unwrapped.action_space["pcs"].shape)
        
        # Create joint action dict - ISO must go first!
        action_dict = {
            "iso": action,
            "pcs": pcs_action
        }
        
        # Step the environment
        obs_dict, rewards, terminations, truncations, info = self.env.step(action_dict)
        
        # Store updated PCS observation
        self.last_pcs_obs = obs_dict["pcs"]
        
        # Return only ISO related outputs
        return (
            obs_dict["iso"],
            rewards["iso"],
            terminations["iso"],
            truncations["iso"],
            info
        )


class PCSEnvWrapper(gym.Wrapper):
    """
    Environment wrapper for PCS agent with fixed ISO policy.
    
    This wrapper converts the multi-agent EnergyNetV0 environment into a
    single-agent environment for training the PCS agent. It uses a fixed
    ISO policy to generate actions for the ISO agent.
    
    The wrapper ensures that the PCS agent receives properly formatted
    observations and rewards, and that the environment steps occur in the
    correct sequential order (ISO first, then PCS).
    """
    
    def __init__(self, env, iso_policy=None):
        """
        Initialize the PCS environment wrapper.
        
        Args:
            env: The EnergyNetV0 environment to wrap
            iso_policy: Optional fixed policy for the ISO agent
        """
        super().__init__(env)
        self.iso_policy = iso_policy
        
        # Use only PCS observation and action spaces
        self.observation_space = env.observation_space["pcs"]
        self.action_space = env.action_space["pcs"]
        
        # Store last observed state for ISO policy
        self.last_iso_obs = None
        self.last_pcs_obs = None
        
    def reset(self, **kwargs):
        """
        Reset the environment and return the initial PCS observation.
        
        Returns:
            Initial observation for the PCS agent
            Info dictionary
        """
        obs_dict, info = self.env.reset(**kwargs)
        
        # Store observations for future use
        self.last_iso_obs = obs_dict["iso"]
        self.last_pcs_obs = obs_dict["pcs"]
        
        return obs_dict["pcs"], info
    
    def step(self, action):
        """
        Execute PCS action with prior ISO action from policy.
        
        This method:
        1. Gets ISO action from the fixed policy
        2. Creates an action dictionary with both actions
        3. Steps the environment with the action dictionary
        4. Returns PCS-specific results
        
        Args:
            action: Action from the PCS agent
            
        Returns:
            PCS observation, reward, terminated flag, truncated flag, info dict
        """
        # Get ISO action from policy or use default action
        if self.iso_policy is not None:
            # Convert to batch format for policy prediction
            iso_obs_batch = np.array([self.last_iso_obs])
            
            # Get action from policy
            iso_action, _ = self.iso_policy.predict(
                iso_obs_batch, 
                deterministic=True
            )
            
            # Extract from batch
            iso_action = iso_action[0]
        else:
            # Default action (mid-range price)
            iso_action = np.zeros(self.unwrapped.action_space["iso"].shape)
        
        # Process ISO action to set prices if possible
        if hasattr(self.unwrapped, "controller"):
            controller = self.unwrapped.controller
            controller._process_iso_action(iso_action)
        
        # Create joint action dict - ISO must go first!
        action_dict = {
            "iso": iso_action,
            "pcs": action
        }
        
        # Step the environment
        obs_dict, rewards, terminations, truncations, info = self.env.step(action_dict)
        
        # Store updated observations
        self.last_iso_obs = obs_dict["iso"]
        self.last_pcs_obs = obs_dict["pcs"]
        
        # Return only PCS related outputs
        return (
            obs_dict["pcs"],
            rewards["pcs"],
            terminations["pcs"],
            truncations["pcs"],
            info
        )


class DummyVecEnvCompatibilityWrapper(gym.Wrapper):
    """
    Wrapper to make a gym environment compatible with SB3's dummy vec env.
    
    This wrapper ensures that the environment can be properly vectorized
    with DummyVecEnv by implementing the necessary methods.
    """
    
    def __init__(self, env):
        """Initialize the compatibility wrapper."""
        super().__init__(env)
        self._max_episode_steps = getattr(env, "_max_episode_steps", 1000)
        
    def seed(self, seed=None):
        """Seed the environment's random number generator."""
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return None
        
    def get_attr(self, attr):
        """Get an attribute from the environment."""
        return getattr(self, attr)
    
    def env_method(self, method_name, *method_args, **method_kwargs):
        """Call a method of the environment."""
        method = getattr(self, method_name)
        return method(*method_args, **method_kwargs)


# Factory functions to create wrapped environments
def make_iso_env(base_env, pcs_policy=None):
    """
    Create an ISO-specific environment with a fixed PCS policy.
    
    Args:
        base_env: The base EnergyNetV0 environment
        pcs_policy: Fixed policy for the PCS agent
        
    Returns:
        Wrapped environment for ISO training
    """
    env = ISOEnvWrapper(base_env, pcs_policy)
    env = DummyVecEnvCompatibilityWrapper(env)
    return env


def make_pcs_env(base_env, iso_policy=None):
    """
    Create a PCS-specific environment with a fixed ISO policy.
    
    Args:
        base_env: The base EnergyNetV0 environment
        iso_policy: Fixed policy for the ISO agent
        
    Returns:
        Wrapped environment for PCS training
    """
    env = PCSEnvWrapper(base_env, iso_policy)
    env = DummyVecEnvCompatibilityWrapper(env)
    return env 