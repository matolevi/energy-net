import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging
import yaml
import os
from typing import Dict, Any, Tuple, Union, List, Optional
from stable_baselines3 import PPO

from energy_net.utils.logger import setup_logger
from energy_net.env import PricingPolicy
from energy_net.market.iso.cost_types import CostType, calculate_costs
from energy_net.market.iso.demand_patterns import calculate_demand
from energy_net.market.iso.quadratic_pricing_iso import QuadraticPricingISO
from energy_net.market.iso.iso_base import ISOBase
from energy_net.components.pcsunit import PCSUnit
from energy_net.rewards.iso_reward import ISOReward
from energy_net.rewards.base_reward import BaseReward
from energy_net.controllers.iso.metrics_handler import ISOMetricsHandler
from energy_net.controllers.iso.pcs_simulator import PCSSimulator
from energy_net.controllers.iso.pricing_strategy import PricingStrategyFactory, PricingStrategy


class ISOController:
    """
    Independent System Operator (ISO) Controller responsible for setting electricity prices.
    Can operate with a trained PPO model or other pricing mechanisms.
    
    Observation Space:
        [time, predicted_demand, pcs_demand]
        
    Action Space:
        [b0, b1, b2, s0, s1, s2, dispatch_profile] 
        - b0, b1, b2: Polynomial coefficients for buy pricing
        - s0, s1, s2: Polynomial coefficients for sell pricing
        - dispatch_profile: 48 values for dispatch profile
    """
    
    def __init__(
        self,
        num_pcs_agents: int = 1, 
        pricing_policy: PricingPolicy = None,  
        demand_pattern=None,
        cost_type=None, 
        render_mode: Optional[str] = None,
        env_config_path: Optional[str] = 'configs/environment_config.yaml',
        iso_config_path: Optional[str] = 'configs/iso_config.yaml',
        pcs_unit_config_path: Optional[str] = 'configs/pcs_unit_config.yaml',
        log_file: Optional[str] = 'logs/environments.log',
        reward_type: str = 'iso',
        model_path: Optional[str] = None,
        trained_pcs_model_path: Optional[str] = None,
    ):
        # Set up logger
        self.logger = setup_logger('ISOController', log_file)
        self.logger.info(f"Initializing ISO Controller with {pricing_policy.value} policy")
        
        self.pricing_policy = pricing_policy
        self.demand_pattern = demand_pattern  # Store it as instance variable
        self.logger.info(f"Using demand pattern: {demand_pattern.value}")
        self.cost_type = cost_type
        self.logger.info(f"Using cost type: {cost_type.value}")

        # Load configurations
        self.env_config = self.load_config(env_config_path)
        self.iso_config = self.load_config(iso_config_path)
        self.pcs_unit_config = self.load_config(pcs_unit_config_path)

        # Get costs from cost type
        self.reserve_price, self.dispatch_price = calculate_costs(
            cost_type,
            self.env_config
        )

        # Define observation and action spaces
        obs_config = self.iso_config.get('observation_space', {})
        time_config = obs_config.get('time', {})
        demand_config = obs_config.get('predicted_demand', {})
        pcs_config = obs_config.get('pcs_demand', {})

        # Convert 'inf' strings from yaml to numpy.inf
        def convert_inf(value):
            if value == 'inf':
                return np.inf
            elif value == '-inf':
                return -np.inf
            return value

        self.observation_space = spaces.Box(
            low=np.array([
                time_config.get('min', 0.0),
                demand_config.get('min', 0.0),
                convert_inf(pcs_config.get('min', -np.inf))
            ], dtype=np.float32),
            high=np.array([
                time_config.get('max', 1.0),
                convert_inf(demand_config.get('max', np.inf)),
                convert_inf(pcs_config.get('max', np.inf))
            ], dtype=np.float32),
            dtype=np.float32
        )
        
        self.logger.info(f"Observation space initialized with bounds: low={self.observation_space.low}, high={self.observation_space.high}")

        # Define price bounds from ISO config
        pricing_config = self.iso_config.get('pricing', {})
        price_params = pricing_config.get('parameters', {})
        self.min_price = price_params.get('min_price', pricing_config.get('default_sell_price', 1.0))
        self.max_price = price_params.get('max_price', pricing_config.get('default_buy_price', 10.0))
        self.max_steps_per_episode = self.env_config['time'].get('max_steps_per_episode', 48)
        self.logger.info(f"Price bounds set to: min={self.min_price}, max={self.max_price}")

        # Get action space parameters from config based on pricing policy
        action_spaces_config = self.iso_config.get('action_spaces', {})
        
        # Initialize pricing strategy based on policy
        self.pricing_strategy = PricingStrategyFactory.create_strategy(
            pricing_policy=self.pricing_policy,
            min_price=self.min_price,
            max_price=self.max_price,
            max_steps_per_episode=self.max_steps_per_episode,
            action_spaces_config=action_spaces_config,
            logger=self.logger
        )
        
        # Set action space based on pricing strategy
        self.action_space = self.pricing_strategy.create_action_space()
        
        # Initialize ISO simulation variables
        self.iso_buy_price = 0.0
        self.iso_sell_price = 0.0
        self.first_action_taken = False
        
        # Initialize the PCS simulator
        self.pcs_simulator = PCSSimulator(
            num_pcs_agents=num_pcs_agents,
            pcs_unit_config=self.pcs_unit_config,
            log_file=log_file,
            logger=self.logger
        )
        self.logger.info(f"Initialized PCS simulator with {num_pcs_agents} agents")
        
        # Initialize reward calculator
        self.reward = self.initialize_reward(reward_type)
        self.logger.info(f"Initialized reward calculator with type: {reward_type}")
        
        # Initialize metrics handler
        self.metrics_handler = ISOMetricsHandler(
            self.env_config,
            self.reward,
            self.logger
        )
        self.logger.info(f"Initialized metrics handler with reward type: {reward_type}")
        
        # Initialize state tracking
        self.current_time = 0.0
        self.count = 0
        self.init = False
        self.realized_demand = 0.0
        self.first_action_taken = False
        self.price_history = []
        self.avg_price = 0.0
        
        # These will be populated during step
        self.production = 0.0
        self.consumption = 0.0
        self.pcs_demand = 0.0
        self.iso_buy_price = 0.0
        self.iso_sell_price = 0.0
        self.dispatch_price = 0.0
        self.reserve_price = 0.0
        
        # Initialize time constants
        self.minutes_per_day = self.env_config['time']['minutes_per_day']
        self.time_step_duration = self.env_config['time']['step_duration']
        
        # Initialize PCSUnit component for backward compatibility
        self.PCSUnit = PCSUnit(self.pcs_unit_config, self.logger)
        self.logger.info("Initialized PCSUnit component for backward compatibility")
        
        # Initialize trained PCS agent
        self.trained_pcs_agent = None
        if trained_pcs_model_path:
            self.logger.info(f"Loading trained PCS agent from {trained_pcs_model_path}")
            try:
                self.set_trained_pcs_agent(0, trained_pcs_model_path)
                self.logger.info("Successfully loaded trained PCS agent")
            except Exception as e:
                self.logger.error(f"Failed to load trained PCS agent: {e}")

        self.demand_pattern = demand_pattern
        self.logger.info(f"Using demand pattern: {demand_pattern.value}")

    def load_config(self, config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path):
            self.logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            self.logger.debug(f"Loaded configuration from {config_path}")
        return config

    def build_observation(self) -> np.ndarray:
        return np.array([
            self.current_time,
            self.predicted_demand,
            self.pcs_demand,
        ], dtype=np.float32)

    def calculate_predicted_demand(self, time: float) -> float:
        """
        Calculate predicted demand using selected pattern
        """
        return calculate_demand(
            time=time,
            pattern=self.demand_pattern,
            config=self.env_config['predicted_demand']
        )

    def translate_to_pcs_observation(self) -> np.ndarray:
        """
        Converts current state to PCS observation format.
        
        Returns:
            np.ndarray: Observation array containing:
                - Current battery level
                - Time of day
                - Current production
                - Current consumption
        """
        # Delegate to the PCS simulator
        return self.pcs_simulator.translate_to_pcs_observation(self.current_time)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Resets the ISO controller state.
        """
        self.logger.info("Resetting ISO Controller environment.")

        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.logger.debug(f"Random number generator seeded with: {seed}")
        else:
            self.rng = np.random.default_rng()
            self.logger.debug("Random number generator initialized without seed.")

        # Reset PCS simulator
        self.pcs_simulator.reset()
        self.logger.debug("PCS simulator has been reset.")

        # Reset PCSUnit for backward compatibility
        self.PCSUnit.reset()
        self.logger.debug("PCSUnit has been reset.")

        # Reset internal state
        let_energy = self.pcs_unit_config['battery']['model_parameters']
        self.avg_price = 0.0
        self.energy_lvl = let_energy['init']
        self.PCSUnit.reset(initial_battery_level=self.energy_lvl)
        self.reward_type = 0
        if options and 'reward' in options:
            if options.get('reward') == 1:
                self.reward_type = 1
                self.logger.debug("Reward type set to 1 based on options.")
            else:
                self.logger.debug(f"Reward type set to {self.reward_type} based on options.")
        else:
            self.logger.debug("No reward type option provided; using default.")

        self.count = 0
        self.first_action_taken = False
        self.terminated = False
        self.truncated = False
        self.init = True

        self.current_time = 0.0
        self.predicted_demand = self.calculate_predicted_demand(self.current_time)
        self.pcs_demand = 0.0

        if self.trained_pcs_agent is not None:
            try:
                pcs_obs = self.translate_to_pcs_observation()
                battery_action = self.simulate_pcs_response(pcs_obs)
                self.PCSUnit.update(time=self.current_time, battery_action=battery_action)
                self.production = self.PCSUnit.get_self_production()
                self.consumption = self.PCSUnit.get_self_consumption()
                self.pcs_demand = self.consumption - self.production
                self.logger.info(f"Updated PCS state on reset: battery_action={battery_action:.3f}, production={self.production:.3f}, consumption={self.consumption:.3f}")
            except Exception as e:
                self.logger.error(f"Failed to update PCS state on reset: {e}")
        else:
            self.logger.info("No trained PCS agent available on reset; using default PCS state.")

        self.logger.info(
            f"Environment Reset:\n"
            f"  Time: {self.current_time:.3f}\n"
            f"  Initial Demand: {self.predicted_demand:.2f} MWh\n"
            f"  PCS Demand: {self.pcs_demand:.2f} MWh\n"
            f"  ISO Sell Price: ${self.iso_sell_price:.2f}/MWh\n"
            f"  ISO Buy Price: ${self.iso_buy_price:.2f}/MWh"
        )

        observation = np.array([
            self.current_time,
            self.predicted_demand,
            self.pcs_demand
        ], dtype=np.float32)
        self.logger.debug(f"Initial observation: {observation}")

        info = {"status": "reset"}
        return observation, info

    def step(self, action: Union[float, np.ndarray, int]) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Agent action determining electricity pricing
            
        Returns:
            observation: Next state observation
            reward: Reward for this step
            done: Whether episode is finished
            truncated: Whether episode was truncated
            info: Additional information
        """
        # 1. Update time and environment state
        self.count += 1
        done = (self.count >= self.max_steps_per_episode)
        truncated = False  # No early termination
        
        # Use the original time calculation formula from the old code
        self.current_time = (self.count * self.time_step_duration) / self.env_config['time']['minutes_per_day']
        self.predicted_demand = self.calculate_predicted_demand(self.current_time)
        
        self.logger.debug(
            f"Step {self.count}:\n"
            f"  Time: {self.current_time:.3f}\n"
            f"  Predicted Demand: {self.predicted_demand:.2f} MWh"
        )

        # 2. Process ISO action using pricing strategy
        self.iso_buy_price, self.iso_sell_price, dispatch, self.first_action_taken = (
            self.pricing_strategy.process_action(
                action=action,
                step_count=self.count,
                first_action_taken=self.first_action_taken,
                predicted_demand=self.predicted_demand
            )
        )

        # 3. Get PCS responses using the PCS simulator
        pcs_result = self.pcs_simulator.simulate_response({
            'current_time': self.current_time,
            'iso_buy_price': self.iso_buy_price,
            'iso_sell_price': self.iso_sell_price
        })
        
        self.production = pcs_result['production']
        self.consumption = pcs_result['consumption']
        self.pcs_demand = pcs_result['pcs_demand']

        # 4. Calculate grid state, costs, and reward using the metrics handler
        metrics_result = self.metrics_handler.calculate_grid_state({
            'predicted_demand': self.predicted_demand,
            'pcs_demand': self.pcs_demand,
            'iso_buy_price': self.iso_buy_price,
            'iso_sell_price': self.iso_sell_price,
            'dispatch': dispatch,
            'count': self.count,
            'current_time': self.current_time,
            'production': self.production,
            'consumption': self.consumption,
            'battery_level': pcs_result['battery_levels'],
            'battery_actions': pcs_result['battery_actions']
        })
        
        reward = metrics_result['reward']
        info = metrics_result['info']
        self.realized_demand = metrics_result['realized_demand']
        self.dispatch_price = metrics_result['dispatch_price']
        self.reserve_price = metrics_result['reserve_price']
        
        # Calculate moving average of price
        self.price_history.append(self.iso_buy_price)
        self.avg_price = sum(self.price_history) / len(self.price_history)
        
        # 5. Build observation for next step
        observation = self.build_observation()

        # Log results
        self.logger.info(
            f"Step {self.count} Results:\n"
            f"  Predicted Demand: {self.predicted_demand:.2f} MWh\n"
            f"  Realized Demand: {self.realized_demand:.2f} MWh\n"
            f"  PCS Demand: {self.pcs_demand:.2f} MWh\n" 
            f"  Dispatch: {dispatch:.2f} MWh @ ${self.dispatch_price:.2f}/MWh\n"
            f"  Reserve: {max(0.0, self.realized_demand + self.pcs_demand - dispatch):.2f} MWh @ ${self.reserve_price:.2f}/MWh\n" 
            f"  Total Cost: ${(self.dispatch_price * dispatch + self.reserve_price * max(0.0, self.realized_demand + self.pcs_demand - dispatch)):.2f}"
        )

        return observation, float(reward), done, truncated, info

    def get_info(self) -> Dict[str, float]:
        return {"running_avg": self.avg_price}

    def close(self):
        self.logger.info("Closing ISO Controller environment.")
        for logger_name in []:
            logger = logging.getLogger(logger_name)
            handlers = logger.handlers[:]
            for handler in handlers:
                handler.close()
                logger.removeHandler(handler)
        self.logger.info("ISO Controller environment closed successfully.")

    def set_trained_pcs_agent(self, agent_idx: int, pcs_agent_path: str):
        """Set trained agent for specific PCS unit"""
        success = self.pcs_simulator.set_trained_agent(agent_idx, pcs_agent_path)
        return success

    def simulate_pcs_response(self, observation: np.ndarray) -> float:
        """
        Simulates the PCS unit's response to current market conditions.
        
        Args:
            observation (np.ndarray): Current state observation for PCS unit.
            
        Returns:
            float: Battery action (positive for charging, negative for discharging).
        """
        # Delegate to the PCS simulator
        return self.pcs_simulator.simulate_pcs_response(observation)

    def initialize_reward(self, reward_type: str) -> BaseReward:
        """
        Creates the appropriate reward function instance.
        
        Args:
            reward_type (str): Type of reward ('iso' or 'cost')
            
        Returns:
            BaseReward: Configured reward function
            
        Raises:
            ValueError: If reward_type is not supported
        """
        if reward_type in ['iso', 'cost']:
            self.logger.info(f"Initializing {reward_type} reward function")
            return ISOReward()
        else:
            self.logger.error(f"Unsupported reward type: {reward_type}")
            raise ValueError(f"Unsupported reward type: {reward_type}")
