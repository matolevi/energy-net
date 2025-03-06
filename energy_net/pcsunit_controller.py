from energy_net.components.grid_entity import GridEntity
from typing import Optional, Tuple, Dict, Any, Union, Callable
import numpy as np
import os
from stable_baselines3 import PPO

from gymnasium import spaces
import yaml
import logging

from energy_net.components.pcsunit import PCSUnit
from energy_net.dynamics.energy_dynamcis import EnergyDynamics
from energy_net.dynamics.energy_dynamcis import ModelBasedDynamics
from energy_net.dynamics.production_dynamics.deterministic_production import DeterministicProduction
from energy_net.dynamics.consumption_dynamics.deterministic_consumption import DeterministicConsumption
from energy_net.dynamics.storage_dynamics.deterministic_battery import DeterministicBattery
from energy_net.dynamics.energy_dynamcis import DataDrivenDynamics
from energy_net.utils.iso_factory import iso_factory
from energy_net.utils.logger import setup_logger  

from energy_net.market.iso.demand_patterns import DemandPattern, calculate_demand  
from energy_net.market.iso.cost_types import CostType, calculate_costs
from energy_net.market.iso.quadratic_pricing_iso import QuadraticPricingISO  

from energy_net.rewards.base_reward import BaseReward
from energy_net.rewards.cost_reward import CostReward

# Import controller components
from energy_net.controllers.pcs.metrics_handler import PCSMetricsHandler
from energy_net.controllers.pcs.battery_manager import BatteryManager
from energy_net.controllers.pcs.market_interface import MarketInterface


class PCSUnitController:
    """
    Power Consumption & Storage Unit Controller
    
    Manages a PCS unit's interaction with the power grid by controlling:
    1. Battery charging/discharging
    2. Energy production (optional)
    3. Energy consumption (optional)
    
    The controller handles:
    - Battery state management
    - Price-based decision making
    - Energy exchange with grid
    - Production/consumption coordination
    
    Actions:
        Type: Box
            - If multi_action=False:
                Charging/Discharging Power: continuous scalar
            - If multi_action=True:
                [Charging/Discharging Power, Consumption Action, Production Action]

    Observation:
        Type: Box(4)
            Energy storage level (MWh): [0, ENERGY_MAX]
            Time (fraction of day): [0, 1]
            ISO Buy Price ($/MWh): [0, inf]
            ISO Sell Price ($/MWh): [0, inf]
    """

    def __init__(
        self,
        cost_type=None,            
        demand_pattern=None,          
        render_mode: Optional[str] = None,
        env_config_path: Optional[str] = 'configs/environment_config.yaml',
        iso_config_path: Optional[str] = 'configs/iso_config.yaml',
        pcs_unit_config_path: Optional[str] = 'configs/pcs_unit_config.yaml',
        log_file: Optional[str] = 'logs/environments.log',  
        reward_type: str = 'cost', 
        trained_iso_model_path: Optional[str] = None  
    ):
        """
        Constructs an instance of PCSunitEnv.

        Args:
            render_mode: Optional rendering mode.
            env_config_path: Path to the environment YAML configuration file.
            iso_config_path: Path to the ISO YAML configuration file.
            pcs_unit_config_path: Path to the PCSUnit YAML configuration file.
            log_file: Path to the log file for environment logging.
            reward_type: Type of reward function to use.
        """
        super().__init__()  # Initialize the parent class

        # Store new parameters
        self.cost_type = cost_type
        self.demand_pattern = demand_pattern
        
        # Set up logger
        self.logger = setup_logger('PCSUnitController', log_file)
        self.logger.info(f"Using demand pattern: {demand_pattern.value}")
        self.logger.info(f"Using cost type: {cost_type.value}")

        # Load configurations
        self.env_config: Dict[str, Any] = self.load_config(env_config_path)
        self.iso_config: Dict[str, Any] = self.load_config(iso_config_path)
        self.pcs_unit_config: Dict[str, Any] = self.load_config(pcs_unit_config_path)
        
        # Initialize PCSUnit with dynamics and configuration
        self.PCSUnit: PCSUnit = PCSUnit(
            config=self.pcs_unit_config,
            log_file=log_file
        )
        self.logger.info("Initialized PCSUnit with all components.")

        # Define observation and action spaces
        energy_config: Dict[str, Any] = self.pcs_unit_config['battery']['model_parameters']
        obs_config = self.pcs_unit_config.get('observation_space', {})

        # Get battery level bounds from battery config if specified
        battery_level_config = obs_config.get('battery_level', {})
        battery_min = energy_config['min'] if battery_level_config.get('min') == "from_battery_config" else battery_level_config.get('min', energy_config['min'])
        battery_max = energy_config['max'] if battery_level_config.get('max') == "from_battery_config" else battery_level_config.get('max', energy_config['max'])

        # Get other observation space bounds from config
        time_config = obs_config.get('time', {})
        buy_price_config = obs_config.get('iso_buy_price', {})
        sell_price_config = obs_config.get('iso_sell_price', {})

        self.observation_space: spaces.Box = spaces.Box(
            low=np.array([
                battery_min,
                time_config.get('min', 0.0),
                buy_price_config.get('min', 0.0),
                sell_price_config.get('min', 0.0)
            ], dtype=np.float32),
            high=np.array([
                battery_max,
                time_config.get('max', 1.0),
                buy_price_config.get('max', 100.0),
                sell_price_config.get('max', 100.0)
            ], dtype=np.float32),
            dtype=np.float32
        )
        self.logger.info(f"Defined observation space: low={self.observation_space.low}, high={self.observation_space.high}")

        # Define Action Space
        self.multi_action: bool = self.pcs_unit_config.get('action', {}).get('multi_action', False)
        self.production_action_enabled: bool = self.pcs_unit_config.get('action', {}).get('production_action', {}).get('enabled', False)
        self.consumption_action_enabled: bool = self.pcs_unit_config.get('action', {}).get('consumption_action', {}).get('enabled', False)

        self.action_space: spaces.Box = spaces.Box(
            low=np.array([
                -energy_config['discharge_rate_max']
            ], dtype=np.float32),
            high=np.array([
                energy_config['charge_rate_max']
            ], dtype=np.float32),
            shape=(1,),
            dtype=np.float32
        )
        self.logger.info(f"Defined action space: low={-energy_config['discharge_rate_max']}, high={energy_config['charge_rate_max']}")

        # Initialize state variables
        self.time = 0.0
        
        # Internal State
        self.init: bool = False
        self.rng = np.random.default_rng()
        self.count: int = 0        # Step counter
        self.terminated: bool = False
        self.truncated: bool = False

        # Initialize component modules
        
        # Initialize battery manager
        self.battery_manager = BatteryManager(
            battery_config=self.pcs_unit_config['battery']['model_parameters'],
            logger=self.logger
        )
        # Set initial battery level
        self.battery_level = self.battery_manager.get_level()
        self.logger.info(f"Initialized battery manager with level: {self.battery_level}")
        
        # Initialize market interface
        self.market_interface = MarketInterface(
            env_config=self.env_config,
            iso_config=self.iso_config,
            pcs_config=self.pcs_unit_config,
            logger=self.logger
        )
        self.logger.info("Initialized market interface")

        # Initialize timing parameters
        self.time_steps_per_day_ratio = self.env_config['time']['time_steps_per_day_ratio']
        self.time_step_duration = self.env_config['time']['step_duration']
        self.max_steps_per_episode = self.env_config['time']['max_steps_per_episode']

        # Initialize the Reward Function
        self.logger.info(f"Setting up reward function: {reward_type}")
        self.reward: BaseReward = self.initialize_reward(reward_type)
        
        # Initialize the Metrics Handler
        self.metrics_handler = PCSMetricsHandler(
            config=self.pcs_unit_config,
            reward_function=self.reward,
            logger=self.logger
        )
        self.logger.info("Initialized metrics handler")
                
        # Load trained ISO model if provided
        if trained_iso_model_path:
            try:
                trained_iso_agent = PPO.load(trained_iso_model_path)
                self.market_interface.set_trained_iso_agent(trained_iso_agent)
                self.logger.info(f"Loaded ISO model: {trained_iso_model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load ISO model: {e}")

        self.logger.info("PCSunitEnv initialization complete.")
                
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Loads a YAML configuration file.

        Args:
            config_path (str): Path to the YAML config file.

        Returns:
            Dict[str, Any]: Configuration parameters.

        Raises:
            FileNotFoundError: If the config file does not exist.
        """
        if not os.path.exists(config_path):
            self.logger.error(f"Configuration file not found at {config_path}")
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        with open(config_path, 'r') as file:
            config: Dict[str, Any] = yaml.safe_load(file)
            self.logger.debug(f"Loaded configuration from {config_path}: {config}")

        return config        

    def initialize_reward(self, reward_type: str) -> BaseReward:
        """
        Initializes the reward function based on the specified type.

        Args:
            reward_type (str): Type of reward ('cost').

        Returns:
            BaseReward: An instance of a reward class.
        
        Raises:
            ValueError: If an unsupported reward_type is provided.
        """
        if reward_type == 'cost':
            return CostReward()
        else:
            self.logger.error(f"Unsupported reward type: {reward_type}")
            raise ValueError(f"Unsupported reward type: {reward_type}")

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Resets the environment to an initial state.

        Args:
            seed: Optional seed for random number generator.
            options: Optional settings like reward type.

        Returns:
            Tuple containing the initial observation and info dictionary.
        """
        self.logger.info("Resetting environment.")

        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.logger.debug(f"Random number generator seeded with: {seed}")
        else:
            self.rng = np.random.default_rng()
            self.logger.debug("Random number generator initialized without seed.")

        # Reset battery manager
        self.battery_manager.reset()
        self.battery_level = self.battery_manager.get_level()
        self.logger.debug(f"Battery manager reset, level: {self.battery_level}")
        
        # Reset market interface
        self.market_interface.reset()
        self.logger.debug("Market interface reset")

        # Reset PCSUnit and ISO
        self.PCSUnit.reset()
        self.logger.debug("PCSUnit has been reset.")
        
        # Pass the initial battery level to PCSUnit
        self.PCSUnit.reset(initial_battery_level=self.battery_level) 

        # Reset reward type if specified in options
        if options and 'reward' in options:
            self.reward_type = options.get('reward', 0)
            self.logger.debug(f"Reward type set to {self.reward_type} based on options.")
        else:
            self.reward_type = 0
            self.logger.debug("Using default reward type.")

        # Reset step counter and state
        self.count = 0
        self.terminated = False
        self.truncated = False
        self.init = True

        # Initialize current time (fraction of day)
        self.time = 0.0
        time: float = (self.count * self.time_step_duration) / 1440  # 1440 minutes in a day
        self.logger.debug(f"Initial time set to {time} fraction of day.")

        # Calculate initial predicted demand
        predicted_demand = self.calculate_predicted_demand(self.time)
        
        # Update market with initial demand
        self.market_interface.update_market_prices(time, predicted_demand, 0.0)
        
        # Generate realized demand
        realized_demand = self.market_interface.update_realized_demand()

        # Update PCSUnit with current time and no action
        self.PCSUnit.update(time=time, battery_action=0.0)
        self.logger.debug("PCSUnit updated with initial time and no action.")

        # Fetch self-production and self-consumption
        production: float = self.PCSUnit.get_self_production()
        consumption: float = self.PCSUnit.get_self_consumption()
        self.logger.debug(f"Initial pcs-production: {production}, pcs-consumption: {consumption}")
        
        # Get market state
        market_state = self.market_interface.get_state()
        iso_buy_price = market_state['iso_buy_price']
        iso_sell_price = market_state['iso_sell_price']

        # Build state dictionary
        battery_state = self.battery_manager.get_state()
        self.state = {
            **battery_state,
            'time': time,
            'production': production,
            'consumption': consumption,
            **market_state  # Include all market state metrics
        }

        # Reset metrics handler
        self.metrics_handler.reset()

        observation: np.ndarray = np.array([
            self.battery_level,
            time,
            iso_buy_price,   
            iso_sell_price   
        ], dtype=np.float32)
        self.logger.debug(f"Initial observation: {observation}")

        # Build info using metrics handler
        info = self.metrics_handler.build_info_dict(self.state, 0.0)
        self.logger.debug(f"Initial info: {info}")

        return (observation, info)

    def step(self, action: Union[float, np.ndarray, int]) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Executes one time step of the PCS unit.
        
        Process flow:
        1. Update time and get current ISO prices
        2. Validate and process battery action
        3. Update PCS unit state
        4. Calculate costs and rewards
        
        Args:
            action: Battery charging/discharging power
                   Positive = charging, Negative = discharging
                   
        Returns:
            observation: Current state [battery_level, time, buy_price, sell_price]
            reward: Cost-based reward for this step
            done: Whether episode is complete
            truncated: Whether episode was truncated
            info: Additional metrics and state information
        """
        assert self.init, "Environment must be reset before stepping."
        
        # 1. Update time and state
        self.count += 1
        self.time = (self.count * self.time_step_duration) / self.env_config['time']['minutes_per_day']
        self.logger.debug(f"Time updated to {self.time:.3f} (day fraction)")

        # Calculate predicted demand for this timestep
        predicted_demand = self.calculate_predicted_demand(self.time)
        
        # Get PCS demand from previous step (or 0.0 if first step)
        pcs_demand = self.state.get('pcs_demand', 0.0)
        
        # 2. Update market prices based on time and demand
        self.market_interface.update_market_prices(self.time, predicted_demand, pcs_demand)
        
        # Update realized demand with noise
        realized_demand = self.market_interface.update_realized_demand()

        # 3. Process PCS action
        self.logger.debug(f"Processing PCS action: {action}")
        if isinstance(action, np.ndarray):
            if self.multi_action and action.shape != (3,):
                raise ValueError(f"Action array must have shape (3,) for multi-action mode")
            elif not self.multi_action and action.shape != (1,):
                raise ValueError(f"Action array must have shape (1,) for single-action mode")
            
            if not self.action_space.contains(action):
                self.logger.warning(f"Action {action} outside bounds, clipping to valid range")
                action = np.clip(action, self.action_space.low, self.action_space.high)
                
            if self.multi_action:
                battery_action, consumption_action, production_action = action
            else:
                battery_action = action.item()
                consumption_action = None
                production_action = None
                
        elif isinstance(action, float):
            if self.multi_action:
                raise TypeError("Expected array action for multi-action mode")
            battery_action = action
            consumption_action = None
            production_action = None
        else:
            raise TypeError(f"Invalid action type: {type(action)}")

        # Validate battery action using battery manager
        battery_action = self.battery_manager.validate_action(battery_action)
        
        if self.multi_action:
            self.PCSUnit.update(
                time=self.time,
                battery_action=battery_action,
                consumption_action=consumption_action,
                production_action=production_action
            )
        else:
            self.PCSUnit.update(
                time=self.time,
                battery_action=battery_action
            )

        # Update battery manager with the action
        energy_change = self.battery_manager.update(battery_action)
            
        # Get updated battery level from the manager
        self.battery_level = self.battery_manager.get_level()

        # Get updated production and consumption
        production = self.PCSUnit.get_self_production()
        consumption = self.PCSUnit.get_self_consumption()
        
        # 4. Calculate market position
        market_position = self.market_interface.calculate_market_position(
            production=production,
            consumption=consumption,
            energy_change=energy_change
        )
        
        # 5. Build complete state dictionary
        battery_state = self.battery_manager.get_state()
        self.state = {
            **battery_state,  # Include all battery state metrics
            'time': self.time,
            'current_time': self.time,
            'production': production,
            'consumption': consumption,
            'battery_action': battery_action,
            **market_position  # Include all market position metrics
        }

        self.logger.info(
            f"PCS State Step {self.count}:\n"
            f"  Time: {self.time:.3f}\n"
            f"  Battery Level: {self.battery_level:.2f} MWh\n"
            f"  Battery Action: {battery_action:.2f} MWh\n"
            f"  Energy Change: {energy_change:.2f} MWh\n"
            f"  Production: {production:.2f} MWh\n"
            f"  Consumption: {consumption:.2f} MWh\n"
            f"  Net Exchange: {market_position['net_exchange']:.2f} MWh"
        )

        self.logger.info(
            f"Financial Metrics:\n"
            f"  ISO Buy Price: ${market_position['iso_buy_price']:.2f}/MWh\n"
            f"  ISO Sell Price: ${market_position['iso_sell_price']:.2f}/MWh\n"
            f"  Revenue: ${market_position['revenue']:.2f}"
        )

        # 6. Compute reward using metrics handler
        reward = self.metrics_handler.calculate_reward(self.state)
        self.logger.info(f"Step reward: {reward:.2f}")

        # 7. Create next observation
        observation = np.array([
            self.battery_level,
            self.time,
            market_position['iso_buy_price'],
            market_position['iso_sell_price']
        ], dtype=np.float32)

        # Check if episode is done
        done = self.count >= self.max_steps_per_episode
        if done:
            self.logger.info("Episode complete")
            # Call metrics handler end episode
            self.metrics_handler.end_episode()
        
        # Build info dictionary with metrics handler
        info = self.metrics_handler.build_info_dict(self.state, reward)
        
        return observation, float(reward), done, False, info


    def get_info(self) -> Dict[str, float]:
        """
        Provides additional information about the environment's state.

        Returns:
            Dict[str, float]: Dictionary containing environment summary metrics.
        """
        return self.metrics_handler.get_metrics_summary()
 
    def close(self):
        """
        Cleanup method. Closes loggers and releases resources.
        """
        self.logger.info("Closing environment.")
        
        # Get final metrics summary
        metrics_summary = self.metrics_handler.get_metrics_summary()
        self.logger.info(f"Final metrics summary: {metrics_summary}")

        logger_names = ['PCSunitEnv', 'Battery', 'ProductionUnit', 'ConsumptionUnit', 'PCSUnit'] 
        for logger_name in logger_names:
            logger = logging.getLogger(logger_name)
            handlers = logger.handlers[:]
            for handler in handlers:
                handler.close()
                logger.removeHandler(handler)
        self.logger.info("Environment closed successfully.")
            
    def calculate_predicted_demand(self, time: float) -> float:
        """
        Calculate predicted demand using selected pattern
        """
        return calculate_demand(
            time=time,
            pattern=self.demand_pattern,
            config=self.env_config['predicted_demand']
        )

