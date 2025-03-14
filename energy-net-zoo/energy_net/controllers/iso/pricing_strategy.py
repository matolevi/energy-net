from typing import Dict, Any, Union, Tuple, List, Optional
import numpy as np
import logging
from abc import ABC, abstractmethod
from energy_net.env import PricingPolicy
from gymnasium import spaces
from energy_net.market.iso.quadratic_pricing_iso import QuadraticPricingISO

class PricingStrategy(ABC):
    """
    Base strategy interface for pricing policies.
    
    This abstract class defines the interface for all pricing strategies.
    Each concrete strategy handles a specific pricing policy (Quadratic, Online, Constant).
    """
    
    def __init__(
        self, 
        min_price: float, 
        max_price: float,
        max_steps_per_episode: int,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the base pricing strategy.
        
        Args:
            min_price: Minimum price boundary
            max_price: Maximum price boundary
            max_steps_per_episode: Maximum number of steps per episode
            logger: Logger instance for logging
        """
        self.min_price = min_price
        self.max_price = max_price
        self.max_steps_per_episode = max_steps_per_episode
        self.logger = logger
    
    @abstractmethod
    def create_action_space(self) -> spaces.Space:
        """
        Create the appropriate action space for this pricing strategy.
        
        Returns:
            A gymnasium Space object representing the action space
        """
        pass
    
    @abstractmethod
    def process_action(
        self, 
        action: Union[float, np.ndarray, int], 
        step_count: int,
        first_action_taken: bool
    ) -> Tuple[float, float, np.ndarray, bool]:
        """
        Process the agent's action according to the pricing strategy.
        
        Args:
            action: The action taken by the agent
            step_count: The current step count in the episode
            first_action_taken: Whether the first action has been taken
            
        Returns:
            Tuple containing:
            - buy_price: Current buying price
            - sell_price: Current selling price
            - dispatch: Current dispatch value
            - first_action_taken: Updated first_action_taken flag
        """
        pass


class QuadraticPricingStrategy(PricingStrategy):
    """
    Strategy for the Quadratic pricing policy.
    This strategy uses polynomial coefficients to determine prices.
    """
    
    def __init__(
        self, 
        min_price: float, 
        max_price: float,
        max_steps_per_episode: int,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the quadratic pricing strategy.
        
        Args:
            min_price: Minimum price boundary
            max_price: Maximum price boundary
            max_steps_per_episode: Maximum number of steps per episode
            config: Configuration for the quadratic pricing policy
            logger: Logger instance for logging
        """
        super().__init__(min_price, max_price, max_steps_per_episode, logger)
        
        policy_config = config.get('quadratic', {})
        dispatch_config = policy_config.get('dispatch', {})
        poly_config = policy_config.get('polynomial', {})
        
        self.dispatch_min = dispatch_config.get('min', 0.0)
        self.dispatch_max = dispatch_config.get('max', 300.0)
        self.low_poly = poly_config.get('min', -100.0)
        self.high_poly = poly_config.get('max', 100.0)
        
        # Initialize price coefficients and dispatch profile
        self.buy_coef = np.zeros(3, dtype=np.float32)   # [b0, b1, b2]
        self.sell_coef = np.zeros(3, dtype=np.float32)  # [s0, s1, s2]
        self.dispatch_profile = np.zeros(max_steps_per_episode, dtype=np.float32)
        
        # Initialize ISO pricing objects
        self.buy_iso = None
        self.sell_iso = None
    
    def create_action_space(self) -> spaces.Space:
        """
        Create the action space for quadratic pricing.
        
        Returns:
            A Box space with dimensions for polynomial coefficients and dispatch profile
        """
        low_array = np.concatenate((
            np.full(6, self.low_poly, dtype=np.float32),
            np.full(self.max_steps_per_episode, self.dispatch_min, dtype=np.float32)
        ))
        high_array = np.concatenate((
            np.full(6, self.high_poly, dtype=np.float32),
            np.full(self.max_steps_per_episode, self.dispatch_max, dtype=np.float32)
        ))
                
        return spaces.Box(
            low=low_array,
            high=high_array,
            dtype=np.float32
        )
    
    def process_action(
        self, 
        action: Union[float, np.ndarray, int], 
        step_count: int,
        first_action_taken: bool,
        predicted_demand: float = 0.0
    ) -> Tuple[float, float, float, bool]:
        """
        Process the agent's action according to the quadratic pricing strategy.
        
        Args:
            action: The action taken by the agent (polynomial coefficients + dispatch profile)
            step_count: The current step count in the episode
            first_action_taken: Whether the first action has been taken
            predicted_demand: The predicted demand for the current time step
            
        Returns:
            Tuple containing:
            - buy_price: Current buying price
            - sell_price: Current selling price
            - dispatch: Current dispatch value
            - first_action_taken: Updated first_action_taken flag
        """
        iso_buy_price = 0.0
        iso_sell_price = 0.0
        dispatch = 0.0
        
        if step_count == 1 and not first_action_taken:
            action = np.array(action).flatten()
            expected_length = 6 + self.max_steps_per_episode
            
            if len(action) != expected_length: 
                if self.logger:
                    self.logger.error(
                        f"Expected action of length {expected_length}, "
                        f"got {len(action)}"
                    )
                raise ValueError(
                    f"Expected action of length {expected_length}, "
                    f"got {len(action)}"
                )
            
            self.buy_coef = action[0:3]    # [b0, b1, b2] 
            self.sell_coef = action[3:6]   # [s0, s1, s2] 
            self.dispatch_profile = action[6:]  # values for dispatch profile
            self.dispatch_profile = np.clip(self.dispatch_profile, self.dispatch_min, self.dispatch_max)
        

            self.buy_iso = QuadraticPricingISO(
                buy_a=float(self.buy_coef[0]),
                buy_b=float(self.buy_coef[1]), 
                buy_c=float(self.buy_coef[2])
            )
            self.sell_iso = QuadraticPricingISO(
                buy_a=float(self.sell_coef[0]),
                buy_b=float(self.sell_coef[1]),
                buy_c=float(self.sell_coef[2])
            )

            first_action_taken = True
            if self.logger:
                self.logger.info(
                    f"Day-ahead polynomial for BUY: {self.buy_coef}, "
                    f"SELL: {self.sell_coef}, "
                    f"Dispatch profile: {self.dispatch_profile}"
                )
        else:
            if self.logger:
                self.logger.debug("Ignoring action - day-ahead polynomial & dispatch are already set.")
        
        buy_pricing_fn = self.buy_iso.get_pricing_function({'demand': predicted_demand})
        iso_buy_price = max(buy_pricing_fn(1.0), 0)

        sell_pricing_fn = self.sell_iso.get_pricing_function({'demand': predicted_demand})
        iso_sell_price = max(sell_pricing_fn(1.0), 0)
        
        if step_count > 0 and step_count <= len(self.dispatch_profile):
            dispatch = self.dispatch_profile[step_count - 1]
        
        if self.logger:
            self.logger.info(f"Step {step_count} - ISO Prices: Sell {iso_sell_price:.2f}, Buy {iso_buy_price:.2f}")
        
        return iso_buy_price, iso_sell_price, dispatch, first_action_taken


class ConstantPricingStrategy(PricingStrategy):
    """
    Strategy for the Constant pricing policy.
    This strategy uses constant prices for an entire episode.
    """
    
    def __init__(
        self, 
        min_price: float, 
        max_price: float,
        max_steps_per_episode: int,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the constant pricing strategy.
        
        Args:
            min_price: Minimum price boundary
            max_price: Maximum price boundary
            max_steps_per_episode: Maximum number of steps per episode
            config: Configuration for the constant pricing policy
            logger: Logger instance for logging
        """
        super().__init__(min_price, max_price, max_steps_per_episode, logger)
        
        policy_config = config.get('constant', {})
        dispatch_config = policy_config.get('dispatch', {})
        poly_config = policy_config.get('polynomial', {})
        
        self.dispatch_min = dispatch_config.get('min', 0.0)
        self.dispatch_max = dispatch_config.get('max', 300.0)
        self.low_const = poly_config.get('min', min_price)
        self.high_const = poly_config.get('max', max_price)
        
        # Initialize constant prices and dispatch profile
        self.const_buy = 0.0
        self.const_sell = 0.0
        self.dispatch_profile = np.zeros(max_steps_per_episode, dtype=np.float32)
        
        # Initialize ISO pricing objects
        self.buy_iso = None
        self.sell_iso = None
    
    def create_action_space(self) -> spaces.Space:
        """
        Create the action space for constant pricing.
        
        Returns:
            A Box space with dimensions for constant buy/sell prices and dispatch profile
        """
        low_array = np.concatenate((
            np.array([self.min_price, self.min_price], dtype=np.float32),
            np.full(self.max_steps_per_episode, self.dispatch_min, dtype=np.float32)
        ))
        high_array = np.concatenate((
            np.array([self.max_price, self.max_price], dtype=np.float32),
            np.full(self.max_steps_per_episode, self.dispatch_max, dtype=np.float32)
        ))
        
        return spaces.Box(
            low=low_array,
            high=high_array,
            dtype=np.float32
        )
    
    def process_action(
        self, 
        action: Union[float, np.ndarray, int], 
        step_count: int,
        first_action_taken: bool,
        predicted_demand: float = 0.0
    ) -> Tuple[float, float, float, bool]:
        """
        Process the agent's action according to the constant pricing strategy.
        
        Args:
            action: The action taken by the agent (constant buy/sell prices + dispatch profile)
            step_count: The current step count in the episode
            first_action_taken: Whether the first action has been taken
            predicted_demand: The predicted demand for the current time step
            
        Returns:
            Tuple containing:
            - buy_price: Current buying price
            - sell_price: Current selling price
            - dispatch: Current dispatch value
            - first_action_taken: Updated first_action_taken flag
        """
        iso_buy_price = 0.0
        iso_sell_price = 0.0
        dispatch = 0.0
        
        if step_count == 1 and not first_action_taken:
            action = np.array(action).flatten()
            expected_length = 2 + self.max_steps_per_episode
            
            if len(action) != expected_length:
                if self.logger:
                    self.logger.error(
                        f"Expected action of length {expected_length}, got {len(action)}"
                    )
                raise ValueError(
                    f"Expected action of length {expected_length}, got {len(action)}"
                )
                
            self.const_buy = float(action[0])
            self.const_sell = float(action[1])
            self.dispatch_profile = action[2:]
            self.dispatch_profile = np.clip(self.dispatch_profile, self.dispatch_min, self.dispatch_max)
            
            self.buy_iso = QuadraticPricingISO(
                buy_a=0.0,
                buy_b=0.0,
                buy_c=self.const_buy
            )
            self.sell_iso = QuadraticPricingISO(
                buy_a=0.0,
                buy_b=0.0,
                buy_c=self.const_sell
            )   
            
            first_action_taken = True
            if self.logger:
                self.logger.info(
                    f"Day-ahead constant prices - BUY: {self.const_buy}, "
                    f"SELL: {self.const_sell}, "
                    f"Dispatch profile: {self.dispatch_profile}"
                )
        else:
            if self.logger:
                self.logger.debug("Ignoring action - day-ahead constant pricing & dispatch already set.")
        
        buy_pricing_fn = self.buy_iso.get_pricing_function({'demand': predicted_demand}) if self.buy_iso else lambda x: 0
        iso_buy_price = buy_pricing_fn(1.0)

        sell_pricing_fn = self.sell_iso.get_pricing_function({'demand': predicted_demand}) if self.sell_iso else lambda x: 0
        iso_sell_price = sell_pricing_fn(1.0)
        
        if step_count > 0 and step_count <= len(self.dispatch_profile):
            dispatch = self.dispatch_profile[step_count - 1]
        
        if self.logger:
            self.logger.info(f"Step {step_count} - ISO Prices: Sell {iso_sell_price:.2f}, Buy {iso_buy_price:.2f}")
        
        return iso_buy_price, iso_sell_price, dispatch, first_action_taken


class OnlinePricingStrategy(PricingStrategy):
    """
    Strategy for the Online pricing policy.
    This strategy updates prices at each time step.
    """
    
    def __init__(
        self, 
        min_price: float, 
        max_price: float,
        max_steps_per_episode: int,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the online pricing strategy.
        
        Args:
            min_price: Minimum price boundary
            max_price: Maximum price boundary
            max_steps_per_episode: Maximum number of steps per episode
            config: Configuration for the online pricing policy
            logger: Logger instance for logging
        """
        super().__init__(min_price, max_price, max_steps_per_episode, logger)
        
        # Extract specific bounds from the config
        online_config = config.get('online', {})
        self.buy_price_min = online_config.get('buy_price', {}).get('min', min_price)
        self.buy_price_max = online_config.get('buy_price', {}).get('max', max_price)
        self.sell_price_min = online_config.get('sell_price', {}).get('min', min_price)
        self.sell_price_max = online_config.get('sell_price', {}).get('max', max_price)
        self.dispatch_min = online_config.get('dispatch', {}).get('min', 0.0)
        self.dispatch_max = online_config.get('dispatch', {}).get('max', 300.0)
        
        if self.logger:
            self.logger.info(
                f"Initialized OnlinePricingStrategy with bounds: "
                f"Buy Price [{self.buy_price_min}, {self.buy_price_max}], "
                f"Sell Price [{self.sell_price_min}, {self.sell_price_max}]"
            )
    
    def create_action_space(self) -> spaces.Space:
        """
        Create the action space for online pricing.
        
        Returns:
            A Box space with dimensions for buy/sell prices
        """
        return spaces.Box(
            low=np.array([self.buy_price_min, self.sell_price_min], dtype=np.float32),
            high=np.array([self.buy_price_max, self.sell_price_max], dtype=np.float32),
            dtype=np.float32
        )
    
    def process_action(
        self, 
        action: Union[float, np.ndarray, int], 
        step_count: int,
        first_action_taken: bool,
        predicted_demand: float = 0.0
    ) -> Tuple[float, float, float, bool]:
        """
        Process the agent's action according to the online pricing strategy.
        
        Args:
            action: The action taken by the agent (buy/sell prices)
            step_count: The current step count in the episode
            first_action_taken: Whether the first action has been taken
            predicted_demand: The predicted demand for the current time step
            
        Returns:
            Tuple containing:
            - buy_price: Current buying price
            - sell_price: Current selling price
            - dispatch: Current dispatch value (set to predicted_demand for online pricing)
            - first_action_taken: Updated first_action_taken flag
        """
        if self.logger:
            self.logger.info(f"Processing ISO action: {action}")
            
        dispatch = predicted_demand
        
        if isinstance(action, np.ndarray):
            action = action.flatten()
        else:
            if self.logger:
                self.logger.info(f"Converting scalar action to array: {action}")
            action = np.array([action, action])
            
        # Ensure the action is within bounds - we do this explicitly to guarantee bounds are respected
        action = np.clip(
            action,
            np.array([self.buy_price_min, self.sell_price_min]),
            np.array([self.buy_price_max, self.sell_price_max])
        )
        
        if self.logger:
            self.logger.info(f"Clipped action: {action}")
            
        iso_buy_price, iso_sell_price = action
        
        # Ensure prices are properly constrained
        iso_buy_price = float(np.clip(iso_buy_price, self.buy_price_min, self.buy_price_max))
        iso_sell_price = float(np.clip(iso_sell_price, self.sell_price_min, self.sell_price_max))
        
        if self.logger:
            self.logger.info(
                f"Step {step_count} - ISO Prices: "
                f"Buy {iso_buy_price:.2f} [{self.buy_price_min}-{self.buy_price_max}], "
                f"Sell {iso_sell_price:.2f} [{self.sell_price_min}-{self.sell_price_max}]"
            )
        
        return iso_buy_price, iso_sell_price, dispatch, first_action_taken


class PricingStrategyFactory:
    """
    Factory class for creating pricing strategy instances.
    """
    
    @staticmethod
    def create_strategy(
        pricing_policy: PricingPolicy,
        min_price: float,
        max_price: float,
        max_steps_per_episode: int,
        action_spaces_config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ) -> PricingStrategy:
        """
        Create the appropriate pricing strategy based on the pricing policy.
        
        Args:
            pricing_policy: The pricing policy enum value
            min_price: Minimum price boundary
            max_price: Maximum price boundary
            max_steps_per_episode: Maximum number of steps per episode
            action_spaces_config: Configuration for action spaces
            logger: Logger instance for logging
            
        Returns:
            An instance of the appropriate pricing strategy
            
        Raises:
            ValueError: If the pricing policy is not supported
        """
        if pricing_policy == PricingPolicy.QUADRATIC:
            return QuadraticPricingStrategy(
                min_price, 
                max_price, 
                max_steps_per_episode, 
                action_spaces_config,
                logger
            )
        elif pricing_policy == PricingPolicy.CONSTANT:
            return ConstantPricingStrategy(
                min_price, 
                max_price, 
                max_steps_per_episode, 
                action_spaces_config,
                logger
            )
        elif pricing_policy == PricingPolicy.ONLINE:
            return OnlinePricingStrategy(
                min_price, 
                max_price, 
                max_steps_per_episode, 
                action_spaces_config,
                logger
            )
        else:
            if logger:
                logger.error(f"Unsupported pricing policy: {pricing_policy}")
            raise ValueError(f"Unsupported pricing policy: {pricing_policy}") 