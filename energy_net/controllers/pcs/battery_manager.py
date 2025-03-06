from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import logging

class BatteryManager:
    """
    Manages battery operations for the PCS controller.
    
    Responsibilities:
    1. Tracking battery state (charge level)
    2. Managing charge/discharge operations with constraints
    3. Enforcing battery capacity and rate limits
    4. Calculating energy changes and efficiency impacts
    """
    
    def __init__(
        self, 
        battery_config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the battery manager.
        
        Args:
            battery_config: Configuration for the battery including capacity and efficiency
            logger: Optional logger for tracking operations
        """
        self.logger = logger
        
        # Extract battery parameters from config
        self.battery_min = battery_config.get('min', 0.0)
        self.battery_max = battery_config.get('max', 100.0)
        self.charge_rate_max = battery_config.get('charge_rate_max', 10.0)
        self.discharge_rate_max = battery_config.get('discharge_rate_max', 10.0)
        self.charge_efficiency = battery_config.get('charge_efficiency', 1.0)
        self.discharge_efficiency = battery_config.get('discharge_efficiency', 1.0)
        self.lifetime_constant = battery_config.get('lifetime_constant', 100.0)
        
        # Initialize state
        self.battery_level = battery_config.get('init', 0.0)
        self.previous_level = self.battery_level
        self.energy_change = 0.0
        self.current_time_step = 0
        
        if self.logger:
            self.logger.info(f"Battery Manager initialized with capacity: [{self.battery_min}, {self.battery_max}] MWh")
    
    def calculate_energy_change(self, action: float) -> Tuple[float, float]:
        """
        Calculate energy change from a battery action.
        
        Args:
            action: Battery action (positive for charging, negative for discharging)
            
        Returns:
            Tuple of (energy_change, new_battery_level)
        """
        # Apply rate limits
        if action > 0:  # Charging
            # Limit to maximum charge rate
            action = min(action, self.charge_rate_max)
            
            # Apply charging efficiency
            energy_added = action * self.charge_efficiency
            
            # Ensure we don't exceed battery capacity
            space_available = self.battery_max - self.battery_level
            energy_change = min(energy_added, space_available)
            
        elif action < 0:  # Discharging
            # Limit to maximum discharge rate and available energy
            max_discharge = min(abs(action), self.discharge_rate_max)
            available_energy = self.battery_level
            energy_removed = min(max_discharge, available_energy / self.discharge_efficiency)
            
            # Convert to negative value for discharging
            energy_change = -energy_removed * self.discharge_efficiency
            
        else:  # No action
            energy_change = 0.0
        
        # Calculate new battery level
        new_battery_level = self.battery_level + energy_change
        
        # Ensure battery level stays within bounds
        new_battery_level = max(self.battery_min, min(new_battery_level, self.battery_max))
        
        if self.logger:
            self.logger.debug(f"Battery action: {action:.2f}, Energy change: {energy_change:.2f}, " 
                             f"New level: {new_battery_level:.2f}/{self.battery_max:.2f} MWh")
            
        return energy_change, new_battery_level
    
    def update(self, action: float) -> float:
        """
        Update battery state based on action.
        
        Args:
            action: Battery action (positive for charging, negative for discharging)
            
        Returns:
            Actual energy change after applying constraints
        """
        energy_change, new_level = self.calculate_energy_change(action)
        
        # Update internal state
        self.previous_level = self.battery_level
        self.battery_level = new_level
        self.energy_change = energy_change
        self.current_time_step += 1
        
        if self.logger:
            self.logger.info(f"Battery updated: {self.previous_level:.2f} → {self.battery_level:.2f} MWh "
                            f"(Δ: {energy_change:.2f} MWh)")
            
        return energy_change
    
    def validate_action(self, action: float) -> float:
        """
        Validate and constrain a proposed battery action based on current state.
        
        Args:
            action: Proposed battery action
            
        Returns:
            Validated action within allowable bounds
        """
        if action > 0:  # Charging
            # Ensure we don't exceed maximum charge rate
            validated_action = min(action, self.charge_rate_max)
            
            if validated_action != action and self.logger:
                self.logger.warning(f"Charge action {action:.2f} exceeds maximum rate, limiting to {validated_action:.2f}")
                
        elif action < 0:  # Discharging
            # Ensure we don't exceed maximum discharge rate or available energy
            max_discharge_rate = min(self.discharge_rate_max, self.battery_level)
            validated_action = max(action, -max_discharge_rate)
            
            if validated_action != action and self.logger:
                self.logger.warning(
                    f"Discharge action {action:.2f} exceeds available capacity "
                    f"(battery level: {self.battery_level:.2f}), limiting to {validated_action:.2f}"
                )
        else:
            validated_action = 0.0
            
        return validated_action
    
    def get_state(self) -> Dict[str, float]:
        """
        Get current battery state.
        
        Returns:
            Dictionary with battery state information
        """
        return {
            'battery_level': self.battery_level,
            'energy_change': self.energy_change,
            'available_capacity': self.battery_max - self.battery_level,
            'used_capacity_ratio': self.battery_level / self.battery_max if self.battery_max > 0 else 0.0,
            'previous_level': self.previous_level
        }
    
    def get_level(self) -> float:
        """
        Get current battery level.
        
        Returns:
            Current battery level in MWh
        """
        return self.battery_level
    
    def reset(self, initial_level: Optional[float] = None) -> None:
        """
        Reset battery to initial or specified level.
        
        Args:
            initial_level: Optional level to reset to (uses default if None)
        """
        if initial_level is not None:
            self.battery_level = max(self.battery_min, min(initial_level, self.battery_max))
        else:
            self.battery_level = self.battery_min
            
        self.previous_level = self.battery_level
        self.energy_change = 0.0
        self.current_time_step = 0
        
        if self.logger:
            self.logger.info(f"Battery reset to {self.battery_level:.2f} MWh") 