from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import logging
from energy_net.components.pcsunit import PCSUnit

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
        pcsunit: Optional[PCSUnit] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the battery manager.
        
        Args:
            battery_config: Configuration for the battery including capacity and efficiency
            pcsunit: Reference to the PCSUnit instance to use for battery operations
            logger: Optional logger for tracking operations
        """
        self.logger = logger
        self.pcsunit = pcsunit
        
        # Extract battery parameters - use PCSUnit.battery if available, otherwise use config
        if self.pcsunit:
            self.battery_min = self.pcsunit.battery.energy_min
            self.battery_max = self.pcsunit.battery.energy_max
            self.charge_rate_max = self.pcsunit.battery.charge_rate_max
            self.discharge_rate_max = self.pcsunit.battery.discharge_rate_max
            self.charge_efficiency = self.pcsunit.battery.charge_efficiency
            self.discharge_efficiency = self.pcsunit.battery.discharge_efficiency
            self.battery_level = self.pcsunit.battery.energy_level
        else:
            # Fallback to config when PCSUnit is not available
            self.battery_min = battery_config.get('min', 0.0)
            self.battery_max = battery_config.get('max', 100.0)
            self.charge_rate_max = battery_config.get('charge_rate_max', 10.0)
            self.discharge_rate_max = battery_config.get('discharge_rate_max', 10.0)
            self.charge_efficiency = battery_config.get('charge_efficiency', 1.0)
            self.discharge_efficiency = battery_config.get('discharge_efficiency', 1.0)
            self.battery_level = battery_config.get('init', 0.0)
            
        self.lifetime_constant = battery_config.get('lifetime_constant', 100.0)
        
        # Initialize state tracking
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
        if self.pcsunit:
            # Get current state from PCSUnit
            current_level = self.pcsunit.battery.energy_level
            
            # Get constraints from PCSUnit battery
            if action > 0:  # Charging
                # Limit to maximum charge rate
                action = min(action, self.pcsunit.battery.charge_rate_max)
                
                # Apply charging efficiency
                energy_added = action * self.pcsunit.battery.charge_efficiency
                
                # Ensure we don't exceed battery capacity
                space_available = self.pcsunit.battery.energy_max - current_level
                energy_change = min(energy_added, space_available)
                
            elif action < 0:  # Discharging
                # Limit to maximum discharge rate and available energy
                max_discharge = min(abs(action), self.pcsunit.battery.discharge_rate_max)
                available_energy = current_level
                energy_removed = min(max_discharge, available_energy / self.pcsunit.battery.discharge_efficiency)
                
                # Convert to negative value for discharging
                energy_change = -energy_removed * self.pcsunit.battery.discharge_efficiency
                
            else:  # No action
                energy_change = 0.0
            
            # Calculate predicted new battery level
            new_battery_level = current_level + energy_change
            
            # Ensure battery level stays within bounds
            new_battery_level = max(self.pcsunit.battery.energy_min, min(new_battery_level, self.pcsunit.battery.energy_max))
            
            if self.logger:
                self.logger.debug(f"Battery action: {action:.2f}, Predicted energy change: {energy_change:.2f}, " 
                                f"Predicted new level: {new_battery_level:.2f}/{self.pcsunit.battery.energy_max:.2f} MWh")
                
            return energy_change, new_battery_level
        else:
            # Original implementation when not using PCSUnit
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
        if self.pcsunit:
            # Store previous level for tracking (use the exact value, not rounded)
            self.previous_level = self.pcsunit.battery.energy_level
            
            # IMPORTANT: Get the energy change directly from the PCSUnit battery
            # This ensures we get the correct value even if there's precision issues
            energy_change = self.pcsunit.battery.energy_change
            current_level = self.pcsunit.battery.energy_level
            
            # Update internal tracking
            self.battery_level = current_level
            self.energy_change = energy_change
            self.current_time_step += 1
            
            if self.logger:
                self.logger.info(f"Battery updated through PCSUnit: {self.previous_level:.5f} → {self.battery_level:.5f} MWh "
                                f"(Δ: {energy_change:.5f} MWh)")
                
            return energy_change
        else:
            # Original implementation when not using PCSUnit
            energy_change, new_level = self.calculate_energy_change(action)
            
            # Update internal state
            self.previous_level = self.battery_level
            self.battery_level = new_level
            self.energy_change = energy_change
            self.current_time_step += 1
            
            if self.logger:
                self.logger.info(f"Battery updated: {self.previous_level:.5f} → {self.battery_level:.5f} MWh "
                                f"(Δ: {energy_change:.5f} MWh)")
                
            return energy_change
    
    def validate_action(self, action: float) -> float:
        """
        Validate and constrain a proposed battery action based on current state.
        
        Args:
            action: Proposed battery action
            
        Returns:
            Validated action within allowable bounds
        """
        # Use a small epsilon to avoid floating point precision issues
        EPSILON = 1e-6
        
        if self.pcsunit:
            current_level = self.pcsunit.battery.energy_level
            
            # Check if battery is effectively at min level with epsilon tolerance
            if current_level <= self.pcsunit.battery.energy_min + EPSILON and action < 0:
                if self.logger:
                    self.logger.warning(
                        f"STRICTLY PREVENTED discharge action {action:.4f} at min battery level "
                        f"({current_level:.6f}/{self.pcsunit.battery.energy_min:.6f})"
                    )
                # CRITICAL FIX: Return exactly 0.0 instead of calculating any discharge amount
                return 0.0
            
            # Additional check for near-empty battery (within 1% of minimum)
            small_amount_threshold = (self.pcsunit.battery.energy_max - self.pcsunit.battery.energy_min) * 0.01
            if current_level <= self.pcsunit.battery.energy_min + small_amount_threshold and action < 0:
                # Scale down discharge action based on how close we are to the minimum
                ratio = (current_level - self.pcsunit.battery.energy_min) / small_amount_threshold
                scaled_action = action * ratio
                if self.logger:
                    self.logger.warning(
                        f"Near minimum: Scaled discharge from {action:.4f} to {scaled_action:.4f} "
                        f"(battery level: {current_level:.6f}, min: {self.pcsunit.battery.energy_min:.6f})"
                    )
                action = scaled_action
            
            # Check if battery is effectively at max level with epsilon tolerance
            if current_level >= self.pcsunit.battery.energy_max - EPSILON and action > 0:
                if self.logger:
                    self.logger.warning(
                        f"Prevented charge action {action:.2f} at max battery level "
                        f"({current_level:.2f}/{self.pcsunit.battery.energy_max:.2f})"
                    )
                # Calculate maximum possible charge (instead of returning 0)
                # At max level, this should be 0, but using formula for consistency
                available_space = (self.pcsunit.battery.energy_max - current_level) / self.pcsunit.battery.charge_efficiency
                return min(available_space, action)
            
            # Continue with original validation logic...
            if action > 0:  # Charging
                # Limit to maximum charge rate
                validated_action = min(action, self.pcsunit.battery.charge_rate_max)
                
                # Limit by available space
                max_charge = (self.pcsunit.battery.energy_max - current_level) / self.pcsunit.battery.charge_efficiency
                validated_action = min(validated_action, max_charge)
                
                if validated_action != action and self.logger:
                    self.logger.warning(f"Charge action {action:.2f} exceeds limits, limiting to {validated_action:.2f}")
                    
            elif action < 0:  # Discharging
                # Limit by available energy and max discharge rate
                available_energy = (current_level - self.pcsunit.battery.energy_min) * self.pcsunit.battery.discharge_efficiency
                max_discharge_rate = min(self.pcsunit.battery.discharge_rate_max, available_energy)
                validated_action = max(action, -max_discharge_rate)
                
                # Double-check that we're not discharging below minimum
                if current_level + validated_action * self.pcsunit.battery.discharge_efficiency < self.pcsunit.battery.energy_min:
                    if self.logger:
                        self.logger.error(f"Validation would allow discharge below minimum - forcing to 0.0")
                    validated_action = 0.0
                
                if validated_action != action and self.logger:
                    self.logger.warning(f"Discharge action {action:.2f} exceeds limits, limiting to {validated_action:.2f}")
                
            else:  # No action
                validated_action = 0.0
                
            return validated_action
        else:
            # For the standalone implementation without PCSUnit reference
            # Also apply the same strict check
            if self.battery_level <= self.battery_min + EPSILON and action < 0:
                if self.logger:
                    self.logger.warning(
                        f"STRICTLY PREVENTED discharge action {action:.4f} at min battery level "
                        f"({self.battery_level:.6f}/{self.battery_min:.6f})"
                    )
                # CRITICAL FIX: Return exactly 0.0 instead of calculating any discharge amount
                return 0.0
            
            if self.battery_level >= self.battery_max - EPSILON and action > 0:
                # Calculate max possible charge instead of returning 0
                available_space = (self.battery_max - self.battery_level) / self.charge_efficiency
                validated_action = min(available_space, action)
                
                if self.logger:
                    self.logger.warning(
                        f"Limited charge action {action:.2f} at max battery level "
                        f"({self.battery_level:.2f}/{self.battery_max:.2f}) to {validated_action:.2f}"
                    )
                return validated_action
            
            # Continue with original validation logic...
            if action > 0:  # Charging
                # Ensure we don't exceed maximum charge rate
                validated_action = min(action, self.charge_rate_max)
                
                # Ensure we don't exceed battery capacity
                max_charge = (self.battery_max - self.battery_level) / self.charge_efficiency
                validated_action = min(validated_action, max_charge)
                
                if validated_action != action and self.logger:
                    self.logger.warning(f"Charge action {action:.2f} exceeds limits, limiting to {validated_action:.2f}")
                    
            elif action < 0:  # Discharging
                # Ensure we don't exceed maximum discharge rate or available energy
                max_discharge_energy = (self.battery_level - self.battery_min) * self.discharge_efficiency
                max_discharge_rate = min(self.discharge_rate_max, max_discharge_energy)
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
        if self.pcsunit:
            current_level = self.pcsunit.battery.energy_level
            # Get energy change safely - try to use PCSUnit's function or calculate it
            try:
                # Try to get energy change directly from PCSUnit
                energy_change = self.pcsunit.get_energy_change()
            except (AttributeError, Exception):
                # Fall back to calculated change
                energy_change = current_level - self.previous_level
                
            return {
                'battery_level': current_level,
                'energy_change': energy_change,
                'available_capacity': self.pcsunit.battery.energy_max - current_level,
                'used_capacity_ratio': current_level / self.pcsunit.battery.energy_max if self.pcsunit.battery.energy_max > 0 else 0.0,
                'previous_level': self.previous_level
            }
        else:
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
        # Always use the most up-to-date value
        if self.pcsunit:
            return self.pcsunit.battery.energy_level
        else:
            return self.battery_level
    
    def reset(self, initial_level: Optional[float] = None) -> None:
        """
        Reset battery to initial or specified level.
        
        Args:
            initial_level: Optional level to reset to (uses default if None)
        """
        if self.pcsunit:
            # Handle reset using PCSUnit - already done in the PCSUnit.reset() call
            # Just update our internal tracking
            if initial_level is not None:
                self.previous_level = initial_level
            else:
                self.previous_level = self.pcsunit.battery.energy_level
                
            self.battery_level = self.pcsunit.battery.energy_level
            self.energy_change = 0.0
            self.current_time_step = 0
            
            if self.logger:
                self.logger.info(f"Battery tracking reset to {self.battery_level:.2f} MWh")
        else:
            # Original implementation when not using PCSUnit
            if initial_level is not None:
                self.battery_level = max(self.battery_min, min(initial_level, self.battery_max))
            else:
                self.battery_level = self.battery_min
                
            self.previous_level = self.battery_level
            self.energy_change = 0.0
            self.current_time_step = 0
            
            if self.logger:
                self.logger.info(f"Battery reset to {self.battery_level:.2f} MWh") 