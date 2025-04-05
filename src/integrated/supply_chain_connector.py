"""
Connector class that integrates System Dynamics and Reinforcement Learning models
for Supply Chain Management.
"""

import numpy as np
from integrated.supply_chain_sd import SupplyChainSDModel
from integrated.supply_chain_rl import SupplyChainRL

class SupplyChainConnector:
    def __init__(self, num_warehouses=3):
        # Initialize System Dynamics model
        self.sd_model = SupplyChainSDModel(num_warehouses)
        
        # Calculate state and action sizes
        self.state_size = (
            1 +  # factory inventory
            num_warehouses +  # warehouse inventories
            num_warehouses +  # demand rates
            1  # time step
        )
        
        self.action_size = (
            1 +  # production rate
            num_warehouses  # shipping rates
        )
        
        # Initialize RL model
        self.rl_model = SupplyChainRL(self.state_size, self.action_size)
        
        # Simulation parameters
        self.time_step = 0
        self.max_steps = 100
        self.batch_size = 32
        
        # Cost parameters
        self.holding_cost = 1.0
        self.stockout_cost = 10.0
        self.production_cost = 2.0
        self.shipping_cost = 1.0
    
    def get_state(self):
        """Get the current state of the system."""
        sd_state = self.sd_model.get_state()
        
        # Combine all state variables into a single array
        state = np.concatenate([
            [sd_state['factory_inventory']],
            sd_state['warehouse_inventories'],
            sd_state['demand_rates'],
            [self.time_step / self.max_steps]  # normalized time step
        ])
        
        return state
    
    def decode_action(self, action_idx):
        """Convert action index to production and shipping rates."""
        # Simple action space: production rate (0-4) and shipping rates (0-4)
        production_rate = action_idx // (5 ** self.sd_model.num_warehouses)
        shipping_rates = []
        
        remaining = action_idx % (5 ** self.sd_model.num_warehouses)
        for i in range(self.sd_model.num_warehouses):
            rate = remaining % 5
            shipping_rates.append(rate)
            remaining //= 5
        
        return production_rate, shipping_rates
    
    def calculate_reward(self):
        """Calculate the reward based on current state."""
        sd_state = self.sd_model.get_state()
        
        # Calculate costs
        holding_cost = (
            self.holding_cost * sd_state['factory_inventory'] +
            sum(self.holding_cost * inv for inv in sd_state['warehouse_inventories'])
        )
        
        stockout_cost = sum(
            self.stockout_cost * max(0, -inv)
            for inv in sd_state['warehouse_inventories']
        )
        
        production_cost = (
            self.production_cost * sd_state['production_rate']
        )
        
        shipping_cost = sum(
            self.shipping_cost * rate
            for rate in sd_state['shipping_rates']
        )
        
        # Total cost is negative reward
        total_cost = holding_cost + stockout_cost + production_cost + shipping_cost
        reward = -total_cost
        
        return reward
    
    def step(self):
        """Execute one step of the simulation."""
        # Get current state
        state = self.get_state()
        
        # Get action from RL model
        action_idx = self.rl_model.act(state)
        production_rate, shipping_rates = self.decode_action(action_idx)
        
        # Apply action to SD model
        self.sd_model.set_production_rate(production_rate)
        for i, rate in enumerate(shipping_rates):
            self.sd_model.set_shipping_rate(i, rate)
        
        # Advance SD model
        self.sd_model.step()
        
        # Get new state and reward
        next_state = self.get_state()
        reward = self.calculate_reward()
        
        # Check if episode is done
        done = self.time_step >= self.max_steps - 1
        
        # Store experience in RL model
        self.rl_model.remember(state, action_idx, reward, next_state, done)
        
        # Train RL model
        self.rl_model.replay(self.batch_size)
        
        # Update time step
        self.time_step += 1
        
        return next_state, reward, done
    
    def reset(self):
        """Reset the simulation."""
        self.sd_model = SupplyChainSDModel(self.sd_model.num_warehouses)
        self.time_step = 0
        return self.get_state()
    
    def save_model(self, path):
        """Save the RL model."""
        self.rl_model.save(path)
    
    def load_model(self, path):
        """Load the RL model."""
        self.rl_model.load(path) 