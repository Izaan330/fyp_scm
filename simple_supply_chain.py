"""
Simplified version of the supply chain management system.
This version doesn't rely on external dependencies like BPTK-Py or PyTorch.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

# Simple Model class for System Dynamics
class Stock:
    def __init__(self, name, initial_value=0):
        self.name = name
        self.value = initial_value
        self.equation = None

class Flow:
    def __init__(self, name, initial_value=0):
        self.name = name
        self.value = initial_value

class Auxiliary:
    def __init__(self, name, initial_value=0):
        self.name = name
        self.value = initial_value

class Model:
    def __init__(self):
        self.stocks = {}
        self.flows = {}
        self.auxiliaries = {}
    
    def stock(self, name, initial_value=0):
        """Create a new stock variable."""
        stock = Stock(name, initial_value)
        self.stocks[name] = stock
        return stock
    
    def flow(self, name, initial_value=0):
        """Create a new flow variable."""
        flow = Flow(name, initial_value)
        self.flows[name] = flow
        return flow
    
    def auxiliary(self, name, initial_value=0):
        """Create a new auxiliary variable."""
        auxiliary = Auxiliary(name, initial_value)
        self.auxiliaries[name] = auxiliary
        return auxiliary
    
    def step(self):
        """Advance the model by one time step."""
        # This is a placeholder - actual implementation would depend on the model
        pass

# System Dynamics model for Supply Chain
class SupplyChainSDModel(Model):
    def __init__(self, num_warehouses=3):
        super().__init__()
        self.num_warehouses = num_warehouses
        
        # Initialize stocks
        self.factory_inventory = self.stock("factory_inventory", 10)  # Start with some inventory
        self.warehouse_inventories = [
            self.stock(f"warehouse_{i}_inventory", 5)  # Start with some inventory
            for i in range(num_warehouses)
        ]
        
        # Initialize flows
        self.production_rate = self.flow("production_rate", 0)
        self.shipping_rates = [
            self.flow(f"shipping_rate_{i}", 0)
            for i in range(num_warehouses)
        ]
        
        # Initialize auxiliaries
        self.demand_rates = [
            self.auxiliary(f"demand_rate_{i}", 0)
            for i in range(num_warehouses)
        ]
        
        # Set up equations - we don't need to store equations as attributes
        # since we handle the calculations in the step() method
    
    def set_production_rate(self, rate):
        """Set the production rate at the factory."""
        self.production_rate.value = rate
    
    def set_shipping_rate(self, warehouse_idx, rate):
        """Set the shipping rate to a specific warehouse."""
        if 0 <= warehouse_idx < self.num_warehouses:
            self.shipping_rates[warehouse_idx].value = rate
    
    def set_demand_rate(self, warehouse_idx, rate):
        """Set the demand rate at a specific warehouse."""
        if 0 <= warehouse_idx < self.num_warehouses:
            self.demand_rates[warehouse_idx].value = rate
    
    def step(self):
        """Advance the model by one time step."""
        # Update factory inventory
        self.factory_inventory.value += self.production_rate.value - sum(s.value for s in self.shipping_rates)
        
        # Ensure factory inventory doesn't go negative
        self.factory_inventory.value = max(0, self.factory_inventory.value)
        
        # Update warehouse inventories
        for i in range(self.num_warehouses):
            self.warehouse_inventories[i].value += (
                self.shipping_rates[i].value - self.demand_rates[i].value
            )
            
            # Ensure warehouse inventory doesn't go negative
            self.warehouse_inventories[i].value = max(0, self.warehouse_inventories[i].value)
    
    def get_state(self):
        """Get the current state of the supply chain."""
        return {
            'factory_inventory': self.factory_inventory.value,
            'warehouse_inventories': [w.value for w in self.warehouse_inventories],
            'production_rate': self.production_rate.value,
            'shipping_rates': [s.value for s in self.shipping_rates],
            'demand_rates': [d.value for d in self.demand_rates]
        }

# Simple RL model for Supply Chain
class SimpleRL:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # RL parameters
        self.memory = deque(maxlen=20000)  # Increased memory size
        self.gamma = 0.95    # Slightly reduced to focus more on immediate rewards
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.15  # Increased minimum exploration
        self.epsilon_decay = 0.999  # Slower decay
        self.learning_rate = 0.02  # Increased learning rate
        
        # Q-table for simple Q-learning
        self.q_table = np.zeros((state_size, action_size))
        
        # Initialize Q-table with optimistic initial values to encourage exploration
        self.q_table = np.ones((state_size, action_size)) * 100.0
        
        # Add a stronger bias toward moderate actions (2) to prevent extreme oscillations
        for i in range(state_size):
            self.q_table[i, 2] += 50.0  # Stronger bias toward production rate of 2
            self.q_table[i, 1] += 25.0  # Also bias toward production rate of 1
            self.q_table[i, 3] += 25.0  # Also bias toward production rate of 3
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose an action based on the current state."""
        if random.random() <= self.epsilon:
            # Bias random actions toward moderate values
            if random.random() < 0.9:  # 90% chance of choosing a moderate action
                return random.choice([1, 2, 3])
            return random.randrange(self.action_size)
        
        # Convert state to index for Q-table
        state_idx = self._state_to_index(state)
        
        # Add noise to Q-values to prevent ties
        noise = np.random.normal(0, 0.1, self.action_size)
        return np.argmax(self.q_table[state_idx] + noise)
    
    def _state_to_index(self, state):
        """Convert state to index for Q-table using improved discretization."""
        # Use more sophisticated state discretization
        factory_inv = state[0]
        warehouse_invs = state[1:4]
        demand_rates = state[4:7]
        time_step = state[7]
        
        # Discretize factory inventory into 10 bins
        factory_bin = min(int(factory_inv / 5), 9)
        
        # Discretize warehouse inventories into 5 bins each
        warehouse_bins = [min(int(inv / 2), 4) for inv in warehouse_invs]
        
        # Discretize demand rates into 3 bins each
        demand_bins = [min(int(rate * 2), 2) for rate in demand_rates]
        
        # Combine all bins into a single index
        state_hash = factory_bin
        for bin_val in warehouse_bins + demand_bins:
            state_hash = state_hash * 5 + bin_val
        
        return state_hash % self.state_size
    
    def replay(self, batch_size):
        """Train the model using experience replay."""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            # Convert states to indices for Q-table
            state_idx = self._state_to_index(state)
            next_state_idx = self._state_to_index(next_state)
            
            # Q-learning update
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(self.q_table[next_state_idx])
            
            self.q_table[state_idx, action] += self.learning_rate * (target - self.q_table[state_idx, action])
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, name):
        """Save the Q-table."""
        np.save(name, self.q_table)
    
    def load(self, name):
        """Load the Q-table."""
        self.q_table = np.load(name)

# Connector class that integrates System Dynamics and RL models
class SupplyChainConnector:
    def __init__(self, num_warehouses=3):
        # Initialize System Dynamics model
        self.sd_model = SupplyChainSDModel(num_warehouses)
        
        # Calculate state and action sizes
        self.state_size = 2000  # Increased state space
        self.action_size = 5  # Simplified action space: 0-4 for production rate
        
        # Initialize RL model
        self.rl_model = SimpleRL(self.state_size, self.action_size)
        
        # Simulation parameters
        self.time_step = 0
        self.max_steps = 100
        self.batch_size = 128  # Increased batch size
        
        # Cost parameters
        self.holding_cost = 1.0  # Increased holding cost
        self.stockout_cost = 50.0  # Increased stockout cost
        self.production_cost = 2.0  # Increased production cost
        self.shipping_cost = 1.0  # Increased shipping cost
        
        # Target inventory levels
        self.target_factory_inventory = 20.0
        self.target_warehouse_inventory = 10.0
        
        # Initialize last_production_rate
        self.last_production_rate = 0
        
        # Set initial demand rates
        for i in range(num_warehouses):
            self.sd_model.set_demand_rate(i, random.uniform(0.8, 1.2))
    
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
        """Convert action index to production and shipping rates with improved logic."""
        # Simplified action space: just production rate (0-4)
        production_rate = action_idx
        
        # Improved shipping policy: ship based on warehouse inventory levels
        factory_inventory = self.sd_model.factory_inventory.value
        shipping_rates = []
        
        if factory_inventory > 0:
            for i in range(self.sd_model.num_warehouses):
                # Get current warehouse inventory and demand
                warehouse_inv = self.sd_model.warehouse_inventories[i].value
                warehouse_demand = self.sd_model.demand_rates[i].value
                
                # Calculate target inventory (dynamic based on demand variability)
                target_inventory = max(
                    self.target_warehouse_inventory,
                    warehouse_demand * 2 + warehouse_demand * 0.5  # Safety stock
                )
                
                # Calculate inventory gap
                inventory_gap = target_inventory - warehouse_inv
                
                # Calculate shipping amount based on inventory gap and demand
                if inventory_gap > 0:
                    # Ship more if inventory is below target
                    shipping_amount = min(
                        factory_inventory / 3,  # Equal share of factory inventory
                        warehouse_demand + inventory_gap * 0.2  # Gradual correction
                    )
                else:
                    # Ship less if inventory is above target
                    shipping_amount = max(
                        warehouse_demand * 0.8,  # Minimum shipping amount
                        warehouse_demand * (1.0 - abs(inventory_gap) * 0.1)  # Reduced shipping
                    )
                
                shipping_rates.append(shipping_amount)
        else:
            shipping_rates = [0] * self.sd_model.num_warehouses
        
        return production_rate, shipping_rates
    
    def calculate_reward(self):
        """Calculate the reward based on current state with improved reward function."""
        sd_state = self.sd_model.get_state()
        
        # Calculate inventory deviations
        factory_inv_deviation = abs(sd_state['factory_inventory'] - self.target_factory_inventory)
        warehouse_inv_deviations = [
            abs(inv - self.target_warehouse_inventory)
            for inv in sd_state['warehouse_inventories']
        ]
        
        # Calculate costs
        holding_cost = (
            self.holding_cost * max(0, sd_state['factory_inventory'] - self.target_factory_inventory) +
            sum(self.holding_cost * max(0, inv - self.target_warehouse_inventory)
                for inv in sd_state['warehouse_inventories'])
        )
        
        stockout_cost = sum(
            self.stockout_cost * abs(min(0, inv))
            for inv in sd_state['warehouse_inventories']
        )
        
        production_cost = (
            self.production_cost * sd_state['production_rate']
        )
        
        shipping_cost = sum(
            self.shipping_cost * rate
            for rate in sd_state['shipping_rates']
        )
        
        # Add a reward for meeting demand
        demand_met = sum(
            min(inv, demand)
            for inv, demand in zip(sd_state['warehouse_inventories'], sd_state['demand_rates'])
        )
        demand_reward = 30.0 * demand_met
        
        # Add a reward for maintaining target inventory levels
        inventory_reward = (
            20.0 * (1.0 / (1.0 + factory_inv_deviation)) +
            sum(20.0 * (1.0 / (1.0 + dev)) for dev in warehouse_inv_deviations)
        )
        
        # Add a penalty for large production rate changes
        production_change = abs(sd_state['production_rate'] - self.last_production_rate)
        oscillation_penalty = 5.0 * production_change
        
        # Add a penalty for having zero inventory
        zero_inventory_penalty = sum(
            20.0 if inv <= 0 else 0
            for inv in sd_state['warehouse_inventories']
        )
        
        # Store current production rate for next step
        self.last_production_rate = sd_state['production_rate']
        
        # Calculate total reward
        total_cost = holding_cost + stockout_cost + production_cost + shipping_cost
        reward = (
            demand_reward +
            inventory_reward -
            total_cost -
            oscillation_penalty -
            zero_inventory_penalty
        )
        
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
        
        # Set initial demand rates
        for i in range(self.sd_model.num_warehouses):
            self.sd_model.set_demand_rate(i, random.uniform(0.8, 1.2))
            
        self.time_step = 0
        self.last_production_rate = 0
        return self.get_state()
    
    def save_model(self, path):
        """Save the RL model."""
        self.rl_model.save(path)
    
    def load_model(self, path):
        """Load the RL model."""
        self.rl_model.load(path)

# Demo script
def run_episode(connector, render=False):
    """Run a single episode of the simulation."""
    state = connector.reset()
    total_reward = 0
    done = False
    
    # Lists to store metrics for plotting
    factory_inventory = []
    warehouse_inventories = [[] for _ in range(connector.sd_model.num_warehouses)]
    rewards = []
    production_rates = []
    shipping_rates = [[] for _ in range(connector.sd_model.num_warehouses)]
    demand_rates = [[] for _ in range(connector.sd_model.num_warehouses)]
    
    while not done:
        # Get action from RL model and execute step
        next_state, reward, done = connector.step()
        total_reward += reward
        
        # Store metrics
        sd_state = connector.sd_model.get_state()
        factory_inventory.append(sd_state['factory_inventory'])
        for i, inv in enumerate(sd_state['warehouse_inventories']):
            warehouse_inventories[i].append(inv)
        rewards.append(reward)
        production_rates.append(sd_state['production_rate'])
        for i, rate in enumerate(sd_state['shipping_rates']):
            shipping_rates[i].append(rate)
        for i, rate in enumerate(sd_state['demand_rates']):
            demand_rates[i].append(rate)
        
        if render:
            print(f"Step {connector.time_step}")
            print(f"Factory Inventory: {sd_state['factory_inventory']:.2f}")
            print(f"Production Rate: {sd_state['production_rate']:.2f}")
            print(f"Warehouse Inventories: {[f'{inv:.2f}' for inv in sd_state['warehouse_inventories']]}")
            print(f"Shipping Rates: {[f'{rate:.2f}' for rate in sd_state['shipping_rates']]}")
            print(f"Demand Rates: {[f'{rate:.2f}' for rate in sd_state['demand_rates']]}")
            print(f"Reward: {reward:.2f}")
            print("---")
    
    return total_reward, factory_inventory, warehouse_inventories, rewards, production_rates, shipping_rates, demand_rates

def plot_results(factory_inventory, warehouse_inventories, rewards, production_rates, shipping_rates, demand_rates):
    """Plot the results of the simulation."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot inventories
    time_steps = range(len(factory_inventory))
    ax1.plot(time_steps, factory_inventory, label='Factory', linewidth=2)
    for i, inv in enumerate(warehouse_inventories):
        ax1.plot(time_steps, inv, label=f'Warehouse {i+1}', linewidth=2)
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Inventory Level')
    ax1.set_title('Inventory Levels Over Time')
    ax1.legend()
    ax1.grid(True)
    
    # Plot rewards
    ax2.plot(time_steps, rewards, color='green', linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Reward')
    ax2.set_title('Rewards Over Time')
    ax2.grid(True)
    
    # Plot production and shipping rates
    ax3.plot(time_steps, production_rates, label='Production', color='blue', linewidth=2)
    for i, rates in enumerate(shipping_rates):
        ax3.plot(time_steps, rates, label=f'Shipping to W{i+1}', linestyle='--', linewidth=1.5)
    
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Rate')
    ax3.set_title('Production and Shipping Rates')
    ax3.legend()
    ax3.grid(True)
    
    # Plot demand rates
    for i, rates in enumerate(demand_rates):
        ax4.plot(time_steps, rates, label=f'Demand at W{i+1}', linewidth=2)
    
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Rate')
    ax4.set_title('Demand Rates')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # Create the supply chain connector
    connector = SupplyChainConnector(num_warehouses=3)
    
    # Training parameters
    num_episodes = 500  # Increased number of episodes
    best_reward = float('-inf')
    running_rewards = deque(maxlen=10)  # Track running average of rewards
    
    print("Starting training...")
    for episode in range(num_episodes):
        total_reward, factory_inv, warehouse_inv, rewards, prod_rates, ship_rates, demand_rates = run_episode(connector)
        running_rewards.append(total_reward)
        running_avg = sum(running_rewards) / len(running_rewards)
        
        # Save the best model
        if total_reward > best_reward:
            best_reward = total_reward
            connector.save_model('best_model.npy')
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Running Average: {running_avg:.2f}")
            print(f"Best Reward: {best_reward:.2f}")
            print(f"Epsilon: {connector.rl_model.epsilon:.4f}")
            print("---")
    
    print("\nTraining completed!")
    print(f"Best Reward: {best_reward:.2f}")
    
    # Load the best model and run a demo episode
    print("\nRunning demo episode with best model...")
    connector.load_model('best_model.npy')
    total_reward, factory_inv, warehouse_inv, rewards, prod_rates, ship_rates, demand_rates = run_episode(connector, render=True)
    
    # Plot the results
    plot_results(factory_inv, warehouse_inv, rewards, prod_rates, ship_rates, demand_rates)

if __name__ == "__main__":
    main() 