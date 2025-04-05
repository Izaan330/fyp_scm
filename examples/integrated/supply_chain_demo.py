"""
Demo script showing how to use the integrated supply chain management system
combining System Dynamics and Reinforcement Learning.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the src directory to the Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
src_path = str(project_root / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from integrated.supply_chain_connector import SupplyChainConnector

def run_episode(connector, render=False):
    """Run a single episode of the simulation."""
    state = connector.reset()
    total_reward = 0
    done = False
    
    # Lists to store metrics for plotting
    factory_inventory = []
    warehouse_inventories = [[] for _ in range(connector.sd_model.num_warehouses)]
    rewards = []
    
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
        
        if render:
            print(f"Step {connector.time_step}")
            print(f"Factory Inventory: {sd_state['factory_inventory']:.2f}")
            print(f"Warehouse Inventories: {[f'{inv:.2f}' for inv in sd_state['warehouse_inventories']]}")
            print(f"Reward: {reward:.2f}")
            print("---")
    
    return total_reward, factory_inventory, warehouse_inventories, rewards

def plot_results(factory_inventory, warehouse_inventories, rewards):
    """Plot the results of the simulation."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot inventories
    time_steps = range(len(factory_inventory))
    ax1.plot(time_steps, factory_inventory, label='Factory')
    for i, inv in enumerate(warehouse_inventories):
        ax1.plot(time_steps, inv, label=f'Warehouse {i+1}')
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Inventory Level')
    ax1.set_title('Inventory Levels Over Time')
    ax1.legend()
    ax1.grid(True)
    
    # Plot rewards
    ax2.plot(time_steps, rewards)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Reward')
    ax2.set_title('Rewards Over Time')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # Create the supply chain connector
    connector = SupplyChainConnector(num_warehouses=3)
    
    # Training parameters
    num_episodes = 100
    best_reward = float('-inf')
    
    print("Starting training...")
    for episode in range(num_episodes):
        total_reward, factory_inv, warehouse_inv, rewards = run_episode(connector)
        
        # Save the best model
        if total_reward > best_reward:
            best_reward = total_reward
            connector.save_model('best_model.pth')
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Best Reward: {best_reward:.2f}")
            print("---")
    
    print("\nTraining completed!")
    print(f"Best Reward: {best_reward:.2f}")
    
    # Load the best model and run a demo episode
    print("\nRunning demo episode with best model...")
    connector.load_model('best_model.pth')
    total_reward, factory_inv, warehouse_inv, rewards = run_episode(connector, render=True)
    
    # Plot the results
    plot_results(factory_inv, warehouse_inv, rewards)

if __name__ == "__main__":
    main() 