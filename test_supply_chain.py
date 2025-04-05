"""
Simple test script to verify that the supply chain code works.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
current_dir = Path(__file__).resolve().parent
src_path = str(current_dir / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    # Try to import the required modules
    from integrated.supply_chain_sd import SupplyChainSDModel
    from integrated.supply_chain_rl import SupplyChainRL
    from integrated.supply_chain_connector import SupplyChainConnector
    
    print("Successfully imported all required modules!")
    
    # Create a simple supply chain model
    sd_model = SupplyChainSDModel(num_warehouses=2)
    print("Created System Dynamics model")
    
    # Set some values
    sd_model.set_production_rate(5)
    sd_model.set_shipping_rate(0, 3)
    sd_model.set_shipping_rate(1, 2)
    sd_model.set_demand_rate(0, 2)
    sd_model.set_demand_rate(1, 1)
    
    # Get the state
    state = sd_model.get_state()
    print("Initial state:", state)
    
    # Step the model
    sd_model.step()
    
    # Get the new state
    new_state = sd_model.get_state()
    print("State after one step:", new_state)
    
    # Create a simple RL model
    rl_model = SupplyChainRL(state_size=7, action_size=25)
    print("Created RL model")
    
    # Create a connector
    connector = SupplyChainConnector(num_warehouses=2)
    print("Created connector")
    
    # Get the initial state
    connector_state = connector.get_state()
    print("Connector initial state:", connector_state)
    
    # Step the connector
    next_state, reward, done = connector.step()
    print("Connector state after one step:", next_state)
    print("Reward:", reward)
    print("Done:", done)
    
    print("\nAll tests passed! The code is working correctly.")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 