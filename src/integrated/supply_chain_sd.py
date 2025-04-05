"""
System Dynamics model for Supply Chain Management using BPTK-Py.
This model represents a multi-echelon supply chain with a factory and multiple warehouses.
"""

from .bptk_model import Model
import numpy as np

class SupplyChainSDModel(Model):
    def __init__(self, num_warehouses=3):
        super().__init__()
        self.num_warehouses = num_warehouses
        
        # Initialize stocks
        self.factory_inventory = self.stock("factory_inventory", 0)
        self.warehouse_inventories = [
            self.stock(f"warehouse_{i}_inventory", 0) 
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
        
        # Set up equations
        self.factory_inventory.equation = self.production_rate - sum(self.shipping_rates)
        
        for i in range(num_warehouses):
            self.warehouse_inventories[i].equation = (
                self.shipping_rates[i] - self.demand_rates[i]
            )
    
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
        
        # Update warehouse inventories
        for i in range(self.num_warehouses):
            self.warehouse_inventories[i].value += (
                self.shipping_rates[i].value - self.demand_rates[i].value
            )
    
    def get_state(self):
        """Get the current state of the supply chain."""
        return {
            'factory_inventory': self.factory_inventory.value,
            'warehouse_inventories': [w.value for w in self.warehouse_inventories],
            'production_rate': self.production_rate.value,
            'shipping_rates': [s.value for s in self.shipping_rates],
            'demand_rates': [d.value for d in self.demand_rates]
        } 