"""
Simplified version of the BPTK-Py Model class for use in the supply chain simulation.
"""

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