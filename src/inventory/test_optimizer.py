import numpy as np
import pandas as pd
from optimizer import InventoryOptimizer

# Datos de ejemplo para predicciones
predictions_df = pd.DataFrame({
    'Product line': ['Health and beauty'] * 10 + ['Electronic accessories'] * 10,
    'Branch': ['A'] * 10 + ['B'] * 10,
    'prediction': np.random.poisson(20, 20)
})

# Información de productos
product_info = pd.DataFrame({
    'Product line': ['Health and beauty', 'Electronic accessories'],
    'unit_cost': [50, 80],
    'order_cost': [100, 120]
})

optimizer = InventoryOptimizer(lead_time=3, service_level=0.95, holding_cost_rate=0.25)
results = optimizer.optimize_inventory(predictions_df, product_info)
print(results)

# Simulación de política de inventario
sim = optimizer.simulate_inventory_policy(
    demand_series=np.random.poisson(20, 30),
    initial_inventory=100,
    reorder_point=60,
    order_quantity=80,
    lead_time=3
)
print(sim.head())
