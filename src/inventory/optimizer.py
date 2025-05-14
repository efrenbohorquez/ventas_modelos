import numpy as np
import pandas as pd
from scipy.stats import norm

class InventoryOptimizer:
    """
    Optimiza los niveles de inventario basado en predicciones de demanda.
    """
    def __init__(self, lead_time=3, service_level=0.95, holding_cost_rate=0.25):
        self.lead_time = lead_time
        self.service_level = service_level
        self.z_score = norm.ppf(service_level)
        self.holding_cost_rate = holding_cost_rate
    
    def calculate_safety_stock(self, demand_std, lead_time_std=None):
        if lead_time_std is None:
            safety_stock = self.z_score * demand_std * np.sqrt(self.lead_time)
        else:
            avg_demand = demand_std * 2  # Estimación aproximada de demanda media
            safety_stock = self.z_score * np.sqrt(
                self.lead_time * demand_std**2 + avg_demand**2 * lead_time_std**2
            )
        return np.ceil(safety_stock)
    
    def calculate_reorder_point(self, avg_demand, safety_stock):
        return np.ceil(avg_demand * self.lead_time + safety_stock)
    
    def calculate_economic_order_quantity(self, annual_demand, order_cost, unit_cost):
        holding_cost = unit_cost * self.holding_cost_rate
        eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost)
        return np.ceil(eoq)
    
    def optimize_inventory(self, predictions_df, product_info):
        results = []
        grouped = predictions_df.groupby(['Product line', 'Branch'])
        for (product, branch), group in grouped:
            avg_demand = group['prediction'].mean()
            demand_std = group['prediction'].std()
            try:
                product_data = product_info[(product_info['Product line'] == product)]
                unit_cost = product_data['unit_cost'].values[0]
                order_cost = product_data['order_cost'].values[0]
            except:
                unit_cost = 50
                order_cost = 100
            safety_stock = self.calculate_safety_stock(demand_std)
            reorder_point = self.calculate_reorder_point(avg_demand, safety_stock)
            annual_demand = avg_demand * 365
            eoq = self.calculate_economic_order_quantity(annual_demand, order_cost, unit_cost)
            results.append({
                'Product line': product,
                'Branch': branch,
                'Average Daily Demand': avg_demand,
                'Demand StdDev': demand_std,
                'Safety Stock': safety_stock,
                'Reorder Point': reorder_point,
                'Economic Order Quantity': eoq,
                'Unit Cost': unit_cost,
                'Annual Inventory Cost': eoq * unit_cost * self.holding_cost_rate / 2 + (annual_demand / eoq) * order_cost
            })
        return pd.DataFrame(results)
    
    def simulate_inventory_policy(self, demand_series, initial_inventory, 
                                  reorder_point, order_quantity, lead_time):
        inventory = initial_inventory
        pending_orders = []
        results = []
        for day, demand in enumerate(demand_series):
            received_qty = 0
            new_pending = []
            for qty, days_left in pending_orders:
                if days_left <= 1:
                    received_qty += qty
                else:
                    new_pending.append((qty, days_left - 1))
            pending_orders = new_pending
            inventory += received_qty
            fulfilled = min(demand, inventory)
            lost_sales = demand - fulfilled
            inventory -= fulfilled
            new_order = 0
            if inventory <= reorder_point and not pending_orders:
                new_order = order_quantity
                pending_orders.append((order_quantity, lead_time))
            results.append({
                'Day': day,
                'Demand': demand,
                'Beginning Inventory': inventory + fulfilled,
                'Ending Inventory': inventory,
                'Fulfilled': fulfilled,
                'Lost Sales': lost_sales,
                'Service Level': fulfilled / demand if demand > 0 else 1.0,
                'New Order': new_order,
                'Pending Orders': sum(qty for qty, _ in pending_orders)
            })
        return pd.DataFrame(results)
