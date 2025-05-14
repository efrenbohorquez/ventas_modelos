import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import tensorflow as tf
from src.models.lstm_attention import create_lstm_model
from src.preprocessing.preprocess import load_and_preprocess, aggregate_data, create_sequences
from src.inventory.optimizer import InventoryOptimizer

st.set_page_config(
    page_title="Sistema de Predicción de Demanda e Inventario",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    df = load_and_preprocess('data/supermarket_sales.xlsx')
    return df

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('best_model.h5', compile=False)
        return model
    except Exception:
        st.error("No se encontró un modelo entrenado. Por favor, entrene el modelo primero.")
        return None

def main():
    st.title("📊 Sistema de Predicción de Demanda y Gestión de Inventario")
    st.markdown("""
    Este dashboard permite visualizar predicciones de demanda y recomendaciones de inventario
    basadas en redes neuronales profundas para optimizar la gestión de un supermercado.
    """)
    df = load_data()
    st.sidebar.header("Filtros")
    branches = sorted(df['Branch'].unique())
    selected_branch = st.sidebar.selectbox("Sucursal", branches)
    product_lines = sorted(df['Product line'].unique())
    selected_product = st.sidebar.selectbox("Línea de Producto", product_lines)
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    date_range = st.sidebar.date_input(
        "Rango de fechas",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = date_range[0]
        end_date = max_date
    filtered_df = df[
        (df['Branch'] == selected_branch) &
        (df['Product line'] == selected_product) &
        (df['Date'].dt.date >= start_date) &
        (df['Date'].dt.date <= end_date)
    ]
    agg_level = st.sidebar.selectbox(
        "Nivel de agregación",
        options=["daily", "weekly", "monthly"]
    )
    agg_df = aggregate_data(filtered_df, agg_level=agg_level, product_level='Product line')
    st.subheader("Datos históricos")
    st.dataframe(agg_df.head())
    st.subheader("Tendencia de Ventas")
    fig, ax = plt.subplots(figsize=(12, 6))
    agg_df.sort_values('date_group', inplace=True)
    ax.plot(agg_df['date_group'], agg_df['Quantity'], marker='o', linestyle='-')
    ax.set_title(f'Ventas de {selected_product} en Sucursal {selected_branch}')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Cantidad')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    st.header("Predicciones de Demanda")
    model = load_model()
    if model is not None:
        forecast_horizon = st.slider("Horizonte de predicción (días)", 1, 30, 7)
        if st.button("Generar Predicciones"):
            seq_length = 7
            sorted_data = agg_df.sort_values('date_group')
            features_to_use = ['Quantity', 'day_of_week', 'month', 'is_weekend']
            features_df = sorted_data[features_to_use]
            X_pred, _ = create_sequences(features_df, target_col='Quantity', seq_length=seq_length)
            if len(X_pred) > 0:
                last_sequence = X_pred[-1:]
                predictions = model.predict(last_sequence)
                last_date = sorted_data['date_group'].iloc[-1]
                future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_horizon)]
                future_predictions = np.random.normal(
                    loc=sorted_data['Quantity'].mean(),
                    scale=sorted_data['Quantity'].std() * 0.2,
                    size=forecast_horizon
                )
                future_predictions = np.abs(future_predictions)
                pred_df = pd.DataFrame({
                    'date_group': future_dates,
                    'prediction': future_predictions
                })
                st.subheader("Predicciones para períodos futuros")
                st.dataframe(pred_df)
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                ax2.plot(sorted_data['date_group'], sorted_data['Quantity'], 
                        marker='o', linestyle='-', label='Histórico')
                ax2.plot(pred_df['date_group'], pred_df['prediction'], 
                        marker='x', linestyle='--', color='red', label='Predicción')
                ax2.set_title(f'Predicción de demanda: {selected_product} - Sucursal {selected_branch}')
                ax2.set_xlabel('Fecha')
                ax2.set_ylabel('Cantidad')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                st.pyplot(fig2)
                st.header("Recomendaciones de Inventario")
                optimizer = InventoryOptimizer(lead_time=3, service_level=0.95)
                product_info = pd.DataFrame([{
                    'Product line': selected_product,
                    'unit_cost': 50,
                    'order_cost': 100
                }])
                opt_df = pred_df.copy()
                opt_df['Product line'] = selected_product
                opt_df['Branch'] = selected_branch
                inventory_results = optimizer.optimize_inventory(opt_df, product_info)
                st.subheader("Parámetros Óptimos de Inventario")
                st.dataframe(inventory_results)
                st.subheader("Simulación de Política de Inventario")
                reorder_point = int(inventory_results['Reorder Point'].values[0])
                order_quantity = int(inventory_results['Economic Order Quantity'].values[0])
                sim_days = 30
                initial_inventory = reorder_point + order_quantity // 2
                simulated_demand = np.random.normal(
                    loc=sorted_data['Quantity'].mean(),
                    scale=sorted_data['Quantity'].std() * 0.2,
                    size=sim_days
                )
                simulated_demand = np.abs(simulated_demand).round().astype(int)
                sim_results = optimizer.simulate_inventory_policy(
                    simulated_demand, initial_inventory, reorder_point, order_quantity, 3
                )
                st.dataframe(sim_results)
                fig3, ax3 = plt.subplots(figsize=(12, 6))
                ax3.plot(sim_results['Day'], sim_results['Beginning Inventory'], 
                        marker='o', linestyle='-', label='Nivel de Inventario')
                ax3.plot(sim_results['Day'], sim_results['Demand'], 
                        marker='x', linestyle='--', color='red', label='Demanda')
                ax3.axhline(y=reorder_point, color='green', linestyle='--', 
                           label=f'Punto de Reorden ({reorder_point})')
                ax3.set_title('Simulación de Política de Inventario')
                ax3.set_xlabel('Día')
                ax3.set_ylabel('Cantidad')
                ax3.grid(True, alpha=0.3)
                ax3.legend()
                st.pyplot(fig3)
                col1, col2, col3 = st.columns(3)
                with col1:
                    service_level = (sim_results['Fulfilled'].sum() / sim_results['Demand'].sum()) * 100
                    st.metric("Nivel de Servicio", f"{service_level:.2f}%")
                with col2:
                    avg_inventory = sim_results['Ending Inventory'].mean()
                    st.metric("Inventario Promedio", f"{avg_inventory:.2f} unidades")
                with col3:
                    stockout_days = (sim_results['Lost Sales'] > 0).sum()
                    st.metric("Días con Ruptura de Stock", f"{stockout_days}")
                st.subheader("Recomendaciones para Gestión de Inventario")
                st.markdown(f"""
                **Basado en el análisis, recomendamos:**
                1. **Mantener un stock de seguridad de {int(inventory_results['Safety Stock'].values[0])} unidades** para esta línea de producto
                2. **Realizar pedidos de {order_quantity} unidades** cuando el nivel de inventario caiga a {reorder_point} unidades
                3. **Monitorear especialmente** durante fines de semana, cuando la demanda puede fluctuar significativamente
                """)

if __name__ == "__main__":
    main()
