import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.preprocessing.preprocess import load_and_preprocess, aggregate_data, create_sequences
from src.models.lstm_attention import create_lstm_model

# 1. Cargar y procesar los datos
file_path = 'data/supermarket_sales.xlsx'
df = load_and_preprocess(file_path)
agg_df = aggregate_data(df, agg_level='daily', product_level='Product line')
agg_df = agg_df.sort_values(['Branch', 'Product line', 'date_group'])

# 2. Seleccionar un grupo de ejemplo (ajusta según tus datos reales)
sample_data = agg_df[(agg_df['Branch'] == 'A') & (agg_df['Product line'] == 'Health and beauty')].reset_index(drop=True)

# Unir variables temporales originales al sample_data
# (date_group equivale a Date en df original)
temp_vars = ['date_group', 'day_of_week', 'month', 'is_weekend']
temp_df = df[['Date', 'day_of_week', 'month', 'is_weekend']].copy()
temp_df = temp_df.rename(columns={'Date': 'date_group'})
sample_data = sample_data.merge(temp_df, on='date_group', how='left')

if len(sample_data) > 20:
    # 3. Crear secuencias para el modelo
    features = ['Quantity', 'day_of_week', 'month', 'is_weekend']
    X, y = create_sequences(sample_data[features], target_col='Quantity', seq_length=7)
    # 4. Separar en train y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 5. Crear y entrenar el modelo
    model, _ = create_lstm_model(seq_length=7, n_features=X.shape[2])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=8)
    # 6. Guardar el modelo entrenado
    model.save('best_model.h5')
    print('Modelo entrenado y guardado como best_model.h5')
else:
    print('No hay suficientes datos para entrenar el modelo.')
