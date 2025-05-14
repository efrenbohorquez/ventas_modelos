import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import datetime as dt

def load_and_preprocess(file_path):
    """
    Carga y preprocesa los datos de ventas del supermercado.

    Parámetros:
    - file_path: Ruta al archivo de datos (CSV o XLSX)

    Retorna:
    - DataFrame con variables temporales y de contexto agregadas.
    """
    # Cargar datos
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    
    # Convertir columnas de fecha y hora
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extraer características temporales
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['year'] = df['Date'].dt.year
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Crear variables para temporadas
    df['season'] = df['month'].apply(lambda x: 0 if 3 <= x <= 5 else 
                                            (1 if 6 <= x <= 8 else 
                                            (2 if 9 <= x <= 11 else 3)))
    
    # Extraer la hora (si existe)
    if 'Time' in df.columns:
        # Si la columna es tipo datetime.time, extraer la hora directamente
        if np.issubdtype(df['Time'].dtype, np.dtype('O')) and isinstance(df['Time'].iloc[0], dt.time):
            df['hour'] = df['Time'].apply(lambda t: t.hour)
        else:
            df['hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour
        df['part_of_day'] = pd.cut(df['hour'], 
                                  bins=[0, 6, 12, 18, 24], 
                                  labels=['night', 'morning', 'afternoon', 'evening'],
                                  right=False, include_lowest=True)
    
    return df

def aggregate_data(df, agg_level='daily', product_level='Product line'):
    """
    Agrega los datos según el nivel temporal y de producto deseado.

    Parámetros:
    - df: DataFrame de ventas preprocesado
    - agg_level: Nivel de agregación temporal ('daily', 'weekly', 'monthly')
    - product_level: Nivel de agregación de producto ('Product line', 'individual')

    Retorna:
    - DataFrame agregado por fecha y producto
    """
    # Crear columna de fecha según nivel de agregación
    if agg_level == 'daily':
        df['date_group'] = df['Date']
    elif agg_level == 'weekly':
        df['date_group'] = df['Date'] - pd.to_timedelta(df['Date'].dt.dayofweek, unit='d')
    elif agg_level == 'monthly':
        df['date_group'] = df['Date'].dt.strftime('%Y-%m-01')
        df['date_group'] = pd.to_datetime(df['date_group'])
    
    # Agrupar datos
    if product_level == 'Product line':
        agg_df = df.groupby(['date_group', 'Branch', product_level]).agg({
            'Quantity': 'sum',
            'Total': 'sum',
            'gross income': 'sum',
            'Rating': 'mean'
        }).reset_index()
    else:  # individual product level
        agg_df = df.groupby(['date_group', 'Branch', 'Invoice ID']).agg({
            'Quantity': 'sum',
            'Total': 'sum',
            'gross income': 'sum',
            'Rating': 'mean'
        }).reset_index()
    
    return agg_df

def create_sequences(data, target_col, seq_length=7, forecast_horizon=1):
    """
    Crea secuencias para el entrenamiento de redes neuronales recurrentes.

    Parámetros:
    - data: DataFrame con los datos (ordenados por fecha)
    - target_col: Columna objetivo a predecir
    - seq_length: Longitud de secuencia histórica (ventana)
    - forecast_horizon: Horizonte de predicción (pasos futuros)

    Retorna:
    - X: array de secuencias de entrada (n_samples, seq_length, n_features)
    - y: array de valores objetivo (n_samples,)
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data.iloc[i:(i+seq_length)].values)
        y.append(data.iloc[i+seq_length+forecast_horizon-1][target_col])
    
    return np.array(X), np.array(y)

def prepare_features(df, cat_features, num_features):
    """
    Prepara un pipeline de transformación para características categóricas y numéricas.

    Parámetros:
    - df: DataFrame de entrada
    - cat_features: Lista de nombres de columnas categóricas
    - num_features: Lista de nombres de columnas numéricas

    Retorna:
    - preprocessor: Pipeline de preprocesamiento listo para usar en scikit-learn
    """
    # Definir preprocesadores
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Combinar preprocesadores en un ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ])
    
    return preprocessor

def main():
    """
    Ejemplo de uso del preprocesamiento y generación de secuencias.
    """
    # Ejemplo de uso
    df = load_and_preprocess('c:/Users/efren/Desktop/demandEnvairomentPrediction/data/supermarket_sales.xlsx')
    print("Datos cargados y preprocesados. Forma:", df.shape)
    
    # Agregar datos por día y línea de producto
    agg_df = aggregate_data(df, agg_level='daily', product_level='Product line')
    print("Datos agregados. Forma:", agg_df.shape)
    
    # Definir características para el modelo
    cat_features = ['Branch', 'Product line', 'day_of_week', 'season', 'is_weekend']
    num_features = ['month', 'Rating']
    
    # Crear pipeline de preprocesamiento
    preprocessor = prepare_features(agg_df, cat_features, num_features)
    
    # Ejemplo de secuencias
    # Primero ordenamos por fecha
    agg_df = agg_df.sort_values(['Branch', 'Product line', 'date_group'])
    
    # Seleccionamos un grupo específico para ejemplo
    sample_data = agg_df[
        (agg_df['Branch'] == 'A') & 
        (agg_df['Product line'] == 'Health and beauty')
    ].reset_index(drop=True)
    
    if len(sample_data) > 10:  # Aseguramos que hay suficientes datos
        X, y = create_sequences(sample_data, target_col='Quantity', seq_length=7)
        print("Forma de X:", X.shape)
        print("Forma de y:", y.shape)
    
if __name__ == "__main__":
    main()
