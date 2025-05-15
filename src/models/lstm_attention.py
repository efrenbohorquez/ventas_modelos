import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Dropout, Attention
from tensorflow.keras.layers import Concatenate, TimeDistributed, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class AttentionLayer(tf.keras.layers.Layer):
    """
    Capa de atención personalizada para destacar períodos relevantes en la secuencia.
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        # Calcular puntuación de atención
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        
        # Obtener pesos de atención
        a = tf.nn.softmax(e, axis=1)
        
        # Aplicar pesos al contexto
        output = x * a
        
        return output  # Solo retornamos el tensor de salida
        
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2]

def create_lstm_model(seq_length, n_features, n_outputs=1):
    """
    Crea un modelo LSTM con mecanismo de atención para predicción de demanda.
    
    Parámetros:
    - seq_length: Longitud de secuencia histórica
    - n_features: Número de características por paso de tiempo
    - n_outputs: Número de valores a predecir (default: 1)
    """
    # Capa de entrada
    inputs = Input(shape=(seq_length, n_features))
    
    # Capas LSTM bidireccionales
    lstm1 = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    lstm1 = Dropout(0.2)(lstm1)
    
    lstm2 = Bidirectional(LSTM(32, return_sequences=True))(lstm1)
    lstm2 = Dropout(0.2)(lstm2)
    
    # Capa de atención
    attention_output, attention_weights = AttentionLayer()(lstm2)
    
    # Pooling global para reducir la dimensionalidad
    gap_layer = GlobalAveragePooling1D()(attention_output)
    
    # Características contextuales adicionales pueden concatenarse aquí
    
    # Capas densas finales
    dense1 = Dense(64, activation='relu')(gap_layer)
    dense1 = Dropout(0.2)(dense1)
    
    dense2 = Dense(32, activation='relu')(dense1)
    
    # Capa de salida
    outputs = Dense(n_outputs, activation='linear')(dense2)
    
    # Definir el modelo
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compilar modelo
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='mean_squared_error',
                  metrics=['mae', 'mse'])
    
    return model, attention_weights

def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=100):
    """
    Entrena el modelo con early stopping y checkpoints.
    """
    # Definir callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model_checkpoint = ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    # Entrenar modelo
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo en el conjunto de prueba.
    """
    # Predecir
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    mse = tf.keras.metrics.mean_squared_error(y_test, y_pred).numpy().mean()
    mae = tf.keras.metrics.mean_absolute_error(y_test, y_pred).numpy().mean()
    
    # Métricas específicas de negocio podrían añadirse aquí
    
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    
    return mse, mae, y_pred
