import numpy as np
from lstm_attention import create_lstm_model

# Prueba rápida con datos sintéticos
seq_length = 7
n_features = 5
n_samples = 100

X = np.random.rand(n_samples, seq_length, n_features)
y = np.random.rand(n_samples, 1)

model, _ = create_lstm_model(seq_length, n_features)
model.summary()
model.fit(X, y, epochs=1, batch_size=8)
print("Modelo LSTM con atención probado exitosamente con datos sintéticos.")
