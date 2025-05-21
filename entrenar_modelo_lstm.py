import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Cargar datos
secuencias = np.load("secuencias.npy", allow_pickle=True)
etiquetas = np.load("etiquetas.npy", allow_pickle=True)

# Validar dimensiones y convertir a numpy array correctamente
NUM_FRAMES = 50
FEATURES_PER_FRAME = 42

X_limpio = []
y_limpio = []

for i in range(len(secuencias)):
    seq = secuencias[i]
    if len(seq) == NUM_FRAMES and all(len(f) == FEATURES_PER_FRAME for f in seq):
        X_limpio.append(seq)
        y_limpio.append(etiquetas[i])

# Asegurarnos que X sea un array 3D (ej. (16, 50, 42))
X = np.array(X_limpio, dtype=np.float32)
y = np.array(y_limpio)

print(f"✅ Se cargaron {X.shape[0]} secuencias válidas con shape {X.shape}")

# Codificar etiquetas
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Guardar las clases
np.save("clases_lstm.npy", encoder.classes_)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Modelo LSTM
model = Sequential()
model.add(LSTM(64, return_sequences=False, input_shape=(NUM_FRAMES, FEATURES_PER_FRAME)))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(encoder.classes_), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar
print("🚀 Entrenando modelo LSTM...")
model.fit(X_train, y_train, epochs=60, batch_size=16, validation_data=(X_test, y_test))

# Guardar el modelo
model.save("modelo_secuencial_lstm.h5")
print("✅ Modelo LSTM guardado como 'modelo_secuencial_lstm.h5'")

