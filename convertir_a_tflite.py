import tensorflow as tf

# Cargar el modelo entrenado
modelo = tf.keras.models.load_model('modelo_gestos.h5')

# Convertir a formato TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
modelo_tflite = converter.convert()

# Guardar archivo .tflite
with open('modelo_gestos.tflite', 'wb') as f:
    f.write(modelo_tflite)

print("✅ Modelo convertido y guardado como 'modelo_gestos.tflite'")
