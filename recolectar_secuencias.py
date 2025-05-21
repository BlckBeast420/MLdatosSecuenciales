import cv2
import mediapipe as mp
import numpy as np
from collections import defaultdict
import time

NUM_FRAMES = 50
SECUENCIAS = []
ETIQUETAS = []
SECUENCIA_ACTUAL = []
conteo_por_letra = defaultdict(int)
letra_mostrada = ""

# Estado de control
esperando_tecla = False
grabando = True

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print("🎥 Recolectando datos. Presiona una letra (A-Z) para etiquetar. ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    frame_data = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                frame_data.append(lm.x)
            for lm in hand_landmarks.landmark:
                frame_data.append(lm.y)

    if len(frame_data) == 42 and grabando:
        SECUENCIA_ACTUAL.append(frame_data)

    if len(SECUENCIA_ACTUAL) >= NUM_FRAMES:
        grabando = False
        esperando_tecla = True
        cv2.putText(frame, "Presiona una letra (A-Z) para guardar", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"Frames: {len(SECUENCIA_ACTUAL)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if letra_mostrada:
        cv2.putText(frame, f"Ultima letra: {letra_mostrada} ({conteo_por_letra[letra_mostrada]})",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow("Recolección de secuencias", frame)
    key = cv2.waitKey(1) & 0xFF

    if esperando_tecla:
        if 65 <= key <= 90 or 97 <= key <= 122:
            letra = chr(key).upper()
            SECUENCIAS.append(SECUENCIA_ACTUAL.copy())
            ETIQUETAS.append(letra)
            conteo_por_letra[letra] += 1
            letra_mostrada = letra
            print(f"✅ Secuencia guardada con etiqueta: {letra} (total: {conteo_por_letra[letra]})")
            SECUENCIA_ACTUAL = []
            esperando_tecla = False
            grabando = True
            time.sleep(3)  # Retraso para permitir reposicionamiento

    if key == 27:
        break

np.save("secuencias.npy", np.array(SECUENCIAS, dtype=object), allow_pickle=True)
np.save("etiquetas.npy", np.array(ETIQUETAS), allow_pickle=True)

print("✅ Datos guardados en 'secuencias.npy' y 'etiquetas.npy'")
print("📊 Conteo por letra:")
for letra, count in conteo_por_letra.items():
    print(f"  {letra}: {count} muestras")

cap.release()
cv2.destroyAllWindows()
