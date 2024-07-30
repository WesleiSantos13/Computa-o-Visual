import cv2
import numpy as np
from ultralytics import YOLO
import time

# Carregue o modelo treinado
model = YOLO('best.pt')

# Inicializar a captura de vídeo e definir a resolução
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Criar uma imagem em branco para desenhar
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
prev_x, prev_y = None, None

# Definir o intervalo de inferência
last_inference_time = 0
inference_interval = 0.1  # Inferência a cada 100ms

# Defina o intervalo para apagar o canvas
reset_interval = 20  # Tempo em segundos para apagar o canvas
last_reset_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    if current_time - last_inference_time >= inference_interval:
        # Execute a inferência
        results = model(frame)
        last_inference_time = current_time

        # Processar as detecções
        detections = results[0].boxes if len(results) > 0 else []
        for det in detections:
            xyxy = det.xyxy[0].tolist()
            if len(xyxy) >= 4:
                x_min, y_min, x_max, y_max = xyxy
                x_centro = int((x_min + x_max) / 2)
                y_centro = int((y_min + y_max) / 2)
                
                # Desenhar um retângulo ao redor da área detectada
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
                
                # Ajuste a posição de escrita para um pouco acima do centro detectado
                offset_y = 10  # Ajuste para mover a escrita acima do centro
                y_centro_adjusted = y_centro - offset_y

                # Desenhar no canvas
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (x_centro, y_centro_adjusted), (255, 255, 255), 4)
                prev_x, prev_y = x_centro, y_centro_adjusted

    # Verificar se é hora de resetar o canvas
    if current_time - last_reset_time >= reset_interval:
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        last_reset_time = current_time

    # Combinar a imagem original com o canvas
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Mostra a imagem processada
    cv2.imshow('Dedo Indicador', combined)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
