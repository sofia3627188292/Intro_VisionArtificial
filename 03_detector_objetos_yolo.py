from ultralytics import YOLO

# Modelo nano: liviano y rápido
modelo = YOLO("yolov8n.pt")

# source=0 usa la webcam, conf=0.5 filtra detecciones débiles
modelo.predict(source=0, show=True, conf=0.5)
