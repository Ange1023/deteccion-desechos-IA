from ultralytics import YOLO

# Carga del modelo preentrenado
model = YOLO("yolov8n.pt")

# Entrenamiento del modelo
results = model.train(data=r'C:\Users\PC\OneDrive\Escritorio\URU\Python\.venv\Include\dataset\data.yaml', epochs=50)

# Imprimir resultados
print(results)

