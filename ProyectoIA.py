import cv2
from ultralytics import YOLO


model = YOLO(r'C:\Users\PC\OneDrive\Escritorio\URU\Python\runs\detect\train13\weights\best.pt')

cap = cv2.VideoCapture(1)  #En este caso uso 1 para usar la camara del celular

while True:
    
    ret, frame = cap.read()

    if not ret:
        print("Error al acceder a la cámara")
        break

    
    results = model.predict(frame, imgsz = 640, conf = 0.61)

   
    annotated_frame = results[0].plot()  

    
    cv2.imshow('Detección de Residuos', annotated_frame)

    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()

    


#r'C:\Users\PC\OneDrive\Escritorio\URU\Python\runs\detect\train13\weights\best.pt'