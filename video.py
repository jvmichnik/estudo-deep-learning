from ultralytics import YOLO, solutions
import cv2

model_path = './runs/detect/pnct/weights/best.pt'

model = YOLO(model_path)

cap = cv2.VideoCapture("d:\Videos\pnct.mp4")


while cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
      break

    frame_small = cv2.resize(frame, (1280, 720))        
    
    results = model(frame_small)
    # Process results list
    for result in results:
        # Visualize the results on the frame
        img = result.plot()        

    cv2.imshow("Tela", img)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("desligando")