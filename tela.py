from ultralytics import YOLO
import cv2
import pyautogui

model_path = './runs/detect/train5/weights/best.pt'

model = YOLO(model_path)

while True:
    img = pyautogui.screenshot()
    results = model(img)

    # Process results list
    for result in results:
        # Visualize the results on the frame
        img = result.plot()        

    cv2.imshow("Tela", img)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cv2.destroyAllWindows()
print("desligando")