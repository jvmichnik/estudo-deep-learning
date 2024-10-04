from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np


model = YOLO("yolov8n.pt")
source = "ortofoto.jpg"
results = model(source)

for result in results:
    plot_image = result.plot()
    cv2.imwrite('new_image.jpg', plot_image)
    
print("done")