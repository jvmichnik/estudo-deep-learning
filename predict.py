from ultralytics import YOLO
import cv2

# model_path = './runs/segment/train2/weights/best.pt'
model_path = './runs/detect/train5/weights/best.pt'
image_path = "6.jpg"
img = cv2.imread(image_path)
H, W, _ = img.shape

model = YOLO(model_path)
results = model(img)

for result in results:
    result.show()
    # for j, mask in enumerate(result.masks):
    #     mask = mask.numpy() * 255
    #     mask = cv2.resize(mask, (W, H))
    #     cv2.imwrite('new_image.jpg', mask)
    
print("done")