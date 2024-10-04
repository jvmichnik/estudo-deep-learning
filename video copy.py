from ultralytics import YOLO, solutions
import cv2

model_path = './runs/detect/train/weights/best.pt'

model = YOLO(model_path)

cap = cv2.VideoCapture("d:\Videos\pnct2.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

region_points = [(400, 50), (1200, 50), (1200, 1000), (400, 1000)]
video_writer = cv2.VideoWriter("object_counting_output2.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=region_points,
    names=model.names,
    draw_tracks=True,
    line_thickness=2,
    view_in_counts=False
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False)

    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()