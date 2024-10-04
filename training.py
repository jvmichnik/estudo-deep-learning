if __name__ == '__main__':
  from ultralytics import YOLO
  import torch

  torch.cuda.set_device(0)
  print("GPU Configurada:", torch.cuda.is_available())

  model = YOLO("yolov8l.pt")
  results = model.train(data="./datasets/pnct_v2/data.yaml", epochs=30, device=0)