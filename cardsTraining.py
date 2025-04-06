import os
import torch
from ultralytics import YOLO

if __name__ == "__main__":
    yolo_model = YOLO("yolov8n.pt")
    yolo_model.to("cuda" if torch.cuda.is_available() else "cpu")

    print(os.getcwd())

    yolo_model.train(
        data="C:/Home/YoloTraining/RealYoloPlayingCardDataset/data.yaml",
        epochs=100,
        imgsz=416,
        batch=-1,
        device=0,
        project="cardsTraining",
        name="cardsTraining",
        exist_ok=True
    )

    os.chdir("models")
    yolo_model.export()
    os.chdir("..")