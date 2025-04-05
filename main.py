from ultralytics import YOLO
import torch
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
import supervision as sv
import cv2
import os
from ultralytics import YOLO

if torch.cuda.is_available():
    print("CUDA is available")
else:
    print("CUDA is not available")
    print(torch.version.cuda)

# ontology = CaptionOntology({"DELL logo": "logo DELL"})
# base_model = GroundedSAM(ontology)

# result = base_model.label(
#   extension=".jpg",
#   input_folder="./datasets/unlabeled",
#   output_folder="./datasets/labeled",
# )

# # show the images

# boxAnnotator = sv.BoxAnnotator()
# labelAnnotator = sv.LabelAnnotator()

# annotationsDataset = sv.DetectionDataset.from_yolo(
#     data_yaml_path="./datasets/labeled/data.yaml",
#     images_directory_path="./datasets/labeled/valid/images",
#     annotations_directory_path="./datasets/labeled/valid/labels",
# )

# save_dir = "./preview"

# for image_path in result.image_paths:
#     image = cv2.imread(image_path)
#     image = boxAnnotator.annotate(scene=image, detections=result.annotations[image_path])
#     #map the class_id to the class_name
#     classes = result.annotations[image_path].class_id.tolist()
#     class_names = [ontology.classes()[x] for x in classes]
#     image = labelAnnotator.annotate(scene=image, detections=result.annotations[image_path], labels=class_names)
#     image_name = image_path.split("\\")[-1]
#     if not cv2.imwrite(save_dir + "/" + image_name, image):
#         print("Could not write image")
#     else:
#         print(f"Saved {image_name} to {save_dir}")


yolo_model = YOLO("yolov8n.pt")
yolo_model.to("cuda" if torch.cuda.is_available() else "cpu")
video_source = cv2.VideoCapture(0)

while True:
    ret, frame = video_source.read()
    if not ret:
        break

    # Perform inference on the frame
    results = yolo_model(frame)

    # Visualize the results
    annotated_frame = results[0].plot()

    # Display the frame with annotations
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

