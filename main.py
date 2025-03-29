from ultralytics import YOLO
import torch
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
import supervision as sv
import cv2

if torch.cuda.is_available():
    print("CUDA is available")
else:
    print("CUDA is not available")
    print(torch.version.cuda)

ontology = CaptionOntology({"human": "human"})
# base_model = GroundedSAM(ontology)

# base_model.label(
#   extension=".png",
#   input_folder="./datasets/unlabeled",
#   output_folder="./datasets/labeled",
# )

# show the images

boxAnnotator = sv.BoxAnnotator()
labelAnnotator = sv.LabelAnnotator()

annotationsDataset = sv.DetectionDataset.from_yolo(
    data_yaml_path="./datasets/labeled/data.yaml",
    images_directory_path="./datasets/labeled/train/images",
    annotations_directory_path="./datasets/labeled/train/labels",
)

for image_path in annotationsDataset.image_paths:
    image = cv2.imread(image_path)
    image = boxAnnotator.annotate(scene=image, detections=annotationsDataset.annotations[image_path])
    #map the class_id to the class_name
    classes = annotationsDataset.annotations[image_path].class_id.tolist()
    class_names = [ontology.classes()[x] for x in classes]
    image = labelAnnotator.annotate(scene=image, detections=annotationsDataset.annotations[image_path], labels=class_names)
    image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("image", image)
    cv2.waitKey(0)


