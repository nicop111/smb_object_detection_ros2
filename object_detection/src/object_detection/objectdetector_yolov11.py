import torch
import numpy as np
import os
import pandas as pd
from ultralytics import YOLO
from typing import Tuple, Any


class ObjectDetector:
    def __init__(self, config):
        self.architecture = config["architecture"]
        self.model = config["model"]
        self.model_dir_path = config["model_dir_path"]
        self.checkpoint = config["checkpoint"]
        self.device = config["device"]
        self.confident = config["confident"]
        self.iou = config["iou"]
        self.classes = config["classes"]
        self.multiple_instance = config["multiple_instance"]
        self.detector = None

        if self.architecture == "yolo":
            if self.model_dir_path:
                print("Path: ", os.path.join(self.model_dir_path, self.model + ".pt"))
                print("Device: ", self.device)
                model_path = os.path.join(self.model_dir_path, self.model + ".pt")
                self.detector = YOLO(model_path)
            else:
                print("No model path defined, loading from hub")
                self.detector = YOLO("yolov11n")  # or other YOLOv11 variants

            # Set model parameters
            self.detector.conf = self.confident
            self.detector.iou = self.iou
            if self.classes is not None:
                self.detector.classes = self.classes

    def results_to_pandas(self, results) -> pd.DataFrame:
        """Convert YOLOv11 results to pandas DataFrame in YOLOv5 format"""
        boxes = results[0].boxes
        data = []

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf)
            cls = int(box.cls)
            name = results[0].names[cls]

            data.append(
                {
                    "xmin": xyxy[0],
                    "ymin": xyxy[1],
                    "xmax": xyxy[2],
                    "ymax": xyxy[3],
                    "confidence": conf,
                    "class": name,
                    "name": name,
                }
            )

        return pd.DataFrame(data)

    def filter_detection(self, detection):
        """Detects objects on the given image using set model (YOLO V11)
        Args:
            detection: object infos in Pandas data frame

        Returns:
            detection: one instance with the highest confidence for every class.
        """
        # pick the one with highest confidence for every class.
        detected_objects = []
        row_to_delete = []
        for i in range(len(detection)):
            if detection["class"][i] in detected_objects:
                row_to_delete.append(i)
            else:
                detected_objects.append(detection["class"][i])

        detection = detection.drop(row_to_delete, axis=0)
        detection.reset_index(inplace=True)

        return detection

    def detect(self, image) -> Tuple[pd.DataFrame, np.ndarray]:
        """Detects objects on the given image using YOLOv11
        Args:
            image: numpy matrix, RGB or Gray Scale image

        Returns:
            detection: object infos in Pandas data frame
            detection_image: image with bounding boxes
        """
        if self.architecture == "yolo":
            # Run inference
            results = self.detector(image, verbose=False)

            # Convert results to pandas DataFrame
            detection = self.results_to_pandas(results)

            if not self.multiple_instance:
                detection = self.filter_detection(detection)

            # Get annotated image
            detection_image = results[0].plot()
            detection_image = detection_image[..., ::-1]  # BGR to RGB

            return detection, detection_image
