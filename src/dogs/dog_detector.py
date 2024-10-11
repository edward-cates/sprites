import os

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from pydantic import BaseModel

class BoundingBox(BaseModel):
    name: str
    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    def overlaps(self, other: "BoundingBox", iou_threshold: float = 0.0) -> bool:
        # Calculate the intersection area
        x_left = max(self.x1, other.x1)
        y_top = max(self.y1, other.y1)
        x_right = min(self.x2, other.x2)
        y_bottom = min(self.y2, other.y2)

        if x_right < x_left or y_bottom < y_top:
            return False

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate the union area
        self_area = (self.x2 - self.x1) * (self.y2 - self.y1)
        other_area = (other.x2 - other.x1) * (other.y2 - other.y1)
        union_area = self_area + other_area - intersection_area

        # Calculate IoU (Intersection over Union)
        iou = intersection_area / union_area

        return iou > iou_threshold

    def calc_percent_covered(self, other: "BoundingBox") -> float:
        """
        Calcluate what percent of my area is covered by the other.
        """
        # Calculate the intersection area
        x_left = max(self.x1, other.x1)
        y_top = max(self.y1, other.y1)
        x_right = min(self.x2, other.x2)
        y_bottom = min(self.y2, other.y2)

        # If there's no overlap, return 0
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate my area
        my_area = self.area

        # Calculate the percentage of my area that is covered
        percent_covered = (intersection_area / my_area)

        return percent_covered

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def interpolate(self, other: "BoundingBox", slider: float) -> "BoundingBox":
        """
        Average each corner.
        :slider: between 0 and 1
        """
        # Ensure slider is between 0 and 1
        slider = max(0, min(1, slider))
        
        # Interpolate each coordinate
        x1 = self.x1 + (other.x1 - self.x1) * slider
        y1 = self.y1 + (other.y1 - self.y1) * slider
        x2 = self.x2 + (other.x2 - self.x2) * slider
        y2 = self.y2 + (other.y2 - self.y2) * slider
        
        # Interpolate the score
        score = self.score + (other.score - self.score) * slider
        
        # Create and return a new BoundingBox
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, name=self.name, score=score)

    @staticmethod
    def deduplicate_boxes(boxes: list["BoundingBox"]) -> list["BoundingBox"]:
        """
        No two boxes should overlap such that one of the boxes is covered > 50%.
        Prioritize size over score.
        """
        result = []
        sorted_boxes = sorted(boxes, key=lambda box: box.area, reverse=True)

        for box in sorted_boxes:
            is_valid = True
            for existing_box in result:
                if box.calc_percent_covered(existing_box) > 0.5:
                    is_valid = False
                    break
            
            if is_valid:
                result.append(box)

        return result


class DogDetector:
    image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
    model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny").to(os.getenv("device", "cuda"))

    def __init__(self):
        pass

    def detect_objects(self, images: list[Image.Image], object_name: str | None, batch_size: int) -> list[list[BoundingBox]]:
        # Chunk the images into batches of size batch_size
        batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]
        
        all_bounding_boxes = []
        
        for batch in batches:
            batch_results = self.detect_objects_batch(batch, object_name)
            all_bounding_boxes.extend(batch_results)
        
        return all_bounding_boxes

    @property
    def device(self):
        return self.model.device

    def detect_objects_batch(self, images: list[Image.Image], object_name: str | None) -> list[list[BoundingBox]]:
        inputs = self.image_processor(images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Process the model outputs
        target_sizes = [(image.height, image.width) for image in images]
        # https://huggingface.co/docs/transformers/en/model_doc/yolos#transformers.YolosForObjectDetection
        results = self.image_processor.post_process_object_detection(outputs, threshold=0.65, target_sizes=target_sizes)

        all_bounding_boxes = []

        for result in results:
            image_boxes = []
            for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                box_label = self.model.config.id2label[label.item()]
                if object_name is None or box_label == object_name:
                    x1, y1, x2, y2 = box.tolist()
                    bounding_box = BoundingBox(
                        name=box_label,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        score=score.item(),
                    )
                    image_boxes.append(bounding_box)
            all_bounding_boxes.append(
                BoundingBox.deduplicate_boxes(image_boxes),
            )

        return all_bounding_boxes

