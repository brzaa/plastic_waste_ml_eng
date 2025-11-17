"""Visualization utilities for detection results."""

from typing import List

import cv2
import numpy as np

# Class colors (BGR format for OpenCV)
CLASS_COLORS = {
    "light_blue": (255, 200, 100),  # Light blue
    "dark_blue": (200, 100, 0),  # Dark blue
    "others": (0, 255, 255),  # Yellow
}


def draw_detections(
    image: np.ndarray, detections: List, class_names: List[str] = None
) -> np.ndarray:
    """
    Draw bounding boxes and labels on image.

    Args:
        image: Input image (BGR)
        detections: List of Detection objects
        class_names: List of class names

    Returns:
        Image with drawn detections
    """
    if class_names is None:
        class_names = ["light_blue", "dark_blue", "others"]

    output = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        class_name = det.class_name
        confidence = det.confidence

        # Get color
        color = CLASS_COLORS.get(class_name, (0, 255, 0))

        # Draw bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{class_name} {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Background for text
        cv2.rectangle(
            output,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            color,
            -1,
        )

        # Text
        cv2.putText(
            output,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output
