"""Tests for utility functions."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from bsort.utils.visualization import draw_detections


@dataclass
class MockDetection:
    """Mock detection for testing."""

    class_id: int
    confidence: float
    bbox: Tuple[int, int, int, int]
    class_name: str


def test_draw_detections_empty():
    """Test drawing with no detections."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = []

    result = draw_detections(image, detections)

    # Should return copy of original image
    assert result.shape == image.shape
    assert np.array_equal(result, image)


def test_draw_detections_single():
    """Test drawing a single detection."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    detection = MockDetection(
        class_id=0, confidence=0.95, bbox=(100, 100, 200, 200), class_name="light_blue"
    )

    result = draw_detections(image, [detection])

    # Image should be modified (not equal to original)
    assert not np.array_equal(result, image)

    # Check that some pixels are non-zero (bounding box drawn)
    assert np.any(result > 0)


def test_draw_detections_multiple():
    """Test drawing multiple detections."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = [
        MockDetection(class_id=0, confidence=0.95, bbox=(50, 50, 100, 100), class_name="light_blue"),
        MockDetection(class_id=1, confidence=0.85, bbox=(200, 200, 300, 300), class_name="dark_blue"),
        MockDetection(class_id=2, confidence=0.75, bbox=(400, 300, 500, 400), class_name="others"),
    ]

    result = draw_detections(image, detections)

    # Image should be modified
    assert not np.array_equal(result, image)

    # Check dimensions preserved
    assert result.shape == image.shape
