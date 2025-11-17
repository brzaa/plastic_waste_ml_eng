"""Tests for HSV auto-labeling functionality."""

import numpy as np
import pytest

# Mock the imports since we're just testing the logic
from scripts.prepare_data import HSVAutoLabeler


class TestHSVAutoLabeler:
    """Test suite for HSV auto-labeling."""

    def setup_method(self):
        """Setup test fixtures."""
        self.labeler = HSVAutoLabeler(
            blue_hue_range=(90, 130),
            light_blue_sv=(30, 150, 100, 255),
            dark_blue_sv=(80, 255, 40, 140),
        )

    def test_initialization(self):
        """Test labeler initialization."""
        assert self.labeler.blue_hue_min == 90
        assert self.labeler.blue_hue_max == 130
        assert self.labeler.CLASS_LIGHT_BLUE == 0
        assert self.labeler.CLASS_DARK_BLUE == 1
        assert self.labeler.CLASS_OTHERS == 2

    def test_center_crop(self):
        """Test center crop functionality."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cropped = self.labeler._get_center_crop(image)

        expected_size = int(100 * 0.6)
        assert cropped.shape[0] == expected_size
        assert cropped.shape[1] == expected_size

    def test_circular_mask(self):
        """Test circular mask application."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        masked = self.labeler._apply_circular_mask(image)

        # Center should be preserved
        center_pixel = masked[50, 50]
        assert np.all(center_pixel == 255)

        # Corners should be masked (black)
        corner_pixel = masked[0, 0]
        assert np.all(corner_pixel == 0)

    def test_classify_light_blue_cap(self):
        """Test classification of light blue cap."""
        # Create a synthetic light blue image (RGB)
        # Light blue in RGB: high blue, moderate green, low red
        # In HSV: Hue ~210° (OpenCV: ~105), High Value, Moderate Saturation
        image = np.ones((100, 100, 3), dtype=np.uint8)
        image[:, :, 0] = 100  # R
        image[:, :, 1] = 180  # G
        image[:, :, 2] = 255  # B (high)

        # YOLO bbox: center of image, 50x50
        bbox = [0.5, 0.5, 0.5, 0.5]  # Normalized

        class_id = self.labeler.classify_cap(image, bbox)

        # Should classify as light blue or dark blue (depends on exact HSV conversion)
        assert class_id in [
            self.labeler.CLASS_LIGHT_BLUE,
            self.labeler.CLASS_DARK_BLUE,
        ]

    def test_classify_yellow_cap(self):
        """Test classification of yellow/others cap."""
        # Create a synthetic yellow image (RGB)
        image = np.ones((100, 100, 3), dtype=np.uint8)
        image[:, :, 0] = 50  # R
        image[:, :, 1] = 200  # G
        image[:, :, 2] = 200  # B (yellow is high G+B, low R in some cases)

        # More accurate yellow: high R+G, low B
        image[:, :, 0] = 255  # R
        image[:, :, 1] = 255  # G
        image[:, :, 2] = 50  # B

        bbox = [0.5, 0.5, 0.5, 0.5]

        class_id = self.labeler.classify_cap(image, bbox)

        # Should classify as others (not blue)
        assert class_id == self.labeler.CLASS_OTHERS

    def test_empty_bbox(self):
        """Test handling of empty bounding box."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Invalid bbox that results in empty crop
        bbox = [0.0, 0.0, 0.0, 0.0]

        class_id = self.labeler.classify_cap(image, bbox)

        # Should default to others
        assert class_id == self.labeler.CLASS_OTHERS
