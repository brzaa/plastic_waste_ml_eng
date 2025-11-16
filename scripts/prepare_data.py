"""
HSV-based Auto-Labeling Script for Bottle Cap Dataset

This script addresses the critical labeling discrepancy: all samples are currently
labeled as class 0, but we need 3 classes (light_blue, dark_blue, others).

Strategy:
1. Read existing YOLO labels (all class 0)
2. Crop each object using bounding box
3. Convert to HSV color space (decouples lighting from color)
4. Classify based on HSV thresholds:
   - Light Blue: Blue hue + High Value (brightness) + Moderate Saturation
   - Dark Blue: Blue hue + Lower Value + High Saturation
   - Others: Non-blue hues (yellow, white, green, trash)
5. Write new labels with corrected class IDs

HSV Color Space Rationale:
- RGB is coupled to lighting conditions (shadow darkens ALL channels)
- HSV separates Hue (pure color) from Value (brightness)
- Blue hue range: 90-130 degrees (in OpenCV's 0-179 scale)
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from rich.console import Console
from rich.progress import track
from tqdm import tqdm

console = Console()
logger = logging.getLogger(__name__)


class HSVAutoLabeler:
    """Automatic labeling using HSV color space analysis."""

    def __init__(
        self,
        blue_hue_range: Tuple[int, int] = (90, 130),
        light_blue_sv: Tuple[int, int, int, int] = (30, 150, 100, 255),
        dark_blue_sv: Tuple[int, int, int, int] = (80, 255, 40, 140),
        center_crop_ratio: float = 0.6,
    ):
        """
        Initialize auto-labeler with HSV thresholds.

        Args:
            blue_hue_range: (min, max) hue values for blue (0-179 in OpenCV)
            light_blue_sv: (s_min, s_max, v_min, v_max) for light blue
            dark_blue_sv: (s_min, s_max, v_min, v_max) for dark blue
            center_crop_ratio: Ratio of center region to sample (avoid edges)
        """
        self.blue_hue_min, self.blue_hue_max = blue_hue_range
        self.light_blue_sv = light_blue_sv
        self.dark_blue_sv = dark_blue_sv
        self.center_crop_ratio = center_crop_ratio

        # Class mapping
        self.CLASS_LIGHT_BLUE = 0
        self.CLASS_DARK_BLUE = 1
        self.CLASS_OTHERS = 2

        logger.info(f"Initialized HSVAutoLabeler with blue hue range: {blue_hue_range}")

    def _get_center_crop(self, image: np.ndarray) -> np.ndarray:
        """
        Extract center region of image to avoid edge artifacts.

        Args:
            image: Input image

        Returns:
            Center-cropped image
        """
        h, w = image.shape[:2]
        crop_h = int(h * self.center_crop_ratio)
        crop_w = int(w * self.center_crop_ratio)

        y1 = (h - crop_h) // 2
        x1 = (w - crop_w) // 2

        return image[y1 : y1 + crop_h, x1 : x1 + crop_w]

    def _apply_circular_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Apply circular mask to focus on cap (they're circular objects).

        Args:
            image: Input image

        Returns:
            Masked image
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        radius = min(w, h) // 2
        cv2.circle(mask, center, radius, 255, -1)

        # Apply mask
        if len(image.shape) == 3:
            mask_3ch = cv2.merge([mask, mask, mask])
            return cv2.bitwise_and(image, mask_3ch)
        return cv2.bitwise_and(image, mask)

    def classify_cap(self, image: np.ndarray, bbox: List[float]) -> int:
        """
        Classify bottle cap color using HSV analysis.

        Args:
            image: Full image (RGB)
            bbox: YOLO format [x_center, y_center, width, height] (normalized)

        Returns:
            Class ID (0: light_blue, 1: dark_blue, 2: others)
        """
        h, w = image.shape[:2]

        # Convert YOLO normalized coords to pixel coords
        x_center = int(bbox[0] * w)
        y_center = int(bbox[1] * h)
        box_w = int(bbox[2] * w)
        box_h = int(bbox[3] * h)

        # Extract bounding box
        x1 = max(0, x_center - box_w // 2)
        y1 = max(0, y_center - box_h // 2)
        x2 = min(w, x_center + box_w // 2)
        y2 = min(h, y_center + box_h // 2)

        cap_region = image[y1:y2, x1:x2]

        if cap_region.size == 0:
            logger.warning("Empty bounding box, defaulting to 'others'")
            return self.CLASS_OTHERS

        # Apply circular mask (caps are circular)
        cap_region = self._apply_circular_mask(cap_region)

        # Get center region (avoid edge artifacts from shadows/reflections)
        center_region = self._get_center_crop(cap_region)

        # Convert to HSV
        hsv = cv2.cvtColor(center_region, cv2.COLOR_RGB2HSV)

        # Calculate mean HSV values (ignoring black pixels from mask)
        mask = cv2.cvtColor(center_region, cv2.COLOR_RGB2GRAY) > 10
        if not np.any(mask):
            return self.CLASS_OTHERS

        mean_h = np.mean(hsv[:, :, 0][mask])
        mean_s = np.mean(hsv[:, :, 1][mask])
        mean_v = np.mean(hsv[:, :, 2][mask])

        # Classification logic
        is_blue_hue = self.blue_hue_min <= mean_h <= self.blue_hue_max

        if is_blue_hue:
            # Distinguish light vs dark blue using Saturation and Value
            s_min, s_max, v_min, v_max = self.light_blue_sv
            if s_min <= mean_s <= s_max and v_min <= mean_v <= v_max:
                return self.CLASS_LIGHT_BLUE

            s_min, s_max, v_min, v_max = self.dark_blue_sv
            if s_min <= mean_s <= s_max and v_min <= mean_v <= v_max:
                return self.CLASS_DARK_BLUE

            # Blue hue but doesn't fit criteria, likely edge case
            # Classify based on value: high V = light, low V = dark
            return self.CLASS_LIGHT_BLUE if mean_v > 120 else self.CLASS_DARK_BLUE

        # Not blue (yellow, white, green, trash)
        return self.CLASS_OTHERS

    def process_dataset(
        self, image_dir: Path, label_dir: Path, output_dir: Path
    ) -> Tuple[int, dict]:
        """
        Process entire dataset and generate new labels.

        Args:
            image_dir: Directory containing images
            label_dir: Directory containing original YOLO labels
            output_dir: Directory to save new labels

        Returns:
            (total_objects_processed, class_distribution)
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all label files
        label_files = list(label_dir.glob("*.txt"))
        console.print(f"[cyan]Found {len(label_files)} label files[/cyan]")

        total_objects = 0
        class_counts = {0: 0, 1: 0, 2: 0}

        for label_file in track(label_files, description="Processing labels..."):
            # Find corresponding image
            image_path = image_dir / f"{label_file.stem}.jpg"
            if not image_path.exists():
                image_path = image_dir / f"{label_file.stem}.png"
            if not image_path.exists():
                logger.warning(f"Image not found for {label_file.name}, skipping")
                continue

            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Failed to read {image_path}, skipping")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Read labels
            with open(label_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Process each object
            new_labels = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                # Original class (ignore, will be replaced)
                bbox = [float(x) for x in parts[1:5]]

                # Classify using HSV
                new_class = self.classify_cap(image, bbox)
                class_counts[new_class] += 1
                total_objects += 1

                # Write new label
                new_labels.append(f"{new_class} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

            # Save new labels
            output_file = output_dir / label_file.name
            with open(output_file, "w", encoding="utf-8") as f:
                f.writelines(new_labels)

        console.print("\n[bold green]✓ Auto-labeling complete![/bold green]")
        console.print(f"Total objects processed: {total_objects}")
        console.print(f"Class distribution:")
        console.print(f"  Light Blue: {class_counts[0]} ({class_counts[0]/total_objects*100:.1f}%)")
        console.print(f"  Dark Blue:  {class_counts[1]} ({class_counts[1]/total_objects*100:.1f}%)")
        console.print(f"  Others:     {class_counts[2]} ({class_counts[2]/total_objects*100:.1f}%)")

        return total_objects, class_counts


def main():
    """Main entry point for auto-labeling script."""
    parser = argparse.ArgumentParser(
        description="Auto-label bottle caps using HSV color space analysis"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Directory containing images",
    )
    parser.add_argument(
        "--label-dir",
        type=str,
        required=True,
        help="Directory containing original YOLO labels",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save new labels",
    )
    parser.add_argument(
        "--blue-hue-min",
        type=int,
        default=90,
        help="Minimum hue value for blue (0-179)",
    )
    parser.add_argument(
        "--blue-hue-max",
        type=int,
        default=130,
        help="Maximum hue value for blue (0-179)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    console.print("[bold cyan]HSV-Based Auto-Labeling System[/bold cyan]")
    console.print("=" * 60)

    # Initialize labeler
    labeler = HSVAutoLabeler(
        blue_hue_range=(args.blue_hue_min, args.blue_hue_max),
    )

    # Process dataset
    labeler.process_dataset(
        image_dir=Path(args.image_dir),
        label_dir=Path(args.label_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
