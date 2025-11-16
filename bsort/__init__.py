"""
bsort - Real-time Bottle Cap Detection and Sorting System
Optimized for Raspberry Pi 5 (BCM2712 SoC)
"""

__version__ = "0.1.0"
__author__ = "ML Engineering Team"

from bsort.config import load_config

__all__ = ["load_config", "__version__"]
