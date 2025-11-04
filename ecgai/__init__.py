"""
ECGAI - ECG Classification with Deep Learning
"""
__version__ = "2.0.0"
__author__ = "Your Name"

from ecgai.models.cnn import ECG_CNN1D
from ecgai.config import Config, load_config

__all__ = [
    "ECG_CNN1D",
    "Config",
    "load_config",
]
