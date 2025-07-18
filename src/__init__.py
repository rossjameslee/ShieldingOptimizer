"""
ShieldingOptimizer - ML-based optimization of nuclear reactor shielding

A production-quality implementation for optimizing molten salt microreactor shielding
using machine learning and mathematical optimization techniques.
"""

__version__ = "1.0.0"
__author__ = "Larsen, A., Lee, R., Wilson, C., Hedengren, J.D., Benson, J., Memmott, M."
__email__ = "memmott@byu.edu"

from .optimizer import ShieldingOptimizer
from .models import MLModel
from .data import MaterialDatabase
from .config import Config

__all__ = [
    "ShieldingOptimizer",
    "MLModel", 
    "MaterialDatabase",
    "Config"
] 