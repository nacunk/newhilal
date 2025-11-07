# hilalpy/__init__.py
# Menandakan bahwa folder ini adalah modul Python.


"""
HilalPy - Modul perhitungan visibilitas hilal
Versi sederhana untuk operasi matematika dasar
"""

from . import cond
from . import divide
from . import equa
from . import multiply
from . import subtract
from . import thres

__version__ = "1.0.0"
__author__ = "Hilal Detection Team"

__all__ = [
    'cond',
    'divide',
    'equa',
    'multiply',
    'subtract',
    'thres'
]