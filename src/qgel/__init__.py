"""
QGEL
===

QGEL is a Python package for the creation, simulation and analysis
of graph-oriented quantum walks and their applications.

Provides
  1. A DTQW class for simulation and circuit construction
  2. A Link_Prediction class built upon DTQW

"""

from qgel.dtqw import DTQW, Node
from qgel.lp import Link_Prediction

__all__ = [
  'DTQW',
  'Link_Prediction',
  'Node'
]