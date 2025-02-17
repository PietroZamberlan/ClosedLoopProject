"""
The `pystim.datasets` module includes utilities to load datasets,
including methods to load and fetch datasets.
"""

from .base import get_path
from .utils import fetch, get, load
from .van_hateren import fetch as fetch_van_hateren


__all__ = [
    'checkerboard',
    'grey',
    'van_hateren',
    'get_path',
    'fetch',
    'fetch_van_hateren',
    'get',
    'load',
]


from . import *
