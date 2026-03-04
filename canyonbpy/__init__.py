from .core import canyonb
from .utils import calculate_decimal_year, adjust_arctic_latitude
from .preprocessing import DatasetToNumpy
from . import accessor  # noqa: F401 — registers ds.canyonb on import

__version__ = "0.3.0"
__all__ = [
    "canyonb",
    "DatasetToNumpy",
    "calculate_decimal_year",
    "adjust_arctic_latitude",
]
