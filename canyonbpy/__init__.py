from .core import canyonb
from .content import co2content
from .utils import calculate_decimal_year, adjust_arctic_latitude
from .preprocessing import DatasetToNumpy
from . import accessor  # noqa: F401 — registers ds.canyonb on import

__version__ = "0.4.0"
__all__ = [
    "canyonb",
    "co2content",
    "DatasetToNumpy",
    "calculate_decimal_year",
    "adjust_arctic_latitude",
]
