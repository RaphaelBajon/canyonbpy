# Welcome to **CanyonbPy**

**`CanyonbPy`** is a Python package implementing the CANYON-B neural network for predicting ocean parameters. It provides tools for analyzing and predicting various ocean parameters using neural network models.

!!! info "Matlab version: CANYON-B v1.0"
    
    This package has been developped using the `CANYON-B v1.0` Matlab version of the code.

!!! info "Python version: PyCO2SYS v1.8.0+"
    
    The calculations from `canyonbpy` could require `PyCO2SYS` Python package. Here, we used `PyCO2SYS v1.8.0`.

## Features

!!! success "Predict ocean parameters using CANYON-B neural network models"
    * Multi-parameters prediction
        * **Total Alkalinity** (`AT`) in μmol·kg<sup>−1</sup>
        * **Dissolved Inorganic Carbon** (`CT`) in μmol·kg<sup>−1</sup>
        * **`pH`**
        * **`pCO2`** in μatm
        * **Nitrate** (`NO3`) in μmol·kg<sup>−1</sup>
        * **Phosphate** (`PO4`) in μmol·kg<sup>−1</sup>
        * **Silicate** (`SiOH4`) in μmol·kg<sup>−1</sup> 
    * Handle various input formats including datetime and decimal years
    * Automatic Arctic latitude adjustments
    * Uncertainty estimates for all predictions
    * `numpy` integration for multi-dimensional data variables

## Installation

```bash
pip install canyonbpy
```

For more detailed installation instructions, see the [Installation](installation.md) page.

## Main commands

!!! tip "`canyonbpy` main function"

    `canyonb` - performs the CANYON-B calculation.

## Quick Start

In your Python environment with `numpy`, `datetime` and `canyonbpy` libraries:

```python
import numpy as np
from datetime import datetime
from canyonbpy import canyonb

# Sample data
data = {
    'gtime': [datetime(2024, 1, 1)],
    'lat': np.array([45.0]),
    'lon': np.array([-20.0]),
    'pres': np.array([100.0]),
    'temp': np.array([15.0]),
    'psal': np.array([35.0]),
    'doxy': np.array([250.0])
}

# Get predictions
results = canyonb(**data)
```