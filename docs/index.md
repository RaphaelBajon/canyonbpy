# Welcome to **CanyonbPy**

**`CanyonbPy`** is a Python package implementing the CANYON-B neural network for predicting ocean nutrients and carbonate system variables, just from temperature, salinity and dissolved oxygen.

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
    * Flexible time input: `datetime`, `datetime64`/`Timestamp`, date strings, or decimal years
    * Automatic Arctic latitude adjustments
    * Uncertainty estimates for all predictions
    * `numpy` integration for multi-dimensional data variables

## Installation

```bash
pip install canyonbpy
```

See the [Installation](installation.md) page for details.

## Quick Start

```python
from datetime import datetime
from canyonbpy import canyonb

results = canyonb(
    gtime=[datetime(2024, 1, 1)], lat=[45.0], lon=[-20.0],
    pres=[100.0], temp=[15.0], psal=[35.0], doxy=[250.0],
)
```

`canyonbpy` also registers a `ds.canyonb` accessor, so predictions can run
directly on an `xr.Dataset`: `results = ds.canyonb.predict()`.

Continue with [Getting Started](user-guide/getting-started.md) for a full
walkthrough of both interfaces.