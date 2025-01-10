# Getting Started

This guide will help you get started with `canyonbpy` and understand its basic functionality.

## Basic Concepts

The `canyonbpy` name comes from CANYON-B (CArbonate system and Nutrients concentration from hYdrological properties and Oxygen using Neural networks) matlab impplementation. It is a method for estimating various ocean parameters using neural networks.

## Your First Prediction

Here's a simple example of how to use `canyonbpy` with its main function `canyonb`:

```python
import numpy as np
from datetime import datetime
from canyonbpy import canyonb

# Prepare your data
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

# Access specific parameters
ph = results['pH']
ph_uncertainty = results['pH_ci']
```

## Understanding the Output

**`canyonb`** can predict the following parameters:

- `AT`: Total Alkalinity
- `CT`: Total Dissolved Inorganic Carbon
- `pH`: pH scale
- `pCO2`: Partial pressure of CO2
- `NO3`: Nitrate
- `PO4`: Phosphate
- `SiOH4`: Silicate

Each parameter comes with uncertainty estimates:

- `_ci`: Total uncertainty
- `_cim`: Measurement uncertainty
- `_cin`: Neural network uncertainty
- `_cii`: Input propagation uncertainty

## Next Steps

- Read the [User Guide - Input](input.md) for detailed informations about the inputs provided to `canyonb` function
- Read the [User Guide - Advanced Features](advanced-features.md) to deepen your knowledge of the library
- Check out the [Examples](../examples/basic-usage.md) for more use cases
- See the [API Reference](../api/core.md) for detailed function documentation