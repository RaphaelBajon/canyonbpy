# CANYON-B Python (canyonbpy)

A Python implementation of CANYON-B (CArbonate system and Nutrients concentration from hYdrological properties and Oxygen using Neural networks) based on Bittig et al. (2018).

## Features

- Convert datetime objects to decimal years for CANYON-B calculations
- Adjust Arctic latitude calculations
- Coordinate transformations for CANYON-B neural network predictions

## Installation

You can install canyonbpy using pip:

```bash
pip install canyonbpy
```

## Usage

Here's a simple example of how to use canyonbpy:

```python
from datetime import datetime
from canyonbpy import canyonb


# Prepare your data
data = {
    'gtime': [datetime(2024, 1, 1)],  # Date/time 
    'lat': [45.0],          # Latitude (-90 to 90)
    'lon': [-20.0],         # Longitude (-180 to 180)
    'pres': [100.0],        # Pressure (dbar)
    'temp': [15.0],         # Temperature (°C)
    'psal': [35.0],         # Salinity
    'doxy': [250.0]         # Dissolved oxygen (µmol/kg)
}

# Make predictions
results = canyonb(**data)

# Access results
ph = results['pH']           # pH prediction
ph_error = results['pH_ci']  # pH uncertainty
```

Available parameters for prediction:
- AT: Total Alkalinity
- CT: Total Dissolved Inorganic Carbon
- pH: pH
- pCO2: Partial pressure of CO2
- NO3: Nitrate
- PO4: Phosphate
- SiOH4: Silicate

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite both the original CANYON-B paper and this implementation:

```
@article{bittig2018canyon,
  title={An alternative to static climatologies: Robust estimation of open ocean CO2 variables and nutrient concentrations from T, S, and O2 data using Bayesian neural networks},
  author={Bittig, Henry C and Steinhoff, Tobias and Claustre, Hervé and Körtzinger, Arne and others},
  journal={Frontiers in Marine Science},
  volume={5},
  pages={328},
  year={2018},
  publisher={Frontiers}
}

@misc{canyonbpy2024,
  author = {Raphaël Bajon},
  title = {canyonbpy: A Python implementation of CANYON-B},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/RaphaelBajon/canyonbpy}
}
```