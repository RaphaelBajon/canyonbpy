# CanyonbPy: CANYON-B Python 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18868524.svg)](https://doi.org/10.5281/zenodo.18868524)


A Python implementation of CANYON-B (CArbonate system and Nutrients concentration from hYdrological properties and Oxygen using Neural networks) based on [Bittig et al., 2018](https://doi.org/10.3389/fmars.2018.00328). It was developped from the MATLAB [CANYON-B v1.0](https://github.com/HCBScienceProducts/CANYON-B).

## Features

Calculate macronutrients and carbonate system variables using CANYON-B neural network. 

## Installation

You can install `canyonbpy` using pip:

```bash
pip install canyonbpy
```

## Usage

Here's a simple example of how to use `canyonbpy` with `numpy`:

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
And now directly with `xarray`:

```python
import xarray as xr
import canyonbpy  # accessor ds.canyonb is registered here

# ds must contain: time, latitude, longitude, pressure, temperature, salinity, doxy
ds = xr.Dataset(
    {
        "temperature": (("time", "pressure", "latitude", "longitude"), 16.0 * np.ones((2, 3, 3, 4))), 
        "salinity": (("time", "pressure", "latitude", "longitude"), 36.1 * np.ones((2, 3, 3, 4))),
        "doxy": (("time", "pressure", "latitude", "longitude"), 104 * np.ones((2, 3, 3, 4))),
    },
    coords={
        "time": ("time", [datetime(2014, 12, 9, 8, 45), datetime(2020, 12, 10, 8, 45)]),
        "pressure": ("pressure", np.array([180, 181, 182])),
        "latitude": ("latitude", np.array([17.6, 17.6, 17.6])),
        "longitude": ("longitude", np.array([-24.3, -24.3, -24.3, -24.3])),
    },
)

# Predict with CANYON-B
results = ds.canyonb.predict()

# CONTENT
results_content = ds.canyonb.content()


# results is an xr.Dataset with the same dims/coords as ds
print(results["pH"])       # xr.DataArray
print(results["pH_ci"])    # total uncertainty

# Merge predictions back into the source dataset
ds_enriched = xr.merge([ds, results])
```
Available parameters for prediction:
- `AT`: Total Alkalinity
- `CT`: Total Dissolved Inorganic Carbon
- `pH`: pH
- `pCO2`: Partial pressure of CO2
- `NO3`: Nitrate
- `PO4`: Phosphate
- `SiOH4`: Silicate

## Documentation

Documentation is available [here](https://canyonbpy.readthedocs.io/).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite both the original CANYON-B paper and this implementation with the corresponding version for bug tracking:

``` bibtex
@article{bittig2018canyon,
  title={An alternative to static climatologies: Robust estimation of open ocean CO2 variables and nutrient concentrations from T, S, and O2 data using Bayesian neural networks},
  author={Bittig, Henry C and Steinhoff, Tobias and Claustre, Hervé and Körtzinger, Arne and others},
  journal={Frontiers in Marine Science},
  volume={5},
  pages={328},
  year={2018},
  publisher={Frontiers}, 
  doi={10.3389/fmars.2018.00328},
}

@software{bajon_canyonbpy,
  author    = {Bajon, Raphaël},
  title     = {canyonbpy: A Python implementation of CANYON-B},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18868524},
  url       = {https://doi.org/10.5281/zenodo.18868524}
}
```
