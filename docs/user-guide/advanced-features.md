# Advanced Features

## Working with `xarray`

`canyonbpy` registers a **`canyonb` accessor** on :class:`xarray.Dataset` objects.
It is available as `ds.canyonb` as soon as `canyonbpy` is imported — no extra
function import needed.

This follows the same design used by [`argopy`](https://argopy.readthedocs.io),
[`cf_xarray`](https://cf-xarray.readthedocs.io), and other scientific Python
packages that extend xarray.

---

### Quick start

```python
import xarray as xr
import canyonbpy  # accessor ds.canyonb is registered here

# ds must contain: time, latitude, longitude, pressure, temperature, salinity, doxy
results = ds.canyonb.predict()

# results is an xr.Dataset with the same dims/coords as ds
print(results["pH"])       # xr.DataArray
print(results["pH_ci"])    # total uncertainty

# Merge predictions back into the source dataset
ds_enriched = xr.merge([ds, results])
```

### Selecting parameters

```python
results = ds.canyonb.predict(param=["pH", "AT", "NO3"])
```

### Custom error specification

```python
results = ds.canyonb.predict(
    epres=1.0,    # pressure error (dbar)
    etemp=0.01,   # temperature error (°C)
    epsal=0.01,   # salinity error
    edoxy=0.02,   # oxygen error (µmol/kg)
)
```

---

### Custom variable names

If your dataset uses different variable names, pass a `var_map` dict.
Only the keys that differ from the defaults need to be specified.

**Default mapping:**

| `canyonb` argument | Default variable name |
|--------------------|-----------------------|
| `gtime`            | `time`                |
| `lat`              | `latitude`            |
| `lon`              | `longitude`           |
| `pres`             | `pressure`            |
| `temp`             | `temperature`         |
| `psal`             | `salinity`            |
| `doxy`             | `doxy`                |

**Argo BGC delayed-mode example:**

```python
var_map = {
    "temp": "TEMP_ADJUSTED",
    "psal": "PSAL_ADJUSTED",
    "doxy": "DOXY_ADJUSTED",
    "pres": "PRES_ADJUSTED",
    "lat":  "LATITUDE",
    "lon":  "LONGITUDE",
}

results = ds.canyonb.predict(var_map=var_map, param=["pH", "NO3"])
```

---

### Low-level access with `converter()`

When you need to inspect or modify the numpy arrays before running the neural
network, use `ds.canyonb.converter()` to get the underlying
:class:`~canyonbpy.preprocessing.DatasetToNumpy` object:

```python
from canyonbpy import canyonb

conv   = ds.canyonb.converter()          # DatasetToNumpy instance
inputs = conv.to_dict()                  # dict[str, np.ndarray]
shape  = conv.original_shape()           # e.g. (n_prof, n_depth)

results = canyonb(**inputs, param=["pH"])

# Reshape outputs back to the original grid
import numpy as np
ph_grid = results["pH"].reshape(shape)
```

---

## Custom Weight Files

```python
# canyonb (numpy)
results = canyonb(**data, weights_dir="/your/path/to/weights_dir/")

# accessor
results = ds.canyonb.predict(weights_dir="/your/path/to/weights_dir/")
```

## Error Specification

### Global Errors
```python
results = canyonb(
    **data,
    epres=1.0,
    etemp=0.01,
    epsal=0.01,
    edoxy=0.02
)
```

### Variable Errors
```python
oxygen_errors = np.full_like(data['doxy'], 0.02)
results = canyonb(**data, edoxy=oxygen_errors)
```

## Parameter Selection

```python
results = canyonb(**data, param=['pH', 'AT'])
results = canyonb(**data, param=['NO3', 'PO4', 'SiOH4'])
```
