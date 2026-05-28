# xarray Accessor

The `canyonb` accessor is registered on `xr.Dataset` automatically when `canyonbpy` is imported.
It provides `predict()` and `content()` as methods directly on any Dataset that contains the
required ocean variables.

```python
import canyonbpy  # registers ds.canyonb
result = ds.canyonb.predict(param=["AT", "pH"])
```

::: canyonbpy.accessor.CanyonBAccessor
    options:
      show_root_heading: true
      show_source: true
      members:
        - predict
        - content
