# Advanced Features

## Working with `xarray`

!!! warning "xarray"
    
    This is not yet implemented but will be available soon. `canyonbpy` will work with `xarray` `datasets` and `DataArrays` in future releases.
    
## Custom Weight Files

Using custom neural network weights:

```python
# Directory structure for custom weights
weights_dir/
├── wgts_AT.txt
├── wgts_CT.txt
├── wgts_pH.txt
├── wgts_pCO2.txt
├── wgts_NO3.txt
├── wgts_PO4.txt
└── wgts_SiOH4.txt

# Use custom weights
results = canyonb(**data, weights_dir="/your/path/to/your/weights_dir/")
```

## Error Specification

### Global Errors
```python
results = canyonb(
    **data,
    epres=1.0,    # Custom pressure error
    etemp=0.01,   # Custom temperature error
    epsal=0.01,   # Custom salinity error
    edoxy=0.02    # Custom oxygen error
)
```

### Variable Errors
```python
# Array of oxygen errors
oxygen_errors = np.full_like(data['doxy'], 0.02)
results = canyonb(**data, edoxy=oxygen_errors)
```

## Parameter Selection

Calculate specific parameters:
```python
# Only pH and Total Alkalinity
results = canyonb(**data, param=['pH', 'AT'])

# Only nutrients
results = canyonb(**data, param=['NO3', 'PO4', 'SiOH4'])
```
