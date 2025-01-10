# Uncertainties

CANYON-B provides comprehensive uncertainty estimates for all predictions. These uncertainties are composed of multiple components and are provided as standard uncertainties (1σ).

## Uncertainty Components

### Measurement Uncertainty (`_cim`)
- Based on input measurement errors
- Configurable via input parameters
- Default values:
  ```python
  epres = 0.5    # Pressure error (dbar)
  etemp = 0.005  # Temperature error (°C)
  epsal = 0.005  # Salinity error
  edoxy = None   # Oxygen error (defaults to 1% of value)
  ```

### Neural Network Uncertainty (`_cin`)

- Derived from committee disagreement
- Represents model uncertainty
- Includes:
    - Committee variance
    - Bias terms
    - Network-specific uncertainties

### Input Propagation Uncertainty (`_cii`)
- How input errors affect prediction
- Calculated using local sensitivity
- Based on error propagation theory

### Total Uncertainty (`_ci`)
Combines all components, and provided as a standard uncertainty:
```
total_uncertainty = sqrt(cim² + cin² + cii²)
```

## Accessing Uncertainties

```python
results = canyonb(**data)

# Total uncertainty
ph_uncertainty = results['pH_ci']

# Component uncertainties
measurement_unc = results['pH_cim']
network_unc = results['pH_cin']
input_unc = results['pH_cii']
```

## Parameter-Specific Considerations

### Carbonate System
- `pH`: Additional term for scale conversion
- `pCO2`: Non-linear error propagation
- `AT`/`CT`: Fixed measurement uncertainty terms

### Nutrients
- Relative errors increase at low concentrations
- Additional terms for seasonal variability
- Regional uncertainty considerations
