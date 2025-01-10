# Neural Networks

## Architecture Overview

CANYON-B (and so `canyonb`) uses a committee of neural networks for each predicted parameter. Each network in the committee:

1. Contains 2-3 hidden layers
2. Uses hyperbolic tangent activation functions
3. Is trained on slightly different datasets
4. Has different initializations

!!! success "Network Committee Advantages"

    - Improved prediction stability
    - Better generalization
    - Uncertainty estimation
    - Robustness to outliers

## Training Data

The networks are trained on high-quality reference data from:

1. GO-SHIP cruises
2. Time series stations
3. Research expeditions

## Parameter-Specific Networks

!!! info "2 Parameter-Specific Networks"

    1. Carbonate System
    
        * `AT` 
        * `CT` 
        * `pH`
        * `pCO2` 

    2. Nutrients
    
        * `NO3` 
        * `PO4` 
        * `SiOH4`

## Input Preprocessing

!!! info "Before neural network prediction"

    1. Inputs are normalized
    2. Geographic coordinates are adjusted
    3. Time is converted to decimal years
    4. Arrays are reshaped as needed

## Output Processing

!!! info "After neural network prediction"

    1. Results are denormalized
    2. Uncertainties are calculated
    3. Arrays are reshaped to match input
    4. Results are packaged into dictionary
