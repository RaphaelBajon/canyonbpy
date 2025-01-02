import numpy as np
import xarray as xr
from typing import Union, List, Dict, Optional, Tuple
import PyCO2SYS as pyco2


from .utils import calculate_decimal_year, adjust_arctic_latitude, load_weight_file


def canyonb(
    gtime: Union[np.ndarray, List],
    lat: np.ndarray,
    lon: np.ndarray,
    pres: np.ndarray,
    temp: np.ndarray,
    psal: np.ndarray,
    doxy: np.ndarray,
    param: Optional[List[str]] = None,
    epres: Optional[float] = 0.5,
    etemp: Optional[float] = 0.005,
    epsal: Optional[float] = 0.005,
    edoxy: Optional[Union[float, np.ndarray]] = None,
    weights_dir: str = None
) -> Dict[str, xr.DataArray]:
    # TODO order='F' should be checked if needed for 2d-4d arrays as inputs
    # Using xarray:
    # For now 2d-matrix not handle because we are creating xarray matrix at the end with multiple dimension wihtout multiple dimensions names. Should address this issue.
    # This might be needed but not sure since all reshape, flatten and eisum are order='C'(or 'K') for indexing right now. 
    """
    CANYON-B neural network prediction for ocean parameters.
    
    Parameters
    ----------
    gtime : array-like
        Date (UTC) as datetime objects or decimal years
    lat : array-like
        Latitude (-90 to 90)
    lon : array-like
        Longitude (-180 to 180 or 0 to 360)
    pres : array-like
        Pressure (dbar)
    temp : array-like
        In-situ temperature (°C)
    psal : array-like
        Salinity
    doxy : array-like
        Dissolved oxygen (µmol/kg)
    param : list of str, optional
        Parameters to calculate. Default calculates all.
    epres, etemp, epsal : float, optional
        Input errors
    edoxy : float or array-like, optional
        Oxygen input error (default: 1% of doxy)
    weights_dir : str
        Directory containing weight files
        
    Returns
    -------
    Dict[str, xr.DataArray]
        Dictionary containing predictions and uncertainties
    """
    # Convert inputs to numpy arrays
    arrays = [np.asarray(x) for x in (lat, lon, pres, temp, psal, doxy)]
    lat, lon, pres, temp, psal, doxy = arrays
    
    # Get array shape and number of elements
    shape = pres.shape
    nol = pres.size
    
    # Set default edoxy if not provided
    if edoxy is None:
        edoxy = 0.01 * doxy
    
    # Expand scalar error values
    errors = [epres, etemp, epsal, edoxy]
    errors = [np.full(nol, e) if np.isscalar(e) else np.asarray(e).flatten() 
             for e in errors]
    epres, etemp, epsal, edoxy = errors
    
    # Define parameters and their properties
    paramnames = ['AT', 'CT', 'pH', 'pCO2', 'NO3', 'PO4', 'SiOH4']
    inputsigma = np.array([6, 4, 0.005, np.nan, 0.02, 0.02, 0.02])
    betaipCO2 = np.array([-3.114e-05, 1.087e-01, -7.899e+01])
    
    # Adjust pH uncertainty
    inputsigma[2] = np.sqrt(0.005**2 + 0.01**2)
    
    # Set parameters to calculate
    if param is None:
        param = paramnames
    paramflag = np.array([p in param for p in paramnames])
    
    # Prepare input data
    year = calculate_decimal_year(np.asarray(gtime).flatten()) 
    adj_lat = adjust_arctic_latitude(lat.flatten(), lon.flatten()) 
    
    # Create input matrix
    data = np.column_stack([
        year,
        adj_lat / 90,
        np.abs(1 - np.mod(lon.flatten() - 110, 360) / 180), 
        np.abs(1 - np.mod(lon.flatten() - 20, 360) / 180), 
        temp.flatten(), 
        psal.flatten(), 
        doxy.flatten(), 
        pres.flatten() / 2e4 + 1 / ((1 + np.exp(-pres.flatten() / 300))**3) 
    ])
    
    out = {}
    
    # Process each parameter
    for i, param_name in enumerate(paramnames):
        if not paramflag[i]:
            continue
            
        # Load weights
        inwgts = load_weight_file(weights_dir, param_name)
        noparsets = inwgts.shape[1] - 1
        
        # Determine input normalization based on parameter type
        if i > 3:  # nutrients
            ni = data[:, 1:].shape[1]
            ioffset = -1
            mw = inwgts[:ni+1, -1]
            sw = inwgts[ni+1:2*ni+2, -1]
            data_N = (data[:, 1:] - mw[:ni]) / sw[:ni]
        else:  # carbonate system
            ni = data.shape[1]
            ioffset = 0
            mw = inwgts[:ni+1, -1]
            sw = inwgts[ni+1:2*ni+2, -1]
            data_N = (data - mw[:ni]) / sw[:ni]
        
        # Extract weights and prepare arrays
        wgts = inwgts[3, :noparsets]
        betaciw = inwgts[2*ni+2:, -1]
        betaciw = betaciw[~np.isnan(betaciw)]
        
        # Preallocate arrays
        cval = np.full((nol, noparsets), np.nan)
        cvalcy = np.full(noparsets, np.nan)
        inval = np.full((nol, ni, noparsets), np.nan)
        
        # Process each network in committee
        for l in range(noparsets):
            nlayerflag = 1 + bool(inwgts[1, l])
            nl1 = int(inwgts[0, l])
            nl2 = int(inwgts[1, l])
            beta = inwgts[2, l]
            
            # Extract weights
            idx = 4
            w1 = inwgts[idx:idx + nl1 * ni, l].reshape(nl1, ni, order='F') # Here, order='F'needed for sure to proper do the calculation as in matlab version !
            idx += nl1*ni
            b1 = inwgts[idx:idx + nl1, l] 
            idx += nl1
            w2 = inwgts[idx:idx + nl2*nl1, l].reshape(nl2, nl1, order='F')
            idx += nl2*nl1
            b2 = inwgts[idx:idx + nl2, l]
            
            if nlayerflag == 2:
                idx += nl2
                w3 = inwgts[idx:idx + nl2, l].reshape(1, nl2, order='F')
                idx += nl2
                b3 = inwgts[idx:idx + 1, l]
            
            # Forward pass
            a = np.dot(data_N, w1.T) + b1
            if nlayerflag == 1:
                y = np.dot(np.tanh(a), w2.T) + b2
            else:
                b = np.dot(np.tanh(a), w2.T) + b2
                y = np.dot(np.tanh(b), w3.T) + b3
            
            # Store results
            cval[:, l] = y.flatten()
            cvalcy[l] = 1/beta
            
            # Calculate input effects
            x1 = w1[None, :, :] * (1 - np.tanh(a)[:, :, None]**2)
            # jusque-là okay 
            if nlayerflag == 1:
                #inx = np.einsum('ij,jkl->ikl', w2, x1)
                inx = np.einsum('ij,...jk->...ik', w2, x1)[:, 0, :] 
            else:
                x2 = w2[None, :, :] * (1 - np.tanh(b)[:, :, None]**2)
                #inx = np.einsum('ij,jkl,kln->ikn', w3, x2, x1, order='F')
                inx = np.einsum('ij,...jk,...kl->...il', w3, x2, x1)[:, 0, :] 
            inval[:, :, l] = inx
        
        # Denormalization
        cval = cval * sw[ni] + mw[ni]
        cvalcy = cvalcy * sw[ni]**2
        
        # Calculate committee statistics
        V1 = np.sum(wgts)
        V2 = np.sum(wgts**2)
        pred = np.sum(wgts[None, :] * cval, axis=1) / V1
        
        # Calculate uncertainties
        cvalcu = np.sum(wgts[None, :] * (cval - pred[:, None])**2, axis=1) / (V1 - V2/V1)
        cvalcib = np.sum(wgts * cvalcy) / V1
        cvalciw = np.polyval(betaciw, np.sqrt(cvalcu))**2
        
        # Calculate input effects
        inx = np.sum(wgts[None, None, :] * inval, axis=2) / V1
        #inx = sw[ni] / sw[:ni] * inx
        inx = np.tile((sw[ni] / sw[0:ni].T), (nol, 1)) * inx
        
        # Pressure scaling
        ddp = 1/2e4 + 1/((1 + np.exp(-pres.flatten()/300))**4) * np.exp(-pres.flatten()/300)/100 # TODO order='F' ?
        inx[:, 7+ioffset] *= ddp
        
        # Calculate input variance
        error_matrix = np.column_stack([etemp, epsal, edoxy, epres])
        cvalcin = np.sum(inx[:, 4+ioffset:8+ioffset]**2 * error_matrix**2, axis=1)
        
        # Calculate measurement uncertainty
        if i > 3:
            cvalcimeas = (inputsigma[i] * pred)**2
        elif i == 3:
            cvalcimeas = np.polyval(betaipCO2, pred)**2
        else:
            cvalcimeas = inputsigma[i]**2
            
        # Calculate total uncertainty
        uncertainty = np.sqrt(cvalcimeas + cvalcib + cvalciw + cvalcu + cvalcin)
        
        # Create numpy arrays
        out[param_name] = np.reshape(pred, shape)
        out[f'{param_name}_ci'] = np.reshape(uncertainty, shape)
        out[f'{param_name}_cim'] = np.sqrt(cvalcimeas)
        out[f'{param_name}_cin'] = np.reshape(np.sqrt(cvalcib + cvalciw + cvalcu), shape)
        out[f'{param_name}_cii'] = np.reshape(np.sqrt(cvalcin), shape)

        # TODO: should be implemented here with xarray such as 
        #coords = {'depth': pres.reshape(shape)} 
        #out[param_name] = xr.DataArray(
        #    pred.reshape(shape),
        #    coords=coords,
        #    dims=['depth'],
        #    name=param_name
        #)
        
        # pCO2
        if i == 3:
            # ipCO2 = 'DIC' / umol kg-1 -> pCO2 / uatm
            outcalc = pyco2.sys(
                par1=2300, 
                par2=out[param_name], 
                par1_type=1, 
                par2_type=2, 
                salinity=35., 
                temperature=25., 
                temperature_out=np.nan, 
                pressure_out=0., 
                pressure_atmosphere_out=np.nan, 
                total_silicate=0., 
                total_phosphate=0., 
                opt_pH_scale=1., 
                opt_k_carbonic=10., 
                opt_k_bisulfate=1.,
                grads_of=["pCO2"],
                grads_wrt=["par2"],
            ) 

            out[f'{paramnames[i]}'] = outcalc['pCO2']
            
            # epCO2 = dpCO2/dDIC * e'DIC'
            for unc in ['_ci', '_cim', '_cin', '_cii']:
                out[param_name + unc] = outcalc['d_pCO2__d_par2'] * out[param_name + unc] 
                
    return out
