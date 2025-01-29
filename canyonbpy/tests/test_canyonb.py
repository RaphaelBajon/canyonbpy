"""
Unit tests for canyonb and helper functions.
Simple versions.
"""

import numpy as np
import pytest
import xarray as xr
from datetime import datetime
from pathlib import Path
from ..core import canyonb
from ..utils import calculate_decimal_year, adjust_arctic_latitude

@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {
        'gtime': [datetime(2024, 1, 1)],
        'lat': np.array([45.0]),
        'lon': np.array([-20.0]),
        'pres': np.array([100.0]),
        'temp': np.array([15.0]),
        'psal': np.array([35.0]),
        'doxy': np.array([250.0])
    }

@pytest.fixture
def input_data_1d():
    """Sample data (from CANYONB.m) for testing the output values."""
    return {
        'gtime': np.array([datetime(2014, 12, 9, 8, 45, 0)]),
        'lat': np.array([17.6]),
        'lon': np.array([-24.3]),
        'pres': np.array([180]),
        'temp': np.array([16]),
        'psal': np.array([36.1]),
        'doxy': np.array([104]),
    }
    
@pytest.fixture
def input_data_2d():
    """Sample data (from CANYONB.m) for testing the output values, using 2d arrays."""
    return {
        'gtime': np.array([
            [datetime(2014, 1, 1, 0, 0, 0), datetime(2014, 12, 9, 8, 45, 0)],
            [datetime(2014, 1, 1, 0, 0, 0), datetime(2014, 12, 9, 8, 45, 0)]
        ]),
        'lat': np.array([[17.6, 20], [17.6, 20]]),
        'lon': np.array([[-24.3, -15], [-24.3, -15]]),
        'pres': np.array([[180, 190], [180, 190]]),
        'temp': np.array([[16, 15], [16, 15]]),
        'psal': np.array([[36.1, 36.2], [36.1, 36.2]]),
        'doxy': np.array([[104, 170], [104, 120]]),
    }

@pytest.fixture
def wrong_data():
    """Wrong data for testing."""
    return {
        'gtime': [datetime(2024, 1, 1)],
        'lat': np.array([91.0]),
        'lon': np.array([-20.0]),
        'pres': np.array([100.0]),
        'temp': np.array([15.0]),
        'psal': np.array([35.0]),
        'doxy': np.array([250.0])
    }

@pytest.fixture
def nan_data():
    """Wrong data for testing."""
    return {
        'gtime': [datetime(2024, 1, 1)],
        'lat': np.array([91.0]),
        'lon': np.array([-20.0]),
        'pres': np.array([100.0]),
        'temp': np.array([np.nan]),
        'psal': np.array([35.0]),
        'doxy': np.array([250.0])
    }

def test_calculate_decimal_year():
    """Test decimal year calculation."""
    dates = np.array([
        datetime(2024, 1, 1),
        datetime(2024, 6, 30),
        datetime(2024, 12, 31, 23, 59, 59)
    ])
    
    result = calculate_decimal_year(dates)
    
    assert len(result) == 3
    assert result[0] == pytest.approx(2024.0, abs=0.01)
    assert result[1] == pytest.approx(2024.5, abs=0.01)
    assert result[2] == pytest.approx(2025.0, abs=0.01)

def test_calculate_decimal_year_float():
    """Test decimal year calculation with float inputs."""
    years = np.array([2024.0, 2024.5, 2025.0])
    result = calculate_decimal_year(years)
    np.testing.assert_array_equal(result, years)

def test_adjust_arctic_latitude():
    """Test Arctic latitude adjustment."""
    lat = np.array([75.0, 80.0, 60.0])
    lon = np.array([-150.0, -120.0, -90.0])
    
    result = adjust_arctic_latitude(lat, lon)
    
    assert len(result) == 3
    # Point outside Arctic basin should remain unchanged
    assert result[2] == lat[2]
    # Points inside Arctic basin should be adjusted
    assert result[0] != lat[0]
    assert result[1] != lat[1]

def test_adjust_arctic_longitude_conversion():
    """Test longitude conversion in Arctic latitude adjustment."""
    lat = np.array([75.0])
    lon1 = np.array([-150.0])
    lon2 = np.array([210.0])  # Equivalent to -150.0
    
    result1 = adjust_arctic_latitude(lat, lon1)
    result2 = adjust_arctic_latitude(lat, lon2)
    
    np.testing.assert_array_almost_equal(result1, result2)

def test_canyonb_basic(sample_data):
    """Test basic canyonb functionality."""
    result = canyonb(**sample_data)
    
    # Check that we get all expected parameters
    expected_params = ['AT', 'CT', 'pH', 'pCO2', 'NO3', 'PO4', 'SiOH4']
    for param in expected_params:
        assert param in result
        assert f'{param}_ci' in result  # Uncertainty estimate
        assert isinstance(result[param], np.ndarray)

def test_canyonb_value_1d(input_data_1d): 
    mat_answer_1d = {
        'NO3': 17.91522,
        'NO3_ci': 1.32494,
        'PO4': 1.081163,
        'PO4_ci': 0.073566,
        'SiOH4': 5.969813,
        'SiOH4_ci': 2.485283,
        'AT': 2359.331,
        'AT_ci': 9.020,
        'CT': 2197.927,
        'CT_ci': 9.151,
        'pH': 7.866380,
        'pH_ci': 0.022136,
        'pCO2': 637.0937,
        'pCO2_ci': 56.5193,    
    }
    results_1d = canyonb(**input_data_1d)

    for name in ['NO3', 'SiOH4', 'PO4']:
        np.testing.assert_almost_equal(results_1d[name], mat_answer_1d[name], decimal=6)
        np.testing.assert_almost_equal(results_1d[name + '_ci'], mat_answer_1d[name + '_ci'], decimal=6)
       
    for name in ['pH', 'pH_ci']:
        np.testing.assert_almost_equal(results_1d[name], mat_answer_1d[name], decimal=5)
        
    for name in ['AT', 'AT_ci']:
        np.testing.assert_almost_equal(results_1d[name], mat_answer_1d[name], decimal=3)
    
    for name in ['CT', 'pCO2']:
        #print(f'{name}: {mat_answer_1d[name] - results_1d[name]}')
        np.testing.assert_almost_equal(results_1d[name], mat_answer_1d[name], decimal=2) 

def test_canyonb_value_2d(input_data_2d): 
    mat_answer_2d = {
        'NO3': np.array([[17.91522114, 11.10990034],
               [17.91522114, 16.57063762]]),
        'NO3_ci': np.array([[1.3249401 , 0.90928634],
               [1.3249401 , 1.32149058]]),
        'PO4': np.array([[1.08116325, 0.67984347],
               [1.08116325, 1.02467124]]),
        'PO4_ci': np.array([[0.07356576, 0.07092262],
               [0.07356576, 0.07297769]]),
        'SiOH4': np.array([[5.96981323, 3.5214663 ],
               [5.96981323, 5.66877531]]),
        'SiOH4_ci': np.array([[2.48528281, 2.49405517],
               [2.48528281, 2.72257296]]),
        'AT': np.array([[2359.31488649, 2369.78081037],
               [2359.31488649, 2366.7615541 ]]),
        'AT_ci': np.array([[9.01249325, 9.00579569],
               [9.01249325, 9.09570314]]),
        'CT': np.array([[2197.33693084, 2165.49041543],
               [2197.33693084, 2197.37359661]]),
        'CT_ci': np.array([[9.12060476, 9.81062657],
               [9.12060476, 9.65245549]]),
        'pH': np.array([[7.86775554, 7.97252544],
               [7.86775554, 7.89779157]]),
        'pH_ci': np.array([[0.02160347, 0.0188441],
               [0.02160347, 0.02401866]]),
        'pCO2': np.array([[634.0151462 , 484.32989726],
               [634.0151462 , 587.15266557]]),
        'pCO2_ci': np.array([[55.91783328, 41.19324986],
               [55.91783328, 52.90195643]]), 
    }
    
    results_2d = canyonb(**input_data_2d)

    for name in ['NO3', 'SiOH4', 'PO4']:
        np.testing.assert_almost_equal(results_2d[name], mat_answer_2d[name], decimal=6)
        np.testing.assert_almost_equal(results_2d[name + '_ci'], mat_answer_2d[name + '_ci'], decimal=6)
       
    for name in ['pH', 'pH_ci']:
        np.testing.assert_almost_equal(results_2d[name], mat_answer_2d[name], decimal=5)
        
    for name in ['AT', 'AT_ci']:
        np.testing.assert_almost_equal(results_2d[name], mat_answer_2d[name], decimal=3)
    
    for name in ['CT', 'pCO2']:
        #print(f'{name}: {mat_answer_1d[name] - results_1d[name]}')
        np.testing.assert_almost_equal(results_2d[name], mat_answer_2d[name], decimal=2)   
        
def test_canyonb_nan(nan_data):
    """Test basic canyonb functionality."""
    result = canyonb(**nan_data)
    
    # Check that we get all expected parameters
    expected_params = ['AT', 'CT', 'pH', 'pCO2', 'NO3', 'PO4', 'SiOH4']
    for param in expected_params:
        assert param in result
        assert f'{param}_ci' in result  # Uncertainty estimate
        assert isinstance(result[param], np.ndarray)
        assert np.isnan(result[param][0])

def test_canyonb_specific_params(sample_data):
    """Test canyonb with specific parameter selection."""
    params = ['pH', 'AT']
    result = canyonb(**sample_data, param=params)
    
    # Check that we only get requested parameters
    assert 'pH' in result
    assert 'AT' in result
    assert 'NO3' not in result
    
    # Check for uncertainty estimates
    assert 'pH_ci' in result
    assert 'AT_ci' in result

def test_canyonb_with_errors(sample_data):
    """Test canyonb with custom error specifications."""
    result = canyonb(
        **sample_data,
        epres=1.0,
        etemp=0.01,
        epsal=0.01,
        edoxy=0.02
    )
    
    # Results should still have the same structure
    assert 'pH' in result
    assert 'pH_ci' in result

def test_canyonb_array_inputs():
    """Test canyonb with array inputs."""
    data = {
        'gtime': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
        'lat': np.array([45.0, 46.0]),
        'lon': np.array([-20.0, -21.0]),
        'pres': np.array([100.0, 200.0]),
        'temp': np.array([15.0, 16.0]),
        'psal': np.array([35.0, 35.1]),
        'doxy': np.array([250.0, 251.0])
    }
    
    result = canyonb(**data)
    
    # Check that output shape matches input
    assert result['pH'].shape == (2,)

def test_canyonb_shape_mismatch(sample_data):
    """Test handling of mismatched input shapes."""
    invalid_data = sample_data.copy()
    invalid_data['lat'] = np.array([45.0, 46.0])  # Different length
    
    with pytest.raises(ValueError):
        canyonb(**invalid_data)