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