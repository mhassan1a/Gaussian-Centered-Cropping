import numpy as np
import pytest
from  cropping import GaussianCrops  # Import the class containing the gcc method


@pytest.fixture
def your_instance():
    # Create an instance of YourClass with necessary parameters
    instance = GaussianCrops()
    instance.crop_percentage = 0.5
    instance.std_scale = 1.0
    instance.min_max = [0, 1]
    return instance


def test_gcc_with_valid_input(your_instance):
    # Test with valid input
    img = np.random.rand(3, 256, 256)  # Assuming a random RGB image of size 256x256
    result = your_instance.gcc(img)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(view, np.ndarray) for view in result)


def test_gcc_with_invalid_crop_percentage(your_instance):
    # Test with invalid crop_percentage
    your_instance.crop_percentage = -0.1
    img = np.random.rand(3, 256, 256)
    with pytest.raises(ValueError):
        your_instance.gcc(img)


def test_gcc_with_invalid_std_scale(your_instance):
    # Test with invalid std_scale
    your_instance.std_scale = -1.0
    img = np.random.rand(3, 256, 256)
    with pytest.raises(ValueError):
        your_instance.gcc(img)


def test_gcc_with_invalid_min_max(your_instance):
    # Test with invalid min_max
    your_instance.min_max = [1, 0]  # Min_max where min > max
    img = np.random.rand(3, 256, 256)
    with pytest.raises(ValueError):
        your_instance.gcc(img)


def test_gcc_with_invalid_image_dimensions(your_instance):
    # Test with invalid image dimensions
    img = np.random.rand(256, 256)  # Assuming a single-channel image
    with pytest.raises(ValueError):
        your_instance.gcc(img)


def test_gcc_with_invalid_image_channels(your_instance):
    # Test with invalid image channels
    img = np.random.rand(4, 256, 256)  # Assuming a 4-channel image
    with pytest.raises(ValueError):
        your_instance.gcc(img)

def test_gcc_with_chw_image_format(your_instance):
    # Test with CHW image format
    img = np.random.rand(3, 256, 256)  # RGB image with CHW format
    result = your_instance.gcc(img)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(view, np.ndarray) for view in result)

def test_gcc_with_hwc_image_format(your_instance):
    # Test with HWC image format
    img = np.random.rand(256, 256, 3)  # RGB image with HWC format
    result = your_instance.gcc(img)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(view, np.ndarray) for view in result)
