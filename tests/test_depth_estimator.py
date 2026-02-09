"""
Tests for the refactored DepthEstimator class.

This test suite verifies:
1. Basic initialization and model loading
2. The estimate() dispatcher method
3. Dict return format
4. All output modes (depth, gradient, pointcloud, mask)
"""

import os
import sys
import pytest
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vistac_sdk.vistac_reconstruct import (
    DepthEstimator,
    BGRXYMLPNet,
    image2bgrxys,
    poisson_dct_neumaan,
    height2pointcloud,
    refine_contact_mask
)


class TestBGRXYMLPNet:
    """Test the neural network architecture."""
    
    def test_initialization(self):
        """Test network can be initialized."""
        net = BGRXYMLPNet()
        assert net is not None
        assert isinstance(net, torch.nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        net = BGRXYMLPNet()
        net.eval()
        
        # Input: batch of BGRXY features
        batch_size = 10
        input_tensor = torch.randn(batch_size, 5)
        
        with torch.no_grad():
            output = net(input_tensor)
        
        # Output should be gradient angles (gx, gy)
        assert output.shape == (batch_size, 2)


class TestDepthEstimator:
    """Test the refactored DepthEstimator class."""
    
    @pytest.fixture
    def model_path(self):
        """Get path to a test model (use D21119 as example)."""
        model_path = "sensors/D21119/model/nnmodel.pth"
        if not os.path.exists(model_path):
            pytest.skip(f"Model not found: {model_path}")
        return model_path
    
    @pytest.fixture
    def bg_image(self):
        """Get background image for testing."""
        bg_path = "sensors/D21119/calibration/background.png"
        if not os.path.exists(bg_path):
            pytest.skip(f"Background image not found: {bg_path}")
        
        import cv2
        bg_image = cv2.imread(bg_path)
        return bg_image
    
    @pytest.fixture
    def estimator(self, model_path):
        """Create a DepthEstimator instance."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return DepthEstimator(model_path, device=device)
    
    def test_initialization(self, model_path):
        """Test DepthEstimator initializes correctly."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        estimator = DepthEstimator(model_path, device=device)
        
        assert estimator is not None
        assert estimator.model_path == model_path
        assert estimator.device == device
        assert estimator.bg_image is None
        assert estimator.gxy_net is not None
    
    def test_initialization_invalid_path(self):
        """Test initialization fails with invalid model path."""
        with pytest.raises(FileNotFoundError):
            DepthEstimator("/invalid/path/model.pth")
    
    def test_load_bg(self, estimator, bg_image):
        """Test background loading."""
        estimator.load_bg(bg_image)
        
        assert estimator.bg_image is not None
        assert hasattr(estimator, 'bg_G')
        assert estimator.bg_G.shape == (bg_image.shape[0], bg_image.shape[1], 2)
    
    def test_estimate_depth(self, estimator, bg_image):
        """Test estimate() method returns depth."""
        estimator.load_bg(bg_image)
        
        # Create a test image (same as background for simplicity)
        test_image = bg_image.copy()
        
        result = estimator.estimate(test_image, outputs=['depth'], ppmm=10.0)
        
        # Check result is a dict
        assert isinstance(result, dict)
        assert 'depth' in result
        
        # Check depth properties
        depth = result['depth']
        assert depth.shape == (test_image.shape[0], test_image.shape[1])
        assert depth.dtype == np.uint8
        assert depth.min() >= 0
        assert depth.max() <= 255
    
    def test_estimate_gradient(self, estimator, bg_image):
        """Test estimate() method returns gradient."""
        estimator.load_bg(bg_image)
        test_image = bg_image.copy()
        
        result = estimator.estimate(test_image, outputs=['gradient'], ppmm=10.0)
        
        assert isinstance(result, dict)
        assert 'gradient' in result
        
        gradient = result['gradient']
        assert gradient.shape == (test_image.shape[0], test_image.shape[1], 2)
        assert gradient.dtype == np.float32 or gradient.dtype == np.float64
    
    def test_estimate_pointcloud(self, estimator, bg_image):
        """Test estimate() method returns pointcloud."""
        estimator.load_bg(bg_image)
        test_image = bg_image.copy()
        
        result = estimator.estimate(test_image, outputs=['pointcloud'], ppmm=10.0)
        
        assert isinstance(result, dict)
        assert 'pointcloud' in result
        
        pc = result['pointcloud']
        assert pc.ndim == 2
        assert pc.shape[1] == 3  # XYZ coordinates
    
    def test_estimate_mask(self, estimator, bg_image):
        """Test estimate() method returns contact mask."""
        estimator.load_bg(bg_image)
        test_image = bg_image.copy()
        
        result = estimator.estimate(test_image, outputs=['mask'], ppmm=10.0)
        
        assert isinstance(result, dict)
        assert 'mask' in result
        
        mask = result['mask']
        assert mask.shape == (test_image.shape[0], test_image.shape[1])
        assert mask.dtype == bool
    
    def test_estimate_multiple_outputs(self, estimator, bg_image):
        """Test estimate() can return multiple outputs at once."""
        estimator.load_bg(bg_image)
        test_image = bg_image.copy()
        
        result = estimator.estimate(
            test_image, 
            outputs=['depth', 'gradient', 'pointcloud', 'mask'], 
            ppmm=10.0
        )
        
        assert isinstance(result, dict)
        assert 'depth' in result
        assert 'gradient' in result
        assert 'pointcloud' in result
        assert 'mask' in result
    
    def test_estimate_requires_ppmm(self, estimator, bg_image):
        """Test estimate() raises error if ppmm not provided."""
        estimator.load_bg(bg_image)
        test_image = bg_image.copy()
        
        with pytest.raises(ValueError, match="ppmm"):
            estimator.estimate(test_image, outputs=['depth'])
    
    def test_old_methods_still_work(self, estimator, bg_image):
        """Test that old methods (get_depth, get_gradient, etc.) still work."""
        estimator.load_bg(bg_image)
        test_image = bg_image.copy()
        ppmm = 10.0
        
        # Test get_depth
        depth = estimator.get_depth(test_image, ppmm)
        assert depth.shape == (test_image.shape[0], test_image.shape[1])
        assert depth.dtype == np.uint8
        
        # Test get_gradient
        gradient = estimator.get_gradient(test_image, ppmm)
        assert gradient.shape == (test_image.shape[0], test_image.shape[1], 2)
        
        # Test get_point_cloud
        pc = estimator.get_point_cloud(test_image, ppmm)
        assert pc.ndim == 2
        assert pc.shape[1] == 3


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_image2bgrxys(self):
        """Test image2bgrxys conversion."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bgrxys = image2bgrxys(image)
        
        assert bgrxys.shape == (100, 100, 5)
        assert bgrxys.dtype == np.float32
        # Check BGR channels are normalized
        assert bgrxys[:, :, :3].min() >= 0
        assert bgrxys[:, :, :3].max() <= 1
    
    def test_poisson_dct_neumaan(self):
        """Test Poisson DCT Neumann solver."""
        # Create simple gradients
        gx = np.random.randn(50, 50).astype(np.float32)
        gy = np.random.randn(50, 50).astype(np.float32)
        
        height = poisson_dct_neumaan(gx, gy)
        
        assert height.shape == (50, 50)
        assert not np.isnan(height).any()
        assert not np.isinf(height).any()
    
    def test_height2pointcloud(self):
        """Test height to pointcloud conversion."""
        height = np.random.randn(50, 50).astype(np.float32)
        ppmm = 10.0
        
        pc = height2pointcloud(height, ppmm)
        
        assert pc.shape == (50 * 50, 3)
        assert pc.dtype == np.float64 or pc.dtype == np.float32
    
    def test_refine_contact_mask(self):
        """Test contact mask refinement."""
        # Create a simple mask with noise
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True  # Large contact region
        mask[10, 10] = True  # Single noisy pixel
        
        refined = refine_contact_mask(mask)
        
        assert refined.shape == mask.shape
        assert refined.dtype == bool
        # Noise should be removed
        assert not refined[10, 10]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
