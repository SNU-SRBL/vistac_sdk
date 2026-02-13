"""
Tests for visualization utilities.
"""

import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from vistac_sdk.viz_utils import (
    plot_gradients,
    force_field_to_rgb,
    visualize_force_field,
    visualize_force_vector
)


class TestPlotGradients(unittest.TestCase):
    """Tests for plot_gradients function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fig, self.ax = plt.subplots()
        self.h, self.w = 240, 320
        
    def tearDown(self):
        """Clean up after tests."""
        plt.close(self.fig)
    
    def test_plot_gradients_separate_arrays(self):
        """Test plotting with separate gx, gy arrays."""
        gx = np.random.randn(self.h, self.w)
        gy = np.random.randn(self.h, self.w)
        
        # Should not raise
        plot_gradients(self.fig, self.ax, gx, gy, mode="rgb")
        
    def test_plot_gradients_combined_array(self):
        """Test plotting with combined [H, W, 2] array."""
        gradient = np.random.randn(self.h, self.w, 2)
        
        # Should not raise
        plot_gradients(self.fig, self.ax, gradient, mode="rgb")
        
    def test_plot_gradients_dict_format(self):
        """Test plotting with dict format."""
        gradient = np.random.randn(self.h, self.w, 2)
        result_dict = {'gradient': gradient}
        
        # Should not raise
        plot_gradients(self.fig, self.ax, result_dict, mode="rgb")
        
    def test_plot_gradients_with_mask(self):
        """Test plotting with mask."""
        gx = np.random.randn(self.h, self.w)
        gy = np.random.randn(self.h, self.w)
        mask = np.random.rand(self.h, self.w) > 0.5
        
        # Should not raise
        plot_gradients(self.fig, self.ax, gx, gy, mask=mask, mode="rgb")
        
    def test_plot_gradients_quiver_mode(self):
        """Test plotting in quiver mode."""
        gx = np.random.randn(self.h, self.w)
        gy = np.random.randn(self.h, self.w)
        
        # Should not raise
        plot_gradients(self.fig, self.ax, gx, gy, mode="quiver")
        
    def test_plot_gradients_invalid_dict(self):
        """Test error handling for invalid dict."""
        invalid_dict = {'wrong_key': np.zeros((10, 10, 2))}
        
        with self.assertRaises(ValueError):
            plot_gradients(self.fig, self.ax, invalid_dict)
            
    def test_plot_gradients_missing_gy(self):
        """Test error handling when gy is missing."""
        gx = np.random.randn(self.h, self.w)
        
        with self.assertRaises(ValueError):
            plot_gradients(self.fig, self.ax, gx)


class TestVisualizeForceField(unittest.TestCase):
    """Tests for visualize_force_field function."""
    
    def test_basic_force_field(self):
        """Test basic force field visualization."""
        normal = np.random.randn(224, 224) * 0.5
        shear = np.random.randn(224, 224, 2) * 0.5
        
        result = visualize_force_field(normal, shear)
        
        self.assertEqual(result.shape, (224, 224, 3))
        self.assertEqual(result.dtype, np.uint8)
        
    def test_force_field_with_overlay(self):
        """Test force field with image overlay."""
        normal = np.random.randn(224, 224) * 0.5
        shear = np.random.randn(224, 224, 2) * 0.5
        overlay = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        result = visualize_force_field(normal, shear, overlay_image=overlay)
        
        self.assertEqual(result.shape, (224, 224, 3))
        self.assertEqual(result.dtype, np.uint8)
        
    def test_force_field_with_grayscale_overlay(self):
        """Test force field with grayscale overlay."""
        normal = np.random.randn(224, 224) * 0.5
        shear = np.random.randn(224, 224, 2) * 0.5
        overlay = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        
        result = visualize_force_field(normal, shear, overlay_image=overlay)
        
        self.assertEqual(result.shape, (224, 224, 3))
        self.assertEqual(result.dtype, np.uint8)
        
    def test_force_field_different_size_overlay(self):
        """Test force field with different size overlay (should resize)."""
        normal = np.random.randn(224, 224) * 0.5
        shear = np.random.randn(224, 224, 2) * 0.5
        overlay = np.random.randint(0, 255, (320, 240, 3), dtype=np.uint8)
        
        result = visualize_force_field(normal, shear, overlay_image=overlay)
        
        self.assertEqual(result.shape, (320, 240, 3))
        self.assertEqual(result.dtype, np.uint8)
        
    def test_force_field_alpha_blending(self):
        """Test alpha blending parameter."""
        normal = np.ones((224, 224)) * 0.5
        shear = np.ones((224, 224, 2)) * 0.5
        overlay = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # With alpha=0, should be mostly overlay (black)
        result_low_alpha = visualize_force_field(normal, shear, overlay_image=overlay, alpha=0.1)
        
        # With alpha=1, should be fully force field
        result_high_alpha = visualize_force_field(normal, shear, overlay_image=overlay, alpha=1.0)
        
        # High alpha should have higher values (more force field visible)
        self.assertGreater(result_high_alpha.mean(), result_low_alpha.mean())

    def test_force_field_normal_preserves_0_1_range(self):
        """Normal already in [0,1] should not be remapped as [-1,1]."""
        normal = np.linspace(0.0, 1.0, 224 * 224, dtype=np.float32).reshape(224, 224)
        shear = np.zeros((224, 224, 2), dtype=np.float32)

        result = visualize_force_field(normal, shear)

        blue = result[..., 2].astype(np.int32)
        self.assertEqual(int(blue.min()), 0)
        self.assertEqual(int(blue.max()), 255)

    def test_force_field_normal_out_of_range_clips(self):
        """Out-of-range normal values should be clipped to [0,1], not remapped."""
        normal = np.array([
            [-0.5, 0.0],
            [0.5, 1.5],
        ], dtype=np.float32)
        shear = np.zeros((2, 2, 2), dtype=np.float32)

        result = visualize_force_field(normal, shear)
        blue = result[..., 2].astype(np.int32)

        # Clipping expectation: [-0.5, 0.0, 0.5, 1.5] -> [0, 0, 127, 255]
        self.assertEqual(int(blue[0, 0]), 0)
        self.assertEqual(int(blue[0, 1]), 0)
        self.assertEqual(int(blue[1, 0]), 127)
        self.assertEqual(int(blue[1, 1]), 255)

    def test_force_field_rgb_channel_mapping(self):
        """Validate (R,G,B) = (Fx,Fy,Fz) mapping."""
        normal = np.zeros((4, 4), dtype=np.float32)
        shear = np.zeros((4, 4, 2), dtype=np.float32)

        # Pixel A: fx=+1, fy=-1, fz=0.25 => R=255, G=0, B~64
        shear[1, 1, 0] = 1.0
        shear[1, 1, 1] = -1.0
        normal[1, 1] = 0.25

        # Pixel B: fx=-1, fy=+1, fz=1.0 => R=0, G=255, B=255
        shear[2, 2, 0] = -1.0
        shear[2, 2, 1] = 1.0
        normal[2, 2] = 1.0

        result = visualize_force_field(normal, shear)

        self.assertEqual(tuple(result[1, 1].tolist()), (255, 0, 63))
        self.assertEqual(tuple(result[2, 2].tolist()), (0, 255, 255))

    def test_force_field_to_rgb_matches_visualize_without_overlay(self):
        """Shared mapping helper should match visualize_force_field base output."""
        normal = np.random.uniform(-0.5, 1.5, (32, 32)).astype(np.float32)
        shear = np.random.uniform(-2.0, 2.0, (32, 32, 2)).astype(np.float32)

        expected = force_field_to_rgb(normal, shear)
        actual = visualize_force_field(normal, shear, overlay_image=None)
        np.testing.assert_array_equal(actual, expected)


class TestVisualizeForceVector(unittest.TestCase):
    """Tests for visualize_force_vector function."""
    
    def test_basic_force_vector(self):
        """Test basic force vector visualization."""
        fx, fy, fz = 0.5, -0.3, 0.8
        image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        
        result = visualize_force_vector(fx, fy, fz, image)
        
        self.assertEqual(result.shape, (240, 320, 3))
        self.assertEqual(result.dtype, np.uint8)
        
    def test_force_vector_grayscale_input(self):
        """Test force vector with grayscale input."""
        fx, fy, fz = 0.5, -0.3, 0.8
        image = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
        
        result = visualize_force_vector(fx, fy, fz, image)
        
        self.assertEqual(result.shape, (240, 320, 3))
        self.assertEqual(result.dtype, np.uint8)
        
    def test_force_vector_zero_forces(self):
        """Test visualization with zero forces."""
        fx, fy, fz = 0.0, 0.0, 0.0
        image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        
        # Should not raise
        result = visualize_force_vector(fx, fy, fz, image)
        
        self.assertEqual(result.shape, image.shape)
        
    def test_force_vector_large_forces(self):
        """Test visualization with large forces."""
        fx, fy, fz = 1.0, 1.0, 1.0
        image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        
        # Should not raise
        result = visualize_force_vector(fx, fy, fz, image)
        
        self.assertEqual(result.shape, image.shape)
        
    def test_force_vector_without_magnitude_text(self):
        """Test visualization without magnitude text."""
        fx, fy, fz = 0.5, -0.3, 0.8
        image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        
        result = visualize_force_vector(fx, fy, fz, image, show_magnitude=False)
        
        self.assertEqual(result.shape, (240, 320, 3))
        
    def test_force_vector_custom_arrow_params(self):
        """Test visualization with custom arrow parameters."""
        fx, fy, fz = 0.5, -0.3, 0.8
        image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        
        result = visualize_force_vector(
            fx, fy, fz, image,
            arrow_scale=100.0,
            arrow_color=(255, 0, 0),
            arrow_thickness=3
        )
        
        self.assertEqual(result.shape, (240, 320, 3))
        
    def test_force_vector_original_not_modified(self):
        """Test that original image is not modified."""
        fx, fy, fz = 0.5, -0.3, 0.8
        image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        image_copy = image.copy()
        
        result = visualize_force_vector(fx, fy, fz, image)
        
        # Original should be unchanged
        np.testing.assert_array_equal(image, image_copy)
        # Result should be different
        self.assertFalse(np.array_equal(result, image))


if __name__ == '__main__':
    unittest.main()
