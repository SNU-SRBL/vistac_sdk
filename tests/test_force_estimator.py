"""Tests for force estimation module."""

import os
import sys
import unittest
import tempfile

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vistac_sdk.vistac_force import (
    SparshEncoder,
    ForceFieldDecoder,
    ForceEstimator,
    _load_encoder_checkpoint
)


class TestSparshEncoder(unittest.TestCase):
    """Test Vision Transformer encoder."""
    
    def setUp(self):
        """Create encoder instance."""
        self.encoder = SparshEncoder(
            img_size=224,
            patch_size=16,
            in_chans=6,
            embed_dim=768,
            depth=12,
            num_heads=12
        )
        self.encoder.eval()
    
    def test_initialization(self):
        """Test encoder initializes correctly."""
        self.assertEqual(self.encoder.num_patches, 196)
        self.assertEqual(len(self.encoder.blocks), 12)
        self.assertEqual(self.encoder.feature_layers, [2, 5, 8, 11])
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        batch_size = 2
        x = torch.randn(batch_size, 6, 224, 224)
        
        with torch.no_grad():
            output = self.encoder(x)
        
        # Output should be [B, N+1, C] where N=196 patches + 1 register token
        self.assertEqual(output.shape, (batch_size, 197, 768))
    
    def test_intermediate_features(self):
        """Test intermediate feature extraction."""
        x = torch.randn(1, 6, 224, 224)
        
        with torch.no_grad():
            _ = self.encoder(x)
            features = self.encoder.get_intermediate_features()
        
        # Should have 4 intermediate features
        self.assertEqual(len(features), 4)
        
        # All should have same shape
        for feat in features:
            self.assertEqual(feat.shape, (1, 197, 768))
    
    def test_patch_embedding(self):
        """Test patch embedding layer."""
        x = torch.randn(1, 6, 224, 224)
        patches = self.encoder.patch_embed(x)
        
        # Should produce 14x14 grid of 768-dim patches
        self.assertEqual(patches.shape, (1, 768, 14, 14))


class TestForceFieldDecoder(unittest.TestCase):
    """Test DPT-style decoder."""
    
    def setUp(self):
        """Create decoder instance."""
        self.decoder = ForceFieldDecoder(embed_dim=768, out_dim=128)
        self.decoder.eval()
    
    def test_initialization(self):
        """Test decoder initializes correctly."""
        self.assertEqual(len(self.decoder.reassembles), 4)
    
    def test_forward_shape(self):
        """Test decoder output shapes."""
        # Create dummy intermediate features
        batch_size = 2
        intermediate_features = [
            torch.randn(batch_size, 197, 768) for _ in range(4)
        ]
        
        with torch.no_grad():
            normal, shear = self.decoder(intermediate_features)
        
        # Check output shapes
        self.assertEqual(normal.shape, (batch_size, 1, 224, 224))
        self.assertEqual(shear.shape, (batch_size, 2, 224, 224))
    
    def test_output_types(self):
        """Test decoder outputs are float tensors and follow Sparsh activations.

        - normal should be in [0, 1] (sigmoid)
        - shear should be bounded by the decoder.scale_flow after tanh
        """
        intermediate_features = [
            torch.randn(1, 197, 768) for _ in range(4)
        ]
        
        with torch.no_grad():
            normal, shear = self.decoder(intermediate_features)
        
        self.assertTrue(normal.dtype == torch.float32)
        self.assertTrue(shear.dtype == torch.float32)

        # normal in [0, 1]
        self.assertGreaterEqual(float(normal.min()), 0.0)
        self.assertLessEqual(float(normal.max()), 1.0)

        # shear should be within [-scale_flow, +scale_flow]
        scale_flow = getattr(self.decoder, 'scale_flow', 20.0)
        self.assertLessEqual(float(torch.abs(shear).max()), scale_flow + 1e-6)


class TestForceEstimator(unittest.TestCase):
    """Test main ForceEstimator interface."""
    
    @classmethod
    def setUpClass(cls):
        """Check if model files exist."""
        cls.encoder_path = 'models/sparsh_dino_base_encoder.ckpt'
        cls.decoder_path = 'models/sparsh_digit_forcefield_decoder.pth'
        
        cls.models_exist = (
            os.path.exists(cls.encoder_path) and 
            os.path.exists(cls.decoder_path)
        )
        
        if not cls.models_exist:
            print("\nWarning: Model files not found. Skipping integration tests.")
            print("Run 'python scripts/download_models.py' to download models.")
    
    def setUp(self):
        """Create test data."""
        self.image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        self.background = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    
    def test_file_not_found_errors(self):
        """Test that missing files raise appropriate errors."""
        with self.assertRaises(FileNotFoundError) as ctx:
            ForceEstimator(
                encoder_path='nonexistent.ckpt',
                decoder_path='nonexistent.pth',
                device='cpu'
            )
        self.assertIn("Run: python scripts/download_models.py", str(ctx.exception))
    
    @unittest.skipUnless(os.path.exists('models/sparsh_dino_base_encoder.ckpt'), 
                         "Model files not available")
    def test_initialization(self):
        """Test estimator initializes with real models."""
        estimator = ForceEstimator(
            encoder_path=self.encoder_path,
            decoder_path=self.decoder_path,
            temporal_stride=5,
            bg_offset=0.5,
            device='cpu'  # Use CPU for testing
        )
        
        self.assertEqual(estimator.device, 'cpu')
        self.assertEqual(estimator.temporal_stride, 5)
        self.assertEqual(estimator.bg_offset, 0.5)
        self.assertIsNotNone(estimator.encoder)
        self.assertIsNotNone(estimator.decoder)
    
    @unittest.skipUnless(os.path.exists('models/sparsh_dino_base_encoder.ckpt'),
                         "Model files not available")
    def test_background_loading(self):
        """Test background image loading."""
        estimator = ForceEstimator(
            encoder_path=self.encoder_path,
            decoder_path=self.decoder_path,
            device='cpu'
        )
        
        estimator.load_background(self.background)
        self.assertIsNotNone(estimator.background)
        self.assertEqual(estimator.background.shape, self.background.shape)
        # Ensure it's a copy, not reference
        self.assertIsNot(estimator.background, self.background)
    
    @unittest.skipUnless(os.path.exists('models/sparsh_dino_base_encoder.ckpt'),
                         "Model files not available")
    def test_temporal_buffer_warmup(self):
        """Test that estimator returns None during warmup."""
        estimator = ForceEstimator(
            encoder_path=self.encoder_path,
            decoder_path=self.decoder_path,
            temporal_stride=5,
            device='cpu'
        )
        estimator.load_background(self.background)
        
        # Add frames until warmup
        for i in range(5):  # Less than stride + 1
            result = estimator.estimate(self.image, timestamp=i * 0.016)
            self.assertIsNone(result, f"Should return None during warmup (frame {i})")
    
    @unittest.skipUnless(os.path.exists('models/sparsh_dino_base_encoder.ckpt'),
                         "Model files not available")
    def test_force_estimation_output_format(self):
        """Test force estimation returns correct format after warmup."""
        estimator = ForceEstimator(
            encoder_path=self.encoder_path,
            decoder_path=self.decoder_path,
            temporal_stride=5,
            device='cpu'
        )
        estimator.load_background(self.background)
        
        # Warmup buffer (stride + 1 frames)
        for i in range(6):
            result = estimator.estimate(self.image, timestamp=i * 0.016)
        
        # Should now have valid output
        self.assertIsNotNone(result)
        self.assertIn('force_field', result)
        self.assertIn('force_vector', result)
        self.assertIn('force_vector_physical', result)
        
        # Check force field structure
        force_field = result['force_field']
        self.assertIn('normal', force_field)
        self.assertIn('shear', force_field)
        
        # Check shapes
        normal = force_field['normal']
        shear = force_field['shear']
        self.assertEqual(normal.shape, (224, 224))
        self.assertEqual(shear.shape, (224, 224, 2))
        
        # Check force vector structure
        force_vector = result['force_vector']
        self.assertIn('fx', force_vector)
        self.assertIn('fy', force_vector)
        self.assertIn('fz', force_vector)
        
        # Check physical vector structure
        phys = result['force_vector_physical']
        self.assertIn('fx', phys)
        self.assertIn('fy', phys)
        self.assertIn('fz', phys)
        
        # Check types (should be Python floats)
        self.assertIsInstance(force_vector['fx'], float)
        self.assertIsInstance(force_vector['fy'], float)
        self.assertIsInstance(force_vector['fz'], float)
        self.assertIsInstance(phys['fx'], float)
        self.assertIsInstance(phys['fy'], float)
        self.assertIsInstance(phys['fz'], float)
    
    @unittest.skipUnless(os.path.exists('models/sparsh_dino_base_encoder.ckpt'),
                         "Model files not available")
    def test_preprocessing_pipeline(self):
        """Test preprocessing produces correct tensor shape."""
        estimator = ForceEstimator(
            encoder_path=self.encoder_path,
            decoder_path=self.decoder_path,
            device='cpu'
        )
        estimator.load_background(self.background)
        
        img_t = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        img_t_minus = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        
        tensor = estimator._preprocess(img_t, img_t_minus)
        
        # Check shape
        self.assertEqual(tensor.shape, (1, 6, 224, 224))
        
        # Check range [0, 1]
        self.assertGreaterEqual(tensor.min().item(), 0.0)
        self.assertLessEqual(tensor.max().item(), 1.0)

    @unittest.skipUnless(os.path.exists('models/sparsh_dino_base_encoder.ckpt'), "Model files not available")
    def test_baseline_subtraction(self):
        """Ensure force_vector baseline (from background) is subtracted from outputs."""
        estimator = ForceEstimator(
            encoder_path=self.encoder_path,
            decoder_path=self.decoder_path,
            device='cpu',
            temporal_stride=5,
        )
        estimator.load_background(self.background)
        # baseline must be stored
        self.assertIn('fx', estimator.force_vector_baseline)
        self.assertIn('fy', estimator.force_vector_baseline)
        self.assertIn('fz', estimator.force_vector_baseline)

        # Clear temporal buffer and compute result without baseline by forcing baseline=0
        estimator.temporal_buffer.clear()
        estimator.force_vector_baseline = {'fx': 0.0, 'fy': 0.0, 'fz': 0.0}
        for i in range(6):
            r_no = estimator.estimate(self.image, timestamp=float(i))
        self.assertIsNotNone(r_no)

        # Reset buffer, restore computed baseline and compute again
        estimator.temporal_buffer.clear()
        estimator.load_background(self.background)  # recompute baseline
        for i in range(6):
            r_yes = estimator.estimate(self.image, timestamp=float(i))
        self.assertIsNotNone(r_yes)

        # Check that baseline subtraction was applied approximately
        b = estimator.force_vector_baseline
        self.assertAlmostEqual(r_no['force_vector']['fx'] - b['fx'], r_yes['force_vector']['fx'], places=5)
        self.assertAlmostEqual(r_no['force_vector']['fy'] - b['fy'], r_yes['force_vector']['fy'], places=5)
        self.assertAlmostEqual(r_no['force_vector']['fz'] - b['fz'], r_yes['force_vector']['fz'], places=5)

    @unittest.skipUnless(os.path.exists('models/sparsh_dino_base_encoder.ckpt'),
                         "Model files not available")
    def test_force_vector_scale_property(self):
        """Verify constructor `force_vector_scale` is stored correctly."""
        est = ForceEstimator(
            encoder_path=self.encoder_path,
            decoder_path=self.decoder_path,
            device='cpu',
            temporal_stride=5,
            force_vector_scale=[2.0, 3.0, 4.0]
        )
        # Preferred attribute
        self.assertTrue(hasattr(est, 'force_vector_scale'))
        import numpy as _np
        self.assertTrue(_np.allclose(est.force_vector_scale, _np.array([2.0, 3.0, 4.0])))

    @unittest.skipUnless(os.path.exists('models/sparsh_dino_base_encoder.ckpt'),
                         "Model files not available")
    def test_force_field_baseline_subtraction(self):
        """Ensure optional runtime baseline subtraction is recorded and applied for force_field."""
        estimator = ForceEstimator(
            encoder_path=self.encoder_path,
            decoder_path=self.decoder_path,
            device='cpu',
            temporal_stride=5,
            force_field_baseline=True,
        )
        estimator.load_background(self.background)

        # Baseline template must be stored when enabled
        self.assertIsNotNone(estimator.force_field_baseline_template)
        self.assertIn('normal', estimator.force_field_baseline_template)
        self.assertIn('shear', estimator.force_field_baseline_template)
        self.assertEqual(estimator.force_field_baseline_template['normal'].shape, (224, 224))
        self.assertEqual(estimator.force_field_baseline_template['shear'].shape, (224, 224, 2))

        # Compute result without template subtraction by zeroing the template
        estimator.temporal_buffer.clear()
        estimator.force_field_baseline_template = {'normal': np.zeros((224, 224)), 'shear': np.zeros((224, 224, 2))}
        for i in range(6):
            r_no = estimator.estimate(self.image, timestamp=float(i))
        self.assertIsNotNone(r_no)

        # Reset buffer, restore computed template and compute again
        estimator.temporal_buffer.clear()
        estimator.load_background(self.background)  # recompute template
        for i in range(6):
            r_yes = estimator.estimate(self.image, timestamp=float(i))
        self.assertIsNotNone(r_yes)

        # The per-pixel subtraction should hold approximately:
        np.testing.assert_allclose(r_no['force_field']['normal'] - estimator.force_field_baseline_template['normal'],
                                   r_yes['force_field']['normal'], atol=1e-5)
        np.testing.assert_allclose(r_no['force_field']['shear'] - estimator.force_field_baseline_template['shear'],
                                   r_yes['force_field']['shear'], atol=1e-5)
    
    def test_preprocessing_without_background_raises_error(self):
        """Test that preprocessing without background raises error."""
        if not self.models_exist:
            self.skipTest("Model files not available")
        
        estimator = ForceEstimator(
            encoder_path=self.encoder_path,
            decoder_path=self.decoder_path,
            device='cpu'
        )
        
        img_t = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        img_t_minus = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        
        with self.assertRaises(ValueError) as ctx:
            estimator._preprocess(img_t, img_t_minus)
        
        self.assertIn("Background not loaded", str(ctx.exception))


class TestModelLoading(unittest.TestCase):
    """Test model checkpoint loading utilities."""
    
    @unittest.skipUnless(os.path.exists('models/sparsh_dino_base_encoder.ckpt'),
                         "Encoder checkpoint not available")
    def test_load_encoder_checkpoint(self):
        """Test encoder checkpoint loading."""
        state_dict = _load_encoder_checkpoint('models/sparsh_dino_base_encoder.ckpt')
        
        # Should be a dict
        self.assertIsInstance(state_dict, dict)
        
        # Should have parameters
        self.assertGreater(len(state_dict), 0)
        
        # Check for expected keys (without 'student_encoder.backbone.' prefix)
        expected_keys = [
            'patch_embed.proj.weight',
            'patch_embed.proj.bias',
            'blocks.0.norm1.weight',
            'register_tokens'
        ]
        
        for key in expected_keys:
            self.assertIn(key, state_dict, f"Expected key '{key}' not found")
        
        # Check patch embedding shape (should be for 6 input channels)
        patch_embed_weight = state_dict['patch_embed.proj.weight']
        self.assertEqual(patch_embed_weight.shape[1], 6, "Patch embedding should accept 6 channels")
        self.assertEqual(patch_embed_weight.shape[0], 768, "Embed dim should be 768")


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestSparshEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestForceFieldDecoder))
    suite.addTests(loader.loadTestsFromTestCase(TestForceEstimator))
    suite.addTests(loader.loadTestsFromTestCase(TestModelLoading))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
