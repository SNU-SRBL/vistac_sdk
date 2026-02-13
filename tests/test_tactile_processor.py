"""
Tests for TactileProcessor unified interface.
"""

import unittest
import time
import numpy as np
import os

from vistac_sdk.tactile_processor import TactileProcessor
from vistac_sdk.utils import load_config


class TestTactileProcessorInit(unittest.TestCase):
    """Test initialization and configuration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.model_path = "sensors/D21119/model/nnmodel.pth"
        cls.encoder_path = "models/sparsh_dino_base_encoder.ckpt"
        cls.decoder_path = "models/sparsh_digit_forcefield_decoder.pth"
        
        # Skip tests if models not found
        if not os.path.exists(cls.model_path):
            raise unittest.SkipTest(f"Depth model not found: {cls.model_path}")
        if not os.path.exists(cls.encoder_path):
            raise unittest.SkipTest(f"Force encoder not found: {cls.encoder_path}")
        if not os.path.exists(cls.decoder_path):
            raise unittest.SkipTest(f"Force decoder not found: {cls.decoder_path}")
    
    def test_init_both_enabled(self):
        """Test initialization with both estimators enabled."""
        processor = TactileProcessor(
            model_path=self.model_path,
            enable_depth=True,
            enable_force=True,
            device='cpu'
        )
        
        self.assertIsNotNone(processor.depth_estimator)
        self.assertIsNotNone(processor.force_estimator)
        self.assertTrue(processor.enable_depth)
        self.assertTrue(processor.enable_force)
    
    def test_init_depth_only(self):
        """Test initialization with depth only."""
        processor = TactileProcessor(
            model_path=self.model_path,
            enable_depth=True,
            enable_force=False,
            device='cpu'
        )
        
        self.assertIsNotNone(processor.depth_estimator)
        self.assertIsNone(processor.force_estimator)
        self.assertTrue(processor.enable_depth)
        self.assertFalse(processor.enable_force)
    
    def test_init_force_only(self):
        """Test initialization with force only."""
        processor = TactileProcessor(
            model_path=None,
            enable_depth=False,
            enable_force=True,
            device='cpu'
        )
        
        self.assertIsNone(processor.depth_estimator)
        self.assertIsNotNone(processor.force_estimator)
        self.assertFalse(processor.enable_depth)
        self.assertTrue(processor.enable_force)
    
    def test_init_depth_no_model_path(self):
        """Test that depth requires model_path."""
        with self.assertRaises(ValueError) as cm:
            TactileProcessor(
                model_path=None,
                enable_depth=True,
                enable_force=False
            )
        self.assertIn("model_path required", str(cm.exception))
    
    def test_set_ppmm(self):
        """Test setting ppmm."""
        processor = TactileProcessor(
            model_path=self.model_path,
            enable_depth=True,
            enable_force=False,
            device='cpu'
        )
        
        processor.set_ppmm(25.0)
        self.assertEqual(processor.ppmm, 25.0)

    def test_sensor_yaml_force_keys(self):
        """Verify sensor YAML contains new force keys (`force_field_baseline`, `force_vector_scale`)."""
        cfg = load_config(config_path='sensors/D21242/D21242.yaml')
        self.assertIn('force', cfg)
        force_cfg = cfg['force']
        # force_field_baseline is runtime-only; YAML should not contain it
        self.assertNotIn('force_field_baseline', force_cfg)
        self.assertIn('force_vector_scale', force_cfg)
        self.assertEqual(force_cfg['force_vector_scale'], [2.0, 2.0, 2.0])

class TestTactileProcessorBackground(unittest.TestCase):
    """Test background loading."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.model_path = "sensors/D21119/model/nnmodel.pth"
        cls.encoder_path = "models/sparsh_dino_base_encoder.ckpt"
        cls.decoder_path = "models/sparsh_digit_forcefield_decoder.pth"
        
        if not os.path.exists(cls.model_path):
            raise unittest.SkipTest(f"Depth model not found: {cls.model_path}")
        if not os.path.exists(cls.encoder_path):
            raise unittest.SkipTest(f"Force encoder not found: {cls.encoder_path}")
    
    def test_load_background_both(self):
        """Test loading background for both estimators."""
        processor = TactileProcessor(
            model_path=self.model_path,
            enable_depth=True,
            enable_force=True,
            device='cpu'
        )
        
        bg_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        processor.load_background(bg_image)
        
        self.assertTrue(processor.is_background_loaded())
        self.assertIsNotNone(processor.depth_estimator.bg_image)
        self.assertIsNotNone(processor.force_estimator.background)
    
    def test_load_background_depth_only(self):
        """Test loading background for depth only."""
        processor = TactileProcessor(
            model_path=self.model_path,
            enable_depth=True,
            enable_force=False,
            device='cpu'
        )
        
        bg_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        processor.load_background(bg_image)
        
        self.assertTrue(processor.is_background_loaded())
        self.assertIsNotNone(processor.depth_estimator.bg_image)


class TestTactileProcessorProcess(unittest.TestCase):
    """Test selective processing."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.model_path = "sensors/D21119/model/nnmodel.pth"
        
        if not os.path.exists(cls.model_path):
            raise unittest.SkipTest(f"Depth model not found: {cls.model_path}")
        
        # Create synthetic background image (240x320x3 BGR)
        # Use a grayish background typical of tactile sensors
        cls.bg_image = np.ones((240, 320, 3), dtype=np.uint8) * 128
        
        # Create test image (slightly different from background)
        cls.test_image = np.ones((240, 320, 3), dtype=np.uint8) * 130
        cls.ppmm = 25.0
    
    def test_process_depth_only(self):
        """Test processing depth only."""
        processor = TactileProcessor(
            model_path=self.model_path,
            enable_depth=True,
            enable_force=False,
            device='cpu',
            ppmm=self.ppmm
        )
        processor.load_background(self.bg_image)
        
        result = processor.process(self.test_image, outputs=['depth'])
        
        self.assertIn('depth', result)
        self.assertNotIn('force_field', result)
        self.assertNotIn('force_vector', result)
        self.assertEqual(result['depth'].shape, (240, 320))
        self.assertEqual(result['depth'].dtype, np.uint8)
    
    def test_process_multiple_depth_outputs(self):
        """Test processing multiple depth outputs."""
        processor = TactileProcessor(
            model_path=self.model_path,
            enable_depth=True,
            enable_force=False,
            device='cpu',
            ppmm=self.ppmm
        )
        processor.load_background(self.bg_image)
        
        result = processor.process(
            self.test_image,
            outputs=['depth', 'gradient', 'mask']
        )
        
        self.assertIn('depth', result)
        self.assertIn('gradient', result)
        self.assertIn('mask', result)
        self.assertEqual(result['depth'].shape, (240, 320))
        self.assertEqual(result['gradient'].shape, (240, 320, 2))
        self.assertEqual(result['mask'].shape, (240, 320))
    
    def test_process_force_warmup(self):
        """Test force estimation during warmup (returns None)."""
        processor = TactileProcessor(
            model_path=None,
            enable_depth=False,
            enable_force=True,
            device='cpu',
            temporal_stride=5
        )
        processor.load_background(self.bg_image)
        
        # First frame (buffer not ready)
        result = processor.process(
            self.test_image,
            outputs=['force_field', 'force_vector'],
            timestamp=0.0
        )
        
        self.assertIn('force_field', result)
        self.assertIn('force_vector', result)
        self.assertIsNone(result['force_field'])
        self.assertIsNone(result['force_vector'])

    def test_force_field_baseline_via_constructor(self):
        """Ensure TactileProcessor passes force_field_baseline to the estimator."""
        processor = TactileProcessor(
            model_path=None,
            enable_depth=False,
            enable_force=True,
            device='cpu',
            temporal_stride=5,
            force_field_baseline=True
        )
        self.assertTrue(processor.force_estimator.force_field_baseline_enabled)
    
    def test_process_force_after_warmup(self):
        """Test force estimation after warmup."""
        processor = TactileProcessor(
            model_path=None,
            enable_depth=False,
            enable_force=True,
            device='cpu',
            temporal_stride=5
        )
        processor.load_background(self.bg_image)
        
        # Warmup (add 6 frames for stride=5)
        for i in range(6):
            processor.process(
                self.test_image,
                outputs=['force_field'],
                timestamp=float(i)
            )
        
        # 7th frame should have result
        result = processor.process(
            self.test_image,
            outputs=['force_field', 'force_vector'],
            timestamp=6.0
        )
        
        self.assertIsNotNone(result['force_field'])
        self.assertIsNotNone(result['force_vector'])
        self.assertIn('force_vector_physical', result)
        self.assertIn('normal', result['force_field'])
        self.assertIn('shear', result['force_field'])
        self.assertEqual(result['force_field']['normal'].shape, (224, 224))
        self.assertEqual(result['force_field']['shear'].shape, (224, 224, 2))
        self.assertIn('fx', result['force_vector'])
        self.assertIn('fy', result['force_vector'])
        self.assertIn('fz', result['force_vector'])
        self.assertIn('fx', result['force_vector_physical'])
        self.assertIn('fy', result['force_vector_physical'])
        self.assertIn('fz', result['force_vector_physical'])


    
    def test_process_invalid_output_depth_disabled(self):
        """Test error when requesting depth output with depth disabled."""
        processor = TactileProcessor(
            model_path=None,
            enable_depth=False,
            enable_force=True,
            device='cpu'
        )
        
        with self.assertRaises(ValueError) as cm:
            processor.process(self.test_image, outputs=['depth'])
        self.assertIn("enable_depth=False", str(cm.exception))
    
    def test_process_invalid_output_force_disabled(self):
        """Test error when requesting force output with force disabled."""
        processor = TactileProcessor(
            model_path=self.model_path,
            enable_depth=True,
            enable_force=False,
            device='cpu'
        )
        
        with self.assertRaises(ValueError) as cm:
            processor.process(self.test_image, outputs=['force_field'])
        self.assertIn("enable_force=False", str(cm.exception))
    
    def test_process_missing_ppmm(self):
        """Test error when ppmm not provided for depth."""
        processor = TactileProcessor(
            model_path=self.model_path,
            enable_depth=True,
            enable_force=False,
            device='cpu'
        )
        processor.load_background(self.bg_image)
        
        with self.assertRaises(ValueError) as cm:
            processor.process(self.test_image, outputs=['depth'])
        self.assertIn("ppmm must be provided", str(cm.exception))
    
    def test_process_ppmm_override(self):
        """Test ppmm can be overridden in process call."""
        processor = TactileProcessor(
            model_path=self.model_path,
            enable_depth=True,
            enable_force=False,
            device='cpu',
            ppmm=20.0  # Constructor value
        )
        processor.load_background(self.bg_image)
        
        # Override with different value
        result = processor.process(
            self.test_image,
            outputs=['depth'],
            ppmm=25.0
        )
        
        self.assertIn('depth', result)


class TestTactileProcessorThreading(unittest.TestCase):
    """Test threading functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.model_path = "sensors/D21119/model/nnmodel.pth"
        
        if not os.path.exists(cls.model_path):
            raise unittest.SkipTest(f"Depth model not found: {cls.model_path}")
        
        # Create synthetic background
        cls.bg_image = np.ones((240, 320, 3), dtype=np.uint8) * 128
        cls.test_image = np.ones((240, 320, 3), dtype=np.uint8) * 130
    
    def test_start_stop_thread(self):
        """Test starting and stopping thread."""
        processor = TactileProcessor(
            model_path=self.model_path,
            enable_depth=True,
            enable_force=False,
            device='cpu'
        )
        processor.load_background(self.bg_image)
        
        processor.start_thread(outputs=['depth'], ppmm=25.0)
        self.assertIsNotNone(processor._thread)
        self.assertTrue(processor._thread.is_alive())
        
        processor.stop_thread()
        time.sleep(0.1)
        # Thread should no longer be alive after stop
        self.assertFalse(processor._thread.is_alive())
    
    def test_thread_processing(self):
        """Test background thread processes frames."""
        processor = TactileProcessor(
            model_path=self.model_path,
            enable_depth=True,
            enable_force=False,
            device='cpu'
        )
        processor.load_background(self.bg_image)
        
        processor.start_thread(outputs=['depth'], ppmm=25.0)
        
        # Set input frame
        processor.set_input_frame(self.test_image, timestamp=0.0)
        
        # Wait for processing
        time.sleep(0.5)
        
        # Get result
        result = processor.get_latest_result()
        
        processor.stop_thread()
        
        self.assertIn('depth', result)
        self.assertEqual(result['depth'].shape, (240, 320))
    
    def test_thread_multiple_frames(self):
        """Test thread processes multiple frames."""
        processor = TactileProcessor(
            model_path=self.model_path,
            enable_depth=True,
            enable_force=False,
            device='cpu'
        )
        processor.load_background(self.bg_image)
        
        processor.start_thread(outputs=['depth'], ppmm=25.0)
        
        # Send multiple frames
        for i in range(5):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            processor.set_input_frame(frame, timestamp=float(i))
            time.sleep(0.1)
        
        # Get final result
        result = processor.get_latest_result()
        
        processor.stop_thread()
        
        self.assertIn('depth', result)


class TestTactileProcessorDefaultOutputs(unittest.TestCase):
    """Test default output selection."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.model_path = "sensors/D21119/model/nnmodel.pth"
        
        if not os.path.exists(cls.model_path):
            raise unittest.SkipTest(f"Depth model not found: {cls.model_path}")
        
        # Create synthetic background
        cls.bg_image = np.ones((240, 320, 3), dtype=np.uint8) * 128
        cls.test_image = np.ones((240, 320, 3), dtype=np.uint8) * 130
    
    def test_default_outputs_depth_only(self):
        """Test default outputs when only depth enabled."""
        processor = TactileProcessor(
            model_path=self.model_path,
            enable_depth=True,
            enable_force=False,
            device='cpu',
            ppmm=25.0
        )
        processor.load_background(self.bg_image)
        
        # Call process without outputs argument
        result = processor.process(self.test_image)
        
        # Should default to depth
        self.assertIn('depth', result)
        self.assertNotIn('force_field', result)
        self.assertNotIn('force_vector', result)
    
    def test_default_outputs_force_only(self):
        """Test default outputs when only force enabled."""
        processor = TactileProcessor(
            model_path=None,
            enable_depth=False,
            enable_force=True,
            device='cpu'
        )
        processor.load_background(self.bg_image)
        
        # Call process without outputs argument
        result = processor.process(self.test_image, timestamp=0.0)
        
        # Should default to force outputs (None during warmup)
        self.assertIn('force_field', result)
        self.assertIn('force_vector', result)
        self.assertNotIn('depth', result)


if __name__ == '__main__':
    unittest.main()
