"""
Unit tests for TemporalBuffer class.

Tests circular buffer behavior, stride handling, and warmup period.
"""

import time
import numpy as np
import pytest
from vistac_sdk.temporal_buffer import TemporalBuffer


class TestTemporalBufferBasics:
    """Test basic TemporalBuffer functionality."""
    
    def test_initialization(self):
        """Test buffer initialization with default and custom parameters."""
        # Default initialization
        buffer = TemporalBuffer()
        assert buffer.max_size == 60
        assert buffer.stride == 5
        assert len(buffer) == 0
        assert not buffer.is_ready()
        
        # Custom initialization
        buffer = TemporalBuffer(max_size=10, stride=3)
        assert buffer.max_size == 10
        assert buffer.stride == 3
    
    def test_invalid_initialization(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError, match="max_size must be >= 1"):
            TemporalBuffer(max_size=0)
        
        with pytest.raises(ValueError, match="stride must be >= 1"):
            TemporalBuffer(stride=0)
        
        with pytest.raises(ValueError, match="stride .* must be < max_size"):
            TemporalBuffer(max_size=5, stride=5)
    
    def test_add_frames(self):
        """Test adding frames to buffer."""
        buffer = TemporalBuffer(max_size=10, stride=3)
        
        # Add first frame
        frame1 = np.zeros((240, 320, 3), dtype=np.uint8)
        buffer.add(frame1)
        
        assert len(buffer) == 1
        assert buffer.frames_added == 1
        assert buffer.last_timestamp is not None
        
        # Add more frames
        for i in range(5):
            frame = np.ones((240, 320, 3), dtype=np.uint8) * i
            buffer.add(frame)
        
        assert len(buffer) == 6
        assert buffer.frames_added == 6


class TestTemporalPairs:
    """Test temporal pair retrieval."""
    
    def test_insufficient_frames(self):
        """Test that get_pair returns None during warmup."""
        buffer = TemporalBuffer(max_size=10, stride=5)
        
        # No frames
        assert buffer.get_pair() is None
        assert not buffer.is_ready()
        
        # Add frames but not enough (need stride + 1)
        for i in range(5):
            frame = np.ones((10, 10), dtype=np.uint8) * i
            buffer.add(frame)
            result = buffer.get_pair()
            assert result is None  # Still not ready (need 6 frames for stride=5)
            assert not buffer.is_ready()
    
    def test_get_pair_after_warmup(self):
        """Test get_pair returns valid pairs after warmup."""
        buffer = TemporalBuffer(max_size=20, stride=5)
        
        # Add 10 frames with distinct values
        frames = []
        for i in range(10):
            frame = np.ones((10, 10), dtype=np.uint8) * i
            frames.append(frame)
            buffer.add(frame)
        
        # After 6th frame (index 5), should be ready
        assert buffer.is_ready()
        
        # Get pair should return (frame_9, frame_4)
        frame_t, frame_t_minus_5 = buffer.get_pair()
        
        assert frame_t is not None
        assert frame_t_minus_5 is not None
        
        # Verify correct frames
        np.testing.assert_array_equal(frame_t, frames[9])
        np.testing.assert_array_equal(frame_t_minus_5, frames[4])
    
    def test_custom_stride(self):
        """Test get_pair with custom stride parameter."""
        buffer = TemporalBuffer(max_size=20, stride=5)
        
        # Add 10 frames
        frames = []
        for i in range(10):
            frame = np.ones((10, 10), dtype=np.uint8) * i
            frames.append(frame)
            buffer.add(frame)
        
        # Get pair with stride=3 (overrides default 5)
        frame_t, frame_t_minus_3 = buffer.get_pair(stride=3)
        
        np.testing.assert_array_equal(frame_t, frames[9])
        np.testing.assert_array_equal(frame_t_minus_3, frames[6])
        
        # Get pair with stride=7
        frame_t, frame_t_minus_7 = buffer.get_pair(stride=7)
        
        np.testing.assert_array_equal(frame_t, frames[9])
        np.testing.assert_array_equal(frame_t_minus_7, frames[2])


class TestCircularBuffer:
    """Test circular buffer behavior."""
    
    def test_buffer_overflow(self):
        """Test that old frames are dropped when buffer is full."""
        buffer = TemporalBuffer(max_size=5, stride=2)
        
        # Add more frames than max_size
        for i in range(10):
            frame = np.ones((5, 5), dtype=np.uint8) * i
            buffer.add(frame)
        
        # Buffer should only contain last 5 frames
        assert len(buffer) == 5
        assert buffer.frames_added == 10  # Total added
        
        # Get pair should use frames 9 and 7 (not 0-4)
        frame_t, frame_t_minus_2 = buffer.get_pair(stride=2)
        
        # Latest frame should have value 9
        assert np.all(frame_t == 9)
        # Frame from 2 steps ago should have value 7
        assert np.all(frame_t_minus_2 == 7)
    
    def test_continuous_streaming(self):
        """Test buffer behavior during continuous frame streaming."""
        buffer = TemporalBuffer(max_size=10, stride=3)
        
        # Simulate streaming 50 frames
        for i in range(50):
            frame = np.ones((5, 5), dtype=np.uint8) * (i % 256)
            buffer.add(frame)
            
            if i > 3:  # After warmup
                assert buffer.is_ready()
                pair = buffer.get_pair()
                assert pair is not None
        
        # Buffer should stabilize at max_size
        assert len(buffer) == 10
        assert buffer.frames_added == 50


class TestUtilityMethods:
    """Test utility methods."""
    
    def test_get_latest_frame(self):
        """Test retrieving latest frame."""
        buffer = TemporalBuffer()
        
        # Empty buffer
        assert buffer.get_latest_frame() is None
        
        # Add frames
        frame1 = np.ones((5, 5), dtype=np.uint8)
        frame2 = np.ones((5, 5), dtype=np.uint8) * 2
        
        buffer.add(frame1)
        latest = buffer.get_latest_frame()
        np.testing.assert_array_equal(latest, frame1)
        
        buffer.add(frame2)
        latest = buffer.get_latest_frame()
        np.testing.assert_array_equal(latest, frame2)
    
    def test_clear(self):
        """Test buffer clearing."""
        buffer = TemporalBuffer(max_size=10, stride=3)
        
        # Add frames
        for i in range(5):
            buffer.add(np.ones((5, 5), dtype=np.uint8) * i)
        
        assert len(buffer) == 5
        assert buffer.frames_added == 5
        
        # Clear buffer
        buffer.clear()
        
        assert len(buffer) == 0
        assert buffer.get_pair() is None
        assert not buffer.is_ready()
        
        # Settings preserved
        assert buffer.max_size == 10
        assert buffer.stride == 3
        
        # Statistics preserved
        assert buffer.frames_added == 5
    
    def test_timestamps(self):
        """Test timestamp handling."""
        buffer = TemporalBuffer(max_size=5, stride=2)
        
        # Add frames with custom timestamps
        timestamps = [1.0, 2.0, 3.0, 4.0, 5.0]
        for ts in timestamps:
            frame = np.zeros((5, 5), dtype=np.uint8)
            buffer.add(frame, timestamp=ts)
        
        # Check timestamps
        retrieved_ts = buffer.get_timestamps()
        assert retrieved_ts == timestamps
        assert buffer.last_timestamp == 5.0
    
    def test_frame_rate_estimation(self):
        """Test frame rate estimation."""
        buffer = TemporalBuffer(max_size=10, stride=2)
        
        # Empty buffer
        assert buffer.get_frame_rate() is None
        
        # Add single frame
        buffer.add(np.zeros((5, 5)), timestamp=0.0)
        assert buffer.get_frame_rate() is None
        
        # Add frames at 60 FPS (1/60 = 0.0167 seconds apart)
        for i in range(1, 10):
            buffer.add(np.zeros((5, 5)), timestamp=i / 60.0)
        
        frame_rate = buffer.get_frame_rate()
        assert frame_rate is not None
        assert 59.0 < frame_rate < 61.0  # Should be close to 60 FPS
    
    def test_repr(self):
        """Test string representation."""
        buffer = TemporalBuffer(max_size=10, stride=3)
        
        # Before warmup
        repr_str = repr(buffer)
        assert "TemporalBuffer" in repr_str
        assert "warmup" in repr_str
        assert "stride=3" in repr_str
        
        # After warmup
        for i in range(5):
            buffer.add(np.zeros((5, 5)))
        
        repr_str = repr(buffer)
        assert "ready" in repr_str


class TestFrameCopying:
    """Test that frames are properly copied (not referenced)."""
    
    def test_frame_independence(self):
        """Test that stored frames are independent copies."""
        buffer = TemporalBuffer(max_size=5, stride=2)
        
        # Create frame and add to buffer
        frame = np.ones((5, 5), dtype=np.uint8)
        buffer.add(frame)
        
        # Modify original frame
        frame[:] = 99
        
        # Retrieved frame should be unchanged
        latest = buffer.get_latest_frame()
        assert np.all(latest == 1)  # Original value preserved
        assert not np.all(latest == 99)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_minimum_buffer_size(self):
        """Test buffer with minimum valid size."""
        buffer = TemporalBuffer(max_size=2, stride=1)
        
        buffer.add(np.zeros((5, 5)))
        assert buffer.get_pair() is None
        
        buffer.add(np.ones((5, 5)))
        pair = buffer.get_pair()
        assert pair is not None
    
    def test_large_stride(self):
        """Test with stride close to max_size."""
        buffer = TemporalBuffer(max_size=20, stride=18)
        
        # Need 19 frames for stride=18
        for i in range(18):
            buffer.add(np.ones((5, 5), dtype=np.uint8) * i)
            assert buffer.get_pair() is None
        
        # 19th frame should enable pairs
        buffer.add(np.ones((5, 5), dtype=np.uint8) * 18)
        pair = buffer.get_pair()
        assert pair is not None
    
    def test_different_frame_shapes(self):
        """Test buffer works with different frame shapes."""
        buffer = TemporalBuffer(max_size=10, stride=2)
        
        # Different shapes
        shapes = [(240, 320, 3), (480, 640, 3), (224, 224, 3), (100, 100)]
        
        for shape in shapes:
            buffer.clear()
            for i in range(5):
                frame = np.random.randint(0, 255, shape, dtype=np.uint8)
                buffer.add(frame)
            
            if i > 2:
                pair = buffer.get_pair()
                assert pair is not None
                frame_t, frame_t_minus = pair
                assert frame_t.shape == shape
                assert frame_t_minus.shape == shape


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
