#!/usr/bin/env python3
"""
Test Suite for Video Enhancement Tool
RIPER-Ω v2.6 Compliant Testing
"""

import sys
import os
import pytest
import numpy as np
import torch
import cv2
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_enhancer import VideoEnhancerApp

class TestVideoEnhancerApp:
    """Test suite for VideoEnhancerApp"""
    
    @pytest.fixture
    def app(self):
        """Create app instance for testing"""
        # Mock tkinter to avoid GUI during tests
        with patch('tkinter.Tk'), patch('tkinter.StringVar'), patch('tkinter.BooleanVar'), patch('tkinter.DoubleVar'):
            mock_root = Mock()
            app = VideoEnhancerApp(mock_root)
            return app
    
    def test_gpu_setup(self, app):
        """Test GPU initialization"""
        # Should not raise exceptions
        app.setup_gpu()
        
        if torch.cuda.is_available():
            assert app.device.type == 'cuda'
        else:
            assert app.device.type == 'cpu'
    
    def test_load_model_tap(self, app):
        """Test TAP model loading"""
        with patch.object(app, 'load_tap_model') as mock_load:
            mock_model = Mock(spec=torch.nn.Module)
            mock_load.return_value = mock_model
            
            result = app.load_model("TAP")
            
            assert result is not None
            assert "TAP" in app.models
            mock_load.assert_called_once()
    
    def test_load_model_mvdenoiser(self, app):
        """Test MVDenoiser model loading"""
        with patch.object(app, 'load_mvdenoiser_model') as mock_load:
            mock_model = Mock(spec=torch.nn.Module)
            mock_load.return_value = mock_model
            
            result = app.load_model("MVDenoiser")
            
            assert result is not None
            assert "MVDenoiser" in app.models
            mock_load.assert_called_once()
    
    def test_load_model_darkir(self, app):
        """Test DarkIR model loading"""
        with patch.object(app, 'load_darkir_model') as mock_load:
            mock_model = Mock(spec=torch.nn.Module)
            mock_load.return_value = mock_model
            
            result = app.load_model("DarkIR")
            
            assert result is not None
            assert "DarkIR" in app.models
            mock_load.assert_called_once()
    
    def test_load_model_invalid(self, app):
        """Test loading invalid model name"""
        result = app.load_model("InvalidModel")
        assert result is None
    
    def test_frame_to_tensor(self, app):
        """Test frame to tensor conversion"""
        # Create dummy 1280x720 frame
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        tensor = app.frame_to_tensor(frame)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 720, 1280)  # BCHW format
        assert tensor.dtype == torch.float32
        assert tensor.min() >= 0.0 and tensor.max() <= 1.0
    
    def test_tensor_to_frame(self, app):
        """Test tensor to frame conversion"""
        # Create dummy tensor
        tensor = torch.rand(1, 3, 720, 1280, dtype=torch.float32)
        
        frame = app.tensor_to_frame(tensor)
        
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (720, 1280, 3)
        assert frame.dtype == np.uint8
        assert frame.min() >= 0 and frame.max() <= 255
    
    def test_apply_darkir(self, app):
        """Test DarkIR application"""
        # Mock model
        mock_model = Mock()
        mock_output = torch.rand(1, 3, 720, 1280)
        mock_model.return_value = mock_output
        
        # Input tensor
        input_tensor = torch.rand(1, 3, 720, 1280)
        
        result = app.apply_darkir(mock_model, input_tensor)
        
        assert torch.equal(result, mock_output)
        mock_model.assert_called_once_with(input_tensor)
    
    def test_apply_darkir_error_handling(self, app):
        """Test DarkIR error handling"""
        # Mock model that raises exception
        mock_model = Mock(side_effect=Exception("Model error"))
        input_tensor = torch.rand(1, 3, 720, 1280)
        
        result = app.apply_darkir(mock_model, input_tensor)
        
        # Should return original tensor on error
        assert torch.equal(result, input_tensor)
    
    def test_apply_denoising_tap(self, app):
        """Test TAP denoising"""
        # Setup frame buffer
        for i in range(3):
            frame_tensor = torch.rand(1, 3, 720, 1280)
            app.frame_buffer.append(frame_tensor)
        
        # Mock model
        mock_model = Mock()
        mock_output = torch.rand(1, 3, 720, 1280)
        mock_model.return_value = mock_output
        
        result = app.apply_denoising(mock_model, "TAP")
        
        assert result is not None
        mock_model.assert_called_once()
    
    def test_apply_denoising_mvdenoiser(self, app):
        """Test MVDenoiser denoising"""
        # Add frame to buffer
        frame_tensor = torch.rand(1, 3, 720, 1280)
        app.frame_buffer.append(frame_tensor)
        
        # Mock model
        mock_model = Mock()
        mock_output = torch.rand(1, 3, 720, 1280)
        mock_model.return_value = mock_output
        
        result = app.apply_denoising(mock_model, "MVDenoiser")
        
        assert torch.equal(result, mock_output)
        mock_model.assert_called_once_with(frame_tensor)
    
    def test_auto_select_denoiser(self, app):
        """Test automatic denoiser selection"""
        # Mock video capture
        mock_cap = Mock()
        mock_frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(5)]
        mock_cap.read.side_effect = [(True, frame) for frame in mock_frames]
        mock_cap.get.return_value = 0  # Mock position
        
        result = app.auto_select_denoiser(mock_cap)
        
        assert result in ["TAP", "MVDenoiser"]
    
    def test_vram_monitoring(self, app):
        """Test VRAM monitoring and adaptive window sizing"""
        if torch.cuda.is_available():
            # Test high VRAM scenario
            with patch('torch.cuda.memory_allocated', return_value=2e9):  # 2GB
                app.monitor_vram()
                assert app.window_size == 3
            
            # Test low VRAM scenario
            with patch('torch.cuda.memory_allocated', return_value=1e9):  # 1GB
                app.monitor_vram()
                assert app.window_size == 5
    
    def test_fitness_scoring(self, app):
        """Test fitness scoring system"""
        # Mock model loading with VRAM tracking
        with patch('torch.cuda.memory_allocated', return_value=1e9):  # 1GB
            with patch.object(app, 'load_tap_model', return_value=Mock()):
                app.load_model("TAP")
        
        assert len(app.fitness_scores) > 0
        assert 0 <= app.fitness_scores[-1] <= 1
    
    def test_bias_detection(self, app):
        """Test bias detection in RL selection"""
        # Simulate bias toward TAP
        app.fitness_scores = [0.9] * 12  # High fitness scores
        original_threshold = app.rl_threshold
        
        # Mock cap for auto selection
        mock_cap = Mock()
        mock_cap.read.return_value = (True, np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8))
        mock_cap.get.return_value = 0
        
        app.auto_select_denoiser(mock_cap)
        
        # Threshold should be mutated down due to bias
        assert app.rl_threshold < original_threshold

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
