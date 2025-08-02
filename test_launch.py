#!/usr/bin/env python3
"""
Quick smoke test for video enhancer application
Tests import, GPU detection, and basic functionality
"""

import sys
import os

def test_imports():
    """Test all critical imports"""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ CUDA version: {torch.version.cuda}")
        
        import cv2
        print(f"✅ OpenCV {cv2.__version__} imported successfully")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__} imported successfully")
        
        from PIL import Image
        print(f"✅ Pillow imported successfully")
        
        # Test video enhancer import
        import video_enhancer
        print("✅ Video enhancer module imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_gpu_setup():
    """Test GPU initialization"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            
            # Test tensor creation on GPU
            test_tensor = torch.rand(1, 3, 256, 256).cuda()
            print(f"✅ GPU tensor creation successful: {test_tensor.shape}")
            print(f"✅ VRAM usage: {torch.cuda.memory_allocated() / 1e9:.3f}GB")
            
            # Test mixed precision
            with torch.cuda.amp.autocast():
                result = test_tensor * 2
            print("✅ Mixed precision (AMP) working")
            
            return True
        else:
            print("⚠️ CUDA not available - CPU mode")
            return True
    except Exception as e:
        print(f"❌ GPU setup error: {e}")
        return False

def test_model_structure():
    """Test model loading structure"""
    try:
        from video_enhancer import VideoEnhancerApp
        import tkinter as tk

        # Mock tkinter for headless testing
        root = tk.Tk()
        root.withdraw()  # Hide window

        app = VideoEnhancerApp(root)
        print("✅ VideoEnhancerApp instantiated successfully")

        # Test model cache
        assert hasattr(app, 'models'), "Model cache not found"
        assert isinstance(app.models, dict), "Model cache not a dictionary"
        print("✅ Model cache structure verified")

        # Test device setup
        assert hasattr(app, 'device'), "Device not set"
        print(f"✅ Device set to: {app.device}")

        # Test colorization variables
        assert hasattr(app, 'colorizer_var'), "Colorizer variable not found"
        assert hasattr(app, 'color_var'), "Color enable variable not found"
        assert hasattr(app, 'text_prompt'), "Text prompt variable not found"
        assert hasattr(app, 'ref_path'), "Reference path variable not found"
        print("✅ Colorization variables verified")

        root.destroy()
        return True
    except Exception as e:
        print(f"❌ Model structure test error: {e}")
        return False

def test_colorization_features():
    """Test colorization-specific features"""
    try:
        from video_enhancer import VideoEnhancerApp
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()

        app = VideoEnhancerApp(root)

        # Test colorizer selection
        app.colorizer_var.set("VanGogh")
        assert app.colorizer_var.get() == "VanGogh"
        print("✅ VanGogh colorizer selection works")

        app.colorizer_var.set("Cobra")
        assert app.colorizer_var.get() == "Cobra"
        print("✅ Cobra colorizer selection works")

        # Test text prompt
        app.text_prompt.set("natural colors")
        assert app.text_prompt.get() == "natural colors"
        print("✅ Text prompt functionality works")

        # Test auto-selection method exists
        assert hasattr(app, 'auto_select_colorizer'), "Auto-select method not found"
        print("✅ Auto-selection method available")

        root.destroy()
        return True
    except Exception as e:
        print(f"❌ Colorization features test error: {e}")
        return False

def main():
    """Run all smoke tests"""
    print("🚀 Video Enhancement Tool - Smoke Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("GPU Setup Test", test_gpu_setup),
        ("Model Structure Test", test_model_structure),
        ("Colorization Features Test", test_colorization_features),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        if test_func():
            print(f"✅ {test_name} PASSED")
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Application ready for use!")
        return True
    else:
        print("⚠️ Some tests failed - check configuration")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
