# Video Enhancement Tool

**RIPER-Ω v2.6 Compliant Implementation**  
**RTX 3080 Optimized with VRAM Monitoring**

A comprehensive video enhancement tool that integrates multiple state-of-the-art denoising and low-light enhancement models with evolutionary algorithm-based auto-selection.

## Features

- **Multi-Model Support**: TAP, MVDenoiser, DarkIR, VanGogh, and Cobra integration
- **Video Colorization**: Advanced B&W to color conversion with text prompts and reference images
- **GPU Optimization**: RTX 3080 optimized with VRAM monitoring (<5GB target for colorization)
- **RL Auto-Selection**: Evolutionary algorithm for automatic model selection
- **Bias Mitigation**: Fitness-based bias detection and correction
- **Temporal Processing**: Frame buffering for video denoising and colorization
- **Low-Light Enhancement**: DarkIR integration for dark video processing
- **Processing Chain**: Configurable Low-light → Denoise → Colorize pipeline
- **Audio Preservation**: Automatic audio track copying

## Requirements

- Python 3.10+
- CUDA 11.8+ (for RTX 3080 optimization)
- 8GB+ VRAM recommended
- Dependencies listed in `requirements.txt`

## Installation

1. **Clone Repository**
   ```bash
   git clone https://github.com/gigamonkeyx/videoenh.git
   cd videoenh
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv env
   .\env\Scripts\activate  # Windows
   # source env/bin/activate  # Linux/Mac
   ```

3. **Install Dependencies**
   ```bash
   pip install opencv-python torch==2.0.0+cu118 torchvision pillow numpy
   pip install -r DarkIR/requirements.txt
   ```

4. **Download Model Weights**
   - MVDenoiser: Download from [Google Drive](https://github.com/mxxx99/MVDenoiser)
   - TAP: Download from [Google Drive](https://github.com/zfu006/TAP)
   - DarkIR: Download from [Hugging Face](https://github.com/cidautai/DarkIR)
   - Cobra: Download from [Hugging Face](https://huggingface.co/JunhaoZhuang/Cobra)
   - VanGogh: Repository is sparse - using fallback implementation
   - Place weights in `weights/` subfolder

## Usage

### GUI Application
```bash
python video_enhancer.py
```

### Features Overview

#### Model Selection
- **TAP**: Optimized for low-noise videos
- **MVDenoiser**: Better for high-noise scenarios
- **Auto**: RL-based automatic selection

#### Processing Options
- **Low-Light Enhancement**: Enable DarkIR preprocessing
- **Video Colorization**: Convert B&W videos to color
- **Text Prompts**: Guide VanGogh colorization with descriptions
- **Reference Images**: Use Cobra with reference color images
- **VRAM Monitoring**: Real-time memory usage display
- **Progress Tracking**: ETA calculation and progress bar

#### Colorization Usage
- **VanGogh Mode**:
  - Enable colorization checkbox
  - Select "VanGogh" from dropdown
  - Enter text prompt (e.g., "natural colors", "vintage film", "warm sunset tones")
  - Process B&W video with text-guided colorization

- **Cobra Mode**:
  - Enable colorization checkbox
  - Select "Cobra" from dropdown
  - Upload reference JPG image with desired color palette
  - Process B&W video matching reference colors

- **Auto Mode**: RL algorithm selects best colorizer based on content analysis

#### IR/Surveillance Video Tuning
- **DarkIR Gain Slider**: Adjust exposure/brightness (0.5-2.0)
- **Denoise Strength**: Control noise reduction intensity (5-50)
- **VanGogh Steps**: Diffusion model steps for quality vs speed (10-50)
- **Cobra Reference Weight**: Blend automatic vs reference colors (0-1)
- **User Feedback**: Thumbs up/down buttons for RL parameter optimization
- **IR Detection**: Automatic histogram equalization for surveillance footage
- **Console Logging**: Real-time processing information and VRAM monitoring

#### Advanced Features
- **Real-time Preview**: Slider adjustments show immediate effects
- **Parallel Processing**: Multi-threaded frame processing for RTX 3080
- **Bias Detection**: RL system prevents parameter drift
- **Fallback Logic**: Automatic model switching on failures
- **VRAM Optimization**: Dynamic batch sizing and memory monitoring

#### GPU Optimizations
- **Mixed Precision**: Automatic AMP for VRAM reduction
- **Adaptive Buffering**: Window size adjusts based on VRAM usage
- **Memory Monitoring**: Real-time VRAM tracking

## Architecture

### RIPER-Ω Protocol Compliance
- **Fitness Scoring**: Models scored based on VRAM efficiency
- **Bias Detection**: Automatic detection of model selection bias
- **RL Evolution**: Threshold mutation for balanced selection
- **Error Handling**: Graceful fallbacks on model failures

### Processing Pipeline
1. **Frame Extraction**: OpenCV-based video reading
2. **Preprocessing**: B&W detection and RGB conversion
3. **Low-Light Enhancement**: Optional DarkIR processing
4. **Temporal Buffering**: Frame sequence preparation
5. **Denoising**: Model-specific processing
6. **Post-processing**: Tensor to frame conversion
7. **Audio Integration**: MoviePy-based audio copying

### RL Auto-Selection Algorithm
```python
# Noise variance analysis
noise_var = np.var(sample_frames)

# Reward calculation
reward_tap = 1.0 / (noise_var + 1e-6) if noise_var < threshold else 0.5
reward_mv = 1.0 / (noise_var + 1e-6) if noise_var >= threshold else 0.3

# Bias detection and mutation
if tap_selections > 80%:
    threshold *= 0.9  # Mutate down
```

## Testing

Run the test suite:
```bash
pytest tests/test_video_enhancer.py -v
```

### Test Coverage
- Model loading and caching
- Tensor conversions
- VRAM monitoring
- RL auto-selection
- Bias detection
- Error handling

## Performance Targets

### VRAM Usage
- **Baseline**: <2GB idle
- **Denoising**: <1.5GB average
- **Colorization**: <5GB peak (FP16 optimization)
- **High-res fallback**: Automatic downscaling

### Processing Speed
- **1280x720**: ~3.77fps target
- **Batch processing**: Adaptive batch sizes
- **Memory optimization**: Mixed precision enabled

## Console Usage
- **Toggle Console**: Button to show/hide processing logs
- **Real-time Logging**: Timestamps, levels (INFO/WARNING/ERROR)
- **VRAM Monitoring**: Memory usage after each processing step
- **Parameter Logging**: Slider changes and RL mutations
- **Error Tracking**: Detailed error messages and stack traces
- **File Logging**: All messages saved to `app.log`

## Development Notes

### Model Integration Status
- ✅ Framework structure complete
- ✅ GPU optimization implemented
- ✅ VRAM monitoring active
- ✅ RL auto-selection functional
- ✅ IR tuning and console integration
- ✅ LVCD colorization fallback
- 🔄 Model weight loading (requires manual download)
- 🔄 Audio integration (MoviePy placeholder)

### Fitness Metrics
- **Model Loading**: >80% confidence required
- **VRAM Efficiency**: Reward = 1 - (vram_used / 2GB)
- **Bias Detection**: <80% single-model preference
- **Processing Success**: >70% frame completion rate

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce window_size or enable CPU fallback
2. **Model Loading Errors**: Verify weight file paths and formats
3. **Network Timeouts**: Retry operations (satellite network compatible)
4. **Audio Sync Issues**: Check input video codec compatibility

### VRAM Optimization
- Monitor real-time usage in GUI
- Automatic window size adjustment
- Mixed precision enabled by default
- Batch size adaptation based on memory

## Contributing

This project follows RIPER-Ω Protocol v2.6 for safe, auditable development:
- All changes must pass fitness checks (>70%)
- Bias mitigation required for RL components
- GPU optimization mandatory for RTX 3080
- Comprehensive testing required

## License

See LICENSE file for details.

## Acknowledgments

- **MVDenoiser**: [mxxx99/MVDenoiser](https://github.com/mxxx99/MVDenoiser)
- **TAP**: [zfu006/TAP](https://github.com/zfu006/TAP)
- **DarkIR**: [cidautai/DarkIR](https://github.com/cidautai/DarkIR)
- **RIPER-Ω Protocol**: Developed by Grok (xAI) for safe AI development
