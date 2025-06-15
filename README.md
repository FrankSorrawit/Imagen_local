# 🎨 Illustrious-XL Advanced Runner

A powerful web interface for running Illustrious-XL-v2.0 with advanced features including LoRA/LyCORIS support, multiple samplers, and ControlNet integration for high-quality anime-style image generation.

## ✨ Features

### 🔧 Model Management
- **Automatic Model Scanning**: Scans your model folder and organizes files by type
- **LoRA & LyCORIS Support**: Load multiple LoRA and LyCORIS models simultaneously
- **Multiple Base Models**: Support for various SDXL-based models
- **Flexible Model Organization**: Automatically detects model types from folder structure

### 🎛️ Advanced Sampling
- **Multiple Samplers**: Euler a, DDIM, DPM++ 2M Karras, UniPC, Heun
- **Optimized Settings**: Pre-configured settings for best quality
- **Custom Scheduler**: Dynamic scheduler switching

### 🖼️ Image Generation
- **Text-to-Image**: High-quality anime image generation
- **Image-to-Image**: Transform existing images while preserving key features
- **ControlNet Integration**: Face and pose preservation using Canny edge detection
- **Multiple Resize Modes**: Resize, Crop and Resize, Resize and Fill

### 🎯 Face Preservation
- **Advanced ControlNet**: Preserves facial features from reference images
- **Minimal Face Changes**: Maintains character identity while changing clothes/style
- **Structural Preservation**: Keeps pose and body proportions intact

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ VRAM (for optimal performance)

### Quick Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/illustrious-xl-runner.git
cd illustrious-xl-runner
```

2. **Create virtual environment**
```bash
python -m venv illustrious_env
```

3. **Activate environment**
```bash
# Windows
illustrious_env\Scripts\activate

# Linux/Mac
source illustrious_env/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Download base model**
- Download Illustrious-XL-v2.0 from [Hugging Face](https://huggingface.co/OnomaAI/Illustrious-XL-v2.0)
- Place in `./models/Illustrious-XL-v2.0/` folder

6. **Create model directory structure**
```
D:/Side project/man Dream/Model/
├── base_models/
│   └── your_model.safetensors
├── lora/
│   ├── character_lora.safetensors
│   └── style_lora.safetensors
├── lycoris/
│   └── your_lycoris.safetensors
└── controlnet/
    └── control_model.safetensors
```

## 🎮 Usage

1. **Start the application**
```bash
python illustrious_runner.py
```

2. **Open web interface**
   - Navigate to `http://127.0.0.1:7860`
   - Interface will open automatically in your browser

3. **Load models**
   - Click "🔍 Scan Models" to detect available models
   - Select desired base model, LoRA, and LyCORIS files
   - Choose your preferred sampler
   - Click "⚡ Load" to initialize

4. **Generate images**
   - **Text-to-Image**: Enter prompts and generate new images
   - **Image-to-Image**: Upload reference image and transform it

## 🎯 Model Organization

### Recommended Folder Structure
```
Model/
├── base_models/           # SDXL base models (.safetensors)
├── lora/                  # LoRA files
│   ├── characters/        # Character-specific LoRAs
│   ├── styles/           # Art style LoRAs
│   └── concepts/         # Concept LoRAs
├── lycoris/              # LyCORIS files
└── controlnet/           # ControlNet models
```

### File Naming Convention
- Use descriptive names: `character_rem_v2.safetensors`
- Include version numbers: `style_anime_v1.1.safetensors`
- Separate categories: `outfit_schooluniform.safetensors`

## ⚙️ Configuration

### Model Path Setup
Edit the model path in `illustrious_runner.py`:
```python
runner = IllustriousXLRunner(
    model_folder="D:/Side project/man Dream/Model"  # Change this path
)
```

### Performance Optimization
- **High VRAM (12GB+)**: Use 1024x1024+ resolution
- **Medium VRAM (8GB)**: Use 768x768 resolution  
- **Low VRAM (6GB)**: Use 512x512 resolution, enable CPU offload

## 🎨 Sampler Guide

| Sampler | Speed | Quality | Best For |
|---------|-------|---------|----------|
| **Euler a** | ⚡⚡⚡ | ⭐⭐⭐ | General use, fast generation |
| **DDIM** | ⚡⚡ | ⭐⭐⭐ | Consistent results, reproducible |
| **DPM++ 2M Karras** | ⚡ | ⭐⭐⭐⭐⭐ | Highest quality, slower |
| **UniPC** | ⚡⚡⚡ | ⭐⭐⭐⭐ | Good quality, low steps |
| **Heun** | ⚡ | ⭐⭐⭐⭐ | High quality, needs more steps |

## 🎭 Prompt Examples

### Character Generation
```
Positive: 1girl, masterpiece, detailed face, long hair, school uniform, anime style
Negative: lowres, bad anatomy, worst quality, blurry
```

### Style Transfer (Image-to-Image)
```
Positive: same character, oil painting style, detailed brush strokes
Settings: Strength 0.6, ControlNet ON
```

### Outfit Change
```
Positive: same face, same body, evening gown, elegant dress, formal wear
Negative: original clothes, casual wear, different character
Settings: Strength 0.7, ControlNet ON
```

## 🛠️ Troubleshooting

### Common Issues

**LoRA Loading Failed**
```bash
# Install missing dependencies
pip install peft transformers

# Check file format (must be .safetensors or .pt)
# Ensure LoRA is compatible with SDXL
```

**CUDA Out of Memory**
```python
# Reduce resolution: 1024 → 768 → 512
# Lower steps: 28 → 20
# Enable CPU offload (automatic)
```

**ControlNet Not Working**
```bash
# Check internet connection (downloads model on first use)
# Fallback to regular img2img if download fails
```

**Model Not Found**
```bash
# Check model path in code
# Ensure folder structure is correct
# Run "Scan Models" to refresh
```

## 📋 Requirements

- **Python**: 3.8+
- **PyTorch**: 2.0+ with CUDA support
- **GPU**: NVIDIA GPU with 6GB+ VRAM
- **Storage**: 10GB+ free space for models
- **RAM**: 16GB+ recommended

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Illustrious-XL-v2.0](https://huggingface.co/OnomaAI/Illustrious-XL-v2.0) by OnomaAI
- [Diffusers](https://github.com/huggingface/diffusers) by Hugging Face
- [Gradio](https://github.com/gradio-app/gradio) for the web interface
- [ControlNet](https://github.com/lllyasviel/ControlNet) for advanced image control

## 📞 Support

If you encounter any issues or have questions:
- Open an [Issue](https://github.com/yourusername/illustrious-xl-runner/issues)
- Check the [Discussions](https://github.com/yourusername/illustrious-xl-runner/discussions)
- Read the documentation above

---

⭐ **Star this repository if you find it useful!** ⭐