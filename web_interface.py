import gradio as gr
from illustrious_engine import IllustriousEngine
from PIL import Image
from pathlib import Path
import json
from datetime import datetime

class WebInterface:
    """Gradio web interface for Illustrious Engine"""
    
    def __init__(self):
        self.engine = IllustriousEngine()
        self.last_generation_info = None
        
    # ==================== MODEL MANAGEMENT ====================
    
    def scan_models_wrapper(self):
        """Scan for available models"""
        models = self.engine.scan_models()
        if not models["base_models"]:
            return (
                gr.update(choices=[], value=None),
                gr.update(choices=models["loras"], value=[]),
                gr.update(choices=models["lycoris"], value=[]),
                "⚠️ No base models found. Please add models to the ./Model folder."
            )
        return (
            gr.update(choices=models["base_models"], value=None),
            gr.update(choices=models["loras"], value=[]),
            gr.update(choices=models["lycoris"], value=[]),
            f"✅ Found {len(models['base_models'])} base models, {len(models['loras'])} LoRAs, {len(models['lycoris'])} LyCORIS"
        )
    
    def load_model_wrapper(self, base_model, loras, lycoris, use_safetensors, scheduler):
        """Load selected models"""
        if not base_model:
            return "❌ Please select a base model first"
            
        selected_models = {
            "base_model": base_model,
            "loras": loras,
            "lycoris": lycoris,
            "controlnets": []
        }
        
        success, message = self.engine.load_model(selected_models, use_safetensors, scheduler)
        
        if success and self.engine.loaded_loras:
            lora_names = [Path(x).stem for x in self.engine.loaded_loras[:3]]
            message += f"\n📎 Loaded: {', '.join(lora_names)}"
            if len(self.engine.loaded_loras) > 3:
                message += f" (+{len(self.engine.loaded_loras)-3} more)"
        
        return message
    
    # ==================== IMAGE GENERATION ====================
    
    def txt2img_wrapper(self, prompt, negative_prompt, width, height, steps, guidance, seed):
        """Generate image from text prompt"""
        if not prompt.strip():
            return None, "Please enter a prompt", gr.update(visible=False)
        
        image, message, generation_info = self.engine.generate_image(
            prompt, negative_prompt, width, height, steps, guidance, seed
        )
        
        self.last_generation_info = generation_info
        save_button_update = gr.update(visible=(image is not None))
        
        return image, message, save_button_update
    
    def img2img_wrapper(self, prompt, init_image, negative_prompt, strength, steps, guidance, seed, 
                       use_controlnet, resize_mode, target_width, target_height):
        """Generate image from reference image"""
        if not prompt.strip():
            return None, "Please enter a prompt", gr.update(visible=False)
        if init_image is None:
            return None, "Please upload a reference image", gr.update(visible=False)
        
        image, message, generation_info = self.engine.generate_from_image(
            prompt, init_image, negative_prompt, strength, steps, guidance, seed,
            use_controlnet, resize_mode, target_width, target_height
        )
        
        self.last_generation_info = generation_info
        save_button_update = gr.update(visible=(image is not None))
        
        return image, message, save_button_update
    
    # ==================== PRESET MANAGEMENT ====================
    
    def save_preset_wrapper(self, preset_name, generated_image):
        """Save current generation as preset"""
        if not preset_name.strip():
            return "❌ Please enter a preset name"
        if self.last_generation_info is None:
            return "❌ No generation info available"
        if generated_image is None:
            return "❌ No image to save"
        
        success, message = self.engine.save_preset(preset_name, generated_image, self.last_generation_info)
        return message
    
    def load_presets_wrapper(self):
        """Load and display all presets"""
        presets = self.engine.load_presets()
        
        if not presets:
            return gr.update(value="No presets found"), gr.update(choices=[], value=None)
        
        preset_choices = []
        preset_display = []
        
        for preset in presets:
            gen_info = preset.get('generation_info', {})
            
            # Create display text
            prompt_short = gen_info.get('prompt', '')[:50] + "..." if len(gen_info.get('prompt', '')) > 50 else gen_info.get('prompt', '')
            neg_prompt_short = gen_info.get('negative_prompt', '')[:30] + "..." if len(gen_info.get('negative_prompt', '')) > 30 else gen_info.get('negative_prompt', '')
            
            display_text = f"""**{preset['name']}**
- Prompt: {prompt_short}
- Negative: {neg_prompt_short}
- Seed: {gen_info.get('seed', 'N/A')}
- Size: {gen_info.get('width', 'N/A')}x{gen_info.get('height', 'N/A')}
- Steps: {gen_info.get('num_inference_steps', 'N/A')}
- CFG: {gen_info.get('guidance_scale', 'N/A')}
- Created: {preset.get('creation_date', '')[:19]}
---"""
            
            preset_choices.append((preset['name'], preset['id']))
            preset_display.append(display_text)
        
        display_text = "\n".join(preset_display)
        return gr.update(value=display_text), gr.update(choices=preset_choices, value=None)
    
    def load_preset_to_generation(self, preset_id, target_tab):
        """Load preset parameters to generation interface"""
        if not preset_id:
            return "Please select a preset", *[gr.update() for _ in range(17)]
        
        preset = self.engine.get_preset_by_id(preset_id)
        if not preset:
            return "Preset not found", *[gr.update() for _ in range(17)]
        
        gen_info = preset.get('generation_info', {})
        
        # Common parameters
        prompt = gen_info.get('prompt', '')
        negative_prompt = gen_info.get('negative_prompt', '')
        seed = gen_info.get('seed', -1)
        steps = gen_info.get('num_inference_steps', 28)
        guidance = gen_info.get('guidance_scale', 7.0)
        
        if gen_info.get('generation_type') == 'txt2img':
            # Text-to-Image parameters
            width = gen_info.get('width', 1024)
            height = gen_info.get('height', 1024)
            
            return (
                f"✅ Loaded preset: {preset['name']}",
                # txt2img parameters (7 items)
                gr.update(value=prompt),          # txt_prompt
                gr.update(value=negative_prompt), # txt_negative
                gr.update(value=width),           # txt_width
                gr.update(value=height),          # txt_height
                gr.update(value=steps),           # txt_steps
                gr.update(value=guidance),        # txt_guidance
                gr.update(value=seed),            # txt_seed
                # img2img parameters (10 items) - no change
                gr.update(),                      # img_prompt
                gr.update(),                      # img_negative
                gr.update(),                      # img_strength
                gr.update(),                      # img_steps
                gr.update(),                      # img_guidance
                gr.update(),                      # img_seed
                gr.update(),                      # use_controlnet
                gr.update(),                      # resize_mode
                gr.update(),                      # target_width
                gr.update()                       # target_height
            )
        else:
            # Image-to-Image parameters
            strength = gen_info.get('strength', 0.75)
            use_controlnet = gen_info.get('use_controlnet', True)
            resize_mode = gen_info.get('resize_mode', 'Resize')
            target_width = gen_info.get('target_width', 1024)
            target_height = gen_info.get('target_height', 1024)
            
            return (
                f"✅ Loaded preset: {preset['name']}",
                # txt2img parameters (7 items) - no change
                gr.update(),                      # txt_prompt
                gr.update(),                      # txt_negative
                gr.update(),                      # txt_width
                gr.update(),                      # txt_height
                gr.update(),                      # txt_steps
                gr.update(),                      # txt_guidance
                gr.update(),                      # txt_seed
                # img2img parameters (10 items)
                gr.update(value=prompt),          # img_prompt
                gr.update(value=negative_prompt), # img_negative
                gr.update(value=strength),        # img_strength
                gr.update(value=steps),           # img_steps
                gr.update(value=guidance),        # img_guidance
                gr.update(value=seed),            # img_seed
                gr.update(value=use_controlnet),  # use_controlnet
                gr.update(value=resize_mode),     # resize_mode
                gr.update(value=target_width),    # target_width
                gr.update(value=target_height)    # target_height
            )
    
    def delete_preset_wrapper(self, preset_id):
        """Delete selected preset"""
        if not preset_id:
            return "Please select a preset to delete"
        
        success, message = self.engine.delete_preset(preset_id)
        return message
    
    # ==================== HELP GUIDE ====================
    
    def load_help_guide(self):
        """Load help guide from markdown file"""
        help_file = Path("help_guide.md")
        if help_file.exists():
            try:
                with open(help_file, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"Error reading help guide: {e}")
                return "Help guide not available"
        else:
            return """# 🎨 Illustrious-XL Runner Help Guide

## 🚀 Quick Start
1. **Scan Models**: Click "🔍 Scan Models" to find your models in ./Model folder
2. **Load Model**: Select a base model and click "⚡ Load"
3. **Generate**: Enter your prompt and click "🎨 Generate Image"

## 📝 Prompt Syntax (Long Prompts Supported!)
- `(keyword)` = 1.1x stronger
- `((keyword))` = 1.21x stronger  
- `(keyword:1.5)` = 1.5x stronger
- `[keyword]` = 0.9x weaker
- `[[keyword]]` = 0.81x weaker
- `[keyword:0.7]` = 0.7x weaker
- `(concept1:1.2) AND (concept2:0.8)` = separate concepts

## 🎯 Examples
```
(masterpiece:1.3), (best quality:1.2), 1girl, beautiful face, detailed eyes, [bad anatomy:0.5]
```

## 💾 VRAM Optimization
- **12GB+**: Full xformers optimization + Compel
- **6-12GB**: Model CPU offload + xformers + Compel
- **<6GB**: Sequential CPU offload + attention slicing (Compel disabled)

## 📚 Preset System
- Save your favorite settings as presets
- Load presets to any generation tab
- Organize your workflows

## 🔧 Model Support
- **Base Models**: SDXL-compatible models (.safetensors, .ckpt)
- **LoRA**: Additional style and character models
- **LyCORIS**: Advanced fine-tuning models
- **ControlNet**: Structure-preserving img2img

## ⚠️ Notes
- This tool is for educational purposes only
- Respect model licenses and local laws
- Long prompts (>77 tokens) are supported with Compel
"""
    
    # ==================== INTERFACE CREATION ====================
    
    def create_interface(self):
        """Create the main Gradio interface"""
        
        # CSS Styling
        css = """
        .model-loader {
            background: linear-gradient(45deg, #f0f9ff, #e0f2fe);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #0891b2;
            margin-bottom: 20px;
        }
        .preset-manager {
            background: linear-gradient(45deg, #fef7cd, #fef3c7);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #f59e0b;
            margin-bottom: 20px;
        }
        .compact-button {
            min-width: 120px !important;
            height: 35px !important;
            font-size: 12px !important;
        }
        .vram-indicator {
            background: linear-gradient(45deg, #dcfce7, #bbf7d0);
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #16a34a;
            margin: 10px 0;
        }
        .preset-display {
            max-height: 400px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            background-color: #f9fafb;
        }
        .disclaimer {
            background: linear-gradient(45deg, #fef2f2, #fee2e2);
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #f87171;
            margin: 10px 0;
        }
        """
        
        with gr.Blocks(title="Illustrious-XL Runner (Modular)", theme=gr.themes.Soft(), css=css) as demo:
            
            # Header
            gr.Markdown("# 🎨 Illustrious-XL Runner (Modular Architecture)")
            gr.Markdown("High-quality anime-style image generation with preset management system")
            
            # Disclaimer
            gr.Markdown("""
            <div class="disclaimer">
            ⚠️ <strong>DISCLAIMER:</strong> This project is for <strong>educational purposes only</strong>. 
            Not for commercial use or illegal activities. Users are responsible for complying with local laws and model licenses.
            </div>
            """)
            
            # VRAM Status with Compel info
            if self.engine.vram_gb > 0:
                compel_status = "✅ Long prompts supported" if self.engine.vram_gb >= 6 else "⚠️ Long prompts disabled (<6GB VRAM)"
                vram_info = f"💾 VRAM: {self.engine.vram_gb:.1f} GB | {compel_status}"
            else:
                vram_info = "💾 CPU Mode | ⚠️ Long prompts disabled"
            gr.Markdown(f"<div class='vram-indicator'>{vram_info}</div>")
            
            # Model Loading Section
            self._create_model_section()
            
            # Main Interface Tabs
            with gr.Tabs():
                # Text-to-Image Tab
                txt_components = self._create_txt2img_tab()
                
                # Image-to-Image Tab
                img_components = self._create_img2img_tab()
                
                # Preset Manager Tab
                preset_components = self._create_preset_tab()
                
                # Help Guide Tab
                self._create_help_tab()
            
            # Event Handlers
            self._setup_event_handlers(txt_components, img_components, preset_components)
        
        return demo
    
    def _create_model_section(self):
        """Create model loading section"""
        with gr.Group(elem_classes="model-loader"):
            gr.Markdown("### 🔧 Model Configuration")
            
            with gr.Row():
                with gr.Column(scale=1):
                    self.scan_btn = gr.Button("🔍 Scan Models", variant="secondary", elem_classes="compact-button")
                    self.load_btn = gr.Button("⚡ Load", variant="primary", elem_classes="compact-button")
                    
                with gr.Column(scale=3):
                    self.load_status = gr.Textbox(
                        label="Status", 
                        interactive=False, 
                        value="Click 'Scan Models' to find available models in ./Model folder",
                        lines=3
                    )
            
            with gr.Row():
                with gr.Column():
                    self.base_model_dropdown = gr.Dropdown(
                        label="Base Model (Required)",
                        choices=[],
                        value=None,
                        interactive=True,
                        info="Select a base model from ./Model folder"
                    )
                    
                with gr.Column():
                    self.scheduler_dropdown = gr.Dropdown(
                        label="Sampler",
                        choices=["Euler a", "DDIM", "DPM++ 2M Karras", "UniPC", "Heun"],
                        value="Euler a"
                    )
            
            with gr.Row():
                with gr.Column():
                    self.lora_checkboxes = gr.CheckboxGroup(
                        label="LoRA Models (Optional)",
                        choices=[],
                        value=[]
                    )
                    
                with gr.Column():
                    self.lycoris_checkboxes = gr.CheckboxGroup(
                        label="LyCORIS Models (Optional)", 
                        choices=[],
                        value=[]
                    )
            
            with gr.Row():
                self.use_safetensors = gr.Checkbox(label="Use SafeTensors", value=True)
    
    def _create_txt2img_tab(self):
        """Create Text-to-Image tab"""
        with gr.TabItem("✨ Text to Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    txt_prompt = gr.Textbox(
                        label="Prompt", 
                        placeholder="1girl, masterpiece, best quality, highly detailed...",
                        lines=4
                    )
                    txt_negative = gr.Textbox(
                        label="Negative Prompt",
                        lines=3,
                        value="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
                    )
                    
                    with gr.Row():
                        txt_width = gr.Slider(512, 1536, 1024, step=64, label="Width")
                        txt_height = gr.Slider(512, 1536, 1024, step=64, label="Height")
                    
                    with gr.Row():
                        txt_steps = gr.Slider(10, 50, 28, step=1, label="Steps")
                        txt_guidance = gr.Slider(1.0, 20.0, 7.0, step=0.5, label="CFG Scale")
                    
                    txt_seed = gr.Number(label="Seed (-1 for random)", value=-1)
                    txt_generate = gr.Button("🎨 Generate Image", variant="primary", size="lg")
                    
                    # Save Preset Section
                    with gr.Group(visible=False) as txt_save_group:
                        gr.Markdown("### 💾 Save as Preset")
                        txt_preset_name = gr.Textbox(
                            label="Preset Name",
                            placeholder="Enter preset name...",
                            value=""
                        )
                        txt_save_preset = gr.Button("💾 Save Preset", variant="secondary")
                        txt_save_message = gr.Textbox(label="Save Status", interactive=False)
                
                with gr.Column(scale=1):
                    txt_output = gr.Image(label="Generated Image", height=600)
                    txt_message = gr.Textbox(label="Status", interactive=False)
        
        return {
            'prompt': txt_prompt,
            'negative': txt_negative,
            'width': txt_width,
            'height': txt_height,
            'steps': txt_steps,
            'guidance': txt_guidance,
            'seed': txt_seed,
            'generate': txt_generate,
            'output': txt_output,
            'message': txt_message,
            'save_group': txt_save_group,
            'preset_name': txt_preset_name,
            'save_preset': txt_save_preset,
            'save_message': txt_save_message
        }
    
    def _create_img2img_tab(self):
        """Create Image-to-Image tab"""
        with gr.TabItem("🖼️ Image to Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe how you want to modify the image...",
                        lines=4
                    )
                    img_input = gr.Image(label="Reference Image", type="pil", height=300)
                    img_negative = gr.Textbox(
                        label="Negative Prompt",
                        lines=2,
                        value="lowres, bad anatomy, bad hands, text, error, missing fingers, cropped, worst quality, low quality, blurry"
                    )
                    
                    with gr.Row():
                        use_controlnet = gr.Checkbox(label="Use ControlNet (Structure Preservation)", value=True)
                        resize_mode = gr.Dropdown(
                            label="Resize Mode",
                            choices=["Resize", "Crop and Resize", "Resize and Fill"],
                            value="Resize"
                        )
                    
                    with gr.Row():
                        target_width = gr.Slider(512, 1536, 1024, step=64, label="Target Width")
                        target_height = gr.Slider(512, 1536, 1024, step=64, label="Target Height")
                    
                    with gr.Row():
                        img_strength = gr.Slider(0.1, 1.0, 0.75, step=0.05, label="Strength")
                        img_steps = gr.Slider(10, 50, 28, step=1, label="Steps")
                    
                    with gr.Row():
                        img_guidance = gr.Slider(1.0, 20.0, 7.0, step=0.5, label="CFG Scale")
                        img_seed = gr.Number(label="Seed (-1 for random)", value=-1)
                    
                    img_generate = gr.Button("🎨 Generate Image", variant="primary", size="lg")
                    
                    # Save Preset Section
                    with gr.Group(visible=False) as img_save_group:
                        gr.Markdown("### 💾 Save as Preset")
                        img_preset_name = gr.Textbox(
                            label="Preset Name",
                            placeholder="Enter preset name...",
                            value=""
                        )
                        img_save_preset = gr.Button("💾 Save Preset", variant="secondary")
                        img_save_message = gr.Textbox(label="Save Status", interactive=False)
                
                with gr.Column(scale=1):
                    img_output = gr.Image(label="Generated Image", height=600)
                    img_message = gr.Textbox(label="Status", interactive=False)
        
        return {
            'prompt': img_prompt,
            'input': img_input,
            'negative': img_negative,
            'strength': img_strength,
            'steps': img_steps,
            'guidance': img_guidance,
            'seed': img_seed,
            'use_controlnet': use_controlnet,
            'resize_mode': resize_mode,
            'target_width': target_width,
            'target_height': target_height,
            'generate': img_generate,
            'output': img_output,
            'message': img_message,
            'save_group': img_save_group,
            'preset_name': img_preset_name,
            'save_preset': img_save_preset,
            'save_message': img_save_message
        }
    
    def _create_preset_tab(self):
        """Create Preset Manager tab"""
        with gr.TabItem("🎯 Preset Manager"):
            with gr.Group(elem_classes="preset-manager"):
                gr.Markdown("### 📚 Preset Library")
                
                with gr.Row():
                    refresh_presets = gr.Button("🔄 Refresh", variant="secondary")
                    load_preset_btn = gr.Button("📥 Load to Generation", variant="primary")
                    delete_preset_btn = gr.Button("🗑️ Delete", variant="stop")
                
                preset_status = gr.Textbox(label="Action Status", interactive=False)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        preset_display = gr.Markdown(
                            value="Click 'Refresh' to load presets",
                            elem_classes="preset-display"
                        )
                    
                    with gr.Column(scale=1):
                        preset_selector = gr.Dropdown(
                            label="Select Preset",
                            choices=[],
                            value=None,
                            info="Choose a preset to load or delete"
                        )
                        
                        target_tab_selector = gr.Radio(
                            label="Load to Tab",
                            choices=["Text to Image", "Image to Image"],
                            value="Text to Image",
                            info="Select which generation tab to load preset into"
                        )
        
        return {
            'refresh': refresh_presets,
            'load': load_preset_btn,
            'delete': delete_preset_btn,
            'status': preset_status,
            'display': preset_display,
            'selector': preset_selector,
            'target_tab': target_tab_selector
        }
    
    def _create_help_tab(self):
        """Create Help Guide tab"""
        with gr.TabItem("💡 Help Guide"):
            help_content = self.load_help_guide()
            gr.Markdown(help_content)
    
    def _setup_event_handlers(self, txt_components, img_components, preset_components):
        """Setup all event handlers"""
        
        # Model Management
        self.scan_btn.click(
            self.scan_models_wrapper,
            outputs=[self.base_model_dropdown, self.lora_checkboxes, self.lycoris_checkboxes, self.load_status]
        )
        
        self.load_btn.click(
            self.load_model_wrapper,
            inputs=[self.base_model_dropdown, self.lora_checkboxes, self.lycoris_checkboxes, 
                   self.use_safetensors, self.scheduler_dropdown],
            outputs=[self.load_status]
        )
        
        # Text-to-Image Generation
        txt_components['generate'].click(
            self.txt2img_wrapper,
            inputs=[txt_components['prompt'], txt_components['negative'], txt_components['width'], 
                   txt_components['height'], txt_components['steps'], txt_components['guidance'], 
                   txt_components['seed']],
            outputs=[txt_components['output'], txt_components['message'], txt_components['save_group']]
        )
        
        # Image-to-Image Generation
        img_components['generate'].click(
            self.img2img_wrapper,
            inputs=[img_components['prompt'], img_components['input'], img_components['negative'], 
                   img_components['strength'], img_components['steps'], img_components['guidance'], 
                   img_components['seed'], img_components['use_controlnet'], img_components['resize_mode'],
                   img_components['target_width'], img_components['target_height']],
            outputs=[img_components['output'], img_components['message'], img_components['save_group']]
        )
        
        # Preset Saving
        txt_components['save_preset'].click(
            self.save_preset_wrapper,
            inputs=[txt_components['preset_name'], txt_components['output']],
            outputs=[txt_components['save_message']]
        )
        
        img_components['save_preset'].click(
            self.save_preset_wrapper,
            inputs=[img_components['preset_name'], img_components['output']],
            outputs=[img_components['save_message']]
        )
        
        # Preset Management
        preset_components['refresh'].click(
            self.load_presets_wrapper,
            outputs=[preset_components['display'], preset_components['selector']]
        )
        
        preset_components['load'].click(
            self.load_preset_to_generation,
            inputs=[preset_components['selector'], preset_components['target_tab']],
            outputs=[
                preset_components['status'],
                # txt2img outputs
                txt_components['prompt'], txt_components['negative'], txt_components['width'], 
                txt_components['height'], txt_components['steps'], txt_components['guidance'], 
                txt_components['seed'],
                # img2img outputs
                img_components['prompt'], img_components['negative'], img_components['strength'], 
                img_components['steps'], img_components['guidance'], img_components['seed'],
                img_components['use_controlnet'], img_components['resize_mode'], 
                img_components['target_width'], img_components['target_height']
            ]
        )
        
        preset_components['delete'].click(
            self.delete_preset_wrapper,
            inputs=[preset_components['selector']],
            outputs=[preset_components['status']]
        )

def main():
    """Main function to launch the web interface"""
    interface = WebInterface()
    demo = interface.create_interface()
    
    # Launch the interface
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )

if __name__ == "__main__":
    main()