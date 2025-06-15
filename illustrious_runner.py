import torch
from diffusers import (
    StableDiffusionXLPipeline, 
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler,
    HeunDiscreteScheduler
)
from PIL import Image, ImageOps
import gradio as gr
import os
from pathlib import Path
import json
import cv2
import numpy as np

class IllustriousXLRunner:
    def __init__(self, base_model_path="./models/Illustrious-XL-v2.0", model_folder="D:/Side project/man Dream/Model"):
        self.base_model_path = base_model_path
        self.model_folder = Path(model_folder)
        self.txt2img_pipe = None
        self.img2img_pipe = None
        self.controlnet_pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded_loras = []
        self.available_models = {}
        self.current_scheduler = None
        
    def scan_models(self):
        """สแกนหาโมเดลทั้งหมดใน folder"""
        models = {
            "base_models": [],
            "loras": [],
            "lycoris": [],
            "controlnets": []
        }
        
        if not self.model_folder.exists():
            self.model_folder.mkdir(parents=True, exist_ok=True)
            return models
            
        # สแกนไฟล์ในโฟลเดอร์
        for file_path in self.model_folder.rglob("*"):
            if file_path.is_file():
                file_name = file_path.name.lower()
                relative_path = str(file_path.relative_to(self.model_folder))
                
                if file_name.endswith(('.safetensors', '.ckpt', '.pt')):
                    if 'lora' in file_name or 'lora' in str(file_path.parent).lower():
                        models["loras"].append(relative_path)
                    elif 'lycoris' in file_name or 'lyco' in file_name:
                        models["lycoris"].append(relative_path)
                    elif 'controlnet' in file_name or 'control' in file_name:
                        models["controlnets"].append(relative_path)
                    else:
                        models["base_models"].append(relative_path)
        
        # เพิ่ม base model ที่มีอยู่แล้ว
        if Path(self.base_model_path).exists():
            models["base_models"].insert(0, "Illustrious-XL-v2.0 (Default)")
            
        self.available_models = models
        return models
    
    def get_scheduler(self, scheduler_name):
        """สร้าง scheduler ตามชื่อ"""
        schedulers = {
            "Euler a": EulerAncestralDiscreteScheduler,
            "DDIM": DDIMScheduler,
            "DPM++ 2M Karras": DPMSolverMultistepScheduler,
            "UniPC": UniPCMultistepScheduler,
            "Heun": HeunDiscreteScheduler
        }
        
        if scheduler_name in schedulers:
            if scheduler_name == "DPM++ 2M Karras":
                return schedulers[scheduler_name].from_config(
                    self.txt2img_pipe.scheduler.config,
                    use_karras_sigmas=True
                )
            else:
                return schedulers[scheduler_name].from_config(
                    self.txt2img_pipe.scheduler.config
                )
        return self.txt2img_pipe.scheduler
    
    def load_model(self, selected_models, use_safetensors=True, scheduler_name="Euler a"):
        """โหลดโมเดลและส่วนเสริมที่เลือก"""
        print(f"Loading selected models...")
        print(f"Using device: {self.device}")
        
        try:
            # โหลด base model
            base_model = selected_models.get("base_model", "Illustrious-XL-v2.0 (Default)")
            
            if base_model == "Illustrious-XL-v2.0 (Default)":
                model_path = self.base_model_path
            else:
                model_path = self.model_folder / base_model
            
            print(f"Loading base model: {model_path}")
            
            # ตรวจสอบรูปแบบไฟล์
            if str(model_path).endswith('.safetensors'):
                self.txt2img_pipe = StableDiffusionXLPipeline.from_single_file(
                    str(model_path),
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_safetensors=True
                )
            else:
                self.txt2img_pipe = StableDiffusionXLPipeline.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_safetensors=use_safetensors,
                    variant="fp16" if self.device == "cuda" else None
                )
            
            # ตั้งค่า scheduler
            if scheduler_name != "Default":
                self.txt2img_pipe.scheduler = self.get_scheduler(scheduler_name)
                self.current_scheduler = scheduler_name
            
            # สร้าง img2img pipeline
            self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
                vae=self.txt2img_pipe.vae,
                text_encoder=self.txt2img_pipe.text_encoder,
                text_encoder_2=self.txt2img_pipe.text_encoder_2,
                tokenizer=self.txt2img_pipe.tokenizer,
                tokenizer_2=self.txt2img_pipe.tokenizer_2,
                unet=self.txt2img_pipe.unet,
                scheduler=self.txt2img_pipe.scheduler,
            )
            
            # โหลด ControlNet สำหรับ img2img
            try:
                controlnet = ControlNetModel.from_pretrained(
                    "diffusers/controlnet-canny-sdxl-1.0",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                self.controlnet_pipe = StableDiffusionXLControlNetPipeline(
                    vae=self.txt2img_pipe.vae,
                    text_encoder=self.txt2img_pipe.text_encoder,
                    text_encoder_2=self.txt2img_pipe.text_encoder_2,
                    tokenizer=self.txt2img_pipe.tokenizer,
                    tokenizer_2=self.txt2img_pipe.tokenizer_2,
                    unet=self.txt2img_pipe.unet,
                    controlnet=controlnet,
                    scheduler=self.txt2img_pipe.scheduler,
                )
                print("ControlNet loaded successfully")
            except Exception as e:
                print(f"ControlNet loading failed: {e}")
                self.controlnet_pipe = None
            
            # ย้ายไป GPU
            if self.device == "cuda":
                self.txt2img_pipe = self.txt2img_pipe.to(self.device)
                self.img2img_pipe = self.img2img_pipe.to(self.device)
                if self.controlnet_pipe:
                    self.controlnet_pipe = self.controlnet_pipe.to(self.device)
                
                # เปิดการประหยัดหน่วยความจำ
                try:
                    self.txt2img_pipe.enable_xformers_memory_efficient_attention()
                    self.img2img_pipe.enable_xformers_memory_efficient_attention()
                    if self.controlnet_pipe:
                        self.controlnet_pipe.enable_xformers_memory_efficient_attention()
                    print("xformers enabled")
                except:
                    print("xformers not available")
                
                # CPU offload
                try:
                    self.txt2img_pipe.enable_model_cpu_offload()
                    self.img2img_pipe.enable_model_cpu_offload()
                    if self.controlnet_pipe:
                        self.controlnet_pipe.enable_model_cpu_offload()
                    print("CPU offload enabled")
                except:
                    pass
            
            # โหลด LoRA
            self.loaded_loras = []
            loras = selected_models.get("loras", [])
            for lora_path in loras:
                try:
                    full_lora_path = self.model_folder / lora_path
                    self.txt2img_pipe.load_lora_weights(str(full_lora_path))
                    self.loaded_loras.append(lora_path)
                    print(f"LoRA loaded: {lora_path}")
                except Exception as e:
                    print(f"Failed to load LoRA {lora_path}: {e}")
            
            # โหลด LyCORIS (ใช้วิธีเดียวกับ LoRA)
            lycoris = selected_models.get("lycoris", [])
            for lyco_path in lycoris:
                try:
                    full_lyco_path = self.model_folder / lyco_path
                    self.txt2img_pipe.load_lora_weights(str(full_lyco_path))
                    self.loaded_loras.append(lyco_path)
                    print(f"LyCORIS loaded: {lyco_path}")
                except Exception as e:
                    print(f"Failed to load LyCORIS {lyco_path}: {e}")
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def generate_canny_edge(self, image, low_threshold=100, high_threshold=200):
        """สร้าง Canny edge detection สำหรับ ControlNet"""
        image_array = np.array(image)
        canny = cv2.Canny(image_array, low_threshold, high_threshold)
        canny_image = Image.fromarray(canny)
        return canny_image
    
    def generate_image(self, prompt, negative_prompt="", width=1024, height=1024, 
                      num_inference_steps=28, guidance_scale=7.0, seed=-1):
        """สร้างภาพจากข้อความ"""
        if self.txt2img_pipe is None:
            return None, "Model not loaded"
        
        try:
            if seed != -1:
                generator = torch.Generator(device=self.device).manual_seed(int(seed))
            else:
                generator = None
            
            with torch.inference_mode():
                result = self.txt2img_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=int(width),
                    height=int(height),
                    num_inference_steps=int(num_inference_steps),
                    guidance_scale=float(guidance_scale),
                    generator=generator
                )
            
            return result.images[0], "Image generated successfully"
            
        except Exception as e:
            return None, f"Error generating image: {e}"
    
    def generate_from_image(self, prompt, init_image, negative_prompt="", 
                           strength=0.75, num_inference_steps=28, 
                           guidance_scale=7.0, seed=-1, use_controlnet=True,
                           resize_mode="Resize", target_width=1024, target_height=1024):
        """สร้างภาพจากภาพอ้างอิง"""
        if self.img2img_pipe is None:
            return None, "Model not loaded"
        
        try:
            # เตรียมภาพ
            if isinstance(init_image, str):
                init_image = Image.open(init_image)
            elif not isinstance(init_image, Image.Image):
                return None, "Invalid image format"
            
            init_image = init_image.convert("RGB")
            
            # ปรับขนาดภาพตาม mode ที่เลือก
            if resize_mode == "Resize":
                init_image = init_image.resize((target_width, target_height))
            elif resize_mode == "Crop and Resize":
                init_image = ImageOps.fit(init_image, (target_width, target_height), Image.Resampling.LANCZOS)
            elif resize_mode == "Resize and Fill":
                init_image.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
                new_image = Image.new("RGB", (target_width, target_height), (255, 255, 255))
                new_image.paste(init_image, ((target_width - init_image.width) // 2, 
                                           (target_height - init_image.height) // 2))
                init_image = new_image
            
            if seed != -1:
                generator = torch.Generator(device=self.device).manual_seed(int(seed))
            else:
                generator = None
            
            # ใช้ ControlNet ถ้าเปิดใช้งานและมีให้ใช้
            if use_controlnet and self.controlnet_pipe is not None:
                # สร้าง canny edge
                canny_image = self.generate_canny_edge(init_image)
                
                with torch.inference_mode():
                    result = self.controlnet_pipe(
                        prompt=prompt,
                        image=canny_image,
                        negative_prompt=negative_prompt,
                        num_inference_steps=int(num_inference_steps),
                        guidance_scale=float(guidance_scale),
                        controlnet_conditioning_scale=strength,
                        generator=generator
                    )
            else:
                # ใช้ img2img ปกติ
                with torch.inference_mode():
                    result = self.img2img_pipe(
                        prompt=prompt,
                        image=init_image,
                        negative_prompt=negative_prompt,
                        strength=float(strength),
                        num_inference_steps=int(num_inference_steps),
                        guidance_scale=float(guidance_scale),
                        generator=generator
                    )
            
            return result.images[0], "Image generated successfully"
            
        except Exception as e:
            return None, f"Error generating image: {e}"

def create_gradio_interface():
    """สร้าง Gradio web interface"""
    runner = IllustriousXLRunner()
    
    def scan_models_wrapper():
        models = runner.scan_models()
        return (
            gr.update(choices=models["base_models"], value=models["base_models"][0] if models["base_models"] else None),
            gr.update(choices=models["loras"], value=[]),
            gr.update(choices=models["lycoris"], value=[]),
            gr.update(choices=models["controlnets"], value=[])
        )
    
    def load_model_wrapper(base_model, loras, lycoris, controlnets, use_safetensors, scheduler):
        selected_models = {
            "base_model": base_model,
            "loras": loras,
            "lycoris": lycoris,
            "controlnets": controlnets
        }
        success = runner.load_model(selected_models, use_safetensors, scheduler)
        status = "✅ Model loaded successfully!" if success else "❌ Failed to load model"
        if success and runner.loaded_loras:
            status += f"\n📎 Loaded LoRA/LyCORIS: {', '.join(runner.loaded_loras[:3])}"
            if len(runner.loaded_loras) > 3:
                status += f" (+{len(runner.loaded_loras)-3} more)"
        return status
    
    def txt2img_wrapper(prompt, negative_prompt, width, height, steps, guidance, seed):
        if not prompt.strip():
            return None, "Please enter a prompt"
        image, message = runner.generate_image(
            prompt, negative_prompt, width, height, steps, guidance, seed
        )
        return image, message
    
    def img2img_wrapper(prompt, init_image, negative_prompt, strength, steps, guidance, seed, 
                       use_controlnet, resize_mode, target_width, target_height):
        if not prompt.strip():
            return None, "Please enter a prompt"
        if init_image is None:
            return None, "Please upload a reference image"
        image, message = runner.generate_from_image(
            prompt, init_image, negative_prompt, strength, steps, guidance, seed,
            use_controlnet, resize_mode, target_width, target_height
        )
        return image, message
    
    # สไตล์ CSS
    css = """
    .model-loader {
        background: linear-gradient(45deg, #f0f9ff, #e0f2fe);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #0891b2;
        margin-bottom: 20px;
    }
    .compact-button {
        min-width: 120px !important;
        height: 35px !important;
        font-size: 12px !important;
    }
    """
    
    with gr.Blocks(title="Illustrious-XL-v2.0 Advanced Runner", theme=gr.themes.Soft(), css=css) as demo:
        gr.Markdown("# 🎨 Illustrious-XL-v2.0 Advanced Runner")
        gr.Markdown("High-quality anime-style image generation with LoRA, LyCORIS, and ControlNet support")
        
        # Model Loading Section
        with gr.Group(elem_classes="model-loader"):
            gr.Markdown("### 🔧 Model Configuration")
            
            with gr.Row():
                with gr.Column(scale=1):
                    scan_btn = gr.Button("🔍 Scan Models", variant="secondary", elem_classes="compact-button")
                    load_btn = gr.Button("⚡ Load", variant="primary", elem_classes="compact-button")
                    
                with gr.Column(scale=3):
                    load_status = gr.Textbox(
                        label="Status", 
                        interactive=False, 
                        value="Click 'Scan Models' first, then select models and click 'Load'",
                        lines=2
                    )
            
            with gr.Row():
                with gr.Column():
                    base_model_dropdown = gr.Dropdown(
                        label="Base Model",
                        choices=[],
                        value=None,
                        interactive=True
                    )
                    
                with gr.Column():
                    scheduler_dropdown = gr.Dropdown(
                        label="Sampler",
                        choices=["Euler a", "DDIM", "DPM++ 2M Karras", "UniPC", "Heun"],
                        value="Euler a"
                    )
            
            with gr.Row():
                with gr.Column():
                    lora_checkboxes = gr.CheckboxGroup(
                        label="LoRA Models",
                        choices=[],
                        value=[]
                    )
                    
                with gr.Column():
                    lycoris_checkboxes = gr.CheckboxGroup(
                        label="LyCORIS Models", 
                        choices=[],
                        value=[]
                    )
            
            with gr.Row():
                use_safetensors = gr.Checkbox(label="Use SafeTensors", value=True)
        
        # Main Interface
        with gr.Tabs():
            # Text-to-Image Tab
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
                    
                    with gr.Column(scale=1):
                        txt_output = gr.Image(label="Generated Image", height=600)
                        txt_message = gr.Textbox(label="Status", interactive=False)
                
                txt_generate.click(
                    txt2img_wrapper,
                    inputs=[txt_prompt, txt_negative, txt_width, txt_height, txt_steps, txt_guidance, txt_seed],
                    outputs=[txt_output, txt_message]
                )
            
            # Image-to-Image Tab
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
                            use_controlnet = gr.Checkbox(label="Use ControlNet (Face Preservation)", value=True)
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
                    
                    with gr.Column(scale=1):
                        img_output = gr.Image(label="Generated Image", height=600)
                        img_message = gr.Textbox(label="Status", interactive=False)
                
                img_generate.click(
                    img2img_wrapper,
                    inputs=[img_prompt, img_input, img_negative, img_strength, img_steps, img_guidance, img_seed,
                           use_controlnet, resize_mode, target_width, target_height],
                    outputs=[img_output, img_message]
                )
        
        # Tips section
        with gr.Accordion("💡 Tips & Model Organization", open=False):
            gr.Markdown("""
            ### 📁 Model Folder Structure:
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
            
            ### 🎛️ Sampler Guide:
            - **Euler a**: Fast, good quality, good for most cases
            - **DDIM**: Stable, deterministic results
            - **DPM++ 2M Karras**: High quality, slower
            - **UniPC**: Fast convergence, good for low steps
            - **Heun**: High quality, more steps needed
            
            ### 🎨 ControlNet Features:
            - **Face Preservation**: Keeps facial features from reference
            - **Canny Edge Detection**: Preserves structural details
            - **Best for**: Character pose/expression preservation
            
            ### 💡 Usage Tips:
            - Use ControlNet for face preservation in img2img
            - Lower strength (0.3-0.5) for subtle changes
            - Higher strength (0.7-0.9) for major modifications
            - Combine multiple LoRAs for unique styles
            """)
        
        # Event handlers
        scan_btn.click(
            scan_models_wrapper,
            outputs=[base_model_dropdown, lora_checkboxes, lycoris_checkboxes]
        )
        
        load_btn.click(
            load_model_wrapper,
            inputs=[base_model_dropdown, lora_checkboxes, lycoris_checkboxes, 
                   use_safetensors, scheduler_dropdown],
            outputs=[load_status]
        )
    
    return demo

if __name__ == "__main__":
    # Create web interface
    demo = create_gradio_interface()
    
    # Launch the interface
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )