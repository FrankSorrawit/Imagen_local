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
import os
from pathlib import Path
import json
import cv2
import numpy as np
import gc
from datetime import datetime
import uuid

# Compel for long prompts and weight control
try:
    from compel import Compel, ReturnedEmbeddingsType
    COMPEL_AVAILABLE = True
    print("✅ Compel library loaded - Long prompts and weight control enabled")
except ImportError:
    COMPEL_AVAILABLE = False
    print("⚠️ Compel library not found. Install with: pip install compel")

class IllustriousEngine:
    """Core image generation engine for Illustrious-XL models"""
    
    def __init__(self, model_folder="./Model", presets_folder="./Presets"):
        # Paths
        self.model_folder = Path(model_folder)
        self.presets_folder = Path(presets_folder)
        self.presets_folder.mkdir(exist_ok=True)
        
        # Pipeline components
        self.txt2img_pipe = None
        self.img2img_pipe = None
        self.controlnet_pipe = None
        
        # Device and VRAM info
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vram_gb = 0
        if torch.cuda.is_available():
            self.vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"VRAM Available: {self.vram_gb:.1f} GB")
        else:
            print("CUDA not available, using CPU")
        
        # Model management
        self.loaded_loras = []
        self.available_models = {}
        self.current_scheduler = None
        self.current_model_info = {}
        
        # Compel for long prompts and weight control
        self.compel = None
        self.compel_neg = None
        
    def clear_memory(self):
        """Clear GPU memory and cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        
    def scan_models(self):
        """Scan for all models in folder"""
        models = {
            "base_models": [],
            "loras": [],
            "lycoris": [],
            "controlnets": []
        }
        
        if not self.model_folder.exists():
            self.model_folder.mkdir(parents=True, exist_ok=True)
            return models
            
        # Scan files in folder
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
            
        self.available_models = models
        return models
    
    def get_scheduler(self, scheduler_name):
        """Create scheduler by name"""
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
    
    def encode_prompt_with_compel(self, prompt, negative_prompt=""):
        """
        Encode prompts using Compel for long prompt support and weight control
        
        Supported Syntax:
        - (keyword) = 1.1x stronger
        - ((keyword)) = 1.21x stronger  
        - (keyword:1.5) = 1.5x stronger
        - [keyword] = 0.9x weaker
        - [[keyword]] = 0.81x weaker
        - [keyword:0.7] = 0.7x weaker
        - (concept1:1.2) AND (concept2:0.8) = separate concepts
        
        Examples:
        - "(masterpiece:1.3), (best quality:1.2), 1girl, [bad anatomy:0.5]"
        - "(anime style:1.4) AND (detailed background:1.1), beautiful landscape"
        """
        if not self.compel or not self.compel_neg:
            print("⚠️ Compel not initialized")
            return None, None, None, None
            
        try:
            # Encode positive prompt
            if prompt.strip():
                conditioning, pooled = self.compel(prompt)
            else:
                conditioning, pooled = self.compel("")
            
            # Encode negative prompt
            if negative_prompt.strip():
                negative_conditioning, negative_pooled = self.compel_neg(negative_prompt)
            else:
                negative_conditioning, negative_pooled = self.compel_neg("")
            
            # Validate results
            if conditioning is None or negative_conditioning is None:
                print("⚠️ Compel encoding returned None values")
                return None, None, None, None
                
            print(f"✅ Compel encoding successful - Prompt tokens: ~{len(prompt.split())}")
            return conditioning, negative_conditioning, pooled, negative_pooled
            
        except Exception as e:
            print(f"⚠️ Compel encoding failed: {e}")
            return None, None, None, None

    def optimize_for_vram(self, pipe):
        """Optimize pipeline for available VRAM"""
        if self.device != "cuda":
            return pipe
            
        # Clear memory first
        self.clear_memory()
        
        # Apply optimizations based on VRAM
        if self.vram_gb >= 12:
            # High VRAM - full optimization
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("✅ xformers enabled (High VRAM mode)")
            except:
                print("⚠️ xformers not available")
                
        elif self.vram_gb >= 6:
            # Medium VRAM (6-12GB) - Compel-compatible optimization
            try:
                pipe.enable_xformers_memory_efficient_attention()
                pipe.enable_model_cpu_offload()
                print("✅ xformers + CPU offload enabled (Medium VRAM mode - Compel compatible)")
            except:
                try:
                    pipe.enable_model_cpu_offload()
                    pipe.enable_attention_slicing()
                    print("✅ CPU offload + attention slicing (Medium VRAM mode - Compel compatible)")
                except:
                    try:
                        pipe.enable_attention_slicing()
                        print("✅ Attention slicing enabled (Medium VRAM mode)")
                    except:
                        print("⚠️ Optimization failed")
                    
        else:
            # Low VRAM (<6GB) - aggressive optimization (Compel disabled)
            try:
                pipe.enable_sequential_cpu_offload()
                pipe.enable_attention_slicing()
                print("✅ Sequential CPU offload + attention slicing (Low VRAM mode - Compel disabled)")
            except:
                try:
                    pipe.enable_attention_slicing()
                    print("✅ Attention slicing enabled (Low VRAM mode)")
                except:
                    print("⚠️ Low VRAM optimizations failed")
        
        return pipe

    def setup_compel(self):
        """Setup Compel for long prompts and weight control"""
        if not COMPEL_AVAILABLE or self.txt2img_pipe is None:
            print("⚠️ Compel not available or pipeline not loaded")
            return False
        
        # Check if using sequential CPU offload (incompatible with Compel)
        if self.vram_gb < 6:
            print("⚠️ Compel disabled for <6GB VRAM (sequential CPU offload mode)")
            self.compel = None
            self.compel_neg = None
            return False
            
        try:
            # Setup Compel for positive prompts
            self.compel = Compel(
                tokenizer=[self.txt2img_pipe.tokenizer, self.txt2img_pipe.tokenizer_2],
                text_encoder=[self.txt2img_pipe.text_encoder, self.txt2img_pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True]
            )
            
            # Setup Compel for negative prompts  
            self.compel_neg = Compel(
                tokenizer=[self.txt2img_pipe.tokenizer, self.txt2img_pipe.tokenizer_2],
                text_encoder=[self.txt2img_pipe.text_encoder, self.txt2img_pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True]
            )
            
            # Test Compel with a simple prompt
            test_result = self.encode_prompt_with_compel("test", "")
            if test_result and test_result[0] is not None:
                print(f"✅ Compel setup complete - Long prompts enabled (VRAM: {self.vram_gb:.1f}GB)")
                return True
            else:
                print("⚠️ Compel test failed - falling back to regular prompts")
                self.compel = None
                self.compel_neg = None
                return False
                
        except Exception as e:
            print(f"⚠️ Compel setup failed: {e} - falling back to regular prompts")
            self.compel = None
            self.compel_neg = None
            return False
    
    def load_model(self, selected_models, use_safetensors=True, scheduler_name="Euler a"):
        """Load selected model and components"""
        print(f"Loading selected models...")
        print(f"Using device: {self.device} (VRAM: {self.vram_gb:.1f} GB)")
        
        # Clear previous models
        self.clear_memory()
        
        try:
            # Load base model
            base_model = selected_models.get("base_model")
            if not base_model:
                return False, "Please select a base model"
                
            model_path = self.model_folder / base_model
            
            if not model_path.exists():
                return False, f"Model file not found: {model_path}"
            
            print(f"Loading base model: {model_path}")
            
            # Determine loading method based on file format
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
            
            # Set scheduler
            if scheduler_name != "Default":
                self.txt2img_pipe.scheduler = self.get_scheduler(scheduler_name)
                self.current_scheduler = scheduler_name
            
            # Create img2img pipeline (sharing components to save VRAM)
            self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
                vae=self.txt2img_pipe.vae,
                text_encoder=self.txt2img_pipe.text_encoder,
                text_encoder_2=self.txt2img_pipe.text_encoder_2,
                tokenizer=self.txt2img_pipe.tokenizer,
                tokenizer_2=self.txt2img_pipe.tokenizer_2,
                unet=self.txt2img_pipe.unet,
                scheduler=self.txt2img_pipe.scheduler,
            )
            
            # Load ControlNet for img2img (optional)
            try:
                controlnet = ControlNetModel.from_pretrained(
                    "diffusers/controlnet-canny-sdxl-1.0",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    cache_dir="./cache"  # Use local cache
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
                print("✅ ControlNet loaded successfully")
            except Exception as e:
                print(f"⚠️ ControlNet loading failed: {e}")
                self.controlnet_pipe = None
            
            # Move to GPU and optimize
            if self.device == "cuda":
                self.txt2img_pipe = self.txt2img_pipe.to(self.device)
                self.img2img_pipe = self.img2img_pipe.to(self.device)
                if self.controlnet_pipe:
                    self.controlnet_pipe = self.controlnet_pipe.to(self.device)
                
                # Apply VRAM optimizations
                self.txt2img_pipe = self.optimize_for_vram(self.txt2img_pipe)
                self.img2img_pipe = self.optimize_for_vram(self.img2img_pipe)
                if self.controlnet_pipe:
                    self.controlnet_pipe = self.optimize_for_vram(self.controlnet_pipe)
            
            # Setup Compel for long prompts
            self.setup_compel()
            
            # Load LoRA models
            self.loaded_loras = []
            loras = selected_models.get("loras", [])
            for lora_path in loras:
                try:
                    full_lora_path = self.model_folder / lora_path
                    if full_lora_path.exists():
                        # Try different LoRA loading methods
                        adapter_name = Path(lora_path).stem
                        try:
                            # Method 1: Standard LoRA loading
                            self.txt2img_pipe.load_lora_weights(
                                str(full_lora_path), 
                                adapter_name=adapter_name
                            )
                            self.loaded_loras.append(lora_path)
                            print(f"✅ LoRA loaded: {lora_path}")
                        except Exception as e1:
                            try:
                                # Method 2: Load without adapter name
                                self.txt2img_pipe.load_lora_weights(str(full_lora_path))
                                self.loaded_loras.append(lora_path)
                                print(f"✅ LoRA loaded (fallback): {lora_path}")
                            except Exception as e2:
                                print(f"❌ Failed to load LoRA {lora_path}: {e1}")
                                print(f"   Fallback also failed: {e2}")
                    else:
                        print(f"⚠️ LoRA file not found: {lora_path}")
                except Exception as e:
                    print(f"❌ Failed to load LoRA {lora_path}: {e}")
            
            # Load LyCORIS models
            lycoris = selected_models.get("lycoris", [])
            for lyco_path in lycoris:
                try:
                    full_lyco_path = self.model_folder / lyco_path
                    if full_lyco_path.exists():
                        adapter_name = Path(lyco_path).stem
                        try:
                            # Method 1: Standard loading
                            self.txt2img_pipe.load_lora_weights(
                                str(full_lyco_path),
                                adapter_name=adapter_name
                            )
                            self.loaded_loras.append(lyco_path)
                            print(f"✅ LyCORIS loaded: {lyco_path}")
                        except Exception as e1:
                            try:
                                # Method 2: Load without adapter name
                                self.txt2img_pipe.load_lora_weights(str(full_lyco_path))
                                self.loaded_loras.append(lyco_path)
                                print(f"✅ LyCORIS loaded (fallback): {lyco_path}")
                            except Exception as e2:
                                print(f"❌ Failed to load LyCORIS {lyco_path}: {e1}")
                                print(f"   Fallback also failed: {e2}")
                    else:
                        print(f"⚠️ LyCORIS file not found: {lyco_path}")
                except Exception as e:
                    print(f"❌ Failed to load LyCORIS {lyco_path}: {e}")
            
            # Store current model info
            self.current_model_info = {
                "base_model": base_model,
                "loras": selected_models.get("loras", []),
                "lycoris": selected_models.get("lycoris", []),
                "scheduler": scheduler_name,
                "use_safetensors": use_safetensors
            }
            
            print("✅ Model loaded successfully!")
            return True, "Model loaded successfully"
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False, f"Error loading model: {e}"
    
    def generate_canny_edge(self, image, low_threshold=100, high_threshold=200):
        """Generate Canny edge detection for ControlNet"""
        image_array = np.array(image)
        canny = cv2.Canny(image_array, low_threshold, high_threshold)
        canny_image = Image.fromarray(canny)
        return canny_image
    
    def generate_image(self, prompt, negative_prompt="", width=1024, height=1024, 
                      num_inference_steps=28, guidance_scale=7.0, seed=-1):
        """Generate image from text with long prompt support"""
        if self.txt2img_pipe is None:
            return None, "Model not loaded. Please load a model first.", None
        
        try:
            # Clear memory before generation
            self.clear_memory()
            
            # Debug: Check prompt length
            prompt_tokens = len(prompt.split())
            negative_tokens = len(negative_prompt.split())
            print(f"🔍 Prompt analysis:")
            print(f"   - Positive prompt: {prompt_tokens} tokens")
            print(f"   - Negative prompt: {negative_tokens} tokens")
            print(f"   - Compel available: {self.compel is not None}")
            
            if seed != -1:
                generator = torch.Generator(device=self.device).manual_seed(int(seed))
                used_seed = int(seed)
            else:
                used_seed = torch.randint(0, 2**32-1, (1,)).item()
                generator = torch.Generator(device=self.device).manual_seed(used_seed)
            
            # Try to use Compel for long prompt support
            use_compel = False
            conditioning, negative_conditioning, pooled, negative_pooled = None, None, None, None
            
            if self.compel and self.compel_neg:
                print("🎯 Attempting Compel encoding...")
                
                # Encode prompts with Compel
                try:
                    result = self.encode_prompt_with_compel(prompt, negative_prompt)
                    if result and len(result) == 4:
                        conditioning, negative_conditioning, pooled, negative_pooled = result
                        if conditioning is not None and negative_conditioning is not None:
                            use_compel = True
                            print("✅ Compel encoding successful, using Compel for long prompts")
                        else:
                            print("⚠️ Compel returned None values, falling back to regular prompts")
                    else:
                        print("⚠️ Compel returned invalid result, falling back to regular prompts")
                except Exception as e:
                    print(f"⚠️ Compel encoding error: {e}, falling back to regular prompts")
            else:
                print("⚠️ Compel not available, using regular prompts")
            
            # Generate image based on available prompt encoding
            if use_compel:
                print("🎨 Generating with Compel-encoded prompts (No token limit!)")
                with torch.inference_mode():
                    result = self.txt2img_pipe(
                        prompt_embeds=conditioning,
                        negative_prompt_embeds=negative_conditioning,
                        pooled_prompt_embeds=pooled,
                        negative_pooled_prompt_embeds=negative_pooled,
                        width=int(width),
                        height=int(height),
                        num_inference_steps=int(num_inference_steps),
                        guidance_scale=float(guidance_scale),
                        generator=generator
                    )
            else:
                # Fallback to regular prompts (77 token limit)
                print("🎨 Generating with regular prompts (77 token limit)")
                if prompt_tokens > 75 or negative_tokens > 75:
                    print("⚠️ WARNING: Prompt will be truncated!")
                
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
            
            # Clear memory after generation
            self.clear_memory()
            
            # Prepare generation info
            generation_info = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": int(width),
                "height": int(height),
                "num_inference_steps": int(num_inference_steps),
                "guidance_scale": float(guidance_scale),
                "seed": used_seed,
                "model_info": self.current_model_info.copy(),
                "generation_type": "txt2img",
                "timestamp": datetime.now().isoformat(),
                "compel_enabled": self.compel is not None,
                "used_compel": use_compel,
                "prompt_tokens": prompt_tokens,
                "negative_tokens": negative_tokens
            }
            
            success_msg = "✅ Image generated successfully"
            if use_compel:
                success_msg += f" (Long prompts: {prompt_tokens} tokens)"
            else:
                success_msg += f" (Regular mode: {prompt_tokens} tokens)"
            
            return result.images[0], success_msg, generation_info
            
        except Exception as e:
            self.clear_memory()
            return None, f"❌ Error generating image: {e}", None
    
    def generate_from_image(self, prompt, init_image, negative_prompt="", 
                           strength=0.75, num_inference_steps=28, 
                           guidance_scale=7.0, seed=-1, use_controlnet=True,
                           resize_mode="Resize", target_width=1024, target_height=1024):
        """Generate image from reference image with long prompt support"""
        if self.img2img_pipe is None:
            return None, "Model not loaded. Please load a model first.", None
        
        try:
            # Clear memory before generation
            self.clear_memory()
            
            # Debug: Check prompt length
            prompt_tokens = len(prompt.split())
            negative_tokens = len(negative_prompt.split())
            print(f"🔍 IMG2IMG Prompt analysis:")
            print(f"   - Positive prompt: {prompt_tokens} tokens")
            print(f"   - Negative prompt: {negative_tokens} tokens")
            print(f"   - Compel available: {self.compel is not None}")
            
            # Prepare image
            if isinstance(init_image, str):
                init_image = Image.open(init_image)
            elif not isinstance(init_image, Image.Image):
                return None, "Invalid image format", None
            
            init_image = init_image.convert("RGB")
            
            # Resize image based on selected mode
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
                used_seed = int(seed)
            else:
                used_seed = torch.randint(0, 2**32-1, (1,)).item()
                generator = torch.Generator(device=self.device).manual_seed(used_seed)
            
            # Use ControlNet if enabled and available
            if use_controlnet and self.controlnet_pipe is not None:
                print("🎯 Using ControlNet...")
                
                # Generate canny edge
                canny_image = self.generate_canny_edge(init_image)
                
                # Try to use Compel for long prompt support
                use_compel = False
                conditioning, negative_conditioning, pooled, negative_pooled = None, None, None, None
                
                if self.compel and self.compel_neg:
                    print("🎯 Attempting Compel encoding for ControlNet...")
                    try:
                        result_compel = self.encode_prompt_with_compel(prompt, negative_prompt)
                        if result_compel and len(result_compel) == 4:
                            conditioning, negative_conditioning, pooled, negative_pooled = result_compel
                            if conditioning is not None and negative_conditioning is not None:
                                use_compel = True
                                print("✅ Using Compel with ControlNet")
                    except Exception as e:
                        print(f"⚠️ Compel encoding failed for ControlNet: {e}")
                
                if use_compel:
                    print("🎨 Generating with ControlNet + Compel (No token limit!)")
                    with torch.inference_mode():
                        result = self.controlnet_pipe(
                            prompt_embeds=conditioning,
                            negative_prompt_embeds=negative_conditioning,
                            pooled_prompt_embeds=pooled,
                            negative_pooled_prompt_embeds=negative_pooled,
                            image=canny_image,
                            num_inference_steps=int(num_inference_steps),
                            guidance_scale=float(guidance_scale),
                            controlnet_conditioning_scale=strength,
                            generator=generator
                        )
                else:
                    print("🎨 Generating with ControlNet + regular prompts (77 token limit)")
                    if prompt_tokens > 75 or negative_tokens > 75:
                        print("⚠️ WARNING: Prompt will be truncated!")
                    
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
                # Use regular img2img
                print("📝 Using img2img...")
                
                # Try to use Compel for long prompt support
                use_compel = False
                conditioning, negative_conditioning, pooled, negative_pooled = None, None, None, None
                
                if self.compel and self.compel_neg:
                    print("🎯 Attempting Compel encoding for img2img...")
                    try:
                        result_compel = self.encode_prompt_with_compel(prompt, negative_prompt)
                        if result_compel and len(result_compel) == 4:
                            conditioning, negative_conditioning, pooled, negative_pooled = result_compel
                            if conditioning is not None and negative_conditioning is not None:
                                use_compel = True
                                print("✅ Using Compel with img2img")
                    except Exception as e:
                        print(f"⚠️ Compel encoding failed for img2img: {e}")
                
                if use_compel:
                    print("🎨 Generating with img2img + Compel (No token limit!)")
                    with torch.inference_mode():
                        result = self.img2img_pipe(
                            prompt_embeds=conditioning,
                            negative_prompt_embeds=negative_conditioning,
                            pooled_prompt_embeds=pooled,
                            negative_pooled_prompt_embeds=negative_pooled,
                            image=init_image,
                            strength=float(strength),
                            num_inference_steps=int(num_inference_steps),
                            guidance_scale=float(guidance_scale),
                            generator=generator
                        )
                else:
                    print("🎨 Generating with img2img + regular prompts (77 token limit)")
                    if prompt_tokens > 75 or negative_tokens > 75:
                        print("⚠️ WARNING: Prompt will be truncated!")
                    
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
            
            # Clear memory after generation
            self.clear_memory()
            
            # Prepare generation info
            generation_info = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "strength": float(strength),
                "num_inference_steps": int(num_inference_steps),
                "guidance_scale": float(guidance_scale),
                "seed": used_seed,
                "use_controlnet": use_controlnet,
                "resize_mode": resize_mode,
                "target_width": target_width,
                "target_height": target_height,
                "model_info": self.current_model_info.copy(),
                "generation_type": "img2img",
                "timestamp": datetime.now().isoformat(),
                "compel_enabled": self.compel is not None,
                "used_compel": use_compel,
                "prompt_tokens": prompt_tokens,
                "negative_tokens": negative_tokens
            }
            
            success_msg = "✅ Image generated successfully"
            if use_compel:
                success_msg += f" (Long prompts: {prompt_tokens} tokens)"
            else:
                success_msg += f" (Regular mode: {prompt_tokens} tokens)"
            
            return result.images[0], success_msg, generation_info
            
        except Exception as e:
            self.clear_memory()
            return None, f"❌ Error generating image: {e}", None
    
    # Preset Management Methods
    def save_preset(self, preset_name, image, generation_info):
        """Save generation as preset"""
        try:
            # Generate unique ID
            preset_id = str(uuid.uuid4())[:8]
            
            # Create preset data
            preset_data = {
                "id": preset_id,
                "name": preset_name,
                "creation_date": datetime.now().isoformat(),
                "generation_info": generation_info
            }
            
            # Convert image to PIL Image if it's a numpy array
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                # If it's neither numpy array nor PIL Image, try to convert
                try:
                    image = Image.fromarray(np.array(image))
                except:
                    return False, "❌ Invalid image format for saving preset"
            
            # Save image thumbnail
            image_path = self.presets_folder / f"{preset_id}.png"
            # Create thumbnail (256x256)
            thumbnail = image.copy()
            thumbnail.thumbnail((256, 256), Image.Resampling.LANCZOS)
            thumbnail.save(image_path, "PNG")
            
            # Save preset JSON
            json_path = self.presets_folder / f"{preset_id}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(preset_data, f, indent=2, ensure_ascii=False)
            
            return True, f"✅ Preset '{preset_name}' saved successfully"
            
        except Exception as e:
            return False, f"❌ Error saving preset: {e}"
    
    def load_presets(self):
        """Load all available presets"""
        presets = []
        
        try:
            for json_file in self.presets_folder.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        preset_data = json.load(f)
                    
                    # Check if image file exists
                    image_path = self.presets_folder / f"{preset_data['id']}.png"
                    if image_path.exists():
                        preset_data['image_path'] = str(image_path)
                        presets.append(preset_data)
                        
                except Exception as e:
                    print(f"Error loading preset {json_file}: {e}")
                    continue
            
            # Sort by creation date (newest first)
            presets.sort(key=lambda x: x.get('creation_date', ''), reverse=True)
            
        except Exception as e:
            print(f"Error loading presets: {e}")
        
        return presets
    
    def delete_preset(self, preset_id):
        """Delete a preset"""
        try:
            json_path = self.presets_folder / f"{preset_id}.json"
            image_path = self.presets_folder / f"{preset_id}.png"
            
            if json_path.exists():
                json_path.unlink()
            if image_path.exists():
                image_path.unlink()
                
            return True, "✅ Preset deleted successfully"
            
        except Exception as e:
            return False, f"❌ Error deleting preset: {e}"
    
    def get_preset_by_id(self, preset_id):
        """Get specific preset by ID"""
        try:
            json_path = self.presets_folder / f"{preset_id}.json"
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading preset {preset_id}: {e}")
        return None
    
    def get_status(self):
        """Get current engine status"""
        status = {
            "device": self.device,
            "vram_gb": self.vram_gb,
            "model_loaded": self.txt2img_pipe is not None,
            "current_model": self.current_model_info,
            "loaded_loras": self.loaded_loras,
            "available_models": self.available_models
        }
        return status