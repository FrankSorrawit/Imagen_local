import os

base_path = os.path.join(os.getcwd(), "Model")

structure = {
    "base_models": ["your_model.safetensors"],
    "lora": ["character_lora.safetensors", "style_lora.safetensors"],
    "lycoris": ["your_lycoris.safetensors"],
    "controlnet": ["control_model.safetensors"]
}

for folder, files in structure.items():
    dir_path = os.path.join(base_path, folder)
    os.makedirs(dir_path, exist_ok=True)
    for file in files:
        file_path = os.path.join(dir_path, file)
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                pass

print("Model directory structure created in your project folder.")