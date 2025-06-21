### Prompt Structure:
1. Quality tags first: `masterpiece, best quality`
2. Character description: `1girl, beautiful face, long hair`
3. Clothing & pose: `school uniform, sitting`
4. Style & mood: `anime style, soft lighting`
5. Background: `detailed background, classroom`

### Weight Guidelines:
- Important elements: 1.2-1.4
- Style emphasis: 1.1-1.3  
- Negative suppression: 0.5-0.8
- Avoid over-we# 🎨 Illustrious-XL Runner - Help Guide

## ⚠️ DISCLAIMER / ข้อปฏิเสธความรับผิดชอบ

**🎓 การใช้งานเพื่อการศึกษาเท่านั้น**
- โปรเจคนี้พัฒนาขึ้นเพื่อการศึกษาและการวิจัยด้าน AI Image Generation เท่านั้น
- **ไม่ได้มีวัตถุประสงค์เพื่อการค้า** หรือการหาผลกำไรใดๆ
- **ไม่ได้มีเจตนาทำผิดกฎหมาย** หรือละเมิดลิขสิทธิ์ของผู้อื่น

**📋 ข้อควรปฏิบัติ:**
- ผู้ใช้งานควรปฏิบัติตามกฎหมายของประเทศของตนเอง
- ไม่ควรสร้างภาพที่ละเมิดลิขสิทธิ์หรือสิทธิบุคคล
- ไม่ควรสร้างเนื้อหาที่ไม่เหมาะสมหรือผิดกฎหมาย
- การใช้โมเดล AI ควรเป็นไปตามเงื่อนไขของผู้พัฒนาโมเดล

**🔒 ความรับผิดชอบ:**
- ผู้พัฒนาโปรเจคนี้ไม่รับผิดชอบต่อการใช้งานที่ผิดกฎหมายหรือไม่เหมาะสม
- ผู้ใช้งานต้องรับผิดชอบการใช้งานของตนเองทั้งหมด

---

## 🎯 Long Prompt & Weight Control (Compel)

### Basic Weight Syntax:
- `(keyword)` = เน้น 10% เพิ่ม (1.1x)
- `((keyword))` = เน้น 21% เพิ่ม (1.21x)
- `(keyword:1.5)` = เน้น 50% เพิ่ม (1.5x)
- `[keyword]` = ลด 10% (0.9x)
- `[[keyword]]` = ลด 19% (0.81x)
- `[keyword:0.7]` = ลด 30% (0.7x)

### Advanced Syntax:
- `(concept1:1.2) AND (concept2:0.8)` = แยก concepts
- `masterpiece, (detailed face:1.3), [bad anatomy:0.5]` = ผสม weight

### Practical Examples:

#### Portrait:
```
(masterpiece:1.3), (best quality:1.2), (1girl:1.1), 
(beautiful face:1.3), (detailed eyes:1.2), long hair,
[bad anatomy:0.5], [blurry:0.7]
```

#### Landscape:

```
(fantasy landscape:1.4), (magical forest:1.2), 
(golden hour lighting:1.3) AND (cinematic:1.1),
[oversaturated:0.6]
```

#### Style Control:
```
(anime style:1.3), (studio ghibli:1.2) AND (makoto shinkai:0.8),
(vibrant colors:1.2), [realistic:0.3]
```

### Tips:
- 📝 No Token Limit: ใช้ prompt ยาวได้ไม่จำกัด
- 🎯 Weight Range: แนะนำ 0.5-1.5 สำหรับผลลัพธ์ที่ดี
- ⚖️ Balance: ใช้ positive weight กับ negative weight สมดุล
- 🔗 AND: ใช้แยก concepts ที่ไม่ต้องการให้ blend กัน

## 🏗️ Modular Architecture:
- illustrious_engine.py: Core image generation engine
- web_interface.py: Gradio web interface
- ./Model/: Model storage folder
- ./Presets/: Preset storage folder

## 📁 Model Folder Structure:
```
./Model/
├── base_model.safetensors
├── another_model.ckpt
├── lora/
│   ├── character_lora.safetensors
│   └── style_lora.safetensors
├── lycoris/
│   └── your_lycoris.safetensors
└── controlnet/
    └── control_model.safetensors
```

## 🎯 Preset System:
- Save: Generate image → Enter preset name → Save
- Load: Go to Preset Manager → Select preset → Load to Generation
- Manage: View all presets with thumbnails and parameters

## 🔧 VRAM Optimizations:
- High VRAM (12GB+): Full optimization with xformers
- Medium VRAM (8-12GB): xformers + CPU offload
- Low VRAM (<8GB): Sequential CPU offload + attention slicing

## 💡 Performance Tips:
- Use preset system to save favorite settings
- Lower resolution for faster generation
- Reduce steps for quicker results (15-20 for testing)
- Close other GPU-intensive applications

## 📦 Installation Requirements:
```bash
pip install compel  # For long prompts and weight control
```

## 🎨 Character & Style Prompts:

### Character Tags:
- `1girl, 1boy, multiple girls, multiple boys`
- `solo, duo, group`
- `loli, shota, mature, elderly`

### Face & Expression:
- `beautiful face, cute face, detailed face`
- `smile, laugh, serious, angry, sad, surprised, embarrassed`
- `eyes closed, wink, looking at viewer, looking away`

### Hair:
- `long hair, short hair, medium hair, very long hair`
- `straight hair, wavy hair, curly hair, messy hair`
- `blonde hair, brown hair, black hair, white hair, pink hair, blue hair`
- `twin tails, ponytail, braided hair, hair bun`

### Eyes:
- `detailed eyes, beautiful eyes, sparkling eyes`
- `blue eyes, green eyes, brown eyes, red eyes, purple eyes`
- `heterochromia, gradient eyes`

### Clothing:
- `school uniform, casual clothes, formal wear, kimono, dress`
- `t-shirt, hoodie, sweater, jacket, coat`
- `skirt, pants, shorts, jeans`

### Art Styles:
- `anime style, manga style, realistic, semi-realistic`
- `studio ghibli style, makoto shinkai style, kyoto animation style`
- `cel shading, soft shading, hard shading`

### Quality Tags:
- `masterpiece, best quality, high quality, ultra-detailed`
- `highly detailed, extremely detailed, intricate details`
- `8k wallpaper, official art, illustration`

### Lighting:
- `natural lighting, soft lighting, dramatic lighting`
- `golden hour, sunset, sunrise, moonlight`
- `rim lighting, backlighting, studio lighting`

### Background:
- `simple background, white background, transparent background`
- `detailed background, outdoor, indoor, nature, city`
- `school, bedroom, kitchen, park, beach, forest`

### Camera & Composition:
- `portrait, close-up, medium shot, full body`
- `from above, from below, from side, from behind`
- `depth of field, bokeh, focus blur`

## 🚫 Common Negative Prompts:
```
lowres, bad anatomy, bad hands, text, error, missing fingers, 
extra digit, fewer digits, cropped, worst quality, low quality, 
normal quality, jpeg artifacts, signature, watermark, username, 
blurry, bad feet, bad proportions, extra limbs, disfigured, 
deformed, mutation, mutated, ugly, disgusting, amputation
```

## 📝 Tips for Better Results:

### **Prompt Structure:**
1. **Quality tags first**: `masterpiece, best quality`
2. **Character description**: `1girl, beautiful face, long hair`
3. **Clothing & pose**: `school uniform, sitting`
4. **Style & mood**: `anime style, soft lighting`
5. **Background**: `detailed background, classroom`

### **Weight Guidelines:**
- **Important elements**: 1.2-1.4
- **Style emphasis**: 1.1-1.3  
- **Negative suppression**: 0.5-0.8
- **Avoid over-weighting**: Stay under 1.5

### **Common Issues:**
- **Blurry results**: Add `(sharp focus:1.2), detailed`
- **Wrong anatomy**: Strengthen negative prompts for anatomy
- **Style not working**: Increase style weight `(anime style:1.3)`
- **Too realistic**: Add `[realistic:0.3]` to negative area

### **Optimization Tips:**
- Use shorter prompts for faster generation
- Test with low steps (15-20) first
- Save successful prompts as presets
- Experiment with different samplers for various styles