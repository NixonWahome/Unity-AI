import torch
import cv2
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
from svgwrite import Drawing
import os
import time
from PIL import Image
import warnings
#run the download.py first.
# Suppress all warnings
warnings.filterwarnings("ignore")

# Configuration - Using smaller models for CPU
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-scribble"
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "cpu_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Force CPU-only mode
device = "cpu"
torch_dtype = torch.float32  # Must use float32 on CPU

def initialize_pipeline():
    """Initialize models with robust error handling"""
    try:
        print("\nInitializing models (this may take 5-10 minutes on first run)...")
        
        # Initialize with explicit CPU settings
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_MODEL,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            BASE_MODEL,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        return pipe
        
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)[:500]}")
        print("\nCRITICAL FIX REQUIRED:")
        print("1. Delete the 'models' folder if it exists")
        print("2. Run these commands:")
        print("   git lfs install")
        print(f"   git clone https://huggingface.co/{BASE_MODEL} ./models/stable-diffusion-v1-5")
        print(f"   git clone https://huggingface.co/{CONTROLNET_MODEL} ./models/controlnet-scribble")
        print("3. Try again after downloads complete")
        exit()

def create_sketch():
    """Create smaller sketch for CPU compatibility"""
    size = 256  # Reduced size for CPU
    sketch = np.zeros((size, size, 3), dtype=np.uint8)
    margin = 30
    cv2.rectangle(sketch, (margin, margin), (size-margin, size-margin), (255,255,255), 2)
    cv2.line(sketch, (size//2, margin), (size//2, size-margin), (255,255,255), 1)
    cv2.line(sketch, (margin, size//2), (size-margin, size//2), (255,255,255), 1)
    return sketch

def generate_floorplan(pipe, prompt):
    """Simplified generation for CPU"""
    try:
        sketch = create_sketch()
        sketch_path = os.path.join(OUTPUT_DIR, "sketch.png")
        cv2.imwrite(sketch_path, sketch)
        
        print("\nGenerating (this will take 5-15 minutes on CPU)...")
        
        image = pipe(
            f"{prompt}, clean architectural floor plan, no furniture, monochrome line drawing",
            image=load_image(sketch_path),
            num_inference_steps=15,  # Reduced for CPU
            guidance_scale=7.0,
            negative_prompt="text, colors, furniture, messy",
            generator=torch.Generator(device="cpu")
        ).images[0]
        
        image = image.convert("L")
        img_np = np.array(image)
        _, binary = cv2.threshold(img_np, 150, 255, cv2.THRESH_BINARY_INV)
        return binary
        
    except Exception as e:
        print(f"\nGeneration failed: {str(e)[:500]}")
        print("Try reducing the sketch size further in create_sketch()")
        exit()

def main():
    print("\n" + "="*50)
    print("CPU-ONLY FLOOR PLAN GENERATOR".center(50))
    print("="*50)
    
    pipe = initialize_pipeline()
    prompt = input("\nDescribe your floor plan (e.g. '2 bedroom modern house'):\n> ")
    
    # Generate
    timestamp = str(int(time.time()))
    floorplan = generate_floorplan(pipe, prompt)
    output_png = os.path.join(OUTPUT_DIR, f"floorplan_{timestamp}.png")
    cv2.imwrite(output_png, floorplan)
    
    print("\n" + "="*50)
    print("SUCCESS!".center(50))
    print("="*50)
    print(f"\nFloor plan saved to:\n{os.path.abspath(output_png)}")
    print("\nNext steps:")
    print("1. Open the PNG in Inkscape")
    print("2. Use Path > Trace Bitmap to convert to SVG")
    print("3. Import SVG into Blender for 3D conversion")

if __name__ == "__main__":
    main()
