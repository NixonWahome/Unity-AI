import torch
import cv2
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from svgwrite import Drawing, shapes
import subprocess
from PIL import Image
import os
import time
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-scribble"
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "generated_floorplans"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32
print(f"\nRunning on: {device.upper()} with {torch_dtype} precision")

def initialize_models():
    """Initialize models with error handling and memory management"""
    try:
        print("\nInitializing models... (This may take several minutes)")
        
        # Initialize ControlNet
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_MODEL,
            torch_dtype=torch_dtype,
            cache_dir="./models",
            local_files_only=False
        ).to(device)

        # Initialize Stable Diffusion Pipeline
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            BASE_MODEL,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None,
            cache_dir="./models",
            local_files_only=False
        ).to(device)

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        
        if device == "cpu":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.enable_model_cpu_offload()
            
        return pipe

    except Exception as e:
        print(f"\nERROR: Model initialization failed - {str(e)[:500]}")
        print("\nSOLUTIONS:")
        print("1. Check your internet connection")
        print("2. Manually download models:")
        print(f"   git lfs install")
        print(f"   git clone https://huggingface.co/{BASE_MODEL} ./models/stable-diffusion-v1-5")
        print(f"   git clone https://huggingface.co/{CONTROLNET_MODEL} ./models/controlnet-scribble")
        print("3. For CPU systems, reduce image size in create_sketch()")
        exit()

def get_user_prompt():
    """Get user input for floor plan description"""
    print("\n" + "="*40)
    print("FLOOR PLAN GENERATOR".center(40))
    print("="*40)
    print("\nEnter your requirements (e.g., '3 bedroom modern apartment with open kitchen'):")
    return input("> ").strip()

def create_sketch():
    """Generate a basic architectural sketch template"""
    # Reduced size for CPU compatibility
    size = 256 if device == "cpu" else 512
    sketch = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Outer walls
    margin = int(size * 0.1)
    cv2.rectangle(sketch, (margin, margin), (size-margin, size-margin), (255,255,255), 2)
    
    # Room dividers
    cv2.line(sketch, (size//2, margin), (size//2, size-margin), (255,255,255), 1)
    cv2.line(sketch, (margin, size//2), (size-margin, size//2), (255,255,255), 1)
    
    return sketch

def generate_floor_plan(prompt, pipe):
    """Generate floor plan from text prompt"""
    try:
        print("\nGenerating floor plan... (This may take 2-5 minutes on CPU, ~1 min on GPU)")
        
        # Step 1: Create guiding sketch
        sketch = create_sketch()
        sketch_path = os.path.join(OUTPUT_DIR, "sketch.png")
        cv2.imwrite(sketch_path, sketch)
        
        # Step 2: AI generation with ControlNet
        image = pipe(
            f"{prompt}, professional architectural floor plan, clean lines, no furniture, monochrome",
            image=load_image(sketch_path),
            num_inference_steps=20 if device == "cuda" else 15,  # Fewer steps on CPU
            guidance_scale=7.5,
            negative_prompt="text, labels, furniture, messy, blurry, color"
        ).images[0]
        
        # Step 3: Post-processing
        image = image.convert("L")
        img_np = np.array(image)
        _, binary = cv2.threshold(img_np, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up small artifacts
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        output_path = os.path.join(OUTPUT_DIR, "floorplan_raw.png")
        cv2.imwrite(output_path, cleaned)
        return cleaned

    except Exception as e:
        print(f"\nERROR: Generation failed - {str(e)[:500]}")
        print("\nTry reducing the image size in create_sketch() or using simpler prompts")
        exit()

def vectorize_to_svg(image, output_svg):
    """Convert raster image to vector SVG"""
    try:
        print("\nConverting to vector format...")
        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        dwg = Drawing(output_svg, size=(image.shape[1], image.shape[0]))
        
        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Filter small artifacts
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                points = [f"{point[0][0]},{point[0][1]}" for point in approx.squeeze()]
                if len(points) >= 3:
                    dwg.add(dwg.path(d=f"M {' L '.join(points)} Z", fill="none", stroke="black", stroke_width=1))
        
        dwg.save()
    except Exception as e:
        print(f"\nERROR: Vectorization failed - {e}")
        exit()

def export_to_3d(svg_path, output_obj):
    """Convert SVG to 3D using Blender"""
    try:
        print("\nConverting to 3D model...")
        
        blender_script = f"""
import bpy
import os

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import SVG
svg_path = r"{os.path.abspath(svg_path)}"
bpy.ops.import_curve.svg(filepath=svg_path)

# Convert to mesh and extrude
for obj in bpy.context.scene.objects:
    if obj.type == 'CURVE':
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.convert(target='MESH')
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.extrude_region_move(
            TRANSFORM_OT_translate={{"value":(0, 0, 3)}}
        )
        bpy.ops.object.mode_set(mode='OBJECT')

# Export
bpy.ops.export_scene.obj(
    filepath=r"{os.path.abspath(output_obj)}",
    use_selection=True,
    use_materials=False
)
"""
        script_path = os.path.join(OUTPUT_DIR, "blender_export.py")
        with open(script_path, "w") as f:
            f.write(blender_script)
        
        subprocess.run([
            "blender",
            "--background",
            "--python",
            script_path
        ], check=True)
    except Exception as e:
        print(f"\nERROR: 3D conversion failed - {e}")
        print("\nMake sure Blender is installed and in your PATH")
        exit()

def main():
    # Initialize pipeline
    pipe = initialize_models()
    
    # Get user input
    prompt = get_user_prompt()
    
    # Generate files
    timestamp = str(int(time.time()))
    raw_img = generate_floor_plan(prompt, pipe)
    
    svg_path = os.path.join(OUTPUT_DIR, f"floorplan_{timestamp}.svg")
    vectorize_to_svg(raw_img, svg_path)
    
    obj_path = os.path.join(OUTPUT_DIR, f"floorplan_{timestamp}.obj")
    export_to_3d(svg_path, obj_path)
    
    # Print results
    print("\n" + "="*40)
    print("GENERATION COMPLETE!".center(40))
    print("="*40)
    print(f"\nResults saved to:")
    print(f"- 2D Floor Plan: {os.path.abspath(svg_path)}")
    print(f"- 3D Model: {os.path.abspath(obj_path)}")
    print("\nYou can now import the .obj file into Unity or Archicad!")

if __name__ == "__main__":
    main()