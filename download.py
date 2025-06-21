import os
import requests
from tqdm import tqdm

MODEL_FILES = {
    "stable-diffusion-v1-5": [
        "v1-5-pruned.safetensors",
        "config.json",
        "model_index.json"
    ],
    "controlnet-scribble": [
        "diffusion_pytorch_model.safetensors",
        "config.json"
    ]
}

def download_file(url, dest):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest, 'wb') as f, tqdm(
        desc=os.path.basename(dest),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def download_models():
    os.makedirs("models/stable-diffusion-v1-5", exist_ok=True)
    os.makedirs("models/controlnet-scribble", exist_ok=True)
    
    print("Downloading Stable Diffusion v1.5...")
    for file in MODEL_FILES["stable-diffusion-v1-5"]:
        url = f"https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/{file}"
        dest = f"models/stable-diffusion-v1-5/{file}"
        download_file(url, dest)
    
    print("\nDownloading ControlNet model...")
    for file in MODEL_FILES["controlnet-scribble"]:
        url = f"https://huggingface.co/lllyasviel/sd-controlnet-scribble/resolve/main/{file}"
        dest = f"models/controlnet-scribble/{file}"
        download_file(url, dest)
    
    print("\nAll models downloaded successfully!")

if __name__ == "__main__":
    download_models()
