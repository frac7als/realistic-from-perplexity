import os
import subprocess
import modal

# Base image setup with essentials and comfy-cli for ComfyUI
image = (
    modal.Image
    .debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "ffmpeg",
        "build-essential",
        "cmake",
        "wget",
        "curl",
    )
    .pip_install(
        "imageio[ffmpeg]",
        "moviepy",
        "fastapi[standard]==0.115.4",
        "comfy-cli==1.5.1",
        "huggingface_hub[hf_transfer]>=0.34.0,<1.0",
        "torch",  # Make sure torch is installed for models
        "transformers",
        "diffusers",
        "accelerate",
    )
    .run_commands(
        "comfy --skip-prompt install --fast-deps --nvidia --version 0.3.59",
        "mkdir -p /root/comfy/ComfyUI/custom_nodes",
        # Add any custom nodes if needed here
    )
)

# Persistent volume for huggingface cache and ComfyUI cache
vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

# Function to download models & LoRAs from Hugging Face repo or direct URLs
def download_models():
    from huggingface_hub import hf_hub_download

    # Paths inside the container for ComfyUI models
    base_dir = "/root/comfy/ComfyUI/models"
    diffusion_dir = os.path.join(base_dir, "diffusion_models")
    lora_dir = os.path.join(base_dir, "loras")

    os.makedirs(diffusion_dir, exist_ok=True)
    os.makedirs(lora_dir, exist_ok=True)

    # Flux base model download (example from CivitAI, replace with actual link if needed)
    flux_base = hf_hub_download(
        repo_id="runcomfy/flux1-kontext-dev",
        filename="flux1-kontext-dev.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(f"ln -sf {flux_base} {os.path.join(diffusion_dir, 'flux1-kontext-dev.safetensors')}", shell=True, check=True)

    # Download all recommended LoRAs (repo_id and filenames assumed or replaced with real ones)
    loras = {
        "time_tale.safetensors": ("some-user/time-tale-lora", "time_tale.safetensors"),
        "ultrarealistic_v2.safetensors": ("runcomfy/ultrarealistic-lora-v2", "ultrarealistic_v2.safetensors"),
        "awportrait_cn_1_0.safetensors": ("some-user/awportrait-cn", "awportrait_cn_1_0.safetensors"),
        "detail_enhancer_f1.safetensors": ("some-user/detail-enhancer-f1", "detail_enhancer_f1.safetensors"),
        "instagirl_wan_v2_3.safetensors": ("some-user/instagirl-wan-lora", "instagirl_wan_v2_3.safetensors"),
        "pulid_2.safetensors": ("some-user/pulid-2-lora", "pulid_2.safetensors"),
    }

    for filename, (repo, file) in loras.items():
        path = hf_hub_download(repo_id=repo, filename=file, cache_dir="/cache")
        subprocess.run(f"ln -sf {path} {os.path.join(lora_dir, filename)}", shell=True, check=True)

# Decorate Modal app and function
app = modal.App(name="comfyui-flux-kontext")

@app.function(
    gpu="A100",  # or whichever GPU you want, L40S/H100 if available
    timeout=600,
    volumes={"/cache": vol},
    container_idle_timeout=1800,
    image=image,
)
@modal.asgi()
def comfyui_api():
    import subprocess
    # Run ComfyUI, listen on all interfaces port 8000
    subprocess.Popen(
        "comfy launch -- --listen 0.0.0.0 --port 8000 --verbose",
        shell=True,
    )
    # No return, ASGI app serves web UI

@app.function(
    volumes={"/cache": vol},
    image=image,
)
def setup_and_download():
    download_models()
    return "Models and LoRAs downloaded and linked."

# Entry for local debug or modal run
if __name__ == "__main__":
    print("Starting download setup...")
    setup_and_download()
