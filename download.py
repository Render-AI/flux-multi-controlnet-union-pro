import os
import sys
import time
import subprocess
from huggingface_hub import hf_hub_download

# Constants
MODEL_CACHE = "FLUX.1-dev"
MODEL_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
CONTROLNET_CACHE = "controlnet_cache"
LORA_CACHE = "lora_cache"

# ControlNets
CONTROLNETS = {
    "upscaler": {
        "repo": "jasperai/Flux.1-dev-Controlnet-Upscaler",
        "files": ["config.json", "diffusion_pytorch_model.safetensors"]
    },
    "lineart": {
        "repo": "promeai/FLUX.1-controlnet-lineart-promeai",
        "files": ["config.json", "diffusion_pytorch_model.safetensors"]
    },
    "canny": {
        "repo": "InstantX/FLUX.1-dev-controlnet-canny",
        "files": ["config.json", "diffusion_pytorch_model.safetensors"]
    },
    "depth": {
        "repo": "Xlabs-AI/flux-controlnet-depth-diffusers",
        "files": ["config.json", "diffusion_pytorch_model.safetensors"]
    }
}

# LoRAs
LORAS = {
    "hyperflex": {
        "repo": "ByteDance/Hyper-SD",
        "file": "Hyper-FLUX.1-dev-8steps-lora.safetensors"
    },
    "add_details": {
        "repo": "Shakker-Labs/FLUX.1-dev-LoRA-add-details",
        "file": "FLUX-dev-lora-add_details.safetensors"
    },
    "realism": {
        "repo": "XLabs-AI/flux-RealismLora",
        "file": "lora.safetensors"
    }
}

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

def create_cache_dirs():
    """Create all necessary cache directories"""
    for dir_path in [MODEL_CACHE, CONTROLNET_CACHE, LORA_CACHE]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # Create subdirectories
    for controlnet in CONTROLNETS.keys():
        os.makedirs(os.path.join(CONTROLNET_CACHE, controlnet), exist_ok=True)
    for lora in LORAS.keys():
        os.makedirs(os.path.join(LORA_CACHE, lora), exist_ok=True)

def download_main_model():
    """Download main Flux model"""
    if not os.path.exists(MODEL_CACHE):
        download_weights(MODEL_URL, ".")
    print(f"Main model downloaded to {MODEL_CACHE}")

def download_controlnets():
    """Download ControlNet files directly"""
    for name, config in CONTROLNETS.items():
        print(f"Downloading ControlNet {name} from {config['repo']}")
        cache_dir = os.path.join(CONTROLNET_CACHE, name)
        
        for file in config['files']:
            try:
                hf_hub_download(
                    repo_id=config['repo'],
                    filename=file,
                    local_dir=cache_dir
                )
                print(f"Downloaded {file} for {name}")
            except Exception as e:
                print(f"Error downloading {file} for {name}: {e}")

def download_loras():
    """Download LoRA weights"""
    for name, config in LORAS.items():
        print(f"Downloading LoRA {name} from {config['repo']}")
        cache_dir = os.path.join(LORA_CACHE, name)
        
        try:
            hf_hub_download(
                repo_id=config['repo'],
                filename=config['file'],
                local_dir=cache_dir
            )
            print(f"Downloaded LoRA {name}")
        except Exception as e:
            print(f"Error downloading LoRA {name}: {e}")

def install_detector_packages():
    """Install detector packages via pip"""
    print("Installing detector packages...")
    packages = [
        "controlnet_aux",
    ]
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Installed {package}")
        except Exception as e:
            print(f"Error installing {package}: {e}")

def main():
    try:
        print("Creating cache directories...")
        create_cache_dirs()

        print("\nDownloading main model...")
        download_main_model()

        print("\nDownloading ControlNets...")
        download_controlnets()

        print("\nDownloading LoRAs...")
        download_loras()

        print("\nInstalling detector packages...")
        install_detector_packages()

        print("\nAll downloads completed!")
    except Exception as e:
        print(f"Error during download process: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()