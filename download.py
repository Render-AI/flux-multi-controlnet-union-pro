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

def is_folder_empty(folder_path):
    """Check if a folder is empty."""
    return len(os.listdir(folder_path)) == 0

def install_requirements():
    """Install required system packages"""
    try:
        subprocess.check_call(["apt-get", "update"])
        subprocess.check_call(["apt-get", "install", "-y", "wget", "tar"])
    except Exception as e:
        print(f"Error installing system requirements: {e}")

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    
    downloaded_file = os.path.join(dest, url.split('/')[-1])
    
    try:
        # First try wget
        subprocess.check_call(["wget", "-P", dest, url], close_fds=False)
    except FileNotFoundError:
        try:
            # If wget not found, try curl
            subprocess.check_call(["curl", "-o", downloaded_file, url], close_fds=False)
        except FileNotFoundError:
            # If curl not found, use Python's urllib
            import urllib.request
            print("Downloading using urllib...")
            urllib.request.urlretrieve(url, downloaded_file)

    print("Download completed, extracting...")
    
    # Extract the tar file
    try:
        subprocess.check_call(["tar", "-xf", downloaded_file, "-C", dest])
        # Remove the tar file after extraction
        os.remove(downloaded_file)
    except Exception as e:
        print(f"Error extracting file: {e}")
    
    print("Total process took: ", time.time() - start)

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
    if not os.path.exists(MODEL_CACHE) or is_folder_empty(MODEL_CACHE):
        download_weights(MODEL_URL, ".")
    print(f"Main model downloaded to {MODEL_CACHE}")

def download_controlnets():
    """Download ControlNet files directly"""
    for name, config in CONTROLNETS.items():
        print(f"Downloading ControlNet {name} from {config['repo']}")
        cache_dir = os.path.join(CONTROLNET_CACHE, name)
        
        # Download if directory is empty or missing files
        expected_files = set(config['files'])
        existing_files = set(os.listdir(cache_dir)) if os.path.exists(cache_dir) else set()
        missing_files = expected_files - existing_files

        if missing_files or is_folder_empty(cache_dir):
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
        
        # Check if file exists or directory is empty
        file_path = os.path.join(cache_dir, config['file'])
        if not os.path.exists(file_path) or is_folder_empty(cache_dir):
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
        print("Installing system requirements...")
        install_requirements()

        print("\nCreating cache directories...")
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