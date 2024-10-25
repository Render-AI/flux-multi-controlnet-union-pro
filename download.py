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
    
    # Ensure destination directory exists with proper permissions
    os.makedirs(dest, exist_ok=True)
    os.chmod(dest, 0o755)  # Ensure directory is writable
    
    filename = url.split('/')[-1]
    downloaded_file = os.path.join(dest, filename)
    
    # Check available disk space
    import shutil
    total, used, free = shutil.disk_usage(dest)
    required_space = 32 * 1024 * 1024 * 1024  # Assume we need ~32GB free space
    if free < required_space:
        raise RuntimeError(f"Not enough disk space. Need at least {required_space/(1024**3):.1f}GB, but only {free/(1024**3):.1f}GB available")
    
    # Try different download methods with resume capability
    success = False
    
    # First try wget with resume capability
    try:
        subprocess.check_call([
            "wget",
            "-c",  # Resume capability
            "--progress=bar:force:noscroll",
            "--tries=5",  # Number of retries
            "--timeout=60",  # Timeout in seconds
            "-P", dest,
            url
        ], close_fds=False)
        success = True
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"wget failed: {e}")
    
    # If wget failed, try curl
    if not success:
        try:
            subprocess.check_call([
                "curl",
                "-C", "-",  # Resume capability
                "--retry", "5",  # Number of retries
                "--retry-delay", "10",  # Delay between retries
                "--connect-timeout", "60",  # Connection timeout
                "-o", downloaded_file,
                url
            ], close_fds=False)
            success = True
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"curl failed: {e}")
    
    # If both wget and curl failed, try urllib
    if not success:
        try:
            import urllib.request
            print("Downloading using urllib...")
            
            def report_progress(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                print(f"\rDownloading: {percent}%", end="")
            
            urllib.request.urlretrieve(url, downloaded_file, reporthook=report_progress)
            print()  # New line after progress
            success = True
        except Exception as e:
            print(f"urllib failed: {e}")
    
    if not success:
        raise RuntimeError("All download methods failed")

    print("Download completed, extracting...")
    
    # Verify the downloaded file exists and has size > 0
    if not os.path.exists(downloaded_file) or os.path.getsize(downloaded_file) == 0:
        raise RuntimeError("Downloaded file is missing or empty")
    
    # Extract the tar file
    try:
        # Create a temporary extraction directory
        extract_dir = os.path.join(dest, "temp_extract")
        os.makedirs(extract_dir, exist_ok=True)
        
        # Extract with progress feedback
        subprocess.check_call([
            "tar",
            "-xvf",
            downloaded_file,
            "-C",
            extract_dir
        ])
        
        # Move extracted contents to final destination
        import glob
        extracted_files = glob.glob(os.path.join(extract_dir, "*"))
        for f in extracted_files:
            final_path = os.path.join(dest, os.path.basename(f))
            if os.path.exists(final_path):
                if os.path.isdir(final_path):
                    shutil.rmtree(final_path)
                else:
                    os.remove(final_path)
            shutil.move(f, dest)
        
        # Cleanup
        shutil.rmtree(extract_dir)
        os.remove(downloaded_file)
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        # Cleanup on failure
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        if os.path.exists(downloaded_file):
            os.remove(downloaded_file)
        raise
    
    print(f"Total process took: {time.time() - start:.1f} seconds")


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
        download_weights(MODEL_URL, MODEL_CACHE)  # Changed from "." to MODEL_CACHE
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