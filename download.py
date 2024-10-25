import os
import sys
import time
import subprocess
import torch
from diffusers import FluxControlNetModel
from controlnet_aux import CannyDetector, MidasDetector, LineartDetector
from huggingface_hub import hf_hub_download

# Constants
MODEL_CACHE = "FLUX.1-dev"
MODEL_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
CONTROLNET_CACHE = "controlnet_cache"
LORA_CACHE = "lora_cache"
DETECTOR_CACHE = "detector_cache"

# ControlNets
CONTROLNETS = {
    "upscaler": "jasperai/Flux.1-dev-Controlnet-Upscaler",
    "lineart": "promeai/FLUX.1-controlnet-lineart-promeai",
    "canny": "InstantX/FLUX.1-dev-controlnet-canny",
    "depth": "Xlabs-AI/flux-controlnet-depth-diffusers"
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

# Detectors
DETECTORS = {
    "canny": {
        "class": CannyDetector,
        "path": "lllyasviel/Annotators"
    },
    "depth": {
        "class": MidasDetector,
        "path": "lllyasviel/ControlNet"
    },
    "lineart": {
        "class": LineartDetector,
        "path": "lllyasviel/Annotators"
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
    for dir_path in [MODEL_CACHE, CONTROLNET_CACHE, LORA_CACHE, DETECTOR_CACHE]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        for controlnet in CONTROLNETS.keys():
            os.makedirs(os.path.join(CONTROLNET_CACHE, controlnet), exist_ok=True)
        for lora in LORAS.keys():
            os.makedirs(os.path.join(LORA_CACHE, lora), exist_ok=True)
        for detector in DETECTORS.keys():
            os.makedirs(os.path.join(DETECTOR_CACHE, detector), exist_ok=True)

def download_main_model():
    """Download main Flux model using pget"""
    if not os.path.exists(MODEL_CACHE):
        download_weights(MODEL_URL, ".")
    print(f"Main model downloaded to {MODEL_CACHE}")

def download_controlnets():
    """Download and save ControlNets"""
    for name, repo_id in CONTROLNETS.items():
        print(f"Downloading ControlNet {name} from {repo_id}")
        cache_dir = os.path.join(CONTROLNET_CACHE, name)
        
        controlnet = FluxControlNetModel.from_pretrained(
            repo_id,
            torch_dtype=torch.float16
        )
        controlnet.save_pretrained(cache_dir, safe_serialization=True)
        print(f"ControlNet {name} saved to {cache_dir}")

def download_loras():
    """Download LoRA weights"""
    for name, config in LORAS.items():
        print(f"Downloading LoRA {name} from {config['repo']}")
        cache_dir = os.path.join(LORA_CACHE, name)
        
        try:
            file_path = hf_hub_download(
                repo_id=config['repo'],
                filename=config['file'],
                local_dir=cache_dir
            )
            print(f"LoRA {name} saved to {file_path}")
        except Exception as e:
            print(f"Error downloading LoRA {name}: {e}")

def download_detectors():
    """Download and cache detectors"""
    for name, config in DETECTORS.items():
        print(f"Downloading detector {name} from {config['path']}")
        cache_dir = os.path.join(DETECTOR_CACHE, name)
        
        try:
            # Initialize detector to trigger download
            detector = config['class'].from_pretrained(
                config['path'],
                cache_dir=cache_dir
            )
            print(f"Detector {name} cached to {cache_dir}")
        except Exception as e:
            print(f"Error downloading detector {name}: {e}")

def main():
    print("Creating cache directories...")
    create_cache_dirs()

    print("\nDownloading main model...")
    download_main_model()

    print("\nDownloading ControlNets...")
    download_controlnets()

    print("\nDownloading LoRAs...")
    download_loras()

    print("\nDownloading detectors...")
    download_detectors()

    print("\nAll downloads completed!")

if __name__ == "__main__":
    main()