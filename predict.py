from cog import BasePredictor, Input, Path
import os
import time
import subprocess
import torch
from PIL import Image
from diffusers import (
    FluxControlNetPipeline,
    FluxControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverSinglestepScheduler,
    FlowMatchEulerDiscreteScheduler
)
from controlnet_aux import CannyDetector, MidasDetector, LineartDetector
from huggingface_hub import hf_hub_download

MODEL_CACHE = "FLUX.1-dev"
MODEL_NAME = 'black-forest-labs/FLUX.1-dev'
MODEL_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
HYPERFLEX_LORA_REPO_NAME = "ByteDance/Hyper-SD"
HYPERFLEX_LORA_CKPT_NAME = "Hyper-FLUX.1-dev-8steps-lora.safetensors"
ADD_DETAILS_LORA_REPO = "Shakker-Labs/FLUX.1-dev-LoRA-add-details"
ADD_DETAILS_LORA_CKPT_NAME = "FLUX-dev-lora-add_details.safetensors"
REALISM_LORA_REPO = "XLabs-AI/flux-RealismLora"
REALISM_LORA_CKPT_NAME = "lora.safetensors"

# ControlNet model IDs
CONTROLNET_UPSCALER = "jasperai/Flux.1-dev-Controlnet-Upscaler"
CONTROLNET_LINEART = "promeai/FLUX.1-controlnet-lineart-promeai"
CONTROLNET_CANNY = "Xlabs-AI/flux-controlnet-canny-diffusers"
CONTROLNET_DEPTH = "Xlabs-AI/flux-controlnet-depth-diffusers"

class Predictor(BasePredictor):
    def setup(self) -> None:
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, ".")
            
        # Initialize all detectors
        self.canny_detector = CannyDetector()
        self.depth_detector = MidasDetector.from_pretrained("lllyasviel/ControlNet")
        self.lineart_detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
        
        # Initialize all controlnet models
        self.controlnet_models = {
            "upscaler": FluxControlNetModel.from_pretrained(
                CONTROLNET_UPSCALER,
                torch_dtype=torch.float16
            ),
            "lineart": FluxControlNetModel.from_pretrained(
                CONTROLNET_LINEART,
                torch_dtype=torch.float16
            ),
            "canny": FluxControlNetModel.from_pretrained(
                CONTROLNET_CANNY,
                torch_dtype=torch.float16
            ),
            "depth": FluxControlNetModel.from_pretrained(
                CONTROLNET_DEPTH,
                torch_dtype=torch.float16
            )
        }

        # Initialize with default controlnet (can be changed later)
        self.pipe = FluxControlNetPipeline.from_pretrained(
            MODEL_CACHE,
            controlnet=self.controlnet_models["canny"],  # default controlnet
            torch_dtype=torch.float16
        )
        
        self.pipe.to("cuda")

        # Load LoRA weights
        self.pipe.load_lora_weights(HYPERFLEX_LORA_REPO_NAME, weight_name=HYPERFLEX_LORA_CKPT_NAME, adapter_name="hyperflex")
        self.pipe.load_lora_weights(ADD_DETAILS_LORA_REPO, weight_name=ADD_DETAILS_LORA_CKPT_NAME, adapter_name="add_details")
        self.pipe.load_lora_weights(REALISM_LORA_REPO, weight_name=REALISM_LORA_CKPT_NAME, adapter_name="realism")

    def process_image(self, image: Image.Image, controlnet_type: str) -> Image.Image:
        """Process the input image based on the controlnet type."""
        if controlnet_type == "canny":
            return self.canny_detector(image)
        elif controlnet_type == "depth":
            return self.depth_detector(image)
        elif controlnet_type == "lineart":
            return self.lineart_detector(image)
        elif controlnet_type == "upscaler":
            # Upscaler doesn't need preprocessing
            return image
        else:
            raise ValueError(f"Unknown controlnet type: {controlnet_type}")

    def predict(
        self,
        prompt: str = Input(description="Input prompt", default="A girl in city, 25 years old, cool, futuristic"),
        controlnet_type: str = Input(description="Type of ControlNet to use", default="canny", choices=["upscaler", "lineart", "canny", "depth"]),
        control_image: Path = Input(description="Input image for ControlNet", default=None),
        guidance_scale: float = Input(description="Guidance scale", default=3.5, ge=0, le=20),
        steps: int = Input(description="Number of inference steps", default=8, ge=1, le=50),
        seed: int = Input(description="Set a seed for reproducibility. Random by default.", default=None),
        control_strength: float = Input(description="ControlNet strength", default=0.6, ge=0, le=2),
        hyperflex_lora_weight: float = Input(description="HyperFlex LoRA weight", default=0.125, ge=0, le=1),
        add_details_lora_weight: float = Input(description="Add Details LoRA weight", default=0, ge=0, le=1),
        realism_lora_weight: float = Input(description="Realism LoRA weight", default=0, ge=0, le=1),
        widthh: int = Input(description="width", default=0, ge=0, le=5000),
        heightt: int = Input(description="height", default=0, ge=0, le=5000),
    ) -> Path:
        # Set the appropriate controlnet
        self.pipe.controlnet = self.controlnet_models[controlnet_type]

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        # Process control image
        if not control_image:
            raise ValueError("Control image must be provided")
            
        img = Image.open(control_image)
        width, height = img.size
        reference_size = (width // 8 * 8, height // 8 * 8)
        img = img.resize(reference_size)
        
        # Process the image according to the controlnet type
        processed_image = self.process_image(img, controlnet_type)

        # Handle LoRA weights
        lora_weights = []
        loras = []

        if hyperflex_lora_weight > 0:
            lora_weights.append(hyperflex_lora_weight)
            loras.append("hyperflex")
        
        if add_details_lora_weight > 0:
            lora_weights.append(add_details_lora_weight)
            loras.append("add_details")
        
        if realism_lora_weight > 0:
            lora_weights.append(realism_lora_weight)
            loras.append("realism")

        if loras:
            self.pipe.set_adapters(loras, adapter_weights=lora_weights)
            self.pipe.fuse_lora(adapter_names=loras)

        generated_image = self.pipe(
            prompt,
            control_image=processed_image,
            width=reference_size[0] if widthh == 0 else widthh,
            height=reference_size[1] if heightt == 0 else heightt,
            controlnet_conditioning_scale=control_strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        if loras:
            self.pipe.unfuse_lora()

        output_path = f"/tmp/output.png"
        generated_image.save(output_path)
        return Path(output_path)