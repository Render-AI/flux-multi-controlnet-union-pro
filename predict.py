from cog import BasePredictor, Input, Path
import os
import time
import subprocess
import torch
from PIL import Image
from diffusers import (
    FluxControlNetPipeline,
    FluxControlNetModel,
    FluxMultiControlNetModel,
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
CONTROLNET_CANNY = "InstantX/FLUX.1-dev-controlnet-canny"
CONTROLNET_DEPTH = "Xlabs-AI/flux-controlnet-depth-diffusers"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, ".")
            
        # Initialize all detectors
        self.canny_detector = CannyDetector()
        self.depth_detector = MidasDetector.from_pretrained("lllyasviel/ControlNet")
        self.lineart_detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
        
        # Initialize all possible controlnet models but don't load them yet
        self.controlnet_models = {}
        
        # Initialize with two default ControlNets (can be updated later)
        controlnet = FluxMultiControlNetModel([
            FluxControlNetModel.from_pretrained(CONTROLNET_CANNY, torch_dtype=torch.float16),
            FluxControlNetModel.from_pretrained(CONTROLNET_CANNY, torch_dtype=torch.float16),
        ])

        # Initialize pipeline
        self.pipe = FluxControlNetPipeline.from_pretrained(
            MODEL_CACHE,
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to("cuda")

        # Load LoRA weights
        self.pipe.load_lora_weights(HYPERFLEX_LORA_REPO_NAME, weight_name=HYPERFLEX_LORA_CKPT_NAME, adapter_name="hyperflex")
        self.pipe.load_lora_weights(ADD_DETAILS_LORA_REPO, weight_name=ADD_DETAILS_LORA_CKPT_NAME, adapter_name="add_details")
        self.pipe.load_lora_weights(REALISM_LORA_REPO, weight_name=REALISM_LORA_CKPT_NAME, adapter_name="realism")

    def get_controlnet_model(self, model_type: str) -> FluxControlNetModel:
        """Get or load a controlnet model."""
        if model_type not in self.controlnet_models:
            model_id = {
                "upscaler": CONTROLNET_UPSCALER,
                "lineart": CONTROLNET_LINEART,
                "canny": CONTROLNET_CANNY,
                "depth": CONTROLNET_DEPTH
            }[model_type]
            
            self.controlnet_models[model_type] = FluxControlNetModel.from_pretrained(
                model_id,
                torch_dtype=torch.float16
            )
        return self.controlnet_models[model_type]

    def process_image(self, image: Image.Image, controlnet_type: str) -> Image.Image:
        """Process the input image based on the controlnet type."""
        if controlnet_type == "canny":
            return self.canny_detector(image)
        elif controlnet_type == "depth":
            return self.depth_detector(image)
        elif controlnet_type == "lineart":
            return self.lineart_detector(image)
        elif controlnet_type == "upscaler":
            return image
        else:
            raise ValueError(f"Unknown controlnet type: {controlnet_type}")

    def predict(
        self,
        prompt: str = Input(description="Input prompt", default="A girl in city, 25 years old, cool, futuristic"),
        canny_image: Path = Input(description="Input image for Canny ControlNet", default=None),
        depth_image: Path = Input(description="Input image for Depth ControlNet", default=None),
        lineart_image: Path = Input(description="Input image for Lineart ControlNet", default=None),
        upscaler_image: Path = Input(description="Input image for Upscaler ControlNet", default=None),
        canny_strength: float = Input(description="Canny ControlNet strength", default=0.6, ge=0, le=2),
        depth_strength: float = Input(description="Depth ControlNet strength", default=0.6, ge=0, le=2),
        lineart_strength: float = Input(description="Lineart ControlNet strength", default=0.6, ge=0, le=2),
        upscaler_strength: float = Input(description="Upscaler ControlNet strength", default=0.6, ge=0, le=2),
        guidance_scale: float = Input(description="Guidance scale", default=3.5, ge=0, le=20),
        steps: int = Input(description="Number of inference steps", default=8, ge=1, le=50),
        seed: int = Input(description="Set a seed for reproducibility. Random by default.", default=None),
        hyperflex_lora_weight: float = Input(description="HyperFlex LoRA weight", default=0.125, ge=0, le=1),
        add_details_lora_weight: float = Input(description="Add Details LoRA weight", default=0, ge=0, le=1),
        realism_lora_weight: float = Input(description="Realism LoRA weight", default=0, ge=0, le=1),
        widthh: int = Input(description="width", default=0, ge=0, le=5000),
        heightt: int = Input(description="height", default=0, ge=0, le=5000),
    ) -> Path:
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        # Process control images
        control_images = []
        active_controlnets = []
        control_strengths = []
        reference_size = None

        image_configs = [
            (upscaler_image, "upscaler", upscaler_strength),
            (lineart_image, "lineart", lineart_strength),
            (canny_image, "canny", canny_strength),
            (depth_image, "depth", depth_strength),
        ]

        for img_path, controlnet_type, strength in image_configs:
            if img_path:
                img = Image.open(img_path)
                if reference_size is None:
                    width, height = img.size
                    reference_size = (width // 8 * 8, height // 8 * 8)
                img = img.resize(reference_size)
                
                processed_image = self.process_image(img, controlnet_type)
                control_images.append(processed_image)
                active_controlnets.append(self.get_controlnet_model(controlnet_type))
                control_strengths.append(strength)

        if not control_images:
            raise ValueError("At least one control image must be provided")

        # Update MultiControlNet with active controlnets
        self.pipe.controlnet = FluxMultiControlNetModel(active_controlnets)

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

        # Generate image
        generated_image = self.pipe(
            prompt,
            control_image=control_images[0] if len(control_images) == 1 else control_images,
            controlnet_conditioning_scale=control_strengths[0] if len(control_strengths) == 1 else control_strengths,
            width=reference_size[0] if widthh == 0 else widthh,
            height=reference_size[1] if heightt == 0 else heightt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        if loras:
            self.pipe.unfuse_lora()

        output_path = f"/tmp/output.png"
        generated_image.save(output_path)
        return Path(output_path)