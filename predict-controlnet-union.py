from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
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
from diffusers.models import FluxMultiControlNetModel

MODEL_CACHE = "FLUX.1-dev"
MODEL_NAME = 'black-forest-labs/FLUX.1-dev'
CONTROLNET_CACHE = "FLUX.1-dev-ControlNet-Union-Pro"
CONTROLNET_MODEL_UNION = 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro'
MODEL_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
CONTROLNET_URL = "https://weights.replicate.delivery/default/shakker-labs/FLUX.1-dev-ControlNet-Union-Pro/model.tar"

CONTROL_TYPES = ["canny", "tile", "depth", "blur", "pose", "gray", "low-quality"]

SCHEDULERS = {
    "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "EulerAncestralDiscrete": EulerAncestralDiscreteScheduler,
    "EulerDiscrete": EulerDiscreteScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "LMSDiscrete": LMSDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "UniPCMultistep": UniPCMultistepScheduler,
    "KDPM2AncestralDiscrete": KDPM2AncestralDiscreteScheduler,
    "KDPM2Discrete": KDPM2DiscreteScheduler,
    "DDPM": DDPMScheduler,
    "DEISMultistep": DEISMultistepScheduler,
    "DPMSolverSinglestep": DPMSolverSinglestepScheduler,
}

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(CONTROLNET_CACHE):
            download_weights(CONTROLNET_URL, CONTROLNET_CACHE)
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, ".")
        
        controlnet_union = FluxControlNetModel.from_pretrained(
            CONTROLNET_CACHE,
            torch_dtype=torch.bfloat16
        )
        controlnet = FluxMultiControlNetModel([controlnet_union])
        self.pipe = FluxControlNetPipeline.from_pretrained(
            MODEL_CACHE,
            controlnet=controlnet,
            torch_dtype=torch.bfloat16
        ).to("cuda")

    def predict(
        self,
        prompt: str = Input(description="Input prompt", default="A bohemian-style female travel blogger with sun-kissed skin and messy beach waves"),
        guidance_scale: float = Input(description="Guidance scale", default=3.5, ge=0, le=20),
        steps: int = Input(description="Number of steps", default=20, ge=1, le=50),
        seed: int = Input(description="Set a seed for reproducibility. Random by default.", default=None),
        scheduler: str = Input(description="Scheduler to use", choices=list(SCHEDULERS.keys()), default="FlowMatchEulerDiscreteScheduler"),
        canny_image: Path = Input(description="Canny control image", default=None),
        tile_image: Path = Input(description="Tile control image", default=None),
        depth_image: Path = Input(description="Depth control image", default=None),
        blur_image: Path = Input(description="Blur control image", default=None),
        pose_image: Path = Input(description="Pose control image", default=None),
        gray_image: Path = Input(description="Gray control image", default=None),
        low_quality_image: Path = Input(description="Low-quality control image", default=None),
        canny_strength: float = Input(description="Canny ControlNet strength", default=0.4, ge=0, le=1),
        tile_strength: float = Input(description="Tile ControlNet strength", default=0.4, ge=0, le=1),
        depth_strength: float = Input(description="Depth ControlNet strength", default=0.2, ge=0, le=1),
        blur_strength: float = Input(description="Blur ControlNet strength", default=0.4, ge=0, le=1),
        pose_strength: float = Input(description="Pose ControlNet strength", default=0.4, ge=0, le=1),
        gray_strength: float = Input(description="Gray ControlNet strength", default=0.4, ge=0, le=1),
        low_quality_strength: float = Input(description="Low-quality ControlNet strength", default=0.4, ge=0, le=1)
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        # Set the scheduler
        self.pipe.scheduler = SCHEDULERS[scheduler].from_config(self.pipe.scheduler.config)

        control_images = []
        control_modes = []
        control_strengths = []

        image_inputs = [
            (canny_image, 0, canny_strength),
            (tile_image, 1, tile_strength),
            (depth_image, 2, depth_strength),
            (blur_image, 3, blur_strength),
            (pose_image, 4, pose_strength),
            (gray_image, 5, gray_strength),
            (low_quality_image, 6, low_quality_strength)
        ]

        reference_size = None

        for img_path, mode, strength in image_inputs:
            if img_path:
                img = Image.open(img_path)
                if reference_size is None:
                    # Set the reference size based on the first provided image
                    width, height = img.size
                    reference_size = (width // 8 * 8, height // 8 * 8)
                    
                # Resize the image to match the reference size
                img = img.resize(reference_size)
                control_images.append(img)
                control_modes.append(mode)
                control_strengths.append(strength)

        if not control_images:
            raise ValueError("At least one control image must be provided")

        image = self.pipe(
            prompt,
            control_image=control_images,
            control_mode=control_modes,
            width=reference_size[0],
            height=reference_size[1],
            controlnet_conditioning_scale=control_strengths,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        output_path = f"/tmp/output.png"
        image.save(output_path)
        return Path(output_path)