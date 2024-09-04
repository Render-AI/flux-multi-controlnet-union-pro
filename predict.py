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
from controlnet_aux import CannyDetector
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
CONTROLNET_CACHE = "FLUX.1-dev-ControlNet-Union-Pro"
CONTROLNET_MODEL_UNION = 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro'
CONTROLNET_CANNY = "InstantX/FLUX.1-dev-Controlnet-Canny-alpha"

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
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, ".")

        self.canny_controlnet = FluxControlNetModel.from_pretrained(
            CONTROLNET_CANNY,
            torch_dtype=torch.float16
        )
        self.controlnet_union = FluxControlNetModel.from_pretrained(
            CONTROLNET_MODEL_UNION,
            torch_dtype=torch.float16
        )

        controlnet = FluxMultiControlNetModel([self.canny_controlnet, self.controlnet_union])
        self.pipe = FluxControlNetPipeline.from_pretrained(
            MODEL_CACHE,
             controlnet=controlnet,
            torch_dtype=torch.float16
        )
        
        self.pipe.to("cuda")
        self.canny_detector = CannyDetector()

        # Load LoRA weights
        self.pipe.load_lora_weights(HYPERFLEX_LORA_REPO_NAME, weight_name=HYPERFLEX_LORA_CKPT_NAME, adapter_name="hyperflex")
        self.pipe.load_lora_weights(ADD_DETAILS_LORA_REPO, weight_name=ADD_DETAILS_LORA_CKPT_NAME, adapter_name="add_details")
        self.pipe.load_lora_weights(REALISM_LORA_REPO, weight_name=REALISM_LORA_CKPT_NAME, adapter_name="realism")


    def predict(
        self,
        prompt: str = Input(description="Input prompt", default="A girl in city, 25 years old, cool, futuristic"),
        canny_image: Path = Input(description="Input image for Canny ControlNet", default=None),
        tile_image: Path = Input(description="Input image for Tile ControlNet", default=None),
        depth_image: Path = Input(description="Input image for Depth ControlNet", default=None),
        blur_image: Path = Input(description="Input image for Blur ControlNet", default=None),
        pose_image: Path = Input(description="Input image for Pose ControlNet", default=None),
        gray_image: Path = Input(description="Input image for Gray ControlNet", default=None),
        low_quality_image: Path = Input(description="Input image for Low Quality ControlNet", default=None),
        guidance_scale: float = Input(description="Guidance scale", default=3.5, ge=0, le=20),
        steps: int = Input(description="Number of inference steps", default=8, ge=1, le=50),
        seed: int = Input(description="Set a seed for reproducibility. Random by default.", default=None),
        canny_strength: float = Input(description="Canny ControlNet strength", default=0.6, ge=0, le=2),
        tile_strength: float = Input(description="Tile ControlNet strength", default=0.6, ge=0, le=2),
        depth_strength: float = Input(description="Depth ControlNet strength", default=0.6, ge=0, le=2),
        blur_strength: float = Input(description="Blur ControlNet strength", default=0.6, ge=0, le=2),
        pose_strength: float = Input(description="Pose ControlNet strength", default=0.6, ge=0, le=2),
        gray_strength: float = Input(description="Gray ControlNet strength", default=0.6, ge=0, le=2),
        low_quality_strength: float = Input(description="Low Quality ControlNet strength", default=0.6, ge=0, le=2),
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
            print(loras)
            self.pipe.set_adapters(loras, adapter_weights=lora_weights)
            self.pipe.fuse_lora(adapter_names=loras)

        print("active adapters:", self.pipe.get_active_adapters())


        image_inputs = [
            (canny_image, 0, canny_strength),
            (tile_image, 1, tile_strength),
            (depth_image, 2, depth_strength),
            (blur_image, 3, blur_strength),
            (pose_image, 4, pose_strength),
            (gray_image, 5, gray_strength),
            (low_quality_image, 6, low_quality_strength)
        ]

        control_images = []
        control_modes = []
        control_strengths = []
        reference_size = None
        has_canny = False
        has_others = False

        for img_path, mode, strength in image_inputs:
            if img_path:
                img = Image.open(img_path)
                if reference_size is None:
                    # Set the reference size based on the first provided image
                    width, height = img.size
                    reference_size = (width // 8 * 8, height // 8 * 8)
                
                # Resize the image to match the reference size
                img = img.resize(reference_size)
                if mode == 0:  # Canny
                    img = self.canny_detector(img)
                    has_canny = True
                else:
                    has_others = True
                control_images.append(img)
                control_modes.append(mode)
                control_strengths.append(strength)

        if not control_images:
            raise ValueError("At least one control image must be provided")

        # Configure the ControlNet based on the provided images
        if has_canny and not has_others:
            self.pipe.controlnet = FluxMultiControlNetModel([self.canny_controlnet])
            control_modes = None
        elif has_canny and has_others:
            self.pipe.controlnet = FluxMultiControlNetModel([self.canny_controlnet, self.controlnet_union])
        elif not has_canny and has_others:
            self.pipe.controlnet = FluxMultiControlNetModel([self.controlnet_union])
        else:
            raise ValueError("Invalid combination of control images")

        generated_image = self.pipe(
            prompt,
            control_image=control_images,
            control_mode=control_modes,
            width=reference_size[0] if widthh == 0 else widthh,
            height=reference_size[1] if heightt==0 else heightt,
            controlnet_conditioning_scale=control_strengths,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        if loras:
            self.pipe.unfuse_lora()

        output_path = f"/tmp/output.png"
        generated_image.save(output_path)
        return Path(output_path)

