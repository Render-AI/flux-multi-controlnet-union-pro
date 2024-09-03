from cog import BasePredictor, Input, Path
import os
import torch
from PIL import Image
from diffusers import (
    FluxPipeline,
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
from controlnet_aux import CannyDetector
from huggingface_hub import hf_hub_download

MODEL_NAME = 'black-forest-labs/FLUX.1-dev'
LORA_REPO_NAME = "ByteDance/Hyper-SD"
LORA_CKPT_NAME = "Hyper-FLUX.1-dev-8steps-lora.safetensors"

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

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.canny_controlnet = FluxControlNetModel.from_pretrained(
            "InstantX/FLUX.1-dev-Controlnet-Canny-alpha",
            torch_dtype=torch.float16
        )
        self.pipe = FluxPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16
        )
        
        # Load and fuse LoRA weights
        lora_path = hf_hub_download(LORA_REPO_NAME, LORA_CKPT_NAME)
        self.pipe.load_lora_weights(lora_path)
        self.pipe.fuse_lora(lora_scale=0.125)
        
        self.pipe.to("cuda")
        
        self.canny_detector = CannyDetector()

    def predict(
        self,
        prompt: str = Input(description="Input prompt", default="A girl in city, 25 years old, cool, futuristic"),
        canny_image: Path = Input(description="Input image for Canny ControlNet", default=None),
        guidance_scale: float = Input(description="Guidance scale", default=3.5, ge=0, le=5),
        steps: int = Input(description="Number of inference steps", default=8, ge=1, le=50),
        seed: int = Input(description="Set a seed for reproducibility. Random by default.", default=None),
        canny_strength: float = Input(description="Canny ControlNet strength", default=0.6, ge=0, le=1),
        use_controlnet: bool = Input(description="Whether to use ControlNet", default=False),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        if use_controlnet and canny_image:
            canny_input = Image.open(canny_image)
            canny_processed = self.canny_detector(canny_input)

            # Update the pipeline with the Canny ControlNet
            self.pipe.controlnet = self.canny_controlnet

            generated_image = self.pipe(
                prompt,
                control_image=[canny_processed],
                controlnet_conditioning_scale=[canny_strength],
                width=canny_input.size[0],
                height=canny_input.size[1],
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
        elif use_controlnet and not canny_image:
            raise ValueError("Canny image must be provided when use_controlnet is True")
        else:
            generated_image = self.pipe(
                prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]

        output_path = f"/tmp/output.png"
        generated_image.save(output_path)
        return Path(output_path)