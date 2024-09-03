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
from controlnet_aux import CannyDetector, MidasDetector
from huggingface_hub import hf_hub_download

MODEL_NAME = 'black-forest-labs/FLUX.1-dev'
CONTROLNET_REPO = "XLabs-AI/flux-controlnet-collections"
CANNY_CONTROLNET_FILE = "flux-canny-controlnet-v3.safetensors"
DEPTH_CONTROLNET_FILE = "flux-depth-controlnet-v3.safetensors"
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
        canny_path = hf_hub_download(CONTROLNET_REPO, CANNY_CONTROLNET_FILE)
        depth_path = hf_hub_download(CONTROLNET_REPO, DEPTH_CONTROLNET_FILE)
        
        self.canny_controlnet = FluxControlNetModel.from_pretrained(
            canny_path,
            torch_dtype=torch.float16
        )
        self.depth_controlnet = FluxControlNetModel.from_pretrained(
            depth_path,
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
        self.midas_detector = MidasDetector.from_pretrained("lllyasviel/Annotators")

    def predict(
        self,
        prompt: str = Input(description="Input prompt", default="A girl in city, 25 years old, cool, futuristic"),
        canny_image: Path = Input(description="Input image for Canny ControlNet", default=None),
        depth_image: Path = Input(description="Input image for Depth ControlNet", default=None),
        guidance_scale: float = Input(description="Guidance scale", default=3.5, ge=0, le=5),
        steps: int = Input(description="Number of inference steps", default=8, ge=1, le=50),
        seed: int = Input(description="Set a seed for reproducibility. Random by default.", default=None),
        canny_strength: float = Input(description="Canny ControlNet strength", default=0.6, ge=0, le=1),
        depth_strength: float = Input(description="Depth ControlNet strength", default=0.6, ge=0, le=1),
        use_controlnet: bool = Input(description="Whether to use ControlNet", default=False),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        if use_controlnet:
            control_images = []
            control_nets = []
            conditioning_scales = []

            reference_size = None

            if canny_image:
                canny_input = Image.open(canny_image)
                if reference_size is None:
                    reference_size = canny_input.size
                else:
                    canny_input = canny_input.resize(reference_size)
                canny_processed = self.canny_detector(canny_input)
                control_images.append(canny_processed)
                control_nets.append(self.canny_controlnet)
                conditioning_scales.append(canny_strength)

            if depth_image:
                depth_input = Image.open(depth_image)
                if reference_size is None:
                    reference_size = depth_input.size
                else:
                    depth_input = depth_input.resize(reference_size)
                depth_map = self.midas_detector(depth_input)
                control_images.append(depth_map)
                control_nets.append(self.depth_controlnet)
                conditioning_scales.append(depth_strength)

            if not control_images:
                raise ValueError("At least one control image must be provided when use_controlnet is True")

            # Update the pipeline with the selected ControlNets
            self.pipe.controlnet = control_nets

            generated_image = self.pipe(
                prompt,
                control_image=control_images,
                controlnet_conditioning_scale=conditioning_scales,
                width=reference_size[0] if reference_size else 512,
                height=reference_size[1] if reference_size else 512,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
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