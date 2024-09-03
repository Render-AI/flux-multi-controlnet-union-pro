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
            "InstantX/FLUX.1-dev-Controlnet-Canny-alpha",
            torch_dtype=torch.float16
        )
        self.pipe = FluxControlNetPipeline.from_pretrained(
            MODEL_CACHE,
            controlnet=self.canny_controlnet,
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
        guidance_scale: float = Input(description="Guidance scale", default=3.5, ge=0, le=20),
        steps: int = Input(description="Number of inference steps", default=8, ge=1, le=50),
        seed: int = Input(description="Set a seed for reproducibility. Random by default.", default=None),
        canny_strength: float = Input(description="Canny ControlNet strength", default=0.6, ge=0, le=2),
        hyperflex_lora_weight: float = Input(description="HyperFlex LoRA weight", default=0.125, ge=0, le=1),
        add_details_lora_weight: float = Input(description="Add Details LoRA weight", default=0, ge=0, le=1),
        realism_lora_weight: float = Input(description="Realism LoRA weight", default=0, ge=0, le=1),
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

        print("active adapters:", self.pipe.get_active_adapters())
        use_controlnet = True
        if use_controlnet and canny_image:
            canny_input = Image.open(canny_image)
            canny_processed = self.canny_detector(canny_input)

            generated_image = self.pipe(
                prompt,
                control_image=canny_processed,
                controlnet_conditioning_scale=canny_strength,
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

        if loras:
            self.pipe.unfuse_lora()

        output_path = f"/tmp/output.png"
        generated_image.save(output_path)
        return Path(output_path)

