from cog import BasePredictor, Input, Path
import os
import torch
from PIL import Image
from diffusers import (
    FluxControlNetPipeline,
    FluxControlNetModel,
    FluxMultiControlNetModel
)
from controlnet_aux import CannyDetector, MidasDetector, LineartDetector

MODEL_CACHE = "FLUX.1-dev"
CONTROLNET_CACHE = "controlnet_cache"
LORA_CACHE = "lora_cache"

class Predictor(BasePredictor):
    def setup(self) -> None:
        # Initialize detectors directly (they're installed via pip)
        self.canny_detector = CannyDetector()
        self.depth_detector = MidasDetector.from_pretrained("lllyasviel/ControlNet")
        self.lineart_detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
        
        # Initialize controlnet models dict
        self.controlnet_models = {}
        
        # Initialize with two default ControlNets from cache
        controlnet = FluxMultiControlNetModel([
            FluxControlNetModel.from_pretrained(
                os.path.join(CONTROLNET_CACHE, "canny"),
                torch_dtype=torch.float16
            ).to("cuda"),
            FluxControlNetModel.from_pretrained(
                os.path.join(CONTROLNET_CACHE, "canny"),
                torch_dtype=torch.float16
            ).to("cuda"),
        ])

        # Initialize pipeline from cache
        self.pipe = FluxControlNetPipeline.from_pretrained(
            MODEL_CACHE,
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to("cuda")

        # Load LoRA weights from cache
        for name, adapter_name in [
            ("hyperflex", "hyperflex"),
            ("add_details", "add_details"),
            ("realism", "realism")
        ]:
            lora_dir = os.path.join(LORA_CACHE, name)
            # Get the first .safetensors file in the directory
            lora_files = [f for f in os.listdir(lora_dir) if f.endswith('.safetensors')]
            if lora_files:
                lora_path = os.path.join(lora_dir, lora_files[0])
                self.pipe.load_lora_weights(
                    lora_path,
                    adapter_name=adapter_name
                )

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