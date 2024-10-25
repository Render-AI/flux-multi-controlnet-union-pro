from cog import BasePredictor, Input, Path
import os
import time
import subprocess
import torch
from PIL import Image
from diffusers import (
    FluxControlNetPipeline,
    FluxControlNetModel,
    FluxMultiControlNetModel
)
from controlnet_aux import CannyDetector, MidasDetector, LineartDetector
from huggingface_hub import hf_hub_download

MODEL_CACHE = "FLUX.1-dev"
MODEL_NAME = 'black-forest-labs/FLUX.1-dev'
CONTROLNET_CACHE = "controlnet_cache"
LORA_CACHE = "lora_cache"
DETECTOR_CACHE = "detector_cache"

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
        # Initialize detectors from cache
        self.canny_detector = CannyDetector.from_pretrained(
            os.path.join(DETECTOR_CACHE, "canny")
        )
        self.depth_detector = MidasDetector.from_pretrained(
            os.path.join(DETECTOR_CACHE, "depth")
        )
        self.lineart_detector = LineartDetector.from_pretrained(
            os.path.join(DETECTOR_CACHE, "lineart")
        )
        
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
        for name in ["hyperflex", "add_details", "realism"]:
            self.pipe.load_lora_weights(
                os.path.join(LORA_CACHE, name),
                adapter_name=name
            )

    def get_controlnet_model(self, model_type: str) -> FluxControlNetModel:
        """Get or load a controlnet model from cache."""
        if model_type not in self.controlnet_models:
            cache_path = os.path.join(CONTROLNET_CACHE, model_type)
            self.controlnet_models[model_type] = FluxControlNetModel.from_pretrained(
                cache_path,
                torch_dtype=torch.float16
            ).to("cuda")
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
        prompt: str = Input(
            description="The text prompt that guides image generation. Be detailed and specific about the image you want to create. Include style, mood, colors, and specific details.",
            default="A girl in city, 25 years old, cool, futuristic style"
        ),
        canny_image: Path = Input(
            description="Input image for edge detection control. The Canny ControlNet will use the edges detected in this image to guide the generation. Best for preserving structural elements and outlines.",
            default=None
        ),
        depth_image: Path = Input(
            description="Input image for depth control. The Depth ControlNet will preserve the spatial relationships and 3D structure of this image in the generated result. Excellent for maintaining perspective and spatial layout.",
            default=None
        ),
        lineart_image: Path = Input(
            description="Input image for line art control. The Lineart ControlNet will follow the artistic lines and sketches in this image. Perfect for turning sketches into detailed artwork while maintaining the original composition.",
            default=None
        ),
        upscaler_image: Path = Input(
            description="Input image for upscaling control. The Upscaler ControlNet will enhance and improve the resolution of this image while maintaining its core details and structure. Ideal for improving image quality and adding details.",
            default=None
        ),
        canny_strength: float = Input(
            description="Controls how strongly the edge detection influences the final image. Higher values (closer to 2.0) follow edge guidance more strictly, lower values (closer to 0) allow more creative freedom.",
            default=0.6,
            ge=0,
            le=2
        ),
        depth_strength: float = Input(
            description="Determines how strictly the depth information influences the generation. Higher values preserve spatial relationships more faithfully, lower values allow more artistic interpretation.",
            default=0.6,
            ge=0,
            le=2
        ),
        lineart_strength: float = Input(
            description="Controls how closely the generated image follows the input line art. Higher values stick closer to the original lines, lower values allow more artistic freedom while maintaining basic composition.",
            default=0.6,
            ge=0,
            le=2
        ),
        upscaler_strength: float = Input(
            description="Determines how much the upscaler influences the final result. Higher values preserve more details from the original image, lower values allow more creative reinterpretation while upscaling.",
            default=0.6,
            ge=0,
            le=2
        ),
        guidance_scale: float = Input(
            description="Controls how closely the image follows the prompt. Higher values (7-20) result in images that more strictly follow the prompt but may be less natural. Lower values (1-7) allow more creative freedom but may stray from the prompt.",
            default=3.5,
            ge=0,
            le=20
        ),
        steps: int = Input(
            description="Number of denoising steps. More steps generally result in higher quality images but take longer to generate. 8-15 steps for quick results, 20-50 for higher quality. Diminishing returns after 30 steps.",
            default=8,
            ge=1,
            le=50
        ),
        seed: int = Input(
            description="Random seed for reproducible results. Using the same seed with identical parameters will generate the same image. Leave as None for random results.",
            default=None
        ),
        hyperflex_lora_weight: float = Input(
            description="Weight of the HyperFlex LoRA adaptation. Higher values enhance the model's flexibility in interpreting prompts. Recommended range 0.1-0.3 for balanced results.",
            default=0.125,
            ge=0,
            le=1
        ),
        add_details_lora_weight: float = Input(
            description="Weight of the Add Details LoRA adaptation. Higher values enhance fine details and textures in the generated image. Recommended range 0.2-0.5 for enhanced detail.",
            default=0,
            ge=0,
            le=1
        ),
        realism_lora_weight: float = Input(
            description="Weight of the Realism LoRA adaptation. Higher values enhance photorealistic qualities in the generated image. Recommended range 0.3-0.7 for balanced realism.",
            default=0,
            ge=0,
            le=1
        ),
        widthh: int = Input(
            description="Output image width in pixels. Must be divisible by 8. Higher values create wider images but require more memory. Set to 0 to use input image width. Recommended: 512-1024 for optimal quality.",
            default=0,
            ge=0,
            le=5000
        ),
        heightt: int = Input(
            description="Output image height in pixels. Must be divisible by 8. Higher values create taller images but require more memory. Set to 0 to use input image height. Recommended: 512-1024 for optimal quality.",
            default=0,
            ge=0,
            le=5000
        ),
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
            control_image=[control_images[0]] if len(control_images) == 1 else control_images,
            controlnet_conditioning_scale=[control_strengths[0]] if len(control_strengths) == 1 else control_strengths,
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