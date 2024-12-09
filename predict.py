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
    FluxControlNetInpaintPipeline
)
from controlnet_aux import (
    CannyDetector, 
    MidasDetector, 
    LineartDetector,
    HEDdetector,
    MLSDdetector
)
# from huggingface_hub import hf_hub_download
# from torchao.quantization import quantize_, int8_weight_only
from sd_embed.embedding_funcs import get_weighted_text_embeddings_flux1
import numpy as np

MODEL_CACHE = "FLUX.1-dev/FLUX.1-dev"
MODEL_NAME = 'black-forest-labs/FLUX.1-dev'
CONTROLNET_CACHE = "controlnet_cache"
LORA_CACHE = "lora_cache"
DETECTOR_CACHE = "detector_cache"

MODEL_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Initialize all models and detectors during setup"""
        # Initialize detectors
        self.canny_detector = CannyDetector()
        self.depth_detector = MidasDetector.from_pretrained("lllyasviel/Annotators", cache_dir=DETECTOR_CACHE)
        self.lineart_detector = LineartDetector.from_pretrained("lllyasviel/Annotators", cache_dir=DETECTOR_CACHE)
        self.hed_detector = HEDdetector.from_pretrained("lllyasviel/Annotators", cache_dir=DETECTOR_CACHE)
        self.mlsd_detector = MLSDdetector.from_pretrained("lllyasviel/Annotators", cache_dir=DETECTOR_CACHE)
        
        # Initialize all controlnet models
        print("Loading all controlnet models...")
        self.controlnet_models = {}
        model_types = ["canny", "depth", "lineart", "upscaler"]
        
        for model_type in model_types:
            print(f"Loading {model_type} controlnet...")
            self.controlnet_models[model_type] = FluxControlNetModel.from_pretrained(
                os.path.join(CONTROLNET_CACHE, model_type),
                torch_dtype=torch.float16
            ).to("cuda")
        
        # Initialize with all ControlNets
        controlnet = FluxMultiControlNetModel([
            self.controlnet_models[model_type] for model_type in model_types
        ])

        print("Loading main pipeline...")
        self.pipe = FluxControlNetPipeline.from_pretrained(
            MODEL_CACHE,
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to("cuda")

        # Quantize transformer
        # quantize_(self.pipe.transformer, int8_weight_only())

        # Create inpaint pipe and copy components
        print("Creating inpaint pipeline...")
        try:
            self.inpaint_pipe = FluxControlNetInpaintPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer, 
                text_encoder_2=self.pipe.text_encoder_2,
                tokenizer_2=self.pipe.tokenizer_2,
                transformer=self.pipe.transformer,
                controlnet=controlnet,
                scheduler=self.pipe.scheduler,
            ).to("cuda")
            print("inpaint pipeline loaded")
        except Exception as e:
            print('error:', e)
            for component in [
                self.inpaint_pipe.vae,
                self.inpaint_pipe.text_encoder,
                self.inpaint_pipe.text_encoder_2,
                self.inpaint_pipe.transformer
            ]:
                if hasattr(component, "to"):
                    component.to(dtype=torch.float16)

        try:
            # Initialize auto mask generator if needed
            if not hasattr(self, 'auto_mask_generator'):
                from auto_mask import AutoMaskGenerator
                self.auto_mask_generator = AutoMaskGenerator()
                print("auto mask loaded")
        except Exception as e:
            print("error, ", e)

        print("Loading LoRA weights...")
        for name, adapter_name in [
            ("hyperflex", "hyperflex"),
            ("add_details", "add_details"),
            ("realism", "realism")
        ]:
            lora_dir = os.path.join(LORA_CACHE, name)
            lora_files = [f for f in os.listdir(lora_dir) if f.endswith('.safetensors')]
            if lora_files:
                lora_path = os.path.join(lora_dir, lora_files[0])
                print(f"Loading {name} LoRA...")
                # Load for main pipeline
                self.pipe.load_lora_weights(
                    lora_path,
                    adapter_name=adapter_name
                )
                # Load for inpainting pipeline
                # self.inpaint_pipe.load_lora_weights(
                #     lora_path,
                #     adapter_name=adapter_name
                # )

    def get_controlnet_model(self, model_type: str) -> FluxControlNetModel:
        """Get a controlnet model from the pre-loaded models."""
        if model_type not in self.controlnet_models:
            raise ValueError(f"Unknown controlnet type: {model_type}")
        return self.controlnet_models[model_type]

    def process_image(self, image: Image.Image, detector_type: str) -> Image.Image:
        """Process the input image based on the detector type."""
        if detector_type == "canny":
            return self.canny_detector(image)
        elif detector_type == "depth":
            return self.depth_detector(image)
        elif detector_type == "lineart":
            return self.lineart_detector(image)
        elif detector_type == "hed":
            return self.hed_detector(image)
        elif detector_type == "mlsd":
            return self.mlsd_detector(image)
        elif detector_type == "upscaler":
            return image
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")

    def predict(
        self,
        prompt: str = Input(
            description="The text prompt that guides image generation",
            default="A girl in city, 25 years old, cool, futuristic style"
        ),
        canny_image: Path = Input(
            description="Input image for Canny ControlNet",
            default=None
        ),
        depth_image: Path = Input(
            description="Input image for Depth ControlNet",
            default=None
        ),
        lineart_image: Path = Input(
            description="Input image for Lineart ControlNet",
            default=None
        ),
        upscaler_image: Path = Input(
            description="Input image for Upscaler ControlNet",
            default=None
        ),
        # Detector choices for each ControlNet
        canny_detector_type: str = Input(
            description="Detector type to use with Canny ControlNet",
            default="canny",
            choices=["canny", "mlsd", "hed", "lineart"]
        ),
        depth_detector_type: str = Input(
            description="Detector type to use with Depth ControlNet",
            default="depth",
            choices=["depth", "canny", "mlsd", "hed", "lineart"]
        ),
        lineart_detector_type: str = Input(
            description="Detector type to use with Lineart ControlNet",
            default="lineart",
            choices=["lineart", "canny", "mlsd", "hed"]
        ),
        canny_strength: float = Input(
            description="Strength for Canny ControlNet",
            default=0.6,
            ge=0,
            le=2
        ),
        depth_strength: float = Input(
            description="Strength for Depth ControlNet",
            default=0.6,
            ge=0,
            le=2
        ),
        lineart_strength: float = Input(
            description="Strength for Lineart ControlNet",
            default=0.6,
            ge=0,
            le=2
        ),
        upscaler_strength: float = Input(
            description="Strength for Upscaler ControlNet",
            default=0.6,
            ge=0,
            le=2
        ),
        guidance_scale: float = Input(
            description="Controls how closely the image follows the prompt",
            default=3.5,
            ge=0,
            le=50
        ),
        steps: int = Input(
            description="Number of denoising steps",
            default=8,
            ge=1,
            le=50
        ),
        seed: int = Input(
            description="Random seed for reproducibility",
            default=None
        ),
        hyperflex_lora_weight: float = Input(
            description="Weight of the HyperFlex LoRA adaptation",
            default=0.125,
            le=1
        ),
        add_details_lora_weight: float = Input(
            description="Weight of the Add Details LoRA adaptation",
            default=0,
            le=1
        ),
        realism_lora_weight: float = Input(
            description="Weight of the Realism LoRA adaptation",
            default=0,
            le=1
        ),
        widthh: int = Input(
            description="Output image width in pixels",
            default=0,
            ge=0,
            le=5000
        ),
        heightt: int = Input(
            description="Output image height in pixels",
            default=0,
            ge=0,
            le=5000
        ),

        image: Path = Input(
            description="Input image for inpainting",
            default=None
        ),
        mask_image: Path = Input(
            description="Mask image for inpainting (white pixels will be repainted)",
            default=None
        ),
        strength: float = Input(
            description="Strength of inpainting (how much to repaint masked area)",
            default=0.6,
            ge=0,
            le=1
        ),
        padding_mask_crop: int = Input(
            description="Size of padding when cropping the mask",
            default=None
        ),
        control_guidance_start: float = Input(
            description="Percentage of steps at which ControlNet starts applying",
            default=0.0,
            ge=0.0,
            le=1.0
        ),
        control_guidance_end: float = Input(
            description="Percentage of steps at which ControlNet stops applying",
            default=1.0,
            ge=0.0,
            le=1.0
        ),
        use_weighted_embeddings: bool = Input(
            description="Whether to use weighted text embeddings",
            default=False
        ),
        auto_mask: bool = Input(
            description="Whether to use automatic masking",
            default=False
        ),
        auto_mask_prompt: str = Input(
            description="Text prompt for automatic object detection and masking",
            default=None
        ),
        auto_mask_box_threshold: float = Input(
            description="Box threshold for automatic masking",
            default=0.3,
            ge=0,
            le=1
        ),
        auto_mask_text_threshold: float = Input(
            description="Text threshold for automatic masking",
            default=0.25,
            ge=0,
            le=1
        ),
        use_background_mask: bool = Input(
            description="Whether to use background mask instead of object mask",
            default=False
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

        # Configure image processing with detector types
        image_configs = [
            (upscaler_image, "upscaler", upscaler_strength, None),  # Upscaler doesn't use a detector
            (lineart_image, "lineart", lineart_strength, lineart_detector_type),
            (canny_image, "canny", canny_strength, canny_detector_type),
            (depth_image, "depth", depth_strength, depth_detector_type),
        ]

        # First, determine reference size from the first available image
        for img_path, _, _, _ in image_configs:
            if img_path:
                img = Image.open(img_path)
                width = (img.width // 16) * 16
                height = (img.height // 16) * 16
                reference_size = (width, height)
                print(f"Reference size set to: {reference_size}")
                break

        if reference_size is None:
            raise ValueError("At least one image must be provided")

        # Process images with chosen detector types
        for img_path, controlnet_type, strength, detector_type in image_configs:
            if img_path:
                img = Image.open(img_path)
                img = img.resize(reference_size, Image.LANCZOS)
                
                if controlnet_type == "upscaler":
                    processed_image = img  # No detection for upscaler
                else:
                    processed_image = self.process_image(img, detector_type)
                
                control_images.append(processed_image)
                active_controlnets.append(self.controlnet_models[controlnet_type])
                control_strengths.append(strength)
                print(f"Processed {controlnet_type} image" + 
                      (f" using {detector_type} detector" if detector_type else ""))

        if not control_images:
            raise ValueError("At least one control image must be provided")

        # Handle custom dimensions
        final_width = reference_size[0]
        final_height = reference_size[1]
        
        if widthh != 0:
            final_width = (widthh // 16) * 16
            print(f"Using custom width: {final_width}")
        if heightt != 0:
            final_height = (heightt // 16) * 16
            print(f"Using custom height: {final_height}")

        # Update MultiControlNet with active controlnets
        self.pipe.controlnet = FluxMultiControlNetModel(active_controlnets)
        print(f"Active controlnets: {len(active_controlnets)}")

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

        
        # Prepare common generation parameters
        generation_params = {
            'control_image': [control_images[0]] if len(control_images) == 1 else control_images,
            'controlnet_conditioning_scale': [control_strengths[0]] if len(control_strengths) == 1 else control_strengths,
            'width': final_width,
            'height': final_height,
            'control_guidance_start': control_guidance_start,
            'control_guidance_end': control_guidance_end,
            'num_inference_steps': steps,
            'guidance_scale': guidance_scale,
            'generator': generator
        }

        if auto_mask and auto_mask_prompt and image:
            # Generate masks using auto masking
            object_mask, background_mask = self.auto_mask_generator.generate_masks(
                image_path=str(image),
                text_prompt=auto_mask_prompt,
                box_threshold=auto_mask_box_threshold,
                text_threshold=auto_mask_text_threshold
            )
            
            # Use the appropriate mask based on user preference
            mask_image = background_mask if use_background_mask else object_mask

            # Convert to PIL Image if needed
            if isinstance(mask_image, np.ndarray):
                mask_image = Image.fromarray(mask_image)

        # Add inpainting-specific parameters if needed
        if mask_image is not None:
            if image is None:
                raise ValueError("Inpainting requires both image and mask_image")
                
            # Open and process the input image and mask
            init_image = Image.open(image)
            mask = Image.open(mask_image)
            
            # Resize if needed
            if widthh != 0 or heightt != 0:
                init_image = init_image.resize((final_width, final_height), Image.LANCZOS)
                mask = mask.resize((final_width, final_height), Image.LANCZOS)

            generation_params.update({
                'image': init_image,
                'mask_image': mask,
                'strength': strength,
                'padding_mask_crop': padding_mask_crop
            })
            selected_pipe = self.inpaint_pipe
        else:
            selected_pipe = self.pipe

        # Add prompt or embeddings
        if use_weighted_embeddings:
            prompt_embeds, pooled_prompt_embeds = get_weighted_text_embeddings_flux1(
                pipe=self.pipe,
                prompt=prompt
            )
            generation_params.update({
                'prompt_embeds': prompt_embeds,
                'pooled_prompt_embeds': pooled_prompt_embeds
            })
        else:
            generation_params['prompt'] = prompt
        
        if loras:
            selected_pipe.set_adapters(loras, adapter_weights=lora_weights)
            selected_pipe.fuse_lora(adapter_names=loras)
            print(f"Active LoRAs: {loras} with weights: {lora_weights}")
        
        generated_image = selected_pipe(**generation_params).images[0]
        
        if loras:
            selected_pipe.unfuse_lora()

        output_path = f"/tmp/output.png"
        generated_image.save(output_path)
        print(f"Generation complete. Output saved to: {output_path}")
        return Path(output_path)
    

