import os
import torch
import time
import numpy as np
from PIL import Image
from segment_anything import build_sam, SamPredictor
from groundingdino.util.inference import load_image, predict
from groundingdino.util.utils import clean_state_dict
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig

AUTO_MASK_CACHE = "auto_mask_cache"

class AutoMaskGenerator:
    def __init__(self):
        print("\nInitializing AutoMaskGenerator...")
        start_time = time.time()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize models with timing
        print("\nInitializing SAM model...")
        sam_start = time.time()
        self.sam_predictor = self._initialize_sam()
        print(f"SAM initialization took {time.time() - sam_start:.2f} seconds")
        
        print("\nInitializing GroundingDINO model...")
        dino_start = time.time()
        self.groundingdino_model = self._initialize_groundingdino()
        print(f"GroundingDINO initialization took {time.time() - dino_start:.2f} seconds")
        
        print(f"\nTotal initialization time: {time.time() - start_time:.2f} seconds")

    def _initialize_sam(self):
        """Initialize SAM model from cache"""
        model_start = time.time()
        sam_checkpoint = os.path.join(AUTO_MASK_CACHE, "sam", "pytorch_model.bin")  # or "model.safetensors"
        
        if not os.path.exists(sam_checkpoint):
            raise RuntimeError(f"SAM model not found in cache: {sam_checkpoint}")
            
        sam = build_sam(checkpoint=sam_checkpoint).to(self.device)
        print(f"SAM model loading took {time.time() - model_start:.2f} seconds")
        return SamPredictor(sam)

    def _initialize_groundingdino(self):
        """Initialize Grounding DINO model from cache"""
        model_start = time.time()
        
        config_file = os.path.join(AUTO_MASK_CACHE, "groundingdino", "GroundingDINO_SwinB.cfg.py")
        ckpt_file = os.path.join(AUTO_MASK_CACHE, "groundingdino", "groundingdino_swinb_cogcoor.pth")
        
        if not os.path.exists(config_file) or not os.path.exists(ckpt_file):
            raise RuntimeError(f"GroundingDINO files not found in cache: {config_file} or {ckpt_file}")
        
        args = SLConfig.fromfile(config_file)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(ckpt_file, map_location=self.device)
        model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        model.eval()
        print(f"GroundingDINO model loading took {time.time() - model_start:.2f} seconds")
        return model

    # generate_masks method remains the same
    def generate_masks(self, image_path, text_prompt, box_threshold=0.3, text_threshold=0.25):
        """Generate both object mask and background mask"""
        print(f"\nGenerating masks for prompt: '{text_prompt}'")
        total_start = time.time()
        
        # Load and prepare image
        print("\nLoading image...")
        image_load_start = time.time()
        image_source, image = load_image(image_path)
        print(f"Image loading took {time.time() - image_load_start:.2f} seconds")
        
        # Detect objects using Grounding DINO
        print("\nDetecting objects with GroundingDINO...")
        dino_start = time.time()
        boxes, _, _ = predict(
            model=self.groundingdino_model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        print(f"GroundingDINO detection took {time.time() - dino_start:.2f} seconds")
        print(f"Number of boxes detected: {len(boxes)}")
        
        # Generate SAM mask
        print("\nGenerating SAM masks...")
        sam_start = time.time()
        
        set_image_start = time.time()
        self.sam_predictor.set_image(image_source)
        print(f"Setting SAM image took {time.time() - set_image_start:.2f} seconds")
        
        H, W, _ = image_source.shape
        boxes_xyxy = boxes * torch.Tensor([W, H, W, H])
        
        transform_start = time.time()
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes_xyxy.to(self.device),
            image_source.shape[:2]
        )
        print(f"Box transformation took {time.time() - transform_start:.2f} seconds")
        
        predict_start = time.time()
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        print(f"SAM prediction took {time.time() - predict_start:.2f} seconds")
        print(f"Total SAM processing took {time.time() - sam_start:.2f} seconds")
        
        # Convert to PIL Images
        print("\nConverting masks to PIL Images...")
        conversion_start = time.time()
        object_mask = Image.fromarray((masks[0][0].cpu().numpy() * 255).astype(np.uint8))
        background_mask = Image.fromarray(((1 - masks[0][0].cpu().numpy()) * 255).astype(np.uint8))
        print(f"Mask conversion took {time.time() - conversion_start:.2f} seconds")
        
        print(f"\nTotal mask generation time: {time.time() - total_start:.2f} seconds")
        
        return object_mask, background_mask