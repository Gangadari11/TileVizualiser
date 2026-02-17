"""
Advanced Floor Detector - Industry-Level Floor Area Detection
Combines multiple state-of-the-art AI models for precise floor detection
"""

import cv2
import numpy as np
import torch
from .base_detector import BaseDetector


class AdvancedFloorDetector(BaseDetector):
    """
    Industry-level floor detection system that combines:
    - Semantic segmentation (DeepLabV3)
    - Instance segmentation (SAM - Segment Anything Model)
    - Depth estimation (MiDaS)
    - Perspective analysis
    - Material classification
    """
    
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("🚀 Initializing Advanced Floor Detection System...")
        self._load_models()
        
    def _load_models(self):
        """Load all required AI models"""
        # 1. DeepLabV3 for semantic segmentation
        print("  📦 Loading DeepLabV3 (semantic segmentation)...")
        from torchvision import models
        self.deeplabv3 = models.segmentation.deeplabv3_resnet101(weights='DEFAULT')
        self.deeplabv3.to(self.device)
        self.deeplabv3.eval()
        
        # 2. SAM for precise segmentation (if available)
        print("  📦 Loading SAM (Segment Anything Model)...")
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            
            # Check for SAM checkpoint
            sam_checkpoint = "sam_vit_b_01ec64.pth"
            import os
            if os.path.exists(sam_checkpoint):
                sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
                sam.to(self.device)
                self.sam_mask_generator = SamAutomaticMaskGenerator(
                    model=sam,
                    points_per_side=32,
                    pred_iou_thresh=0.86,
                    stability_score_thresh=0.92,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=100,
                )
                self.has_sam = True
                print("    ✓ SAM loaded successfully")
            else:
                print(f"    ⚠ SAM checkpoint not found at {sam_checkpoint}")
                self.has_sam = False
        except ImportError:
            print("    ⚠ SAM not available (install segment-anything)")
            self.has_sam = False
        
        # 3. MiDaS for depth estimation
        print("  📦 Loading MiDaS (depth estimation)...")
        try:
            self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.midas.to(self.device)
            self.midas.eval()
            
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.midas_transform = midas_transforms.small_transform
            self.has_midas = True
            print("    ✓ MiDaS loaded successfully")
        except Exception as e:
            print(f"    ⚠ MiDaS not available: {e}")
            self.has_midas = False
        
        print("✓ Model initialization complete!\n")
    
    def detect_floor(self, image):
        """
        Advanced multi-stage floor detection
        """
        print("🔍 Starting Advanced Floor Detection Pipeline...")
        
        original_height, original_width = image.shape[:2]
        
        # Stage 1: Semantic Segmentation
        print("\n[Stage 1/5] Semantic Segmentation...")
        semantic_floor = self._semantic_floor_detection(image)
        self._print_coverage("Semantic floor", semantic_floor)
        
        # Stage 2: Depth-based Floor Detection
        print("\n[Stage 2/5] Depth-based Floor Plane Detection...")
        depth_floor = self._depth_floor_detection(image)
        self._print_coverage("Depth-based floor", depth_floor)
        
        # Stage 3: Instance Segmentation for Furniture
        print("\n[Stage 3/5] Precise Furniture Detection...")
        furniture_mask = self._detect_furniture_precise(image)
        self._print_coverage("Furniture detected", furniture_mask)
        
        # Stage 4: Geometric & Perspective Analysis
        print("\n[Stage 4/5] Geometric Floor Analysis...")
        geometric_floor = self._geometric_floor_detection(image)
        self._print_coverage("Geometric floor", geometric_floor)
        
        # Stage 5: Intelligent Fusion
        print("\n[Stage 5/5] Intelligent Multi-Model Fusion...")
        final_floor = self._fuse_detections(
            semantic_floor, 
            depth_floor, 
            furniture_mask,
            geometric_floor,
            image.shape[:2]
        )
        
        # Post-processing
        final_floor = self._post_process_mask(final_floor)
        
        # Resize to original dimensions
        final_floor = cv2.resize(final_floor, (original_width, original_height), 
                                interpolation=cv2.INTER_NEAREST)
        
        self._print_coverage("FINAL floor detection", final_floor)
        print("\n✅ Advanced floor detection complete!\n")
        
        return final_floor
    
    def _semantic_floor_detection(self, image):
        """Use DeepLabV3 for semantic segmentation"""
        from torchvision import transforms
        
        # Prepare image
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((520, 520)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.deeplabv3(input_batch)['out'][0]
        
        output_predictions = output.argmax(0).byte().cpu().numpy()
        
        # Create floor mask
        height, width = 520, 520
        floor_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Background class often contains floor
        # COCO classes: 0=background (can be floor)
        # We'll use spatial reasoning + class filtering
        background_mask = (output_predictions == 0).astype(np.uint8)
        
        # Apply STRICTER spatial constraints - floor is ONLY in lower portion
        # Exclude upper 40% which typically contains walls, furniture, beds
        spatial_mask = np.zeros((height, width), dtype=np.uint8)
        spatial_mask[int(height * 0.40):, :] = 1  # STRICTER: Only bottom 60%
        
        floor_mask = cv2.bitwise_and(background_mask, spatial_mask) * 255
        
        return floor_mask
    
    def _depth_floor_detection(self, image):
        """Use depth estimation to identify the floor plane"""
        if not self.has_midas:
            # Fallback to color-based floor detection
            return self._color_based_floor_detection(image)
        
        # Prepare image for MiDaS
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_batch = self.midas_transform(img_rgb).to(self.device)
        
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(520, 520),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Normalize depth map
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map = depth_map.astype(np.uint8)
        
        # Floor is typically at maximum depth (farthest from camera in downward direction)
        # In MiDaS output, lower values = farther, higher values = closer
        # So floor should have LOWER depth values in lower part of image
        
        height, width = depth_map.shape
        
        # Analyze depth in lower region
        lower_region = depth_map[int(height * 0.5):, :]
        floor_depth_threshold = np.percentile(lower_region, 30)  # Lower 30% depth values
        
        # Create floor mask based on depth
        floor_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Floor pixels: low depth values in lower region
        floor_candidates = depth_map < floor_depth_threshold
        
        # Apply STRICTER spatial constraints - only bottom 55%
        spatial_mask = np.zeros((height, width), dtype=np.uint8)
        spatial_mask[int(height * 0.45):, :] = 1  # STRICTER: Exclude upper 45%
        
        floor_mask = (floor_candidates & (spatial_mask == 1)).astype(np.uint8) * 255
        
        return floor_mask
    
    def _detect_furniture_precise(self, image):
        """Precise furniture detection using SAM or enhanced heuristics"""
        if self.has_sam:
            return self._detect_furniture_with_sam(image)
        else:
            return self._detect_furniture_enhanced_heuristics(image)
    
    def _detect_furniture_with_sam(self, image):
        """Use SAM to precisely segment furniture"""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (520, 520))
        
        # Generate masks
        masks = self.sam_mask_generator.generate(img_resized)
        
        height, width = 520, 520
        furniture_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Analyze each mask to determine if it's furniture
        for mask_data in masks:
            mask = mask_data['segmentation']
            bbox = mask_data['bbox']  # x, y, w, h
            area = mask_data['area']
            
            x, y, w, h = bbox
            center_y = y + h / 2
            
            # Furniture characteristics (including beds!):
            # 1. In middle vertical region (not at very bottom) - BEDS ARE HERE
            # 2. Reasonable size (not tiny, can be large for beds)
            # 3. Not spanning full width (walls/floor span full width)
            
            is_furniture = (
                0.10 * height < center_y < 0.70 * height and  # Middle region (beds included)
                300 < area < 150000 and  # Include larger objects (beds are large)
                w < width * 0.90  # Allow wider objects
            )
            
            # AGGRESSIVE: Mark large horizontal objects in upper-middle as furniture (beds)
            is_bed = (
                0.15 * height < center_y < 0.55 * height and  # Upper-middle (typical bed position)
                area > 30000 and  # Large objects
                w > width * 0.4  # Wide objects
            )
            
            if is_furniture or is_bed:
                furniture_mask[mask] = 255
        
        return furniture_mask
    
    def _detect_furniture_enhanced_heuristics(self, image):
        """Enhanced furniture detection using computer vision"""
        img_resized = cv2.resize(image, (520, 520))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        furniture_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Method 1: Edge-based object detection
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 120000:  # Wider range for beds
                x, y, w, h = cv2.boundingRect(contour)
                center_y = y + h / 2
                
                # Furniture in middle region (STRICTER)
                if 0.10 * height < center_y < 0.65 * height and w < width * 0.85:
                    cv2.drawContours(furniture_mask, [contour], -1, 255, -1)
                
                # Detect BEDS: large horizontal objects in upper-middle
                if 0.15 * height < center_y < 0.55 * height and area > 20000:
                    cv2.drawContours(furniture_mask, [contour], -1, 255, -1)
        
        # Method 2: Detect beds and darker objects
        _, dark_regions = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        furniture_zone = np.zeros_like(dark_regions)
        furniture_zone[int(height * 0.10):int(height * 0.65), :] = 255
        dark_furniture = cv2.bitwise_and(dark_regions, furniture_zone)
        
        # Method 3: Detect walls (upper portion with vertical edges)
        wall_zone = np.zeros_like(gray)
        wall_zone[:int(height * 0.40), :] = 255  # Upper 40% is likely walls
        walls = cv2.bitwise_and(edges_closed, wall_zone)
        furniture_mask = cv2.bitwise_or(furniture_mask, walls)
        
        # Combine all detection methods
        furniture_mask = cv2.bitwise_or(furniture_mask, dark_furniture)
        
        # Dilate MORE to include margins around furniture/beds
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))  # Larger kernel
        furniture_mask = cv2.dilate(furniture_mask, kernel_dilate, iterations=3)  # More iterations
        
        return furniture_mask
    
    def _geometric_floor_detection(self, image):
        """Detect floor using geometric and perspective cues"""
        img_resized = cv2.resize(image, (520, 520))
        height, width = img_resized.shape[:2]
        
        # Detect vanishing point and floor plane
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, 
                               minLineLength=50, maxLineGap=10)
        
        floor_mask = np.zeros((height, width), dtype=np.uint8)
        
        if lines is not None:
            # Analyze line orientations
            horizontal_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                # Near-horizontal lines in lower region likely indicate floor
                if angle < 20 or angle > 160:
                    if y1 > height * 0.3:
                        horizontal_lines.append((x1, y1, x2, y2))
            
            # If we have horizontal lines, assume floor is below them
            if horizontal_lines:
                # Find highest horizontal line
                min_y = min(min(y1, y2) for x1, y1, x2, y2 in horizontal_lines)
                floor_mask[int(min_y):, :] = 255
            else:
                # Default: lower 60% is floor
                floor_mask[int(height * 0.4):, :] = 255
        else:
            # Default: lower 60% is floor
            floor_mask[int(height * 0.4):, :] = 255
        
        return floor_mask
    
    def _color_based_floor_detection(self, image):
        """Detect floor based on color consistency in lower region"""
        img_resized = cv2.resize(image, (520, 520))
        height, width = img_resized.shape[:2]
        
        # Convert to LAB color space for better color analysis
        lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
        
        # Analyze color in bottom region (likely floor)
        bottom_region = lab[int(height * 0.7):, :]
        mean_color = np.mean(bottom_region, axis=(0, 1))
        std_color = np.std(bottom_region, axis=(0, 1))
        
        # Create mask for similar colors
        lower_bound = mean_color - 2.0 * std_color
        upper_bound = mean_color + 2.0 * std_color
        
        floor_mask = cv2.inRange(lab, lower_bound, upper_bound)
        
        # Apply STRICTER spatial constraint
        spatial_mask = np.zeros((height, width), dtype=np.uint8)
        spatial_mask[int(height * 0.45):, :] = 255  # STRICTER: Only bottom 55%
        
        floor_mask = cv2.bitwise_and(floor_mask, spatial_mask)
        
        return floor_mask
    
    def _fuse_detections(self, semantic, depth, furniture, geometric, shape):
        """Intelligently fuse all detection methods"""
        height, width = shape
        
        # Ensure all masks are same size
        semantic = cv2.resize(semantic, (width, height), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        furniture = cv2.resize(furniture, (width, height), interpolation=cv2.INTER_NEAREST)
        geometric = cv2.resize(geometric, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # Convert to binary
        semantic = (semantic > 127).astype(np.uint8)
        depth = (depth > 127).astype(np.uint8)
        furniture = (furniture > 127).astype(np.uint8)
        geometric = (geometric > 127).astype(np.uint8)
        
        # STRICTER voting: Additional wall exclusion
        # Walls are typically in upper 35% of image
        wall_exclusion = np.ones((height, width), dtype=np.uint8)
        wall_exclusion[:int(height * 0.35), :] = 0  # Exclude upper 35%
        
        # Voting system: pixel is floor if at least 2 out of 3 methods agree
        # (excluding furniture AND walls)
        floor_votes = semantic.astype(int) + depth.astype(int) + geometric.astype(int)
        
        # MORE CONSERVATIVE: Require at least 2 votes AND not furniture AND not wall area
        floor_mask = ((floor_votes >= 2) & (furniture == 0) & (wall_exclusion == 1)).astype(np.uint8) * 255
        
        return floor_mask
    
    def _post_process_mask(self, mask):
        """Clean up and refine the floor mask"""
        # Remove small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
        # Keep only largest connected component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if num_labels > 1:
            # Find largest component (excluding background)
            largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest_component).astype(np.uint8) * 255
        
        # Smooth edges
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def _print_coverage(self, name, mask):
        """Print coverage percentage"""
        if mask is not None and mask.size > 0:
            coverage = (np.sum(mask > 0) / mask.size) * 100
            print(f"  ✓ {name}: {coverage:.1f}% coverage")
        else:
            print(f"  ✗ {name}: No detection")
    
    def load_model(self):
        """Required by base class"""
        pass
    
    def preprocess_image(self, image):
        """Required by base class"""
        return image
    
    def postprocess_mask(self, mask):
        """Required by base class"""
        return self._post_process_mask(mask)
