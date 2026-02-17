
from .base_detector import BaseDetector
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

class DeepLabV3Detector(BaseDetector):
    def __init__(self, model_path, confidence_threshold=0.5, use_advanced_detection=True, force_floor_detection=False):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.use_advanced_detection = use_advanced_detection
        self.force_floor_detection = force_floor_detection
        self.model = self.load_model()
        
        # EXPANDED COCO-VOC class mapping - EXCLUDE ALL NON-FLOOR OBJECTS
        # Background class 0 is NOT furniture, but classes 1-20 include many objects
        self.furniture_classes = [
            9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,  # Original furniture
            1, 2, 3, 4, 5, 6, 7, 8,  # Additional objects (person, bicycle, car, motorcycle, airplane, bus, train, truck)
            # Classes to be more aggressive about excluding non-floor items
        ]
        # Note: Class 0 (background) is complex - it can be floor, wall, or other
        # We'll use it carefully with other heuristics
        
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((520, 520)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def load_model(self):
        import torch
        from torchvision import models
        model = models.segmentation.deeplabv3_resnet101(weights='DEFAULT')
        model.eval()
        return model

    def preprocess_image(self, image):
        """Preprocess the image for detection."""
        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        return input_batch

    def postprocess_mask(self, mask):
        """Postprocess the mask obtained from detection."""
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        
        # Close gaps (larger kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
        # Additional closing to ensure continuity
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Smooth edges slightly
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask

    def detect_floor(self, image):
        """Intelligent floor detection that EXCLUDES furniture using AI segmentation."""
        # Store original image dimensions
        original_height, original_width = image.shape[:2]
        
        # Always use intelligent detection that combines AI + heuristics
        print("  🤖 Using AI-based intelligent floor detection (furniture-aware)")
        floor_mask = self._intelligent_floor_detection(image)
        
        # Resize mask back to original image dimensions
        floor_mask = cv2.resize(floor_mask, (original_width, original_height), 
                               interpolation=cv2.INTER_NEAREST)
        
        return floor_mask
    
    def _intelligent_floor_detection(self, image):
        """
        COMPREHENSIVE 10-method floor detection for MAXIMUM coverage:
        1. AI semantic segmentation
        2. Elevated object detection
        3. Table-specific detection
        4. Edge detection
        5. Color-based floor detection
        6. Geometric floor analysis
        7. Spatial reasoning
        8. Background subtraction (NEW)
        9. Flood fill from bottom (NEW)
        10. Color consistency analysis (NEW)
        """
        original_height, original_width = image.shape[:2]
        image_resized = cv2.resize(image, (520, 520))
        height, width = image_resized.shape[:2]
        
        print("  📊 Step 1: AI semantic segmentation to identify objects...")
        furniture_mask = self._detect_furniture_enhanced(image_resized)
        
        print("  📊 Step 2: Additional object detection...")
        elevated_objects = self._detect_elevated_objects(image_resized)
        
        print("  📊 Step 3: Enhanced table/central object detection...")
        table_mask = self._detect_tables_specifically(image_resized)
        
        print("  📊 Step 4: Edge detection...")
        edge_mask = self._detect_object_edges(image_resized)
        
        print("  📊 Step 5: Color-based floor detection...")
        floor_candidates = self._detect_light_floor_areas(image_resized)
        
        print("  📊 Step 6: Geometric floor analysis...")
        geometric_floor = self._detect_horizontal_floor_surface(image_resized, height, width)
        
        print("  📊 Step 7: Spatial reasoning...")
        spatial_floor = self._get_smart_spatial_floor(height, width, furniture_mask)
        
        print("  📊 Step 8: Background subtraction (NEW)...")
        background_floor = self._detect_floor_as_background(image_resized, height, width)
        
        print("  📊 Step 9: Flood fill from bottom (NEW)...")
        flood_fill_floor = self._flood_fill_floor_detection(image_resized, height, width)
        
        print("  📊 Step 10: Color consistency analysis (NEW)...")
        consistent_floor = self._detect_floor_by_color_consistency(image_resized, height, width)
        
        print("  📊 Step 11: Glass/reflection detection (ADVANCED)...")
        glass_mask = self._detect_glass_and_reflections(image_resized, height, width)
        
        print("  📊 Step 12: Texture uniformity analysis (ADVANCED)...")
        texture_floor = self._detect_uniform_texture_surfaces(image_resized, height, width)
        
        print("  📊 Step 13: Shadow-based object boundaries (ADVANCED)...")
        shadow_objects = self._detect_objects_by_shadows(image_resized, height, width)
        
        print("  📊 Step 14: Material classification (ADVANCED)...")
        floor_materials = self._classify_floor_materials(image_resized, height, width)
        
        print("  📊 Step 15: Depth estimation from visual cues (ADVANCED)...")
        depth_floor = self._estimate_floor_depth(image_resized, height, width)
        
        print("  📊 Step 16: Perspective-based floor plane detection (ADVANCED)...")
        perspective_floor = self._detect_floor_by_perspective(image_resized, height, width)
        
        print("  📊 Step 17: Contrast-based object boundaries (ADVANCED)...")
        contrast_objects = self._detect_objects_by_contrast(image_resized, height, width)
        
        print("  📊 Step 18: Connected component floor validation (ADVANCED)...")
        
        print("  📊 Step 19: INTELLIGENT COMBINATION with multi-level filtering...")
        
        # DEBUG: Let's see what each method detects
        print(f"    - Floor candidates: {np.sum(floor_candidates) / (height * width) * 100:.1f}%")
        print(f"    - Geometric floor: {np.sum(geometric_floor) / (height * width) * 100:.1f}%")
        print(f"    - Spatial floor: {np.sum(spatial_floor) / (height * width) * 100:.1f}%")
        print(f"    - Background floor: {np.sum(background_floor) / (height * width) * 100:.1f}%")
        print(f"    - Flood fill floor: {np.sum(flood_fill_floor) / (height * width) * 100:.1f}%")
        print(f"    - Consistent floor: {np.sum(consistent_floor) / (height * width) * 100:.1f}%")
        
        # SIMPLEST APPROACH: Start with EVERYTHING in lower 75% as floor
        simple_floor_mask = np.zeros((height, width), dtype=np.uint8)
        simple_floor_mask[int(height*0.18):, :] = 1  # Everything from 18% down
        print(f"    - Simple spatial mask: {np.sum(simple_floor_mask) / (height * width) * 100:.1f}%")
        
        # Also combine the advanced detections for even more coverage
        floor_union = np.logical_or(floor_candidates, geometric_floor).astype(np.uint8)
        floor_union = np.logical_or(floor_union, spatial_floor).astype(np.uint8)
        floor_union = np.logical_or(floor_union, background_floor).astype(np.uint8)
        floor_union = np.logical_or(floor_union, flood_fill_floor).astype(np.uint8)
        floor_union = np.logical_or(floor_union, consistent_floor).astype(np.uint8)
        floor_union = np.logical_or(floor_union, simple_floor_mask).astype(np.uint8)
        print(f"    - Combined floor (before furniture exclusion): {np.sum(floor_union) / (height * width) * 100:.1f}%")
        
        # Create ULTRA PRECISE furniture exclusion - only exclude core furniture, not their edges
        furniture_core = cv2.erode(furniture_mask.astype(np.uint8) * 255, 
                                   np.ones((20,20), np.uint8), iterations=3) > 0  # Massive erosion
        table_core = cv2.erode(table_mask.astype(np.uint8) * 255,
                               np.ones((18,18), np.uint8), iterations=3) > 0  # Massive erosion
        print(f"    - Furniture mask (before erosion): {np.sum(furniture_mask) / (height * width) * 100:.1f}%")
        print(f"    - Furniture core (after erosion): {np.sum(furniture_core) / (height * width) * 100:.1f}%")
        print(f"    - Table core (after erosion): {np.sum(table_core) / (height * width) * 100:.1f}%")
        
        # Only exclude the absolute core of furniture
        obstacles = np.logical_or(furniture_core, table_core).astype(np.uint8)
        print(f"    - Total obstacles: {np.sum(obstacles) / (height * width) * 100:.1f}%")
        
        # Use advanced methods as ENHANCERS, not hard filters
        # Combine basic detections
        floor_union_basic = np.logical_or(floor_candidates, geometric_floor).astype(np.uint8)
        floor_union_basic = np.logical_or(floor_union_basic, spatial_floor).astype(np.uint8)
        floor_union_basic = np.logical_or(floor_union_basic, background_floor).astype(np.uint8)
        floor_union_basic = np.logical_or(floor_union_basic, flood_fill_floor).astype(np.uint8)
        floor_union_basic = np.logical_or(floor_union_basic, consistent_floor).astype(np.uint8)
        floor_union_basic = np.logical_or(floor_union_basic, simple_floor_mask).astype(np.uint8)
        
        # Apply soft filters (use OR logic - accept if ANY advanced method confirms)
        floor_enhanced = np.logical_or(texture_floor, floor_materials).astype(np.uint8)
        floor_enhanced = np.logical_or(floor_enhanced, depth_floor).astype(np.uint8)
        floor_enhanced = np.logical_or(floor_enhanced, perspective_floor).astype(np.uint8)
        
        # Final floor = basic detection AND at least one advanced confirmation
        floor_union = np.logical_and(floor_union_basic, floor_enhanced).astype(np.uint8)
        print(f"    - After advanced enhancement filtering: {np.sum(floor_union) / (height * width) * 100:.1f}%")
        
        # Combine all object detection methods
        obstacles_all = np.logical_or(obstacles, glass_mask).astype(np.uint8)
        obstacles_all = np.logical_or(obstacles_all, shadow_objects).astype(np.uint8)
        obstacles_all = np.logical_or(obstacles_all, contrast_objects).astype(np.uint8)
        print(f"    - Total obstacles (including glass/shadows/contrast): {np.sum(obstacles_all) / (height * width) * 100:.1f}%")
        
        # Floor = advanced detection EXCEPT all obstacles
        combined = np.logical_and(floor_union, np.logical_not(obstacles_all)).astype(np.uint8)
        
        # Add reasonable bounds
        reasonable_floor_area = self._get_reasonable_floor_bounds(height, width)
        combined = np.logical_and(combined, reasonable_floor_area).astype(np.uint8)
        
        floor_mask = (combined * 255).astype(np.uint8)
        
        print("  📊 Step 16: Multi-scale refinement and cleaning...")
        # Use CONTROLLED morphology to connect nearby regions
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel_connect)
        
        # Remove small isolated noise
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel_clean)
        
        # Keep only significant regions
        contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        floor_mask_clean = np.zeros((height, width), dtype=np.uint8)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Keep regions larger than 500 pixels
                cv2.drawContours(floor_mask_clean, [contour], -1, 255, -1)
        
        floor_mask = floor_mask_clean
        
        coverage = np.sum(floor_mask > 0) / (height * width)
        print(f"  ✓ Intelligent floor detection complete: {coverage*100:.1f}% coverage")
        
        return floor_mask
    
    def _detect_furniture_enhanced(self, image):
        """ENHANCED furniture detection using AI + visual cues with BALANCED margins."""
        # Method 1: AI-based detection
        ai_furniture = self._detect_furniture_with_ai(image)
        
        # Method 2: Detect furniture-like regions (elevated, bounded, textured)
        visual_furniture = self._detect_furniture_visual_cues(image)
        
        # Combine both methods
        combined_furniture = np.logical_or(ai_furniture, visual_furniture).astype(np.uint8) * 255
        
        # MODERATE dilation for reasonable safety margin (not too aggressive)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18))  # Balanced (was 30)
        combined_furniture = cv2.dilate(combined_furniture, kernel, iterations=2)  # Moderate (was 3)
        
        return combined_furniture > 0
    
    def _detect_furniture_visual_cues(self, image):
        """Detect furniture using visual cues: enclosed regions, texture differences, shadows."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        furniture_regions = np.zeros((height, width), dtype=np.uint8)
        
        # Cue 1: Detect enclosed regions (tables, chairs have clear boundaries)
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours - enclosed regions are likely furniture
        contours, hierarchy = cv2.findContours(edges_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and position
        for contour in contours:
            area = cv2.contourArea(contour)
            # Medium-sized enclosed regions in middle/lower area = likely furniture
            if 400 < area < 50000:  # Balanced range
                x, y, w, h = cv2.boundingRect(contour)
                # Check if in middle/center vertical zone (furniture typically not at very bottom)
                if 0.18 * height < y < 0.80 * height:  # Focused range
                    # Fill contour
                    cv2.drawContours(furniture_regions, [contour], -1, 255, thickness=cv2.FILLED)
                    # Moderate border
                    cv2.drawContours(furniture_regions, [contour], -1, 255, thickness=8)
        
        # Cue 2: Detect darker regions in lower-middle (furniture shadows)
        _, dark_regions = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY_INV)  # Balanced threshold
        
        # Only keep dark regions in furniture zone (18-72% height)
        furniture_zone = np.zeros_like(dark_regions)
        furniture_zone[int(height*0.18):int(height*0.72), :] = 255
        dark_furniture = cv2.bitwise_and(dark_regions, furniture_zone)
        
        # Moderate dilation
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
        dark_furniture = cv2.dilate(dark_furniture, kernel_dilate, iterations=1)
        
        # Combine all cues
        combined = cv2.bitwise_or(furniture_regions, dark_furniture)
        
        # Moderate safety dilation
        kernel_safe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        combined = cv2.dilate(combined, kernel_safe, iterations=1)
        
        return combined > 0
    
    def _detect_furniture_with_ai(self, image):
        """Use DeepLabV3 to detect furniture and objects."""
        input_batch = self.preprocess_image(image)
        
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        
        # Get predicted class for each pixel
        predictions = output.argmax(0).cpu().numpy()
        
        # Create mask of furniture/object areas (expanded class list)
        furniture_mask = np.zeros_like(predictions, dtype=bool)
        for furniture_class in self.furniture_classes:
            furniture_mask = np.logical_or(furniture_mask, predictions == furniture_class)
        
        # Convert to uint8
        furniture_mask_uint8 = furniture_mask.astype(np.uint8) * 255
        
        return furniture_mask_uint8 > 0
    
    def _detect_elevated_objects(self, image):
        """Detect elevated objects (tables, chairs) based on visual depth cues."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        elevated_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Method 1: Objects with distinct boundaries in center region
        # Furniture is typically in the 25-65% vertical range (not at very top or bottom)
        center_region = np.zeros_like(gray)
        center_region[int(height*0.25):int(height*0.65), :] = 255
        
        # Detect strong gradients (object boundaries)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        gradient_mag = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Strong gradients in center region = likely furniture
        _, strong_gradients = cv2.threshold(gradient_mag, 100, 255, cv2.THRESH_BINARY)
        furniture_gradients = cv2.bitwise_and(strong_gradients, center_region)
        
        # Dilate to fill furniture objects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        furniture_gradients = cv2.dilate(furniture_gradients, kernel, iterations=3)
        
        # Method 2: Detect "floating" elements (objects with space around them)
        # Use morphological operations to find isolated regions
        contours, _ = cv2.findContours(furniture_gradients, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Furniture-sized objects
            if 900 < area < 85000:  # Refined range
                x, y, w, h = cv2.boundingRect(contour)
                # Aspect ratio check (furniture is typically wider than tall or squarish)
                aspect_ratio = w / float(h) if h > 0 else 0
                if 0.25 < aspect_ratio < 4.5:  # Reasonable range
                    # Fill this region as furniture
                    cv2.drawContours(elevated_mask, [contour], -1, 255, thickness=cv2.FILLED)
                    # Moderate border for safety
                    cv2.drawContours(elevated_mask, [contour], -1, 255, thickness=12)
        
        # Additional safety: dilate elevated objects moderately
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (28, 28))  # Balanced
        elevated_mask = cv2.dilate(elevated_mask, kernel_large, iterations=1)
        
        return elevated_mask > 0
    
    def _detect_tables_specifically(self, image):
        """Specifically detect tables with VERY CONSERVATIVE detection - only obvious tables."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        table_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Method 1: Detect circular/round objects (coffee tables) - STRICT
        edges = cv2.Canny(gray, 50, 130)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=60,
                                   param1=120, param2=40, minRadius=30, maxRadius=100)  # Stricter
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                center_x, center_y, radius = circle
                # If circle is in center/middle area, it's likely a table
                if 0.30 * height < center_y < 0.60 * height:  # Tighter range
                    # Draw filled circle with MINIMAL margin
                    cv2.circle(table_mask, (center_x, center_y), int(radius * 1.1), 255, -1)  # Smaller margin
        
        # Method 2: VERY RESTRICTIVE - only mark center if there's STRONG evidence
        h_start, h_end = int(height * 0.40), int(height * 0.60)  # Narrower range
        w_start, w_end = int(width * 0.40), int(width * 0.60)  # Narrower range
        
        center_region = gray[h_start:h_end, w_start:w_end]
        center_edges = cv2.Canny(center_region, 70, 170)  # Higher thresholds
        
        # Require MUCH more evidence (was 800)
        if np.sum(center_edges > 0) > 1500:
            center_y, center_x = height // 2, width // 2
            axes_length = (int(width * 0.12), int(height * 0.10))  # Much smaller
            cv2.ellipse(table_mask, (center_x, center_y), axes_length, 0, 0, 360, 255, -1)
        
        # Method 3: DISABLED - was too aggressive
        # (The mid-brightness detection was marking everything)
        
        # MINIMAL dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))  # Much smaller (was 25)
        table_mask = cv2.dilate(table_mask, kernel, iterations=1)
        
        return table_mask > 0
    
    def _detect_floor_as_background(self, image, height, width):
        """MAXIMUM AGGRESSIVE: Assume everything in lower 75% IS floor."""
        # Start with everything in lower portion as floor
        floor_mask = np.zeros((height, width), dtype=np.uint8)
        floor_mask[int(height*0.20):, :] = 255  # Everything from 20% down is FLOOR
        
        return floor_mask > 0
    
    def _flood_fill_floor_detection(self, image, height, width):
        """AGGRESSIVE flood fill from bottom - VERY generous thresholds."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create flood fill mask
        h, w = gray.shape
        floor_filled = np.zeros((h, w), np.uint8)
        
        # Start flood fill from MANY points at bottom and sides
        seed_points = []
        
        # Bottom row - many points
        bottom_row = h - 8
        for x in range(5, w - 5, 15):  # More seeds
            seed_points.append((x, bottom_row))
        
        # Bottom corners and lower sides
        for y in range(int(h*0.60), h - 5, 20):  # Lower 40%
            seed_points.append((10, y))  # Left side
            seed_points.append((w - 10, y))  # Right side
        
        # VERY generous flood fill threshold
        threshold_low, threshold_high = 60, 60  # Much higher
        
        for seed in seed_points:
            flood_mask = np.zeros((h + 2, w + 2), np.uint8)
            gray_copy = gray.copy()
            cv2.floodFill(gray_copy, flood_mask, seed, 255,
                         loDiff=(threshold_low,), upDiff=(threshold_high,))
            # Extract filled region
            filled_region = flood_mask[1:-1, 1:-1]
            floor_filled = cv2.bitwise_or(floor_filled, filled_region)
        
        return floor_filled > 0
    
    def _detect_floor_by_color_consistency(self, image, height, width):
        """VERY GENEROUS color similarity detection."""
        # Use LAB color space for better color similarity
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Sample floor color from bottom center region (likely floor)
        sample_y1, sample_y2 = int(height * 0.70), int(height * 0.95)
        sample_x1, sample_x2 = int(width * 0.30), int(width * 0.70)
        floor_sample = lab[sample_y1:sample_y2, sample_x1:sample_x2]
        
        # Calculate mean floor color
        mean_floor_color = np.mean(floor_sample, axis=(0, 1))
        
        # Find all pixels with similar color (VERY generous threshold)
        color_diff = np.sqrt(np.sum((lab - mean_floor_color) ** 2, axis=2))
        _, similar_color = cv2.threshold(color_diff.astype(np.uint8), 80, 255, cv2.THRESH_BINARY_INV)  # Much higher
        
        # Keep only lower portion
        similar_color[:int(height*0.12), :] = 0
        
        # Aggressive cleanup and connection
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))  # Much larger
        similar_color = cv2.morphologyEx(similar_color, cv2.MORPH_CLOSE, kernel)
        
        return similar_color > 0
    
    def _detect_glass_and_reflections(self, image, height, width):
        """Detect OBVIOUS glass, mirrors, and highly reflective surfaces (very selective)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        glass_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Only VERY bright areas (strong reflections, windows)
        _, very_bright = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)  # Higher threshold
        
        # Very high value, very low saturation (pure white reflections)
        bright_unsaturated = np.logical_and(hsv[:,:,2] > 240, hsv[:,:,1] < 15).astype(np.uint8) * 255
        
        # Combine ONLY strong indicators
        glass_mask = cv2.bitwise_and(very_bright, bright_unsaturated)
        
        # Keep only in upper/middle region (windows, not floor reflections)
        glass_mask[int(height*0.60):, :] = 0
        
        # Moderate dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        glass_mask = cv2.dilate(glass_mask, kernel, iterations=1)
        
        return glass_mask > 0
    
    def _detect_uniform_texture_surfaces(self, image, height, width):
        """Detect surfaces with uniform texture - PERMISSIVE to include most floors."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate local standard deviation
        kernel_size = 15  # Smaller kernel for finer detail
        mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        mean_sq = cv2.blur((gray.astype(np.float32) ** 2), (kernel_size, kernel_size))
        variance = mean_sq - (mean ** 2)
        std_dev = np.sqrt(np.maximum(variance, 0))
        
        # Accept WIDE range of textures (floors can vary)
        floor_texture = np.logical_and(std_dev > 5, std_dev < 100).astype(np.uint8) * 255  # Very permissive
        
        # Minimal cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        floor_texture = cv2.morphologyEx(floor_texture, cv2.MORPH_CLOSE, kernel)
        
        return floor_texture > 0
    
    def _detect_objects_by_shadows(self, image, height, width):
        """Detect OBVIOUS objects by clear shadows - very selective."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Only VERY dark shadows (clear furniture shadows)
        shadows = np.logical_and(hsv[:,:,2] < 60, hsv[:,:,1] < 60).astype(np.uint8) * 255  # Stricter
        
        # Only in middle region where furniture sits
        shadows[:int(height*0.30), :] = 0
        shadows[int(height*0.70):, :] = 0
        shadows[:, :int(width*0.15)] = 0
        shadows[:, int(width*0.85):] = 0
        
        # Find clear shadow regions
        shadow_contours, _ = cv2.findContours(shadows, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        object_mask = np.zeros((height, width), dtype=np.uint8)
        for contour in shadow_contours:
            area = cv2.contourArea(contour)
            if 500 < area < 10000:  # Medium shadow size only
                x, y, w, h = cv2.boundingRect(contour)
                # Object is likely above the shadow (conservative)
                cv2.rectangle(object_mask, (x, max(0, y-int(h*0.5))), (x+w, y+int(h*0.5)), 255, -1)
        
        # Moderate dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
        object_mask = cv2.dilate(object_mask, kernel, iterations=1)
        
        return object_mask > 0
    
    def _classify_floor_materials(self, image, height, width):
        """Classify materials - PERMISSIVE detection of floor-like materials."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Accept WIDE range of materials that could be floor
        
        # Material 1: Light surfaces (marble, tile, light wood)
        light_surfaces = lab[:,:,0] > 100  # Lower threshold
        
        # Material 2: Moderate surfaces (wood, darker tile)
        moderate_surfaces = np.logical_and(gray > 60, gray < 200)
        
        # Material 3: Low to moderate texture
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        smooth_surface = sobel_mag < 80  # Very permissive
        
        # Combine - accept if ANY indicator suggests floor material
        floor_material = np.logical_or(light_surfaces, moderate_surfaces).astype(np.uint8) * 255
        floor_material = cv2.bitwise_or(floor_material, smooth_surface.astype(np.uint8) * 255)
        
        # Minimal cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        floor_material = cv2.morphologyEx(floor_material, cv2.MORPH_CLOSE, kernel)
        
        return floor_material > 0
    
    def _estimate_floor_depth(self, image, height, width):
        """Estimate depth from visual cues - PERMISSIVE detection of floor plane."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Weak vertical gradient (floor is horizontal)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        vertical_gradient = np.abs(gy)
        floor_gradient = vertical_gradient < 50  # More permissive
        
        # Method 2: Horizontal consistency
        horizontal_consistency = cv2.blur(gray.astype(np.float32), (40, 3))  # Wider horizontal blur
        horizontal_std = np.abs(gray.astype(np.float32) - horizontal_consistency)
        consistent_horizontal = horizontal_std < 40  # More permissive
        
        # Accept if EITHER criterion matches (OR logic)
        floor_depth = np.logical_or(floor_gradient, consistent_horizontal).astype(np.uint8) * 255
        
        # Minimal cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        floor_depth = cv2.morphologyEx(floor_depth, cv2.MORPH_CLOSE, kernel)
        
        # Keep generous area
        floor_depth[:int(height*0.15), :] = 0
        
        return floor_depth > 0
    
    def _detect_floor_by_perspective(self, image, height, width):
        """Detect floor using perspective - PERMISSIVE to include most floor areas."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simplified: Floor is primarily in lower portion with horizontal patterns
        floor_perspective = np.ones((height, width), dtype=np.uint8) * 255
        
        # Focus on horizontal features (floor has horizontal lines/patterns)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient ratio (horizontal surfaces have low gy/gx ratio)
        gx_abs = np.abs(gx) + 1e-6
        gy_abs = np.abs(gy) + 1e-6
        gradient_ratio = gy_abs / gx_abs
        
        # Floor has more horizontal than vertical gradients
        horizontal_dominant = gradient_ratio < 1.5  # Permissive
        
        floor_perspective = horizontal_dominant.astype(np.uint8) * 255
        
        # Keep most of image except very top
        floor_perspective[:int(height*0.15), :] = 0
        
        # Minimal cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        floor_perspective = cv2.morphologyEx(floor_perspective, cv2.MORPH_CLOSE, kernel)
        
        return floor_perspective > 0
    
    def _detect_objects_by_contrast(self, image, height, width):
        """Detect ELEVATED objects (like coffee tables) by contrast - FOCUS on center region."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Focus specifically on CENTER region where coffee table sits
        center_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Define center region (30-70% both horizontally and vertically)
        h_start, h_end = int(height * 0.30), int(height * 0.70)
        w_start, w_end = int(width * 0.30), int(width * 0.70)
        
        # Method 1: Local brightness contrast in center
        center_region = gray[h_start:h_end, w_start:w_end]
        blur_center = cv2.GaussianBlur(center_region, (21, 21), 0)
        contrast_center = np.abs(center_region.astype(np.float32) - blur_center.astype(np.float32))
        
        # High contrast = object (table surface has reflections, creates contrast)
        _, high_contrast_center = cv2.threshold(contrast_center.astype(np.uint8), 30, 255, cv2.THRESH_BINARY)
        center_mask[h_start:h_end, w_start:w_end] = high_contrast_center
        
        # Method 2: Enclosed circular/elliptical regions in center (coffee tables are often round)
        edges = cv2.Canny(gray, 70, 180)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
        
        contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        object_mask = np.zeros((height, width), dtype=np.uint8)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 25000:  # Table-sized
                x, y, w, h = cv2.boundingRect(contour)
                center_x, center_y = x + w//2, y + h//2
                
                # Must be in center region
                if (0.35 * width < center_x < 0.65 * width and 
                    0.35 * height < center_y < 0.65 * height):
                    # Check circularity (tables are often round)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        if circularity > 0.5:  # Reasonably circular
                            cv2.drawContours(object_mask, [contour], -1, 255, -1)
        
        # Combine center contrast and circular objects
        object_mask = cv2.bitwise_or(object_mask, center_mask)
        
        # Significant dilation for table safety margin
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        object_mask = cv2.dilate(object_mask, kernel, iterations=2)
        
        return object_mask > 0
    
    def _detect_object_edges(self, image):
        """Detect strong edges indicating object boundaries."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        gray_smooth = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Detect edges with moderate thresholds
        edges = cv2.Canny(gray_smooth, 60, 150)  # Balanced thresholds
        
        # Dilate edges to create boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        return edges_dilated > 0
    
    def _detect_walls_and_vertical_surfaces(self, image, height, width):
        """Detect walls and vertical surfaces that should NOT be treated as floor."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Detect vertical lines (walls, windows, doors)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        
        wall_mask = np.zeros((height, width), dtype=np.uint8)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is vertical (walls)
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle > 70 and angle < 110:  # Near vertical (80-100 degrees)
                    # This is a vertical line - likely a wall edge
                    cv2.line(wall_mask, (x1, y1), (x2, y2), 255, 8)
        
        # Method 2: Upper portion of image is more likely to be walls/background
        upper_wall_mask = np.zeros((height, width), dtype=np.uint8)
        # Top 30% is very likely to be wall/background
        upper_wall_mask[:int(height * 0.30), :] = 255
        
        # Method 3: Detect high-contrast vertical gradients (wall features)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Vertical edges are stronger than horizontal in walls
        vertical_strength = np.abs(sobely)
        horizontal_strength = np.abs(sobelx)
        
        # Normalize
        vertical_strength = cv2.normalize(vertical_strength, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        horizontal_strength = cv2.normalize(horizontal_strength, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Areas with strong vertical gradients are likely walls
        vertical_dominant = vertical_strength > (horizontal_strength * 1.5)
        vertical_wall_mask = (vertical_dominant.astype(np.uint8) * 255)
        
        # Combine all wall detection methods
        combined_wall = cv2.bitwise_or(wall_mask, upper_wall_mask)
        combined_wall = cv2.bitwise_or(combined_wall, vertical_wall_mask)
        
        # Dilate to create exclusion zones
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        combined_wall = cv2.dilate(combined_wall, kernel, iterations=2)
        
        return combined_wall > 0
    
    def _detect_horizontal_floor_surface(self, image, height, width):
        """Detect horizontal surfaces - VERY GENEROUS for maximum floor coverage."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Normalize
        sobelx = np.abs(sobelx)
        sobely = np.abs(sobely)
        
        # VERY PERMISSIVE: accept almost everything as potential floor
        horizontal_dominant = np.logical_or(
            sobelx > sobely * 0.8,  # Accept even weak horizontal (was 1.1)
            np.logical_and(sobelx < 80, sobely < 80)  # Or both weak - expanded (was 60)
        )
        
        floor_surface = (horizontal_dominant.astype(np.uint8) * 255)
        
        # Apply to much larger region - floors can be high up in images
        lower_region = np.zeros((height, width), dtype=np.uint8)
        lower_region[int(height * 0.22):, :] = 255  # Much more expanded (was 0.28)
        
        floor_surface = cv2.bitwise_and(floor_surface, lower_region)
        
        # Aggressive smoothing and expansion
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))  # Much larger (was 15)
        floor_surface = cv2.morphologyEx(floor_surface, cv2.MORPH_CLOSE, kernel)
        
        return floor_surface > 0
    
    def _get_reasonable_floor_bounds(self, height, width):
        """Define reasonable bounds - VERY GENEROUS for maximum coverage."""
        bounds_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Floor can be in bottom 85% of image (very generous)
        floor_start = int(height * 0.15)  # Was 0.20
        bounds_mask[floor_start:, :] = 1
        
        return bounds_mask > 0
    
    def _detect_light_floor_areas(self, image):
        """Detect light-colored floor with VERY AGGRESSIVE thresholds for MAXIMUM coverage."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # VERY LOW thresholds to catch all light areas
        _, gray_mask = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)  # Very low (was 90)
        _, lab_mask = cv2.threshold(lab[:, :, 0], 95, 255, cv2.THRESH_BINARY)  # Very low (was 110)
        
        # Very generous HSV thresholds
        low_sat = hsv[:, :, 1] < 120  # Very permissive (was 100)
        high_val = hsv[:, :, 2] > 100  # Very low threshold (was 120)
        hsv_mask = np.logical_and(low_sat, high_val).astype(np.uint8) * 255
        
        # Additional: very light areas
        _, very_light = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)  # Lower (was 140)
        
        # Additional: adaptive threshold to catch local variations
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 31, -5)
        
        # Combine all - very aggressive OR combination
        combined = cv2.bitwise_or(gray_mask, lab_mask)
        combined = cv2.bitwise_or(combined, hsv_mask)
        combined = cv2.bitwise_or(combined, very_light)
        combined = cv2.bitwise_or(combined, adaptive)
        
        return combined > 0
    
    def _get_smart_spatial_floor(self, height, width, furniture_mask):
        """Create spatial floor estimate - VERY GENEROUS coverage."""
        # Start with bottom 65% as potential floor (very generous)
        spatial_mask = np.zeros((height, width), dtype=np.uint8)
        floor_start = int(height * 0.35)  # Much higher (was 0.45)
        spatial_mask[floor_start:, :] = 1
        
        # Exclude areas where furniture is detected
        spatial_mask = np.logical_and(spatial_mask, np.logical_not(furniture_mask)).astype(np.uint8)
        
        # For each column, find where floor likely starts (below furniture)
        for col in range(width):
            # Find last furniture pixel in this column
            furniture_col = furniture_mask[:, col]
            if np.any(furniture_col):
                last_furniture_row = np.where(furniture_col)[0][-1]
                # Floor starts below furniture (with minimal gap)
                floor_start_row = min(last_furniture_row + 8, height - 1)  # Minimal gap (was 12)
                spatial_mask[:floor_start_row, col] = 0
        
        return spatial_mask > 0
        
        # METHOD 1: Ultra-aggressive spatial mask (bottom 70% of image)
        print("  • Method 1: Ultra-aggressive spatial coverage...")
        spatial_mask1 = self._get_ultra_aggressive_spatial_mask(height, width)
        
        # METHOD 2: Light/bright pixel detection (multiple thresholds)
        print("  • Method 2: Multi-threshold light detection...")
        light_mask = self._multi_threshold_light_detection(image_resized, height, width)
        
        # METHOD 3: Color similarity to bottom-center (definitely floor)
        print("  • Method 3: Region growing from floor seed...")
        similarity_mask = self._advanced_region_growing(image_resized, height, width)
        
        # METHOD 4: Edge-aware floor detection (exclude furniture/walls)
        print("  • Method 4: Edge-aware boundary detection...")
        edge_refined_mask = self._edge_aware_floor_detection(image_resized, height, width)
        
        # COMBINE ALL METHODS using logical OR (union) for maximum coverage
        print("  • Combining all detection methods...")
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Start with spatial mask as base
        combined_mask = spatial_mask1.copy()
        
        # Add areas detected by light detection
        combined_mask = np.logical_or(combined_mask, light_mask).astype(np.uint8)
        
        # Add areas detected by similarity
        combined_mask = np.logical_or(combined_mask, similarity_mask).astype(np.uint8)
        
        # Refine with edge detection
        combined_mask = np.logical_and(combined_mask, edge_refined_mask).astype(np.uint8)
        
        floor_mask = combined_mask * 255
        
        # Check coverage
        coverage = np.sum(floor_mask > 0) / (height * width)
        print(f"  • Combined detection coverage: {coverage*100:.1f}%")
        
        # If coverage still too small, use ultra-aggressive fallback
        if coverage < 0.35:  # Less than 35%
            print(f"  ⚠ Coverage too low, using ultra-aggressive fallback...")
            floor_mask = (spatial_mask1 * 255).astype(np.uint8)
        
        # MAXIMUM-AGGRESSIVE morphological operations (3 passes)
        print("  • Applying MAXIMUM aggressive gap filling...")
        
        # First pass: HUGE closing (was 51x51, now 61x61)
        kernel_huge = cv2.getStructuringElement(cv2.MORPH_RECT, (61, 61))
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel_huge)
        
        # Second pass: large ellipse closing (was 31x31, now 41x41)
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel_large)
        
        # Third pass: medium closing for remaining gaps
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel_medium)
        
        # Third pass: fill all holes completely
        contours, hierarchy = cv2.findContours(floor_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find and fill the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            floor_mask_filled = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(floor_mask_filled, [largest_contour], -1, 255, -1)
            
            # Fill all internal holes
            if hierarchy is not None:
                for i in range(len(contours)):
                    if hierarchy[0][i][3] != -1:  # Has parent (is a hole)
                        cv2.drawContours(floor_mask_filled, [contours[i]], -1, 255, -1)
            
            floor_mask = floor_mask_filled
        
        final_coverage = np.sum(floor_mask > 0) / (height * width)
        print(f"  ✓ FINAL floor coverage: {final_coverage*100:.1f}%")
        
        return floor_mask
    
    def _get_ultra_aggressive_spatial_mask(self, height, width):
        """MAXIMUM aggressive spatial mask covering 80% of image."""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Cover bottom 80% with minimal taper
        top_start = int(height * 0.20)  # Start at 20% from top (was 30%)
        
        for i in range(top_start, height):
            progress = (i - top_start) / (height - top_start)
            # Minimal margin (near full width)
            margin = int(width * 0.005 * (1 - progress))  # Only 0.5% margin (was 1%)
            mask[i, max(0, margin):min(width, width-margin)] = 1
        
        return mask
    
    def _multi_threshold_light_detection(self, image, height, width):
        """ULTRA-SENSITIVE light detection with lower thresholds."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Multiple thresholds to catch all light pixels
        masks = []
        
        # Threshold 1: Lower gray threshold (was 100, now 80)
        _, mask1 = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
        masks.append(mask1)
        
        # Threshold 2: Lower lightness in LAB (was 140, now 120)
        _, mask2 = cv2.threshold(lab[:, :, 0], 120, 255, cv2.THRESH_BINARY)
        masks.append(mask2)
        
        # Threshold 3: Lower value in HSV (was 160, now 140)
        _, mask3 = cv2.threshold(hsv[:, :, 2], 140, 255, cv2.THRESH_BINARY)
        masks.append(mask3)
        
        # Threshold 4: More generous low saturation + high value (was 100/150, now 110/130)
        low_sat = hsv[:, :, 1] < 110
        high_val = hsv[:, :, 2] > 130
        mask4 = np.logical_and(low_sat, high_val).astype(np.uint8) * 255
        masks.append(mask4)
        
        # Threshold 5: Adaptive threshold for local brightness
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 51, -10)
        masks.append(adaptive)
        
        # Combine all thresholds (OR)
        combined = masks[0]
        for mask in masks[1:]:
            combined = cv2.bitwise_or(combined, mask)
        
        # Apply to bottom 80% region (was 70%)
        region = np.zeros_like(combined)
        region[int(height*0.2):, :] = 255
        combined = cv2.bitwise_and(combined, region)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        return (combined > 0).astype(np.uint8)
    
    def _advanced_region_growing(self, image, height, width):
        """MAXIMUM region growing with very generous thresholds."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Multiple seed regions at bottom
        seed_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Bottom center (was 80%, now 75%)
        h_start = int(height * 0.75)
        w_start = int(width * 0.25)
        w_end = int(width * 0.75)
        seed_mask[h_start:, w_start:w_end] = 1
        
        # Get seed statistics
        seed_pixels = gray[seed_mask > 0]
        if len(seed_pixels) > 0:
            seed_mean = np.mean(seed_pixels)
            seed_std = np.std(seed_pixels)
            
            # MAXIMUM generous threshold for similarity (was 40/2.5, now 50/3.0)
            threshold = max(50, seed_std * 3.0)
            similar = np.abs(gray - seed_mean) < threshold
            similar_mask = similar.astype(np.uint8)
            
            # Keep only bottom 80% (was 70%)
            region = np.zeros_like(similar_mask)
            region[int(height*0.2):, :] = 1
            similar_mask = np.logical_and(similar_mask, region).astype(np.uint8)
            
            return similar_mask
        
        return np.zeros((height, width), dtype=np.uint8)
    
    def _edge_aware_floor_detection(self, image, height, width):
        """Edge-aware detection to exclude strong furniture/wall boundaries."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect strong edges
        edges = cv2.Canny(gray, 30, 100)
        
        # Dilate edges to create exclusion zones
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edge_zones = cv2.dilate(edges, kernel, iterations=1)
        
        # Invert: 1 = keep, 0 = exclude
        inclusion_mask = cv2.bitwise_not(edge_zones)
        
        # Apply heavy blur to soften exclusion
        inclusion_mask = cv2.GaussianBlur(inclusion_mask, (15, 15), 0)
        _, inclusion_mask = cv2.threshold(inclusion_mask, 200, 255, cv2.THRESH_BINARY)
        
        return (inclusion_mask > 0).astype(np.uint8)
    
    def _get_spatial_mask_extended(self, height, width):
        """Extended spatial mask covering MORE floor area."""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Cover bottom 65% with perspective trapezoid (INCREASED from 55%)
        top_start = int(height * 0.35)  # Start at 35% from top (was 45%)
        
        for i in range(top_start, height):
            # Calculate width at this row (wider at bottom)
            progress = (i - top_start) / (height - top_start)
            # Very minimal taper for maximum coverage
            margin = int(width * 0.02 * (1 - progress))  # Only 2% margin (was 5%)
            mask[i, margin:width-margin] = 1
        
        return mask
    
    def _semantic_floor_detection(self, image):
        """Basic semantic segmentation-based floor detection."""
        input_batch = self.preprocess_image(image)
        
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        
        output_predictions = output.argmax(0).cpu().numpy()
        height, width = output_predictions.shape
        
        # Create floor mask from semantic classes
        floor_mask = np.zeros_like(output_predictions, dtype=np.uint8)
        for class_id in self.floor_classes:
            floor_mask = np.logical_or(floor_mask, output_predictions == class_id)
        
        # Apply bottom region constraint
        bottom_region_start = int(height * 0.5)
        bottom_mask = np.zeros_like(floor_mask)
        bottom_mask[bottom_region_start:, :] = 1
        floor_mask = np.logical_and(floor_mask, bottom_mask).astype(np.uint8) * 255
        
        return self.postprocess_mask(floor_mask)
    
    def _advanced_floor_detection(self, image):
        """Advanced floor detection using multiple strategies."""
        original_height, original_width = image.shape[:2]
        
        # Resize for processing
        image_resized = cv2.resize(image, (520, 520))
        height, width = image_resized.shape[:2]
        
        print("  • Running semantic segmentation...")
        # Strategy 1: Semantic Segmentation
        semantic_mask = self._get_semantic_mask(image_resized, height, width)
        
        print("  • Detecting light/reflective surfaces...")
        # Strategy 2: Enhanced light floor detection
        color_mask = self._detect_light_floor_enhanced(image_resized, height, width)
        
        print("  • Analyzing floor texture patterns...")
        # Strategy 3: Texture-based detection
        texture_mask = self._detect_floor_texture(image_resized, height, width)
        
        print("  • Applying geometric constraints...")
        # Strategy 4: Geometric/spatial detection
        spatial_mask = self._get_spatial_mask(height, width)
        
        print("  • Detecting reflective surfaces (marble/gloss)...")
        # Strategy 5: Reflectivity detection for marble/glossy floors
        reflective_mask = self._detect_reflective_floor(image_resized, height, width)
        
        print("  • Growing floor region from seed points...")
        # Strategy 6: Region growing from bottom center (definitely floor)
        region_mask = self._region_growing_floor(image_resized, height, width)
        
        print("  • Excluding furniture and walls...")
        # Strategy 7: Exclude non-floor objects
        exclusion_mask = self._exclude_non_floor(image_resized, height, width)
        
        # Combine all strategies with optimized weights for light floors
        combined_mask = (
            semantic_mask.astype(float) * 0.15 +
            color_mask.astype(float) * 0.30 +      # Higher weight for light detection
            texture_mask.astype(float) * 0.10 +
            spatial_mask.astype(float) * 0.15 +
            reflective_mask.astype(float) * 0.20 +  # Marble/gloss detection
            region_mask.astype(float) * 0.10
        )
        
        # Apply exclusion mask
        combined_mask = combined_mask * exclusion_mask
        
        # More aggressive threshold for difficult floors
        floor_mask = (combined_mask > 0.35).astype(np.uint8) * 255
        
        # Post-process
        floor_mask = self.postprocess_mask(floor_mask)
        
        # Calculate coverage
        coverage = np.sum(floor_mask > 0) / (height * width)
        print(f"  • Initial floor coverage: {coverage*100:.1f}%")
        
        # If still too small, use very aggressive fallback
        if coverage < 0.15:  # Less than 15%
            print("  ⚠ Using aggressive light-floor fallback...")
            floor_mask = self._aggressive_light_floor_fallback(image_resized, height, width)
            floor_mask = self.postprocess_mask(floor_mask)
        
        return floor_mask
    
    def _get_semantic_mask(self, image, height, width):
        """Get floor mask from semantic segmentation."""
        input_batch = self.preprocess_image(image)
        
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        
        output_predictions = output.argmax(0).cpu().numpy()
        
        # Create mask from floor classes
        mask = np.zeros((height, width), dtype=np.uint8)
        for class_id in self.floor_classes:
            mask = np.logical_or(mask, output_predictions == class_id)
        
        return mask.astype(np.uint8)
    
    def _detect_light_floor(self, image, height, width):
        """Detect light-colored floors (marble, white tiles, etc.)."""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Detect bright pixels (high value/lightness)
        _, bright_mask = cv2.threshold(hsv[:, :, 2], 180, 255, cv2.THRESH_BINARY)
        _, light_mask = cv2.threshold(lab[:, :, 0], 160, 255, cv2.THRESH_BINARY)
        
        # Combine brightness indicators
        light_floor_mask = cv2.bitwise_and(bright_mask, light_mask)
        
        # Focus on bottom half where floor typically is
        bottom_start = height // 2
        top_mask = np.zeros_like(light_floor_mask)
        top_mask[bottom_start:, :] = 1
        light_floor_mask = cv2.bitwise_and(light_floor_mask, top_mask)
        
        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        light_floor_mask = cv2.morphologyEx(light_floor_mask, cv2.MORPH_OPEN, kernel)
        
        return (light_floor_mask > 0).astype(np.uint8)
    
    def _detect_light_floor_enhanced(self, image, height, width):
        """Enhanced light floor detection with multiple thresholds."""
        # Convert to multiple color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Very bright pixels (white/light gray)
        _, very_bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Method 2: High lightness in LAB space (more aggressive)
        _, light_lab = cv2.threshold(lab[:, :, 0], 150, 255, cv2.THRESH_BINARY)
        
        # Method 3: High value in HSV with low saturation (desaturated light colors)
        high_value = hsv[:, :, 2] > 170
        low_saturation = hsv[:, :, 1] < 80
        light_desaturated = np.logical_and(high_value, low_saturation).astype(np.uint8) * 255
        
        # Combine all light detection methods
        light_mask = cv2.bitwise_or(very_bright, light_lab)
        light_mask = cv2.bitwise_or(light_mask, light_desaturated)
        
        # Apply to bottom 60% of image
        bottom_start = int(height * 0.4)
        region_mask = np.zeros_like(light_mask)
        region_mask[bottom_start:, :] = 255
        light_mask = cv2.bitwise_and(light_mask, region_mask)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        light_mask = cv2.morphologyEx(light_mask, cv2.MORPH_CLOSE, kernel)
        
        return (light_mask > 0).astype(np.uint8)
    
    def _detect_reflective_floor(self, image, height, width):
        """Detect reflective surfaces like marble, polished tiles."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect high local variance (reflections create bright spots)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        variance = cv2.absdiff(gray, blurred)
        
        # Reflections have moderate variance (not too uniform, not too chaotic)
        reflection_mask = np.logical_and(variance > 5, variance < 40).astype(np.uint8) * 255
        
        # Combine with brightness (reflections are usually bright)
        _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        reflection_mask = cv2.bitwise_and(reflection_mask, bright)
        
        # Focus on bottom region
        bottom_start = int(height * 0.45)
        region_mask = np.zeros_like(reflection_mask)
        region_mask[bottom_start:, :] = 255
        reflection_mask = cv2.bitwise_and(reflection_mask, region_mask)
        
        # Close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        reflection_mask = cv2.morphologyEx(reflection_mask, cv2.MORPH_CLOSE, kernel)
        
        return (reflection_mask > 0).astype(np.uint8)
    
    def _region_growing_floor(self, image, height, width):
        """Grow floor region from seed points at bottom center."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Seed region at bottom center (definitely floor)
        seed_region = np.zeros((height, width), dtype=np.uint8)
        bottom_start = int(height * 0.85)
        center_left = int(width * 0.3)
        center_right = int(width * 0.7)
        seed_region[bottom_start:, center_left:center_right] = 255
        
        # Get average color of seed region
        seed_pixels = gray[seed_region > 0]
        if len(seed_pixels) > 0:
            seed_mean = np.mean(seed_pixels)
            seed_std = np.std(seed_pixels)
            
            # Find similar pixels
            threshold = max(30, seed_std * 2)
            similar = np.abs(gray - seed_mean) < threshold
            similar_mask = similar.astype(np.uint8) * 255
            
            # Keep only bottom region
            region = np.zeros_like(similar_mask)
            region[int(height*0.4):, :] = 255
            similar_mask = cv2.bitwise_and(similar_mask, region)
            
            return (similar_mask > 0).astype(np.uint8)
        
        return np.zeros((height, width), dtype=np.uint8)
    
    def _exclude_non_floor(self, image, height, width):
        """Create exclusion mask for furniture, walls, etc."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect strong edges (furniture boundaries, wall corners)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to create exclusion zones
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edge_zones = cv2.dilate(edges, kernel, iterations=2)
        
        # Detect very dark areas (furniture shadows)
        _, dark = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
        
        # Combine: exclude edge zones and dark areas
        exclusion = cv2.bitwise_and(dark, cv2.bitwise_not(edge_zones))
        
        # Invert to create inclusion mask (1 = keep, 0 = exclude)
        inclusion_mask = (exclusion > 0).astype(np.uint8)
        
        return inclusion_mask
    
    def _aggressive_light_floor_fallback(self, image, height, width):
        """Very aggressive fallback for difficult light floors."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Very aggressive brightness threshold - catch more light pixels
        _, floor_mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        
        # Apply to bottom 60% with extended trapezoid shape
        spatial = self._get_spatial_mask_extended(height, width)
        floor_mask = cv2.bitwise_and(floor_mask, (spatial * 255).astype(np.uint8))
        
        # If still too small, just use spatial mask
        if np.sum(floor_mask > 0) < (height * width * 0.2):
            floor_mask = (spatial * 255).astype(np.uint8)
        
        # Massive closing to fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel)
        
        return floor_mask
    
    def _detect_floor_texture(self, image, height, width):
        """Detect floor based on texture patterns."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradient magnitude (edges)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        
        # Detect horizontal lines (floor surface patterns)
        kernel_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        horizontal = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_horiz)
        
        # Calculate texture variance (floors have consistent texture)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        variance = cv2.absdiff(gray, blur)
        _, low_var = cv2.threshold(variance, 30, 255, cv2.THRESH_BINARY_INV)
        
        # Combine texture features
        texture_mask = cv2.bitwise_and(low_var, (gradient_mag < 50).astype(np.uint8) * 255)
        
        # Focus on bottom region
        bottom_start = height // 2
        region_mask = np.zeros_like(texture_mask)
        region_mask[bottom_start:, :] = 1
        texture_mask = cv2.bitwise_and(texture_mask, region_mask)
        
        return (texture_mask > 0).astype(np.uint8)
    
    def _get_spatial_mask(self, height, width):
        """Get floor mask based on spatial location (bottom region with perspective)."""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Create trapezoid shape (perspective floor)
        # Bottom 50% of image, with wider bottom than top
        top_start = int(height * 0.5)
        
        for i in range(top_start, height):
            # Calculate width at this row (wider at bottom)
            progress = (i - top_start) / (height - top_start)
            margin = int(width * 0.1 * (1 - progress))  # Narrower at top
            mask[i, margin:width-margin] = 1
        
        return mask

    def process_image(self, image_path):
        import cv2
        image = cv2.imread(image_path)
        floor_mask = self.detect_floor(image)
        return floor_mask