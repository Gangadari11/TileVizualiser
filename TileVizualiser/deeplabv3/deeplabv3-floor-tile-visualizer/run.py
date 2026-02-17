"""
Complete Enhanced Floor Detection & Correction Workflow
========================================================
This is the main script that provides the complete workflow:

1. Auto-detect floor using AI models
2. Show enhanced preview with multiple visualization modes
3. Let user accept/reject detection
4. Launch enhanced correction interface with:
   - Photoshop-style smart selection tools
   - Intelligent edge snapping
   - Magic wand, quick selection, intelligent scissors
   - Automatic edge correction
5. Save final corrected mask

Usage:
    python enhanced_floor_workflow.py --image room.jpg
    
Advanced:
    python enhanced_floor_workflow.py --image room.jpg --detector industry --auto-export
"""

import cv2
import numpy as np
import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import our enhanced modules
from utils.enhanced_floor_preview import preview_floor_detection
from utils.enhanced_interactive_gui import enhanced_floor_correction

# Import tile visualization modules
from utils.interactive_floor_capture import InteractiveFloorCapture
from utils.mask_refinement import MaskRefinement
from utils.plane_approximation import PlaneApproximation
from utils.tile_projection_engine import TileProjectionEngine
from utils.realistic_blending import RealisticBlending


def detect_tile_pattern_advanced(image: np.ndarray) -> np.ndarray:
    """
    Advanced tile pattern detection using grout lines and repetitive patterns
    
    Args:
        image: Input image (BGR)
        
    Returns:
        Tile probability map (0-255)
    """
    print("   🔲 Analyzing tile patterns and grout lines...")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Method 1: Detect grout lines (dark lines between tiles)
    # Apply morphological gradient to find edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    
    # Threshold to find potential grout lines
    _, grout_candidate = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Method 2: Detect horizontal and vertical lines (tile grid)
    # Horizontal lines - OPTIMIZED: reduced iterations
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horizontal_lines = cv2.morphologyEx(grout_candidate, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    
    # Vertical lines - OPTIMIZED: reduced iterations
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    vertical_lines = cv2.morphologyEx(grout_candidate, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    
    # Combine grid lines
    tile_grid = cv2.bitwise_or(horizontal_lines, vertical_lines)
    
    # Method 3: Texture uniformity (tiles have consistent texture)
    # Compute local standard deviation
    blur = cv2.GaussianBlur(gray, (0, 0), 5)
    texture_map = cv2.Laplacian(blur, cv2.CV_64F)
    texture_map = np.abs(texture_map).astype(np.uint8)
    texture_uniformity = 255 - cv2.normalize(texture_map, None, 0, 255, cv2.NORM_MINMAX)
    
    # Method 4: Create tile probability map
    tile_prob = np.zeros((height, width), dtype=np.float32)
    
    # Areas with grid lines are likely tiles
    tile_prob[tile_grid > 0] += 0.5
    
    # Areas with uniform texture are likely tiles
    tile_prob += (texture_uniformity / 255.0) * 0.3
    
    # Bottom region is more likely to be floor
    for y in range(height):
        if y > height * 0.3:
            weight = (y - height * 0.3) / (height * 0.7)
            tile_prob[y, :] += weight * 0.2
    
    # Normalize to 0-255
    tile_prob = np.clip(tile_prob * 255, 0, 255).astype(np.uint8)
    
    # Count detected grid lines
    grid_pixels = np.sum(tile_grid > 0)
    if grid_pixels > 100:
        print(f"   ✅ Detected tile grid pattern ({grid_pixels} pixels)")
    
    return tile_prob


def detect_and_remove_objects(image: np.ndarray, floor_mask: np.ndarray) -> np.ndarray:
    """
    Detect and remove objects (furniture, decor) from floor mask using multiple methods
    
    Args:
        image: Input image
        floor_mask: Initial floor mask
        
    Returns:
        Cleaned floor mask with objects removed
    """
    print("   🚫 Removing furniture and objects...")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Method 1: Color-based furniture detection
    # Furniture typically has different colors than floor
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Get floor color from bottom region
    bottom_region = image[int(height*0.8):, :]
    floor_mean = np.mean(bottom_region, axis=(0, 1))
    
    # Calculate color difference from floor
    color_diff = np.sqrt(np.sum((image.astype(np.float32) - floor_mean) ** 2, axis=2))
    
    # Areas VERY different from floor color are likely objects (be lenient with shadows)
    color_threshold = np.percentile(color_diff, 75)  # More lenient
    color_objects = (color_diff > color_threshold * 1.8).astype(np.uint8) * 255  # Even more lenient
    
    # Focus on upper region where furniture is
    color_objects[:int(height*0.5), :] = color_objects[:int(height*0.5), :]
    color_objects[int(height*0.8):, :] = 0  # Clear bottom 20%
    
    # Method 2: Multi-scale edge detection for better object boundaries
    edges_weak = cv2.Canny(gray, 50, 150)
    edges_strong = cv2.Canny(gray, 100, 200)
    edges_combined = cv2.bitwise_or(edges_weak, edges_strong)
    
    # Method 3: Morphological closing to get complete object boundaries - OPTIMIZED
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges_closed = cv2.morphologyEx(edges_combined, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # Method 4: Find filled regions within edges (objects)
    # Invert and fill
    edges_inv = cv2.bitwise_not(edges_closed)
    
    # Flood fill from top corners (usually background/walls)
    filled = edges_inv.copy()
    h, w = filled.shape[:2]
    mask_flood = np.zeros((h+2, w+2), np.uint8)
    
    # Fill from multiple seed points (corners and top edges)
    seed_points = [(0, 0), (w-1, 0), (w//2, 0), (0, h-1), (w-1, h-1)]
    for point in seed_points:
        if 0 <= point[0] < w and 0 <= point[1] < h:
            cv2.floodFill(filled, mask_flood, point, 0)
    
    # Remaining white regions are enclosed by edges (potential objects)
    potential_objects = filled
    
    # Method 5: Find contours of potential objects
    contours, _ = cv2.findContours(potential_objects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create object mask
    object_mask = np.zeros((height, width), dtype=np.uint8)
    
    removed_objects = 0
    total_image_area = height * width
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by size: objects are between 0.05% and 25% of image
        if area > total_image_area * 0.0005 and area < total_image_area * 0.25:
            # Get centroid position
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate shape features
                aspect_ratio = float(w) / h if h > 0 else 0
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Objects criteria:
                # 1. In upper 70% of image (furniture is typically in middle/upper area)
                # 2. Not extremely elongated (0.15 < aspect ratio < 6.0)
                # 3. Reasonably compact or rectangular (circularity > 0.15)
                # 4. Overlaps with floor mask (to avoid false positives)
                
                is_in_upper_region = cy < height * 0.7  # Expanded to 70%
                reasonable_shape = 0.15 < aspect_ratio < 6.0  # More lenient
                compact_enough = circularity > 0.15  # More lenient
                
                # Check overlap with floor mask
                contour_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], -1, 255, -1)
                overlap = cv2.bitwise_and(floor_mask, contour_mask)
                overlap_ratio = np.sum(overlap > 0) / area if area > 0 else 0
                
                is_on_floor = overlap_ratio > 0.3  # Lower threshold (30% overlap)
                
                # Additional check: objects in top half are more likely furniture
                is_definitely_furniture = cy < height * 0.5 and area > total_image_area * 0.02
                
                # Mark as object if criteria met
                if (is_in_upper_region and reasonable_shape and compact_enough and is_on_floor) or is_definitely_furniture:
                    cv2.drawContours(object_mask, [contour], -1, 255, -1)
                    removed_objects += 1
    
    # Method 6: Additional removal - detect vertical objects (furniture legs, chairs)
    # Look for vertical structures in floor region
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    vertical_structures = cv2.morphologyEx(edges_combined, cv2.MORPH_OPEN, kernel_vertical)
    
    # Dilate slightly
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    vertical_structures = cv2.dilate(vertical_structures, kernel_dilate, iterations=2)
    
    # Only keep vertical structures in upper region
    vertical_structures[int(height*0.7):, :] = 0
    
    # Method 7: Combine all object detection methods
    # Be very conservative with color-based detection (can remove shadowed floor)
    color_objects_dilated = cv2.dilate(color_objects, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    
    object_mask = cv2.bitwise_or(object_mask, vertical_structures)
    object_mask = cv2.bitwise_or(object_mask, color_objects_dilated)
    
    # Dilate object mask minimally to preserve shadowed floor areas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    object_mask = cv2.dilate(object_mask, kernel, iterations=1)
    
    # Remove objects from floor mask
    cleaned_mask = cv2.bitwise_and(floor_mask, cv2.bitwise_not(object_mask))
    
    # Calculate removed percentage
    removed_pixels = np.sum(floor_mask > 0) - np.sum(cleaned_mask > 0)
    removed_percentage = (removed_pixels / np.sum(floor_mask > 0)) * 100 if np.sum(floor_mask > 0) > 0 else 0
    
    print(f"   ✅ Removed {removed_objects} objects ({removed_percentage:.1f}% of detected area)")
    
    return cleaned_mask


def simple_floor_detection(image: np.ndarray, auto_mode: bool = False) -> np.ndarray:
    """
    Advanced floor detection with shadow handling and tile pattern recognition
    Handles lighting variations, existing tile patterns, and shadows
    
    Args:
        image: Input image (BGR)
        auto_mode: If True, uses more aggressive detection for automatic workflow
    
    Returns:
        Floor mask (0-255)
    """
    print("\n🔍 Running advanced floor detection with shadow handling...")
    
    height, width = image.shape[:2]
    
    # Method 1: Detect existing tile patterns (grout lines, grid)
    tile_probability = detect_tile_pattern_advanced(image)
    print(f"   Tile pattern map computed")
    
    # Method 2: Multi-region color analysis (handle shadows)
    # Analyze ENTIRE bottom half, not just extreme bottom
    bottom_region = image[int(height * 0.5):, :]  # Bottom 50%
    
    # Get multiple dominant colors (tiles + shadows)
    bottom_flat = bottom_region.reshape(-1, 3).astype(np.float32)
    
    # Use many clusters to capture all floor variations
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 15  # Many clusters for shadows, highlights, tile variations
    _, labels, centers = cv2.kmeans(bottom_flat, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    # Get top 8 dominant colors (covers most floor variations)
    unique, counts = np.unique(labels, return_counts=True)
    top_indices = np.argsort(counts)[-8:]  # Top 8 colors
    floor_colors = centers[top_indices]
    
    print(f"   Detected {len(floor_colors)} floor color variants (including shadows)")
    
    # Method 3: Very lenient multi-color segmentation
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Create combined mask for all floor colors
    combined_mask = np.zeros((height, width), dtype=np.float32)
    
    for i, floor_color in enumerate(floor_colors):
        floor_color_lab = cv2.cvtColor(np.uint8([[floor_color]]), cv2.COLOR_BGR2LAB)[0][0]
        color_diff = np.sqrt(np.sum((image_lab.astype(np.float32) - floor_color_lab) ** 2, axis=2))
        
        # VERY lenient threshold to capture shadows and highlights
        threshold = np.percentile(color_diff[int(height * 0.5):, :], 92)  # 92nd percentile
        
        # Create soft mask
        color_mask = np.clip((threshold - color_diff) / threshold, 0, 1)
        combined_mask = np.maximum(combined_mask, color_mask)
    
    # Method 4: Add tile pattern probability
    tile_boost = (tile_probability.astype(np.float32) / 255.0) * 0.4
    combined_mask = np.clip(combined_mask + tile_boost, 0, 1)
    
    # Method 5: Shadow-aware spatial weighting
    # Very lenient - floor starts from 30% down and weights are more permissive
    spatial_weight = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        if y < height * 0.30:
            weight = 0.0  # Top 30% - walls/windows
        elif y < height * 0.45:
            # Transition 30-45% - gradual ramp up (shadows here)
            normalized_y = (y - height * 0.30) / (height * 0.15)
            weight = normalized_y ** 1.2  # Less aggressive curve
        else:
            # Bottom 55% - floor with high confidence
            normalized_y = (y - height * 0.45) / (height * 0.55)
            weight = 0.5 + normalized_y * 0.5  # Start at 0.5, increase to 1.0
        spatial_weight[y, :] = weight
    
    # Combine all signals
    combined_mask = combined_mask * spatial_weight
    
    # Convert to binary mask with very low threshold to capture shadows
    mask_binary = (combined_mask > 0.18).astype(np.uint8) * 255  # Very low threshold
    
    # Method 6: Texture-based floor detection (marble has consistent texture)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect texture uniformity (floor has similar texture throughout)
    texture = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    texture = np.abs(texture)
    texture_norm = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Floor typically has medium texture (not too smooth, not too busy)
    texture_mask = ((texture_norm > 10) & (texture_norm < 150)).astype(np.uint8) * 255
    
    # Combine with main mask
    mask_binary = cv2.bitwise_or(mask_binary, cv2.bitwise_and(texture_mask, (spatial_weight > 0.3).astype(np.uint8) * 255))
    
    # Method 7: GrabCut for intelligent segmentation
    if auto_mode:
        print("   🤖 Applying AI-based segmentation...")
        grabcut_mask = np.zeros((height, width), dtype=np.uint8)
        
        # More aggressive GrabCut initialization
        grabcut_mask[mask_binary > 100] = cv2.GC_FGD      # Definite foreground
        grabcut_mask[mask_binary > 50] = cv2.GC_PR_FGD    # Probable foreground
        grabcut_mask[spatial_weight < 0.1] = cv2.GC_BGD   # Definite background
        
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            cv2.grabCut(image, grabcut_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
            mask_grabcut = np.where((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            
            # Merge with existing mask
            mask_binary = cv2.bitwise_or(mask_binary, mask_grabcut)
            print("   ✅ AI segmentation applied")
        except Exception as e:
            print(f"   ⚠️ AI segmentation skipped: {e}")
    
    # Method 8: Remove only obvious furniture (not floor variations)
    mask_no_objects = detect_and_remove_objects(image, mask_binary)
    
    # Method 9: Light morphology - preserve floor area
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(mask_no_objects, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # Fill gaps between floor tiles
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_close, iterations=3)
    
    # Method 10: Keep main floor + large connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
    
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        total_area = mask_clean.size
        
        # Keep components that are at least 2% of image (floor sections)
        final_mask = np.zeros_like(mask_clean)
        kept = 0
        for i in range(len(areas)):
            if areas[i] >= total_area * 0.02:  # At least 2% of image
                component_mask = (labels == (i + 1)).astype(np.uint8) * 255
                final_mask = cv2.bitwise_or(final_mask, component_mask)
                kept += 1
        
        mask_clean = final_mask
        print(f"   Merged {kept} floor sections")
    
    # Method 11: Expand to capture floor edges
    kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_clean = cv2.dilate(mask_clean, kernel_expand, iterations=2)
    
    # Final smoothing
    mask_clean = cv2.GaussianBlur(mask_clean, (7, 7), 0)
    _, mask_clean = cv2.threshold(mask_clean, 80, 255, cv2.THRESH_BINARY)
    
    # Calculate coverage
    coverage = (np.sum(mask_clean > 0) / mask_clean.size) * 100
    print(f"✅ Floor detected: {coverage:.1f}% coverage")
    
    return mask_clean


def advanced_floor_detection(image: np.ndarray, auto_mode: bool = False) -> np.ndarray:
    """
    Advanced floor detection using multiple methods
    
    Args:
        image: Input image (BGR)
        auto_mode: If True, uses more aggressive detection for automatic workflow
    
    Returns:
        Floor mask (0-255)
    """
    print("\n🤖 Running advanced floor detection...")
    
    try:
        # Try to use industry detector
        from detectors.industry_floor_detector import IndustryFloorDetector
        
        detector = IndustryFloorDetector(confidence_threshold=0.5, use_gpu=False)
        mask = detector.detect_floor(image)
        
        return mask
    
    except Exception as e:
        print(f"⚠️ Advanced detector failed: {e}")
        print("📍 Falling back to simple detection...")
        return simple_floor_detection(image, auto_mode=auto_mode)


def apply_tile_to_floor(image: np.ndarray, floor_mask: np.ndarray, tile_path: str) -> np.ndarray:
    """
    Apply tile texture to detected floor area with proper perspective matching
    
    Args:
        image: Original image
        floor_mask: Binary floor mask (0-255)
        tile_path: Path to tile texture image
        
    Returns:
        Image with tiled floor following the original floor plane
    """
    print("\n🎨 Applying tile texture with perspective correction...")
    
    # Load tile texture
    tile = cv2.imread(tile_path)
    if tile is None:
        print(f"❌ Error: Could not load tile image: {tile_path}")
        return image
    
    print(f"   Tile size: {tile.shape[1]}x{tile.shape[0]} pixels")
    
    # Get floor region dimensions
    ys, xs = np.where(floor_mask > 0)
    if len(xs) == 0:
        print("❌ Error: Empty floor mask")
        return image
    
    # Find floor contour to detect perspective plane
    contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("❌ Error: No floor contour found")
        return image
    
    # Get largest contour (main floor area)
    floor_contour = max(contours, key=cv2.contourArea)
    
    # Approximate contour to quadrilateral (4 corners)
    epsilon = 0.02 * cv2.arcLength(floor_contour, True)
    approx = cv2.approxPolyDP(floor_contour, epsilon, True)
    
    # If we don't have 4 points, use convex hull and select 4 corners
    if len(approx) != 4:
        hull = cv2.convexHull(floor_contour)
        # Simplify hull to 4 points
        epsilon = 0.05 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        # If still not 4 points, use bounding box
        if len(approx) != 4:
            rect = cv2.minAreaRect(floor_contour)
            approx = cv2.boxPoints(rect)
            approx = np.int0(approx).reshape(-1, 1, 2)
    
    # Get the 4 corner points
    corners = approx.reshape(4, 2).astype(np.float32)
    
    # Sort corners: top-left, top-right, bottom-right, bottom-left
    # Sort by y-coordinate
    corners_sorted_y = corners[np.argsort(corners[:, 1])]
    top_points = corners_sorted_y[:2]
    bottom_points = corners_sorted_y[2:]
    
    # Sort top points by x-coordinate (left to right)
    top_points = top_points[np.argsort(top_points[:, 0])]
    # Sort bottom points by x-coordinate (left to right)
    bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]
    
    # Ordered corners: TL, TR, BR, BL
    floor_corners = np.float32([
        top_points[0],      # top-left
        top_points[1],      # top-right
        bottom_points[1],   # bottom-right
        bottom_points[0]    # bottom-left
    ])
    
    print(f"   Floor plane corners detected:")
    print(f"     Top-left:     ({floor_corners[0][0]:.0f}, {floor_corners[0][1]:.0f})")
    print(f"     Top-right:    ({floor_corners[1][0]:.0f}, {floor_corners[1][1]:.0f})")
    print(f"     Bottom-right: ({floor_corners[2][0]:.0f}, {floor_corners[2][1]:.0f})")
    print(f"     Bottom-left:  ({floor_corners[3][0]:.0f}, {floor_corners[3][1]:.0f})")
    
    # Calculate dimensions for the flat tile pattern
    # Use the maximum width and height from floor corners
    width_top = np.linalg.norm(floor_corners[1] - floor_corners[0])
    width_bottom = np.linalg.norm(floor_corners[2] - floor_corners[3])
    height_left = np.linalg.norm(floor_corners[3] - floor_corners[0])
    height_right = np.linalg.norm(floor_corners[2] - floor_corners[1])
    
    max_width = int(max(width_top, width_bottom))
    max_height = int(max(height_left, height_right))
    
    print(f"   Floor plane dimensions: {max_width}x{max_height} pixels")
    
    # Create tiled pattern large enough to cover the floor
    tile_h, tile_w = tile.shape[:2]
    
    # Calculate how many tiles needed
    num_tiles_x = int(np.ceil(max_width / tile_w)) + 1
    num_tiles_y = int(np.ceil(max_height / tile_h)) + 1
    
    print(f"   Creating {num_tiles_x}x{num_tiles_y} tile pattern...")
    
    # Create large tiled pattern
    tiled_pattern = np.tile(tile, (num_tiles_y, num_tiles_x, 1))
    
    # Crop to exact needed size
    tiled_pattern = tiled_pattern[:max_height, :max_width]
    
    # Define source points (corners of flat tiled pattern)
    src_corners = np.float32([
        [0, 0],                          # top-left
        [max_width - 1, 0],              # top-right
        [max_width - 1, max_height - 1], # bottom-right
        [0, max_height - 1]              # bottom-left
    ])
    
    # Calculate homography matrix to warp tiles onto floor plane
    print("   Computing perspective transformation...")
    H, _ = cv2.findHomography(src_corners, floor_corners)
    
    # Warp tiled pattern onto floor plane
    tiled_warped = cv2.warpPerspective(
        tiled_pattern,
        H,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    print("   Applying lighting and blending...")
    
    # Apply lighting from original image to tiles for realism
    # Convert to LAB color space for better lighting transfer
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    tiles_lab = cv2.cvtColor(tiled_warped, cv2.COLOR_BGR2LAB)
    
    # Extract channels
    l_original, a_original, b_original = cv2.split(image_lab)
    l_tiles, a_tiles, b_tiles = cv2.split(tiles_lab)
    
    # Blend lightness channel (preserves shadows, highlights, and lighting)
    l_blended = cv2.addWeighted(l_original, 0.5, l_tiles, 0.5, 0)
    
    # Merge back
    tiles_lab_adjusted = cv2.merge([l_blended, a_tiles, b_tiles])
    tiles_adjusted = cv2.cvtColor(tiles_lab_adjusted, cv2.COLOR_LAB2BGR)
    
    # Create 3-channel mask for blending
    mask_3channel = cv2.cvtColor(floor_mask, cv2.COLOR_GRAY2BGR) / 255.0
    
    # Apply Gaussian blur for seamless edge blending
    mask_3channel = cv2.GaussianBlur(mask_3channel, (15, 15), 0)
    
    # Blend tiles with original image
    result = (image * (1 - mask_3channel) + tiles_adjusted * mask_3channel).astype(np.uint8)
    
    print("✅ Tile texture applied with perspective matching!")
    
    return result


def run_interactive_tile_workflow(room_image: np.ndarray, 
                                  tile_image_path: str,
                                  output_dir: str) -> int:
    """
    Run interactive tile visualization workflow with multi-polygon support
    
    Args:
        room_image: Room image array
        tile_image_path: Path to tile texture
        output_dir: Output directory for results
        
    Returns:
        0 on success, 1 on error
    """
    print("\n" + "="*70)
    print("  INTERACTIVE TILE VISUALIZATION WORKFLOW")
    print("  Multi-Polygon Floor Capture + Live Tile Preview")
    print("="*70)
    
    # Load tile texture
    print("\n📂 Loading tile texture...")
    tile_texture = cv2.imread(tile_image_path)
    
    if tile_texture is None:
        print(f"❌ Error: Could not load tile texture: {tile_image_path}")
        return 1
    
    print(f"   ✓ Tile texture: {tile_texture.shape[1]}x{tile_texture.shape[0]}")
    
    # =================================================================
    # PHASE 1: INTERACTIVE FLOOR CAPTURE (Multi-Polygon)
    # =================================================================
    print("\n" + "="*70)
    print("PHASE 1: MULTI-POLYGON FLOOR CAPTURE")
    print("="*70)
    print("\n🎮 Instructions:")
    print("   • Click to add points for polygon")
    print("   • Press SPACE to finish current polygon")
    print("   • Add multiple polygons for discontinuous areas")
    print("   • Press ENTER when all polygons are complete")
    
    capture = InteractiveFloorCapture(room_image)
    
    # Create window
    window_capture = "Floor Capture - Multi-Polygon (Click points, SPACE=finish, ENTER=done)"
    cv2.namedWindow(window_capture, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_capture, 1200, 800)
    
    # Mouse callback
    def mouse_callback(event, x, y, flags, param):
        if capture.mode == 'polygon':
            if event == cv2.EVENT_LBUTTONDOWN:
                capture.add_polygon_point(x, y)
        elif capture.mode == 'scribble':
            if event == cv2.EVENT_LBUTTONDOWN:
                capture.start_scribble(x, y)
            elif event == cv2.EVENT_MOUSEMOVE:
                if flags & cv2.EVENT_FLAG_LBUTTON:
                    capture.continue_scribble(x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                capture.stop_scribble()
    
    cv2.setMouseCallback(window_capture, mouse_callback)
    
    print("\n🎮 CONTROLS:")
    print("   [p] - Polygon mode (click to add points)")
    print("   [s] - Scribble mode (drag to draw)")
    print("   [SPACE] - Finish current polygon and start new one")
    print("   [d] - Delete last completed polygon")
    print("   [r] - Refine mask (region growing)")
    print("   [u] - Refine mask (superpixel)")
    print("   [e] - Refine mask (edge snapping)")
    print("   [c] - Clear ALL polygons")
    print("   [ENTER] - Finish capture and proceed")
    print("   [ESC] - Quit")
    
    current_mode = 'polygon'
    current_mask = None
    show_help = True
    
    while True:
        # Create display
        display = room_image.copy()
        
        # Draw rough mask (all polygons combined)
        rough_mask = capture.get_rough_mask()
        if np.any(rough_mask > 0):
            overlay = display.copy()
            overlay[rough_mask > 0] = [0, 255, 0]
            display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)
        
        # Draw all completed polygons
        for idx, poly_points in enumerate(capture.completed_polygons):
            color = (0, 255, 255)  # Yellow for completed
            for i, pt in enumerate(poly_points):
                cv2.circle(display, pt, 4, color, -1)
            # Draw lines
            if len(poly_points) > 1:
                for i in range(len(poly_points)):
                    pt1 = poly_points[i]
                    pt2 = poly_points[(i + 1) % len(poly_points)]
                    cv2.line(display, pt1, pt2, color, 2)
            
            # Label
            if len(poly_points) > 0:
                centroid_x = int(np.mean([pt[0] for pt in poly_points]))
                centroid_y = int(np.mean([pt[1] for pt in poly_points]))
                cv2.putText(display, f"Poly {idx + 1}", (centroid_x - 30, centroid_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw current polygon being drawn
        if current_mode == 'polygon':
            for i, pt in enumerate(capture.polygon_points):
                cv2.circle(display, pt, 5, (0, 0, 255), -1)
                cv2.putText(display, str(i + 1), (pt[0] + 10, pt[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if len(capture.polygon_points) > 1:
                for i in range(len(capture.polygon_points) - 1):
                    cv2.line(display, capture.polygon_points[i], 
                            capture.polygon_points[i + 1], (0, 255, 255), 2)
        
        # Display mode and help
        mode_text = f"Mode: {current_mode.upper()} | Polygons: {capture.get_polygon_count()}"
        cv2.putText(display, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        if show_help:
            help_y = 70
            help_texts = [
                "[p]Polygon [s]Scribble [SPACE]FinishPoly [d]DeleteLast",
                "[r]RegionGrow [u]Superpixel [e]EdgeSnap",
                "[c]Clear [ENTER]Finish [ESC]Quit"
            ]
            for text in help_texts:
                cv2.putText(display, text, (10, help_y), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 1)
                help_y += 25
        
        cv2.imshow(window_capture, display)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('p'):
            current_mode = 'polygon'
            capture.set_mode('polygon')
        
        elif key == ord('s'):
            current_mode = 'scribble'
            capture.set_mode('scribble')
        
        elif key == 32:  # SPACE
            if current_mode == 'polygon':
                if capture.finish_current_polygon():
                    print(f"✅ Polygon completed. Total: {capture.get_polygon_count()}")
        
        elif key == ord('d'):
            if capture.delete_last_polygon():
                print(f"🗑️ Deleted polygon. Remaining: {capture.get_polygon_count()}")
        
        elif key == ord('r'):
            rough_mask = capture.get_rough_mask()
            if np.any(rough_mask > 0):
                refined = capture.refine_mask_region_growing(rough_mask, tolerance=25)
                current_mask = refined
                print("\n✅ Mask refined with region growing")
        
        elif key == ord('u'):
            rough_mask = capture.get_rough_mask()
            if np.any(rough_mask > 0):
                refined = capture.refine_mask_superpixel(rough_mask)
                current_mask = refined
                print("\n✅ Mask refined with superpixels")
        
        elif key == ord('e'):
            rough_mask = capture.get_rough_mask()
            if np.any(rough_mask > 0):
                refined = capture.refine_mask_edge_snapping(rough_mask, snap_distance=20)
                current_mask = refined
                print("\n✅ Mask refined with edge snapping")
        
        elif key == ord('c'):
            capture.clear()
            current_mask = None
            print("\n🗑️ Cleared all polygons")
        
        elif key == 13:  # ENTER
            if len(capture.polygon_points) >= 3:
                capture.finish_current_polygon()
            
            current_mask = capture.get_rough_mask()
            if np.any(current_mask > 0):
                print(f"\n✅ Capture finished with {capture.get_polygon_count()} polygon(s)")
                break
            else:
                print("\n⚠️ No floor area captured yet!")
        
        elif key == 27:  # ESC
            print("\n❌ Cancelled")
            cv2.destroyAllWindows()
            return 1
    
    cv2.destroyWindow(window_capture)
    
    if current_mask is None or np.count_nonzero(current_mask) == 0:
        print("\n❌ No floor area captured. Exiting.")
        return 1
    
    # =================================================================
    # PHASE 2: MASK REFINEMENT
    # =================================================================
    print("\n" + "="*70)
    print("PHASE 2: MASK REFINEMENT")
    print("="*70)
    
    refinement = MaskRefinement()
    
    print("   Filling holes...")
    refined_mask = refinement.fill_holes(current_mask, min_hole_size=100)
    
    print("   Removing islands...")
    refined_mask = refinement.remove_islands(refined_mask, min_island_size=200)
    
    print("   Smoothing edges...")
    refined_mask = refinement.smooth_edges(refined_mask, method='morphological', kernel_size=5)
    
    print(f"\n✅ Mask refined! Area: {np.count_nonzero(refined_mask)} pixels")
    
    # =================================================================
    # PHASE 3: PLANE APPROXIMATION
    # =================================================================
    print("\n" + "="*70)
    print("PHASE 3: PLANE APPROXIMATION")
    print("="*70)
    
    plane_approx = PlaneApproximation(room_image, refined_mask)
    
    # Compute convex hull
    plane_approx.compute_convex_hull()
    
    # Compute quadrilateral
    quad_points = plane_approx.compute_minimum_area_quad()
    
    # Estimate perspective
    plane_approx.estimate_perspective_transform(target_width=1000, target_height=1000)
    
    # Try to detect vanishing point
    plane_approx.detect_vanishing_point()
    
    # Show visualization
    vis = plane_approx.visualize_plane_approximation()
    cv2.namedWindow("Plane Approximation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Plane Approximation", 800, 600)
    cv2.imshow("Plane Approximation", vis)
    print("\n   📊 Showing plane approximation...")
    print("   Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyWindow("Plane Approximation")
    
    print("\n✅ Plane approximation complete!")
    
    # =================================================================
    # PHASE 4: TILE PROJECTION & BLENDING
    # =================================================================
    print("\n" + "="*70)
    print("PHASE 4: TILE PROJECTION & BLENDING")
    print("="*70)
    
    # Create projection engine
    quad_points = plane_approx.get_projection_quadrilateral()
    projection_engine = TileProjectionEngine(tile_texture, refined_mask, quad_points)
    
    # Set initial scale
    tile_scale = 1.0
    projection_engine.set_tile_scale(tile_scale)
    
    # Project tiles
    warped_tiles, warped_tiles_clipped = projection_engine.project_tiles_full(grid_size=1000)
    
    # Create blending engine
    blending = RealisticBlending(room_image, refined_mask)
    
    # Blend
    result_image = blending.blend_complete(
        warped_tiles_clipped,
        match_brightness=True,
        match_color=True,
        apply_lighting=True,
        alpha=0.85,
        feather_size=10
    )
    
    print("\n✅ Tile projection and blending complete!")
    
    # =================================================================
    # PHASE 5: INTERACTIVE PREVIEW
    # =================================================================
    print("\n" + "="*70)
    print("PHASE 5: INTERACTIVE PREVIEW WITH SCALE ADJUSTMENT")
    print("="*70)
    
    window_preview = "Live Preview - Adjust Tile Scale"
    cv2.namedWindow(window_preview, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_preview, 1200, 800)
    
    # Trackbar callback
    def on_scale_change(val):
        nonlocal tile_scale, result_image
        tile_scale = val / 10.0  # Scale 1-50 -> 0.1-5.0
        print(f"\n🔄 Updating tile scale: {tile_scale:.2f}")
        
        # Update projection engine
        projection_engine.set_tile_scale(tile_scale)
        
        # Re-project
        warped_tiles, warped_tiles_clipped = projection_engine.project_tiles_full(grid_size=1000)
        
        # Re-blend
        result_image = blending.blend_complete(
            warped_tiles_clipped,
            match_brightness=True,
            match_color=True,
            apply_lighting=True,
            alpha=0.85,
            feather_size=10
        )
        print("   ✓ Preview updated")
    
    cv2.createTrackbar('Tile Scale x10', window_preview, 10, 50, on_scale_change)
    
    print("\n🎮 INTERACTIVE CONTROLS:")
    print("   [Trackbar] - Adjust tile scale")
    print("   [s] - Save result")
    print("   [ESC/Q] - Quit")
    
    while True:
        cv2.imshow(window_preview, result_image)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            # Save result
            output_path = os.path.join(output_dir, "interactive_tiled_result.jpg")
            cv2.imwrite(output_path, result_image)
            print(f"\n💾 Result saved: {output_path}")
            
            # Also save mask
            mask_path = os.path.join(output_dir, "interactive_floor_mask.png")
            cv2.imwrite(mask_path, refined_mask)
            print(f"💾 Mask saved: {mask_path}")
        
        elif key == 27 or key == ord('q'):  # ESC or Q
            break
    
    cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("✅ INTERACTIVE TILE VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\n📁 Results saved to: {output_dir}/")
    
    return 0


def main():
    """Main workflow"""
    parser = argparse.ArgumentParser(
        description="Enhanced Floor Detection & Correction with Photoshop-like Tools"
    )
    parser.add_argument('--image', '-i', type=str, default='room.jpg',
                       help='Input image path (default: room.jpg from assets)')
    parser.add_argument('--tile', '-t', type=str, default='tile.jpg',
                       help='Tile texture image path (default: tile.jpg from assets)')
    parser.add_argument('--detector', '-d', type=str, choices=['simple', 'industry'],
                       default='simple', help='Detection method')
    parser.add_argument('--auto', '-a', action='store_true', default=False,
                       help='Fully automatic mode - no user interaction')
    parser.add_argument('--manual', '-m', action='store_true',
                       help='Manual mode - enable interactive correction tools')
    parser.add_argument('--interactive-tile', action='store_true', default=True,
                       help='Enable interactive tile visualization with multi-polygon support (DEFAULT)')
    parser.add_argument('--auto-export', action='store_true',
                       help='Automatically export all visualizations')
    parser.add_argument('--output-dir', '-o', type=str, default='outputs',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # If manual mode or interactive tile mode is specified, disable auto mode
    if args.manual or args.interactive_tile:
        args.auto = False
    
    print("\n" + "="*70)
    if args.interactive_tile:
        print("🎨 INTERACTIVE TILE VISUALIZATION MODE")
        print("   Multi-Polygon Floor Capture + Live Tile Preview")
    elif args.auto:
        print("🤖 FULLY AUTOMATIC FLOOR TILE VISUALIZATION")
        print("   AI-Powered Object Removal & Floor Detection")
    else:
        print("🚀 ENHANCED FLOOR DETECTION & CORRECTION WORKFLOW")
        print("   With Photoshop-Style Smart Selection Tools")
    print("="*70 + "\n")
    
    # Auto-detect image and tile in assets folder if not found in current directory
    assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
    
    # Check if image exists, if not, look in assets folder
    if not os.path.exists(args.image):
        asset_image_path = os.path.join(assets_dir, args.image)
        if os.path.exists(asset_image_path):
            args.image = asset_image_path
            print(f"✅ Found image in assets folder: {args.image}")
        else:
            print(f"❌ Error: Image not found: {args.image}")
            print(f"   Also checked: {asset_image_path}")
            print("\n💡 Usage:")
            print(f"   python {sys.argv[0]} --image your_room.jpg")
            if args.auto:
                print(f"   python {sys.argv[0]} --auto  (fully automatic)")
            return 1
    
    # Auto-detect tile.jpg if not specified
    if args.tile is None:
        # Check current directory first
        if os.path.exists('tile.jpg'):
            args.tile = 'tile.jpg'
            print("✅ Auto-detected tile.jpg in current directory")
        # Then check assets folder
        elif os.path.exists(os.path.join(assets_dir, 'tile.jpg')):
            args.tile = os.path.join(assets_dir, 'tile.jpg')
            print("✅ Auto-detected tile.jpg in assets folder")
    
    # If tile is specified but not found, check assets folder
    if args.tile and not os.path.exists(args.tile):
        asset_tile_path = os.path.join(assets_dir, args.tile)
        if os.path.exists(asset_tile_path):
            args.tile = asset_tile_path
            print(f"✅ Found tile in assets folder: {args.tile}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load image
    print(f"📂 Loading image: {args.image}")
    image = cv2.imread(args.image)
    
    if image is None:
        print(f"❌ Error: Could not load image")
        return 1
    
    print(f"   Size: {image.shape[1]}x{image.shape[0]} pixels")
    
    if args.auto:
        print(f"   Mode: AUTOMATIC (no user interaction)")
    elif args.interactive_tile:
        print(f"   Mode: INTERACTIVE TILE VISUALIZATION")
    else:
        print(f"   Mode: INTERACTIVE (manual correction available)")
    
    # =================================================================
    # INTERACTIVE TILE VISUALIZATION WORKFLOW
    # =================================================================
    if args.interactive_tile:
        if not args.tile or not os.path.exists(args.tile):
            print(f"\n❌ Error: Tile texture required for interactive mode!")
            print(f"   Please provide --tile <path_to_tile_image>")
            return 1
        
        return run_interactive_tile_workflow(image, args.tile, args.output_dir)
    
    # =================================================================
    # STEP 1: AUTO-DETECT FLOOR
    # =================================================================
    print("\n" + "="*70)
    print("STEP 1: AUTOMATIC FLOOR DETECTION")
    if args.auto:
        print("       (With Automatic Object Removal)")
    print("="*70)
    
    if args.detector == 'industry':
        initial_mask = advanced_floor_detection(image, auto_mode=args.auto)
    else:
        initial_mask = simple_floor_detection(image, auto_mode=args.auto)
    
    # Save initial detection
    cv2.imwrite(os.path.join(args.output_dir, '01_initial_detection.png'), initial_mask)
    
    # Create visualization
    initial_viz = image.copy()
    initial_viz[initial_mask > 0] = initial_viz[initial_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    cv2.imwrite(os.path.join(args.output_dir, '02_initial_visualization.png'), initial_viz)
    
    # =================================================================
    # STEP 2: ENHANCED PREVIEW (Skip in auto mode)
    # =================================================================
    if not args.auto:
        print("\n" + "="*70)
        print("STEP 2: ENHANCED PREVIEW")
        print("="*70)
        
        # Show enhanced preview with multiple visualization modes
        accepted = preview_floor_detection(image, initial_mask, auto_export=args.auto_export)
        
        if not accepted:
            print("\n❌ Detection rejected. Exiting...")
            return 0
    else:
        print("\n" + "="*70)
        print("STEP 2: AUTO MODE - Skipping Preview")
        print("="*70)
        print("✅ Auto mode enabled - using detection without review")
    
    # =================================================================
    # STEP 3: ENHANCED INTERACTIVE CORRECTION (Skip in auto mode)
    # =================================================================
    if not args.auto:
        print("\n" + "="*70)
        print("STEP 3: INTERACTIVE CORRECTION WITH SMART TOOLS")
        print("="*70)
        
        # Launch enhanced correction interface
        corrected_mask = enhanced_floor_correction(image, initial_mask)
        
        if corrected_mask is None:
            print("\n❌ Correction cancelled. Exiting...")
            return 0
    else:
        print("\n" + "="*70)
        print("STEP 3: AUTO MODE - Skipping Manual Correction")
        print("="*70)
        print("✅ Using automatic detection result (no manual correction needed)")
        corrected_mask = initial_mask
    
    # =================================================================
    # STEP 4: SAVE RESULTS
    # =================================================================
    print("\n" + "="*70)
    print("STEP 4: SAVING RESULTS")
    print("="*70)
    
    # Save corrected mask
    mask_path = os.path.join(args.output_dir, '03_corrected_mask.png')
    cv2.imwrite(mask_path, corrected_mask)
    print(f"✅ Corrected mask saved: {mask_path}")
    
    # Save final visualization
    final_viz = image.copy()
    final_viz[corrected_mask > 0] = final_viz[corrected_mask > 0] * 0.4 + np.array([0, 255, 0]) * 0.6
    viz_path = os.path.join(args.output_dir, '04_final_visualization.png')
    cv2.imwrite(viz_path, final_viz)
    print(f"✅ Final visualization saved: {viz_path}")
    
    # Create before/after comparison
    before = initial_viz
    after = final_viz
    
    # Resize if needed for side-by-side
    max_width = 800
    if image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
        # Display resize with high quality
        before = cv2.resize(before, (new_width, new_height), interpolation=cv2.INTER_AREA)
        after = cv2.resize(after, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Create side-by-side comparison
    comparison = np.hstack([before, after])
    
    # Add labels
    cv2.putText(comparison, "BEFORE (Auto-Detected)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(comparison, "AFTER (Corrected)", (before.shape[1] + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    comparison_path = os.path.join(args.output_dir, '05_comparison.png')
    cv2.imwrite(comparison_path, comparison)
    print(f"✅ Before/after comparison saved: {comparison_path}")
    
    # Calculate improvement statistics
    initial_coverage = (np.sum(initial_mask > 0) / initial_mask.size) * 100
    final_coverage = (np.sum(corrected_mask > 0) / corrected_mask.size) * 100
    
    print("\n" + "="*70)
    print("📊 STATISTICS")
    print("="*70)
    print(f"Initial detection: {initial_coverage:.1f}% coverage")
    print(f"Final corrected:   {final_coverage:.1f}% coverage")
    print(f"Change:            {final_coverage - initial_coverage:+.1f}%")
    
    # Calculate difference
    added = cv2.bitwise_and(corrected_mask, cv2.bitwise_not(initial_mask))
    removed = cv2.bitwise_and(initial_mask, cv2.bitwise_not(corrected_mask))
    
    added_pixels = np.sum(added > 0)
    removed_pixels = np.sum(removed > 0)
    
    print(f"\nUser corrections:")
    print(f"  Added:   {added_pixels:,} pixels ({added_pixels/initial_mask.size*100:.2f}%)")
    print(f"  Removed: {removed_pixels:,} pixels ({removed_pixels/initial_mask.size*100:.2f}%)")
    
    # =================================================================
    # STEP 5: APPLY TILE TEXTURE (if provided)
    # =================================================================
    if args.tile:
        print("\n" + "="*70)
        print("STEP 5: TILE TEXTURE APPLICATION")
        print("="*70)
        
        # Check if tile exists
        if not os.path.exists(args.tile):
            print(f"⚠️ Warning: Tile image not found: {args.tile}")
            print("   Skipping tile application...")
        else:
            # Apply tile texture to floor
            tiled_result = apply_tile_to_floor(image, corrected_mask, args.tile)
            
            # Save tiled result
            tiled_path = os.path.join(args.output_dir, '06_tiled_floor.jpg')
            cv2.imwrite(tiled_path, tiled_result)
            print(f"✅ Tiled floor saved: {tiled_path}")
            
            # Create before/after tile comparison
            tile_before = image.copy()
            tile_after = tiled_result
            
            # Resize if needed
            if image.shape[1] > max_width:
                scale = max_width / image.shape[1]
                new_width = int(image.shape[1] * scale)
                new_height = int(image.shape[0] * scale)
                tile_before = cv2.resize(tile_before, (new_width, new_height))
                tile_after = cv2.resize(tile_after, (new_width, new_height))
            
            # Create side-by-side
            tile_comparison = np.hstack([tile_before, tile_after])
            
            # Add labels
            cv2.putText(tile_comparison, "ORIGINAL", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(tile_comparison, "WITH TILES", (tile_before.shape[1] + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            tile_comparison_path = os.path.join(args.output_dir, '07_tile_comparison.png')
            cv2.imwrite(tile_comparison_path, tile_comparison)
            print(f"✅ Tile comparison saved: {tile_comparison_path}")
    
    print("\n" + "="*70)
    print("✅ WORKFLOW COMPLETE!")
    print("="*70)
    print(f"\nAll outputs saved to: {args.output_dir}/")
    
    if args.auto:
        print("\n🤖 Automatic mode completed successfully!")
        print("   ✅ Floor detected automatically")
        print("   ✅ Objects removed automatically")
        if args.tile and os.path.exists(args.tile):
            print("   ✅ Tiles applied automatically")
            print(f"\n📁 Final result: {args.output_dir}/06_tiled_floor.jpg")
        else:
            print(f"\n📁 Floor mask: {args.output_dir}/03_corrected_mask.png")
    else:
        if args.tile and os.path.exists(args.tile):
            print("\n🎉 Your floor has been perfectly detected, corrected, and tiled!")
            print(f"   Check out: {args.output_dir}/06_tiled_floor.jpg")
        else:
            print("\n🎉 You now have a perfectly corrected floor mask!")
            print("   Ready for tile visualization or further processing.")
            print(f"\n💡 Tip: Add --tile tile.jpg to apply tile texture automatically!")
    
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
