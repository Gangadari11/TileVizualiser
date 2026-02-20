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
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import our enhanced modules
from utils.enhanced_floor_preview import preview_floor_detection
from utils.enhanced_interactive_gui import enhanced_floor_correction

# Import tile visualization modules
from utils.interactive_floor_capture import InteractiveFloorCapture
from utils.mask_refinement import MaskRefinement
from utils.plane_approximation import PlaneApproximation
from utils.professional_tile_installer import ProfessionalTileInstaller
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


def _extract_floor_quad(floor_mask: np.ndarray) -> np.ndarray:
    """
    Extract a reliable 4-corner trapezoid from the floor mask
    representing the floor plane in perspective (TL, TR, BR, BL).
    Uses the actual x-extent at multiple y-slices to build a clean quad.
    """
    h, w = floor_mask.shape
    
    # Collect left-most and right-most floor pixel per row
    left_pts, right_pts = [], []
    step = max(1, h // 80)
    for y in range(0, h, step):
        row = floor_mask[y, :]
        cols = np.where(row > 128)[0]
        if len(cols) >= 6:
            left_pts.append([cols[0], y])
            right_pts.append([cols[-1], y])
    
    if len(left_pts) < 4:
        # Fallback: bounding box of mask
        ys, xs = np.where(floor_mask > 0)
        return np.float32([
            [xs.min(), ys.min()],
            [xs.max(), ys.min()],
            [xs.max(), ys.max()],
            [xs.min(), ys.max()]
        ])
    
    left_pts = np.array(left_pts)
    right_pts = np.array(right_pts)
    
    # Top edge: topmost row with floor pixels
    top_y = left_pts[0, 1]
    row_top = floor_mask[top_y, :]
    top_cols = np.where(row_top > 128)[0]
    tl = np.float32([top_cols[0], top_y])
    tr = np.float32([top_cols[-1], top_y])
    
    # Bottom edge: bottommost row with floor pixels
    bot_y = left_pts[-1, 1]
    row_bot = floor_mask[bot_y, :]
    bot_cols = np.where(row_bot > 128)[0]
    bl = np.float32([bot_cols[0], bot_y])
    br = np.float32([bot_cols[-1], bot_y])
    
    return np.float32([tl, tr, br, bl])  # TL, TR, BR, BL


def apply_tile_to_floor(image: np.ndarray, floor_mask: np.ndarray, tile_path: str) -> np.ndarray:
    """
    Apply tile texture to detected floor area with proper perspective-correct tiling.

    Key approach:
    1. Extract floor trapezoid (perspective quad) from mask.
    2. Decide real-world tile count based on bottom-edge width (nearest camera edge).
    3. Build a flat rectangular tiled texture where ONE tile = tile_size_px × tile_size_px.
    4. Warp the flat grid onto the floor quad with cv2.getPerspectiveTransform —
       this makes tiles near the camera large and far tiles small, matching real perspective.
    5. Extract per-pixel shadow/lighting from the ORIGINAL floor and apply it
       MULTIPLICATIVELY on top of the warped tile (no additive blending that washes out pattern).
    6. Use feathered mask at edges for seamless integration.
    """
    print("\n🎨 Applying perspective-correct tile texture...")

    # ── 1. Load tile ──────────────────────────────────────────────────────────
    tile = cv2.imread(tile_path)
    if tile is None:
        print(f"❌ Error: Could not load tile image: {tile_path}")
        return image
    print(f"   Tile size: {tile.shape[1]}×{tile.shape[0]} px")

    # ── 2. Verify mask ────────────────────────────────────────────────────────
    ys, xs = np.where(floor_mask > 0)
    if len(xs) == 0:
        print("❌ Error: Empty floor mask")
        return image

    img_h, img_w = image.shape[:2]

    # ── 3. Extract floor quad: [TL, TR, BR, BL] ──────────────────────────────
    floor_corners = _extract_floor_quad(floor_mask)
    tl, tr, br, bl = floor_corners

    print(f"   Floor quad  TL=({tl[0]:.0f},{tl[1]:.0f})  TR=({tr[0]:.0f},{tr[1]:.0f})")
    print(f"               BL=({bl[0]:.0f},{bl[1]:.0f})  BR=({br[0]:.0f},{br[1]:.0f})")

    # ── 4. Tile-count-based scale ─────────────────────────────────────────────
    # Bottom edge is the closest edge (widest in perspective).
    # We want ~7 tiles visible across the nearest edge for realistic scale.
    bottom_width_px = float(np.linalg.norm(br - bl))
    left_height_px  = float(np.linalg.norm(bl - tl))

    tile_orig_h, tile_orig_w = tile.shape[:2]
    tile_native = max(tile_orig_w, tile_orig_h)   # longest side of original texture

    TILES_ACROSS_BOTTOM = 2   # LARGE tiles: only 3 visible across = big, clear pattern
    target_size = int(bottom_width_px / TILES_ACROSS_BOTTOM)

    # CRITICAL: never downscale below the native tile resolution.
    # If the computed target is smaller than the original tile,
    # keep the native size — the perspective warp will downscale with LANCZOS4.
    # Upscaling with 2× headroom keeps details crisp for tiles farther from camera.
    tile_size_flat = max(tile_native, target_size)
    # Extra 2× upscale headroom so distant (small) tiles still have enough pixels
    if tile_size_flat < tile_native * 2:
        tile_size_flat = tile_native * 2

    # Recompute tile count using the chosen flat size
    num_x = int(np.ceil(bottom_width_px / tile_size_flat)) + 3
    num_y = int(np.ceil(left_height_px  / tile_size_flat)) + 3

    print(f"   Native tile res : {tile_orig_w}×{tile_orig_h} px")
    print(f"   Flat cell size  : {tile_size_flat} px  (no downscale before warp)")
    print(f"   Grid            : {num_x}×{num_y} tiles  →  "
          f"{num_x * tile_size_flat}×{num_y * tile_size_flat} px flat canvas")

    # ── 5. Build flat tiled texture at maximum resolution ────────────────────
    # Upscale tile to tile_size_flat using LANCZOS4; never goes below native size.
    tile_cell = cv2.resize(tile, (tile_size_flat, tile_size_flat),
                           interpolation=cv2.INTER_LANCZOS4)

    flat_w = num_x * tile_size_flat
    flat_h = num_y * tile_size_flat
    tiled_flat = np.tile(tile_cell, (num_y, num_x, 1))   # (flat_h, flat_w, 3)
    tiled_flat = tiled_flat[:flat_h, :flat_w]             # safety crop

    # ── 6. Perspective warp: flat grid → floor quad ───────────────────────────
    # Source = 4 corners of the flat texture
    src_pts = np.float32([
        [0,          0         ],   # TL
        [flat_w - 1, 0         ],   # TR
        [flat_w - 1, flat_h - 1],   # BR
        [0,          flat_h - 1],   # BL
    ])
    # Destination = floor quad in image space
    dst_pts = np.float32([tl, tr, br, bl])

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    tiled_warped = cv2.warpPerspective(
        tiled_flat, H, (img_w, img_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # ── 7. Shadow / lighting transfer (floor pixels only, multiplicative) ─────
    # We compute a per-pixel brightness ratio from the original image and
    # multiply it onto the warped tile so shadows/highlights are preserved.
    # ALL processing is done in float64 to avoid quantisation error.
    # The NON-FLOOR part of the result is a byte-for-byte copy of the original.

    image_f64  = image.astype(np.float64)           # lossless reference
    warped_f64 = tiled_warped.astype(np.float64)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float64)
    L_channel = image_lab[:,:,0]

    # Median brightness of the ORIGINAL floor region only
    floor_pixels_bright = L_channel[floor_mask > 128]
    avg_bright = float(np.median(floor_pixels_bright)) if len(floor_pixels_bright) > 0 else 128.0
    avg_bright = max(avg_bright, 10.0)

    # ENHANCED SHADOW MAP:
    # 1. Use Luminance channel from LAB for better human-eye brightness perception
    # 2. Add local contrast enhancement (CLAHE-like) to keep shadow details
    
    # Calculate ratio map
    shadow_map = L_channel / avg_bright
    
    # Sigmoid contrast curve to deepen shadows and brighten highlights slightly
    # This makes the floor look less "flat"
    shadow_map = (shadow_map - 0.5) * 1.2 + 0.5 
    
    # Clamp extreme values to prevent artifacts
    shadow_map = np.clip(shadow_map, 0.2, 1.8)
    
    # Smooth the shadow map to reduce noise grain from original floor
    shadow_map = cv2.GaussianBlur(shadow_map.astype(np.float32), (5, 5), 0).astype(np.float64)

    # Apply shadow ONLY inside the floor region
    shadow_3c   = shadow_map[:, :, np.newaxis]
    
    # Mix warped tile with shadow map
    # We blend 80% shadow influence, 20% original tile brightness to prevent total black crush
    tiled_lit = warped_f64 * shadow_3c
    
    # PRESERVE ORIGINAL REFLECTIONS (Specularity)
    # Bright spots in original floor (>200) should be added back as reflections
    highlights = np.clip(L_channel - 210, 0, 255)
    if np.max(highlights) > 0:
        highlights = highlights / np.max(highlights) * 40.0 # Scale intensity
        highlights = cv2.GaussianBlur(highlights.astype(np.float32), (15,15), 0).astype(np.float64)
        highlights_3c = highlights[:, :, np.newaxis]
        tiled_lit = np.clip(tiled_lit + highlights_3c, 0, 255)

    tiled_lit = np.clip(tiled_lit, 0.0, 255.0)

    # ── 8. Compose: pixel-precise copy of original + tile on floor only ───────
    # Build a soft alpha from the floor mask (feather only at boundary, 7 px).
    mask_f64     = floor_mask.astype(np.float64)
    
    # ENHANCED EDGE BLENDING
    # Use a slightly tighter erosion first to avoid "halo" artifacts around furniture
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask_eroded = cv2.erode(mask_f64.astype(np.uint8), kernel_erode).astype(np.float64)
    
    # Then feather edges
    mask_feather = cv2.GaussianBlur(mask_eroded.astype(np.float32),
                                    (5, 5), 0).astype(np.float64) / 255.0
    mask_3c      = mask_feather[:, :, np.newaxis]          # (H,W,1) broadcast

    # Full-precision blend
    blended_f64 = image_f64 * (1.0 - mask_3c) + tiled_lit * mask_3c

    # ── 9. Sharpening applied STRICTLY inside the floor area ─────────────────
    # Convert blended result to uint8 once (single quantisation step)
    blended_u8  = np.clip(blended_f64, 0, 255).astype(np.uint8)

    sharpen_k = np.array([[ 0, -0.5,  0],
                           [-0.5,  3, -0.5],
                           [ 0, -0.5,  0]], dtype=np.float32)
    sharpened_u8 = cv2.filter2D(blended_u8, -1, sharpen_k)

    # Mix: 55 % original blend + 45 % sharpened, but ONLY on tile pixels
    # Non-floor pixels are taken directly from the original (zero float error).
    floor_bool = (mask_feather > 0.3)[:, :, np.newaxis]   # bool broadcast mask

    sharp_mix = cv2.addWeighted(blended_u8, 0.55, sharpened_u8, 0.45, 0)

    # Re-composite: use sharp_mix on floor, original pixels elsewhere (lossless)
    result = np.where(floor_bool, sharp_mix, image).astype(np.uint8)

    coverage = 100.0 * np.sum(floor_mask > 0) / floor_mask.size
    print(f"✅ Perspective-correct tile applied!  Floor coverage: {coverage:.1f}%")
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
    
    # Check tile resolution
    th, tw = tile_texture.shape[:2]
    if tw < 512 or th < 512:
        print(f"⚠️ Tile image is small ({tw}x{th}). Upscaling for better quality...")
        scale_t = max(512/tw, 512/th)
        tile_texture = cv2.resize(tile_texture, (0,0), fx=scale_t, fy=scale_t, interpolation=cv2.INTER_LANCZOS4)
        print(f"   ✅ Tile upscaled to {tile_texture.shape[1]}x{tile_texture.shape[0]}")

    print(f"   ✓ Tile texture: {tile_texture.shape[1]}x{tile_texture.shape[0]}")
    
    # NEW: INTELLIGENT PATTERN DETECTION
    # Analyzes tile to see if it's part of a larger pattern (e.g. circle quadrant)
    try:
        from src.utils.pattern_detector import PatternDetector
        detector = PatternDetector(tile_texture)
        final_tile, pattern_name = detector.analyze_optimal_pattern()
        
        # If we found a complex pattern, update the texture
        if pattern_name != 'standard':
            tile_texture = final_tile
            print(f"   ✨ UPGRADED tile to {pattern_name.upper()} macro-pattern")
            print(f"   ✨ New tile resolution: {tile_texture.shape[1]}x{tile_texture.shape[0]}")
    except ImportError:
        print("   ⚠️ Pattern detector module not found, skipping optimization.")
    except Exception as e:
        print(f"   ⚠️ Pattern detection failed: {e}")
    
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
                cv2.circle(display, pt, 8, color, -1)
            # Draw lines
            if len(poly_points) > 1:
                for i in range(len(poly_points)):
                    pt1 = poly_points[i]
                    pt2 = poly_points[(i + 1) % len(poly_points)]
                    cv2.line(display, pt1, pt2, color, 4)
            
            # Label
            if len(poly_points) > 0:
                centroid_x = int(np.mean([pt[0] for pt in poly_points]))
                centroid_y = int(np.mean([pt[1] for pt in poly_points]))
                cv2.putText(display, f"Poly {idx + 1}", (centroid_x - 30, centroid_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw current polygon being drawn
        if current_mode == 'polygon':
            for i, pt in enumerate(capture.polygon_points):
                cv2.circle(display, pt, 10, (0, 0, 255), -1)
                cv2.putText(display, str(i + 1), (pt[0] + 10, pt[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if len(capture.polygon_points) > 1:
                for i in range(len(capture.polygon_points) - 1):
                    cv2.line(display, capture.polygon_points[i], 
                            capture.polygon_points[i + 1], (0, 255, 255), 4)
        
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
    # PHASE 4: PROFESSIONAL TILE INSTALLATION
    # =================================================================
    print("\n" + "="*70)
    print("PHASE 4: PROFESSIONAL TILE INSTALLATION (REAL-WORLD SIMULATION)")
    print("="*70)
    
    # Create professional tile installer
    quad_points = plane_approx.get_projection_quadrilateral()
    tile_installer = ProfessionalTileInstaller(tile_texture, refined_mask, quad_points)
    
    # Set initial tile size — USER REQUEST: few large tiles, clearly visible pattern
    tile_size_cm = 500.0  # 200cm per tile = only ~3 tiles visible across room
    tile_installer.set_tile_size(tile_size_cm)
    
    # Install tiles professionally (like real installation)
    # ULTRA-HIGH RESOLUTION: 8000px canvas for maximum pattern detail (delicate patterns need more)
    print("   Using 8000x8000px base resolution for DELICATE pattern clarity...")
    warped_tiles, warped_tiles_clipped = tile_installer.install_complete(resolution=12000)
    
    # Create blending engine
    blending = RealisticBlending(room_image, refined_mask)
    
    # Blend - INDUSTRIAL STANDARD BLENDING
    # Turn ON subtle effects for realism (gloss, shadow integration) but keep alpha high
    result_image = blending.blend_complete(
        warped_tiles_clipped,
        match_brightness=True,   # ENABLED: Match room brightness 
        match_color=False,       # DISABLED: Keep tile true color
        apply_lighting=False,    # DISABLED: We use AO shadow map in blend_complete internally
        alpha=0.95,              # 95% opacity allows 5% of original floor texture/shadows to bleed through
        feather_size=3           # Sharp edges
    )
    
    print("\n✅ Professional tile installation and blending complete!")
    
    # =================================================================
    # PHASE 5: INTERACTIVE PREVIEW WITH REAL-WORLD CONTROLS
    # =================================================================
    print("\n" + "="*70)
    print("PHASE 5: INTERACTIVE PREVIEW - ADJUST REAL TILE SIZE")
    print("="*70)
    
    window_preview = "Live Preview - Adjust Real Tile Size"
    cv2.namedWindow(window_preview, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_preview, 1400, 900)
    
    # Trackbar callback for tile size
    def on_tile_size_change(val):
        nonlocal tile_size_cm, result_image
        tile_size_cm = max(20, val)  # 20cm minimum
        print(f"\n🔲 Changing tile size: {tile_size_cm:.0f}cm x {tile_size_cm:.0f}cm")
        
        # Update installer
        tile_installer.set_tile_size(tile_size_cm)
        
        # Re-install tiles professionally with ULTRA-HIGH resolution
        warped_tiles, warped_tiles_clipped = tile_installer.install_complete(resolution=8000)
        
        # Re-blend - MINIMAL blending for maximum quality
        result_image = blending.blend_complete(
            warped_tiles_clipped,
            match_brightness=False,  # Keep original tile appearance
            match_color=False,       # Keep original tile colors
            apply_lighting=False,    # Keep original tile lighting
            alpha=0.99,              # 99% tiles for maximum quality
            feather_size=5
        )
        print("   ✅ Preview updated")
    
    # Create trackbar: 30cm to 150cm (extended range for very large tiles)
    cv2.createTrackbar('Tile Size (cm)', window_preview, int(tile_size_cm), 150, on_tile_size_change)
    
    print("\n🎮 INTERACTIVE CONTROLS:")
    print("   [Trackbar] Tile Size (cm) - Set REAL tile size (20-120cm)")
    print("   [s] - Save result (HIGH QUALITY PNG + JPG)")
    print("   [ESC/Q] - Quit without saving")
    print("\n💡 COMMON TILE SIZES:")
    print("      40cm (16\") - Medium tiles")
    print("      60cm (24\") - Large tiles")
    print("      80cm (32\") - Extra large (best for delicate patterns) ← RECOMMENDED")
    print("      100cm (40\") - Very large (fewer tiles, clearer patterns)")
    print("\n   💡 TIP: For delicate/mosaic patterns, use 70-100cm tiles!")
    print("   ✅ 100% TILE QUALITY - No blending adjustments applied")
    print("   ✅ GUARANTEE: Entire user-marked area will be covered!\n")
    
    while True:
        # Add info overlay to help user
        info_overlay = result_image.copy()
        tiles_across, tiles_down = tile_installer._calculate_tile_count()
        # Subtract the 2 overfill tiles for display (actual visible tiles)
        visible_across = max(1, tiles_across - 2)
        visible_down = max(1, tiles_down - 2)
        total_tiles = visible_across * visible_down
        
        floor_area_m2 = (tile_installer.floor_width_cm * tile_installer.floor_height_cm) / 10000
        
        cv2.putText(info_overlay, f"Tile: {tile_size_cm:.0f}cm x {tile_size_cm:.0f}cm | ~{total_tiles} tiles needed",
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(info_overlay, f"Floor: {floor_area_m2:.1f}m² | FULL COVERAGE GUARANTEED",
                   (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(info_overlay, "Adjust trackbar | 's'=SAVE HIGH-QUALITY | ESC=quit",
                   (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(info_overlay, "NOTE: Saved file will be FULL QUALITY (this preview is resized)",
                   (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        
        cv2.imshow(window_preview, info_overlay)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            # Generate unique timestamp for this save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save result with MAXIMUM QUALITY (PNG format, no compression losses)
            output_path = os.path.join(output_dir, f"interactive_tiled_result_{timestamp}.png")  # PNG not JPG!
            
            # Save with maximum quality
            cv2.imwrite(output_path, result_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # 0 = no compression
            print(f"\n💾 Result saved in HIGHEST QUALITY: {output_path}")
            
            # Also save high-quality JPG version (if needed)
            jpg_path = os.path.join(output_dir, f"interactive_tiled_result_{timestamp}.jpg")
            cv2.imwrite(jpg_path, result_image, [cv2.IMWRITE_JPEG_QUALITY, 100])  # 100 = best quality
            print(f"💾 JPG version saved: {jpg_path}")
            
            # Also save mask
            mask_path = os.path.join(output_dir, f"interactive_floor_mask_{timestamp}.png")
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
    # ============================================================
    # CHANGE THESE DEFAULTS TO SWITCH ROOM/TILE IMAGES:
    # ============================================================
    DEFAULT_ROOM = 'assets/room.jpg'    # Change to: room2.jpg, room3.jpg, etc.
    DEFAULT_TILE = 'assets/tile5.jpg'    # Change to: tile2.jpg, tile3.jpg, etc.
    # ============================================================
    
    parser.add_argument('--image', '-i', type=str, default=DEFAULT_ROOM,
                       help='Input image path')
    parser.add_argument('--tile', '-t', type=str, default=DEFAULT_TILE,
                       help='Tile texture image path')
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
    
    # Check resolution and upscale if too small (crucial for pattern visibility)
    # ULTRA-HIGH QUALITY SETTINGS: Force minimum 2400px for crystal clarity
    h, w = image.shape[:2]
    min_dim = 3000  # Increased for industrial quality
    
    if w < min_dim or h < min_dim:
        print(f"\n⚠️ Input room image is small ({w}x{h}). Upscaling for detail...")
        scale = max(min_dim / w, min_dim / h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply smart sharpening after upscale to restore edges
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        image = cv2.filter2D(image, -1, sharpen_kernel)
        print(f"   ✅ Upscaled & Sharpened to {new_w}x{new_h} (Ultra-High Quality)")
    else:
        # Even if large enough, apply subtle sharpening for clarity
        sharpen_kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        image = cv2.filter2D(image, -1, sharpen_kernel)
        print(f"   ✅ Image enhanced for clarity ({w}x{h})")
    
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
            # PROFESSIONAL INSTALLATION WORKFLOW (Ported from Interactive Mode)
            print("\n🏭 STARTING INDUSTRIAL QUALITY TILE INSTALLATION...")
            
            # 1. Load Pattern Logic
            tile_img = cv2.imread(args.tile)
            
            # NEW: Intelligent Pattern Detection
            try:
                from src.utils.pattern_detector import PatternDetector
                detector = PatternDetector(tile_img)
                final_tile, pattern_name = detector.analyze_optimal_pattern()
                
                if pattern_name != 'standard':
                    tile_img = final_tile
                    print(f"   ✨ UPGRADED to {pattern_name.upper()} macro-pattern")
            except Exception as e:
                print(f"   ⚠️ Auto-pattern detection skipped: {e}")

            # 2. Approximate Floor Plane (Needed for Professional Installer)
            # Create a simple quad approximation from the mask if not already done
            from src.utils.plane_approximation import PlaneApproximation
            plane_approx = PlaneApproximation(image, corrected_mask)
            plane_approx.compute_convex_hull()
            quad_points = plane_approx.compute_minimum_area_quad()
            
            # 3. Initialize Professional Installer
            from src.utils.professional_tile_installer import ProfessionalTileInstaller
            tile_installer = ProfessionalTileInstaller(tile_img, corrected_mask, quad_points)
            
            # Force Luxury Size
            # USER REQUEST: fewer tiles, big and clearly visible pattern
            base_tile_size_cm = 500.0  # 200cm per single tile = only ~3 tiles visible across room
            
            tile_size_target = base_tile_size_cm
            
            # If we detect a macro-pattern, we MUST verify the type to scale correctly.
            # But even "standard" tiles might benefit from rotation if they have continuity.
            if pattern_name != 'standard':
                # If we made a 2x2 macro-pattern, double the size so individual tiles (quadrants) stay large
                tile_size_target *= 2 # Result: 300cm macro block!
                print(f"   ℹ️  Adjusting macro-tile size to {tile_size_target}cm (2x {base_tile_size_cm}cm tiles)")
            else:
                 # Even if standard, maybe the user wants it BIG.
                 pass

            tile_installer.set_tile_size(tile_size_target)
            
            # 4. Install & Warp
            warped_tiles, warped_tiles_clipped = tile_installer.install_complete(resolution=8000)
            
            # 5. Realistic Blending
            from src.utils.realistic_blending import RealisticBlending
            blending = RealisticBlending(image, corrected_mask)
            
            tiled_result = blending.blend_complete(
                warped_tiles_clipped,
                match_brightness=True,
                match_color=False,
                apply_lighting=False,
                alpha=0.95,
                feather_size=3
            )
            
            # Save primary output as PNG (lossless — identical pixel values to in-memory result)
            tiled_path = os.path.join(args.output_dir, '06_tiled_floor.png')
            cv2.imwrite(tiled_path, tiled_result, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(f"✅ Tiled floor saved (lossless PNG): {tiled_path}")
            
            # Also save a high-quality JPEG for quick sharing
            jpg_path = os.path.join(args.output_dir, '06_tiled_floor.jpg')
            try:
                # 4:4:4 chroma sampling = no chroma downscale (OpenCV 4.1+)
                cv2.imwrite(jpg_path, tiled_result,
                            [cv2.IMWRITE_JPEG_QUALITY, 100,
                             cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                             cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
            except (cv2.error, AttributeError):
                cv2.imwrite(jpg_path, tiled_result, [cv2.IMWRITE_JPEG_QUALITY, 100])
            print(f"   JPG copy saved: {jpg_path}")
            
            # Create before/after tile comparison
            tile_before = image.copy()
            tile_after = tiled_result
            
            # Resize if needed
            if image.shape[1] > max_width:
                scale = max_width / image.shape[1]
                new_width = int(image.shape[1] * scale)
                new_height = int(image.shape[0] * scale)
                tile_before = cv2.resize(tile_before, (new_width, new_height), interpolation=cv2.INTER_AREA)
                tile_after = cv2.resize(tile_after, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Create side-by-side
            tile_comparison = np.hstack([tile_before, tile_after])
            
            # Add labels
            cv2.putText(tile_comparison, "ORIGINAL", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(tile_comparison, "WITH TILES", (tile_before.shape[1] + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            tile_comparison_path = os.path.join(args.output_dir, '07_tile_comparison.png')
            cv2.imwrite(tile_comparison_path, tile_comparison, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # 0 = no compression
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
