"""
Headless Tile Visualization Engine
===================================
Non-interactive version of the tile visualization engine for web API usage.
Processes floor masks and applies tiles without GUI requirements.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
import os
import sys

# Direct imports to avoid GUI-related dependencies in __init__.py
# Import from the same directory (utils)
_current_dir = os.path.dirname(os.path.abspath(__file__))

# Import modules directly without going through __init__.py
import importlib.util

def _import_module(name, path):
    """Import a module from a specific file path"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import required modules directly
_mask_refinement = _import_module('mask_refinement', os.path.join(_current_dir, 'mask_refinement.py'))
_plane_approximation = _import_module('plane_approximation', os.path.join(_current_dir, 'plane_approximation.py'))
_professional_tile_installer = _import_module('professional_tile_installer', os.path.join(_current_dir, 'professional_tile_installer.py'))
_realistic_blending = _import_module('realistic_blending', os.path.join(_current_dir, 'realistic_blending.py'))

MaskRefinement = _mask_refinement.MaskRefinement
PlaneApproximation = _plane_approximation.PlaneApproximation
ProfessionalTileInstaller = _professional_tile_installer.ProfessionalTileInstaller
RealisticBlending = _realistic_blending.RealisticBlending


def create_polygon_mask(image: np.ndarray, points: list) -> np.ndarray:
    """
    Create a binary mask from polygon points.
    
    Args:
        image: Input image to determine dimensions
        points: List of [x, y] polygon points
        
    Returns:
        Binary mask (0/255)
    """
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if len(points) >= 3:
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    
    return mask


def refine_mask(mask: np.ndarray) -> np.ndarray:
    """
    Refine the floor mask using morphological operations.
    
    Args:
        mask: Input binary mask
        
    Returns:
        Refined mask
    """
    refinement = MaskRefinement()
    
    # Fill holes
    refined = refinement.fill_holes(mask, min_hole_size=100)
    
    # Remove small islands
    refined = refinement.remove_islands(refined, min_island_size=200)
    
    # Smooth edges
    refined = refinement.smooth_edges(refined, method='morphological', kernel_size=5)
    
    return refined


def extract_floor_quad(floor_mask: np.ndarray) -> np.ndarray:
    """
    Extract a reliable 4-corner trapezoid from the floor mask
    representing the floor plane in perspective (TL, TR, BR, BL).
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
        if len(xs) == 0:
            return np.float32([
                [0, 0], [w-1, 0], [w-1, h-1], [0, h-1]
            ])
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
    
    return np.float32([tl, tr, br, bl])


def visualize(
    room_image: np.ndarray,
    tile_image: np.ndarray,
    polygon_points: list,
    tile_size_cm: float = 200.0,
    grout_width_mm: float = 2.0,
    resolution: int = 4000
) -> Dict[str, Any]:
    """
    Apply tile texture to the floor area defined by polygon points.
    
    Args:
        room_image: Input room image (BGR)
        tile_image: Tile texture image (BGR)
        polygon_points: List of [x, y] points defining the floor area
        tile_size_cm: Real-world tile size in cm (default: 200cm)
        grout_width_mm: Grout width in mm (default: 2mm)
        resolution: Internal resolution for tiling (default: 4000)
        
    Returns:
        Dictionary with:
        - 'result': Final visualization image (BGR)
        - 'mask': Floor mask used
        - 'success': Boolean success flag
        - 'message': Status message
    """
    try:
        # Validate inputs
        if room_image is None or room_image.size == 0:
            return {
                'result': None,
                'mask': None,
                'success': False,
                'message': 'Invalid room image'
            }
        
        if tile_image is None or tile_image.size == 0:
            return {
                'result': None,
                'mask': None,
                'success': False,
                'message': 'Invalid tile image'
            }
        
        if len(polygon_points) < 3:
            return {
                'result': None,
                'mask': None,
                'success': False,
                'message': 'Need at least 3 points to define floor area'
            }
        
        # Create mask from polygon
        floor_mask = create_polygon_mask(room_image, polygon_points)
        
        if np.count_nonzero(floor_mask) == 0:
            return {
                'result': None,
                'mask': None,
                'success': False,
                'message': 'Empty floor mask - invalid polygon'
            }
        
        # Refine the mask
        refined_mask = refine_mask(floor_mask)
        
        # Get plane approximation
        plane_approx = PlaneApproximation(room_image, refined_mask)
        plane_approx.compute_convex_hull()
        quad_points = plane_approx.compute_minimum_area_quad()
        
        # Ensure tile image is large enough
        th, tw = tile_image.shape[:2]
        if tw < 512 or th < 512:
            scale_t = max(512/tw, 512/th)
            tile_image = cv2.resize(
                tile_image, (0, 0), 
                fx=scale_t, fy=scale_t, 
                interpolation=cv2.INTER_LANCZOS4
            )
        
        # Try pattern detection
        try:
            _pattern_detector = _import_module('pattern_detector', os.path.join(_current_dir, 'pattern_detector.py'))
            detector = _pattern_detector.PatternDetector(tile_image)
            tile_image, pattern_name = detector.analyze_optimal_pattern()
        except (ImportError, Exception):
            pattern_name = 'standard'
        
        # Create professional tile installer
        tile_installer = ProfessionalTileInstaller(tile_image, refined_mask, quad_points)
        tile_installer.set_tile_size(tile_size_cm)
        tile_installer.set_grout_width(grout_width_mm)
        
        # Install tiles
        warped_tiles, warped_tiles_clipped = tile_installer.install_complete(resolution=resolution)
        
        # Blend with original image
        blending = RealisticBlending(room_image, refined_mask)
        result_image = blending.blend_complete(
            warped_tiles_clipped,
            match_brightness=True,
            match_color=False,
            apply_lighting=False,
            alpha=0.95,
            feather_size=3
        )
        
        return {
            'result': result_image,
            'mask': refined_mask,
            'success': True,
            'message': f'Tile applied successfully with {pattern_name} pattern'
        }
        
    except Exception as e:
        return {
            'result': None,
            'mask': None,
            'success': False,
            'message': f'Error: {str(e)}'
        }


def auto_detect_floor(image: np.ndarray) -> np.ndarray:
    """
    Automatically detect floor area in the image.
    
    Args:
        image: Input room image (BGR)
        
    Returns:
        Floor mask (0/255)
    """
    height, width = image.shape[:2]
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Analyze bottom region for floor color
    bottom_region = image[int(height * 0.6):, :]
    
    # Get dominant colors from bottom
    bottom_flat = bottom_region.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(bottom_flat, 8, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    # Get top 4 dominant colors
    unique, counts = np.unique(labels, return_counts=True)
    top_indices = np.argsort(counts)[-4:]
    floor_colors = centers[top_indices]
    
    # Create combined color mask
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    combined_mask = np.zeros((height, width), dtype=np.float32)
    
    for floor_color in floor_colors:
        floor_color_lab = cv2.cvtColor(np.uint8([[floor_color]]), cv2.COLOR_BGR2LAB)[0][0]
        color_diff = np.sqrt(np.sum((image_lab.astype(np.float32) - floor_color_lab) ** 2, axis=2))
        threshold = np.percentile(color_diff[int(height * 0.6):, :], 85)
        color_mask = np.clip((threshold - color_diff) / threshold, 0, 1)
        combined_mask = np.maximum(combined_mask, color_mask)
    
    # Spatial weighting - favor bottom of image
    spatial_weight = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        if y < height * 0.35:
            weight = 0.0
        elif y < height * 0.5:
            weight = (y - height * 0.35) / (height * 0.15)
        else:
            weight = 0.6 + (y - height * 0.5) / (height * 0.5) * 0.4
        spatial_weight[y, :] = weight
    
    combined_mask = combined_mask * spatial_weight
    
    # Convert to binary
    mask = (combined_mask > 0.25).astype(np.uint8) * 255
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8) * 255
    
    return mask


def get_floor_polygon(mask: np.ndarray) -> list:
    """
    Extract polygon points from a floor mask.
    
    Args:
        mask: Binary floor mask
        
    Returns:
        List of [x, y] polygon points
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    # Get largest contour
    largest = max(contours, key=cv2.contourArea)
    
    # Simplify contour
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    
    # Convert to list of points
    points = approx.reshape(-1, 2).tolist()
    
    return points
