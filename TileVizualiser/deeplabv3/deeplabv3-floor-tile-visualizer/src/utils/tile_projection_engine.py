"""
Tile Projection Engine
=======================
Projects tile textures onto the detected floor plane using 
perspective transformation and geometric warping.

Key Features:
- Load and prepare tile textures
- Generate repeating tile grids
- Warp tiles into floor plane perspective
- Clip warped tiles using irregular masks
- Support for tile scale adjustment
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class TileProjectionEngine:
    """
    Engine for projecting tile textures onto floor plane
    """
    
    def __init__(self, 
                 tile_texture: np.ndarray,
                 mask: np.ndarray,
                 quad_points: np.ndarray):
        """
        Initialize tile projection engine
        
        Args:
            tile_texture: Tile texture image (BGR)
            mask: Floor mask for clipping (binary, 0 or 255)
            quad_points: 4 corner points for perspective (4 x 2 array)
        """
        self.tile_texture = tile_texture
        self.mask = mask
        self.quad_points = quad_points.astype(np.float32)
        
        self.height, self.width = mask.shape[:2]
        
        # Tile parameters (can be adjusted)
        self.tile_scale = 1.0
        self.grout_width = 2
        self.grout_color = (180, 180, 180)
        
        print("✅ Tile projection engine initialized")
    
    def set_tile_scale(self, scale: float):
        """
        Set tile scale factor
        
        Args:
            scale: Scale factor (1.0 = normal, >1 = larger tiles, <1 = smaller tiles)
        """
        self.tile_scale = max(0.1, min(5.0, scale))
        print(f"📏 Tile scale set to: {self.tile_scale:.2f}")
    
    def set_grout_properties(self, width: int, color: Tuple[int, int, int]):
        """
        Set grout line properties
        
        Args:
            width: Grout line width in pixels
            color: Grout color (B, G, R)
        """
        self.grout_width = width
        self.grout_color = color
        print(f"🎨 Grout: width={width}px, color={color}")
    
    def generate_tile_grid(self, 
                          grid_width: int = 1000,
                          grid_height: int = 1000,
                          tile_size: int = 100) -> np.ndarray:
        """
        Generate a repeating tile grid in top-down view
        
        Args:
            grid_width: Width of grid
            grid_height: Height of grid
            tile_size: Size of each tile in pixels
            
        Returns:
            Tiled image (BGR)
        """
        print("🔲 Generating tile grid...")
        
        # Adjust tile size by scale
        tile_size = int(tile_size * self.tile_scale)
        
        # Resize tile texture with HIGH QUALITY interpolation
        tile_resized = cv2.resize(
            self.tile_texture,
            (tile_size - self.grout_width, tile_size - self.grout_width),
            interpolation=cv2.INTER_CUBIC
        )
        
        # Create grid
        num_tiles_x = (grid_width + tile_size - 1) // tile_size + 1
        num_tiles_y = (grid_height + tile_size - 1) // tile_size + 1
        
        # Create canvas with grout color
        grid = np.full(
            (num_tiles_y * tile_size, num_tiles_x * tile_size, 3),
            self.grout_color,
            dtype=np.uint8
        )
        
        # Place tiles
        for i in range(num_tiles_y):
            for j in range(num_tiles_x):
                y_start = i * tile_size
                x_start = j * tile_size
                
                y_end = y_start + tile_size - self.grout_width
                x_end = x_start + tile_size - self.grout_width
                
                if y_end <= grid.shape[0] and x_end <= grid.shape[1]:
                    grid[y_start:y_end, x_start:x_end] = tile_resized
        
        # Crop to exact size
        grid = grid[:grid_height, :grid_width]
        
        print(f"   ✓ Grid generated: {grid.shape[1]}x{grid.shape[0]}, {num_tiles_x * num_tiles_y} tiles")
        return grid
    
    def warp_tiles_to_perspective(self, 
                                  tile_grid: np.ndarray,
                                  inverse_matrix: np.ndarray) -> np.ndarray:
        """
        Warp tile grid from top-down to image perspective
        
        Args:
            tile_grid: Tile grid in top-down view
            inverse_matrix: Inverse perspective transformation matrix
            
        Returns:
            Warped tile image
        """
        print("🔄 Warping tiles to perspective...")
        
        # Warp tile grid to image perspective with HIGH QUALITY interpolation
        warped = cv2.warpPerspective(
            tile_grid,
            inverse_matrix,
            (self.width, self.height),
            flags=cv2.INTER_CUBIC,  # Changed from LINEAR to CUBIC for better quality
            borderMode=cv2.BORDER_CONSTANT
        )
        
        print("   ✓ Tiles warped")
        return warped
    
    def clip_with_mask(self, warped_tiles: np.ndarray) -> np.ndarray:
        """
        Clip warped tiles using the floor mask
        
        Args:
            warped_tiles: Warped tile image
            
        Returns:
            Clipped tile image (with transparency)
        """
        print("✂️ Clipping tiles with mask...")
        
        # Convert to BGRA (add alpha channel)
        if warped_tiles.shape[2] == 3:
            warped_tiles_alpha = cv2.cvtColor(warped_tiles, cv2.COLOR_BGR2BGRA)
        else:
            warped_tiles_alpha = warped_tiles.copy()
        
        # Set alpha channel based on mask
        warped_tiles_alpha[:, :, 3] = self.mask
        
        print("   ✓ Tiles clipped")
        return warped_tiles_alpha
    
    def project_tiles_full(self, 
                          grid_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete tile projection pipeline
        
        Args:
            grid_size: Size of tile grid to generate
            
        Returns:
            Tuple of (warped_tiles, warped_tiles_clipped)
        """
        print("🎬 Starting full tile projection...")
        
        # Step 1: Generate tile grid
        tile_size = int(80 * self.tile_scale)
        tile_grid = self.generate_tile_grid(grid_size, grid_size, tile_size)
        
        # Step 2: Compute inverse perspective transform
        dst_points = np.array([
            [0, 0],
            [grid_size - 1, 0],
            [grid_size - 1, grid_size - 1],
            [0, grid_size - 1]
        ], dtype=np.float32)
        
        forward_matrix = cv2.getPerspectiveTransform(self.quad_points, dst_points)
        inverse_matrix = np.linalg.inv(forward_matrix)
        
        # Step 3: Warp to perspective
        warped_tiles = self.warp_tiles_to_perspective(tile_grid, inverse_matrix)
        
        # Step 4: Clip with mask
        warped_tiles_clipped = self.clip_with_mask(warped_tiles)
        
        print("✅ Tile projection complete!")
        return warped_tiles, warped_tiles_clipped
    
    def add_tile_variation(self, tile_grid: np.ndarray, 
                          variation_amount: float = 0.1) -> np.ndarray:
        """
        Add subtle brightness variation to tiles for realism
        
        Args:
            tile_grid: Input tile grid
            variation_amount: Amount of variation (0-1)
            
        Returns:
            Tile grid with variation
        """
        print("✨ Adding tile variation...")
        
        # Convert to float
        tile_grid_float = tile_grid.astype(np.float32)
        
        # Generate random variation pattern
        variation = np.random.randn(tile_grid.shape[0], tile_grid.shape[1]) * variation_amount
        variation = cv2.GaussianBlur(variation, (51, 51), 0)
        
        # Apply variation
        for c in range(3):
            tile_grid_float[:, :, c] *= (1.0 + variation)
        
        # Clip and convert back
        tile_grid_varied = np.clip(tile_grid_float, 0, 255).astype(np.uint8)
        
        print("   ✓ Variation added")
        return tile_grid_varied
    
    def add_perspective_distortion(self, 
                                   tile_grid: np.ndarray,
                                   strength: float = 0.1) -> np.ndarray:
        """
        Add subtle perspective distortion to tile grid for more realism
        
        Args:
            tile_grid: Input tile grid
            strength: Strength of distortion (0-1)
            
        Returns:
            Distorted tile grid
        """
        h, w = tile_grid.shape[:2]
        
        # Create subtle distortion map
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        
        for i in range(h):
            for j in range(w):
                # Perspective scaling factor
                scale = 1.0 + strength * (i / h)
                
                map_x[i, j] = j * scale
                map_y[i, j] = i
        
        # Apply distortion
        distorted = cv2.remap(
            tile_grid,
            map_x,
            map_y,
            cv2.INTER_LINEAR
        )
        
        return distorted
    
    def visualize_tile_grid(self, tile_grid: np.ndarray) -> np.ndarray:
        """
        Create visualization of tile grid
        
        Args:
            tile_grid: Tile grid to visualize
            
        Returns:
            Visualization image
        """
        # Resize for display if too large
        max_display_size = 800
        h, w = tile_grid.shape[:2]
        
        if max(h, w) > max_display_size:
            scale = max_display_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            display_grid = cv2.resize(tile_grid, (new_w, new_h))
        else:
            display_grid = tile_grid.copy()
        
        return display_grid
