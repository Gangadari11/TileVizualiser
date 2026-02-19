"""
Professional Tile Installation Engine
======================================
Simulates REAL-WORLD tile installation process like professionals do:

1. Calculate actual floor dimensions (in cm/inches)
2. Define real tile size (30x30cm, 45x45cm, 60x60cm, etc.)
3. Place tiles one-by-one starting from a corner
4. Cut edge tiles to fit remaining space
5. Apply grout spacing between tiles
6. Project result onto floor with perspective

This approach matches professional visualization tools:
- RoomSketcher
- Autodesk Revit
- SketchUp  
- Floorplanner
- Floor Plan Creator
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class ProfessionalTileInstaller:
    """
    Professional tile installation simulator
    """
    
    def __init__(self,
                 tile_texture: np.ndarray,
                 mask: np.ndarray,
                 quad_points: np.ndarray):
        """
        Initialize professional tile installer
        
        Args:
            tile_texture: Tile texture image (BGR)
            mask: Floor mask for clipping (binary, 0 or 255)
            quad_points: 4 corner points for perspective (4 x 2 array)
        """
        self.tile_texture = tile_texture
        self.mask = mask
        self.quad_points = quad_points.astype(np.float32)
        self.image_height, self.image_width = mask.shape[:2]
        
        # ── 1. Calculate Perceived Width and Height ─────────────────────────
        # In a perspective projection, the TOP width is much smaller than the BOTTOM width.
        # But in the real world (top-down view), floor is roughly rectangular.
        
        bottom_width_px = np.sqrt(np.sum((self.quad_points[2] - self.quad_points[3])**2))
        top_width_px    = np.sqrt(np.sum((self.quad_points[1] - self.quad_points[0])**2))
        
        left_height_px  = np.sqrt(np.sum((self.quad_points[3] - self.quad_points[0])**2))
        right_height_px = np.sqrt(np.sum((self.quad_points[2] - self.quad_points[1])**2))
        
        # KEY INSIGHT: The BOTTOM width is the closest to the true "scale" of the floor
        # in pixels relative to the camera, because it's closest to the lens.
        floor_width_px = bottom_width_px
        
        # The height is tricky. Perspective foreshortening crushes the height.
        # We estimate the true depth by assuming a FOV or vanishing point.
        # Simple heuristic: If top width is much smaller than bottom, the floor is DEEP.
        
        perspective_ratio = top_width_px / max(1.0, bottom_width_px)
        # Ratio 1.0 = Flat on (orthographic). Ratio 0.1 = Very deep floor.
        
        # Base height estimate on the average vertical pixel span
        avg_height_px = (left_height_px + right_height_px) / 2.0
        
        # Correct for foreshortening: As perspective gets stronger (ratio smaller),
        # the REAL floor depth is much larger than the pixel height suggests.
        # Multiplier = 1.0 + (1.0 - ratio) * 2.0 => Flat=1x, Deep=3x height
        depth_multiplier = 1.0 + (1.0 - perspective_ratio) * 1.5
        
        floor_height_px = avg_height_px * depth_multiplier
        
        # ── 2. Add Margins for Oversized Canvas ─────────────────────────────
        floor_width_px  *= 1.2
        floor_height_px *= 1.2
        
        # PROFESSIONAL APPROACH: Assume realistic floor dimensions
        # Average living room: 4-6 meters wide.
        # We map the BOTTOM width (closest to camera) to ~450cm.
        self.pixels_per_cm = floor_width_px / 450.0 
        self.floor_width_cm = floor_width_px / self.pixels_per_cm
        self.floor_height_cm = floor_height_px / self.pixels_per_cm
        
        # Real-world tile parameters (standard sizes in cm)
        # DEFAULT: LARGE tile size for delicate patterns to be visible
        self.tile_size_cm = 80.0  # 80cm x 80cm (32 inches) - LARGE for delicate patterns
        self.grout_width_cm = 0.3  # 3mm grout line (standard)
        self.grout_color = (110, 105, 100)  # Dark cement gray grout - clearly visible
        
        # Installation parameters
        self.start_corner = 'bottom-left'  # Where to start laying tiles
        
        print("✅ Professional Tile Installer Initialized")
        print(f"   Floor: {self.floor_width_cm:.1f}cm x {self.floor_height_cm:.1f}cm")
        print(f"   Area: {(self.floor_width_cm * self.floor_height_cm / 10000):.2f} m²")
        print(f"   Default Tiles: {self.tile_size_cm:.0f}cm x {self.tile_size_cm:.0f}cm (LARGE for delicate patterns)")
        print(f"   Grout: {self.grout_width_cm * 10:.1f}mm")
    
    def set_tile_size(self, size_cm: float):
        """
        Set real-world tile size
        
        Args:
            size_cm: Tile size in cm (common: 30, 40, 45, 60, 80, 120)
        """
        self.tile_size_cm = max(20.0, min(150.0, size_cm))
        tiles_across, tiles_down = self._calculate_tile_count()
        total = tiles_across * tiles_down
        print(f"🔲 Tile size: {self.tile_size_cm:.0f}cm x {self.tile_size_cm:.0f}cm")
        print(f"   Estimated: ~{total} tiles ({tiles_across} across x {tiles_down} down)")
    
    def set_grout_width(self, width_mm: float):
        """Set grout width in millimeters"""
        self.grout_width_cm = width_mm / 10.0
        print(f"🎨 Grout: {width_mm:.1f}mm")
    
    def _calculate_tile_count(self) -> Tuple[int, int]:
        """Calculate number of tiles needed (with overfill for complete coverage)"""
        tile_plus_grout = self.tile_size_cm + self.grout_width_cm
        # Add 2 extra tiles in each direction to ensure complete coverage
        tiles_across = int(np.ceil(self.floor_width_cm / tile_plus_grout)) + 2
        tiles_down = int(np.ceil(self.floor_height_cm / tile_plus_grout)) + 2
        return tiles_across, tiles_down
    
    def _cm_to_pixels(self, cm: float) -> int:
        """Convert cm to pixels"""
        return int(cm * self.pixels_per_cm)
    
    def install_tiles_on_flat_floor(self, resolution: int = 4000) -> np.ndarray:
        """
        Install tiles on flat floor (top-down view) - LIKE REAL INSTALLATION
        
        This simulates actual tile installation:
        1. Start from bottom-left corner
        2. Place full tiles row by row
        3. Cut tiles at edges to fit
        4. Add grout lines between tiles
        
        Args:
            resolution: Resolution of flat floor canvas
            
        Returns:
            Flat tiled floor image (top-down view)
        """
        print("\n🔨 Installing tiles (simulating real-world installation)...")
        print("   ⚠️ Creating ULTRA-HIGH RESOLUTION canvas for pattern clarity")
        
        # Create MUCH LARGER canvas for pattern detail preservation
        canvas_size = int(resolution * 1.8)  # Increased from 1.5 to 1.8
        canvas = np.full((canvas_size, canvas_size, 3), self.grout_color, dtype=np.uint8)
        
        # Calculate tile dimensions - PRIORITIZE PATTERN VISIBILITY
        tiles_across, tiles_down = self._calculate_tile_count()
        
        # CRITICAL: Calculate tile size to ensure pattern is ALWAYS visible
        # For delicate patterns (circles, florals, mosaic), each tile needs 1000-1500px minimum
        tile_px_calculated = canvas_size // max(tiles_across, tiles_down)
        
        # ENFORCE MINIMUM: 1200px per tile for delicate patterns (increased from 800px)
        # This ensures overlapping circles, mosaics, etc. are clearly visible
        tile_px = max(tile_px_calculated, 1200)
        
        # If enforcing minimum makes canvas too small, increase canvas size
        required_canvas = tile_px * max(tiles_across, tiles_down)
        if required_canvas > canvas_size:
            canvas_size = int(required_canvas * 1.3)  # 30% extra for safety
            canvas = np.full((canvas_size, canvas_size, 3), self.grout_color, dtype=np.uint8)
            print(f"   🔼 Canvas increased to {canvas_size}x{canvas_size}px for delicate pattern clarity")
        
        grout_px = max(2, int(tile_px * self.grout_width_cm / self.tile_size_cm))
        
        print(f"   Canvas: {canvas_size}x{canvas_size}px (ultra-high res for delicate patterns)")
        print(f"   Tile size in canvas: {tile_px}px (VERY LARGE for pattern detail)")
        print(f"   Real-world tile: {tile_px / self.pixels_per_cm:.1f}cm equivalent")
        print(f"   Grout in canvas: {grout_px}px")
        print(f"   Laying {tiles_across}x{tiles_down} tiles (pattern will be CRYSTAL CLEAR)")
        
        # Resize tile texture with MAXIMUM quality - preserve every detail
        # For complex patterns (geometric, floral), use LANCZOS4 and large size
        print(f"   🎨 Resizing tile texture to {tile_px - grout_px}px (preserving detail)...")
        tile_resized = cv2.resize(
            self.tile_texture,
            (tile_px - grout_px, tile_px - grout_px),
            interpolation=cv2.INTER_LANCZOS4
        )
        print(f"   ✅ Tile resized - pattern detail preserved at {tile_px - grout_px}px")
        
        # INSTALL TILES - Row by row, like real installation
        # Cover ENTIRE canvas to ensure no gaps after perspective warp
        tiles_placed = 0
        tiles_cut = 0
        
        # Calculate how many tiles we need to fill the ENTIRE oversized canvas
        max_tiles_across = int(np.ceil(canvas_size / tile_px)) + 1
        max_tiles_down = int(np.ceil(canvas_size / tile_px)) + 1
        
        # Create 3D Grout Effect Colors
        # Shadow (Top/Left of grout line) - Reduce opacity (mix with grout color)
        grout_shadow = tuple(int(0.6*c) for c in self.grout_color)
        # Highlight (Bottom/Right of grout line) - Subtler highlight
        grout_highlight = tuple(int(min(255, c + 30)) for c in self.grout_color)
        
        for row in range(max_tiles_down):
            for col in range(max_tiles_across):
                # Calculate tile position (starting from bottom-left)
                x = col * tile_px
                y = row * tile_px
                
                # Check if we're at edge - need to cut tile
                x_end = x + tile_px - grout_px
                y_end = y + tile_px - grout_px

                # Apply 3D Grout Beveling Effect (SUBTLER)
                if grout_px >= 2:
                    # Vertical Grout Line (Right side of previous tile)
                    if x > 0 and x < canvas_size:
                        # Shadow on left of grout (1px)
                        cv2.line(canvas, (x-1, y), (x-1, min(y+tile_px, canvas_size)), grout_shadow, 1)
                        
                    # Horizontal Grout Line (Bottom of previous tile)
                    if y > 0 and y < canvas_size:
                        # Shadow on top of grout (1px)
                        cv2.line(canvas, (x, y-1), (min(x+tile_px, canvas_size), y-1), grout_shadow, 1)

                
                # Stop if completely outside canvas
                if x >= canvas_size or y >= canvas_size:
                    continue
                
                # Get tile (or cut portion)
                current_tile = tile_resized.copy()
                
                # CUT TILE if at edge (like real installation)
                if x_end > canvas_size:
                    cut_width = canvas_size - x - grout_px
                    if cut_width > tile_px * 0.1:  # Place even small cuts for coverage
                        current_tile = current_tile[:, :cut_width]
                        tiles_cut += 1
                        x_end = canvas_size
                    else:
                        continue
                
                if y_end > canvas_size:
                    cut_height = canvas_size - y - grout_px
                    if cut_height > tile_px * 0.1:  # Place even small cuts for coverage
                        current_tile = current_tile[:cut_height, :]
                        tiles_cut += 1
                        y_end = canvas_size
                    else:
                        continue
                
                # PLACE TILE on canvas
                try:
                    tile_h, tile_w = current_tile.shape[:2]
                    canvas[y:y+tile_h, x:x+tile_w] = current_tile
                    tiles_placed += 1
                except:
                    pass
        
        print(f"   ✅ Installed: {tiles_placed} tiles on oversized canvas ({tiles_cut} edge tiles cut)")
        print(f"   ✅ Full coverage ensured - all user-marked areas will be tiled")
        
        return canvas
    
    def apply_perspective_to_installed_floor(self, flat_floor: np.ndarray) -> np.ndarray:
        """
        Apply perspective transformation to the flat installed floor
        
        Args:
            flat_floor: Flat tiled floor (top-down view)
            
        Returns:
            Floor with perspective applied
        """
        print("\n📐 Applying perspective transformation...")
        print("   Using oversized canvas to ensure complete floor coverage")
        
        # Define source points (flat floor corners) - CENTER of oversized canvas
        canvas_size = flat_floor.shape[0]
        margin = int(canvas_size * 0.15)  # 15% margin on each side
        
        src_points = np.float32([
            [margin, margin],                                    # top-left
            [canvas_size - margin - 1, margin],                  # top-right
            [canvas_size - margin - 1, canvas_size - margin - 1], # bottom-right
            [margin, canvas_size - margin - 1]                   # bottom-left
        ])
        
        # Compute perspective transform
        M = cv2.getPerspectiveTransform(src_points, self.quad_points)
        
        # Apply perspective warp with highest quality
        warped = cv2.warpPerspective(
            flat_floor,
            M,
            (self.image_width, self.image_height),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.grout_color
        )
        
        print("   ✅ Perspective applied - entire marked area covered")
        return warped
    
    def clip_to_floor_mask(self, warped_floor: np.ndarray) -> np.ndarray:
        """
        Clip warped floor to actual floor mask (ensures only user-marked area is tiled)
        
        Args:
            warped_floor: Floor with perspective
            
        Returns:
            Clipped floor (BGRA with alpha)
        """
        print("   ✂️ Clipping to user-marked floor area...")
        
        # Add alpha channel
        warped_bgra = cv2.cvtColor(warped_floor, cv2.COLOR_BGR2BGRA)
        warped_bgra[:, :, 3] = self.mask
        
        # Count coverage
        coverage = np.count_nonzero(self.mask > 0)
        print(f"   ✅ Clipped to {coverage} pixels of user-marked area")
        return warped_bgra
    
    def install_complete(self, resolution: int = 4000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete professional tile installation workflow
        
        Args:
            resolution: Canvas resolution for tile installation
            
        Returns:
            Tuple of (warped_tiles, warped_tiles_clipped)
        """
        print("\n" + "="*70)
        print("🏗️  PROFESSIONAL TILE INSTALLATION WORKFLOW")
        print("="*70)
        
        # Step 1: Install tiles on flat floor (like real installation)
        flat_tiled_floor = self.install_tiles_on_flat_floor(resolution)
        
        # Step 2: Apply perspective to match room view
        warped_tiles = self.apply_perspective_to_installed_floor(flat_tiled_floor)
        
        # Step 3: Clip to actual floor area
        warped_tiles_clipped = self.clip_to_floor_mask(warped_tiles)
        
        print("\n✅ PROFESSIONAL INSTALLATION COMPLETE!")
        print("="*70)
        
        return warped_tiles, warped_tiles_clipped
