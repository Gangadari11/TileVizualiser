"""
Realistic Blending Module
==========================
Blends projected tiles with the original image using color matching,
brightness adjustment, and alpha blending for realistic results.

Key Features:
- Color tone matching
- Brightness/contrast adjustment
- Alpha blending with smooth transitions
- Shadow preservation
- Lighting adaptation
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class RealisticBlending:
    """
    Tools for realistic blending of tiles with original image
    """
    
    def __init__(self, original_image: np.ndarray, mask: np.ndarray):
        """
        Initialize realistic blending
        
        Args:
            original_image: Original room image (BGR)
            mask: Floor mask (binary, 0 or 255)
        """
        self.original_image = original_image.copy()
        self.mask = mask
        self.height, self.width = original_image.shape[:2]
        
        # Compute statistics of original floor region
        self._compute_floor_statistics()
        
        print("✅ Realistic blending initialized")
    
    def _compute_floor_statistics(self):
        """Compute color and brightness statistics of original floor"""
        # Extract floor region
        floor_region = cv2.bitwise_and(
            self.original_image,
            self.original_image,
            mask=self.mask
        )
        
        # Get non-zero pixels
        floor_pixels = floor_region[self.mask > 0]
        
        if len(floor_pixels) > 0:
            self.floor_mean_color = np.mean(floor_pixels, axis=0)
            self.floor_std_color = np.std(floor_pixels, axis=0)
            self.floor_mean_brightness = np.mean(cv2.cvtColor(
                floor_region, cv2.COLOR_BGR2GRAY
            )[self.mask > 0])
        else:
            self.floor_mean_color = np.array([128, 128, 128])
            self.floor_std_color = np.array([30, 30, 30])
            self.floor_mean_brightness = 128
        
        print(f"   📊 Floor mean color: {self.floor_mean_color}")
        print(f"   📊 Floor mean brightness: {self.floor_mean_brightness:.1f}")
    
    def match_brightness(self, 
                        tile_image: np.ndarray,
                        target_brightness: Optional[float] = None,
                        strength: float = 0.5) -> np.ndarray:
        """
        Match tile image brightness to original floor (with adjustable strength)
        
        Args:
            tile_image: Input tile image (BGR)
            target_brightness: Target brightness (if None, use floor brightness)
            strength: Strength of matching (0.0-1.0), default 0.5 to preserve tile colors
            
        Returns:
            Brightness-adjusted tile image
        """
        print(f"💡 Matching brightness (strength: {strength:.1%})...")
        
        if target_brightness is None:
            target_brightness = self.floor_mean_brightness
        
        # Convert to grayscale to compute current brightness
        tile_gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
        tile_brightness = np.mean(tile_gray[tile_gray > 0])
        
        if tile_brightness == 0:
            return tile_image
        
        # Compute brightness adjustment factor with strength control
        brightness_factor = 1.0 + (target_brightness / tile_brightness - 1.0) * strength
        
        # Apply adjustment
        adjusted = tile_image.astype(np.float32) * brightness_factor
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        print(f"   ✓ Brightness adjusted: {tile_brightness:.1f} → {tile_brightness * brightness_factor:.1f}")
        return adjusted
    
    def match_color_tone(self, 
                        tile_image: np.ndarray,
                        method: str = 'none',
                        strength: float = 0.3) -> np.ndarray:
        """
        Match tile color tone to original floor (OPTIONAL - use 'none' to preserve original colors)
        
        Args:
            tile_image: Input tile image (BGR)
            method: 'histogram', 'mean_std', or 'none' (default: 'none' to preserve colors)
            strength: Strength of color matching (0.0-1.0)
            
        Returns:
            Color-matched tile image
        """
        if method == 'none':
            print("🎨 Preserving original tile colors (no color matching)")
            return tile_image
            
        print(f"🎨 Matching color tone (method: {method}, strength: {strength:.1%})...")
        
        if method == 'histogram':
            matched = self._match_color_histogram(tile_image)
        elif method == 'mean_std':
            matched = self._match_color_mean_std(tile_image)
        else:
            print(f"   ⚠️ Unknown method: {method}")
            return tile_image
        
        # Blend matched with original based on strength
        result = cv2.addWeighted(tile_image, 1.0 - strength, matched, strength, 0)
        return result
    
    def _match_color_histogram(self, tile_image: np.ndarray) -> np.ndarray:
        """Match color using histogram matching"""
        # Extract floor region from original
        floor_region = cv2.bitwise_and(
            self.original_image,
            self.original_image,
            mask=self.mask
        )
        
        # Match histograms for each channel
        matched = tile_image.copy()
        
        for c in range(3):
            # Compute histograms
            source_hist, _ = np.histogram(
                tile_image[:, :, c].flatten(),
                bins=256,
                range=(0, 256)
            )
            template_hist, _ = np.histogram(
                floor_region[:, :, c][self.mask > 0],
                bins=256,
                range=(0, 256)
            )
            
            # Compute cumulative histograms
            source_cdf = source_hist.cumsum()
            source_cdf = source_cdf / source_cdf[-1]
            
            template_cdf = template_hist.cumsum()
            template_cdf = template_cdf / template_cdf[-1]
            
            # Create mapping
            mapping = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                # Find closest value in template CDF
                diff = np.abs(template_cdf - source_cdf[i])
                mapping[i] = np.argmin(diff)
            
            # Apply mapping
            matched[:, :, c] = mapping[tile_image[:, :, c]]
        
        print("   ✓ Histogram matching complete")
        return matched
    
    def _match_color_mean_std(self, tile_image: np.ndarray) -> np.ndarray:
        """Match color using mean and standard deviation"""
        # Compute tile statistics
        tile_pixels = tile_image[tile_image.sum(axis=2) > 0]
        
        if len(tile_pixels) == 0:
            return tile_image
        
        tile_mean = np.mean(tile_pixels, axis=0)
        tile_std = np.std(tile_pixels, axis=0)
        
        # Avoid division by zero
        tile_std = np.maximum(tile_std, 1.0)
        
        # Normalize and scale
        matched = tile_image.astype(np.float32)
        
        for c in range(3):
            matched[:, :, c] = (matched[:, :, c] - tile_mean[c]) / tile_std[c]
            matched[:, :, c] = matched[:, :, c] * self.floor_std_color[c] + self.floor_mean_color[c]
        
        matched = np.clip(matched, 0, 255).astype(np.uint8)
        
        print("   ✓ Mean/std matching complete")
        return matched
    
    def extract_lighting_map(self) -> np.ndarray:
        """
        Extract lighting map from original floor region
        
        Returns:
            Lighting intensity map (0-1 float)
        """
        print("💡 Extracting lighting map...")
        
        # Convert to grayscale
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to get smooth lighting
        lighting = cv2.GaussianBlur(gray, (51, 51), 0)
        
        # Normalize to 0-1
        lighting = lighting.astype(np.float32) / 255.0
        
        print("   ✓ Lighting map extracted")
        return lighting
    
    def apply_lighting(self, 
                      tile_image: np.ndarray,
                      lighting_map: np.ndarray,
                      strength: float = 0.3) -> np.ndarray:
        """
        Apply original lighting to tile image (reduced default strength)
        
        Args:
            tile_image: Input tile image (BGR)
            lighting_map: Lighting intensity map (0-1)
            strength: Strength of lighting effect (0-1), reduced to 0.3
            
        Returns:
            Tile image with lighting applied
        """
        print(f"💡 Applying lighting (strength: {strength:.1%})...")
        
        # Convert to float
        tile_float = tile_image.astype(np.float32) / 255.0
        
        # Apply lighting to each channel
        lighting_3d = np.stack([lighting_map] * 3, axis=2)
        
        # Blend with lighting
        lit_tile = tile_float * (1.0 - strength) + tile_float * lighting_3d * strength
        
        # Convert back to uint8
        lit_tile = np.clip(lit_tile * 255, 0, 255).astype(np.uint8)
        
        print("   ✓ Lighting applied")
        return lit_tile
    
    def alpha_blend(self,
                   tile_image: np.ndarray,
                   alpha: float = 0.8,
                   feather_size: int = 10) -> np.ndarray:
        """
        Blend tiles with original image using alpha blending
        
        Args:
            tile_image: Tile image (BGR or BGRA)
            alpha: Global alpha value (0-1)
            feather_size: Size of edge feathering in pixels
            
        Returns:
            Blended result (BGR)
        """
        print(f"🎨 Alpha blending (alpha: {alpha:.2f})...")
        
        # Create alpha mask
        if tile_image.shape[2] == 4:
            # Use existing alpha channel
            alpha_mask = tile_image[:, :, 3].astype(np.float32) / 255.0
            tile_bgr = tile_image[:, :, :3]
        else:
            # Create alpha from mask
            alpha_mask = self.mask.astype(np.float32) / 255.0
            tile_bgr = tile_image
        
        # Apply feathering to alpha mask
        if feather_size > 0:
            alpha_mask = self._feather_mask(alpha_mask, feather_size)
        
        # Apply global alpha
        alpha_mask = alpha_mask * alpha
        
        # Expand to 3 channels
        alpha_3d = np.stack([alpha_mask] * 3, axis=2)
        
        # Blend
        blended = (tile_bgr * alpha_3d + 
                  self.original_image * (1.0 - alpha_3d))
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        print("   ✓ Blending complete")
        return blended
    
    def _feather_mask(self, mask: np.ndarray, feather_size: int) -> np.ndarray:
        """
        Apply feathering (soft edges) to mask
        
        Args:
            mask: Input mask (0-1 float)
            feather_size: Feathering size in pixels
            
        Returns:
            Feathered mask
        """
        # Compute distance transform
        mask_binary = (mask > 0.5).astype(np.uint8)
        dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
        
        # Normalize by feather size
        feathered = np.clip(dist_transform / feather_size, 0, 1)
        
        return feathered.astype(np.float32)
    
    def blend_complete(self,
                      tile_image: np.ndarray,
                      match_brightness: bool = True,
                      match_color: bool = True,
                      apply_lighting: bool = True,
                      alpha: float = 0.85,
                      feather_size: int = 10) -> np.ndarray:
        """
        Complete blending pipeline
        
        Args:
            tile_image: Input tile image (BGR or BGRA)
            match_brightness: Apply brightness matching
            match_color: Apply color matching
            apply_lighting: Apply lighting map
            alpha: Global alpha value
            feather_size: Edge feathering size
            
        Returns:
            Final blended result
        """
        print("🎬 Starting complete blending pipeline...")
        
        # Extract BGR if BGRA
        if tile_image.shape[2] == 4:
            tile_bgr = tile_image[:, :, :3].copy()
        else:
            tile_bgr = tile_image.copy()
        
        # Step 1: Match brightness
        if match_brightness:
            tile_bgr = self.match_brightness(tile_bgr)
        
        # Step 2: Match color tone
        if match_color:
            tile_bgr = self.match_color_tone(tile_bgr, method='mean_std')
        
        # Step 3: Apply lighting
        if apply_lighting:
            lighting_map = self.extract_lighting_map()
            tile_bgr = self.apply_lighting(tile_bgr, lighting_map, strength=0.5)
        
        # Step 4: Alpha blend
        result = self.alpha_blend(tile_bgr, alpha=alpha, feather_size=feather_size)
        
        print("✅ Complete blending finished!")
        return result
