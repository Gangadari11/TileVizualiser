import cv2
import numpy as np
from pathlib import Path

class InteractiveSmartFloorDetector:
    def __init__(self):
        self.image = None
        self.floor_mask = None
        self.display_image = None
        self.seed_points = []
        self.exclude_points = []
        self.original_size = None
        self.processing_scale = 1.0
        
    def smart_floor_detection(self, image_path):
        """
        Smart floor detection with interactive correction
        """
        original_image = cv2.imread(image_path)
        
        if original_image is None:
            print(f"❌ Error: Could not load image '{image_path}'")
            return None
        
        print("\n" + "="*70)
        print("🎯 SMART INTERACTIVE FLOOR DETECTION")
        print("="*70)
        
        # Store original size and downscale for processing
        self.original_size = (original_image.shape[1], original_image.shape[0])
        height, width = original_image.shape[:2]
        
        # Smart resizing: only process at max 1500px on longest side
        max_dim = max(height, width)
        if max_dim > 1500:
            self.processing_scale = 1500 / max_dim
            new_width = int(width * self.processing_scale)
            new_height = int(height * self.processing_scale)
            self.image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"   📏 Resized for processing: {width}x{height} → {new_width}x{new_height} ({self.processing_scale:.2f}x)")
        else:
            self.image = original_image
            self.processing_scale = 1.0
        
        height, width = self.image.shape[:2]
        
        # =========================================================
        # STEP 1: AUTOMATIC INITIAL DETECTION
        # =========================================================
        print("\n📍 STEP 1: Automatic Initial Detection")
        print("-" * 70)
        
        # Convert to LAB color space
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Sample bottom region for floor color
        bottom_region = self.image[int(height * 0.85):, int(width * 0.2):int(width * 0.8)]
        floor_color_bgr = np.median(bottom_region.reshape(-1, 3), axis=0)
        floor_color_lab = cv2.cvtColor(np.uint8([[floor_color_bgr]]), cv2.COLOR_BGR2LAB)[0][0]
        
        print(f"   ✓ Sampled floor color (LAB): {floor_color_lab}")
        
        # Calculate color distance
        diff = lab.astype(float) - floor_color_lab.astype(float)
        color_distance = np.sqrt(
            (diff[:, :, 0] * 0.6) ** 2 + 
            diff[:, :, 1] ** 2 + 
            diff[:, :, 2] ** 2
        )
        
        # Adaptive threshold
        threshold = np.percentile(color_distance[int(height * 0.7):], 50)
        color_mask = (color_distance < threshold * 2.5).astype(np.uint8) * 255
        
        # Find horizon line
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=width//3, maxLineGap=50)
        
        horizon_y = int(height * 0.5)
        if lines is not None:
            horizontal_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 15 and height * 0.2 < y1 < height * 0.7:
                    horizontal_lines.append((y1 + y2) // 2)
            if horizontal_lines:
                horizon_y = int(np.median(horizontal_lines))
        
        print(f"   ✓ Horizon detected at y={horizon_y}")
        
        # Create position mask (floor is below horizon)
        position_mask = np.zeros((height, width), dtype=np.uint8)
        position_mask[horizon_y:, :] = 255
        
        # Combine masks
        combined = cv2.bitwise_and(color_mask, position_mask)
        
        # Morphological cleanup - optimized (reduced iterations)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Keep largest component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            combined = (labels == largest_label).astype(np.uint8) * 255
        
        self.floor_mask = combined
        
        print(f"   ✓ Initial detection complete")
        
        # =========================================================
        # STEP 2: INTERACTIVE CORRECTION
        # =========================================================
        print("\n✏️ STEP 2: Interactive Correction (Optional)")
        print("-" * 70)
        print("\n   INSTRUCTIONS:")
        print("   • LEFT CLICK: Mark floor regions (add to floor)")
        print("   • RIGHT CLICK: Mark non-floor regions (remove from floor)")
        print("   • Press 'R': Reset to automatic detection")
        print("   • Press 'S': Apply region growing from seed points")
        print("   • Press ENTER: Accept and continue")
        print("   • Press ESC: Cancel")
        print("-" * 70)
        
        accepted = self.interactive_correction()
        
        if not accepted:
            return None
        
        # =========================================================
        # STEP 3: INTELLIGENT REFINEMENT
        # =========================================================
        print("\n🔧 STEP 3: Intelligent Refinement")
        print("-" * 70)
        
        # Apply region growing if seed points were provided
        if len(self.seed_points) > 0:
            print(f"   ✓ Applying region growing from {len(self.seed_points)} seed points...")
            self.floor_mask = self.region_growing_refinement()
        
        # GrabCut refinement
        print("   ✓ Applying GrabCut refinement...")
        self.floor_mask = self.grabcut_refinement()
        
        # Edge smoothing
        print("   ✓ Smoothing edges...")
        for _ in range(3):
            mask_float = self.floor_mask.astype(np.float32) / 255.0
            smoothed = cv2.bilateralFilter(mask_float, 9, 75, 75)
            self.floor_mask = (smoothed * 255).astype(np.uint8)
        
        _, self.floor_mask = cv2.threshold(self.floor_mask, 127, 255, cv2.THRESH_BINARY)
        
        print("   ✓ Refinement complete")
        
        print("\n" + "="*70)
        print("✅ SMART FLOOR DETECTION COMPLETE")
        print("="*70 + "\n")
        
        return self.floor_mask
    
    def interactive_correction(self):
        """Interactive manual correction interface"""
        self.display_image = self.image.copy()
        initial_mask = self.floor_mask.copy()
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Add to floor
                self.seed_points.append((x, y))
                cv2.circle(self.display_image, (x, y), 8, (0, 255, 0), -1)
                cv2.circle(self.floor_mask, (x, y), 25, 255, -1)
                
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Remove from floor
                self.exclude_points.append((x, y))
                cv2.circle(self.display_image, (x, y), 8, (0, 0, 255), -1)
                cv2.circle(self.floor_mask, (x, y), 25, 0, -1)
        
        cv2.namedWindow("Interactive Floor Correction")
        cv2.setMouseCallback("Interactive Floor Correction", mouse_callback)
        
        while True:
            # Create overlay
            overlay = self.display_image.copy()
            mask_colored = cv2.cvtColor(self.floor_mask, cv2.COLOR_GRAY2BGR)
            mask_colored[:, :, 0] = 0  # Remove blue
            mask_colored[:, :, 2] = 0  # Remove red
            
            alpha = 0.4
            overlay = cv2.addWeighted(overlay, 1, mask_colored, alpha, 0)
            
            # Draw seed points
            for pt in self.seed_points:
                cv2.circle(overlay, pt, 8, (0, 255, 0), -1)
                cv2.circle(overlay, pt, 10, (255, 255, 255), 2)
            
            for pt in self.exclude_points:
                cv2.circle(overlay, pt, 8, (0, 0, 255), -1)
                cv2.circle(overlay, pt, 10, (255, 255, 255), 2)
            
            # Info panel
            cv2.rectangle(overlay, (10, 10), (900, 180), (0, 0, 0), -1)
            cv2.rectangle(overlay, (10, 10), (900, 180), (0, 255, 0), 2)
            
            cv2.putText(overlay, "INTERACTIVE FLOOR CORRECTION", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(overlay, "LEFT CLICK: Add floor | RIGHT CLICK: Remove floor", (20, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(overlay, "Press 'S': Smart region growing | Press 'R': Reset", (20, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(overlay, "Press ENTER: Accept | Press ESC: Cancel", (20, 135), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.putText(overlay, f"Seeds: {len(self.seed_points)} | Excludes: {len(self.exclude_points)}", 
                       (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            cv2.imshow("Interactive Floor Correction", overlay)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # ENTER
                cv2.destroyAllWindows()
                return True
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return False
            elif key == ord('r') or key == ord('R'):
                # Reset
                self.floor_mask = initial_mask.copy()
                self.display_image = self.image.copy()
                self.seed_points = []
                self.exclude_points = []
                print("   ↺ Reset to initial detection")
            elif key == ord('s') or key == ord('S'):
                # Apply smart region growing
                if len(self.seed_points) > 0:
                    print(f"   ⚡ Applying smart region growing...")
                    self.floor_mask = self.region_growing_refinement()
                    self.display_image = self.image.copy()
                    print("   ✓ Region growing applied")
    
    def region_growing_refinement(self):
        """Smart region growing from seed points"""
        if len(self.seed_points) == 0:
            return self.floor_mask
        
        height, width = self.image.shape[:2]
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        
        # Initialize output mask
        output_mask = np.zeros((height, width), dtype=np.uint8)
        visited = np.zeros((height, width), dtype=bool)
        
        # Get seed colors
        seed_colors = []
        for x, y in self.seed_points:
            if 0 <= y < height and 0 <= x < width:
                seed_colors.append(lab[y, x])
        
        if len(seed_colors) == 0:
            return self.floor_mask
        
        seed_color_mean = np.mean(seed_colors, axis=0)
        
        # Calculate adaptive threshold
        distances = [np.linalg.norm(c - seed_color_mean) for c in seed_colors]
        threshold = max(np.mean(distances) * 3, 30)
        
        # Region growing using flood fill
        for seed_x, seed_y in self.seed_points:
            if not (0 <= seed_y < height and 0 <= seed_x < width):
                continue
            
            stack = [(seed_x, seed_y)]
            
            while stack:
                x, y = stack.pop()
                
                if x < 0 or x >= width or y < 0 or y >= height:
                    continue
                
                if visited[y, x]:
                    continue
                
                visited[y, x] = True
                
                # Check color similarity
                pixel_color = lab[y, x]
                distance = np.linalg.norm(pixel_color - seed_color_mean)
                
                if distance < threshold:
                    output_mask[y, x] = 255
                    
                    # Add neighbors
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), 
                                  (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        stack.append((x + dx, y + dy))
        
        # Apply exclusion points
        for x, y in self.exclude_points:
            cv2.circle(output_mask, (x, y), 30, 0, -1)
        
        # Morphological cleanup - optimized
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        output_mask = cv2.morphologyEx(output_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        output_mask = cv2.morphologyEx(output_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return output_mask
    
    def grabcut_refinement(self):
        """GrabCut refinement - OPTIMIZED: Reduced iterations for speed"""
        if self.floor_mask.sum() == 0:
            return self.floor_mask
        
        try:
            height, width = self.image.shape[:2]
            gc_mask = np.zeros(self.floor_mask.shape, np.uint8)
            
            # Create trimap - optimized kernel sizes
            kernel_fg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
            
            definite_fg = cv2.erode(self.floor_mask, kernel_fg, iterations=1)
            gc_mask[definite_fg == 255] = cv2.GC_FGD
            
            definite_bg = cv2.dilate(self.floor_mask, kernel_bg, iterations=1)
            gc_mask[definite_bg == 0] = cv2.GC_BGD
            
            probable = cv2.dilate(self.floor_mask, kernel_bg, iterations=1) - \
                      cv2.erode(self.floor_mask, kernel_fg, iterations=1)
            gc_mask[probable == 255] = cv2.GC_PR_FGD
            
            contours, _ = cv2.findContours(self.floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                rect = (max(0, x-20), max(0, y-20), min(width-x, w+40), min(height-y, h+40))
                
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                
                # Reduced from 8 to 3 iterations for speed
                cv2.grabCut(self.image, gc_mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
                
                result_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
                return result_mask
        except:
            pass
        
        return self.floor_mask
    
    def order_points(self, pts):
        """Order corner points: TL, TR, BR, BL"""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    
    def apply_tiles(self, tile_path, tile_size_px=80, output_path="room_with_tiles_interactive.png"):
        """Apply tiles with ultra-realistic blending - OPTIMIZED"""
        if self.floor_mask is None:
            print("❌ No floor mask available")
            return None
        
        tile = cv2.imread(tile_path)
        if tile is None:
            print(f"❌ Could not load tile: {tile_path}")
            return None
        
        print(f"\n🎨 APPLYING TILES (OPTIMIZED)")
        print("="*70)
        
        # Scale mask back to original size if needed
        if self.processing_scale != 1.0:
            print(f"   📏 Scaling mask to original resolution...")
            self.floor_mask = cv2.resize(self.floor_mask, self.original_size, interpolation=cv2.INTER_LINEAR)
            # Also scale image
            self.image = cv2.resize(self.image, self.original_size, interpolation=cv2.INTER_CUBIC)
        
        contours, _ = cv2.findContours(self.floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("❌ No floor region found")
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get corners
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(largest_contour)
            corners = cv2.boxPoints(rect).astype(np.float32)
        
        corners = self.order_points(corners)
        
        # Calculate dimensions
        width1 = np.linalg.norm(corners[1] - corners[0])
        width2 = np.linalg.norm(corners[2] - corners[3])
        height1 = np.linalg.norm(corners[3] - corners[0])
        height2 = np.linalg.norm(corners[2] - corners[1])
        
        max_width = int(max(width1, width2))
        max_height = int(max(height1, height2))
        
        print(f"✓ Floor: {max_width} x {max_height} px")
        print(f"✓ Tile: {tile_size_px} x {tile_size_px} px")
        
        # Create tiled pattern with HIGH QUALITY resize
        tile_resized = cv2.resize(tile, (tile_size_px, tile_size_px), interpolation=cv2.INTER_CUBIC)
        tiles_x = (max_width // tile_size_px) + 2
        tiles_y = (max_height // tile_size_px) + 2
        
        tiled_pattern = np.tile(tile_resized, (tiles_y, tiles_x, 1))
        tiled_pattern = tiled_pattern[:max_height, :max_width]
        
        # Add grout
        grout_color = (180, 180, 180)
        grout_width = max(1, tile_size_px // 45)
        
        for i in range(0, max_height, tile_size_px):
            cv2.line(tiled_pattern, (0, i), (max_width, i), grout_color, grout_width)
        for j in range(0, max_width, tile_size_px):
            cv2.line(tiled_pattern, (j, 0), (j, max_height), grout_color, grout_width)
        
        # Perspective transform with HIGH QUALITY interpolation
        dst_corners = np.array([[0, 0], [max_width - 1, 0], 
                               [max_width - 1, max_height - 1], [0, max_height - 1]], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(dst_corners, corners)
        warped_tiles = cv2.warpPerspective(tiled_pattern, M, 
                                          (self.image.shape[1], self.image.shape[0]),
                                          flags=cv2.INTER_CUBIC,
                                          borderMode=cv2.BORDER_CONSTANT)
        
        # Lighting match - REDUCED AGGRESSION to preserve original tile colors
        print("✓ Subtle lighting adjustment (preserving tile colors)...")
        floor_region = cv2.bitwise_and(self.image, self.image, mask=self.floor_mask)
        floor_lab = cv2.cvtColor(floor_region, cv2.COLOR_BGR2LAB)
        tiles_lab = cv2.cvtColor(warped_tiles, cv2.COLOR_BGR2LAB)
        
        floor_pixels = floor_lab[self.floor_mask > 0]
        
        if len(floor_pixels) > 0:
            floor_l_mean = np.mean(floor_pixels[:, 0])
            tiles_pixels = tiles_lab[self.floor_mask > 0]
            tiles_l_mean = np.mean(tiles_pixels[:, 0])
            
            # Reduce adjustment strength from 100% to 40% to preserve tile color
            adjustment = (floor_l_mean - tiles_l_mean) * 0.4
            tiles_lab[:, :, 0] = np.clip(tiles_lab[:, :, 0] + adjustment, 0, 255)
            warped_tiles = cv2.cvtColor(tiles_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        # Smooth blending - OPTIMIZED
        print("✓ Blending (optimized)...")
        mask_float = self.floor_mask.astype(float) / 255.0
        dist_from_edge = cv2.distanceTransform(self.floor_mask, cv2.DIST_L2, 5)
        dist_normalized = dist_from_edge / np.max(dist_from_edge) if np.max(dist_from_edge) > 0 else dist_from_edge
        
        # Reduced blur kernel for speed
        mask_blur = cv2.GaussianBlur(mask_float, (15, 15), 0)
        mask_blend = 0.7 * mask_float + 0.3 * mask_blur  # More of original mask
        mask_blend = mask_blend * (0.3 + 0.7 * dist_normalized)
        
        mask_3ch = np.stack([mask_blend] * 3, axis=2)
        
        result = (self.image * (1 - mask_3ch) + warped_tiles * mask_3ch).astype(np.uint8)
        
        # Post-processing - REMOVED bilateral filter for speed, use light sharpening only
        print("✓ Post-processing (fast)...")
        # SKIP bilateral filter - it's very slow and reduces detail
        
        # Very subtle sharpening to enhance details
        kernel_sharpen = np.array([[-0.2, -0.2, -0.2], [-0.2, 2.6, -0.2], [-0.2, -0.2, -0.2]]) / 2.0
        sharpened = cv2.filter2D(result, -1, kernel_sharpen)
        result = cv2.addWeighted(result, 0.92, sharpened, 0.08, 0)
        
        # Save as PNG for LOSSLESS quality
        cv2.imwrite(output_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        print(f"   💾 Saved as PNG (lossless quality)")
        
        print(f"\n✅ Saved: {output_path}")
        print("="*70 + "\n")
        
        return result
    
    def show_comparison(self, result):
        """Show before/after comparison"""
        h = 800
        scale = h / self.image.shape[0]
        w = int(self.image.shape[1] * scale)
        
        orig = cv2.resize(self.image, (w, h), interpolation=cv2.INTER_AREA)
        tiled = cv2.resize(result, (w, h), interpolation=cv2.INTER_AREA)
        
        cv2.rectangle(orig, (10, 10), (250, 80), (0, 0, 0), -1)
        cv2.putText(orig, "ORIGINAL", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        cv2.rectangle(tiled, (10, 10), (320, 80), (0, 0, 0), -1)
        cv2.putText(tiled, "WITH NEW TILES", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        comparison = np.hstack([orig, tiled])
        
        cv2.namedWindow("Result - Interactive Smart Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Result - Interactive Smart Detection", 1800, 900)
        cv2.imshow("Result - Interactive Smart Detection", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    print("\n" + "="*70)
    print("   🎯 INTERACTIVE SMART TILE VISUALIZER")
    print("   Manual Correction + AI Region Growing")
    print("="*70)
    
    room_image = "room.jpg"
    tile_image = "tile.jpg"
    
    if not Path(room_image).exists() or not Path(tile_image).exists():
        print("\n❌ Files not found!")
        return
    
    detector = InteractiveSmartFloorDetector()
    
    floor_mask = detector.smart_floor_detection(room_image)
    
    if floor_mask is None:
        print("\n❌ Detection cancelled!")
        return
    
    print("\n" + "="*70)
    print("APPLYING TILES")
    print("="*70)
    
    tile_size = int(input("Tile size in px [80]: ") or "80")
    
    result = detector.apply_tiles(tile_image, tile_size_px=tile_size)
    
    if result is None:
        return
    
    print("\n" + "="*70)
    print("FINAL RESULT")
    print("="*70)
    
    detector.show_comparison(result)
    
    print("\n✅ COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()