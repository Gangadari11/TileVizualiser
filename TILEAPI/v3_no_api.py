"""
Floor Tile Replacement - NO API NEEDED!
Simple tile blending using basic image processing
Works immediately with no setup!
"""

from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np

# ================= CONFIG =================
ROOM_IMAGE_PATH = "room.jpg"
TILE_IMAGE_PATH = "tile.jpg"
OUTPUT_IMAGE_PATH = "room_with_tile_simple.png"
MASK_PATH = "mask_simple.png"

# Adjust floor area (0.0 to 1.0, where floor starts from top)
FLOOR_START_RATIO = 0.4  # Floor starts 40% down from top

# ================= HELPER FUNCTIONS =================

def create_floor_mask(image_path, floor_start_ratio=0.4):
    """Create a mask for the floor area"""
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    
    # Create black image for floor area
    mask = Image.new('L', (width, height), 255)  # White = keep original
    draw = ImageDraw.Draw(mask)
    
    # Draw floor area in black (will be replaced)
    floor_start = int(height * floor_start_ratio)
    draw.rectangle([(0, floor_start), (width, height)], fill=0)
    
    # Smooth the edge for better blending
    mask = mask.filter(ImageFilter.GaussianBlur(radius=5))
    
    mask.save(MASK_PATH)
    print(f"✅ Mask created: {MASK_PATH}")
    return mask


def apply_perspective_to_tile(tile_img, target_width, target_height):
    """
    Apply perspective transform to make tiles look realistic
    """
    # Create a larger tiled pattern first
    tile_size = 200
    tile_resized = tile_img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
    
    # Calculate how many tiles needed
    tiles_wide = (target_width // tile_size) + 2
    tiles_high = (target_height // tile_size) + 2
    
    # Create tiled pattern
    tiled = Image.new('RGB', (tiles_wide * tile_size, tiles_high * tile_size))
    for x in range(tiles_wide):
        for y in range(tiles_high):
            tiled.paste(tile_resized, (x * tile_size, y * tile_size))
    
    # Resize to target dimensions with perspective effect
    # Make it slightly taller to simulate perspective
    tiled = tiled.resize((target_width, int(target_height * 1.2)), Image.Resampling.LANCZOS)
    
    # Crop to actual size (this creates perspective effect)
    tiled = tiled.crop((0, int(target_height * 0.2), target_width, int(target_height * 1.2)))
    
    return tiled


def add_lighting_to_floor(floor_img, original_room):
    """
    Match the lighting from the original room
    """
    # Convert to arrays
    floor_array = np.array(floor_img).astype(float)
    room_array = np.array(original_room).astype(float)
    
    # Calculate average brightness of original floor area
    avg_brightness = room_array.mean()
    floor_brightness = floor_array.mean()
    
    # Adjust floor brightness to match room
    brightness_factor = avg_brightness / floor_brightness
    floor_array = floor_array * brightness_factor
    floor_array = np.clip(floor_array, 0, 255).astype(np.uint8)
    
    result = Image.fromarray(floor_array)
    
    # Add subtle gradient (darker at back, lighter at front)
    gradient = Image.new('L', floor_img.size, 0)
    draw = ImageDraw.Draw(gradient)
    
    for y in range(floor_img.height):
        # Darker at top (back), lighter at bottom (front)
        intensity = int(180 + (75 * y / floor_img.height))
        draw.line([(0, y), (floor_img.width, y)], fill=intensity)
    
    # Apply gradient
    result = Image.composite(result, result.point(lambda p: p * 0.7), gradient)
    
    return result


def blend_with_mask(room_img, floor_img, mask_img):
    """
    Blend floor tiles into room using mask
    """
    # Ensure all images are same size
    if floor_img.size != room_img.size:
        floor_img = floor_img.resize(room_img.size, Image.Resampling.LANCZOS)
    if mask_img.size != room_img.size:
        mask_img = mask_img.resize(room_img.size, Image.Resampling.LANCZOS)
    
    # Convert to arrays
    room_array = np.array(room_img).astype(float)
    floor_array = np.array(floor_img).astype(float)
    mask_array = np.array(mask_img).astype(float) / 255.0
    
    # Expand mask dimensions to match RGB
    if len(mask_array.shape) == 2:
        mask_array = mask_array[:, :, np.newaxis]
    
    # Blend: result = room * mask + floor * (1 - mask)
    result_array = (room_array * mask_array + floor_array * (1 - mask_array))
    result_array = np.clip(result_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(result_array)


def enhance_result(img):
    """
    Apply final enhancements for better quality
    """
    # Slight sharpening
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=2))
    
    # Slight contrast boost
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.1)
    
    # Slight color boost
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.05)
    
    return img


# ================= MAIN FUNCTION =================

def replace_floor_with_tile():
    """
    Main function to replace floor with tile pattern
    """
    print("🚀 Starting Floor Tile Replacement...")
    print("💡 No API needed - 100% local processing!\n")
    
    # Load images
    print("📸 Loading images...")
    room_img = Image.open(ROOM_IMAGE_PATH).convert('RGB')
    tile_img = Image.open(TILE_IMAGE_PATH).convert('RGB')
    original_size = room_img.size
    
    print(f"   Room: {room_img.size}")
    print(f"   Tile: {tile_img.size}")
    
    # Create mask
    print("\n🎭 Creating floor mask...")
    mask_img = create_floor_mask(ROOM_IMAGE_PATH, floor_start_ratio=FLOOR_START_RATIO)
    
    # Calculate floor dimensions
    mask_array = np.array(mask_img)
    floor_start_y = int(room_img.height * FLOOR_START_RATIO)
    floor_height = room_img.height - floor_start_y
    floor_width = room_img.width
    
    print(f"\n🎨 Creating tiled floor pattern...")
    print(f"   Floor area: {floor_width}x{floor_height}px")
    
    # Create tiled floor with perspective
    floor_pattern = apply_perspective_to_tile(tile_img, floor_width, floor_height)
    
    # Match lighting to room
    print("💡 Matching lighting...")
    floor_pattern = add_lighting_to_floor(floor_pattern, room_img.crop((0, floor_start_y, floor_width, room_img.height)))
    
    # Create full-size floor image
    full_floor = Image.new('RGB', room_img.size)
    full_floor.paste(floor_pattern, (0, floor_start_y))
    
    # Blend using mask
    print("🖌️  Blending tiles with room...")
    result = blend_with_mask(room_img, full_floor, mask_img)
    
    # Enhance final result
    print("✨ Enhancing image...")
    result = enhance_result(result)
    
    # Save
    result.save(OUTPUT_IMAGE_PATH)
    print(f"\n✅ SUCCESS! Image saved to: {OUTPUT_IMAGE_PATH}")
    print(f"📐 Output size: {result.size}")
    
    return result


# ================= RUN =================

if __name__ == "__main__":
    print("="*60)
    print("  🏠 FLOOR TILE REPLACEMENT - NO API VERSION")
    print("  ⚡ Works instantly - No setup required!")
    print("="*60 + "\n")
    
    try:
        result = replace_floor_with_tile()
        
        print("\n" + "="*60)
        print("✨ DONE! Check your output image.")
        print("="*60)
        print("\n📚 TIPS:")
        print(f"  - Change FLOOR_START_RATIO (currently {FLOOR_START_RATIO})")
        print("    to adjust where the floor begins")
        print("  - Use high-quality tile images for best results")
        print("  - For AI-powered results, try v3_free.py with DeepAI")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Could not find image file")
        print(f"   Make sure '{ROOM_IMAGE_PATH}' and '{TILE_IMAGE_PATH}' exist")
        print(f"   Details: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
