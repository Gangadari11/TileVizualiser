### APIs are only available for Pro members in good standing

"""
Floor Tile Replacement using DeepAI API (TRULY FREE!)
No billing required, just a free API key
"""

from PIL import Image, ImageDraw
import numpy as np
import requests
import io

# ================= CONFIG =================
# Get your FREE API key from: https://deepai.org/machine-learning-model/text2img
# Sign up is free and no billing required!
DEEPAI_API_KEY = "9ed3b77e-f0e5-4124-8ebe-d19f16108064"  # ⬅️ Get from https://deepai.org/

ROOM_IMAGE_PATH = "room.jpg"
TILE_IMAGE_PATH = "tile.jpg"
OUTPUT_IMAGE_PATH = "room_with_tile_deepai.png"
MASK_PATH = "mask_deepai.png"

# ================= HELPER FUNCTIONS =================

def analyze_tile_colors(tile_path):
    """Analyze tile image to create accurate prompt"""
    tile = Image.open(tile_path).convert('RGB')
    tile_array = np.array(tile)
    
    # Get average colors
    avg_color = tile_array.mean(axis=(0, 1)).astype(int)
    
    # Determine dominant color description
    r, g, b = avg_color
    
    if r > 180 and g > 180 and b > 180:
        color_desc = "white and light beige"
    elif r > 150 and g > 120 and b < 100:
        color_desc = "beige, cream, and brown"
    elif r > 100 and g > 100 and b > 100:
        color_desc = "neutral gray and white"
    elif r < 50 and g < 50 and b < 50:
        color_desc = "dark gray and black"
    else:
        color_desc = "multicolored"
    
    # Detect pattern
    tile_small = tile.resize((50, 50))
    pixels = list(tile_small.convert('RGB').getdata())
    unique_colors = len(set(pixels))
    
    if unique_colors < 10:
        pattern_desc = "solid uniform"
    elif unique_colors < 50:
        pattern_desc = "geometric checkerboard or grid pattern"
    else:
        pattern_desc = "complex textured pattern with veining"
    
    return color_desc, pattern_desc


def create_floor_mask(image_path, floor_start_ratio=0.4):
    """Create a mask for the floor area"""
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    
    # Create white mask (keep everything)
    mask = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(mask)
    
    # Draw black rectangle for floor area (will be replaced)
    floor_start = int(height * floor_start_ratio)
    draw.rectangle([(0, floor_start), (width, height)], fill=(0, 0, 0))
    
    mask.save(MASK_PATH)
    print(f"✅ Mask created and saved to {MASK_PATH}")
    return mask


def resize_image(image_path, size=(512, 512)):
    """Resize image to model-compatible dimensions"""
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize(size, Image.Resampling.LANCZOS)
    return img_resized


def generate_prompt_from_tile(tile_path):
    """Analyze tile and create detailed prompt"""
    color_desc, pattern_desc = analyze_tile_colors(tile_path)
    
    prompt = (
        "Visualize the room with this tile pattern on the floor. "
    )
    
    return prompt


def simple_inpainting_blend(room_img, mask_img, tile_img):
    """
    Simple approach: blend tile pattern into masked floor area
    This is a fallback when AI APIs don't work
    """
    from PIL import ImageFilter
    
    room_array = np.array(room_img)
    mask_array = np.array(mask_img.convert('L'))
    
    # Scale tile to match floor dimensions
    floor_height = np.sum(mask_array < 128, axis=0).max()
    floor_width = room_img.width
    
    # Create tiled pattern
    tile_resized = tile_img.resize((200, 200))
    tiles_h = (floor_height // 200) + 2
    tiles_w = (floor_width // 200) + 2
    
    tiled_floor = Image.new('RGB', (tiles_w * 200, tiles_h * 200))
    for i in range(tiles_w):
        for j in range(tiles_h):
            tiled_floor.paste(tile_resized, (i * 200, j * 200))
    
    tiled_floor = tiled_floor.resize((floor_width, room_img.height), Image.Resampling.LANCZOS)
    tiled_array = np.array(tiled_floor)
    
    # Blend using mask
    mask_blend = mask_array[:, :, None] / 255.0
    result_array = (room_array * mask_blend + tiled_array * (1 - mask_blend)).astype(np.uint8)
    
    result_img = Image.fromarray(result_array)
    result_img = result_img.filter(ImageFilter.SMOOTH)
    
    return result_img


# ================= MAIN INPAINTING =================

def inpaint_with_deepai(api_key=None):
    """
    Use DeepAI API for image editing
    Free API key from: https://deepai.org/
    """
    
    if not api_key or api_key == "your_deepai_key_here":
        print("❌ Please provide a DeepAI API key")
        print("Get one FREE at: https://deepai.org/")
        print("\n1. Sign up at https://deepai.org/")
        print("2. Go to https://deepai.org/dashboard")
        print("3. Copy your API key")
        print("4. Paste it in the DEEPAI_API_KEY variable")
        print("\n💡 100% FREE - No billing required!")
        
        # Fallback to simple blending
        print("\n🎨 Using fallback: Simple tile blending...")
        return fallback_simple_blend()
    
    print("🚀 Starting DeepAI Image Editing...")
    
    # Load and prepare images
    print("\n📸 Loading images...")
    room_img = resize_image(ROOM_IMAGE_PATH, size=(512, 512))
    room_img.save("temp_room.png")
    
    # Create mask
    print("🎭 Creating floor mask...")
    mask_img = create_floor_mask(ROOM_IMAGE_PATH, floor_start_ratio=0.4)
    mask_img = mask_img.resize((512, 512), Image.Resampling.LANCZOS)
    
    # Generate prompt
    print("\n🔍 Analyzing tile image...")
    prompt = generate_prompt_from_tile(TILE_IMAGE_PATH)
    print(f"📝 Generated prompt:\n{prompt}\n")
    
    # DeepAI Image Editor API
    print("🌐 Sending request to DeepAI API...")
    
    response = requests.post(
        "https://api.deepai.org/api/image-editor",
        files={
            'image': open('temp_room.png', 'rb'),
        },
        data={
            'text': prompt,
        },
        headers={'api-key': api_key}
    )
    
    if response.status_code == 200:
        result_data = response.json()
        output_url = result_data.get('output_url')
        
        if output_url:
            print("✅ Generation complete! Downloading...")
            img_response = requests.get(output_url)
            result_img = Image.open(io.BytesIO(img_response.content))
            result_img.save(OUTPUT_IMAGE_PATH)
            print(f"✅ Image saved to {OUTPUT_IMAGE_PATH}")
            
            # Cleanup
            import os
            os.remove("temp_room.png")
            
            return result_img
    else:
        print(f"❌ API Error {response.status_code}: {response.text}")
        print("\n🎨 Using fallback: Simple tile blending...")
        return fallback_simple_blend()


def fallback_simple_blend():
    """Fallback method without AI API"""
    print("\n📸 Loading images...")
    room_img = Image.open(ROOM_IMAGE_PATH).convert('RGB')
    tile_img = Image.open(TILE_IMAGE_PATH).convert('RGB')
    
    print("🎭 Creating floor mask...")
    mask_img = create_floor_mask(ROOM_IMAGE_PATH, floor_start_ratio=0.4)
    
    print("🎨 Blending tiles into floor...")
    result = simple_inpainting_blend(room_img, mask_img, tile_img)
    
    result.save(OUTPUT_IMAGE_PATH)
    print(f"\n✅ Image saved to {OUTPUT_IMAGE_PATH}")
    print("💡 This is a simple blend. For AI-powered results, add an API key.")
    
    return result


# ================= RUN =================

if __name__ == "__main__":
    print("="*60)
    print("  🏠 FLOOR TILE REPLACEMENT - FREE VERSION")
    print("  💯 No billing required!")
    print("="*60)
    
    result = inpaint_with_deepai(DEEPAI_API_KEY)
    
    if result:
        print("\n✨ Done! Check the output image.")
    else:
        print("\n⚠️ See instructions above to get a free API key")
    
    print("\n" + "="*60)
    print("📚 OPTIONS:")
    print("  1. Get DeepAI key for AI-powered results")
    print("  2. Or use the simple blend (no AI needed)")
    print("  3. Adjust 'floor_start_ratio' to change floor area")
    print("="*60)
