"""
Floor Tile Replacement using Replicate API (FREE Credits!)
More reliable than HF API for inpainting
"""

from PIL import Image, ImageDraw
import numpy as np
import requests
import io
import base64
import time

# ================= CONFIG =================
# Get your FREE token from: https://replicate.com/account/api-tokens
REPLICATE_API_TOKEN = ""  # ⬅️ REPLACE THIS

ROOM_IMAGE_PATH = "room.jpg"
TILE_IMAGE_PATH = "tile.jpg"
OUTPUT_IMAGE_PATH = "room_with_tile_replicate.png"
MASK_PATH = "mask_replicate.png"

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
        f"photorealistic luxury interior floor with {pattern_desc} tiles. "
        f"Floor tiles are {color_desc} colored with subtle grout lines. "
        f"The tiles have realistic lighting, reflections, proper perspective vanishing point. "
        f"Professional architectural photography, high detail ceramic floor tiles, "
        f"natural lighting with soft shadows. "
        f"The tiles extend from wall to wall covering the entire floor area."
    )
    
    return prompt


def image_to_data_uri(image_path):
    """Convert image to data URI for API"""
    with open(image_path, "rb") as img_file:
        img_data = base64.b64encode(img_file.read()).decode()
    
    # Detect image type
    ext = image_path.split('.')[-1].lower()
    mime_type = f"image/{ext}" if ext in ['png', 'jpg', 'jpeg', 'webp'] else "image/png"
    return f"data:{mime_type};base64,{img_data}"


# ================= MAIN INPAINTING WITH REPLICATE =================

def inpaint_with_replicate(api_token=None):
    """
    Use Replicate API for inpainting (more reliable!)
    Get free token from: https://replicate.com/account/api-tokens
    """
    
    if not api_token or api_token == "your_replicate_token_here":
        print("❌ Please provide a Replicate API token")
        print("Get one FREE at: https://replicate.com/account/api-tokens")
        print("\n1. Sign up at https://replicate.com")
        print("2. Go to Account > API Tokens")
        print("3. Create a new token and paste it in the REPLICATE_API_TOKEN variable")
        print("\n💡 Replicate offers FREE credits to start!")
        return
    
    print("🚀 Starting Replicate API Inpainting...")
    print("📦 Using model: stability-ai/stable-diffusion-inpainting")
    
    # Load and prepare images
    print("\n📸 Loading images...")
    room_img = resize_image(ROOM_IMAGE_PATH, size=(512, 512))
    room_img.save("temp_room.png")
    
    # Create mask
    print("🎭 Creating floor mask...")
    mask_img = create_floor_mask(ROOM_IMAGE_PATH, floor_start_ratio=0.4)
    mask_img = mask_img.resize((512, 512), Image.Resampling.LANCZOS)
    mask_img.save("temp_mask.png")
    
    # Generate prompt
    print("\n🔍 Analyzing tile image...")
    prompt = generate_prompt_from_tile(TILE_IMAGE_PATH)
    print(f"📝 Generated prompt:\n{prompt}\n")
    
    # Convert to data URIs
    image_uri = image_to_data_uri("temp_room.png")
    mask_uri = image_to_data_uri("temp_mask.png")
    
    # Replicate API endpoint
    api_url = "https://api.replicate.com/v1/predictions"
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json"
    }
    
    # Create prediction
    data = {
        "version": "95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",  # SD inpainting
        "input": {
            "image": image_uri,
            "mask": mask_uri,
            "prompt": prompt,
            "negative_prompt": "blurry, distorted, low quality, unrealistic, cartoon, painting",
            "num_inference_steps": 25,
            "guidance_scale": 7.5,
        }
    }
    
    print("🌐 Sending request to Replicate API...")
    response = requests.post(api_url, headers=headers, json=data)
    
    if response.status_code != 201:
        print(f"❌ Error {response.status_code}: {response.text}")
        return None
    
    prediction = response.json()
    prediction_url = prediction["urls"]["get"]
    
    # Poll for result
    print("⏳ Processing... (this may take 30-60 seconds)")
    max_wait = 120  # 2 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        time.sleep(2)
        result = requests.get(prediction_url, headers=headers).json()
        
        status = result["status"]
        print(f"   Status: {status}...")
        
        if status == "succeeded":
            output_url = result["output"][0] if isinstance(result["output"], list) else result["output"]
            print(f"\n✅ Generation complete! Downloading...")
            
            # Download result
            img_response = requests.get(output_url)
            result_img = Image.open(io.BytesIO(img_response.content))
            result_img.save(OUTPUT_IMAGE_PATH)
            print(f"✅ Image saved to {OUTPUT_IMAGE_PATH}")
            print(f"📐 Output size: {result_img.size}")
            
            # Cleanup
            import os
            os.remove("temp_room.png")
            os.remove("temp_mask.png")
            
            return result_img
        
        elif status == "failed":
            print(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
            return None
    
    print("❌ Timeout waiting for result")
    return None


# ================= RUN =================

if __name__ == "__main__":
    print("="*60)
    print("  🏠 FLOOR TILE REPLACEMENT WITH REPLICATE API")
    print("  💯 FREE Credits - No PyTorch needed!")
    print("="*60)
    
    result = inpaint_with_replicate(REPLICATE_API_TOKEN)
    
    if result:
        print("\n✨ Done! Check the output image.")
    else:
        print("\n❌ Generation failed. Check your token and try again.")
    
    print("\n" + "="*60)
    print("📚 TIPS:")
    print("  - Adjust 'floor_start_ratio' to change floor area")
    print("  - Replicate offers $5 free credits!")
    print("  - Much more reliable than HF Inference API")
    print("="*60)
