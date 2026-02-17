"""
v4.py - Accurate Floor Tile Replacement using Hugging Face API
Uses Hugging Face Inference API with API token (direct and reliable)
"""

from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
import requests
import io
import os
import time

# ================= CONFIG =================
ROOM_IMAGE_PATH = "room.jpg"
TILE_IMAGE_PATH = "tile.jpg"
OUTPUT_IMAGE_PATH = "room_with_tile_v4.png"
MASK_PATH = "mask_v4.png"

# Hugging Face API Configuration
HF_API_TOKEN = "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" # Set your token in environment or below
# Or set directly: HF_API_TOKEN = "hf_your_token_here"

# Available inpainting models on Hugging Face (updated for 2026)
# Using models with confirmed availability and better API support
INPAINTING_MODELS = [
    "timbrooks/instruct-pix2pix",  # Image instruction editing
    "Fantasy-Studio/Paint-by-Example",  # Paint by example inpainting
    "kandinsky-community/kandinsky-2-2-decoder-inpaint",  # Kandinsky inpainting
]

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
        color_desc = "white and light beige marble"
    elif r > 150 and g > 120 and b < 100:
        color_desc = "beige, cream, and brown ceramic"
    elif r > 100 and g > 100 and b > 100:
        color_desc = "neutral gray and white stone"
    elif r < 50 and g < 50 and b < 50:
        color_desc = "dark gray and black marble"
    else:
        color_desc = "multicolored patterned"
    
    # Detect pattern
    tile_small = tile.resize((50, 50))
    pixels = list(tile_small.convert('RGB').getdata())  # Will use numpy instead
    tile_array_small = np.array(tile_small.convert('RGB'))
    unique_colors = len(np.unique(tile_array_small.reshape(-1, 3), axis=0))
    
    if unique_colors < 10:
        pattern_desc = "solid uniform tiles"
    elif unique_colors < 50:
        pattern_desc = "geometric checkerboard pattern tiles"
    else:
        pattern_desc = "complex textured marble tiles with natural veining"
    
    return color_desc, pattern_desc


def create_floor_mask(image_path, floor_start_ratio=0.4):
    """Create a mask for the floor area (white=keep, black=inpaint)"""
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    
    # Create white mask (keep everything initially)
    mask = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(mask)
    
    # Draw black rectangle for floor area (will be inpainted)
    floor_start = int(height * floor_start_ratio)
    draw.rectangle([(0, floor_start), (width, height)], fill=0)
    
    # Smooth edges
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
    
    mask.save(MASK_PATH)
    print(f"✅ Mask created: {MASK_PATH}")
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
        f"high quality interior photograph, luxury room floor covered with beautiful {color_desc} {pattern_desc}, "
        f"the floor tiles have realistic lighting and natural reflections, proper perspective with vanishing point, "
        f"photorealistic ceramic tiles with subtle grout lines, professional architectural photography, "
        f"8k resolution, highly detailed floor texture, natural daylight, sharp focus"
    )
    
    return prompt


# ================= HUGGING FACE API INPAINTING =================

def inpaint_with_huggingface_api():
    """
    Use Hugging Face Inference API for inpainting
    Requires HF_API_TOKEN
    """
    if not HF_API_TOKEN:
        print("❌ Hugging Face API token not found!")
        print("💡 Set your token:")
        print("   - Environment: export HF_API_TOKEN='hf_your_token_here'")
        print("   - Or edit the file and set: HF_API_TOKEN = 'hf_your_token_here'")
        print("   - Get token from: https://huggingface.co/settings/tokens")
        return None
    
    print("🚀 Using Hugging Face Inference API")
    print(f"🤖 Model: {INPAINTING_MODELS[0]}\n")
    
    # Prepare images
    print("📸 Preparing images...")
    room_img = resize_image(ROOM_IMAGE_PATH, size=(512, 512))
    
    mask_img = create_floor_mask(ROOM_IMAGE_PATH, floor_start_ratio=0.4)
    mask_img = mask_img.resize((512, 512), Image.Resampling.LANCZOS)
    
    # Invert mask for HF API (black=keep, white=inpaint)
    mask_inverted = Image.eval(mask_img, lambda x: 255 - x)
    
    prompt = generate_prompt_from_tile(TILE_IMAGE_PATH)
    print(f"📝 Prompt: {prompt[:80]}...\n")
    
    # Prepare request
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    # Convert images to bytes
    room_bytes = io.BytesIO()
    room_img.save(room_bytes, format='PNG')
    room_bytes = room_bytes.getvalue()
    
    mask_bytes = io.BytesIO()
    mask_inverted.save(mask_bytes, format='PNG')
    mask_bytes = mask_bytes.getvalue()
    
    # Build payload
    data = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": "blurry, distorted, low quality, unrealistic, bad perspective",
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
        }
    }
    
    # Try each model
    for model_name in INPAINTING_MODELS:
        try:
            api_url = f"https://api-inference.huggingface.co/models/{model_name}"
            print(f"🌐 Trying model: {model_name}")
            print("⏳ Generating (this may take 20-60 seconds)...")
            
            # Try multipart file upload format
            files = {
                'image': ('image.png', room_bytes, 'image/png'),
                'mask_image': ('mask.png', mask_bytes, 'image/png'),
            }
            
            # Send request with timeout
            response = requests.post(
                api_url,
                headers=headers,
                files=files,
                data={'prompt': prompt, 'negative_prompt': 'blurry, distorted, low quality'},
                timeout=120
            )
            
            # Check response
            if response.status_code == 200:
                try:
                    result_img = Image.open(io.BytesIO(response.content))
                    
                    # Resize back to original size
                    original_size = Image.open(ROOM_IMAGE_PATH).size
                    result_img = result_img.resize(original_size, Image.Resampling.LANCZOS)
                    
                    result_img.save(OUTPUT_IMAGE_PATH)
                    print(f"\n✅ SUCCESS with {model_name}!")
                    print(f"📁 Saved: {OUTPUT_IMAGE_PATH}")
                    print(f"📐 Size: {result_img.size}")
                    return result_img
                except Exception as e:
                    print(f"⚠️ Failed to process response: {str(e)[:100]}")
                    continue
            
            elif response.status_code == 503:
                print(f"⏳ Model loading... waiting 15 seconds")
                time.sleep(15)
                # Retry once
                response = requests.post(
                    api_url,
                    headers=headers,
                    files=files,
                    data={'prompt': prompt, 'negative_prompt': 'blurry, distorted, low quality'},
                    timeout=120
                )
                if response.status_code == 200:
                    try:
                        result_img = Image.open(io.BytesIO(response.content))
                        original_size = Image.open(ROOM_IMAGE_PATH).size
                        result_img = result_img.resize(original_size, Image.Resampling.LANCZOS)
                        result_img.save(OUTPUT_IMAGE_PATH)
                        print(f"\n✅ SUCCESS with {model_name}!")
                        print(f"📁 Saved: {OUTPUT_IMAGE_PATH}")
                        return result_img
                    except Exception as e:
                        print(f"⚠️ Failed to process response: {str(e)[:100]}")
                        continue
                else:
                    error_msg = response.text[:300] if response.text else "No error message"
                    print(f"⚠️ Error {response.status_code}: {error_msg}")
            elif response.status_code == 410:
                print(f"⚠️ Model {model_name} is no longer available (410 Gone)")
            elif response.status_code == 404:
                print(f"⚠️ Model {model_name} not found (404)")
            else:
                error_msg = response.text[:300] if response.text else "No error message"
                print(f"⚠️ Error {response.status_code}: {error_msg}")
                
        except Exception as e:
            print(f"⚠️ {model_name} failed: {str(e)[:150]}")
            continue
    
    print("\n❌ All Hugging Face models failed")
    print("💡 Consider these alternatives:")
    print("   - Use Replicate API (../replicateAPI/v3_replicate.py)")
    print("   - Use DeepAI API (../deepAPI/v1.py)")  
    print("   - Check Hugging Face model hub for new inpainting models")
    return None





# ================= MAIN =================

def main():
    print("="*70)
    print("  🏠 V4 - FLOOR TILE REPLACEMENT")
    print("  🤖 Using Hugging Face Inference API")
    print("="*70 + "\n")
    
    try:
        result = inpaint_with_huggingface_api()
        if result:
            print("\n✨ Success! Check your output image.")
            print("="*70)
            return
        else:
            print("\n" + "="*70)
            print("❌ API PROCESSING FAILED")
            print("="*70)
            print("\n💡 TROUBLESHOOTING:")
            print("  1. Check your Hugging Face API token is valid")
            print("  2. Verify room.jpg and tile.jpg exist")
            print("  3. Check internet connection")
            print("  4. Try a different model or check HF status")
            print("="*70)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Please check your API token and internet connection")


if __name__ == "__main__":
    main()
