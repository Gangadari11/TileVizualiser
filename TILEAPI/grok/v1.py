import os
import base64
import requests
from xai_sdk import Client

# Load API key from environment (recommended)
api_key ="xai-r3b3D2Tpitp7bhgIcWMWMG2haNEry4ok7x9tiUDIxLfi7XcExzJNdxk5KdjjbTNlUPg1saOxmTcPw8L2"
if not api_key:
    api_key = "xai-r3b3D2Tpitp7bhgIcWMWMG2haNEry4ok7x9tiUDIxLfi7XcExzJNdxk5KdjjbTNlUPg1saOxmTcPw8L2"  # ← replace only for quick testing; use env var in production!

client = Client(api_key=api_key)

def file_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Your file paths (update these!)
room_path = "../room.jpg"   # photo of the room (floor to be tiled)
tile_path = "../tile.jpg"   # sample tile image/pattern

room_b64 = file_to_base64(room_path)
tile_b64 = file_to_base64(tile_path)  # we'll describe it in prompt

# Strong editing prompt — be very specific for best results
prompt = (
    "Edit this room photo: Replace the existing floor completely with this exact tile pattern and material shown in the reference tile image. "
    "Make the tiles seamless, realistically laid out following the room's perspective and lighting. "
    "Do NOT change walls, ceiling, furniture, windows, shadows, objects, or any other elements — keep everything else identical. "
    "Photorealistic, high detail, natural tiling alignment, no distortions."
    # If you want to help the model "see" the tile better, you can add: "Reference tile: [describe color, shape, size if known]"
)

# Call image editing/generation
response = client.image.sample(
    model="grok-imagine-image",          # current best for image edits
    prompt=prompt,
    image_url=f"data:image/jpeg;base64,{room_b64}",  # base image to edit
    # Optional params (adjust as needed)
    # aspect_ratio can be: "1:1", "16:9", "9:16", "4:3", "3:4" — omitted to use default
    # If multi-image reference is supported in your SDK version, you might add:
    # reference_images=[f"data:image/jpeg;base64,{tile_b64}"]   # but check docs first — not always available
)

# Response handling (usually returns URL(s))
if hasattr(response, 'url'):
    generated_url = response.url
elif hasattr(response, 'images') and response.images:
    generated_url = response.images[0].url
else:
    print("Unexpected response format:", response)
    exit()

print("Generated image URL:", generated_url)

# Download and save locally
img_data = requests.get(generated_url).content
output_path = "room_tiled_visualized.jpg"
with open(output_path, "wb") as f:
    f.write(img_data)

print(f"Saved result to: {output_path}")