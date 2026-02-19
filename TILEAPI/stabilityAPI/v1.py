import io
import os
import warnings
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

# ================= CONFIG =================
os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
os.environ['STABILITY_KEY'] = 'sk-vsfo9rx6OPllQgpaFsKP5BsiuFqfw1YPShwq9Uw3cDm4p4p5'

ROOM_IMAGE_PATH = "room2.jpg"
TILE_IMAGE_PATH = "tile.jpg"
OUTPUT_IMAGE_PATH = "room_with_tile.png"

# ================= LOAD AND RESIZE IMAGES =================
room_img = Image.open(ROOM_IMAGE_PATH).convert('RGB')
tile_img = Image.open(TILE_IMAGE_PATH).convert('RGB')

print(f"Original room size: {room_img.size}")
print(f"Tile size: {tile_img.size}")

# Resize room to SDXL compatible dimensions
TARGET_SIZE = (1024, 1024)
room_img_resized = room_img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
room_img_resized.save("room_resized.jpg")
print(f"✅ Resized room to {TARGET_SIZE}")

# Analyze tile colors for better prompt
tile_colors = tile_img.getcolors(tile_img.size[0] * tile_img.size[1])
print(f"Tile has {len(tile_colors)} unique colors")

# ================= CONNECT TO API =================
stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'],
    verbose=True,
    engine="stable-diffusion-xl-1024-v1-0",
)

# ================= GENERATE WITH TILE FLOOR =================
# Create a detailed prompt based on your tile
prompt = (
    "Detect the visible floor region in the room image and replace only that surface with the provided tile texture.Requirements:The floor may be irregular, angled, or partially occluded — do not assume a rectangular shape. Preserve the original room geometry and perspective. Warp the tile texture to match the floor plane naturally. Repeat tiles at realistic scale, aligned with perspective. Do not modify walls, furniture, lighting, or objects. Blend edges smoothly with the original image.Maintain natural shadows and brightness.Result should look like a realistic floor renovation preview, not an artistic edit.Goal: produce a professional interior visualization where tiles appear physically installed on the existing floor."
)

answers = stability_api.generate(
    prompt=prompt,
    init_image=room_img_resized,
    start_schedule=0.4,  # 0.4 = keep 60% of original room, modify 40%
    seed=123456789,
    steps=50,
    cfg_scale=7.0,
    width=1024,
    height=1024,
    sampler=generation.SAMPLER_K_DPMPP_2M
)

# ================= SAVE RESULT =================
for resp in answers:
    for artifact in resp.artifacts:
        if artifact.finish_reason == generation.FILTER:
            warnings.warn(
                "Your request activated the API's safety filters and could not be processed. "
                "Please modify the prompt and try again.")
        if artifact.type == generation.ARTIFACT_IMAGE:
            img2 = Image.open(io.BytesIO(artifact.binary))
            img2.save(OUTPUT_IMAGE_PATH)
            print(f"✅ Generated image saved as {OUTPUT_IMAGE_PATH}")