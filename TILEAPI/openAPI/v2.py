import base64
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO

# ================= CONFIG =================
OPENAI_API_KEY = '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'
client = OpenAI(api_key=OPENAI_API_KEY)

ROOM_IMAGE_PATH = "room.jpg"
TILE_IMAGE_PATH = "tile.jpg"

# ================= ANALYZE TILE WITH GPT-4 VISION =================
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

tile_base64 = encode_image(TILE_IMAGE_PATH)

# Get detailed tile description
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe these floor tiles in detail: colors, pattern, style, texture, size, arrangement. Be specific for image generation."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{tile_base64}"
                    }
                }
            ]
        }
    ],
    max_tokens=300
)

tile_description = response.choices[0].message.content
print("🎨 Tile Description from GPT-4 Vision:")
print(tile_description)
print("\n" + "="*50 + "\n")

# ================= PREPARE ROOM IMAGE =================
def resize_for_openai(image_path, size=(1024, 1024)):
    img = Image.open(image_path).convert('RGBA')
    img_resized = img.resize(size, Image.Resampling.LANCZOS)
    resized_path = image_path.replace('.jpg', '_resized.png')
    img_resized.save(resized_path)
    return resized_path

room_resized = resize_for_openai(ROOM_IMAGE_PATH)

# Create mask
room_img = Image.open(room_resized).convert('RGBA')
width, height = room_img.size
mask = Image.new('RGBA', (width, height), (255, 255, 255, 255))
from PIL import ImageDraw
draw = ImageDraw.Draw(mask)
draw.rectangle([(0, int(height * 0.4)), (width, height)], fill=(0, 0, 0, 0))
mask.save('mask_openai.png')

# ================= GENERATE WITH ANALYZED DESCRIPTION =================
prompt = f"A luxury interior room with floor tiles. {tile_description} The tiles should have realistic lighting, reflections, and proper perspective."

try:
    response = client.images.edit(
        model="dall-e-2",
        image=open(room_resized, "rb"),
        mask=open('mask_openai.png', "rb"),
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    
    image_url = response.data[0].url
    img_response = requests.get(image_url)
    img = Image.open(BytesIO(img_response.content))
    img.save("room_with_tile_openai.png")
    print(f"✅ Generated image saved!")
    
except Exception as e:
    print(f"❌ Error: {e}")