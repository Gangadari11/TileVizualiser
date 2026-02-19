from google import genai
from google.genai import types

client = genai.Client(api_key="5efd834a-17bc-467a-ae78-bc627c6f4a15")

# Load your images
room_img = client.files.upload(file="room.jpg")
tile_img = client.files.upload(file="tile.jpg")

response = client.models.generate_content(
    model="gemini-2.5-flash-image",
    contents=[
        "Please replace the existing floor in this room with the tile pattern shown in the second image. Maintain the original furniture and lighting.",
        room_img,
        tile_img
    ],
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"]
    )
)

# Save the result
for part in response.candidates[0].content.parts:
    if part.inline_data:
        with open("visualized_room.png", "wb") as f:
            f.write(part.inline_data.data)