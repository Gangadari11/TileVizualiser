import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image

# 1. SETUP - Replace with your actual Project ID
PROJECT_ID = "" 
vertexai.init(project=PROJECT_ID, location="us-central1")

def run_visualization():
    # Load the "Nano Banana" powered model (Image Gen 006)
    model = ImageGenerationModel.from_pretrained("image-generation-006")

    # Load your images (Make sure these files are in the same folder!)
    base_room = Image.load_from_file("room.jpg")
    tile_sample = Image.load_from_file("tile.jpg")

    print("AI is tiling the floor... please wait.")

    # Generate the edit
    images = model.edit_image(
        base_image=base_room,
        prompt="Keep the room exactly as is, but replace the floor material with the tile pattern from the reference image. Match the perspective and lighting.",
        reference_images=[tile_sample],
        guidance_scale=60
    )

    # Save output
    images[0].save(location="result.png")
    print("Done! Check result.png")

if __name__ == "__main__":
    run_visualization()