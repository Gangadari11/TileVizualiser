# Hugging Face API Tile Replacement - v4.py

## Description
This script replaces floor tiles in room images using the Hugging Face Inference API with inpainting models.

## Requirements
```bash
pip install pillow requests numpy
```

## Setup

### 1. Get Your Hugging Face API Token
- Visit: https://huggingface.co/settings/tokens
- Create a new token (Read access is sufficient)
- Copy your token (starts with `hf_`)

### 2. Set Your API Token

**Option A: Environment Variable (Recommended)**
```bash
# Windows PowerShell
$env:HF_API_TOKEN="hf_your_token_here"

# Windows CMD
set HF_API_TOKEN=hf_your_token_here

# Linux/Mac
export HF_API_TOKEN="hf_your_token_here"
```

**Option B: Edit the File Directly**
Open `v4.py` and replace line 20 with:
```python
HF_API_TOKEN = "hf_your_token_here"  # Replace with your actual token
```

## Usage

1. Place your images in the same folder:
   - `room.jpg` - Your room image
   - `tile.jpg` - Your tile pattern

2. Run the script:
```bash
python v4.py
```

3. Output will be saved as `room_with_tile_v4.png`

## How It Works

### Method 1: Hugging Face API (Primary)
- Uses stable-diffusion-based inpainting models
- Requires API token
- High quality AI-powered results
- May take 20-60 seconds per image

### Method 2: Advanced Blending (Fallback)
- Activates if API fails or no token provided
- Uses perspective transformation and lighting matching
- No API required
- Good quality results

## Models Used
1. `stabilityai/stable-diffusion-2-inpainting` (Primary)
2. `runwayml/stable-diffusion-inpainting` (Backup)
3. `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` (Backup)

## Troubleshooting

### "API token not found"
- Make sure you've set the `HF_API_TOKEN` environment variable or edited the file

### "Model loading" (503 error)
- The model is warming up on Hugging Face servers
- Script will wait 10 seconds and retry automatically

### "All models failed"
- Check your internet connection
- Verify your API token is valid
- The fallback method will activate automatically

## API Rate Limits
- Free Hugging Face Inference API has rate limits
- If you hit limits, the fallback method will work without API
