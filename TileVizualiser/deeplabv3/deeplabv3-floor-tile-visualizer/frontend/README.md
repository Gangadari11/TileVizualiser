# Floor Tile Visualizer - Web Frontend

A modern web-based interface for the Floor Tile Visualization Engine.

## Features

- 🖼️ **Room Image Upload**: Drag and drop or click to upload room photos
- 🎯 **Floor Selection**: 
  - Manual polygon drawing tool
  - Auto-detect floor using AI
- 🎨 **Tile Gallery**: Choose from pre-loaded tiles or upload custom textures
- ⚙️ **Adjustable Settings**:
  - Tile size (30-150cm)
  - Grout width (0-10mm)
- 📥 **Download**: Export high-quality results

## Quick Start

1. **Activate the virtual environment** (from parent project):
   ```bash
   cd ..\..\..
   .\venv10\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   cd deeplabv3\deeplabv3-floor-tile-visualizer\frontend
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

## Project Structure

```
frontend/
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css     # Application styles
│   ├── js/
│   │   └── app.js        # Frontend JavaScript
│   └── tiles/            # Pre-loaded tile textures
├── uploads/              # Uploaded room images
└── outputs/              # Generated results
```

## Usage Guide

### Step 1: Upload Room Image
- Click the upload zone or drag and drop a room photo
- Supported formats: JPG, PNG, WebP

### Step 2: Select Floor Area
**Option A - Manual Selection:**
1. Click on the image to add polygon points around the floor
2. Click near the first point to close the polygon

**Option B - Auto-Detect:**
1. Click the "Auto-Detect" button
2. The AI will automatically detect the floor area

### Step 3: Choose a Tile
- Click on any tile in the gallery
- Or upload your own tile texture using "Upload Custom Tile"

### Step 4: Adjust Settings
- **Tile Size**: Larger tiles = fewer tiles visible (clearer pattern)
- **Grout Width**: Space between tiles (0mm = seamless)

### Step 5: Apply & Download
1. Click "Apply Tile" button
2. Wait for processing (may take a few seconds)
3. Preview the result
4. Download full quality image

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main application page |
| `/api/tiles` | GET | Get available tiles |
| `/api/upload` | POST | Upload room image |
| `/api/upload-tile` | POST | Upload custom tile |
| `/api/auto-detect` | POST | Auto-detect floor area |
| `/api/visualize` | POST | Apply tile to floor |
| `/api/download/<filename>` | GET | Download result |

## Tips for Best Results

1. **Room Images**: Use well-lit photos with clear floor visibility
2. **Floor Selection**: Be precise around furniture edges for cleaner results
3. **Tile Size**: Start with 60-80cm for most rooms
4. **Complex Patterns**: Use larger tile sizes (80-120cm) for detailed patterns

## Troubleshooting

**Image upload fails:**
- Check file size (max 50MB)
- Ensure file is a valid image format

**Auto-detect not working well:**
- Try manual polygon selection
- Ensure floor is clearly visible in the image

**Tile looks stretched:**
- Verify the floor polygon is correctly drawn
- Try adjusting tile size

## Credits

Built using the Floor Tile Visualization Engine with:
- Professional Tile Installer
- Realistic Blending Engine
- Plane Approximation System
