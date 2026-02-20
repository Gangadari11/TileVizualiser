"""
Floor Tile Visualizer - Web Application
========================================
Flask-based web application for interactive floor tile visualization.
Provides a modern UI for:
- Uploading room images
- Selecting floor areas with polygon tool
- Choosing from available tile textures
- Applying tiles with professional quality rendering
"""

import os
import sys
import json
import uuid
import base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
import numpy as np

# Add parent src to path for engine imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils.headless_visualizer import visualize, auto_detect_floor, get_floor_polygon

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'outputs')
TILES_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'tiles')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TILES_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
CORS(app)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_available_tiles():
    """Get list of available tile textures"""
    tiles = []
    for filename in os.listdir(TILES_FOLDER):
        if allowed_file(filename):
            tiles.append({
                'name': filename,
                'path': f'/static/tiles/{filename}',
                'thumbnail': f'/static/tiles/{filename}'
            })
    return tiles


@app.route('/')
def index():
    """Render main application page"""
    tiles = get_available_tiles()
    return render_template('index.html', tiles=tiles)


@app.route('/api/tiles')
def api_tiles():
    """Get available tiles as JSON"""
    return jsonify(get_available_tiles())


@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Handle room image upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, webp'}), 400
    
    try:
        # Generate unique filename
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save file
        file.save(filepath)
        
        # Load and get image dimensions
        img = cv2.imread(filepath)
        if img is None:
            os.remove(filepath)
            return jsonify({'error': 'Could not read image file'}), 400
        
        height, width = img.shape[:2]
        
        # Return success with image info
        return jsonify({
            'success': True,
            'filename': filename,
            'path': f'/uploads/{filename}',
            'width': width,
            'height': height
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload-tile', methods=['POST'])
def upload_tile():
    """Handle custom tile image upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Generate unique filename for custom tile
        ext = file.filename.rsplit('.', 1)[1].lower()
        original_name = secure_filename(file.filename.rsplit('.', 1)[0])
        filename = f"custom_{original_name}_{uuid.uuid4().hex[:8]}.{ext}"
        filepath = os.path.join(TILES_FOLDER, filename)
        
        # Save file
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'name': filename,
            'path': f'/static/tiles/{filename}',
            'thumbnail': f'/static/tiles/{filename}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/auto-detect', methods=['POST'])
def auto_detect():
    """Automatically detect floor area in uploaded image"""
    data = request.get_json()
    
    if not data or 'filename' not in data:
        return jsonify({'error': 'No filename provided'}), 400
    
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], data['filename'])
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404
        
        # Load image
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Could not read image'}), 400
        
        # Auto-detect floor
        mask = auto_detect_floor(img)
        
        # Get polygon points from mask
        points = get_floor_polygon(mask)
        
        if not points:
            return jsonify({'error': 'Could not detect floor area'}), 400
        
        return jsonify({
            'success': True,
            'points': points,
            'coverage': float(np.count_nonzero(mask) / mask.size * 100)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/visualize', methods=['POST'])
def api_visualize():
    """Apply tile to floor area"""
    data = request.get_json()
    
    # Validate required fields
    required = ['room_image', 'tile_image', 'polygon_points']
    for field in required:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    try:
        # Load room image
        room_path = os.path.join(app.config['UPLOAD_FOLDER'], data['room_image'])
        if not os.path.exists(room_path):
            return jsonify({'error': 'Room image not found'}), 404
        
        room_img = cv2.imread(room_path)
        if room_img is None:
            return jsonify({'error': 'Could not read room image'}), 400
        
        # Load tile image
        tile_filename = data['tile_image']
        tile_path = os.path.join(TILES_FOLDER, tile_filename)
        if not os.path.exists(tile_path):
            return jsonify({'error': 'Tile image not found'}), 404
        
        tile_img = cv2.imread(tile_path)
        if tile_img is None:
            return jsonify({'error': 'Could not read tile image'}), 400
        
        # Get options
        tile_size = data.get('tile_size_cm', 200.0)
        grout_width = data.get('grout_width_mm', 2.0)
        resolution = data.get('resolution', 4000)
        
        polygon_points = data['polygon_points']
        
        # Visualize
        result = visualize(
            room_image=room_img,
            tile_image=tile_img,
            polygon_points=polygon_points,
            tile_size_cm=tile_size,
            grout_width_mm=grout_width,
            resolution=resolution
        )
        
        if not result['success']:
            return jsonify({'error': result['message']}), 400
        
        # Save result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"result_{uuid.uuid4().hex[:8]}_{timestamp}.jpg"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        cv2.imwrite(output_path, result['result'], [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Also create a preview (smaller version for quick display)
        preview = result['result'].copy()
        max_dim = 1200
        h, w = preview.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            preview = cv2.resize(preview, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
        preview_filename = f"preview_{output_filename}"
        preview_path = os.path.join(app.config['OUTPUT_FOLDER'], preview_filename)
        cv2.imwrite(preview_path, preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        return jsonify({
            'success': True,
            'message': result['message'],
            'result_path': f'/outputs/{output_filename}',
            'preview_path': f'/outputs/{preview_filename}',
            'filename': output_filename
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/outputs/<filename>')
def serve_output(filename):
    """Serve output files"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


@app.route('/api/download/<filename>')
def download_result(filename):
    """Download full quality result"""
    return send_from_directory(
        app.config['OUTPUT_FOLDER'], 
        filename,
        as_attachment=True
    )


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏠 FLOOR TILE VISUALIZER - Web Application")
    print("="*60)
    print(f"\n📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📁 Output folder: {OUTPUT_FOLDER}")
    print(f"📁 Tiles folder: {TILES_FOLDER}")
    print(f"\n🎨 Available tiles: {len(get_available_tiles())}")
    print("\n🌐 Starting server...")
    print("   Open http://localhost:5000 in your browser")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
