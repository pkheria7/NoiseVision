from flask import Flask, render_template, send_from_directory
import os
from pathlib import Path

app = Flask(__name__)

# Configuration
DETECTED_FOLDER = 'detected'

def get_image_files():
    """Get all image files from the detected folder"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif')
    return [f for f in os.listdir(DETECTED_FOLDER) 
            if f.lower().endswith(image_extensions)]

@app.route('/')
def gallery():
    """Render the gallery page"""
    images = get_image_files()
    return render_template('gallery.html', images=images)

@app.route('/detected/<path:filename>')
def serve_image(filename):
    """Serve images from the detected folder"""
    return send_from_directory(DETECTED_FOLDER, filename)

if __name__ == '__main__':
    # Create detected folder if it doesn't exist
    Path(DETECTED_FOLDER).mkdir(exist_ok=True)
    
    # Create templates folder and gallery template
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    # Create the enhanced gallery template
    gallery_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Manual Checking Gallery</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
            color: white;
            padding: 30px 20px;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .header h1 {
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #fff, #f0f0f0);
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        
        .header .subtitle {
            font-size: 1.2rem;
            opacity: 0.9;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        
        .stats-badge {
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
        }
        
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 25px;
            padding: 20px;
        }
        
        .gallery-item {
            position: relative;
            overflow: hidden;
            border-radius: 15px;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            border: 1px solid rgba(255, 255, 255, 0.3);
            aspect-ratio: 1;
        }
        
        .gallery-item:hover {
            transform: translateY(-10px) scale(1.02);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
        }
        
        .gallery-item img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
            transition: transform 0.4s ease;
        }
        
        .gallery-item:hover img {
            transform: scale(1.1);
        }
        
        .gallery-item .overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(
                to bottom,
                transparent 0%,
                transparent 50%,
                rgba(0, 0, 0, 0.3) 70%,
                rgba(0, 0, 0, 0.8) 100%
            );
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .gallery-item:hover .overlay {
            opacity: 1;
        }
        
        .gallery-item .caption {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(
                to top,
                rgba(0, 0, 0, 0.9) 0%,
                rgba(0, 0, 0, 0.7) 60%,
                transparent 100%
            );
            color: white;
            padding: 20px;
            font-size: 14px;
            font-weight: 500;
            transform: translateY(100%);
            transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }
        
        .gallery-item:hover .caption {
            transform: translateY(0);
        }
        
        .caption .filename {
            font-weight: 600;
            margin-bottom: 5px;
            font-size: 16px;
        }
        
        .caption .file-info {
            font-size: 12px;
            opacity: 0.8;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .no-images {
            grid-column: 1 / -1;
            text-align: center;
            padding: 60px 40px;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white;
        }
        
        .no-images i {
            font-size: 4rem;
            margin-bottom: 20px;
            opacity: 0.5;
        }
        
        .no-images h3 {
            font-size: 1.5rem;
            margin-bottom: 10px;
        }
        
        .no-images p {
            font-size: 1rem;
            opacity: 0.8;
        }
        
        .loading-spinner {
            display: none;
            text-align: center;
            padding: 40px;
            color: white;
        }
        
        .loading-spinner i {
            font-size: 2rem;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .gallery-item .image-actions {
            position: absolute;
            top: 10px;
            right: 10px;
            display: flex;
            gap: 8px;
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .gallery-item:hover .image-actions {
            opacity: 1;
        }
        
        .action-btn {
            background: rgba(255, 255, 255, 0.9);
            border: none;
            border-radius: 50%;
            width: 35px;
            height: 35px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        
        .action-btn:hover {
            background: white;
            transform: scale(1.1);
        }
        
        .action-btn i {
            font-size: 14px;
            color: #333;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .gallery {
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                gap: 20px;
                padding: 15px;
            }
            
            .header h1 {
                font-size: 2.2rem;
            }
            
            .header .subtitle {
                font-size: 1rem;
            }
            
            body {
                padding: 15px;
            }
        }
        
        @media (max-width: 480px) {
            .gallery {
                grid-template-columns: 1fr;
                gap: 15px;
                padding: 10px;
            }
            
            .header h1 {
                font-size: 1.8rem;
            }
            
            .header {
                padding: 20px 15px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-images"></i> Manual Checking Gallery</h1>
            <div class="subtitle">
                <span class="stats-badge">
                    <i class="fas fa-photo-video"></i>
                    {{ images|length }} Images Found
                </span>
            </div>
        </div>
        
        <div class="loading-spinner" id="loadingSpinner">
            <i class="fas fa-spinner"></i>
            <p>Loading images...</p>
        </div>
        
        <div class="gallery" id="gallery">
            {% if images %}
                {% for image in images %}
                    <div class="gallery-item">
                        <img src="{{ url_for('serve_image', filename=image) }}" alt="{{ image }}" loading="lazy">
                        <div class="overlay"></div>
                        <div class="image-actions">
                            <button class="action-btn" title="View Full Size" onclick="viewFullSize('{{ url_for('serve_image', filename=image) }}')">
                                <i class="fas fa-expand"></i>
                            </button>
                            <button class="action-btn" title="Download" onclick="downloadImage('{{ url_for('serve_image', filename=image) }}', '{{ image }}')">
                                <i class="fas fa-download"></i>
                            </button>
                        </div>
                        <div class="caption">
                            <div class="filename">{{ image }}</div>
                            <div class="file-info">
                                <i class="fas fa-file-image"></i>
                                <span>Image File</span>
                            </div>
                        </div>
                    </div>
                {% endfor %}
            {% else %}
                <div class="no-images">
                    <i class="fas fa-folder-open"></i>
                    <h3>No Images Found</h3>
                    <p>The detected folder is empty. Add some images to get started!</p>
                </div>
            {% endif %}
        </div>
    </div>
    
    <script>
        // Image loading animation
        document.addEventListener('DOMContentLoaded', function() {
            const gallery = document.getElementById('gallery');
            const spinner = document.getElementById('loadingSpinner');
            
            // Show loading spinner initially
            if (document.querySelectorAll('.gallery-item').length > 0) {
                spinner.style.display = 'block';
                gallery.style.opacity = '0';
                
                // Hide spinner after images load
                setTimeout(() => {
                    spinner.style.display = 'none';
                    gallery.style.opacity = '1';
                    gallery.style.transition = 'opacity 0.5s ease';
                }, 1000);
            }
        });
        
        // View full size image
        function viewFullSize(imageUrl) {
            const modal = document.createElement('div');
            modal.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.9);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 1000;
                backdrop-filter: blur(5px);
            `;
            
            const img = document.createElement('img');
            img.src = imageUrl;
            img.style.cssText = `
                max-width: 90%;
                max-height: 90%;
                object-fit: contain;
                border-radius: 10px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
            `;
            
            const closeBtn = document.createElement('button');
            closeBtn.innerHTML = '<i class="fas fa-times"></i>';
            closeBtn.style.cssText = `
                position: absolute;
                top: 20px;
                right: 20px;
                background: rgba(255, 255, 255, 0.9);
                border: none;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                font-size: 20px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
            `;
            
            closeBtn.onclick = () => document.body.removeChild(modal);
            modal.onclick = (e) => {
                if (e.target === modal) document.body.removeChild(modal);
            };
            
            modal.appendChild(img);
            modal.appendChild(closeBtn);
            document.body.appendChild(modal);
        }
        
        // Download image
        function downloadImage(imageUrl, filename) {
            const link = document.createElement('a');
            link.href = imageUrl;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    </script>
</body>
</html>
"""
    
    # Write the template to a file
    with open(templates_dir / 'gallery.html', 'w') as f:
        f.write(gallery_template)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5003, debug=True)