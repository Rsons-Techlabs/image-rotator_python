# Quick Usage Guide

## Installation

1. **Activate the virtual environment:**

   ```bash
   source venv/bin/activate
   ```

2. **Install dependencies (if not already installed):**
   ```bash
   pip install -r requirements.txt
   ```

## Basic Usage

### 1. Process a directory with intelligent text detection

```bash
python3 image_orientation_fixer.py /path/to/your/images
```

- Analyzes text orientation using OpenCV
- Only rotates images where text won't become unreadable
- Saves results to `/path/to/your/images/portrait_fixed/`

### 2. Force rotate all landscape images

```bash
python3 image_orientation_fixer.py /path/to/your/images --force-rotate
```

- Rotates ALL landscape images to portrait
- Ignores text orientation analysis

### 3. Specify custom output directory

```bash
python3 image_orientation_fixer.py /path/to/your/images -o /path/to/output
```

### 4. Debug mode (see detailed analysis)

```bash
python3 image_orientation_fixer.py /path/to/your/images --debug
```

### 5. Overwrite original files

```bash
python3 image_orientation_fixer.py /path/to/your/images --overwrite
```

## Test with Sample Image

```bash
# Test with the included sample image
source venv/bin/activate
python3 image_orientation_fixer.py test_images --debug
```

## Programmatic Usage

```python
from image_orientation_fixer import ImageOrientationFixer

# Create fixer instance
fixer = ImageOrientationFixer(debug=True)

# Process a directory
fixer.process_directory('my_images', output_dir='fixed_images')

# Analyze a single image
orientation = fixer.get_image_orientation('image.jpg')
text_orientation, confidence, has_text, regions = fixer.detect_text_orientation('image.jpg')
should_rotate, angle, reason = fixer.should_rotate_image('image.jpg')
```

## Supported Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## How It Works

1. **EXIF Analysis**: Reads and applies EXIF orientation data
2. **Text Detection**: Uses OpenCV for advanced text orientation analysis:
   - Morphological operations for text region detection
   - Edge detection and line analysis
   - Contour-based text shape analysis
3. **Smart Decision Making**:
   - High confidence vertical text → preserve orientation
   - Low confidence or no text → rotate landscape to portrait
   - Document-like aspect ratios → be conservative
4. **Quality Preservation**: Saves with 95% quality and optimization
