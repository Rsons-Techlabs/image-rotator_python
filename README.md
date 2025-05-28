# Image Orientation Fixer

This script detects image orientation and rotates images to portrait orientation if needed. It processes images in a specified directory and saves the corrected versions. **Now uses OpenCV for advanced computer vision-based text detection with significantly improved accuracy.**

## Features

- **OpenCV-Powered Text Detection**: Uses advanced computer vision algorithms for robust text analysis
- **Multi-Method Analysis**: Combines morphological operations, edge detection, and contour analysis
- **High Accuracy**: Much more reliable text orientation detection than simple gradient methods
- **Custom Rotation Angles**: Rotate images by any specified angle (e.g., 90°, -90°, 180°, 45°)
- Automatically detects image orientation using EXIF data and dimensions
- Rotates landscape images to portrait orientation (when text allows)
- Preserves image quality during rotation
- Supports multiple image formats (JPG, PNG, BMP, TIFF)
- Option to overwrite original files or create copies
- Batch processing of entire directories
- Smart detection prevents rotating images with vertical text that would become unreadable

## Setup

1. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage (Text-Aware)

Process all images in a directory with intelligent text detection:

```bash
python image_orientation_fixer.py /path/to/your/images
```

### Specify Output Directory

Save processed images to a specific directory:

```bash
python image_orientation_fixer.py /path/to/your/images -o /path/to/output
```

### Force Rotation (Ignore Text Orientation)

Rotate all landscape images regardless of text orientation:

```bash
python image_orientation_fixer.py /path/to/your/images --force-rotate
```

### Debug Mode

View detailed text analysis information to understand detection decisions:

```bash
python image_orientation_fixer.py /path/to/your/images --debug
```

### Overwrite Original Files

Modify the original files instead of creating copies:

```bash
python image_orientation_fixer.py /path/to/your/images --overwrite
```

### Custom Rotation Angle

Rotate all images by a specific angle (ignores text analysis and orientation checks):

```bash
# Rotate all images 90 degrees clockwise
python image_orientation_fixer.py /path/to/your/images --angle 90

# Rotate all images 90 degrees counter-clockwise
python image_orientation_fixer.py /path/to/your/images --angle -90

# Rotate all images 180 degrees (upside down)
python image_orientation_fixer.py /path/to/your/images --angle 180

# Rotate by 45 degrees and overwrite original files
python image_orientation_fixer.py /path/to/your/images --angle 45 --overwrite

# Custom angle with debug output
python image_orientation_fixer.py /path/to/your/images --angle 135 --debug
```

### Help

View all available options:

```bash
python image_orientation_fixer.py --help
```

## How It Works

1. **EXIF Analysis**: The script first checks the EXIF orientation data in each image
2. **Advanced Text Orientation Detection**: Uses multiple analysis methods:
   - **Edge Detection**: Analyzes horizontal vs vertical edge patterns
   - **Line Pattern Detection**: Identifies consistent text line patterns
   - **Variance Analysis**: Measures content variance in different directions
   - **Confidence Scoring**: Combines all methods for reliable detection
3. **Smart Rotation Logic with Confidence Thresholds**:
   - **High confidence vertical text** (>60%) → preserve orientation
   - **Medium confidence + wide aspect ratio** → likely document, preserve
   - **Low confidence text** (<30%) → be conservative, preserve if text detected
   - **No text detected** → rotate landscape to portrait normally
4. **Quality Preservation**: Images are saved with 95% quality and optimization

## Text Detection Algorithm

The OpenCV-powered algorithm uses three sophisticated computer vision methods:

### 1. Morphological Text Detection

- Uses morphological gradient operations to enhance text edges
- Applies OTSU thresholding for optimal binarization
- Morphological closing to connect text components
- Filters regions by size and aspect ratio for text-like characteristics

### 2. Edge-Based Line Detection

- Canny edge detection for precise edge identification
- Hough Line Transform to detect consistent line patterns
- Analyzes line angles to determine text orientation
- Weights results by line length and consistency

### 3. Contour-Based Analysis

- Adaptive thresholding for robust text segmentation
- Contour detection and filtering for text-like shapes
- Aspect ratio analysis of detected regions
- Area-weighted confidence scoring

### Advanced Confidence System

- **Weighted Voting**: Each method contributes based on detection strength
- **Aspect Ratio Consistency**: Boosts confidence when orientation matches image dimensions
- **Multi-Method Agreement**: Higher confidence when methods agree
- **Text Presence Validation**: Ensures actual text is detected before making decisions

## Supported Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## Requirements

- Python 3.6+
- Pillow (PIL) library
- NumPy library
- **OpenCV library** (for advanced text detection)

## Example Output

### Text-Aware Processing (Default)

```
Processing images in: /Users/example/photos
Output directory: /Users/example/photos/portrait_fixed
Text-aware rotation: Enabled
Debug mode: Disabled
--------------------------------------------------

Processing: document_landscape.jpg
Orientation: landscape
Text orientation: vertical
Text confidence: 0.92
Has text: True
Text regions detected: 15
Preserving orientation: High confidence vertical text (confidence: 0.92)
No rotation needed
Copied: /Users/example/photos/portrait_fixed/document_landscape.jpg

Processing: photo_landscape.jpg
Orientation: landscape
Text orientation: horizontal
Text confidence: 0.25
Has text: True
Text regions detected: 3
Rotation needed: 90°
Rotated 90° and saved: /Users/example/photos/portrait_fixed/photo_landscape.jpg

Processing: screenshot_landscape.jpg
Orientation: landscape
Text orientation: vertical
Text confidence: 0.78
Has text: True
Text regions detected: 8
Preserving orientation: High confidence vertical text (confidence: 0.78)
No rotation needed
Copied: /Users/example/photos/portrait_fixed/screenshot_landscape.jpg

Processing: nature_photo.jpg
Orientation: landscape
Text orientation: unknown
Text confidence: 0.05
Has text: False
Text regions detected: 0
Rotation needed: 90°
Rotated 90° and saved: /Users/example/photos/portrait_fixed/nature_photo.jpg

==================================================
Processing complete!
Total images processed: 4
Images rotated: 2
Images preserved for text readability: 2
Images already portrait: 0
```

### Custom Angle Processing

```
Processing images in: /Users/example/photos
Recursive processing: Enabled
Saving to: Same folder as source images
Text-aware rotation: Enabled
Custom rotation angle: 180°
Debug mode: Disabled
--------------------------------------------------

Processing: document_landscape.jpg
Orientation: landscape
Rotation needed: 180°
Rotated 180° and saved: document_landscape.jpg

Processing: photo_portrait.jpg
Orientation: portrait
Rotation needed: 180°
Rotated 180° and saved: photo_portrait.jpg

Processing: screenshot_landscape.jpg
Orientation: landscape
Rotation needed: 180°
Rotated 180° and saved: screenshot_landscape.jpg

==================================================
Processing complete!
Total images processed: 3
Images rotated: 3
Images preserved for text readability: 0
Images already portrait: 0
```

## Command Line Options

- `input_dir`: Directory containing images to process (required)
- `-o, --output`: Output directory (optional, defaults to 'portrait_fixed' subdirectory)
- `--overwrite`: Overwrite original files instead of creating copies
- `--force-rotate`: Disable text-aware rotation and force rotate all landscape images
- `--angle DEGREES`: Rotate all images by a specific angle in degrees (e.g., 90, -90, 180, 45.5). When specified, ignores text analysis and orientation checks. Accepts values between -360 and 360 degrees.
- `--debug`: View detailed text analysis information
- `--recursive`: Process subdirectories recursively (enabled by default)
- `--no-recursive`: Disable recursive processing, only process files in the specified directory
