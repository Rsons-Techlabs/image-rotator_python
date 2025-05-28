#!/usr/bin/env python3
"""
Image Orientation Fixer with OpenCV-powered Text Detection

This script detects image orientation and rotates images to portrait orientation
while preserving text readability using advanced computer vision techniques.
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class ImageOrientationFixer:
    """Main class for detecting and fixing image orientation."""
    
    def __init__(self, debug=False):
        self.debug = debug
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
    def detect_text_orientation(self, image_path):
        """
        Detect text orientation using multiple OpenCV methods.
        
        Returns:
            tuple: (orientation, confidence, has_text, text_regions_count)
                orientation: 'horizontal', 'vertical', or 'unknown'
                confidence: float between 0 and 1
                has_text: boolean indicating if text was detected
                text_regions_count: number of text regions found
        """
        # Read image with OpenCV
        img = cv2.imread(str(image_path))
        if img is None:
            return 'unknown', 0.0, False, 0
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Morphological text detection
        morph_score, morph_regions = self._morphological_text_detection(gray)
        
        # Method 2: Edge-based line detection
        edge_score, edge_lines = self._edge_based_line_detection(gray)
        
        # Method 3: Contour-based analysis
        contour_score, contour_regions = self._contour_based_analysis(gray)
        
        # Combine results with weighted voting
        total_regions = morph_regions + contour_regions
        has_text = total_regions > 0
        
        if not has_text:
            return 'unknown', 0.0, False, 0
            
        # Weight the scores based on detection strength
        weights = [0.4, 0.3, 0.3]  # morphological, edge, contour
        scores = [morph_score, edge_score, contour_score]
        
        # Calculate weighted average
        weighted_score = sum(w * s for w, s in zip(weights, scores)) / sum(weights)
        
        # Determine orientation based on weighted score
        if weighted_score > 0.3:
            orientation = 'vertical'
        elif weighted_score < -0.3:
            orientation = 'horizontal'
        else:
            orientation = 'unknown'
            
        # Calculate confidence (0-1 scale)
        confidence = min(abs(weighted_score), 1.0)
        
        if self.debug:
            logger.info(f"  Text detection scores: morph={morph_score:.2f}, edge={edge_score:.2f}, contour={contour_score:.2f}")
            logger.info(f"  Weighted score: {weighted_score:.2f}")
            logger.info(f"  Text regions: morph={morph_regions}, contour={contour_regions}, edge_lines={edge_lines}")
            
        return orientation, confidence, has_text, total_regions
    
    def _morphological_text_detection(self, gray):
        """Detect text using morphological operations."""
        # Apply morphological gradient
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # Binarization using OTSU
        _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Connect horizontally oriented regions
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        connected_h = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel_h)
        
        # Connect vertically oriented regions
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
        connected_v = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel_v)
        
        # Find contours for both orientations
        contours_h, _ = cv2.findContours(connected_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_v, _ = cv2.findContours(connected_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and aspect ratio
        h_regions = self._filter_text_contours(contours_h, 'horizontal')
        v_regions = self._filter_text_contours(contours_v, 'vertical')
        
        # Calculate score based on region count and area
        h_score = len(h_regions)
        v_score = len(v_regions)
        
        if h_score + v_score == 0:
            return 0.0, 0
            
        # Normalize score (-1 to 1, negative for horizontal, positive for vertical)
        score = (v_score - h_score) / (v_score + h_score)
        return score, max(h_score, v_score)
    
    def _edge_based_line_detection(self, gray):
        """Detect text orientation using edge detection and line analysis."""
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return 0.0, 0
            
        # Analyze line angles
        horizontal_lines = 0
        vertical_lines = 0
        
        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            
            # Classify lines as horizontal or vertical
            if (angle < 10 or angle > 170) or (80 < angle < 100):
                if angle < 45 or angle > 135:
                    horizontal_lines += 1
                else:
                    vertical_lines += 1
                    
        total_lines = horizontal_lines + vertical_lines
        if total_lines == 0:
            return 0.0, 0
            
        # Calculate score
        score = (vertical_lines - horizontal_lines) / total_lines
        return score, total_lines
    
    def _contour_based_analysis(self, gray):
        """Analyze text orientation using contour detection."""
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and analyze contours
        text_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:  # Filter out noise
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Consider as potential text if reasonable aspect ratio
            if 0.1 < aspect_ratio < 10:
                text_contours.append((contour, aspect_ratio, area))
        
        if not text_contours:
            return 0.0, 0
            
        # Analyze aspect ratios to determine orientation
        horizontal_weight = 0
        vertical_weight = 0
        
        for _, aspect_ratio, area in text_contours:
            weight = np.sqrt(area)  # Weight by area
            
            if aspect_ratio > 2:  # Wide regions (horizontal text)
                horizontal_weight += weight
            elif aspect_ratio < 0.5:  # Tall regions (vertical text)
                vertical_weight += weight
                
        total_weight = horizontal_weight + vertical_weight
        if total_weight == 0:
            return 0.0, len(text_contours)
            
        score = (vertical_weight - horizontal_weight) / total_weight
        return score, len(text_contours)
    
    def _filter_text_contours(self, contours, orientation):
        """Filter contours that look like text regions."""
        text_regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Too small
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Filter based on orientation and aspect ratio
            if orientation == 'horizontal' and aspect_ratio > 1.5:
                text_regions.append(contour)
            elif orientation == 'vertical' and aspect_ratio < 0.7:
                text_regions.append(contour)
                
        return text_regions
    
    def get_image_orientation(self, image_path):
        """
        Determine if image is landscape or portrait.
        
        Returns:
            str: 'landscape' or 'portrait'
        """
        try:
            with Image.open(image_path) as img:
                # Apply EXIF orientation
                img = ImageOps.exif_transpose(img)
                width, height = img.size
                
                return 'landscape' if width > height else 'portrait'
        except Exception as e:
            logger.error(f"Error reading image {image_path}: {e}")
            return 'unknown'
    
    def should_rotate_image(self, image_path, force_rotate=False, custom_angle=None):
        """
        Determine if an image should be rotated based on orientation and text analysis.
        
        Args:
            image_path: Path to the image file
            force_rotate: If True, force rotation regardless of text analysis
            custom_angle: If provided, use this angle instead of default 90 degrees
        
        Returns:
            tuple: (should_rotate, rotation_angle, reason)
        """
        orientation = self.get_image_orientation(image_path)
        
        # If custom angle is provided, always rotate with that angle
        if custom_angle is not None:
            return True, custom_angle, f"Custom rotation angle: {custom_angle}°"
        
        if orientation != 'landscape':
            return False, 0, "Already portrait or unknown orientation"
            
        if force_rotate:
            return True, 90, "Force rotation enabled"
            
        # Analyze text orientation
        text_orientation, confidence, has_text, text_regions = self.detect_text_orientation(image_path)
        
        if self.debug:
            logger.info(f"Text orientation: {text_orientation}")
            logger.info(f"Text confidence: {confidence:.2f}")
            logger.info(f"Has text: {has_text}")
            logger.info(f"Text regions detected: {text_regions}")
        
        # Decision logic based on text analysis
        if not has_text:
            return True, 90, "No text detected"
            
        if text_orientation == 'vertical' and confidence > 0.6:
            return False, 0, f"High confidence vertical text (confidence: {confidence:.2f})"
            
        # Check aspect ratio for documents
        try:
            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img)
                width, height = img.size
                aspect_ratio = width / height
                
                if aspect_ratio > 1.5 and confidence > 0.3:
                    return False, 0, f"Likely document with medium confidence text (confidence: {confidence:.2f})"
        except:
            pass
            
        if confidence < 0.3 and has_text:
            return False, 0, f"Low confidence text detection, being conservative (confidence: {confidence:.2f})"
            
        return True, 90, f"Horizontal text or low confidence vertical text (confidence: {confidence:.2f})"
    
    def rotate_image(self, input_path, output_path, angle):
        """Rotate image by specified angle and save."""
        try:
            with Image.open(input_path) as img:
                # Apply EXIF orientation first
                img = ImageOps.exif_transpose(img)
                
                # Rotate the image
                rotated = img.rotate(angle, expand=True)
                
                # Save with high quality
                save_kwargs = {'quality': 95, 'optimize': True}
                if img.format == 'JPEG':
                    save_kwargs['format'] = 'JPEG'
                elif img.format == 'PNG':
                    save_kwargs['format'] = 'PNG'
                    
                rotated.save(output_path, **save_kwargs)
                return True
                
        except Exception as e:
            logger.error(f"Error rotating image {input_path}: {e}")
            return False
    
    def copy_image(self, input_path, output_path):
        """Copy image without rotation."""
        try:
            with Image.open(input_path) as img:
                # Apply EXIF orientation
                img = ImageOps.exif_transpose(img)
                
                # Save with high quality
                save_kwargs = {'quality': 95, 'optimize': True}
                if img.format == 'JPEG':
                    save_kwargs['format'] = 'JPEG'
                elif img.format == 'PNG':
                    save_kwargs['format'] = 'PNG'
                    
                img.save(output_path, **save_kwargs)
                return True
                
        except Exception as e:
            logger.error(f"Error copying image {input_path}: {e}")
            return False
    
    def process_directory(self, input_dir, output_dir=None, overwrite=False, force_rotate=False, recursive=True, custom_angle=None):
        """Process all images in a directory and optionally subdirectories."""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return
            
        # When processing recursively, we always save to the same folder path
        if recursive:
            overwrite = True
            output_dir = None
            
        # Set up output directory
        if output_dir:
            output_path = Path(output_dir)
        elif overwrite:
            output_path = input_path
        else:
            output_path = input_path / "portrait_fixed"
            
        if not overwrite:
            output_path.mkdir(exist_ok=True)
        
        # Find all image files (recursively if enabled)
        image_files = []
        if recursive:
            # Recursively find all image files in subdirectories
            for ext in self.supported_formats:
                image_files.extend(input_path.rglob(f"*{ext}"))
                image_files.extend(input_path.rglob(f"*{ext.upper()}"))
        else:
            # Only process files in the current directory
            for ext in self.supported_formats:
                image_files.extend(input_path.glob(f"*{ext}"))
                image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            logger.info("No supported image files found.")
            return
            
        # Process images
        logger.info(f"Processing images in: {input_path}")
        if recursive:
            logger.info("Recursive processing: Enabled")
            logger.info("Saving to: Same folder as source images")
        else:
            logger.info(f"Output directory: {output_path}")
        logger.info(f"Text-aware rotation: {'Disabled' if force_rotate else 'Enabled'}")
        if custom_angle is not None:
            logger.info(f"Custom rotation angle: {custom_angle}°")
        logger.info(f"Debug mode: {'Enabled' if self.debug else 'Disabled'}")
        logger.info("-" * 50)
        
        stats = {
            'total': 0,
            'rotated': 0,
            'preserved': 0,
            'already_portrait': 0,
            'errors': 0
        }
        
        for image_file in sorted(image_files):
            # Show relative path for better readability
            relative_path = image_file.relative_to(input_path)
            logger.info(f"\nProcessing: {relative_path}")
            stats['total'] += 1
            
            # Determine output path
            if recursive or overwrite:
                out_path = image_file  # Save to same location
            else:
                out_path = output_path / image_file.name
            
            # Check orientation
            orientation = self.get_image_orientation(image_file)
            logger.info(f"Orientation: {orientation}")
            
            # If custom angle is specified, skip portrait check and always process
            if custom_angle is None and orientation == 'portrait':
                logger.info("Already portrait orientation")
                if not (recursive or overwrite):
                    if self.copy_image(image_file, out_path):
                        logger.info(f"Copied: {out_path}")
                    else:
                        stats['errors'] += 1
                        continue
                stats['already_portrait'] += 1
                continue
            
            # Determine if rotation is needed
            should_rotate, angle, reason = self.should_rotate_image(image_file, force_rotate, custom_angle)
            
            if should_rotate:
                logger.info(f"Rotation needed: {angle}°")
                if self.rotate_image(image_file, out_path, angle):
                    logger.info(f"Rotated {angle}° and saved: {relative_path}")
                    stats['rotated'] += 1
                else:
                    stats['errors'] += 1
            else:
                logger.info(f"Preserving orientation: {reason}")
                if recursive or overwrite:
                    logger.info("No rotation needed")
                else:
                    if self.copy_image(image_file, out_path):
                        logger.info(f"Copied: {out_path}")
                    else:
                        stats['errors'] += 1
                        continue
                stats['preserved'] += 1
        
        # Print summary
        logger.info("\n" + "=" * 50)
        logger.info("Processing complete!")
        logger.info(f"Total images processed: {stats['total']}")
        logger.info(f"Images rotated: {stats['rotated']}")
        logger.info(f"Images preserved for text readability: {stats['preserved']}")
        logger.info(f"Images already portrait: {stats['already_portrait']}")
        if stats['errors'] > 0:
            logger.info(f"Errors encountered: {stats['errors']}")


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fix image orientation using OpenCV-powered text detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/images
  %(prog)s /path/to/images -o /path/to/output
  %(prog)s /path/to/images --force-rotate
  %(prog)s /path/to/images --debug --overwrite
  %(prog)s /path/to/images --recursive
  %(prog)s /path/to/images --no-recursive -o /path/to/output
  %(prog)s /path/to/images --angle 180
  %(prog)s /path/to/images --angle -90 --overwrite
        """
    )
    
    parser.add_argument('input_dir', help='Directory containing images to process')
    parser.add_argument('-o', '--output', help='Output directory (default: input_dir/portrait_fixed, ignored when --recursive is used)')
    parser.add_argument('--overwrite', action='store_true', 
                       help='Overwrite original files instead of creating copies')
    parser.add_argument('--force-rotate', action='store_true',
                       help='Rotate all landscape images regardless of text orientation')
    parser.add_argument('--angle', type=float, metavar='DEGREES',
                       help='Rotate all images by this specific angle in degrees (e.g., 90, -90, 180). When specified, ignores text analysis and orientation checks.')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode for detailed text analysis output')
    parser.add_argument('--recursive', action='store_true', default=True,
                       help='Process subdirectories recursively and save to same folder (default: enabled)')
    parser.add_argument('--no-recursive', dest='recursive', action='store_false',
                       help='Disable recursive processing, only process files in the specified directory')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Validate angle if provided
    if args.angle is not None:
        if not -360 <= args.angle <= 360:
            logger.error("Angle must be between -360 and 360 degrees")
            sys.exit(1)
    
    # Create fixer instance and process
    fixer = ImageOrientationFixer(debug=args.debug)
    fixer.process_directory(
        input_dir=args.input_dir,
        output_dir=args.output,
        overwrite=args.overwrite,
        force_rotate=args.force_rotate,
        recursive=args.recursive,
        custom_angle=args.angle
    )


if __name__ == "__main__":
    main() 