#!/usr/bin/env python3
"""
Example usage of the Image Orientation Fixer
"""

from image_orientation_fixer import ImageOrientationFixer

def main():
    """Demonstrate basic usage patterns."""
    
    # Create an instance of the fixer
    fixer = ImageOrientationFixer(debug=True)
    
    print("Image Orientation Fixer - Example Usage")
    print("=" * 50)
    
    # Example 1: Process a directory with default settings
    print("\n1. Basic directory processing:")
    print("   fixer.process_directory('test_images')")
    print("   - Processes all images in test_images/")
    print("   - Saves results to test_images/portrait_fixed/")
    print("   - Uses intelligent text detection")
    
    # Example 2: Process with custom output directory
    print("\n2. Custom output directory:")
    print("   fixer.process_directory('test_images', output_dir='my_output')")
    print("   - Saves results to my_output/")
    
    # Example 3: Force rotation (ignore text)
    print("\n3. Force rotation (ignore text orientation):")
    print("   fixer.process_directory('test_images', force_rotate=True)")
    print("   - Rotates ALL landscape images to portrait")
    print("   - Ignores text orientation analysis")
    
    # Example 4: Overwrite original files
    print("\n4. Overwrite original files:")
    print("   fixer.process_directory('test_images', overwrite=True)")
    print("   - Modifies original files in place")
    print("   - No backup copies created")
    
    # Example 5: Single image analysis
    print("\n5. Analyze a single image:")
    image_path = "test_images/20250505_223957793_iOS_1747175885253.jpeg"
    
    orientation = fixer.get_image_orientation(image_path)
    text_orientation, confidence, has_text, regions = fixer.detect_text_orientation(image_path)
    should_rotate, angle, reason = fixer.should_rotate_image(image_path)
    
    print(f"   Image: {image_path}")
    print(f"   Orientation: {orientation}")
    print(f"   Text detected: {has_text}")
    print(f"   Text orientation: {text_orientation}")
    print(f"   Confidence: {confidence:.2f}")
    print(f"   Should rotate: {should_rotate}")
    print(f"   Reason: {reason}")
    
    print("\n" + "=" * 50)
    print("To run the actual processing, use:")
    print("python image_orientation_fixer.py test_images")
    print("python image_orientation_fixer.py test_images --debug")
    print("python image_orientation_fixer.py test_images --force-rotate")

if __name__ == "__main__":
    main() 