import cv2
import numpy as np
import os
from PIL import Image

def verify_and_convert_image(input_path: str, output_path: str) -> bool:
    """
    Verify and convert an image to the correct format for face recognition.
    Returns True if successful, False otherwise.
    """
    try:
        # Open image with PIL first (better format handling)
        with Image.open(input_path) as img:
            # Convert to RGB mode
            img = img.convert('RGB')
            # Save as JPEG
            img.save(output_path, 'JPEG', quality=95)
        
        # Verify the saved image
        test_img = cv2.imread(output_path)
        if test_img is None:
            return False
            
        return True
    except Exception as e:
        print(f"Error processing image {input_path}: {str(e)}")
        return False

def process_directory(input_dir: str, output_dir: str):
    """
    Process all images in a directory and convert them to the correct format.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    failed_count = 0
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.jpg')
            
            if verify_and_convert_image(input_path, output_path):
                print(f"Successfully converted: {filename}")
                success_count += 1
            else:
                print(f"Failed to convert: {filename}")
                failed_count += 1
    
    print(f"\nProcessing complete:")
    print(f"Successfully converted: {success_count} images")
    print(f"Failed to convert: {failed_count} images")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "..", "data", "original_faces")
    output_dir = os.path.join(script_dir, "..", "data", "registered_faces")
    
    process_directory(input_dir, output_dir)