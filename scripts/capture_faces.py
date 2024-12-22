import cv2
import os

def capture_face_image():
    """
    Interactive face capture system that allows capturing multiple faces
    with custom names.
    """
    # Ensure the directory exists
    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "original_faces")
    os.makedirs(save_dir, exist_ok=True)
    
    cam = cv2.VideoCapture(0)
    
    while True:
        # Get person's name
        name = input("\nEnter the person's name (or 'quit' to exit): ").strip()
        
        if name.lower() == 'quit':
            break
            
        if not name:
            print("Name cannot be empty! Please try again.")
            continue
            
        # Replace spaces with underscores and remove special characters
        filename = "".join(c for c in name if c.isalnum() or c.isspace())
        filename = filename.replace(" ", "_")
        
        # Create window for this person
        window_name = f"Capture Face Image - {name}"
        cv2.namedWindow(window_name)
        
        print("\nInstructions:")
        print("- Press SPACE to capture the image")
        print("- Press ESC to cancel this capture")
        print("- Make sure the face is clearly visible and well-lit")
        
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Display frame
            cv2.imshow(window_name, frame)
            
            # Handle key presses
            k = cv2.waitKey(1)
            if k % 256 == 27:  # ESC key
                print("Cancelled capture for", name)
                break
            elif k % 256 == 32:  # SPACE key
                # Save the image
                img_path = os.path.join(save_dir, f"{filename}.jpg")
                
                # Check if file already exists
                if os.path.exists(img_path):
                    overwrite = input(f"\nImage for {name} already exists. Overwrite? (yes/no): ").lower()
                    if overwrite != 'yes':
                        print("Skipping capture for", name)
                        break
                
                cv2.imwrite(img_path, frame)
                print(f"\nImage saved as {img_path}")
                break
        
        # Close the window for this person
        cv2.destroyWindow(window_name)
        
        # Ask if user wants to capture another face
        another = input("\nCapture another face? (yes/no): ").lower()
        if another != 'yes':
            break
    
    # Cleanup
    cam.release()
    cv2.destroyAllWindows()
    print("\nFace capture completed!")

if __name__ == "__main__":
    try:
        capture_face_image()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
