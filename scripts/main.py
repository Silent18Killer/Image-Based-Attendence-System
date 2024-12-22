import cv2
from face_utils import load_registered_faces, recognize_faces
from datetime import datetime
import csv
import os
import sys

def mark_attendance(name: str) -> None:
    """
    Mark attendance with timestamp in CSV file.
    """
    now = datetime.now()
    time_str = now.strftime("%H:%M:%S")
    date_str = now.strftime("%Y-%m-%d")
    
    # Get the path to attendance.csv in the data folder (one level up from scripts)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    attendance_file = os.path.join(script_dir, "..", "data", "attendance.csv")
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(attendance_file), exist_ok=True)
    
    # Create file with headers if it doesn't exist
    if not os.path.exists(attendance_file):
        with open(attendance_file, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Date", "Time"])
    
    with open(attendance_file, "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, date_str, time_str])

def main():
    try:
        # Get the correct path to registered_faces directory (one level up from scripts)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        faces_dir = os.path.join(script_dir, "..", "data", "registered_faces")
        
        # Ensure the faces directory exists
        if not os.path.exists(faces_dir):
            print(f"Creating directory: {faces_dir}")
            os.makedirs(faces_dir, exist_ok=True)
        
        # Load registered faces
        print("Loading registered faces...")
        registered_faces = load_registered_faces(faces_dir)
        print(f"Loaded {len(registered_faces)} registered faces")
        
        # Initialize video capture
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            raise RuntimeError("Could not access the camera")
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            try:
                # Process frame
                face_locations, names = recognize_faces(registered_faces, frame)
                
                # Draw results
                for (top, right, bottom, left), name in zip(face_locations, names):
                    # Scale back up face locations
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    
                    # Draw box
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # Draw label
                    cv2.putText(
                        frame,
                        name,
                        (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.6,
                        (255, 255, 255),
                        1
                    )
                    
                    # Mark attendance for recognized faces
                    if name != "Unknown":
                        mark_attendance(name)
                
                # Display results
                cv2.imshow('Attendance System', frame)
                
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                continue
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        video_capture.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()