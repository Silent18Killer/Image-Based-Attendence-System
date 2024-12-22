import cv2
import face_recognition
import os
import numpy as np
from typing import Dict, Tuple, List

def load_registered_faces(directory: str) -> Dict[str, np.ndarray]:
    """
    Load and encode faces from the given directory with proper error handling
    and image format checking.
    """
    face_encodings = {}
    
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist")
        
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(directory, filename)
            try:
                # Read image using cv2 first to check format
                image = cv2.imread(path)
                if image is None:
                    print(f"Warning: Could not read {filename}. Skipping.")
                    continue
                    
                # Convert to RGB (face_recognition expects RGB)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Get face encodings
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    face_encodings[os.path.splitext(filename)[0]] = encodings[0]
                else:
                    print(f"Warning: No face detected in {filename}. Skipping.")
                    
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
                
    if not face_encodings:
        raise ValueError("No valid face encodings could be generated from the provided directory")
        
    return face_encodings

def recognize_faces(
    registered_faces: Dict[str, np.ndarray],
    frame: np.ndarray
) -> Tuple[List[Tuple[int, int, int, int]], List[str]]:
    """
    Recognize faces in a frame using the registered face encodings.
    """
    if frame is None or frame.size == 0:
        raise ValueError("Invalid frame provided")
        
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Convert to RGB (face_recognition expects RGB)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces and get encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(
            list(registered_faces.values()),
            face_encoding,
            tolerance=0.6
        )
        name = "Unknown"
        
        if True in matches:
            matched_idx = matches.index(True)
            name = list(registered_faces.keys())[matched_idx]
            
        names.append(name)
        
    return face_locations, names