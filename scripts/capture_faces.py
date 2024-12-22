import cv2

def capture_face_image(name):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capture Face Image")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("Capture Face Image", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC key
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:  # SPACE key
            # Save the image
            img_name = f"data/original_faces/{name}.jpg"
            cv2.imwrite(img_name, frame)
            print(f"{img_name} saved!")
            break

    cam.release()
    cv2.destroyAllWindows()

# Capture image for "John_Doe"
capture_face_image("John_Doe")
