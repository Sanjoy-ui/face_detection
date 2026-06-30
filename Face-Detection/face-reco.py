import cv2
import time
import os

# Load Haar cascades
cascade_path = cv2.data.haarcascades
face_cascade = cv2.CascadeClassifier(cascade_path + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cascade_path + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cascade_path + 'haarcascade_smile.xml')

# Check if the cascades loaded successfully
if face_cascade.empty() or eye_cascade.empty() or smile_cascade.empty():
    print("Error loading Haar cascades. Ensure OpenCV is properly installed.")
    exit()

# State variables for toggles
blur_faces = False
detect_eyes = False
detect_smiles = False

# Open webcam (0 = default camera)
webcam = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not webcam.isOpened():
    print("Cannot access webcam.")
    exit()

# Variables for FPS calculation
prev_time = 0

print("Controls:")
print("  's' - Save snapshot")
print("  'b' - Toggle face blur")
print("  'e' - Toggle eye detection")
print("  'm' - Toggle smile detection")
print("  'ESC' - Quit")

while True:
    # Capture frame-by-frame
    ret, img = webcam.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        if blur_faces:
            # Extract the face region, blur it, and place it back
            face_roi = img[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
            img[y:y+h, x:x+w] = blurred_face
        else:
            # Draw rectangle around the face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Eye Detection
        if detect_eyes and not blur_faces:
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

        # Smile Detection
        if detect_smiles and not blur_faces:
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25))
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
                cv2.putText(roi_color, "Smile", (sx, sy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # UI Overlay: FPS and Face Count
    cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Faces: {len(faces)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # UI Overlay: Status
    status_text = f"Blur (b): {'ON' if blur_faces else 'OFF'} | Eyes (e): {'ON' if detect_eyes else 'OFF'} | Smile (m): {'ON' if detect_smiles else 'OFF'}"
    cv2.putText(img, status_text, (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Show the resulting frame
    cv2.imshow("Face Detection", img)

    # Keyboard handling
    key = cv2.waitKey(10) & 0xFF
    if key == 27:  # ESC key
        break
    elif key == ord('s'):
        filename = f"snapshot_{int(time.time())}.jpg"
        cv2.imwrite(filename, img)
        print(f"Snapshot saved as {filename}")
    elif key == ord('b'):
        blur_faces = not blur_faces
    elif key == ord('e'):
        detect_eyes = not detect_eyes
    elif key == ord('m'):
        detect_smiles = not detect_smiles

# Release resources
webcam.release()
cv2.destroyAllWindows()
