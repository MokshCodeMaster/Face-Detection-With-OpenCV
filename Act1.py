import cv2
#Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Start the Video Capture from the default webcam (0)
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could Not Open the Camera")
    exit()
while True:
    # Capture Frame-By-Frame
    ret, frame = cap.read()
    # If frame is read correctly, ret will be true
    if not ret:
        print("Error: Failed to Capture image")
        break
    # convert the frame to grayscale (Face dectection works better on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))
    # Draw rectangles Around the Faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) # Blue Rectangle with thickness 2
    # Display the resulting frame
    cv2.imshow('Face Detection - Press q to quit', frame)
    #Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the Video Capture Object  and Close any open windows
cap.release()
cv2.destroyAllWindows()

