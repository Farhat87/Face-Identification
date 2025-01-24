import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0) # Start the video capture 

scale_factor = 0.5  #Scaling down the frame size by half

frame_skip = 2
frame_count = 0

while True:
    ret, frame = cap.read()  # Capture frame from the webcam

    if not ret:
        print("Failed to grab frame.")
        break


    if frame_count % frame_skip == 0:
        small_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR) # Resizing the frame 

        # Converting the smaller frame to grayscale for  detection
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

       
        faces = [(int(x / scale_factor), int(y / scale_factor), int(w / scale_factor), int(h / scale_factor)) for (x, y, w, h) in faces]

   
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

  
    num_faces = len(faces)  # Displaying number of faces detected
    cv2.putText(frame, f'Faces detected: {num_faces}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
cv.waitKey(0)
