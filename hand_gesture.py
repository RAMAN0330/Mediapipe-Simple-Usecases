import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize OpenCV
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            for landmark in hand_landmarks.landmark:
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Draw lines connecting landmarks
            connections = mp_hands.HAND_CONNECTIONS
            for connection in connections:
                x0, y0 = hand_landmarks.landmark[connection[0]].x * w, hand_landmarks.landmark[connection[0]].y * h
                x1, y1 = hand_landmarks.landmark[connection[1]].x * w, hand_landmarks.landmark[connection[1]].y * h
                cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)

    # Display the frame with hand landmarks and connections
    cv2.imshow('Hand Gestures', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()