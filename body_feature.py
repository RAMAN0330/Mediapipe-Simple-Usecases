import cv2
import mediapipe as mp

# Initialize MediaPipe Pose, Face, and Hands models
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

pose = mp_pose.Pose()
face = mp_face.FaceMesh()
hands = mp_hands.Hands()

# Define the connections between pose landmarks
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 for default camera, or provide the camera's index

while cap.isOpened():
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    results_pose = pose.process(frame_rgb)
    results_face = face.process(frame_rgb)
    results_hands = hands.process(frame_rgb)

    if results_pose.pose_landmarks:
        # Label body landmarks with white integers
        for idx, landmark in enumerate(results_pose.pose_landmarks.landmark):
            h, w, _ = frame.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.putText(frame, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        for connection in POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            start_landmark = results_pose.pose_landmarks.landmark[start_idx]
            end_landmark = results_pose.pose_landmarks.landmark[end_idx]
            start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
            end_point = (int(end_landmark.x * w), int(end_landmark.y * h))
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

    if results_face.multi_face_landmarks:
        # Label face landmarks with white integers
        for face_landmarks in results_face.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.putText(frame, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if results_hands.multi_hand_landmarks:
        # Label hand landmarks with white integers and draw lines
        for hand_landmarks in results_hands.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.putText(frame, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Define the connections between hand landmarks
            HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS
            for connection in HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                start_landmark = hand_landmarks.landmark[start_idx]
                end_landmark = hand_landmarks.landmark[end_idx]
                start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
                end_point = (int(end_landmark.x * w), int(end_landmark.y * h))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

    # Display the frame with pose, face, and hand landmarks
    cv2.imshow('Pose, Face, and Hand Estimation', frame)

    # Exit the program when 'Esc' is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()