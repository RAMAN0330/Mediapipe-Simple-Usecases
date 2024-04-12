import cv2
import mediapipe as mp

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

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
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Draw pose landmarks on the frame
        mp.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame with pose estimation
    cv2.imshow('Pose Estimation', frame)

    # Exit the program when 'Esc' is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()