import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import pandas as pd

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Initialize OpenCV for webcam feed
cap = cv2.VideoCapture(0)

# Create an empty DataFrame to store the data
data = []


# Function to calculate the Euclidean distance between two points
def calculate_distance(point1, point2):
    return dist.euclidean(point1, point2)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect facial landmarks
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmark_points = []
            for landmark in face_landmarks.landmark:
                h, w, c = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                landmark_points.append((x, y))
                cv2.putText(frame, f'{landmark_points.index((x, y))}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (0, 0, 255), 1)

            # Calculate and display the lengths and area
            forehead_length = calculate_distance(landmark_points[10], landmark_points[151])
            jaw_length = calculate_distance(landmark_points[152], landmark_points[14])
            lip_length = calculate_distance(landmark_points[57], landmark_points[287])
            cheek_length = calculate_distance(landmark_points[234], landmark_points[454])

            # Calculate the area of the face using a convex hull
            hull = cv2.convexHull(np.array(landmark_points))
            face_area = cv2.contourArea(hull)

            # Display measurements on the frame
            cv2.putText(frame, f'Forehead Length: {forehead_length:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
            cv2.putText(frame, f'Jaw Length: {jaw_length:.2f}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, f'Lip Length: {lip_length:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, f'Cheek Length: {cheek_length:.2f}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
            cv2.putText(frame, f'Face Area: {face_area:.2f}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Append the data to the DataFrame
            data.append({
                'Forehead Length': forehead_length,
                'Jaw Length': jaw_length,
                'Lip Length': lip_length,
                'Cheek Length': cheek_length,
                'Face Area': face_area
            })

    # Display the frame
    cv2.imshow('Face Analysis', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

# # Store the landmark points as a variable
# landmark_points_variable = landmark_points
# print("Landmark Points:")
# for idx, point in enumerate(landmark_points_variable):
#     print(f"Point {idx}: {point}")


# Create a Pandas DataFrame from the collected data
df = pd.DataFrame(data)

# Print or save the DataFrame as needed
print(df)
# You can save the DataFrame to a CSV file using df.to_csv('face_data.csv', index=False)