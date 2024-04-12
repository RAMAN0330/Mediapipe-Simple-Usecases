import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import pandas as pd
import tkinter as tk
from tkinter import filedialog

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Create an empty DataFrame to store the data
data = []

# Global variable to track if a front face has been captured
front_face_captured = False

# Function to calculate the Euclidean distance between two points
def calculate_distance(point1, point2):
    return dist.euclidean(point1, point2)

# Function to process an image file
def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print("Unable to load the image.")
        return

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

            # Calculate and display the lengths and area
            forehead_length = calculate_distance(landmark_points[10], landmark_points[151])
            jaw_length = calculate_distance(landmark_points[152], landmark_points[10])
            cheek_length = calculate_distance(landmark_points[234], landmark_points[454])

            # Calculate the area of the face using a convex hull
            hull = cv2.convexHull(np.array(landmark_points))
            face_area = cv2.contourArea(hull)

            # Append the data to the DataFrame
            data.append({
                'Forehead Length': forehead_length,
                'Jaw Length': jaw_length,
                'Cheek Length': cheek_length,
                'Face Area': face_area
            })

            # Draw the face mesh on the image
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1))

            # Display the frame with landmarks and measurements
            cv2.imshow('Face Analysis', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# Function to capture using the camera
def capture_camera():
    global front_face_captured  # Use the global variable

    # Initialize OpenCV for webcam feed
    cap = cv2.VideoCapture(0)

    while cap.isOpened() and not front_face_captured:
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

                # Check if the face is frontal
                if abs(landmark_points[10][1] - landmark_points[152][1]) < 10:
                    front_face_captured = True

                if front_face_captured:
                    break

        # Draw the face mesh on the frame
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1))

        # Display the frame
        cv2.imshow('Face Analysis', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    # Release the webcam and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

# Ask the user for input
choice = input("Choose an option (1: Upload Image, 2: Capture using Camera): ")

if choice == '1':
    # Open a file dialog to choose an image
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename()
    root.destroy()

    if image_path:
        process_image(image_path)
    else:
        print("No image selected.")
elif choice == '2':
    capture_camera()

# Create a Pandas DataFrame from the collected data
df = pd.DataFrame(data)

# Print or save the DataFrame as needed
print(df)
# You can save the DataFrame to a CSV file using df.to_csv('face_data.csv', index=False)