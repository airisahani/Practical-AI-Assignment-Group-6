import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from importlib.metadata import version
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, 
                    enable_segmentation=False, smooth_segmentation=True, 
                    min_detection_confidence=0.5, min_tracking_confidence=0.8)

# Helper function to calculate angle
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # Endpoint

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Streamlit UI
def main():
    st.title("Exercise Monitoring App")
    st.sidebar.title("Exercise Options")

    exercise_choice = st.sidebar.selectbox(
        "Select an exercise", ["Push-Ups", "Squats", "Bicep Curls", "Pull-Ups"]
    )

    start_button = st.sidebar.button("Start Monitoring")

    st.write(f"### Selected Exercise: {exercise_choice}")

    if start_button:
        st.write("Initializing webcam...")
        run_exercise_monitoring(exercise_choice)

# Real-time monitoring function
def run_exercise_monitoring(exercise):
    # Try different camera indices if the default (0) doesn't work
    for camera_index in [0, 1, 2]:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            break
    
    if not cap.isOpened():
        st.error("Error: Could not access the webcam. Please check if:")
        st.error("1. Your webcam is properly connected")
        st.error("2. You've granted browser/system permission to access the webcam")
        st.error("3. No other application is currently using the webcam")
        return

    # Set counter and status variables
    counter = 0
    status = False

    stframe = st.empty()

    # Add stop button and counter for tracking exercise duration
    stop_button = st.button("Stop Exercise")
    start_time = time.time()
    
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Unable to read from the webcam.")
            break

        # Convert the BGR frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect pose landmarks
        results = pose.process(frame_rgb)
        landmarks = results.pose_landmarks

        if landmarks:
            landmarks_array = [
                [lm.x * frame.shape[1], lm.y * frame.shape[0]] for lm in landmarks.landmark
            ]

            if exercise == "Push-Ups":
                angle = calculate_angle(landmarks_array[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                        landmarks_array[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                        landmarks_array[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])

                # Determine status and count reps
                if angle > 160:
                    status = True
                if status and angle < 90:
                    counter += 1
                    status = False

                cv2.putText(frame, f"Push-Ups: {counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif exercise == "Squats":
                # Calculate angle between hip, knee, and ankle
                angle = calculate_angle(landmarks_array[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                     landmarks_array[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                     landmarks_array[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

                # Determine status and count reps
                if angle > 160:
                    status = True
                if status and angle < 90:
                    counter += 1
                    status = False

                cv2.putText(frame, f"Squats: {counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif exercise == "Bicep Curls":
                # Calculate angle between shoulder, elbow, and wrist
                angle = calculate_angle(landmarks_array[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                     landmarks_array[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                     landmarks_array[mp_pose.PoseLandmark.RIGHT_WRIST.value])

                # Determine status and count reps
                if angle > 160:
                    status = True
                if status and angle < 60:
                    counter += 1
                    status = False

                cv2.putText(frame, f"Bicep Curls: {counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif exercise == "Pull-Ups":
                # Calculate angle between wrist, shoulder, and hip
                angle = calculate_angle(landmarks_array[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                     landmarks_array[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                     landmarks_array[mp_pose.PoseLandmark.RIGHT_HIP.value])

                # Determine status and count reps
                if angle < 60:
                    status = True
                if status and angle > 120:
                    counter += 1
                    status = False

                cv2.putText(frame, f"Pull-Ups: {counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Add visual feedback for form
            cv2.putText(frame, f"Angle: {int(angle)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Draw status indicator
            status_text = "Up" if status else "Down"
            cv2.putText(frame, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Render the frame in Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_container_width=True)

        # Break loop on keypress (optional, for local testing)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update stop button status
        stop_button = st.button("Stop Exercise")
    
    # Release camera
    cap.release()
    
    # Calculate exercise duration
    duration = int(time.time() - start_time)
    minutes = duration // 60
    seconds = duration % 60
    
    # Display summary screen
    st.success("Exercise Session Completed!")
    
    # Create three columns for statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Reps", counter)
    
    with col2:
        st.metric("Duration", f"{minutes}m {seconds}s")
    
    with col3:
        st.metric("Avg Reps/Min", round(counter / max(1, minutes), 1))
    
    # Display exercise-specific feedback
    st.subheader("Exercise Analysis")
    if counter == 0:
        st.warning("No repetitions were detected. Make sure you're visible in the camera frame.")
    else:
        if exercise == "Push-Ups":
            st.info("💪 Great job on those push-ups! Remember to maintain a straight back and control your descent.")
        elif exercise == "Squats":
            st.info("🏋️ Excellent squatting! Focus on keeping your knees aligned with your toes.")
        elif exercise == "Bicep Curls":
            st.info("💪 Well done on the curls! Keep your elbows close to your body for better form.")
        elif exercise == "Pull-Ups":
            st.info("💪 Impressive pull-ups! Remember to fully extend your arms at the bottom of each rep.")
    
    # Add a restart button
    if st.button("Start New Exercise"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()
