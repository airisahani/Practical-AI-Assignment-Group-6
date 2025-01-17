import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose

# Initialize session state variables if they don't exist
if 'exercise_active' not in st.session_state:
    st.session_state.exercise_active = False

if 'counter' not in st.session_state:
    st.session_state.counter = 0

if 'start_time' not in st.session_state:
    st.session_state.start_time = None

# Helper function to calculate angle
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def main():
    st.title("Exercise Monitoring App")
    st.sidebar.title("Exercise Options")

    exercise_choice = st.sidebar.selectbox(
        "Select an exercise", ["Push-Ups", "Squats", "Bicep Curls", "Pull-Ups"]
    )

    # Create a container for the button
    button_container = st.sidebar.empty()

    # Show either Start or Stop button based on exercise_active state
    if st.session_state.exercise_active:
        if button_container.button("Stop Monitoring", key="stop"):
            st.session_state.exercise_active = False
            st.rerun()
    else:
        if button_container.button("Start Monitoring", key="start"):
            st.session_state.exercise_active = True
            st.session_state.counter = 0
            st.session_state.start_time = time.time()
            st.rerun()

    st.write(f"### Selected Exercise: {exercise_choice}")

    # If exercise is active, run the monitoring
    if st.session_state.exercise_active:
        run_exercise_monitoring(exercise_choice)
    else:
        # If the exercise is not active AND we have a start_time (meaning user just finished)
        # show the summary button
        if st.session_state.start_time is not None:
            show_summary_button = st.button("Show Exercise Summary", key="show_summary")
            if show_summary_button:
                display_exercise_summary(exercise_choice)

def run_exercise_monitoring(exercise):
    # Initialize the Pose model
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.8,
    )

    # Try different camera indices
    for camera_index in [0, 1, 2]:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            break

    if not cap.isOpened():
        st.error("Error: Could not access the webcam.")
        return

    stframe = st.empty()

    # We won't break from the loop until the user clicks Stop (which resets exercise_active to False)
    while cap.isOpened() and st.session_state.exercise_active:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Unable to read from the webcam.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect pose landmarks
        results = pose.process(frame_rgb)
        landmarks = results.pose_landmarks

        if landmarks:
            landmarks_array = [
                [lm.x * frame.shape[1], lm.y * frame.shape[0]] for lm in landmarks.landmark
            ]

            # Calculate the angle based on selected exercise
            if exercise == "Push-Ups":
                angle = calculate_angle(
                    landmarks_array[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                    landmarks_array[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                    landmarks_array[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                )
                process_exercise_logic(frame, angle, "Push-Ups")

            elif exercise == "Squats":
                angle = calculate_angle(
                    landmarks_array[mp_pose.PoseLandmark.RIGHT_HIP.value],
                    landmarks_array[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                    landmarks_array[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                )
                process_exercise_logic(frame, angle, "Squats")

            elif exercise == "Bicep Curls":
                angle = calculate_angle(
                    landmarks_array[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                    landmarks_array[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                    landmarks_array[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                )
                process_exercise_logic(frame, angle, "Bicep Curls")

            elif exercise == "Pull-Ups":
                angle = calculate_angle(
                    landmarks_array[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                    landmarks_array[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                    landmarks_array[mp_pose.PoseLandmark.RIGHT_HIP.value],
                )
                process_exercise_logic(frame, angle, "Pull-Ups")

        # Convert frame back to RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_container_width=True)

    cap.release()
    stframe.empty()

def process_exercise_logic(frame, angle, exercise_name):
    """
    Common logic for counting reps, angles, etc.
    """
    # We store a 'status' in session state if you prefer
    if 'status' not in st.session_state:
        st.session_state.status = False

    # Different angle thresholds per exercise (example logic)
    if exercise_name == "Push-Ups":
        # If angle > 160 => top of push-up
        if angle > 160:
            st.session_state.status = True
        # If we go down to <90 => count rep
        if st.session_state.status and angle < 90:
            st.session_state.counter += 1
            st.session_state.status = False

    elif exercise_name == "Squats":
        if angle > 160:
            st.session_state.status = True
        if st.session_state.status and angle < 90:
            st.session_state.counter += 1
            st.session_state.status = False

    elif exercise_name == "Bicep Curls":
        if angle > 160:
            st.session_state.status = True
        if st.session_state.status and angle < 60:
            st.session_state.counter += 1
            st.session_state.status = False

    elif exercise_name == "Pull-Ups":
        if angle < 60:
            st.session_state.status = True
        if st.session_state.status and angle > 120:
            st.session_state.counter += 1
            st.session_state.status = False

    # Show counts and angle on the frame
    cv2.putText(frame, f"{exercise_name}: {st.session_state.counter}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Angle: {int(angle)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    status_text = "Up" if st.session_state.status else "Down"
    cv2.putText(frame, status_text, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def display_exercise_summary(exercise):
    """
    Show the summary metrics once the user has stopped and clicked "Show Summary."
    """
    duration = int(time.time() - st.session_state.start_time)
    minutes = duration // 60
    seconds = duration % 60
    total_reps = st.session_state.counter

    st.success("Exercise Session Completed!")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Reps", total_reps)
    with col2:
        st.metric("Duration", f"{minutes}m {seconds}s")
    with col3:
        # Avoid division by zero in case minutes = 0
        st.metric("Avg Reps/Min", round(total_reps / max(1, minutes), 1))

    st.subheader("Exercise Analysis")
    if total_reps == 0:
        st.warning("No repetitions were detected. Make sure you're visible in the camera frame.")
    else:
        if exercise == "Push-Ups":
            st.info("💪 Great job on those push-ups! Remember to maintain a straight back.")
        elif exercise == "Squats":
            st.info("🏋️ Excellent squatting! Focus on keeping your knees aligned with your toes.")
        elif exercise == "Bicep Curls":
            st.info("💪 Well done on the curls! Keep your elbows close to your body.")
        elif exercise == "Pull-Ups":
            st.info("💪 Impressive pull-ups! Remember to fully extend your arms at the bottom.")

    # Optionally reset for a new session
    if st.button("Start New Exercise"):
        # Reset relevant session state
        st.session_state.counter = 0
        st.session_state.start_time = None
        st.session_state.status = False
        st.rerun()

if __name__ == "__main__":
    main()
