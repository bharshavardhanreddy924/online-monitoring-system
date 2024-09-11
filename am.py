import streamlit as st
import cv2
import numpy as np
import dlib
from simple_facerec import SimpleFacerec
import threading
import time
import webrtcvad
import pyaudio
from pynput import keyboard
import datetime
import os
from scipy.spatial import distance

# Initialize Dlib's face detector and predictor
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks (1).dat")
sfr = SimpleFacerec()
sfr.load_encoding_images("face-recog/images")

# Initialize YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
prohibited_objects = [
    'cell phone', 'book', 'laptop', 'keyboard', 'mouse', 'remote',
    'bottle', 'cup', 'wine glass', 'fork', 'knife', 'spoon'
]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize video capture
vs = cv2.VideoCapture(0)
frame_lock = threading.Lock()
current_frame = None

# Initialize VAD and audio stream
vad = webrtcvad.Vad(1)
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=320)
stream.start_stream()

# Global variables
voice_detected = False
prohibited_keys = {
    frozenset([keyboard.Key.alt_l, keyboard.Key.tab]),
    frozenset([keyboard.Key.esc]),
    frozenset([keyboard.KeyCode.from_char('r')]),
    frozenset([keyboard.KeyCode.from_char('c')]),
}
pressed_keys = set()
key_violation = False
exam_events = []
exam_start_time = None

# Initialize face mesh detector
face_mesh = cv2.face.createFacemarkLBF()
face_mesh.loadModel("lbfmodel.yaml")

def calculate_focus(eye_aspect_ratio, head_angle, eye_closed, gaze_direction):
    EAR_THRESHOLD = 0.25
    HEAD_ANGLE_THRESHOLD = 5
    GAZE_THRESHOLD = 5
    IGNORE_GAZE_THRESHOLD = 10

    eye_focus = 1 - max(0, (eye_aspect_ratio - EAR_THRESHOLD) / EAR_THRESHOLD)
    if eye_closed:
        eye_focus = 0

    head_focus = max(0, 1 - (abs(head_angle) / HEAD_ANGLE_THRESHOLD))
    
    if abs(gaze_direction) > IGNORE_GAZE_THRESHOLD:
        gaze_focus = 1
    else:
        gaze_focus = max(0, 1 - abs(gaze_direction) / GAZE_THRESHOLD)

    focus_level = 0.4 * eye_focus + 0.3 * head_focus + 0.3 * gaze_focus
    return focus_level * 100

def get_eye_aspect_ratio(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C)

def get_gaze_direction(eye_points):
    eye_center = np.mean(eye_points, axis=0)
    eye_left = eye_points[0]
    eye_right = eye_points[3]
    direction = eye_center[0] - (eye_left[0] + eye_right[0]) / 2
    return direction

def process_voice():
    global voice_detected, exam_events, exam_start_time
    while True:
        try:
            audio_frame = stream.read(320, exception_on_overflow=False)
            is_speech = vad.is_speech(audio_frame, 16000)
            if is_speech != voice_detected:
                voice_detected = is_speech
                event_time = time.time() - exam_start_time
                event = f"Voice {'detected' if voice_detected else 'stopped'} at {event_time:.2f} seconds"
                exam_events.append(event)
        except Exception as e:
            print(f"Error in voice detection: {e}")
        time.sleep(0.1)

def capture_frames():
    global vs, current_frame
    while True:
        ret, frame = vs.read()
        if ret:
            with frame_lock:
                current_frame = frame.copy()
        else:
            time.sleep(0.1)

def on_press(key):
    global key_violation, exam_events, exam_start_time
    pressed_keys.add(key)
    for combo in prohibited_keys:
        if combo.issubset(pressed_keys):
            key_violation = True
            event_time = time.time() - exam_start_time
            event = f"Prohibited key combination detected at {event_time:.2f} seconds: {', '.join(str(k) for k in combo)}"
            exam_events.append(event)

def on_release(key):
    if key in pressed_keys:
        pressed_keys.remove(key)

def start_keyboard_listener():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

def detect_objects(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in prohibited_objects:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return [(classes[class_ids[i]], boxes[i]) for i in indexes]

def generate_report(exam_duration, focus_levels):
    report = f"Exam Monitoring Report\n"
    report += f"Date and Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"Exam Duration: {exam_duration} seconds\n\n"

    report += "Events:\n"
    for event in exam_events:
        report += f"- {event}\n"

    report += f"\nAverage Focus Level: {sum(focus_levels) / len(focus_levels):.2f}%\n"

    report_file = f"exam_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, "w") as f:
        f.write(report)

    return report_file

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])  # 51, 59
    B = distance.euclidean(mouth[4], mouth[8])   # 53, 57
    C = distance.euclidean(mouth[0], mouth[6])   # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

def detect_liveness(frame, landmarks):
    # Extract eye and mouth landmarks
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    mouth = landmarks[48:68]

    # Calculate EAR and MAR
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    mar = mouth_aspect_ratio(mouth)

    # Thresholds
    EAR_THRESHOLD = 0.2
    MAR_THRESHOLD = 0.5

    # Check for blink and mouth movement
    if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
        return True  # Blink detected, likely a real face
    if mar > MAR_THRESHOLD:
        return True  # Mouth movement detected, likely a real face

    return False  # No liveness detected

def detect_picture_in_phone(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Check for rectangular shapes (potential phone outline)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.5 < aspect_ratio < 1.5:  # Typical phone aspect ratios
                return True

    return False

def main():
    global exam_start_time, exam_events
    st.title("Exam Monitoring System")

    option = st.sidebar.selectbox("Choose an option", ("Registration", "Exam"))

    if option == "Registration":
        st.header("User Registration")
        name = st.text_input("Enter your name")
        frame_placeholder = st.empty()
        register_button = st.button("Capture Photo")

        if name:
            threading.Thread(target=capture_frames, daemon=True).start()

            while True:
                with frame_lock:
                    if current_frame is not None:
                        frame = current_frame.copy()
                    else:
                        continue

                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_placeholder.image(buffer.tobytes(), channels="BGR")

                if register_button:
                    if name and current_frame is not None:
                        cv2.imwrite(f"face-recog/images/{name}.jpg", current_frame)
                        sfr.load_encoding_images("face-recog/images")
                        st.success(f"Registration successful for {name}!")
                        break
                    else:
                        st.error("Could not capture image. Please try again.")
                    time.sleep(1)

        st.write("Once registered, proceed to the exam section.")

    elif option == "Exam":
        st.header("Exam Monitoring")

        start_exam = st.button("Start Exam")

        if start_exam:
            exam_start_time = time.time()
            exam_events = []
            focus_levels = []

            focus_placeholder = st.empty()
            frame_placeholder = st.empty()
            voice_indicator = st.empty()
            key_violation_indicator = st.empty()
            object_detection_indicator = st.empty()
            time_remaining = st.empty()
            multiple_faces_warning = st.empty()
            liveness_indicator = st.empty()
            picture_in_phone_indicator = st.empty()

            threading.Thread(target=capture_frames, daemon=True).start()
            threading.Thread(target=process_voice, daemon=True).start()
            threading.Thread(target=start_keyboard_listener, daemon=True).start()

            exam_duration = 180  # 1 minute exam

            while time.time() - exam_start_time < exam_duration:
                remaining_time = exam_duration - (time.time() - exam_start_time)
                time_remaining.text(f"Time Remaining: {remaining_time:.2f} seconds")

                with frame_lock:
                    if current_frame is not None:
                        frame = current_frame.copy()
                    else:
                        continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)
                face_locations, face_names = sfr.detect_known_faces(frame)

                num_faces = len(face_locations)
                if num_faces > 1:
                    event_time = time.time() - exam_start_time
                    event = f"Multiple faces detected at {event_time:.2f} seconds"
                    exam_events.append(event)
                    multiple_faces_warning.error("Error: Multiple faces detected in frame!")
                else:
                    multiple_faces_warning.empty()

                count_text = f"Persons Detected: {num_faces}"
                cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                for face_loc, name in zip(face_locations, face_names):
                    y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

                focus_level = 100
                for face in faces:
                    landmarks = predictor(gray, face)
                    landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])
                    
                    left_eye = landmarks[36:42]
                    right_eye = landmarks[42:48]
                    
                    left_EAR = get_eye_aspect_ratio(left_eye)
                    right_EAR = get_eye_aspect_ratio(right_eye)
                    average_EAR = (left_EAR + right_EAR) / 2.0
                    
                    eye_closed = average_EAR < 0.25
                    
                    left_gaze = get_gaze_direction(left_eye)
                    right_gaze = get_gaze_direction(right_eye)
                    average_gaze = (left_gaze + right_gaze) / 2.0
                    
                    head_angle = 0  # Placeholder for head angle calculation
                    
                    focus_level = calculate_focus(average_EAR, head_angle, eye_closed, average_gaze)
                    focus_levels.append(focus_level)
                    
                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                    for (x, y) in landmarks:
                        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                    
                    gaze_color = (0, 255, 0) if abs(average_gaze) < 10 else (0, 0, 255)
                    cv2.line(frame, (face.left(), face.top()), (face.right(), face.bottom()), gaze_color, 2)
                    
                    focus_text = f"Focus: {focus_level:.2f}%"
                    cv2.putText(frame, focus_text, (face.left(), face.bottom() + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                    # Liveness detection
                    is_live = detect_liveness(frame, landmarks)
                    if is_live:
                        liveness_indicator.markdown("**Face Liveness Detected**", unsafe_allow_html=True)
                    else:
                        liveness_indicator.markdown("**Warning: Possible Fake Face**", unsafe_allow_html=True)
                        event = f"Possible fake face detected at {time.time() - exam_start_time:.2f} seconds"
                        exam_events.append(event)   

                    # Picture-in-phone detection
                    is_picture_in_phone = detect_picture_in_phone(frame)
                    if is_picture_in_phone:
                        event = f"Possible picture-in-phone detected at {time.time() - exam_start_time:.2f} seconds"
                        exam_events.append(event)
                    else:
                        picture_in_phone_indicator.markdown("No Picture-in-Phone Detected", unsafe_allow_html=True)

                focus_placeholder.write(f"Current Focus Level: {focus_level:.2f}%")

                if voice_detected:
                    voice_indicator.markdown("**Voice Detected!**", unsafe_allow_html=True)
                else:
                    voice_indicator.markdown("No Voice Detected", unsafe_allow_html=True)

                if key_violation:
                    key_violation_indicator.markdown("**Prohibited Key Press Detected!**", unsafe_allow_html=True)
                else:
                    key_violation_indicator.markdown("No Key Violation Detected", unsafe_allow_html=True)

                # Object detection
                detected_objects = detect_objects(frame)
                if detected_objects:
                    object_text = "Prohibited Objects Detected: " + ", ".join([obj[0] for obj in detected_objects])
                    object_detection_indicator.markdown(f"**{object_text}**", unsafe_allow_html=True)
                    for obj_class, (x, y, w, h) in detected_objects:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, obj_class, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        event = f"Prohibited object '{obj_class}' detected at {time.time() - exam_start_time:.2f} seconds"
                        exam_events.append(event)
                else:
                    object_detection_indicator.markdown("No Prohibited Objects Detected", unsafe_allow_html=True)

                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_placeholder.image(buffer.tobytes(), channels="BGR")
                
                time.sleep(0.1)

            # Exam finished
            st.header("Exam Completed")
            report_file = generate_report(exam_duration, focus_levels)
            st.success(f"Exam report generated: {report_file}")

            # Display report contents
            with open(report_file, "r") as f:
                report_contents = f.read()
            st.text_area("Exam Report", report_contents, height=400)

if __name__ == "__main__":
    main()