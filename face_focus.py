import cv2
import dlib
from imutils.video import VideoStream
import numpy as np
import time
from simple_facerec import SimpleFacerec

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"M:\aiml el\shape_predictor_68_face_landmarks (1).dat")

sfr = SimpleFacerec()
sfr.load_encoding_images("M:/aiml el/face-recog/images")

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

def main():
    vs = VideoStream(src=0).start()
    
    while True:
        start_time = time.time()
        frame = vs.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        face_locations, face_names = sfr.detect_known_faces(frame)
        
        num_faces = len(face_locations) 

        if num_faces > 1:
            print("More than one person detected")
        
        count_text = f"Persons Detected: {num_faces}"
        cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        for i, face in enumerate(faces):
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
            
            head_angle = 0
            
            focus_level = calculate_focus(average_EAR, head_angle, eye_closed, average_gaze)
            
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            for (x, y) in landmarks:
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            
            gaze_color = (0, 255, 0) if abs(average_gaze) < 10 else (0, 0, 255)
            cv2.line(frame, (face.left(), face.top()), (face.right(), face.bottom()), gaze_color, 2)
            
            focus_text = f"Focus: {focus_level:.2f}%"
            text_x = face.left()
            text_y = face.bottom() + 30
            cv2.putText(frame, focus_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("Frame", frame)
        elapsed_time = time.time() - start_time
        sleep_time = max(0, 0.3 - elapsed_time)
        time.sleep(sleep_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
