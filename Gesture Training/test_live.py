import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import deque

# setup
mp_drawing = mp.solutions.drawing_utils # type: ignore[attr-defined]
mp_drawing_styles = mp.solutions.drawing_styles # type: ignore[attr-defined]
mp_hands = mp.solutions.hands # type: ignore[attr-defined]

# gesture labels
GESTURES = {
    0: ('Open Palm',   'Hover / Stop'),
    1: ('Fist',        'Land'),
    2: ('Point Up',    'Move Up'),
    3: ('Point Down',  'Move Down'),
    4: ('Point Left',  'Rotate Left'),
    5: ('Point Right', 'Rotate Right'),
    6: ('Peace Sign',  'Next Mode'),
    7: ('Thumbs Up',   'Confirm Mode'),
}

# load trained model
with open('model/gesture_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Model loaded — press Q to quit")

# buffer for smoothing — only act when last 10 frames agree
BUFFER_SIZE = 10
buffer = deque(maxlen=BUFFER_SIZE)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            # extract keypoints
            lm = result.multi_hand_landmarks[0].landmark
            keypoints = []
            for point in lm:
                keypoints.append(point.x)
                keypoints.append(point.y)

            # predict gesture
            pred = model.predict([keypoints])[0]
            buffer.append(pred)

            # only show command if buffer agrees
            if len(buffer) == BUFFER_SIZE and len(set(buffer)) == 1:
                gesture_name, command = GESTURES[pred]

                # big command box at bottom of screen
                cv2.rectangle(frame, (0, frame.shape[0] - 80),
                              (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                cv2.putText(frame, f"Gesture: {gesture_name}", (10, frame.shape[0] - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Command: {command}", (10, frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

                print(f"Gesture: {gesture_name} → {command}")
            else:
                # buffer still filling or no consensus
                cv2.rectangle(frame, (0, frame.shape[0] - 80),
                              (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                cv2.putText(frame, "Hold gesture steady...", (10, frame.shape[0] - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        else:
            buffer.clear()
            cv2.putText(frame, "No hand detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Gesture Test — Live", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()