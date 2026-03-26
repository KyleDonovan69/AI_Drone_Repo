import cv2
import mediapipe as mp
import csv
import os

# setup
mp_drawing = mp.solutions.drawing_utils # type: ignore[attr-defined]
mp_drawing_styles = mp.solutions.drawing_styles # type: ignore[attr-defined]
mp_hands = mp.solutions.hands # type: ignore[attr-defined]

# gesture labels — key maps to (gesture name, drone command)
GESTURES = {
    '0': ('Open Palm',   'Hover / Stop'),
    '1': ('Fist',        'Land'),
    '2': ('Point Up',    'Move Up'),
    '3': ('Point Down',  'Move Down'),
    '4': ('Point Left',  'Rotate Left'),
    '5': ('Point Right', 'Rotate Right'),
    '6': ('Peace Sign',  'Next Mode'),
    '7': ('Thumbs Up',   'Confirm Mode'),
}

# make sure data folder exists
os.makedirs('data', exist_ok=True)
CSV_PATH = 'data/keypoints.csv'

# count existing samples per gesture
def count_samples():
    counts = {k: 0 for k in GESTURES}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0] in counts:
                    counts[row[0]] += 1
    return counts

cap = cv2.VideoCapture(0)
print("Data collector running")
print("Hold a gesture and press the matching number key to save a sample")
print("Press Q to quit\n")

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

        # draw skeleton if hand detected
        keypoints = None
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            # flatten landmarks to a list of 42 values [x0,y0,x1,y1...]
            lm = result.multi_hand_landmarks[0].landmark
            keypoints = []
            for point in lm:
                keypoints.append(point.x)
                keypoints.append(point.y)

        # draw gesture guide on screen
        counts = count_samples()
        y = 30
        for key, (name, cmd) in GESTURES.items():
            count = counts[key]
            bar = '█' * min(count, 100) 
            color = (0, 255, 0) if count >= 100 else (0, 200, 255)
            cv2.putText(frame, f"[{key}] {name} ({cmd}): {count}/100",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            y += 25

        # hand status
        status = "Hand detected - ready to save" if keypoints else "No hand detected"
        status_color = (0, 255, 0) if keypoints else (0, 0, 255)
        cv2.putText(frame, status, (10, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        cv2.imshow("Gesture Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        # save sample when number key pressed
        if chr(key) in GESTURES and keypoints:
            gesture_id = chr(key)
            with open(CSV_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([gesture_id] + keypoints)
            name = GESTURES[gesture_id][0]
            count = counts[gesture_id] + 1
            print(f"Saved sample for [{gesture_id}] {name} — total: {count}")

        elif chr(key) in GESTURES and not keypoints:
            print("No hand detected — move your hand into frame first")

        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("\nDone! Samples saved to data/keypoints.csv")
print("Final counts:")
for key, (name, _) in GESTURES.items():
    print(f"  [{key}] {name}: {count_samples()[key]} samples")