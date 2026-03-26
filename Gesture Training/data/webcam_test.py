import cv2
import mediapipe as mp

# setup
mp_drawing = mp.solutions.drawing_utils # type: ignore[attr-defined]
mp_drawing_styles = mp.solutions.drawing_styles # type: ignore[attr-defined]
mp_hands = mp.solutions.hands # type: ignore[attr-defined]

cap = cv2.VideoCapture(0)

print("Hand tracking running — press Q to quit")

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # flip so it feels like a mirror
        frame = cv2.flip(frame, 1)

        # mediapipe expects RGB
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

            cv2.putText(frame, "Hand detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("MediaPipe Hand Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()