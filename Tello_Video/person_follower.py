import cv2
import mediapipe as mp
import numpy as np
import time


class PersonFollower:
    """
    Detects a person using MediaPipe Pose and issues RC control commands
    to the Tello drone to keep them centred in frame at a stable distance.

    Activation: call toggle() — returns the new active state.
    Each video loop iteration: call process_frame(bgr_frame) to get an
    annotated frame back and (when active) dispatch RC commands.

    Control axes:
        YAW      — rotates drone left/right to centre person horizontally.
        UP/DOWN  — adjusts altitude to keep shoulders at mid-frame height.
        FWD/BACK — adjusts distance to maintain a target apparent body size.
    """

    # ── RC speed limits (Tello accepts -100 to 100) ─────────────────────
    MAX_YAW: int = 30
    MAX_UD: int = 25
    MAX_FB: int = 25

    # ── Dead-bands (normalised 0–1) — ignore tiny errors to avoid jitter ─
    DEAD_ZONE: float = 0.10
    SIZE_DEAD_ZONE: float = 0.07

    # ── Target apparent size (shoulder-to-hip height as fraction of frame) ─
    TARGET_SIZE: float = 0.60

    # ── Max RC command rate ──────────────────────────────────────────────
    CMD_INTERVAL: float = 0.15  # seconds between RC packets (~7 Hz)

    # ── MediaPipe Pose landmark indices ─────────────────────────────────
    _IDX_L_SHOULDER: int = 11
    _IDX_R_SHOULDER: int = 12
    _IDX_L_HIP: int = 23
    _IDX_R_HIP: int = 24

    def __init__(self, tello) -> None:
        """
        :param tello: Tello instance — must expose send_rc_control().
        """
        self._tello = tello
        self.active: bool = False
        self._last_cmd_time: float = 0.0

        self._pose = mp.solutions.pose.Pose(
            model_complexity=0,              # lightest model — best for real-time
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._draw_utils = mp.solutions.drawing_utils
        self._pose_connections = mp.solutions.pose.POSE_CONNECTIONS

    # ── Public interface ─────────────────────────────────────────────────

    def toggle(self) -> bool:
        """
        Toggle follow mode on/off.

        :return: New active state (True = following).
        """
        self.active = not self.active
        if not self.active:
            # Zero all axes so the drone hovers in place
            self._tello.send_rc_control(0, 0, 0, 0)
        return self.active

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Run pose detection on a BGR frame, overlay landmarks and status,
        and (when active) send RC control commands.

        :param frame: BGR numpy array from the drone camera.
        :return: Annotated BGR frame ready for display.
        """
        if frame is None or frame.size == 0:
            return frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)

        annotated = frame.copy()

        if results.pose_landmarks:
            self._draw_utils.draw_landmarks(
                annotated,
                results.pose_landmarks,
                self._pose_connections,
            )
            if self.active:
                self._compute_and_send(results.pose_landmarks.landmark)
        else:
            # No person visible then stop moving
            if self.active:
                self._throttled_rc(0, 0, 0, 0)

        self._draw_status(annotated)
        return annotated

    def close(self) -> None:
        """Release MediaPipe resources and halt any ongoing RC movement."""
        if self.active:
            self._tello.send_rc_control(0, 0, 0, 0)
        self._pose.close()

    # ── Private helpers ──────────────────────────────────────────────────

    def _compute_and_send(self, landmarks) -> None:
        """
        Calculate proportional RC values from pose landmark positions and
        send them to the drone.
        """
        ls = landmarks[self._IDX_L_SHOULDER]
        rs = landmarks[self._IDX_R_SHOULDER]
        lh = landmarks[self._IDX_L_HIP]
        rh = landmarks[self._IDX_R_HIP]

        # Midpoint in norm image space (0.0 – 1.0)
        shoulder_cx = (ls.x + rs.x) / 2.0
        shoulder_cy = (ls.y + rs.y) / 2.0
        hip_cy = (lh.y + rh.y) / 2.0

        # positive values mean "too far right / down / close"
        x_err = shoulder_cx - 0.5
        y_err = shoulder_cy - 0.5
        size_err = abs(shoulder_cy - hip_cy) - self.TARGET_SIZE

        # ── Yaw: rotate to centre person horizontally ─────────────────
        yaw = 0
        if abs(x_err) > self.DEAD_ZONE:
            yaw = int(np.clip(x_err * 120, -self.MAX_YAW, self.MAX_YAW))

        # ── Up/Down: keep shoulders at vertical mid-frame ───────────-
        # Image Y increases downward, so negate: person low > drone moves down
        ud = 0
        if abs(y_err) > self.DEAD_ZONE:
            ud = int(np.clip(-y_err * 100, -self.MAX_UD, self.MAX_UD))

        # ── Forward/Back: maintain target apparent body size ─────────
        # Positive size_err means person is too close > back off
        fb = 0
        if abs(size_err) > self.SIZE_DEAD_ZONE:
            fb = int(np.clip(-size_err * 150, -self.MAX_FB, self.MAX_FB))

        self._throttled_rc(0, fb, ud, yaw)

    def _throttled_rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        """Send RC only if the command interval has elapsed."""
        now = time.time()
        if now - self._last_cmd_time >= self.CMD_INTERVAL:
            self._tello.send_rc_control(lr, fb, ud, yaw)
            self._last_cmd_time = now

    def _draw_status(self, frame: np.ndarray) -> None:
        """Overlay a follow-mode status label in the bottom-left corner."""
        label = "FOLLOW: ON" if self.active else "FOLLOW: OFF"
        colour = (0, 255, 0) if self.active else (0, 0, 255)
        cv2.putText(
            frame, label,
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2,
        )
