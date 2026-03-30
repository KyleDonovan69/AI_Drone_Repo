import cv2
import numpy as np
import time


# ── Tunable constants ─────────────────────────────────────────────────────────

# Monster-green HSV range  (tweak if detection is off under your lighting)
_HSV_LO = np.array([35,  80, 80],  dtype=np.uint8)
_HSV_HI = np.array([85, 255, 255], dtype=np.uint8)

# Minimum contour area (px²) to count as a real detection, not noise
_MIN_AREA = 1_200

# Frame fractions used for dead-zone logic
_CENTRE_DEAD_X = 0.18   # ±18 % of frame width  → no yaw correction
_CENTRE_DEAD_Y = 0.18   # ±18 % of frame height → no up/down correction
_CLOSE_AREA    = 0.10   # target occupies ≥10 % of frame area → "close enough"

# RC magnitudes
_YAW_SPEED    = 30   # left/right rotation to centre target
_UD_SPEED     = 20   # up/down to keep target vertically centred
_FB_SPEED     = 25   # forward speed when target is centred but far
_SEARCH_YAW   = 28   # slow clockwise spin when searching

# How often (seconds) to emit an RC command while searching
_SEARCH_TICK  = 0.30


class HuntMode:
    """
    Hunt a Monster can.

    Uses HSV colour masking to find the can's neon-green in the camera frame,
    then steers the drone toward it.  When nothing is visible the drone
    rotates slowly to search.

    Interface mirrors PersonFollower so TelloUI can treat both identically.
    """

    def __init__(self, tello):
        self.tello  = tello
        self.active = False

        self._last_tick  = 0.0   # time of last RC send
        self._searching  = False # True while no target visible

        # debug: drawn on frame so you can see what the detector sees
        self._debug_mask = False

    # ── Public API ────────────────────────────────────────────────────────────

    def toggle(self) -> bool:
        """Toggle hunt mode on/off.  Returns the new active state."""
        self.active = not self.active
        if not self.active:
            self._stop_rc()
            self._searching = False
        else:
            print("[HUNT] Hunt mode ON — looking for Monster can")
        return self.active

    def close(self):
        """Zero RC outputs and release any resources."""
        if self.active:
            self._stop_rc()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect the Monster can in *frame*, steer the drone, and return the
        annotated frame.  Call every video frame whether active or not.
        """
        if not self.active:
            return frame

        h, w = frame.shape[:2]
        cx, cy, area = self._detect(frame, w, h)

        if cx is None:
            # ── No target visible — search by rotating ──────────────────────
            self._searching = True
            now = time.time()
            if now - self._last_tick >= _SEARCH_TICK:
                self.tello.send_rc_control(0, 0, 0, _SEARCH_YAW)
                self._last_tick = now
            self._draw_searching(frame, w, h)
        else:
            # ── Target found — steer toward it ──────────────────────────────
            self._searching = False
            self._steer(cx, cy, area, w, h)
            self._draw_target(frame, cx, cy, area, w, h)

        return frame

    # ── Detection ─────────────────────────────────────────────────────────────

    def _detect(self, frame, w, h):
        """
        Return (cx, cy, area) of the largest Monster-green blob, or
        (None, None, None) if nothing found.
        """
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, _HSV_LO, _HSV_HI)

        # clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None

        biggest = max(contours, key=cv2.contourArea)
        area_px = cv2.contourArea(biggest)
        if area_px < _MIN_AREA:
            return None, None, None

        M  = cv2.moments(biggest)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # normalise area to fraction of total frame
        area_frac = area_px / (w * h)
        return cx, cy, area_frac

    # ── Steering ──────────────────────────────────────────────────────────────

    def _steer(self, cx, cy, area_frac, w, h):
        """Compute and send RC controls to home in on the target."""
        now = time.time()
        if now - self._last_tick < _SEARCH_TICK:
            return
        self._last_tick = now

        frame_cx = w / 2
        frame_cy = h / 2

        # normalised offsets  (-1 … +1)
        dx = (cx - frame_cx) / (w / 2)
        dy = (cy - frame_cy) / (h / 2)

        # ── yaw (left/right) ────────────────────────────────────────────────
        if abs(dx) > _CENTRE_DEAD_X:
            yaw = int(np.sign(dx) * _YAW_SPEED)
        else:
            yaw = 0

        # ── up/down ─────────────────────────────────────────────────────────
        if abs(dy) > _CENTRE_DEAD_Y:
            ud = int(-np.sign(dy) * _UD_SPEED)   # negative dy = target above
        else:
            ud = 0

        # ── forward ─────────────────────────────────────────────────────────
        # only move forward when reasonably centred and not already close
        centred = abs(dx) <= _CENTRE_DEAD_X * 1.5
        if centred and area_frac < _CLOSE_AREA:
            fb = _FB_SPEED
        else:
            fb = 0

        self.tello.send_rc_control(0, fb, ud, yaw)

    def _stop_rc(self):
        try:
            self.tello.send_rc_control(0, 0, 0, 0)
        except Exception:
            pass

    # ── HUD drawing ───────────────────────────────────────────────────────────

    def _draw_target(self, frame, cx, cy, area_frac, w, h):
        """Draw a crosshair + info banner on the frame."""
        colour = (0, 255, 80)

        # crosshair
        cv2.circle(frame, (cx, cy), 18, colour, 2)
        cv2.line(frame, (cx - 26, cy), (cx + 26, cy), colour, 2)
        cv2.line(frame, (cx, cy - 26), (cx, cy + 26), colour, 2)

        # line from frame centre to target
        cv2.line(frame, (w // 2, h // 2), (cx, cy), colour, 1)

        # status text
        close_txt = "CLOSE — hovering" if area_frac >= _CLOSE_AREA else "APPROACHING"
        cv2.putText(frame, f"[HUNT] {close_txt}",
                    (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        cv2.putText(frame, f"target area {area_frac*100:.1f}%",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)

    def _draw_searching(self, frame, w, h):
        """Draw a 'searching' banner."""
        colour = (0, 160, 255)
        cv2.putText(frame, "[HUNT] Searching for Monster can...",
                    (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        # small spinning indicator — just a pulsing circle in the corner
        radius = int(12 + 6 * abs(np.sin(time.time() * 3)))
        cv2.circle(frame, (w - 30, 30), radius, colour, 2)
