import cv2
import numpy as np
import time

# ── Tunable constants ─────────────────────────────────────────────────────────

_HSV_LO = np.array([35, 60, 60],  dtype=np.uint8)
_HSV_HI = np.array([82, 255, 255], dtype=np.uint8)

# Strict range for confirmation (stage 2)
_HSV_STRICT_LO = np.array([40,  100, 100], dtype=np.uint8)
_HSV_STRICT_HI = np.array([75,  255, 255], dtype=np.uint8)

_MIN_AREA       = 400
_CENTRE_DEAD_X  = 0.18
_CENTRE_DEAD_Y  = 0.18
_CLOSE_AREA     = 0.075
_CONFIRM_RATIO  = 0.45   # 45% of bounding box must be strict-green to confirm

_YAW_SPEED   = 30
_UD_SPEED    = 15
_FB_SPEED    = 15
_BACK_SPEED  = 10        # reverse speed during failed confirmation
_SEARCH_TICK = 0.30

_CONFIRM_TIME = 1.0      # seconds to hold position and check before deciding
_BACKOFF_TIME = 1.0      # seconds to back away after failed confirmation

# ── Patrol (square + 360 scan) constants ─────────────────────────────────────

_PATROL_SIDE_DURATION  = 1.0   # seconds to fly each side — tune in lab for room size
_PATROL_TURN_DURATION  = 0.8    # seconds to yaw 90 degrees
_PATROL_SCAN_DURATION  = 0.4    # seconds per scan yaw tick
_PATROL_SCAN_TICKS     = 8      # 8 ticks * 0.4s * yaw35 ≈ full 360
_PATROL_FWD_SPEED      = 15     # forward speed during patrol sides
_PATROL_TURN_YAW       = 40     # yaw speed for 90 degree corner turn
_PATROL_SCAN_YAW       = 35     # yaw speed during 360 scan

# ── Debug flag ────────────────────────────────────────────────────────────────
_DEBUG_HSV = False   # set True in lab to print centre-pixel HSV values


class HuntMode:
    def __init__(self, tello):
        self.tello  = tello
        self.active = False

        self._last_tick   = 0.0
        self._searching   = False
        self._confirming  = False
        self._backing_off = False

        self._confirm_start  = 0.0
        self._backoff_start  = 0.0

        # patrol state
        self._patrol_phase       = 'scan'   # 'scan', 'move', 'turn'
        self._patrol_phase_start = 0.0
        self._patrol_scan_tick   = 0
        self._patrol_corner      = 0        # 0-3, which corner of the square

    # ── Public API ────────────────────────────────────────────────────────────

    def toggle(self) -> bool:
        self.active = not self.active
        if not self.active:
            self._stop_rc()
            self._reset_states()
        else:
            print("[HUNT] Hunt mode ON — looking for Monster can")
        return self.active

    def close(self):
        if self.active:
            self._stop_rc()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self.active:
            return frame

        h, w = frame.shape[:2]

        # optional HSV debug — point drone at can and read console
        if _DEBUG_HSV:
            self._debug_hsv(frame, w, h)

        # ── Backing off after failed confirmation ────────────────────────────
        if self._backing_off:
            if time.time() - self._backoff_start >= _BACKOFF_TIME:
                self._backing_off = False
                self._searching   = True
                self._stop_rc()
            else:
                self.tello.send_rc_control(0, -_BACK_SPEED, 0, 0)
                self._draw_status(frame, w, h, "[HUNT] Not the can — backing off", (0, 0, 255))
            return frame

        cx, cy, area = self._detect(frame, w, h, strict=False)

        if cx is None:
            # ── Nothing visible — square patrol with 360 scans ───────────────
            self._confirming = False
            self._searching  = True
            self._run_patrol()
            self._draw_searching(frame, w, h)

        elif area >= _CLOSE_AREA:
            # ── Close enough — run confirmation check ────────────────────────
            if not self._confirming:
                self._confirming     = True
                self._confirm_start  = time.time()
                self._stop_rc()

            elapsed = time.time() - self._confirm_start

            if elapsed >= _CONFIRM_TIME:
                if self._confirm(frame, cx, cy, w, h):
                    self.active = False  
                    self._stop_rc()
                    print("[HUNT] TARGET CONFIRMED — Initiating Safe Landing")
                    
                    # Check height and land
                    current_height = self.tello.get_height()
                    if current_height > 30:
                        self.tello.move_down(0.2)
                        
                    self.tello.land()
                else:
                    self._confirming  = False
                    self._backing_off = True
                    self._backoff_start = time.time()
            else:
                self._stop_rc()
                self._draw_status(frame, w, h,
                    f"[HUNT] Confirming... {elapsed:.1f}s", (0, 200, 255))

        else:
            # ── Visible but not close yet — approach ─────────────────────────
            self._confirming = False
            self._searching  = False
            # reset patrol so it starts fresh next time we lose the target
            self._reset_patrol()
            self._steer(cx, cy, area, w, h)
            self._draw_target(frame, cx, cy, area, w, h)

        return frame

    # ── Patrol ────────────────────────────────────────────────────────────────

    def _run_patrol(self):
        """
        Square patrol: at each corner do a full 360 scan, then fly one side,
        then yaw 90 degrees to face the next side. Loops continuously.

        Pattern per corner:
            [360 scan] → fly forward → [360 scan] → turn 90° → repeat
        """
        now = time.time()

        if self._patrol_phase == 'scan':
            # 360 scan — fire _PATROL_SCAN_TICKS yaw ticks
            if now - self._patrol_phase_start >= _PATROL_SCAN_DURATION:
                if self._patrol_scan_tick < _PATROL_SCAN_TICKS:
                    self.tello.send_rc_control(0, 0, 0, _PATROL_SCAN_YAW)
                    self._patrol_scan_tick  += 1
                    self._patrol_phase_start = now
                else:
                    # scan complete — start flying the next side
                    self._patrol_scan_tick   = 0
                    self._patrol_phase       = 'move'
                    self._patrol_phase_start = now
                    self.tello.send_rc_control(0, _PATROL_FWD_SPEED, 0, 0)

        elif self._patrol_phase == 'move':
            # fly one side of the square
            if now - self._patrol_phase_start >= _PATROL_SIDE_DURATION:
                self._stop_rc()
                self._patrol_phase       = 'turn'
                self._patrol_phase_start = now

        elif self._patrol_phase == 'turn':
            # yaw 90 degrees to face next side
            if now - self._patrol_phase_start >= _PATROL_TURN_DURATION:
                self._stop_rc()
                self._patrol_corner      = (self._patrol_corner + 1) % 4
                self._patrol_phase       = 'scan'
                self._patrol_phase_start = now
                self._patrol_scan_tick   = 0
            else:
                self.tello.send_rc_control(0, 0, 0, _PATROL_TURN_YAW)

    def _reset_patrol(self):
        """Reset patrol to beginning so next search starts cleanly."""
        self._patrol_phase       = 'scan'
        self._patrol_phase_start = 0.0
        self._patrol_scan_tick   = 0
        self._patrol_corner      = 0

    # ── Detection ─────────────────────────────────────────────────────────────

    def _detect(self, frame, w, h, strict=False):
        lo = _HSV_STRICT_LO if strict else _HSV_LO
        hi = _HSV_STRICT_HI if strict else _HSV_HI

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask    = cv2.inRange(hsv, lo, hi)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None

        biggest  = max(contours, key=cv2.contourArea)
        area_px  = cv2.contourArea(biggest)
        if area_px < _MIN_AREA:
            return None, None, None

        # aspect ratio check — Monster can is taller than wide
        x, y, bw, bh = cv2.boundingRect(biggest)
        aspect = bh / bw if bw > 0 else 0
        if aspect < 0.5:
            return None, None, None

        M  = cv2.moments(biggest)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        area_frac = area_px / (w * h)
        return cx, cy, area_frac

    def _confirm(self, frame, cx, cy, w, h):
        """
        Strict check at close range.
        Crops a region around the target and checks what fraction
        of it matches the strict Monster-green HSV range.
        """
        crop_size = int(min(w, h) * 0.25)
        x1 = max(0, cx - crop_size // 2)
        y1 = max(0, cy - crop_size // 2)
        x2 = min(w, cx + crop_size // 2)
        y2 = min(h, cy + crop_size // 2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False

        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, _HSV_STRICT_LO, _HSV_STRICT_HI)

        ratio = np.count_nonzero(mask) / mask.size
        print(f"[HUNT] Confirm ratio: {ratio:.2f} (need {_CONFIRM_RATIO})")
        return ratio >= _CONFIRM_RATIO

    # ── Debug ─────────────────────────────────────────────────────────────────

    def _debug_hsv(self, frame, w, h):
        """Print HSV value of centre pixel — use in lab to tune HSV ranges."""
        cx, cy = w // 2, h // 2
        hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        pixel  = hsv[cy, cx]
        print(f"[DEBUG] Centre HSV: H={pixel[0]} S={pixel[1]} V={pixel[2]}")
        # draw a small crosshair so you know where it's sampling
        cv2.circle(frame, (cx, cy), 6, (255, 255, 0), 2)

    # ── Steering ──────────────────────────────────────────────────────────────

    def _steer(self, cx, cy, area_frac, w, h):
        now = time.time()
        if now - self._last_tick < _SEARCH_TICK:
            return
        self._last_tick = now

        frame_cx = w / 2
        frame_cy = h / 2

        dx = (cx - frame_cx) / (w / 2)
        dy = (cy - frame_cy) / (h / 2)

        yaw = int(np.sign(dx) * _YAW_SPEED) if abs(dx) > _CENTRE_DEAD_X else 0
        ud  = int(-np.sign(dy) * _UD_SPEED)  if abs(dy) > _CENTRE_DEAD_Y else 0

        centred = abs(dx) <= _CENTRE_DEAD_X * 1.5
        fb = _FB_SPEED if centred and area_frac < _CLOSE_AREA else 0

        self.tello.send_rc_control(0, fb, ud, yaw)

    def _stop_rc(self):
        try:
            self.tello.send_rc_control(0, 0, 0, 0)
        except Exception:
            pass

    def _reset_states(self):
        self._searching   = False
        self._confirming  = False
        self._backing_off = False
        self._reset_patrol()

    # ── HUD drawing ───────────────────────────────────────────────────────────

    def _draw_target(self, frame, cx, cy, area_frac, w, h):
        colour = (0, 255, 80)
        cv2.circle(frame, (cx, cy), 18, colour, 2)
        cv2.line(frame, (cx - 26, cy), (cx + 26, cy), colour, 2)
        cv2.line(frame, (cx, cy - 26), (cx, cy + 26), colour, 2)
        cv2.line(frame, (w // 2, h // 2), (cx, cy), colour, 1)
        close_txt = "CLOSE — hovering" if area_frac >= _CLOSE_AREA else "APPROACHING"
        cv2.putText(frame, f"[HUNT] {close_txt}",
                    (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        cv2.putText(frame, f"target area {area_frac*100:.1f}%",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)

    def _draw_searching(self, frame, w, h):
        colour = (0, 160, 255)
        phase_txt = {
            'scan': f"Scanning... ({self._patrol_scan_tick}/{_PATROL_SCAN_TICKS})",
            'move': "Advancing...",
            'turn': f"Turning to corner {self._patrol_corner + 1}/4",
        }
        label = phase_txt.get(self._patrol_phase, "Searching...")
        cv2.putText(frame, f"[HUNT] {label}",
                    (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        radius = int(12 + 6 * abs(np.sin(time.time() * 3)))
        cv2.circle(frame, (w - 30, 30), radius, colour, 2)

    def _draw_status(self, frame, w, h, text, colour):
        cv2.putText(frame, text, (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)