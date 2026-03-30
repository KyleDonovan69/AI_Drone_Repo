import threading
import time
import numpy as np
import cv2


class Tello:
    """
    Simulator version of the Tello wrapper.

    Behaves the same as the real drone, but instead
    of sending UDP packets to a drone it prints every command to the terminal
    and captures frames from the laptop webcam so gesture recognition and
    person following work normally.

    Swap this in main.py to use sim instead.
    """

    def __init__(self, local_ip, local_port, imperial=False, command_timeout=.3,
                 tello_ip='192.168.10.1', tello_port=8889):
        """
        Mirrors the real Tello.__init__ signature exactly so main.py needs
        no changes.

        :param local_ip (str): Ignored in simulator.
        :param local_port (int): Ignored in simulator.
        :param imperial (bool): If True, speed is MPH and distance is feet.
                             If False, speed is KPH and distance is meters.
        :param command_timeout (int|float): Ignored in simulator.
        :param tello_ip (str): Ignored in simulator.
        :param tello_port (int): Ignored in simulator.
        """
        self.imperial   = imperial
        self.is_freeze  = False
        self.last_frame = None
        self.frame      = None

        # flight state — updated by simulated commands
        self._airborne    = False
        self._altitude    = 0     # cm
        self._yaw         = 0     # degrees
        self._battery     = 87    # percent (realistic fake value)
        self._flight_time = 0     # seconds

        # RC state — updated by send_rc_control
        self._rc_lr  = 0
        self._rc_fb  = 0
        self._rc_ud  = 0
        self._rc_yaw = 0

        # open laptop webcam
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            print("[SIM] WARNING: Could not open webcam — falling back to synthetic frames")
            self._use_webcam = False
        else:
            self._use_webcam = True
            print("[SIM] Webcam opened successfully")

        print("[SIM] Tello simulator initialised — no drone required")
        print("[SIM] All commands will be printed here instead of sent over Wi-Fi\n")

        # thread for capturing/generating video frames
        self._stop_event   = threading.Event()
        self._video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self._video_thread.start()

        # thread for updating simulated flight state
        self._state_thread = threading.Thread(target=self._state_loop, daemon=True)
        self._state_thread.start()

    def __del__(self):
        """Stop background threads and release webcam."""
        self.close()

    def close(self):
        """Release all resources cleanly — call this before exiting."""
        self._stop_event.set()
        if hasattr(self, '_cap') and self._cap.isOpened():
            self._cap.release()

    # ── Video ─────────────────────────────────────────────────────────────

    def read(self):
        """Return the latest camera frame (BGR numpy array)."""
        if self.is_freeze:
            return self.last_frame
        return self.frame

    def video_freeze(self, is_freeze=True):
        """Pause video output -- set is_freeze to True"""
        self.is_freeze = is_freeze
        if is_freeze:
            self.last_frame = self.frame

    def _video_loop(self):
        """Capture webcam frames at ~30 fps, overlaying simulator state info."""
        while not self._stop_event.is_set():
            if self._use_webcam:
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    frame = cv2.flip(frame, 1)  # mirror so it feels natural
                    self.frame = self._overlay_status(frame)
                else:
                    # webcam read failed — fall back to synthetic
                    self.frame = self._make_synthetic_frame()
            else:
                self.frame = self._make_synthetic_frame()
            time.sleep(1 / 30)

    def _overlay_status(self, frame):
        """
        Overlay a small simulator status panel in the top-left corner of a
        real webcam frame so you can see drone state while testing.
        """
        overlay = frame.copy()

        # semi-transparent dark background for the text block
        cv2.rectangle(overlay, (0, 0), (340, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        status_text  = "AIRBORNE" if self._airborne else "LANDED"
        status_color = (0, 220, 0) if self._airborne else (0, 80, 220)

        lines = [
            ("[SIMULATOR]",      (180, 180, 180)),
            (f"Status:   {status_text}",                  status_color),
            (f"Altitude: {self._altitude} cm",            (180, 180, 180)),
            (f"Yaw:      {self._yaw % 360}°",             (180, 180, 180)),
            (f"Battery:  {self._battery}%",               (180, 180, 180)),
            (f"RC fb={self._rc_fb:+d} ud={self._rc_ud:+d} yaw={self._rc_yaw:+d}", (180, 180, 180)),
        ]
        y = 22
        for text, colour in lines:
            cv2.putText(frame, text, (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)
            y += 26

        return frame

    def _make_synthetic_frame(self):
        """Fallback synthetic frame used when the webcam is unavailable."""
        frame = np.full((720, 960, 3), 40, dtype=np.uint8)
        cv2.putText(frame, "No webcam available", (300, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
        return frame

    def _state_loop(self):
        """Update simulated flight state every second."""
        while not self._stop_event.is_set():
            if self._airborne:
                self._flight_time += 1
                # simulate altitude drift
                self._altitude = max(0, self._altitude + self._rc_ud // 10)
                # simulate yaw drift
                self._yaw = (self._yaw + self._rc_yaw // 10) % 360
            time.sleep(1)

    # ── Command interface ─────────────────────────────────────────────────

    def send_command(self, command):
        """
        Print a command to the terminal instead of sending it over UDP.

        :param command: Command to send.
        :return (str): Always returns 'ok' so callers behave normally.
        """
        print(f"[SIM] >> {command}")
        return 'ok'

    def send_rc_control(self, left_right, forward_backward, up_down, yaw):
        self._rc_lr  = int(np.clip(left_right,       -100, 100))
        self._rc_fb  = int(np.clip(forward_backward, -100, 100))
        self._rc_ud  = int(np.clip(up_down,          -100, 100))
        self._rc_yaw = int(np.clip(yaw,              -100, 100))

        if any([left_right, forward_backward, up_down, yaw]):
            print(f"[SIM] rc  lr={self._rc_lr:+4d}  fb={self._rc_fb:+4d}  "
                f"ud={self._rc_ud:+4d}  yaw={self._rc_yaw:+4d}")
    # ── Flight commands ───────────────────────────────────────────────────

    def takeoff(self):
        """Initiates take-off.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.
        """
        self._airborne  = True
        self._altitude  = 80  # Tello hovers at ~80 cm after takeoff
        print("[SIM] ** TAKEOFF — drone is now airborne **")
        return self.send_command('takeoff')

    def land(self):
        """Initiates landing.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.
        """
        self._airborne = False
        self._altitude = 0
        print("[SIM] ** LANDING — drone is now on the ground **")
        return self.send_command('land')

    def flip(self, direction):
        """Flips.

        Args:
            direction (str): Direction to flip, 'l', 'r', 'f', 'b'.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.
        """
        return self.send_command(f'flip {direction}')

    def rotate_cw(self, degrees):
        """Rotates clockwise.

        Args:
            degrees (int): Degrees to rotate, 1 to 360.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.
        """
        self._yaw = (self._yaw + degrees) % 360
        return self.send_command(f'cw {degrees}')

    def rotate_ccw(self, degrees):
        """Rotates counter-clockwise.

        Args:
            degrees (int): Degrees to rotate, 1 to 360.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.
        """
        self._yaw = (self._yaw - degrees) % 360
        return self.send_command(f'ccw {degrees}')

    def move(self, direction, distance):
        """Moves in a direction for a distance.

        This method expects meters or feet. The Tello API expects distances
        from 20 to 500 centimeters.

        Metric: .02 to 5 meters
        Imperial: .7 to 16.4 feet

        Args:
            direction (str): Direction to move, 'forward', 'back', 'right' or 'left'.
            distance (int|float): Distance to move.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.
        """
        distance = float(distance)
        if self.imperial:
            distance = int(round(distance * 30.48))
        else:
            distance = int(round(distance * 100))
        return self.send_command(f'{direction} {distance}')

    def move_backward(self, distance):
        """Moves backward for a distance.

        See comments for Tello.move().

        Args:
            distance (int): Distance to move.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.
        """
        return self.move('back', distance)

    def move_down(self, distance):
        """Moves down for a distance.

        See comments for Tello.move().

        Args:
            distance (int): Distance to move.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.
        """
        if self._airborne:
            self._altitude = max(0, self._altitude - int(distance * 100))
        return self.move('down', distance)

    def move_forward(self, distance):
        """Moves forward for a distance.

        See comments for Tello.move().

        Args:
            distance (int): Distance to move.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.
        """
        return self.move('forward', distance)

    def move_left(self, distance):
        """Moves left for a distance.

        See comments for Tello.move().

        Args:
            distance (int): Distance to move.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.
        """
        return self.move('left', distance)

    def move_right(self, distance):
        """Moves right for a distance.

        See comments for Tello.move().

        Args:
            distance (int): Distance to move.

        """
        return self.move('right', distance)

    def move_up(self, distance):
        """Moves up for a distance.

        See comments for Tello.move().

        Args:
            distance (int): Distance to move.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.
        """
        if self._airborne:
            self._altitude += int(distance * 100)
        return self.move('up', distance)

    # ── Telemetry ─────────────────────────────────────────────────────────

    def set_speed(self, speed):
        """
        Sets speed.

        This method expects KPH or MPH. The Tello API expects speeds from
        1 to 100 centimeters/second.

        Metric: .1 to 3.6 KPH
        Imperial: .1 to 2.2 MPH

        Args:
            speed (int|float): Speed.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.
        """
        speed = float(speed)
        if self.imperial:
            speed = int(round(speed * 44.704))
        else:
            speed = int(round(speed * 27.7778))
        return self.send_command(f'speed {speed}')

    def get_response(self):
        """Returns response of tello.

        Returns:
            int: response of tello.
        """
        return 'ok'

    def get_height(self):
        """Returns height(dm) of tello.

        Returns:
            int: Height(dm) of tello.
        """
        return self._altitude // 10  # convert cm to dm

    def get_battery(self):
        """Returns percent battery life remaining.

        Returns:
            int: Percent battery life remaining.
        """
        return self._battery

    def get_flight_time(self):
        """Returns the number of seconds elapsed during flight.

        Returns:
            int: Seconds elapsed during flight.
        """
        return self._flight_time

    def get_speed(self):
        """Returns the current speed.

        Returns:
            int: Current speed in KPH or MPH.
        """
        return 0