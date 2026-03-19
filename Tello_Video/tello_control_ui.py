from PIL import Image
from PIL import ImageTk
import tkinter as tki
from tkinter import Toplevel, Scale
import threading
import datetime
import cv2
import os
import time
import mediapipe as mp
import pickle
from collections import deque

# mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# gesture labels
GESTURES = {
    0: ('Open Palm',   'Hover / Stop'),
    1: ('Fist',        'Land'),
    2: ('Point Up',    'Move Up'),
    3: ('Point Down',  'Move Down'),
    4: ('Point Left',  'Rotate Left'),
    5: ('Point Right', 'Rotate Right'),
}


class TelloUI:
    """Wrapper class to enable the GUI."""

    def __init__(self, tello, outputpath):
        """
        Initial all the element of the GUI, support by Tkinter.

        :param tello: class interacts with the Tello drone.
        :param outputpath: path to save snapshots.
        """

        self.tello = tello
        self.outputPath = outputpath
        self.frame = None
        self.thread = None
        self.stopEvent = None

        # control variables
        self.distance = 0.2  # default distance for 'move' cmd
        self.degree = 30     # default degree for 'cw' or 'ccw' cmd

        self.quit_waiting_flag = False

        # gesture control setup
        self.gesture_mode = False
        self.gesture_buffer = deque(maxlen=15)
        with open('model/gesture_model.pkl', 'rb') as f:
            self.gesture_model = pickle.load(f)
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7)
        print("[INFO] Gesture model loaded")

        # initialize the root window and image panel
        self.root = tki.Tk()
        self.panel = None

        # create buttons
        self.btn_snapshot = tki.Button(self.root, text="Snapshot!",
                                       command=self.takeSnapshot)
        self.btn_snapshot.pack(side="bottom", fill="both",
                               expand="yes", padx=10, pady=5)

        self.btn_pause = tki.Button(self.root, text="Pause", relief="raised",
                                    command=self.pauseVideo)
        self.btn_pause.pack(side="bottom", fill="both",
                            expand="yes", padx=10, pady=5)

        self.btn_cmd = tki.Button(self.root, text="Open Command Panel",
                                  relief="raised", command=self.openCmdWindow)
        self.btn_cmd.pack(side="bottom", fill="both",
                          expand="yes", padx=10, pady=5)

        # gesture toggle button
        self.btn_gesture = tki.Button(self.root, text="Gesture Control: OFF",
                                      relief="raised", command=self.toggleGestureMode)
        self.btn_gesture.pack(side="bottom", fill="both",
                              expand="yes", padx=10, pady=5)

        # start video loop thread
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

        # set a callback to handle when the window is closed
        self.root.wm_title("TELLO Controller")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

        # sending_command sends 'command' to tello every 5 seconds to keep connection alive
        self.sending_command_thread = threading.Thread(target=self._sendingCommand)

    def videoLoop(self):
        """
        The mainloop thread of Tkinter.
        Raises:
            RuntimeError: To get around a RunTime error that Tkinter throws due to threading.
        """
        try:
            time.sleep(0.5)
            self.sending_command_thread.start()

            while not self.stopEvent.is_set():
                self.frame = self.tello.read()

                if self.frame is None or self.frame.size == 0:
                    continue

                # --- gesture recognition ---
                if self.gesture_mode:
                    rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    result = self.hands.process(rgb)
                    if result.multi_hand_landmarks:
                        lm = result.multi_hand_landmarks[0].landmark
                        keypoints = [v for point in lm for v in (point.x, point.y)]
                        pred = self.gesture_model.predict([keypoints])[0]
                        self.gesture_buffer.append(pred)
                        if len(self.gesture_buffer) == 15 and len(set(self.gesture_buffer)) == 1:
                            self.dispatchGestureCommand(pred)
                    else:
                        self.gesture_buffer.clear()
                # --- end gesture recognition ---

                image = Image.fromarray(self.frame)
                self.root.after(0, self._updateGUIImage, image)
                time.sleep(0.03)

        except RuntimeError as e:
            print("[INFO] caught a RuntimeError")

    def _updateGUIImage(self, image):
        """Main operation to update the GUI panel."""
        image = ImageTk.PhotoImage(image)
        if self.panel is None:
            self.panel = tki.Label(image=image)
            self.panel.image = image
            self.panel.pack(side="left", padx=10, pady=10)
        else:
            self.panel.configure(image=image)
            self.panel.image = image

    def _sendingCommand(self):
        """Send 'command' to tello every 5 seconds to keep connection alive."""
        while not self.stopEvent.is_set():
            if hasattr(self, 'tello'):
                self.tello.send_command('command')
            time.sleep(5)

    def _setQuitWaitingFlag(self):
        """Set the quit waiting flag to True."""
        self.quit_waiting_flag = True

    def toggleGestureMode(self):
        """Toggle gesture control on or off."""
        self.gesture_mode = not self.gesture_mode
        self.gesture_buffer.clear()
        if self.gesture_mode:
            self.btn_gesture.config(text="Gesture Control: ON", relief="sunken")
            print("[INFO] Gesture control ON")
        else:
            self.btn_gesture.config(text="Gesture Control: OFF", relief="raised")
            print("[INFO] Gesture control OFF")

    def dispatchGestureCommand(self, gesture_id):
        """Map a recognised gesture ID to a drone command."""
        gesture_name, command = GESTURES[gesture_id]
        print(f"[GESTURE] {gesture_name} → {command}")
        if gesture_id == 0:
            self.tello.send_command('stop')    # hover / stop
        elif gesture_id == 1:
            self.telloLanding()                # land
        elif gesture_id == 2:
            self.telloUp(self.distance)        # move up
        elif gesture_id == 3:
            self.telloDown(self.distance)      # move down
        elif gesture_id == 4:
            self.tello.rotate_ccw(self.degree) # rotate left
        elif gesture_id == 5:
            self.tello.rotate_cw(self.degree)  # rotate right

    def openCmdWindow(self):
        """Open the command panel window."""
        panel = Toplevel(self.root)
        panel.wm_title("Command Panel")

        text0 = tki.Label(panel,
                          text='This Controller maps keyboard inputs to Tello control commands\n'
                               'Adjust the trackbar to reset distance and degree parameters',
                          font='Helvetica 10 bold')
        text0.pack(side='top')

        text1 = tki.Label(panel, text=
                          'W - Move Tello Up\t\t\tArrow Up - Move Tello Forward\n'
                          'S - Move Tello Down\t\t\tArrow Down - Move Tello Backward\n'
                          'A - Rotate Tello Counter-Clockwise\tArrow Left - Move Tello Left\n'
                          'D - Rotate Tello Clockwise\t\tArrow Right - Move Tello Right',
                          justify="left")
        text1.pack(side="top")

        self.btn_landing = tki.Button(panel, text="Land", relief="raised",
                                      command=self.telloLanding)
        self.btn_landing.pack(side="bottom", fill="both",
                              expand="yes", padx=10, pady=5)

        self.btn_takeoff = tki.Button(panel, text="Takeoff", relief="raised",
                                      command=self.telloTakeOff)
        self.btn_takeoff.pack(side="bottom", fill="both",
                              expand="yes", padx=10, pady=5)

        # binding arrow keys to drone control
        self.tmp_f = tki.Frame(panel, width=100, height=2)
        self.tmp_f.bind('<KeyPress-w>', self.on_keypress_w)
        self.tmp_f.bind('<KeyPress-s>', self.on_keypress_s)
        self.tmp_f.bind('<KeyPress-a>', self.on_keypress_a)
        self.tmp_f.bind('<KeyPress-d>', self.on_keypress_d)
        self.tmp_f.bind('<KeyPress-Up>', self.on_keypress_up)
        self.tmp_f.bind('<KeyPress-Down>', self.on_keypress_down)
        self.tmp_f.bind('<KeyPress-Left>', self.on_keypress_left)
        self.tmp_f.bind('<KeyPress-Right>', self.on_keypress_right)
        self.tmp_f.pack(side="bottom")
        self.tmp_f.focus_set()

        self.btn_flip = tki.Button(panel, text="Flip", relief="raised",
                                   command=self.openFlipWindow)
        self.btn_flip.pack(side="bottom", fill="both",
                           expand="yes", padx=10, pady=5)

        self.distance_bar = Scale(panel, from_=0.02, to=5, tickinterval=0.01,
                                  digits=3, label='Distance(m)', resolution=0.01)
        self.distance_bar.set(0.2)
        self.distance_bar.pack(side="left")

        self.btn_distance = tki.Button(panel, text="Reset Distance", relief="raised",
                                       command=self.updateDistancebar)
        self.btn_distance.pack(side="left", fill="both",
                               expand="yes", padx=10, pady=5)

        self.degree_bar = Scale(panel, from_=1, to=360, tickinterval=10, label='Degree')
        self.degree_bar.set(30)
        self.degree_bar.pack(side="right")

        self.btn_degree = tki.Button(panel, text="Reset Degree", relief="raised",
                                     command=self.updateDegreebar)
        self.btn_degree.pack(side="right", fill="both",
                             expand="yes", padx=10, pady=5)

    def openFlipWindow(self):
        """Open the flip command window."""
        panel = Toplevel(self.root)
        panel.wm_title("Flip")

        self.btn_flipl = tki.Button(panel, text="Flip Left", relief="raised",
                                    command=self.telloFlip_l)
        self.btn_flipl.pack(side="bottom", fill="both",
                            expand="yes", padx=10, pady=5)

        self.btn_flipr = tki.Button(panel, text="Flip Right", relief="raised",
                                    command=self.telloFlip_r)
        self.btn_flipr.pack(side="bottom", fill="both",
                            expand="yes", padx=10, pady=5)

        self.btn_flipf = tki.Button(panel, text="Flip Forward", relief="raised",
                                    command=self.telloFlip_f)
        self.btn_flipf.pack(side="bottom", fill="both",
                            expand="yes", padx=10, pady=5)

        self.btn_flipb = tki.Button(panel, text="Flip Backward", relief="raised",
                                    command=self.telloFlip_b)
        self.btn_flipb.pack(side="bottom", fill="both",
                            expand="yes", padx=10, pady=5)

    def takeSnapshot(self):
        """Save the current frame as a jpg file."""
        if self.frame is None:
            print("[WARN] No frame available yet")
            return
        ts = datetime.datetime.now()
        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
        p = os.path.sep.join((self.outputPath, filename))
        cv2.imwrite(p, cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))
        print("[INFO] saved {}".format(filename))

    def pauseVideo(self):
        """Toggle freeze/unfreeze of video."""
        if self.btn_pause.config('relief')[-1] == 'sunken':
            self.btn_pause.config(relief="raised")
            self.tello.video_freeze(False)
        else:
            self.btn_pause.config(relief="sunken")
            self.tello.video_freeze(True)

    def telloTakeOff(self):
        return self.tello.takeoff()

    def telloLanding(self):
        return self.tello.land()

    def telloFlip_l(self):
        return self.tello.flip('l')

    def telloFlip_r(self):
        return self.tello.flip('r')

    def telloFlip_f(self):
        return self.tello.flip('f')

    def telloFlip_b(self):
        return self.tello.flip('b')

    def telloCW(self, degree):
        return self.tello.rotate_cw(degree)

    def telloCCW(self, degree):
        return self.tello.rotate_ccw(degree)

    def telloMoveForward(self, distance):
        return self.tello.move_forward(distance)

    def telloMoveBackward(self, distance):
        return self.tello.move_backward(distance)

    def telloMoveLeft(self, distance):
        return self.tello.move_left(distance)

    def telloMoveRight(self, distance):
        return self.tello.move_right(distance)

    def telloUp(self, dist):
        return self.tello.move_up(dist)

    def telloDown(self, dist):
        return self.tello.move_down(dist)

    def updateDistancebar(self):
        try:
            self.distance = self.distance_bar.get()
            print('reset distance to %.2f' % self.distance)
        except Exception:
            pass

    def updateDegreebar(self):
        try:
            self.degree = self.degree_bar.get()
            print('reset degree to %d' % self.degree)
        except Exception:
            pass

    def on_keypress_w(self, event):
        print("up %d m" % self.distance)
        self.telloUp(self.distance)

    def on_keypress_s(self, event):
        print("down %d m" % self.distance)
        self.telloDown(self.distance)

    def on_keypress_a(self, event):
        print("ccw %d degree" % self.degree)
        self.tello.rotate_ccw(self.degree)

    def on_keypress_d(self, event):
        print("cw %d degree" % self.degree)
        self.tello.rotate_cw(self.degree)

    def on_keypress_up(self, event):
        print("forward %d m" % self.distance)
        self.telloMoveForward(self.distance)

    def on_keypress_down(self, event):
        print("backward %d m" % self.distance)
        self.telloMoveBackward(self.distance)

    def on_keypress_left(self, event):
        print("left %d m" % self.distance)
        self.telloMoveLeft(self.distance)

    def on_keypress_right(self, event):
        print("right %d m" % self.distance)
        self.telloMoveRight(self.distance)

    def on_keypress_enter(self, event):
        if self.frame is not None:
            self.registerFace()
        self.tmp_f.focus_set()

    def onClose(self):
        """Set the stop event and clean up on window close."""
        print("[INFO] closing...")
        self.stopEvent.set()
        self.hands.close()
        del self.tello
        self.root.quit()