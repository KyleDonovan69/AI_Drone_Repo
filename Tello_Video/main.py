import os
import warnings

# get rid of those annoying ass warnings it was spamming
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

from tello_control_ui import TelloUI

# ── Simulator toggle ──────────────────────────────────────────────────────────
# Set USE_SIMULATOR = True to Use the drone sim since I'm not buying a 300 euro drone
# Set USE_SIMULATOR = False when connected to the real Tello drone
USE_SIMULATOR = True

if USE_SIMULATOR:
    import tello_sim as tello
    print("[INFO] Running in SIM mode no drone required")
else:
    import tello
    print("[INFO] Running with REAL drone")
# ─────────────────────────────────────────────────────────────────────────────


def main():

    # make sure img folder exists so snapshots don't crash
    os.makedirs('./img', exist_ok=True)

    drone = tello.Tello('0.0.0.0', 8889)
    vplayer = TelloUI(drone, "./img/")

    try:
        # start the Tkinter mainloop
        vplayer.root.mainloop()
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt received — shutting down cleanly")
    finally:
        # ensure everything shuts down whether we Ctrl+C or close the window
        if not vplayer.stopEvent.is_set():
            vplayer.onClose()

if __name__ == "__main__":
    main()