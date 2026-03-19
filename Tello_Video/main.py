import tello
from tello_control_ui import TelloUI
import os


def main():

    # make sure img folder exists so snapshots don't crash
    os.makedirs('./img', exist_ok=True)

    drone = tello.Tello('0.0.0.0', 8889) 
    vplayer = TelloUI(drone, "./img/")

    # start the Tkinter mainloop
    vplayer.root.mainloop()

if __name__ == "__main__":
    main()