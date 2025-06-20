import tkinter as tk
from threading import Thread
from camWDCalc import run_apriltag_capture
from focusHelper import run_focusApp
from liveFeedback import run_live_feedback

# Configure your camera URL here!
# camera_url = "http://root:tJe9R87pDxG6b62@192.168.110.150/axis-cgi/mjpg/video.cgi" # Axis Camera
# camera_url = "rtsp://192.168.50.175/profile2/media.smp" # Wisenet Camera
# camera_url = "rtsp://service:smB0ston!@10.200.11.33:554/" # Bosch Camera
camera_url = "rtsp://10.200.11.33/axis-media/media.amp"
# camera_url = 0 # For local built-in camera

# 
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Multi-Function App")
        self.geometry("1080x720")
        self.current_frame = None
        self.show_home()

    def switch_frame(self, new_frame_class):
        if self.current_frame:
            self.current_frame.destroy()
        self.current_frame = new_frame_class(self)
        self.current_frame.pack(fill="both", expand=True)

    def show_home(self):
        self.switch_frame(HomeScreen)

    def show_function1(self):
        self.switch_frame(Function1Screen)

    def show_function2(self):
        self.switch_frame(Function2Screen)

    def show_function3(self):
        self.switch_frame(Function3Screen)

class HomeScreen(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        tk.Label(self, text="Home Screen", font=("Arial", 16)).pack(pady=20)
        tk.Button(self, text="Function 1", command=master.show_function1).pack(pady=5)
        tk.Button(self, text="Function 2", command=master.show_function2).pack(pady=5)
        tk.Button(self, text="Function 3", command=master.show_function3).pack(pady=5)

class Function1Screen(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        tk.Label(self, text="Function 1 Screen", font=("Arial", 16)).pack(pady=20)
        tk.Label(self, text="Press ESC to return to home").pack()

        # Start button
        tk.Button(self, text="Start Focus Helper", command=self.start_processing).pack(pady=10)

        self.bind_all("<Escape>", lambda e: master.show_home())

    def start_processing(self):
        Thread(target=run_focusApp, args=(camera_url,)).start()


class Function2Screen(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        tk.Label(self, text="Function 2 Screen", font=("Arial", 16)).pack(pady=10)
        tk.Label(self, text="Press ESC to return to home").pack()

        # Input fields
        self.focal_entry = self._create_labeled_entry("Focal Length (mm):")
        self.fov_entry = self._create_labeled_entry("FOV (deg):")
        self.sensor_w_entry = self._create_labeled_entry("Sensor Width (mm):")
        self.sensor_h_entry = self._create_labeled_entry("Sensor Height (mm):")

        # Start button
        tk.Button(self, text="Start AprilTag Capture", command=self.start_processing).pack(pady=10)

        # Escape key binding
        self.bind_all("<Escape>", lambda e: master.show_home())

    def _create_labeled_entry(self, label_text):
        frame = tk.Frame(self)
        frame.pack(pady=2)
        tk.Label(frame, text=label_text, width=20, anchor="w").pack(side="left")
        entry = tk.Entry(frame)
        entry.pack(side="left")
        return entry

    def start_processing(self):
        try:
            F_MM = float(self.focal_entry.get())
            FOV_DEG = float(self.fov_entry.get())
            SENSOR_W_MM = float(self.sensor_w_entry.get())
            SENSOR_H_MM = float(self.sensor_h_entry.get())
        except ValueError:
            print("Invalid input — please enter numeric values.")
            return

        # Start processing in a new thread with GUI values
        Thread(target=run_apriltag_capture, args=(camera_url, F_MM, FOV_DEG, SENSOR_W_MM, SENSOR_H_MM)).start()


class Function3Screen(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        tk.Label(self, text="Function 3 Screen", font=("Arial", 16)).pack(pady=20)
        tk.Label(self, text="Press ESC to return to home").pack()

        # Start button
        tk.Button(self, text="Start Live Feedback", command=self.start_processing).pack(pady=10)

        self.bind_all("<Escape>", lambda e: master.show_home())

    def start_processing(self):
        Thread(target=run_live_feedback, args=(camera_url,)).start()

if __name__ == "__main__":
    app = App()
    app.mainloop()