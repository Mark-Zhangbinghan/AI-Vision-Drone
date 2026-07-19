import djitellopy
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import threading
import time
import queue
import sys
from datetime import datetime

# Module-level constants
MOVE_DISTANCE = 30        # cm
ROTATE_DEGREES = 30       # degrees
VIDEO_REFRESH_MS = 30     # ~33fps
STATUS_REFRESH_MS = 2000  # 2s
KEEPALIVE_MS = 10000      # 10s
RECORD_FPS = 30.0                    # video recording frame rate
PHOTO_DEFAULT_EXT = ".jpg"
VIDEO_DEFAULT_EXT = ".mp4"
VIDEO_FOURCC = "mp4v"
BLACK_FRAME_SHAPE = (300, 400, 3)    # djitellopy initial placeholder frame (H, W, C)
CMD_QUEUE_MAXSIZE = 5               # drop excess commands to prevent thread pile-up
CMD_RATE_LIMIT_MS = 150             # minimum ms between same-command dispatches


class DroneController:
    def __init__(self, master):
        self.master = master
        self.master.title("Tello Drone Controller")
        self.master.geometry("1000x650")
        self.master.minsize(900, 600)

        # Drone object
        self.drone = djitellopy.Tello()

        # State attributes
        self._connected = False
        self.frame_read = None
        self._photo = None
        self._streaming = False
        self._video_job = None
        self._status_job = None
        self._keepalive_job = None

        # Media capture state
        self._recording = False
        self._video_writer = None
        self._recorder_thread = None
        self._stop_recording_flag = threading.Event()
        self._record_file_path = None
        self._btn_default_bg = None  # saved default bg for record button

        # Command queue — bounded to prevent pile-up under rapid-fire input
        self._cmd_queue = queue.Queue(maxsize=CMD_QUEUE_MAXSIZE)
        self._cmd_cooldown = {}  # {fn_name: last_dispatch_time}
        self._cmd_worker = threading.Thread(target=self._cmd_loop, daemon=True)
        self._cmd_worker.start()

        # Build UI
        self._build_ui()

        # Register window close handler
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Connect to drone
        self._connect_drone()

        # Start periodic loops
        self.update_status()
        if self._connected:
            self.send_keepalive()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        """Configure main window grid layout."""
        self.master.columnconfigure(0, weight=0)   # control panel fixed width
        self.master.columnconfigure(1, weight=1)   # video area expandable
        self.master.rowconfigure(0, weight=1)      # main content area expandable
        self.master.rowconfigure(1, weight=0)      # status bar fixed height

        self._build_left_panel()
        self._build_video_panel()
        self._build_status_bar()

    def _build_left_panel(self):
        """Left control panel with 3 grouped LabelFrames."""
        self.left_panel = tk.Frame(self.master)
        self.left_panel.grid(row=0, column=0, sticky="ns", padx=10, pady=10)

        # --- Flight Control group ---
        self.flight_frame = tk.LabelFrame(self.left_panel, text="Flight Control")
        self.flight_frame.pack(fill="x", pady=(0, 10))
        self.flight_frame.columnconfigure(0, weight=1)
        self.flight_frame.columnconfigure(1, weight=1)
        self.takeoff_button = tk.Button(self.flight_frame, text="Take Off",
                                        command=self.takeoff)
        self.takeoff_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.land_button = tk.Button(self.flight_frame, text="Land",
                                     command=self.land)
        self.land_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # --- Movement group ---
        self.move_frame = tk.LabelFrame(self.left_panel, text="Movement")
        self.move_frame.pack(fill="x", pady=(0, 10))

        # D-pad sub-frame (3x3 cross layout)
        self.dpad_frame = tk.Frame(self.move_frame)
        self.dpad_frame.pack(fill="x")
        self.forward_button = tk.Button(self.dpad_frame, text="Forward",
                                        command=self.move_forward)
        self.forward_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.left_button = tk.Button(self.dpad_frame, text="Left",
                                     command=self.move_left)
        self.left_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.right_button = tk.Button(self.dpad_frame, text="Right",
                                      command=self.move_right)
        self.right_button.grid(row=1, column=2, padx=5, pady=5, sticky="ew")
        self.backward_button = tk.Button(self.dpad_frame, text="Backward",
                                         command=self.move_backward)
        self.backward_button.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        self.dpad_frame.columnconfigure(0, weight=1)
        self.dpad_frame.columnconfigure(1, weight=1)
        self.dpad_frame.columnconfigure(2, weight=1)

        # Separator
        tk.Frame(self.move_frame, height=2, bd=1, relief="sunken").pack(
            fill="x", pady=5)

        # Extra movement sub-frame (up/down + rotate)
        self.extra_move_frame = tk.Frame(self.move_frame)
        self.extra_move_frame.pack(fill="x")
        self.extra_move_frame.columnconfigure(0, weight=1)
        self.extra_move_frame.columnconfigure(1, weight=1)
        self.up_button = tk.Button(self.extra_move_frame, text="Up",
                                   command=self.move_up)
        self.up_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.down_button = tk.Button(self.extra_move_frame, text="Down",
                                     command=self.move_down)
        self.down_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.rotate_cw_button = tk.Button(self.extra_move_frame,
                                          text="Rotate CW",
                                          command=self.rotate_cw)
        self.rotate_cw_button.grid(row=0, column=1, padx=5, pady=5,
                                   sticky="ew")
        self.rotate_ccw_button = tk.Button(self.extra_move_frame,
                                           text="Rotate CCW",
                                           command=self.rotate_ccw)
        self.rotate_ccw_button.grid(row=1, column=1, padx=5, pady=5,
                                    sticky="ew")

        # --- Camera group ---
        self.camera_frame = tk.LabelFrame(self.left_panel, text="Camera")
        self.camera_frame.pack(fill="x")
        self.start_camera_button = tk.Button(self.camera_frame,
                                             text="Start Camera",
                                             command=self.start_video)
        self.start_camera_button.pack(fill="x", padx=5, pady=2)
        self.stop_camera_button = tk.Button(self.camera_frame,
                                            text="Stop Camera",
                                            command=self.stop_video,
                                            state="disabled")
        self.stop_camera_button.pack(fill="x", padx=5, pady=2)

        # --- Media Capture group ---
        self.media_frame = tk.LabelFrame(self.left_panel, text="Media Capture")
        self.media_frame.pack(fill="x", pady=(10, 0))

        self.capture_photo_button = tk.Button(
            self.media_frame, text="Capture Photo",
            command=self.take_photo, state="disabled")
        self.capture_photo_button.pack(fill="x", padx=5, pady=2)

        self.record_button = tk.Button(
            self.media_frame, text="Start Recording",
            command=self.toggle_recording, state="disabled")
        self.record_button.pack(fill="x", padx=5, pady=2)
        self._btn_default_bg = self.record_button.cget("bg")

    def _build_video_panel(self):
        """Right video display panel."""
        self.video_frame = tk.LabelFrame(self.master, text="Live Video")
        self.video_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.video_frame.rowconfigure(0, weight=1)
        self.video_frame.columnconfigure(0, weight=1)

        self.video_label = tk.Label(
            self.video_frame,
            text="Camera Off\nClick 'Start Camera'",
            bg="black", fg="white")
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    def _build_status_bar(self):
        """Bottom status bar."""
        self.status_bar = tk.Label(
            self.master,
            text="Battery: --  |  Height: --  |  Status: Connecting...",
            relief="sunken", bd=1, anchor="w", padx=10, pady=5)
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")

    def _set_control_state(self, enabled):
        """Enable/disable flight + movement + camera + media buttons (offline mode)."""
        state = "normal" if enabled else "disabled"
        for btn in (self.takeoff_button, self.land_button,
                    self.forward_button, self.backward_button,
                    self.left_button, self.right_button,
                    self.up_button, self.down_button,
                    self.rotate_cw_button, self.rotate_ccw_button,
                    self.start_camera_button,
                    self.capture_photo_button, self.record_button):
            btn.config(state=state)

    # ------------------------------------------------------------------ #
    # Drone connection
    # ------------------------------------------------------------------ #
    def _connect_drone(self):
        """Connect to Tello; keep GUI open in offline mode on failure."""
        try:
            self.drone.connect()
            self._connected = True
        except Exception as e:
            self._connected = False
            messagebox.showwarning(
                "Connection",
                f"Could not connect to Tello:\n{e}\n"
                "GUI opened in offline mode.")
        self._set_control_state(self._connected)

    # ------------------------------------------------------------------ #
    # Command dispatch (dedicated thread — never blocks the main loop)
    # ------------------------------------------------------------------ #
    def _enqueue_cmd(self, fn, *args):
        """Push a drone command onto the bounded worker queue.

        Skips duplicate commands within CMD_RATE_LIMIT_MS (cooldown)
        and silently drops when the queue is full (protects video stream).
        """
        now = time.time()
        name = fn.__name__
        last = self._cmd_cooldown.get(name, 0)
        if now - last < CMD_RATE_LIMIT_MS / 1000.0:
            return  # cooldown active — ignore rapid duplicate
        try:
            self._cmd_queue.put_nowait((fn, args))
        except queue.Full:
            pass  # queue full — silently drop, video stays smooth

    def _cmd_loop(self):
        """Dedicated thread: dequeues and executes drone commands sequentially."""
        while True:
            item = self._cmd_queue.get()
            if item is None:          # sentinel — shutdown
                break
            fn, args = item
            if not self._connected:    # skip commands while disconnected
                continue
            try:
                fn(*args)
            except Exception:
                pass  # silently drop; connection errors caught by update_status
            self._cmd_cooldown[fn.__name__] = time.time()  # reset cooldown

    # ------------------------------------------------------------------ #
    # Flight commands (non-blocking — dispatched to worker thread)
    # ------------------------------------------------------------------ #
    def takeoff(self):
        self._enqueue_cmd(self.drone.takeoff)

    def land(self):
        self._enqueue_cmd(self.drone.land)

    # ------------------------------------------------------------------ #
    # Movement commands (non-blocking — dispatched to worker thread)
    # ------------------------------------------------------------------ #
    def move_forward(self):
        self._enqueue_cmd(self.drone.move_forward, MOVE_DISTANCE)

    def move_backward(self):
        self._enqueue_cmd(self.drone.move_back, MOVE_DISTANCE)

    def move_left(self):
        self._enqueue_cmd(self.drone.move_left, MOVE_DISTANCE)

    def move_right(self):
        self._enqueue_cmd(self.drone.move_right, MOVE_DISTANCE)

    def move_up(self):
        self._enqueue_cmd(self.drone.move_up, MOVE_DISTANCE)

    def move_down(self):
        self._enqueue_cmd(self.drone.move_down, MOVE_DISTANCE)

    def rotate_cw(self):
        self._enqueue_cmd(self.drone.rotate_clockwise, ROTATE_DEGREES)

    def rotate_ccw(self):
        self._enqueue_cmd(self.drone.rotate_counter_clockwise, ROTATE_DEGREES)

    # ------------------------------------------------------------------ #
    # Camera / video stream
    # ------------------------------------------------------------------ #
    def start_video(self):
        """Start the Tello video stream and the display loop."""
        try:
            self.drone.streamon()
            self.frame_read = self.drone.get_frame_read(with_queue=True, max_queue_len=32)
            self._streaming = True
            self.start_camera_button.config(state="disabled")
            self.stop_camera_button.config(state="normal")
            self.capture_photo_button.config(state="normal")
            self.record_button.config(state="normal")
            self.update_video()
        except Exception as e:
            self._streaming = False
            messagebox.showerror("Camera Error",
                                 f"Failed to start camera:\n{e}")

    def stop_video(self):
        """Stop the video stream and clear the display."""
        # Stop recording first if active (streamoff destroys frame_read)
        if self._recording:
            self.stop_recording()

        self._streaming = False
        if self._video_job is not None:
            self.master.after_cancel(self._video_job)
            self._video_job = None
        try:
            self.drone.streamoff()
        except Exception:
            pass
        self.frame_read = None
        self._photo = None
        self.video_label.config(image="",
                                text="Camera Off\nClick 'Start Camera'")
        self.start_camera_button.config(state="normal")
        self.stop_camera_button.config(state="disabled")
        self.capture_photo_button.config(state="disabled")
        self.record_button.config(state="disabled")

    def update_video(self):
        """Periodically refresh the video label with the latest frame.
        
        In queue mode, popleft() returns the oldest frame.  We drain all
        queued frames and render only the newest one — zero latency.
        """
        if not self._streaming or self.frame_read is None:
            return
        # Drain the queue: pop every frame, keep the last one
        frame = None
        while True:
            f = self.frame_read.frame  # popleft() in queue mode
            if f is None:
                break
            frame = f                  # overwrite → latest wins
        if frame is not None:
            w = self.video_label.winfo_width()
            h = self.video_label.winfo_height()
            if w < 10 or h < 10:
                w, h = 640, 480
            img = Image.fromarray(frame)            # RGB numpy -> PIL
            img = img.resize((w, h))                # fit to label
            self._photo = ImageTk.PhotoImage(img)   # keep ref to avoid GC
            self.video_label.config(image=self._photo, text="")
        self._video_job = self.master.after(VIDEO_REFRESH_MS, self.update_video)

    # ------------------------------------------------------------------ #
    # Media capture (photo + video recording)
    # ------------------------------------------------------------------ #
    def _is_black_placeholder(self, frame):
        """Check if frame is the djitellopy initial black placeholder (300x400)."""
        if frame is None:
            return True
        if frame.shape == BLACK_FRAME_SHAPE and not frame.any():
            return True
        return False

    def take_photo(self):
        """Capture current video frame and save as an image file."""
        if not self._streaming or self.frame_read is None:
            messagebox.showwarning("Capture Photo",
                                   "Camera stream is not running.")
            return

        frame = self.frame_read.frame  # RGB numpy, thread-safe
        if self._is_black_placeholder(frame):
            messagebox.showwarning("Capture Photo",
                                   "No valid frame yet (black placeholder). "
                                   "Please wait a moment and retry.")
            return

        default_name = "tello_photo_" + datetime.now().strftime("%Y%m%d_%H%M%S") \
                       + PHOTO_DEFAULT_EXT
        file_path = filedialog.asksaveasfilename(
            title="Save Photo",
            defaultextension=PHOTO_DEFAULT_EXT,
            initialfile=default_name,
            filetypes=[("JPEG image", "*.jpg"), ("PNG image", "*.png"),
                       ("All files", "*.*")])
        if not file_path:  # user cancelled
            return

        try:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ok = cv2.imwrite(file_path, bgr)
            if not ok:
                raise RuntimeError("cv2.imwrite returned False")
            messagebox.showinfo("Capture Photo",
                                f"Photo saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Capture Photo",
                                 f"Failed to save photo:\n{e}")

    def toggle_recording(self):
        """Toggle video recording on/off based on current state."""
        if self._recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        """Start video recording: pick path -> create VideoWriter -> launch thread."""
        if not self._streaming or self.frame_read is None:
            messagebox.showwarning("Recording",
                                   "Camera stream is not running.")
            return
        if self._recording:
            return

        # Use first frame to determine dimensions and validity
        frame = self.frame_read.frame
        if self._is_black_placeholder(frame):
            messagebox.showwarning("Recording",
                                   "No valid frame yet (black placeholder). "
                                   "Please wait a moment and retry.")
            return

        default_name = "tello_video_" + datetime.now().strftime("%Y%m%d_%H%M%S") \
                       + VIDEO_DEFAULT_EXT
        file_path = filedialog.asksaveasfilename(
            title="Save Video",
            defaultextension=VIDEO_DEFAULT_EXT,
            initialfile=default_name,
            filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")])
        if not file_path:  # user cancelled
            return

        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
        try:
            writer = cv2.VideoWriter(file_path, fourcc, RECORD_FPS, (w, h))
            if not writer.isOpened():
                raise RuntimeError("VideoWriter failed to open "
                                   "(codec mp4v unavailable?)")
        except Exception as e:
            messagebox.showerror("Recording",
                                 f"Failed to create video writer:\n{e}")
            return

        # Write first frame immediately
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        self._video_writer = writer
        self._record_file_path = file_path
        self._stop_recording_flag.clear()
        self._recording = True
        self._recorder_thread = threading.Thread(
            target=self._record_loop, daemon=True)
        self._recorder_thread.start()

        self.record_button.config(text="Stop Recording",
                                  bg="#d9534f", fg="white")
        self.capture_photo_button.config(state="disabled")

    def _record_loop(self):
        """Background recording thread: read frame -> RGB->BGR -> write."""
        frame_interval = 1.0 / RECORD_FPS
        writer = self._video_writer
        while not self._stop_recording_flag.is_set():
            try:
                if self.frame_read is None:
                    break
                frame = self.frame_read.frame
                if frame is None or self._is_black_placeholder(frame):
                    time.sleep(frame_interval)
                    continue
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
            except Exception:
                pass
            time.sleep(frame_interval)

    def stop_recording(self):
        """Stop recording: signal thread -> join -> release writer -> restore UI."""
        if not self._recording:
            return

        self._stop_recording_flag.set()

        if self._recorder_thread is not None:
            self._recorder_thread.join(timeout=2.0)
            self._recorder_thread = None

        if self._video_writer is not None:
            try:
                self._video_writer.release()
            except Exception:
                pass
            self._video_writer = None

        saved_path = self._record_file_path
        self._recording = False
        self._record_file_path = None

        self.record_button.config(text="Start Recording",
                                  bg=self._btn_default_bg, fg="black")
        if self._streaming:
            self.capture_photo_button.config(state="normal")

        if saved_path:
            messagebox.showinfo("Recording",
                                f"Video saved to:\n{saved_path}")

    # ------------------------------------------------------------------ #
    # Status / keepalive / cleanup
    # ------------------------------------------------------------------ #
    def update_status(self):
        """Refresh the status bar every STATUS_REFRESH_MS."""
        battery, height, status = "?", "?", "Disconnected"
        try:
            if self._connected:
                battery = f"{self.drone.get_battery()}%"
                height = f"{self.drone.get_height()} cm"
                status = "Connected"
        except Exception:
            status = "Error"
        rec_tag = "[REC] " if self._recording else ""
        self.status_bar.config(
            text=f"{rec_tag}Battery: {battery}  |  Height: {height}  |  "
                 f"Status: {status}")
        self._status_job = self.master.after(STATUS_REFRESH_MS,
                                             self.update_status)

    # ------------------------------------------------------------------ #
    # Keepalive helper — SDK 1.3 compatible
    # ------------------------------------------------------------------ #
    def _keepalive_cmd(self):
        """Reset the 15s auto-land timer with a universally recognised command.

        djitellopy.send_keepalive() sends 'keepalive' which only Tello EDU /
        SDK-2.0 firmware understands.  The original Tello (SDK 1.3) returns
        'unknown command: keepalive'.  Re-sending 'command' is harmless on
        all firmware versions and reliably resets the timer.
        """
        self.drone.send_command_without_return("command")

    def send_keepalive(self):
        """Enqueue a keepalive every KEEPALIVE_MS (non-blocking)."""
        self._enqueue_cmd(self._keepalive_cmd)
        self._keepalive_job = self.master.after(KEEPALIVE_MS,
                                                self.send_keepalive)

    def on_closing(self):
        """Clean up resources on window close."""
        # Stop command worker thread
        self._cmd_queue.put(None)
        # Stop recording first and release VideoWriter
        if self._recording:
            self.stop_recording()

        self._streaming = False
        for job in (self._video_job, self._status_job, self._keepalive_job):
            if job is not None:
                self.master.after_cancel(job)
        try:
            self.drone.end()  # lands if flying, streamoff if streaming
        except Exception:
            pass
        self.master.destroy()


if __name__ == "__main__":
    # Bump process priority so video decoding and PIL get more CPU slices
    if sys.platform == "win32":
        try:
            import ctypes
            HIGH_PRIORITY_CLASS = 0x00000080
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(),
                HIGH_PRIORITY_CLASS)
        except Exception:
            pass
    root = tk.Tk()
    app = DroneController(root)
    root.mainloop()
