"""
AI Object Detection and Recognition
=====================================
Desktop application built with CustomTkinter + OpenCV.

Libraries used:
  - opencv-python   : webcam capture, frame processing, CLAHE enhancement
  - ultralytics     : YOLOv8 object detection model (inference + plotting)
  - customtkinter   : modern dark-themed desktop GUI
  - Pillow (PIL)    : image conversion for display inside the GUI
  - numpy           : array operations and colour generation

Features:
  - Live webcam feed with real-time object detection (YOLOv8)
  - Webcam selector — scans all connected cameras and lets you pick one
  - Load and detect objects in still images
  - Adjustable confidence threshold and NMS threshold via sliders
  - Optional CLAHE image enhancement for better detection in low light
  - Live FPS counter and object count shown as stat cards
  - Coloured status pill (Running / Stopped / Idle)
  - Detection log with per-frame results
  - Snapshot capture from webcam

Usage:
    python app.py
"""

import os
import time
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import customtkinter as ctk
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
LABELS_FILE   = os.path.join(SCRIPT_DIR, "coco_labels.txt")
SNAPSHOTS_DIR = os.path.join(SCRIPT_DIR, "snapshots")

APP_NAME = "AI Object Detection and Recognition"

# Maximum camera index to probe during scan.
# Increase if you have more than 10 cameras connected.
MAX_CAMERA_INDEX = 10


# ── Camera scanner ─────────────────────────────────────────────────────────────
def scan_cameras() -> list[dict]:
    """
    Probe camera indices 0 to MAX_CAMERA_INDEX and return a list of dicts
    for every index that successfully opens.

    Works cross-platform:
      - macOS  : finds built-in FaceTime camera, iPhone Continuity Camera,
                 any USB webcam — each at its own index
      - Windows: finds DirectShow / Media Foundation devices by index
      - Linux  : finds /dev/video* devices by index

    OpenCV does not expose camera display names, so each entry is labelled
    "Camera 0", "Camera 1", etc. The index is stored separately so it can
    be passed directly to cv2.VideoCapture().

    Returns a list of dicts:  [{"index": 0, "label": "Camera 0"}, ...]
    Returns [{"index": 0, "label": "Camera 0 (default)"}] as a fallback
    if nothing is found, so the dropdown is never empty.
    """
    found = []
    for i in range(MAX_CAMERA_INDEX):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()          # confirm it can actually deliver frames
            cap.release()
            if ret:
                found.append({"index": i, "label": f"Camera {i}"})
    if not found:
        found = [{"index": 0, "label": "Camera 0 (default)"}]
    return found


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_class_labels(filepath: str) -> list:
    with open(filepath, "rt") as f:

      
        return f.read().rstrip("\n").split("\n")


# ── Application ────────────────────────────────────────────────────────────────
class ObjectDetectionApp(ctk.CTk):
    """CustomTkinter desktop GUI for live AI object detection."""

    def __init__(self):
        super().__init__()

        # ── Window setup ───────────────────────────────────────────────────────
        self.title(APP_NAME)
        self.geometry("1200x740")
        self.minsize(1050, 660)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # ── State ──────────────────────────────────────────────────────────────
        self.cap              = None
        self.running          = False
        self.model            = None
        self.class_labels     = []
        self.conf_threshold   = 0.45
        self.nms_threshold    = 0.40
        self.enhance_image    = True
        self.latest_frame     = None
        self._after_id        = None
        self._cameras         = []   # list of dicts from scan_cameras()
        self._selected_cam_idx = 0   # actual cv2 index to open

        # ── Load labels ────────────────────────────────────────────────────────
        if os.path.exists(LABELS_FILE):
            self.class_labels = load_class_labels(LABELS_FILE)

        # ── Build UI then load model ────────────────────────────────────────────
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._load_model()

        # ── Scan cameras after UI is ready so the log is visible ───────────────
        self.after(200, self._refresh_cameras)

    # ── Model loading ──────────────────────────────────────────────────────────
    def _load_model(self):
        try:
            self.model = YOLO("yolov8n.pt")
            self._log("Model loaded: YOLOv8 Nano (yolov8n.pt)")
            self.start_btn.configure(state="normal")
            self.load_btn.configure(state="normal")
        except Exception as e:
            self._log(f"[ERROR] Could not load model: {e}")
            self.model = None

    # ── UI construction ────────────────────────────────────────────────────────
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        # ── Video canvas ───────────────────────────────────────────────────────
        self.video_label = ctk.CTkLabel(
            self, text="", fg_color="#0a0a0a", corner_radius=10
        )
        self.video_label.grid(
            row=0, column=0, padx=(12, 6), pady=12, sticky="nsew"
        )

        # ── Sidebar (fixed, non-scrollable) ───────────────────────────────────
        sidebar = ctk.CTkFrame(self, width=320, corner_radius=10)
        sidebar.grid(row=0, column=1, padx=(6, 12), pady=12, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.pack_propagate(False)

        # ── App title ─────────────────────────────────────────────────────────
        ctk.CTkLabel(
            sidebar,
            text=APP_NAME,
            font=ctk.CTkFont(size=14, weight="bold"),
            wraplength=288,
            justify="center",
        ).pack(padx=16, pady=(18, 2))

        ctk.CTkLabel(
            sidebar,
            text="YOLOv8  |  80 COCO classes",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        ).pack(padx=16, pady=(0, 14))

        # ── Divider ────────────────────────────────────────────────────────────
        ctk.CTkFrame(sidebar, height=1, fg_color="#2a2a2a").pack(
            fill="x", padx=16, pady=(0, 12)
        )

        # ── Status pill ────────────────────────────────────────────────────────
        status_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        status_row.pack(fill="x", padx=16, pady=(0, 10))

        ctk.CTkLabel(
            status_row,
            text="Status",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(side="left")

        self.status_pill = ctk.CTkLabel(
            status_row,
            text="  Idle  ",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#3a3a00",
            text_color="#e0c000",
            corner_radius=8,
        )
        self.status_pill.pack(side="right")

        # ── Stats cards (FPS + Objects) ────────────────────────────────────────
        stats_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        stats_row.pack(fill="x", padx=16, pady=(0, 14))
        stats_row.columnconfigure(0, weight=1)
        stats_row.columnconfigure(1, weight=1)

        fps_box = ctk.CTkFrame(stats_row, corner_radius=8)
        fps_box.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ctk.CTkLabel(
            fps_box, text="FPS", font=ctk.CTkFont(size=10), text_color="gray"
        ).pack(pady=(8, 0))
        self.fps_label = ctk.CTkLabel(
            fps_box, text="--", font=ctk.CTkFont(size=20, weight="bold")
        )
        self.fps_label.pack(pady=(0, 8))

        obj_box = ctk.CTkFrame(stats_row, corner_radius=8)
        obj_box.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ctk.CTkLabel(
            obj_box, text="Objects", font=ctk.CTkFont(size=10), text_color="gray"
        ).pack(pady=(8, 0))
        self.objects_label = ctk.CTkLabel(
            obj_box, text="--", font=ctk.CTkFont(size=20, weight="bold")
        )
        self.objects_label.pack(pady=(0, 8))

        # ── Divider ────────────────────────────────────────────────────────────
        ctk.CTkFrame(sidebar, height=1, fg_color="#2a2a2a").pack(
            fill="x", padx=16, pady=(0, 12)
        )

        # ── Camera selector ────────────────────────────────────────────────────
        cam_header = ctk.CTkFrame(sidebar, fg_color="transparent")
        cam_header.pack(fill="x", padx=16, pady=(0, 6))

        ctk.CTkLabel(
            cam_header,
            text="Camera",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(side="left")

        self.refresh_btn = ctk.CTkButton(
            cam_header,
            text="Refresh",
            command=self._refresh_cameras,
            width=68,
            height=26,
            font=ctk.CTkFont(size=11),
        )
        self.refresh_btn.pack(side="right")

        self.cam_var = ctk.StringVar(value="Scanning...")
        self.cam_dropdown = ctk.CTkOptionMenu(
            sidebar,
            variable=self.cam_var,
            values=["Scanning..."],
            command=self._on_camera_change,
            font=ctk.CTkFont(size=12),
            height=34,
        )
        self.cam_dropdown.pack(fill="x", padx=16, pady=(0, 14))

        # ── Divider ────────────────────────────────────────────────────────────
        ctk.CTkFrame(sidebar, height=1, fg_color="#2a2a2a").pack(
            fill="x", padx=16, pady=(0, 12)
        )

        # ── Controls ───────────────────────────────────────────────────────────
        ctk.CTkLabel(
            sidebar, text="Controls", font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=16, pady=(0, 6))

        btn_row1 = ctk.CTkFrame(sidebar, fg_color="transparent")
        btn_row1.pack(fill="x", padx=16, pady=(0, 6))

        self.start_btn = ctk.CTkButton(
            btn_row1,
            text="Start Webcam",
            command=self._start_webcam,
            fg_color="#1a6b35",
            hover_color="#145228",
            height=38,
            font=ctk.CTkFont(size=13),
        )
        self.start_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self.stop_btn = ctk.CTkButton(
            btn_row1,
            text="Stop",
            command=self._stop_webcam,
            fg_color="#6b1a24",
            hover_color="#52141b",
            height=38,
            font=ctk.CTkFont(size=13),
            state="disabled",
        )
        self.stop_btn.pack(side="left", expand=True, fill="x", padx=(4, 0))

        btn_row2 = ctk.CTkFrame(sidebar, fg_color="transparent")
        btn_row2.pack(fill="x", padx=16, pady=(0, 14))

        self.snap_btn = ctk.CTkButton(
            btn_row2,
            text="Snapshot",
            command=self._take_snapshot,
            height=34,
            font=ctk.CTkFont(size=12),
            state="disabled",
        )
        self.snap_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self.load_btn = ctk.CTkButton(
            btn_row2,
            text="Load Image",
            command=self._load_image,
            height=34,
            font=ctk.CTkFont(size=12),
        )
        self.load_btn.pack(side="left", expand=True, fill="x", padx=(4, 0))

        # ── Divider ────────────────────────────────────────────────────────────
        ctk.CTkFrame(sidebar, height=1, fg_color="#2a2a2a").pack(
            fill="x", padx=16, pady=(0, 12)
        )

        # ── Confidence threshold ───────────────────────────────────────────────
        conf_header = ctk.CTkFrame(sidebar, fg_color="transparent")
        conf_header.pack(fill="x", padx=16, pady=(0, 4))
        ctk.CTkLabel(
            conf_header,
            text="Confidence Threshold",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(side="left")
        self.thresh_value_label = ctk.CTkLabel(
            conf_header,
            text="45%",
            font=ctk.CTkFont(size=12),
            text_color="#5b9bd5",
        )
        self.thresh_value_label.pack(side="right")

        self.thresh_slider = ctk.CTkSlider(
            sidebar,
            from_=10,
            to=95,
            number_of_steps=17,
            command=self._on_threshold_change,
        )
        self.thresh_slider.set(45)
        self.thresh_slider.pack(fill="x", padx=16, pady=(0, 14))

        # ── NMS threshold ──────────────────────────────────────────────────────
        nms_header = ctk.CTkFrame(sidebar, fg_color="transparent")
        nms_header.pack(fill="x", padx=16, pady=(0, 4))
        ctk.CTkLabel(
            nms_header,
            text="NMS Threshold",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(side="left")
        self.nms_value_label = ctk.CTkLabel(
            nms_header,
            text="40%",
            font=ctk.CTkFont(size=12),
            text_color="#5b9bd5",
        )
        self.nms_value_label.pack(side="right")

        self.nms_slider = ctk.CTkSlider(
            sidebar,
            from_=10,
            to=80,
            number_of_steps=14,
            command=self._on_nms_change,
        )
        self.nms_slider.set(40)
        self.nms_slider.pack(fill="x", padx=16, pady=(0, 14))

        # ── Divider ────────────────────────────────────────────────────────────
        ctk.CTkFrame(sidebar, height=1, fg_color="#2a2a2a").pack(
            fill="x", padx=16, pady=(0, 10)
        )

        # ── CLAHE toggle ───────────────────────────────────────────────────────
        self.enhance_var = ctk.BooleanVar(value=True)
        self.enhance_checkbox = ctk.CTkCheckBox(
            sidebar,
            text="Enhance Image (CLAHE)",
            variable=self.enhance_var,
            command=self._on_enhance_toggle,
            font=ctk.CTkFont(size=12),
        )
        self.enhance_checkbox.pack(anchor="w", padx=16, pady=(0, 14))

        # ── Divider ────────────────────────────────────────────────────────────
        ctk.CTkFrame(sidebar, height=1, fg_color="#2a2a2a").pack(
            fill="x", padx=16, pady=(0, 10)
        )

        # ── Detection log ──────────────────────────────────────────────────────
        ctk.CTkLabel(
            sidebar,
            text="Detection Log",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(0, 6))

        self.log_textbox = ctk.CTkTextbox(
            sidebar,
            font=ctk.CTkFont(size=11),
            state="disabled",
            corner_radius=8,
        )
        self.log_textbox.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        # ── Initial placeholder ────────────────────────────────────────────────
        self._show_placeholder()

    # ── Camera scanning ────────────────────────────────────────────────────────
    def _refresh_cameras(self):
        """
        Scan for connected cameras and populate the dropdown.
        Disabled while the webcam is running to avoid mid-stream switching.
        """
        if self.running:
            self._log("Stop the webcam before refreshing cameras.")
            return

        self._log("Scanning for cameras...")
        self.cam_dropdown.configure(values=["Scanning..."], state="disabled")
        self.refresh_btn.configure(state="disabled")
        self.update()  # flush UI so "Scanning..." appears immediately

        self._cameras = scan_cameras()
        labels = [c["label"] for c in self._cameras]

        self.cam_dropdown.configure(values=labels, state="normal")
        self.cam_var.set(labels[0])
        self._selected_cam_idx = self._cameras[0]["index"]
        self.refresh_btn.configure(state="normal")

        self._log(f"Found {len(self._cameras)} camera(s): {', '.join(labels)}")

    def _on_camera_change(self, label: str):
        """Called when the user picks a different camera from the dropdown."""
        for cam in self._cameras:
            if cam["label"] == label:
                self._selected_cam_idx = cam["index"]
                self._log(f"Camera selected: {label} (index {cam['index']})")
                break

    # ── Placeholder ────────────────────────────────────────────────────────────
    def _show_placeholder(self):
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            "Press 'Start Webcam' or 'Load Image'",
            (55, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (60, 60, 60),
            2,
            cv2.LINE_AA,
        )
        self._display_frame(placeholder)

    # ── Display helpers ────────────────────────────────────────────────────────
    def _display_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        lw = self.video_label.winfo_width() or 640
        lh = self.video_label.winfo_height() or 480
        img.thumbnail((lw, lh), Image.LANCZOS)
        photo = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.video_label.configure(image=photo, text="")
        self.video_label._image = photo

    def _log(self, msg: str):
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", msg + "\n")
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    def _set_status(self, text: str, state: str):
        """Update status pill colour. state: 'running' | 'stopped' | 'idle'"""
        colours = {
            "running": ("#0d3d1a", "#34d058"),
            "stopped": ("#3d0d10", "#f97583"),
            "idle":    ("#3a3a00", "#e0c000"),
        }
        bg, fg = colours.get(state, colours["idle"])
        self.status_pill.configure(text=f"  {text}  ", fg_color=bg, text_color=fg)

    # ── Callbacks ──────────────────────────────────────────────────────────────
    def _on_threshold_change(self, value):
        self.conf_threshold = value / 100.0
        self.thresh_value_label.configure(text=f"{int(value)}%")

    def _on_nms_change(self, value):
        self.nms_threshold = value / 100.0
        self.nms_value_label.configure(text=f"{int(value)}%")

    def _on_enhance_toggle(self):
        self.enhance_image = self.enhance_var.get()
        self._log(f"Image enhancement {'ON' if self.enhance_image else 'OFF'}")

    # ── Webcam start / stop ────────────────────────────────────────────────────
    def _start_webcam(self):
        if self.model is None:
            self._log("[ERROR] Model not loaded")
            return

        # Use the index chosen in the dropdown, not a hardcoded 0
        self._log(f"Opening {self.cam_var.get()} (index {self._selected_cam_idx})...")
        self.cap = cv2.VideoCapture(self._selected_cam_idx)
        if not self.cap.isOpened():
            self._log(
                f"[ERROR] Cannot open camera at index {self._selected_cam_idx}. "
                "Try pressing Refresh to re-scan."
            )
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.snap_btn.configure(state="normal")
        # Lock the camera dropdown and refresh button while running
        self.cam_dropdown.configure(state="disabled")
        self.refresh_btn.configure(state="disabled")
        self._set_status("Running", "running")
        self._log("Webcam started.")
        self._update_frame()

    def _stop_webcam(self):
        self.running = False
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.snap_btn.configure(state="disabled")
        # Unlock the camera dropdown and refresh button
        self.cam_dropdown.configure(state="normal")
        self.refresh_btn.configure(state="normal")
        self._set_status("Stopped", "stopped")
        self.fps_label.configure(text="--")
        self.objects_label.configure(text="--")
        self._log("Webcam stopped.")
        self._show_placeholder()

    # ── Frame loop ─────────────────────────────────────────────────────────────
    def _update_frame(self):
        if not self.running or self.cap is None:
            return

        t0 = time.perf_counter()

        ret, frame = self.cap.read()
        if not ret:
            self._log("[ERROR] Failed to read frame")
            self._stop_webcam()
            return

        try:
            detect_frame = self._enhance_frame(frame) if self.enhance_image else frame

            results = self.model(
                detect_frame,
                conf=self.conf_threshold,
                iou=self.nms_threshold,
                verbose=False,
            )

            frame = results[0].plot()
            num_objects = len(results[0].boxes)

        except Exception as e:
            self._log(f"[ERROR] Detection failed: {e}")
            self._after_id = self.after(30, self._update_frame)
            return

        self.latest_frame = frame.copy()
        self._display_frame(frame)

        elapsed = time.perf_counter() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0
        self.fps_label.configure(text=f"{fps:.1f}")
        self.objects_label.configure(text=str(num_objects))

        self._after_id = self.after(10, self._update_frame)

    # ── Snapshot ───────────────────────────────────────────────────────────────
    def _take_snapshot(self):
        if self.latest_frame is None:
            return
        os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SNAPSHOTS_DIR, f"snapshot_{ts}.jpg")
        cv2.imwrite(path, self.latest_frame)
        self._log(f"Snapshot saved: {os.path.basename(path)}")

    # ── Load image ─────────────────────────────────────────────────────────────
    def _load_image(self):
        if self.model is None:
            self._log("[ERROR] Model not loaded.")
            return

        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        if self.running:
            self._stop_webcam()

        img = cv2.imread(file_path)
        if img is None:
            self._log(f"[ERROR] Cannot read: {file_path}")
            return

        self._log(f"Loaded: {os.path.basename(file_path)}")
        detect_img = self._enhance_frame(img) if self.enhance_image else img

        try:
            results = self.model(
                detect_img,
                conf=self.conf_threshold,
                iou=self.nms_threshold,
                verbose=False,
            )
        except Exception as e:
            self._log(f"[ERROR] Detection failed: {e}")
            return

        boxes = results[0].boxes
        num_objects = len(boxes)

        if num_objects > 0:
            img = results[0].plot()
            self._log(f"Detected {num_objects} object(s):")
            for box in boxes:
                cls   = int(box.cls[0])
                conf  = float(box.conf[0])
                label = self.model.names[cls]
                self._log(f"  {label}  {conf * 100:.1f}%")
            self.objects_label.configure(text=str(num_objects))
        else:
            self._log("No objects detected. Try lowering the threshold.")
            self.objects_label.configure(text="0")

        self.latest_frame = img.copy()
        self.snap_btn.configure(state="normal")
        self._display_frame(img)

        base, ext = os.path.splitext(file_path)
        out_path = base + "_detected" + ext
        cv2.imwrite(out_path, img)
        self._log(f"Saved: {os.path.basename(out_path)}")

    # ── Image enhancement ──────────────────────────────────────────────────────
    @staticmethod
    def _enhance_frame(frame):
        """Apply CLAHE to the L channel of LAB colour space.
        Improves detection in poor lighting or low-contrast scenes."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # ── Cleanup ────────────────────────────────────────────────────────────────
    def _on_close(self):
        self.running = False
        if self._after_id:
            self.after_cancel(self._after_id)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.destroy()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = ObjectDetectionApp()
    app.mainloop()
