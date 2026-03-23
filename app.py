"""
Object Detection GUI – Live Webcam & Image Detection
=====================================================
Desktop application built with CustomTkinter + OpenCV.

Features:
  • Two models: SSD MobileNet V3 (fast) & YOLOv4-tiny (more accurate)
  • Input resolution selector: 320 / 416 / 512 / 608
  • Live webcam feed with real-time object detection
  • Load and detect objects in still images
  • Adjustable confidence threshold + NMS threshold via sliders
  • Live FPS counter and detection log
  • Snapshot capture from webcam
  • Start / Stop / Snapshot controls

Usage:
    python app.py
"""

import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import customtkinter as ctk
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
LABELS_FILE = os.path.join(SCRIPT_DIR, "coco_labels.txt")
SNAPSHOTS_DIR = os.path.join(SCRIPT_DIR, "snapshots")

# SSD MobileNet V3
SSD_CONFIG = os.path.join(MODELS_DIR, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
SSD_MODEL = os.path.join(MODELS_DIR, "frozen_inference_graph.pb")

# YOLOv4-tiny
YOLOV4T_CFG = os.path.join(MODELS_DIR, "yolov4-tiny.cfg")
YOLOV4T_WEIGHTS = os.path.join(MODELS_DIR, "yolov4-tiny.weights")

# YOLOv3 (full — best accuracy)
YOLOV3_CFG = os.path.join(MODELS_DIR, "yolov3.cfg")
YOLOV3_WEIGHTS = os.path.join(MODELS_DIR, "yolov3.weights")

# ── Model definitions ─────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "SSD MobileNet V3 (Fast)": {
        "type": "ssd",
        "config": SSD_CONFIG,
        "weights": SSD_MODEL,
        "short": "SSD",
        "description": "Fastest. Lower accuracy, may miss small objects.",
    },
    "YOLOv4-tiny (Balanced)": {
        "type": "yolo",
        "config": YOLOV4T_CFG,
        "weights": YOLOV4T_WEIGHTS,
        "short": "YOLOv4-tiny",
        "description": "Good speed/accuracy balance for real-time.",
    },
    "YOLOv3 (Best Accuracy)": {
        "type": "yolo",
        "config": YOLOV3_CFG,
        "weights": YOLOV3_WEIGHTS,
        "short": "YOLOv3",
        "description": "Best for bottles, phones & small objects. Slower.",
    },
}

RESOLUTION_OPTIONS = ["320", "416", "512", "608"]


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_class_labels(filepath: str) -> list:
    with open(filepath, "rt") as f:
        return f.read().rstrip("\n").split("\n")


def build_yolo_model():
    model = YOLO("yolov8n.pt")  # or yolov8l.pt for better accuracy
    return model


# Generate distinct colours for each of the 80 classes
np.random.seed(42)
COLORS = [
    tuple(int(c) for c in color)
    for color in np.random.randint(80, 255, size=(80, 3))
]


# ── Application ────────────────────────────────────────────────────────────────
class ObjectDetectionApp(ctk.CTk):
    """CustomTkinter desktop GUI for live object detection."""

    def __init__(self):
        super().__init__()

        # ── Window setup ───────────────────────────────────────────────────────
        self.title("ROBOVORTEX VISION – Object Detection Framework")
        self.geometry("1150x720")
        self.minsize(1000, 650)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # ── State ──────────────────────────────────────────────────────────────
        self.cap = None
        self.running = False
        self.model = None
        self.model_type = "ssd"
        self.class_labels = []
        self.conf_threshold = 0.40
        self.nms_threshold = 0.40
        self.input_size = 416
        self.enhance_image = True         # CLAHE preprocessing
        self.latest_frame = None
        self._after_id = None

        # ── Load labels ────────────────────────────────────────────────────────
        if os.path.exists(LABELS_FILE):
            self.class_labels = load_class_labels(LABELS_FILE)

        # ── Build UI then load default model ───────────────────────────────────
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._load_selected_model()

    # ── Model loading ──────────────────────────────────────────────────────────
    def _load_selected_model(self):
        try:
            self.model = YOLO("yolov8n.pt")
            self._log("YOLOv8 loaded successfully 🚀")

            self.start_btn.configure(state="normal")
            self.load_btn.configure(state="normal")
            self.model_info_label.configure(text="Model: YOLOv8")

        except Exception as e:
            self._log(f"[ERROR] {e}")
            self.model = None

    # ── UI construction ────────────────────────────────────────────────────────
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        # ── Video canvas ───────────────────────────────────────────────────────
        self.video_label = ctk.CTkLabel(
            self, text="", fg_color="black", corner_radius=8
        )
        self.video_label.grid(
            row=0, column=0, padx=(10, 5), pady=10, sticky="nsew"
        )

        # ── Sidebar (scrollable) ──────────────────────────────────────────────
        sidebar = ctk.CTkScrollableFrame(self, width=300)
        sidebar.grid(row=0, column=1, padx=(5, 10), pady=10, sticky="nsew")

        # Title
        ctk.CTkLabel(
            sidebar,
            text="🎯 ROBOVORTEX VISION",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(padx=15, pady=(15, 2))
        ctk.CTkLabel(
            sidebar,
            text="Real-time Object Detection",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        ).pack(padx=15, pady=(0, 10))

        # ── Model selection ────────────────────────────────────────────────────
        model_frame = ctk.CTkFrame(sidebar)
        model_frame.pack(padx=10, pady=5, fill="x")

        ctk.CTkLabel(
            model_frame, text="Model", font=ctk.CTkFont(weight="bold")
        ).pack(padx=10, pady=(8, 2), anchor="w")

        self.model_var = ctk.StringVar(value=list(MODEL_CONFIGS.keys())[0])
        self.model_dropdown = ctk.CTkOptionMenu(
            model_frame,
            values=list(MODEL_CONFIGS.keys()),
            variable=self.model_var,
            command=self._on_model_change,
        )
        self.model_dropdown.pack(padx=10, pady=(0, 2), fill="x")

        self.model_desc_label = ctk.CTkLabel(
            model_frame,
            text=MODEL_CONFIGS[self.model_var.get()]["description"],
            font=ctk.CTkFont(size=11),
            text_color="gray",
            wraplength=250,
        )
        self.model_desc_label.pack(padx=10, pady=(0, 4), anchor="w")

        # ── Resolution selector ────────────────────────────────────────────────
        ctk.CTkLabel(
            model_frame, text="Input Resolution", font=ctk.CTkFont(size=12)
        ).pack(padx=10, pady=(4, 0), anchor="w")

        self.res_var = ctk.StringVar(value="320")
        res_frame = ctk.CTkFrame(model_frame, fg_color="transparent")
        res_frame.pack(padx=10, pady=(2, 8), fill="x")

        for res in RESOLUTION_OPTIONS:
            ctk.CTkRadioButton(
                res_frame,
                text=f"{res}x{res}",
                variable=self.res_var,
                value=res,
                command=self._on_resolution_change,
                font=ctk.CTkFont(size=12),
            ).pack(side="left", padx=(0, 8))

        # ── Controls ───────────────────────────────────────────────────────────
        controls = ctk.CTkFrame(sidebar)
        controls.pack(padx=10, pady=5, fill="x")

        ctk.CTkLabel(
            controls, text="Controls", font=ctk.CTkFont(weight="bold")
        ).pack(padx=10, pady=(8, 2), anchor="w")

        btn_frame = ctk.CTkFrame(controls, fg_color="transparent")
        btn_frame.pack(padx=10, pady=5, fill="x")

        self.start_btn = ctk.CTkButton(
            btn_frame,
            text="▶  Start Webcam",
            command=self._start_webcam,
            fg_color="#2ea44f",
            hover_color="#22863a",
            height=36,
        )
        self.start_btn.pack(side="left", expand=True, fill="x", padx=(0, 3))

        self.stop_btn = ctk.CTkButton(
            btn_frame,
            text="⏹  Stop",
            command=self._stop_webcam,
            fg_color="#d73a49",
            hover_color="#b31d28",
            height=36,
            state="disabled",
        )
        self.stop_btn.pack(side="left", expand=True, fill="x", padx=(3, 0))

        btn_frame2 = ctk.CTkFrame(controls, fg_color="transparent")
        btn_frame2.pack(padx=10, pady=(0, 8), fill="x")

        self.snap_btn = ctk.CTkButton(
            btn_frame2,
            text="📸 Snapshot",
            command=self._take_snapshot,
            height=32,
            state="disabled",
        )
        self.snap_btn.pack(side="left", expand=True, fill="x", padx=(0, 3))

        self.load_btn = ctk.CTkButton(
            btn_frame2,
            text="🖼 Load Image",
            command=self._load_image,
            height=32,
        )
        self.load_btn.pack(side="left", expand=True, fill="x", padx=(3, 0))

        # ── Confidence threshold slider ────────────────────────────────────────
        thresh_frame = ctk.CTkFrame(sidebar)
        thresh_frame.pack(padx=10, pady=5, fill="x")

        ctk.CTkLabel(
            thresh_frame,
            text="Confidence Threshold",
            font=ctk.CTkFont(weight="bold"),
        ).pack(padx=10, pady=(8, 0), anchor="w")

        self.thresh_value_label = ctk.CTkLabel(
            thresh_frame,
            text="45%",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#3b82f6",
        )
        self.thresh_value_label.pack(padx=10, pady=(2, 0))

        self.thresh_slider = ctk.CTkSlider(
            thresh_frame,
            from_=10,
            to=95,
            number_of_steps=17,
            command=self._on_threshold_change,
        )
        self.thresh_slider.set(45)
        self.thresh_slider.pack(padx=15, pady=(0, 4), fill="x")

        # NMS threshold
        ctk.CTkLabel(
            thresh_frame,
            text="NMS Threshold (overlap removal)",
            font=ctk.CTkFont(size=12),
        ).pack(padx=10, pady=(4, 0), anchor="w")

        self.nms_value_label = ctk.CTkLabel(
            thresh_frame,
            text="40%",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#8b5cf6",
        )
        self.nms_value_label.pack(padx=10, pady=(0, 0))

        self.nms_slider = ctk.CTkSlider(
            thresh_frame,
            from_=10,
            to=80,
            number_of_steps=14,
            command=self._on_nms_change,
        )
        self.nms_slider.set(40)
        self.nms_slider.pack(padx=15, pady=(0, 10), fill="x")

        # Image enhancement toggle
        self.enhance_var = ctk.BooleanVar(value=True)
        self.enhance_checkbox = ctk.CTkCheckBox(
            sidebar,
            text="Enhance Image (CLAHE)",
            variable=self.enhance_var,
            command=self._on_enhance_toggle,
            font=ctk.CTkFont(size=13),
        )
        self.enhance_checkbox.pack(padx=15, pady=(5, 10), anchor="w")

        # ── Stats ──────────────────────────────────────────────────────────────
        stats_frame = ctk.CTkFrame(sidebar)
        stats_frame.pack(padx=10, pady=5, fill="x")

        ctk.CTkLabel(
            stats_frame, text="Statistics", font=ctk.CTkFont(weight="bold")
        ).pack(padx=10, pady=(8, 2), anchor="w")

        self.fps_label = ctk.CTkLabel(
            stats_frame, text="FPS: —", font=ctk.CTkFont(size=13)
        )
        self.fps_label.pack(padx=10, anchor="w")

        self.objects_label = ctk.CTkLabel(
            stats_frame, text="Objects: —", font=ctk.CTkFont(size=13)
        )
        self.objects_label.pack(padx=10, anchor="w")

        self.model_info_label = ctk.CTkLabel(
            stats_frame, text="Model: —", font=ctk.CTkFont(size=13)
        )
        self.model_info_label.pack(padx=10, anchor="w")

        self.status_label = ctk.CTkLabel(
            stats_frame,
            text="Status: Idle",
            font=ctk.CTkFont(size=13),
            text_color="orange",
        )
        self.status_label.pack(padx=10, pady=(0, 8), anchor="w")

        # ── Detection log ──────────────────────────────────────────────────────
        log_frame = ctk.CTkFrame(sidebar)
        log_frame.pack(padx=10, pady=5, fill="both", expand=True)

        ctk.CTkLabel(
            log_frame, text="Detection Log", font=ctk.CTkFont(weight="bold")
        ).pack(padx=10, pady=(8, 2), anchor="w")

        self.log_textbox = ctk.CTkTextbox(
            log_frame, font=ctk.CTkFont(size=12), state="disabled", height=180
        )
        self.log_textbox.pack(padx=8, pady=(0, 8), fill="both", expand=True)

        # ── COCO classes info ──────────────────────────────────────────────────
        info_frame = ctk.CTkFrame(sidebar)
        info_frame.pack(padx=10, pady=(5, 10), fill="x")
        ctk.CTkLabel(
            info_frame,
            text="Detectable Objects (80 classes)",
            font=ctk.CTkFont(weight="bold"),
        ).pack(padx=10, pady=(8, 2), anchor="w")

        sample_classes = (
            "person, bicycle, car, motorcycle, bus, truck, "
            "bottle, cup, fork, knife, spoon, bowl, banana, apple, "
            "pizza, chair, couch, bed, tv, laptop, mouse, remote, "
            "keyboard, cell phone, book, clock, scissors, "
            "cat, dog, horse, bear, bird, ..."
        )
        ctk.CTkLabel(
            info_frame,
            text=sample_classes,
            font=ctk.CTkFont(size=11),
            text_color="gray",
            wraplength=260,
            justify="left",
        ).pack(padx=10, pady=(0, 8), anchor="w")

        # ── Placeholder image ──────────────────────────────────────────────────
        self._show_placeholder()

    # ── Placeholder ────────────────────────────────────────────────────────────
    def _show_placeholder(self):
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            "Press 'Start Webcam' or 'Load Image'",
            (60, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (100, 100, 100),
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

    def _on_model_change(self, choice):
        was_running = self.running
        if was_running:
            self._stop_webcam()
        self._load_selected_model()
        if was_running:
            self._start_webcam()

    def _on_resolution_change(self):
        new_res = int(self.res_var.get())
        if new_res == self.input_size:
            return
        self.input_size = new_res
        self._log(f"Resolution changed to {new_res}x{new_res}")
        was_running = self.running
        if was_running:
            self._stop_webcam()
        self._load_selected_model()
        if was_running:
            self._start_webcam()

    # ── Webcam start / stop ────────────────────────────────────────────────────
    def _start_webcam(self):
        if self.model is None:
            self._log("[ERROR] Model not loaded")
            return

        # FIX: removed cv2.CAP_AVFOUNDATION — that backend is Mac-only and
        # silently fails to open the webcam on Windows / Linux.
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self._log("[ERROR] Cannot open webcam")
            return

        # Lower resolution = faster + more stable
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.snap_btn.configure(state="normal")
        self.model_dropdown.configure(state="disabled")
        self.status_label.configure(text="Status: Running", text_color="green")
        self._log("Webcam started 🎥")

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
        self.model_dropdown.configure(state="normal")
        self.status_label.configure(text="Status: Stopped", text_color="orange")
        self.fps_label.configure(text="FPS: —")
        self.objects_label.configure(text="Objects: —")
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
            # Apply enhancement BEFORE detection
            detect_frame = self._enhance_frame(frame) if self.enhance_image else frame

            # FIX: model called exactly once per frame (duplicate block removed)
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
            # Still schedule next frame so the loop doesn't die on one bad frame
            self._after_id = self.after(30, self._update_frame)
            return

        self.latest_frame = frame.copy()
        self._display_frame(frame)

        elapsed = time.perf_counter() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0
        self.fps_label.configure(text=f"FPS: {fps:.1f}")
        self.objects_label.configure(text=f"Objects: {num_objects}")

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

        # FIX: replaced old OpenCV .detect() call (crashes with YOLOv8)
        # with the correct Ultralytics API
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
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names[cls]
                self._log(f"  • {label}  {conf * 100:.1f}%")
            self.objects_label.configure(text=f"Objects: {num_objects}")
        else:
            self._log("No objects detected. Try lowering the threshold.")
            self.objects_label.configure(text="Objects: 0")

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
        """Apply CLAHE (contrast-limited adaptive histogram equalization).
        Improves detection of objects in poor lighting / low contrast."""
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