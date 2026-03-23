"""
Object Detection Using OpenCV
==============================
Based on: https://medium.com/@hastisutaria/object-detection-using-opencv-f94f61e88b23

Uses SSD MobileNet V3 (trained on COCO dataset) with OpenCV's DNN module
to detect 80 object classes in images or live webcam feed.

Usage:
    python object_detection.py --image <path_to_image>
    python object_detection.py --webcam
"""

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
CONFIG_FILE = os.path.join(MODELS_DIR, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
FROZEN_MODEL = os.path.join(MODELS_DIR, "frozen_inference_graph.pb")
LABELS_FILE = os.path.join(SCRIPT_DIR, "coco_labels.txt")


def load_class_labels(filepath: str) -> list:
    """Read COCO class labels from a text file (one label per line)."""
    with open(filepath, "rt") as f:
        class_labels = f.read().rstrip("\n").split("\n")
    print(f"Loaded {len(class_labels)} class labels.")
    return class_labels


def build_model(config: str, frozen: str) -> cv2.dnn_DetectionModel:
    """Build and configure the SSD MobileNet V3 detection model."""
    model = cv2.dnn_DetectionModel(frozen, config)
    model.setInputSize(320, 320)
    model.setInputScale(1.0 / 127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)
    print("Model loaded and configured successfully.")
    return model


"""def detect_objects(model, img, conf_threshold: float = 0.5):
    """"""
    Run object detection on a single image.

    Returns:
        class_indices  – array of class IDs for each detection
        confidences    – array of confidence scores
        bboxes         – array of bounding boxes [x, y, w, h]
    """"""""
    class_indices, confidences, bboxes = (img, confThreshold=conf_threshold)
    return class_indices, confidences, bboxes"""



def draw_detections(img, class_indices, confidences, bboxes, class_labels):
    """Draw bounding boxes and labels on the image."""
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 2

    for class_idx, conf, box in zip(
        class_indices.flatten(), confidences.flatten(), bboxes
    ):
        idx = int(class_idx) - 1
        if idx < 0 or idx >= len(class_labels):
            continue
        label = class_labels[idx]
        confidence_pct = f"{conf * 100:.1f}%"

        # Draw rectangle
        cv2.rectangle(img, box, (0, 255, 0), 3)

        # Draw label + confidence
        text = f"{label}  {confidence_pct}"
        cv2.putText(
            img,
            text,
            (box[0] + 10, box[1] + 40),
            font,
            fontScale=font_scale,
            color=(0, 0, 255),
            thickness=2,
        )

    return img


# ── Image mode ─────────────────────────────────────────────────────────────────
def detect_from_image(image_path: str, conf_threshold: float = 0.5):
    """Detect objects in a still image and display the result."""
    # Validate paths
    for path, name in [
        (CONFIG_FILE, "Config file"),
        (FROZEN_MODEL, "Frozen model"),
        (LABELS_FILE, "Labels file"),
    ]:
        if not os.path.exists(path):
            print(f"[ERROR] {name} not found: {path}")
            print("Run  python download_models.py  first to fetch the model files.")
            sys.exit(1)

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    # Load resources
    class_labels = load_class_labels(LABELS_FILE)
    model = build_model(CONFIG_FILE, FROZEN_MODEL)

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        sys.exit(1)

    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    # Detect
    '''class_indices, confidences, bboxes = detect_objects(model, img, conf_threshold)'''
    results = model(img)
    '''if len(class_indices) == 0:
        print("No objects detected.")
    else:
        print(f"\nDetected {len(class_indices.flatten())} object(s):")
        print("-" * 45)
        for idx, conf in zip(class_indices.flatten(), confidences.flatten()):
            print(f"  {class_labels[idx - 1]:20s}  {conf * 100:5.1f}%")
        print("-" * 45)'''
    
    boxes = results[0].boxes
    if len(boxes) == 0:
        print("No objects detected.")
    else:
        print(f"\nDetected {len(boxes)} object(s):")
        print("-" * 45)

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]

            print(f"{label:20s} {conf * 100:5.1f}%")

        print("-" * 45)

         # Draw and display
        
        img = results[0].plot()
    

    # Save the output image
    output_path = os.path.splitext(image_path)[0] + "_detected.jpg"
    cv2.imwrite(output_path, img)
    print(f"\nOutput saved to: {output_path}")

    # Display using matplotlib (works without a GUI window manager)
    plt.figure(figsize=(14, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Object Detection Result")
    plt.tight_layout()
    plt.show()


# ── Webcam mode ────────────────────────────────────────────────────────────────
def detect_from_webcam(conf_threshold: float = 0.5):
    """Run real-time object detection on the webcam feed."""
    for path, name in [
        (CONFIG_FILE, "Config file"),
        (FROZEN_MODEL, "Frozen model"),
        (LABELS_FILE, "Labels file"),
    ]:
        if not os.path.exists(path):
            print(f"[ERROR] {name} not found: {path}")
            print("Run  python download_models.py  first to fetch the model files.")
            sys.exit(1)

    class_labels = load_class_labels(LABELS_FILE)
    model = build_model(CONFIG_FILE, FROZEN_MODEL)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        sys.exit(1)

    print("Webcam opened. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        '''class_indices, confidences, bboxes = detect_objects(
            model, frame, conf_threshold
        )'''
        results = model(frame)

        '''if len(class_indices) > 0:
            frame = draw_detections(
                frame, class_indices, confidences, bboxes, class_labels
            )'''
        frame = results[0].plot()

        cv2.imshow("Object Detection - Press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released.")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Object Detection using OpenCV (SSD MobileNet V3 + COCO)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to the input image")
    group.add_argument(
        "--webcam", action="store_true", help="Use webcam for real-time detection"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold (0-1, default: 0.5)",
    )

    args = parser.parse_args()

    if args.image:
        detect_from_image(args.image, args.threshold)
    else:
        detect_from_webcam(args.threshold)


if __name__ == "__main__":
    main()
