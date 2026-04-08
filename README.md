# AI Object Detection Framework

A Python desktop application for real-time object detection using YOLOv8. Supports live webcam feed and still image analysis through a modern GUI built with CustomTkinter.

---

## Features

- Real-time object detection via webcam
- Still image loading and detection
- YOLOv8 model with 80 COCO object classes
- Adjustable confidence threshold and NMS threshold via sliders
- Optional CLAHE image enhancement for better detection in low light
- Live FPS counter and per-frame detection log
- Snapshot capture from webcam feed
- Input resolution selector: 320 / 416 / 512 / 608

---

## Libraries

| Library | Version | Purpose |
|---|---|---|
| opencv-python | >= 4.5.0 | Webcam capture, frame reading, CLAHE image enhancement, and writing output images |
| ultralytics | latest | YOLOv8 model loading, inference, bounding box plotting, and class name lookup |
| customtkinter | >= 5.2.0 | Modern dark-themed desktop GUI with sliders, buttons, and scrollable frames |
| Pillow | >= 9.0.0 | Converting OpenCV BGR frames to RGB images for display inside the CustomTkinter GUI |
| numpy | >= 1.21.0 | Array operations and generating distinct colours for the 80 detection classes |
| matplotlib | >= 3.4.0 | Displaying detection results for still images in object_detection.py |

---

## Project Structure

```
project/
|-- app.py                  Main GUI application
|-- object_detection.py     Standalone CLI detection script
|-- download_models.py      Downloads SSD and YOLO model files
|-- coco_labels.txt         80 COCO class names, one per line
|-- requirements.txt        Python dependencies
|-- README.md               This file
|-- snapshots/              Saved snapshots from webcam (created at runtime)
```

---

## Setup and install

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download model files

The following command downloads all model files into the `models/` directory:

```bash
python download_models.py
```

Models downloaded:

- SSD MobileNet V3 - fastest, lower accuracy
- YOLOv4-tiny - balanced speed and accuracy
- YOLOv3 - best accuracy, slower

Note: The GUI application (app.py) uses YOLOv8 and downloads its weights automatically on first run via the Ultralytics library. No manual download is needed for the GUI.

---

## Usage

### GUI Application

```bash
python app.py
```

Controls inside the GUI:

- Start Webcam - opens the default camera and begins detection
- Stop - stops the webcam feed
- Snapshot - saves the current annotated frame to the `snapshots/` folder
- Load Image - opens a file dialog to run detection on a still image
- Confidence Threshold slider - filters out detections below this score
- NMS Threshold slider - controls overlap removal between bounding boxes
- Enhance Image (CLAHE) - toggles contrast enhancement before detection

### CLI Script (SSD MobileNet V3)

Detect objects in a still image:

```bash
python object_detection.py --image path/to/image.jpg
```

Run real-time detection via webcam:

```bash
python object_detection.py --webcam
```

Adjust the confidence threshold (default is 0.5):

```bash
python object_detection.py --image sample.jpg --threshold 0.6
```

---

## How It Works

1. Model loading - YOLOv8 is loaded via the Ultralytics library on startup
2. Frame capture - OpenCV reads frames from the webcam at 640x480
3. Enhancement - if enabled, CLAHE is applied to the L channel of the LAB colour space to improve contrast
4. Detection - the frame is passed to the YOLOv8 model with the configured confidence and IOU thresholds
5. Annotation - Ultralytics plots bounding boxes and class labels directly onto the frame
6. Display - the annotated frame is converted from BGR to RGB via Pillow and rendered in the GUI

---

## Detectable Object Classes (80 total)

The model is trained on the COCO dataset and can detect the following categories:

People and vehicles: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat

Outdoor objects: traffic light, fire hydrant, stop sign, parking meter, bench

Animals: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

Accessories: backpack, umbrella, handbag, tie, suitcase

Sports: frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket

Kitchen and food: bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake

Furniture and household: chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator

Miscellaneous: book, clock, vase, scissors, teddy bear, hair drier, toothbrush

---

## Requirements File Reference

```
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.4.0
customtkinter>=5.2.0
Pillow>=9.0.0
```

Install Ultralytics separately as it is not included in requirements.txt:

```bash
pip install ultralytics
```

---

## Notes

- On first run, app.py will automatically download the YOLOv8 nano weights file (yolov8n.pt, approximately 6 MB) from the Ultralytics servers.
- Snapshots are saved as JPEG files in the `snapshots/` directory with a timestamp in the filename.
- When loading a still image, the annotated result is also saved alongside the original file with `_detected` appended to the filename.
- Lowering the confidence threshold will detect more objects but may increase false positives.
