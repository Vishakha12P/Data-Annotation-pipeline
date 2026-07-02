# YOLO + YOLO-World Annotation Pipeline

A simpler replacement for the GroundingDINO-based pipeline — same job
(auto-labeling person + PPE classes for training data), but with **zero
custom C++/CUDA extensions to compile**. GPU acceleration works automatically
through plain PyTorch, since `ultralytics` doesn't need any special build step.

## Why this is simpler than the old GroundingDINO pipeline

- `pip install ultralytics` — that's it, no separate repo to clone, no
  `setup.py`/`pyproject.toml` build step, no BERT text encoder, no
  `transformers` version conflicts, no Visual Studio / CUDA Toolkit version
  matching.
- **Person detection** uses a pretrained YOLOv8 model — it already knows
  "person" from COCO training. No custom classes, no training needed for this part.
- **PPE detection** uses YOLO-World, an open-vocabulary model (like
  GroundingDINO) that takes text prompts — but built entirely in PyTorch.

## Setup

```bash
# 1. Activate your existing venv
source venv/Scripts/activate      # Git Bash
# or: venv\Scripts\activate       # Command Prompt

# 2. Install CUDA-enabled PyTorch FIRST (you already confirmed this works)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Install everything else
pip install -r requirements.txt
```

That's the entire setup. No cloning, no compiling, no C++ compiler needed.

The first time you run `annotate.py`, `ultralytics` will automatically
download the pretrained model weights (`yolov8s.pt` and `yolov8s-worldv2.pt`)
— a few hundred MB total, one-time only.

## Configure your classes

Edit `config.yaml` — your PPE classes are already filled in under
`secondary.classes`. Adjust `conf_threshold` values or `skip_frames` as needed.

## Add your videos

```bash
mkdir -p data/raw_videos
cp /path/to/your/video.mp4 data/raw_videos/
echo "data/raw_videos/your_video.mp4" > data/video_list.txt
```
(One video path per line if you have multiple.)

## Run it

```bash
python annotate.py
```

Progress prints per-video, with a tqdm progress bar per frame. It supports
resuming — if interrupted, just rerun the same command; frames that already
have a saved label file are skipped.

## Output structure

```
data/
├── frames/                       (not used directly — frames processed in-memory)
├── primary/
│   ├── labels/                   YOLO labels for "person" (full frame)
│   └── visualizations/           full frames with person boxes drawn
└── secondary/
    ├── images/                   cropped person images
    ├── labels/                   YOLO labels for PPE classes (relative to crop)
    ├── visualizations/           crops with PPE boxes drawn
    └── metadata/                 JSON crop offset info (for mapping back to full frame)

dataset_primary.yaml     — for training a person-detection YOLO model
dataset_secondary.yaml   — for training a PPE-detection YOLO model
```

## A note on "missing PPE" logic

This pipeline (and any object detector) only detects things that ARE present
— there's no box for "an apron that isn't there." To flag missing PPE, you
apply a simple rule at inference time in your final application: for each
detected person, check whether a PPE box overlaps their region; if not,
flag it as missing. That logic lives outside the annotation/training pipeline.

## Next step: training

Once you've run this across your videos and spot-checked label quality,
train dedicated YOLO models on the generated datasets:

```python
from ultralytics import YOLO

# Person model (optional — the pretrained one may already be good enough)
model = YOLO('yolov8n.pt')
model.train(data='dataset_primary.yaml', epochs=100, imgsz=640)

# PPE model (this one you actually need to train — PPE items are custom classes)
model = YOLO('yolov8n.pt')
model.train(data='dataset_secondary.yaml', epochs=100, imgsz=640)
```
