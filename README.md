# YOLO + YOLO-World PPE Annotation Pipeline

An automated data annotation pipeline for brewery/facility CCTV footage. Detects **people** in each frame, then checks each person for **8 categories of PPE compliance** — all without any manual labeling or model training required for the annotation stage itself.

Built as a simpler, GPU-friendly alternative to GroundingDINO-based pipelines: everything here runs on plain PyTorch via the `ultralytics` library, with **no custom C++/CUDA extensions to compile**.

---

## What this does

```
CCTV Video -> Extract frames -> Detect people (YOLOv8, pretrained)
                                       |
                                       v
                           Crop each detected person
                                       |
                                       v
                 Detect PPE items on each crop (YOLO-World, open-vocabulary)
                                       |
                                       v
               Save YOLO-format labels + visualizations + dataset YAMLs
```

**Primary detection** — finds every person in the full frame using a pretrained YOLOv8 model (already knows "person" from COCO training, zero setup).

**Secondary detection** — for each detected person, crops them out and checks for PPE items using YOLO-World, an open-vocabulary detector that finds objects from a text description instead of needing to be pre-trained on your exact classes.

### PPE classes detected

| Class | Detects |
|---|---|
| `safety shoes` | Mandatory safety footwear |
| `hand gloves` | Hand protection |
| `safety goggles` | Eye protection |
| `hi-vis vest` | High-visibility vest / safety jacket |
| `apron` | Process aprons |
| `helmet` | Head protection |
| `arm sleeve` | Arm protection near moving equipment |
| `leg sleeve` | Leg protection / protective clothing |

> **Note on "missing PPE":** this pipeline (like any object detector) only detects things that ARE present in an image — there's no box for "an apron that isn't there." To flag *missing* PPE, apply a simple rule at inference time in your final application: for each detected person, check whether a PPE box overlaps their region; if none does, flag it as missing. That logic lives outside this annotation pipeline.

---

## Requirements

- Windows, Python 3.11 (3.13/3.14 are too new for some ML packages as of writing — stick to 3.11)
- NVIDIA GPU strongly recommended (tested on RTX 5050 Laptop GPU, 8GB VRAM) — CPU works but is roughly 10-15x slower
- Git (for cloning/version control, optional)

---

## Setup (Windows, Git Bash)

### 1. Get Python 3.11 if you don't already have it

```bash
py -0
```
Check if `3.11` is listed. If not:
```bash
py install 3.11
```
(If that command isn't supported by your `py` launcher version, download the installer manually from python.org/downloads and make sure **"Add python.exe to PATH"** is checked during install.)

### 2. Create and activate a virtual environment

```bash
py -3.11 -m venv venv
source venv/Scripts/activate
python --version   # should print Python 3.11.x
```

> **Important:** every time you open a **new** terminal/tab, you must re-run `source venv/Scripts/activate` before any `python` command will work correctly. If you see `Python was not found; run without arguments to install from the Microsoft Store...`, this is why — your venv isn't active in that window.

### 3. Install CUDA-enabled PyTorch (do this BEFORE the other requirements)

Check your GPU and driver first:
```bash
nvidia-smi
```
Note the `CUDA Version` shown at the top — install a matching PyTorch build. For CUDA 12.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Verify it actually got the GPU build (not a silent CPU-only fallback):
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
This must print `CUDA available: True` and your GPU's name. If it prints `False`, PyTorch installed a CPU-only wheel — double check the `--index-url` matches your CUDA version exactly.

### 4. Install everything else

```bash
pip install -r requirements.txt
```

This installs `ultralytics` (which also pulls in `clip`, `opencv-python`, etc.) — all pure Python/pip installs, no compiling required.

The **first time** you run `annotate.py`, it will automatically download the model weights (`yolov8s.pt` and `yolov8s-worldv2.pt`, a few hundred MB total). This only happens once; they're cached afterward.

---

## Usage

### 1. Add your video(s)

```bash
mkdir -p data/raw_videos
cp "/c/path/to/your/video.mp4" data/raw_videos/
```

### 2. Point the pipeline at it

```bash
echo "data/raw_videos/your_video.mp4" > data/video_list.txt
```

For multiple videos, add one path per line:
```bash
echo "data/raw_videos/video1.mp4" > data/video_list.txt
echo "data/raw_videos/video2.mp4" >> data/video_list.txt
```

### 3. Run it

```bash
python annotate.py
```

You should see output like:
```
Using device: cuda
GPU: NVIDIA GeForce RTX 5050 Laptop GPU
Loading person detection model (yolov8s.pt)...
Loading PPE detection model (yolov8s-worldv2.pt)...
Models loaded.
Found 1 video(s) to process
> your_video.mp4  (1598x882, 30.0fps, 21.2s, ~8 frames to check)
  your_video: 9it [00:07, 1.20it/s]
Created dataset_primary.yaml and dataset_secondary.yaml
============================================================
ANNOTATION COMPLETE
============================================================
Frames checked   : 9
Frames kept      : 9
Frames discarded : 0
Primary detections:
  person: 79
Secondary (PPE) detections:
  helmet: 62
  hand gloves: 57
  hi-vis vest: 117
  ...
============================================================
```

If it says `Using device: cpu` or `CUDA requested but not available`, GPU acceleration isn't active — see the CUDA verification step above.

### 4. Resume support

If a run gets interrupted (closed terminal, crashed, etc.), just run `python annotate.py` again. Frames that were already fully processed (have a saved label file) are automatically skipped — you won't lose progress or reprocess from frame zero.

---

## Configuration (`config.yaml`)

### Frame sampling

```yaml
video:
  skip_frames: 74
```
Number of frames to skip between processed frames. CCTV footage is highly redundant — you don't need every frame.
- `0` = every frame
- `2` = every 3rd frame
- `74` roughly 1 frame every 2.5 seconds at 30fps

For long footage (3-4 hour videos), sample sparsely to keep runtime realistic. Increase this number for longer/repetitive footage, decrease it if you need finer temporal coverage.

### Model size

```yaml
model:
  person_model: "yolov8s.pt"       # nano -> small -> medium -> large: n, s, m, l
  ppe_model: "yolov8s-worldv2.pt"
```
Larger models are more accurate but slower and need more VRAM. On an 8GB laptop GPU, `s` (small) is a reasonable default. Drop to `n` (nano) if you hit out-of-memory errors; go up to `m`/`l` if you have VRAM to spare and want higher accuracy.

### Confidence thresholds

```yaml
primary:
  conf_threshold: 0.35   # person detection

secondary:
  conf_threshold: 0.20   # PPE detection
```
Lower = more detections (higher recall, more false positives). Higher = fewer, more confident detections (higher precision, may miss things). YOLO-World (secondary) typically needs a lower threshold than closed-set YOLO (primary) since it's working from open-vocabulary text prompts.

### PPE classes

```yaml
secondary:
  classes:
    - "safety shoes"
    - "hand gloves"
    - "safety goggles"
    - "hi-vis vest"
    - "apron"
    - "helmet"
    - "arm sleeve"
    - "leg sleeve"
```
This list is passed directly to YOLO-World via `set_classes()`, which **hard-restricts** the model's output to only these classes — it cannot output anything outside this list. To add/remove/rename a class, edit this list (and update `crop_padding`/thresholds if a specific class needs different tuning).

---

## Output structure

```
data/
|-- raw_videos/                   your input video files
|-- primary/
|   |-- labels/                   YOLO labels for "person" (relative to full frame)
|   |-- visualizations/           full frames with person boxes drawn
|-- secondary/
    |-- images/                   cropped person images
    |-- labels/                   YOLO labels for PPE classes (relative to crop)
    |-- visualizations/           crops with PPE boxes drawn
    |-- metadata/                 JSON crop offset info (maps crop coords back to original frame)

dataset_primary.yaml     - for training a person-detection YOLO model
dataset_secondary.yaml   - for training a PPE-detection YOLO model
```

Label format (standard YOLO): `class_id x_center y_center width height`, all normalized to `[0, 1]`.

---

## Scaling to many videos (e.g. 300+ CCTV recordings)

A few practical notes before pointing this at a large batch:

- **Sample sparsely.** For hours-long footage, a low `skip_frames` value will make runtime impractical even on GPU. Start sparse (e.g. every few seconds), and only densify on specific videos if the resulting dataset turns out too small.
- **Queue everything at once.** List every video in `video_list.txt` (one path per line) and let it run — the resume logic means an overnight interruption doesn't cost you progress.
- **Spot-check label quality before training.** No auto-labeler is perfect. Review a sample of `data/secondary/visualizations/` for false positives/negatives before using the labels to train a model — bad labels at scale will hurt your trained model more than a smaller, cleaner dataset would.
- **GPU VRAM.** If you hit CUDA out-of-memory errors when processing many people per frame, drop to a smaller model size (`yolov8n.pt` / `yolov8n-worldv2.pt`) rather than reducing batch size (this pipeline processes one frame at a time by design, so there's no batch size to tune here).

---

## Next step: training your own model

Once you've annotated your videos and are happy with label quality, train dedicated models on the generated datasets:

```python
from ultralytics import YOLO

# Optional - the pretrained person model may already be good enough as-is
model = YOLO('yolov8n.pt')
model.train(data='dataset_primary.yaml', epochs=100, imgsz=640)

# Recommended - train on your actual PPE data for best results
model = YOLO('yolov8n.pt')
model.train(data='dataset_secondary.yaml', epochs=100, imgsz=640)
```

---

## Troubleshooting

**`Python was not found; run without arguments to install from the Microsoft Store`**
Your venv isn't active in this terminal. Run `source venv/Scripts/activate`.

**`git : The term 'git' is not recognized...` (in VS Code's PowerShell terminal)**
VS Code's default terminal is PowerShell, which may not have Git in its PATH even if Git Bash does. Switch VS Code's default terminal to Git Bash: `Ctrl+Shift+P` -> "Terminal: Select Default Profile" -> Git Bash.

**`CUDA available: False` after installing torch**
You likely installed the CPU-only wheel. Reinstall explicitly with the CUDA index URL:
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
(Match the CUDA version to what `nvidia-smi` reports.)

**`FileNotFoundError: Video list file not found: data/video_list.txt`**
You haven't created `data/video_list.txt` yet, or you're running the command from the wrong folder. Check with `pwd` and `ls data/`.

**cp: cannot stat '...': No such file or directory**
The path you gave doesn't match the real file location or has a typo. Run `ls` on the parent folder to get the exact filename (watch for spaces — wrap paths with spaces in quotes).

**Annotation runs but detects nothing / very few things**
- Check `Using device: cuda` actually printed — if it silently fell back to CPU it still works, just slowly.
- Lower `conf_threshold` for the relevant stage in `config.yaml`.
- Confirm the PPE item is actually visible/unoccluded in the source footage at that frame.

**Pushing to GitHub asks for a password and rejects it**
GitHub no longer accepts account passwords for git operations over HTTPS. Generate a **Personal Access Token** (GitHub -> Settings -> Developer settings -> Personal access tokens -> Tokens (classic) -> Generate new token, with `repo` scope checked) and use that in place of your password when prompted.

---

## Project structure

```
yolo_annotation_pipeline/
|-- annotate.py           # main pipeline script
|-- config.yaml            # all settings: classes, thresholds, paths
|-- requirements.txt        # pip dependencies
|-- README.md               # this file
|-- .gitignore              # excludes venv/, data/, model weights, logs
|-- data/                   # created at runtime (not committed to git)
    |-- raw_videos/
    |-- video_list.txt
    |-- primary/
    |-- secondary/
```
