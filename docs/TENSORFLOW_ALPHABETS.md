# TensorFlow alphabet training and webcam recognition

This path implements static A-Z recognition with one or two hands and text output.
Two hands are included at the user's request. This expands the original PS's
single-hand constraint. Dynamic word videos are not included in this classifier.

## Dataset

Use the supplied Training, Validation, and Testing folders, each containing A-Z
subfolders. The Drive inventory on 14 October 2026 contained 14,232 Training,
2,540 Validation, and 4,591 Testing images (21,363 total). Counts refer to listed
files, before download, duplicate removal, or landmark rejection.

The Letters folder contains 26 reference pictures. It is not the training corpus.
Numbers and word-video collections in the parent folder are not mixed into A-Z.

Download the three folders to `data/Dataset/`. Expected structure:

```text
data/Dataset/Training/A/*.jpg
data/Dataset/Training/B/*.jpg
data/Dataset/Validation/A/*.jpg
data/Dataset/Testing/A/*.jpg
```

## Setup

From the repository root, use a separate environment for this workflow.

```powershell
python -m venv .venv-tf
.\.venv-tf\Scripts\Activate.ps1
python -m pip install -r requirements-tensorflow.txt
python scripts/download_hand_model.py
```

On Linux, activate with `source .venv-tf/bin/activate`. MediaPipe may require
the system `libgles2`, `libegl1`, and `libgl1` libraries even for CPU processing.
Use desktop OpenCV on the laptop to display a webcam window. Do not install
headless OpenCV in the laptop environment.

## Audit, preprocessing, and training

```powershell
python scripts/prepare_tf_alphabets.py --data data/Dataset
python scripts/train_tf_alphabets.py
python -m pytest tests/test_tf_pipeline.py -q
```

Raw images remain unchanged. The manifest excludes corrupt images, exact byte
copies, identical decoded pixels, and conservatively confirmed near copies.
Near copies require pHash Hamming distance <=3 AND resized RGB RMS <=2/255.
Perceptual similarity alone does not cause removal. Different-label identical
copies are quarantined, and near copies across labels are flagged for review.

Duplicate groups retain a representative in Testing first, then Validation,
then Training. This removes their copies from training without moving held-out
images into training. The manifest and audit report record decisions.

The detector extracts up to two hands. Both use a common spatial center and
scale. Hand slots are ordered by wrist x-coordinate, not handedness labels.
Features contain 126 landmark values plus two hand-presence flags. Relative
hand positions remain available. Each hand's z values are wrist-relative and
are not treated as measured inter-hand depth. Images are square-padded to
preserve aspect ratio. No horizontal flip is applied before inference.

Only Training fits the classifier and its normalization layer. Validation
selects one of two small dense Keras networks, early stopping, and acceptance
rules. The test set is evaluated after those choices. The output contains:

- `alphabet.keras`: classifier plus fitted feature normalization.
- `metadata.json`: exact class ordering, preprocessing version, acceptance rules.
- `reference.npz`: training features for the distribution-distance gate.
- `evaluation.json`: test metrics and limitations.
- `history.json`: training and validation learning curves.

Output directory: `models/checkpoints/tf_alphabets/`. These generated files and
raw data are ignored by Git. Share the complete model directory separately.

Acceptance requires confidence, top-two probability margin, and proximity to
training examples of the predicted class. Validation policy selection targets
95% accuracy among accepted examples. If no setting meets that target with
enough examples, text acceptance is disabled instead of lowering the target.
This is a validation target, not a promised real-world accuracy level.

## Webcam

```powershell
python app/live_tf_alphabets.py
```

Use `--camera 1` for another camera. `q` quits, and `c` clears the displayed
text. Five consistent accepted predictions commit a letter. An uncertain
prediction clears pending consensus. No hand suppresses output. The preview
is mirrored for convenience, but inference uses the unmirrored frame.

For a prerecorded integration check:

```powershell
python app/live_tf_alphabets.py --video sample.mp4 --headless
```

## Existing browser application

Install `fastapi` and `uvicorn` in the same environment if running the existing
browser UI, then start the server from the repository root:

```powershell
python -m pip install fastapi uvicorn
python -m uvicorn api.server:app --host 127.0.0.1 --port 8000
```

The alphabet API loads the TensorFlow model when its directory exists. The
frontend sends unmirrored hand arrays and frame dimensions. The backend applies
the same common-center/scale features after mapping coordinates to a square
frame. Existing legacy model paths remain available when no TensorFlow model
exists. Missing models no longer return a fabricated confident A prediction.
WebSocket responses use actual model probabilities and an explicit acceptance
flag; low-confidence outputs cannot be automatically committed as text.

The browser's legacy MediaPipe detector and Python Tasks detector may give
slightly different landmarks. Test this browser path on the laptop; the desktop
entry point uses exactly the same detector as offline preprocessing.

## Evaluation limits

The supplied folder names alone do not establish signer independence. No signer
IDs have been supplied. Report dataset-split test accuracy accurately and record
a separate unseen signer before claiming the PS's unseen-signer demonstration.
Source recording and augmentation group IDs are also unavailable. Conservative
duplicate removal does not prove that all related frames or augmented variants
are separated across splits. Treat the supplied test split as provisional until
its provenance can be verified.
Unsupported gestures and webcam lighting/orientation variations need separate
live tests. High softmax confidence is not proof of a valid ISL gesture.

Some two-hand signs overlap or occlude fingers. The detector can miss a hand or
produce unstable keypoints; no-hand detections are recorded and excluded.
The live system recognizes static alphabet labels, not dynamic signs or ISL
sentence grammar. Inspect `evaluation.json` before enabling a demonstration.

## References

- https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
- https://www.tensorflow.org/guide/keras/training_with_built_in_methods
