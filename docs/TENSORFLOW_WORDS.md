# TensorFlow real-life word poses

This separate model uses the supplied real-life word **images**. It recognizes
the supplied static poses with one or two hands and produces word text. It does
not learn motion from these images, and does not train numbers or replace the
alphabet model. No sentences, translation, or speech features are added.

The image model uses a frozen ImageNet MobileNetV2 backbone (alpha 0.35, 160×160
RGB images), optional spatial hand landmarks, and a trained TensorFlow fusion
head. It retains the whole pose, including overlapping hands. A hand-landmark-only
approach could not detect the held-out prayer poses; a missing hand vector is
zero-filled while image information remains available. It pads images to a square without stretching them and uses
the same preprocessing for offline images and live unmirrored webcam frames.

## Data and duplicate removal

Place the twenty original word ZIP files in `data/drive_words/`. The preparation
script validates each ZIP member, preserves the archives, and assigns whole
provider User folders to splits before extraction:

| Split | Provider folders |
|---|---|
| Training | User_1, User_2, User_3, User_6 |
| Validation | User_4 |
| Testing | User_5 |

The doctor archive contains Users 1–5 only. Provider folder identities are not
independently verified. Images within one user may be related frames or
augmentations, so they must not be randomly scattered across these splits.
Exact byte/pixel duplicates and conservative near duplicates are excluded from
the processed dataset. Near duplicates require both pHash Hamming distance ≤3
and resized RGB RMS difference ≤2/255. Held-out representatives take priority.
Identical images with conflicting labels are quarantined. Similar cross-label
poses are flagged for review. Original ZIPs and images are not deleted.

## Prepare and train

Use the same Python 3.12 environment as the alphabet project:

```powershell
.\.venv-tf\Scripts\python.exe -m pip install -r requirements-tensorflow.txt
.\.venv-tf\Scripts\python.exe scripts/prepare_tf_words.py --data data/drive_words
.\.venv-tf\Scripts\python.exe scripts/train_tf_words.py --data data/processed/tf_words --output models/checkpoints/tf_words
```

Word outputs are `word.keras`, `metadata.json`, `reference.npz`, `evaluation.json`
and `history.json` in the separate `tf_words` directory. The first training run
downloads the official ImageNet backbone weights; the delivered complete model
does not require downloading those weights to run. Classification-head fitting
uses training data only. Model selection and confidence/distance gates use
validation data. The fixed test group is evaluated only after those choices.
The delivered experiment also records that this test group was evaluated in an
earlier image-only prototype; it is not a fresh untouched benchmark for the
whole development process. User folders alone do not verify signer identity.

## Webcam test

```powershell
.\.venv-tf\Scripts\python.exe -u app/live_tf_words.py --camera 0
```

Keep the relevant hand or both hands fully visible. Match the supplied reference
pose and hold it steady. Include your upper body as in the dataset. Five agreeing
frames commit a word; an uncertain interval resets the repeat filter. Press `c`
to clear text and `q` to quit. Preview mirroring does not affect inference.
Uncertain or distant image embeddings produce no word. This is a closed-vocabulary
image classifier: rejection of every unsupported gesture or no-hand scene is
not guaranteed. Live background/clothing changes can affect its predictions.
If no validation policy achieves the accepted-accuracy target, live word output
is disabled in metadata rather than presenting unreliable predictions as text.

This command opens an OpenCV webcam window. The existing browser alphabet mode
continues using the alphabet model; browser word mode is not part of this bundle.
Preregistered image replay can check the camera pipeline, but performance on a
new live signer still needs testing on the user's computer.
