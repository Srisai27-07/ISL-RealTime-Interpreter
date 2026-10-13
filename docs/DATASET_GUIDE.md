# 📊 Dataset Guide for Indian Sign Language (ISL)

> **Assigned to Member 1 (Dataset Engineer)**: This document outlines public dataset sources, collection protocols, data formats, and best practices for creating a robust ISL dataset.

---

## 🇮🇳 1. Why Indian Sign Language (ISL) is Unique

When pitching to hackathon judges, emphasize these crucial differences:
1. **Two-Handed Grammar**: While American Sign Language (ASL) uses one hand for its alphabet (fingerspelling) and many signs, ISL uses **both hands** for almost all alphabet letters and words.
2. **Spatial Dynamics**: Meaning in ISL is heavily conveyed by the relative spatial trajectory between both hands.
3. **Scarcity of Data**: ISL has far fewer labeled public datasets compared to ASL, making efficient few-shot keypoint modeling essential.

---

## 🌐 2. Recommended Public Datasets

If you are sourcing pre-existing data, prioritize these verified datasets:

| Dataset Name | Source / Institution | Description | Format |
| :--- | :--- | :--- | :--- |
| **INCLUDE Dataset** | AI4Bharat / IIT Madras | 4,287 videos, 263 ISL signs, recorded by deaf individuals. The gold-standard benchmark for ISL. | Raw MP4 videos |
| **ISL-CSLTR** | Continuous Sign Language Translation | Continuous sentences in ISL for hospital, banking, and travel domains. | Video clips |
| **ISLRTC Dictionary** | Ministry of Social Justice & Empowerment, Govt. of India | 10,000+ official standard ISL dictionary video clips. | Online video catalog |

> **Pro Tip**: If downloading raw videos, run our batch converter to transform them into fast 3D keypoint arrays:
> ```bash
> python scripts/02_extract_landmarks.py --input-dir path/to/videos --output-dir data/processed
> ```

---

## 📹 3. Custom Dataset Collection (Recommended for Hackathon Demo)

For a real-time live demo, recording **10–15 well-selected signs** across your 4 team members produces the highest live accuracy and fastest turnaround.

### High-Impact Hackathon Sign Vocabulary:
1. `Hello` (Greeting)
2. `Thank You` (Politeness)
3. `Please` (Request)
4. `Yes` (Affirmation)
5. `No` (Negation)
6. `Help` (Emergency / Accessibility)
7. `Water` (Basic need)
8. `Food` (Basic need)
9. `Good Morning` (Politeness)
10. `I Love You` (Universal expression)

### Step-by-Step Recording with `scripts/01_collect_data.py`:
1. Run the data collector:
   ```bash
   python scripts/01_collect_data.py --samples 25
   ```
2. Position yourself:
   - Sit 2–3 feet from the webcam so both hands and chest are clearly visible.
   - Avoid strong backlight (e.g., sitting directly with a window behind you).
3. The script will guide you:
   - Displays the target sign name.
   - Press **SPACE** when ready.
   - 3-second countdown (3.. 2.. 1..).
   - Perform the sign smoothly for ~1 second (30 frames).
   - Automatically saves the numpy array `(30, 126)` into `data/processed/<Sign_Name>/seq_<timestamp>.npy`.
4. **Collect 20–25 samples per class**:
   - Have at least 2 different team members perform each sign to teach the model invariance to hand size, skin tone, and speed!

---

## 📐 4. Landmark Array Format & Math

Each recorded sample is stored as a 2D float32 numpy array:
$$\text{Sample Shape} = (30 \text{ frames}, 126 \text{ features})$$

- **Frames 0 through 29**: 30 consecutive frames at ~30 FPS (1.0 second duration).
- **Features 0 through 62**: Left Hand (21 MediaPipe landmarks $\times [x, y, z]$).
- **Features 63 through 125**: Right Hand (21 MediaPipe landmarks $\times [x, y, z]$).
- **Missing Hand Handling**: If a sign only uses one hand (e.g. one hand is idle), the missing hand landmarks are zero-padded $[0.0, \dots, 0.0]$.

### Why Landmark Arrays instead of Raw Videos?
- **File size**: 1 raw video $\approx 5$ MB. 1 landmark array $\approx 15$ KB (**99.7% reduction**).
- **Training speed**: 50 epochs of BiLSTM takes under **3 minutes** on a standard CPU/MacBook!
- **Privacy & Invariance**: Landmark coordinates ignore background clutter, lighting variations, and personal appearance, focusing strictly on gesture geometry.
