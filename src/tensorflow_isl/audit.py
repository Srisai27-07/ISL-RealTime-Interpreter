"""Non-destructive deduplication across the original split assignments."""
import hashlib
import json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

SPLITS = ("Testing", "Validation", "Training")  # preserve held-out representatives
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def phash(image):
    values = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float32)
    low = dctn(values, norm="ortho")[:8, :8].ravel()
    bits = low > np.median(low[1:])
    return sum(int(v) << i for i, v in enumerate(bits))


def audit_dataset(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, thumbs = [], []
    for split in SPLITS:
        for path in sorted((root / split).glob("*/*")):
            if path.suffix.lower() not in EXTENSIONS:
                continue
            row = {"path": str(path.resolve()), "split": split,
                   "label": path.parent.name, "status": "candidate"}
            try:
                raw = path.read_bytes()
                with Image.open(path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                    row.update(sha256=hashlib.sha256(raw).hexdigest(),
                               pixel_hash=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest(),
                               phash=phash(image), width=image.width, height=image.height)
                    thumbs.append(np.asarray(image.resize((32, 32)), dtype=np.int16))
            except Exception as error:
                row.update(status="invalid_image", reason=str(error))
                thumbs.append(None)
            rows.append(row)
    # Exact byte/pixel duplicate components, including differently compressed copies.
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def join(a, b):
        parent[find(b)] = find(a)
    seen = {}
    for i, row in enumerate(rows):
        if row["status"] != "candidate":
            continue
        for key in ("sha256", "pixel_hash"):
            token = (key, row[key])
            if token in seen:
                join(seen[token], i)
            else:
                seen[token] = i
    # Conservative near-duplicate check. A matching pHash alone is insufficient.
    # Four 16-bit buckets guarantee candidate retrieval at Hamming distance <=3.
    buckets, near_pairs, cross_label_pairs = defaultdict(list), [], []
    for i, row in enumerate(rows):
        if row["status"] != "candidate":
            continue
        h = row["phash"]
        keys = [(part, (h >> (part*16)) & 65535) for part in range(4)]
        candidates = set(j for key in keys for j in buckets[key])
        for j in candidates:
            if find(i) == find(j) or (h ^ rows[j]["phash"]).bit_count() > 3:
                continue
            # Confirm almost identical appearance at a common resolution.
            rms = float(np.sqrt(np.mean((thumbs[i].astype(float)-thumbs[j])**2)))
            if rms > 2.0:
                continue
            pair = {"first": rows[j]["path"], "second": row["path"], "rms_255": rms}
            if rows[j]["label"] == row["label"]:
                join(j, i)
                near_pairs.append(pair)
            else:
                # Similar poses in different classes require review, not deletion.
                cross_label_pairs.append(pair)
        for key in keys:
            buckets[key].append(i)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if row["status"] == "candidate":
            groups[find(i)].append(i)
    rank = {split: n for n, split in enumerate(SPLITS)}
    conflicts = []
    for indices in groups.values():
        labels = {rows[i]["label"] for i in indices}
        if len(labels) > 1:
            conflicts.append([rows[i]["path"] for i in indices])
            for i in indices:
                rows[i]["status"] = "label_conflict"
            continue
        keep = min(indices, key=lambda i: (rank[rows[i]["split"]], rows[i]["path"]))
        for i in indices:
            rows[i]["status"] = "keep" if i == keep else "duplicate"
            rows[i]["representative"] = rows[keep]["path"]
    summary = {
        "total_images": len(rows), "status_counts": dict(Counter(r["status"] for r in rows)),
        "original_split_counts": dict(Counter(r["split"] for r in rows)),
        "retained_split_counts": dict(Counter(r["split"] for r in rows if r["status"] == "keep")),
        "retained_by_class": {s: dict(Counter(r["label"] for r in rows
                                    if r["split"] == s and r["status"] == "keep")) for s in SPLITS},
        "exact_label_conflicts": conflicts,
        "near_duplicate_pairs": len(near_pairs),
        "similar_cross_label_pairs": cross_label_pairs,
        "near_duplicate_rule": "pHash Hamming <=3 AND resized RGB RMS <=2 on 0..255",
        "signer_evaluation": "Signer identity not supplied. Dataset test split is not verified signer-independent.",
        "raw_files_deleted": False,
    }
    (output / "manifest.json").write_text(json.dumps(rows, indent=2))
    (output / "audit.json").write_text(json.dumps(summary, indent=2))
    return rows, summary
