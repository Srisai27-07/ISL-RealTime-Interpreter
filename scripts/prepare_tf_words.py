"""Extract word ZIPs, split by provider user, audit and cache optional hands."""
import argparse
import json
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath


def extract_archives(root):
    root = Path(root)
    records = []
    archives = sorted(root.glob("*.zip"))
    if not archives:
        raise ValueError(f"No word ZIP files found in {root}")
    for archive in archives:
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                if member.is_dir():
                    continue
                parts = PurePosixPath(member.filename).parts
                if (len(parts) != 3 or ".." in parts or parts[0] != archive.stem
                        or parts[1] not in {f"User_{i}" for i in range(1, 7)}
                        or Path(parts[2]).suffix.lower() not in {".jpg", ".jpeg", ".png"}):
                    raise ValueError(f"Unexpected archive member: {member.filename}")
                label, user, name = parts
                split = {"User_4": "Validation", "User_5": "Testing"}.get(user, "Training")
                target = root / "split" / split / label / (user + "__" + name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read(member))
                records.append({"archive": archive.name, "member": member.filename,
                                "label": label, "user": user, "split": split,
                                "path": str(target.resolve())})
    expected = {r["path"] for r in records}
    actual = {str(p.resolve()) for p in (root / "split").rglob("*") if p.is_file()}
    if expected != actual:
        raise ValueError("Split folder contains stale files. Use a new dataset directory.")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Directory containing word ZIPs")
    parser.add_argument("--output", default="data/processed/tf_words")
    parser.add_argument("--hand-model", default="models/checkpoints/hand_landmarker.task")
    args = parser.parse_args()
    records = extract_archives(args.data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "source_records.json").write_text(json.dumps(records, indent=2))
    print(f"Extracted {len(records)} word images; fixed provider-user split", flush=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name("prepare_tf_alphabets.py")),
                    "--data", str(Path(args.data) / "split"), "--output", str(output),
                    "--hand-model", args.hand_model], check=True)
    summary = json.loads((output / "audit.json").read_text())
    summary.update(user_disjoint_split=True,
                   split_users={"Training": ["User_1", "User_2", "User_3", "User_6"],
                                "Validation": ["User_4"], "Testing": ["User_5"]},
                   signer_evaluation="Provider User folders are disjoint; actual identity metadata is unverified.",
                   data_type="Still images; movement recognition is not evaluated",
                   original_by_user=dict(Counter(r["user"] for r in records)))
    (output / "audit.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
