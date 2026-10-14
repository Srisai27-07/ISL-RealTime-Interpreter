import zipfile
import numpy as np
from PIL import Image
from scripts.prepare_tf_words import extract_archives
from src.tensorflow_isl.word_images import prepare_rgb, prepare_bgr


def test_word_training_and_webcam_pixels_match_without_stretching():
    rgb = np.zeros((80, 40, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255
    offline = prepare_rgb(Image.fromarray(rgb))
    live = prepare_bgr(rgb[:, :, ::-1])
    np.testing.assert_array_equal(offline, live)
    assert offline.shape == (160, 160, 3)
    assert offline[:, :35].max() == 0
    assert offline[80, 80, 0] == 255
    assert offline[80, 80, 2] == 0


def test_provider_user_groups_do_not_cross_splits(tmp_path):
    archive = tmp_path / "pray.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for i in range(1, 7):
            z.writestr(f"pray/User_{i}/pose.jpg", b"preserved raw bytes")
    rows = extract_archives(tmp_path)
    users = {s: {r["user"] for r in rows if r["split"] == s}
             for s in ("Training", "Validation", "Testing")}
    assert users["Training"] == {"User_1", "User_2", "User_3", "User_6"}
    assert users["Validation"] == {"User_4"}
    assert users["Testing"] == {"User_5"}
    assert not users["Training"] & (users["Validation"] | users["Testing"])
    assert archive.exists()
