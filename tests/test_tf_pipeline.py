import json
import numpy as np
from PIL import Image
from src.tensorflow_isl.features import normalize_hands, browser_features
from src.tensorflow_isl.stability import StablePrediction
from src.tensorflow_isl.audit import audit_dataset


def hand(x=0.2):
    values = np.zeros((21, 3), dtype=np.float32)
    values[:, 0] = x + np.linspace(0, .08, 21)
    values[:, 1] = .3 + np.linspace(0, .16, 21)
    return values


def test_common_translation_and_scale_invariance():
    a, b = hand(.2), hand(.6)
    expected = normalize_hands([a, b])
    offset = np.array([.1, .2, 0])
    assert np.allclose(expected, normalize_hands([b*1.5+offset, a*1.5+offset]), atol=1e-5)


def test_inter_hand_distance_is_preserved():
    assert not np.allclose(normalize_hands([hand(.2), hand(.6)]),
                           normalize_hands([hand(.2), hand(.8)]))


def test_browser_aspect_ratio_matches_square_padding():
    a, b = hand(.2), hand(.6)
    padded = []
    for h in (a,b):
        h = h.copy()
        h[:,1] = h[:,1]*.75+.125
        padded.append(h)
    assert np.allclose(browser_features([a,b],640,480), normalize_hands(padded))


def test_missing_hand_is_not_a_training_vector():
    assert normalize_hands([]) is None
    assert normalize_hands([np.zeros((21,3))]) is None
    v = normalize_hands([hand()])
    assert np.array_equal(v[126:], [1, 0])
    assert np.all(v[63:126] == 0)


def test_uncertainty_breaks_consensus_and_rearms_repeat():
    stable = StablePrediction(3)
    assert stable.update('A') == (None, False)
    stable.update('A')
    stable.update(None)
    assert stable.update('A') == (None, False)
    stable.update('A')
    assert stable.update('A') == ('A', True)
    assert stable.update('A') == ('A', False)
    stable.update(None)
    stable.update('A'); stable.update('A')
    assert stable.update('A') == ('A', True)


def test_duplicate_is_excluded_from_training_without_deleting_raw(tmp_path):
    image = Image.fromarray(np.random.default_rng(2).integers(0,256,(40,40,3),dtype=np.uint8))
    for split in ('Training','Testing'):
        p = tmp_path/split/'A'/'same.png'
        p.parent.mkdir(parents=True)
        image.save(p)
    rows, report = audit_dataset(tmp_path, tmp_path/'report')
    assert report['status_counts'] == {'keep': 1, 'duplicate': 1}
    assert next(r for r in rows if r['status']=='keep')['split'] == 'Testing'
    assert (tmp_path/'Training'/'A'/'same.png').is_file()


def test_conflicting_duplicate_labels_are_all_quarantined(tmp_path):
    for label in ('A','B'):
        p = tmp_path/'Training'/label/'same.png'
        p.parent.mkdir(parents=True)
        Image.new('RGB',(40,40),'red').save(p)
    rows, _ = audit_dataset(tmp_path, tmp_path/'report')
    assert all(r['status']=='label_conflict' for r in rows)
