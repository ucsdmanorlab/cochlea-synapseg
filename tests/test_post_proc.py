import numpy as np

from napari_cochlea_synapse_seg.utils.post_proc import sdt_to_labels


def _blob(shape, center, amplitude, sigma=2.0):
    zz, yy, xx = np.indices(shape)
    d2 = sum((g - c) ** 2 for g, c in zip((zz, yy, xx), center))
    return amplitude * np.exp(-d2 / (2 * sigma**2))


def test_sdt_to_labels_separates_blobs_and_drops_weak_peaks():
    shape = (16, 40, 40)
    pred = np.zeros(shape, dtype=np.float32)
    pred += _blob(shape, (8, 10, 10), 1.0)
    pred += _blob(shape, (8, 10, 30), 1.0)
    # peak below peak_thresh, so strict_peak_thresh should discard it
    pred += _blob(shape, (8, 30, 20), 0.1)

    labels = sdt_to_labels(pred, peak_thresh=0.5, mask_thresh=0.02)

    assert labels.shape == shape
    assert set(np.unique(labels)) == {0, 1, 2}
    assert labels[8, 10, 10] != labels[8, 10, 30]
    assert labels[8, 10, 10] > 0 and labels[8, 10, 30] > 0
    assert labels[8, 30, 20] == 0
    assert labels[0, 0, 0] == 0
