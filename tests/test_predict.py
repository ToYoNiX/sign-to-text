import numpy as np

from features import N_FEATURES, extract


def _make_landmarks(n=21):
    return [{"x": float(i), "y": float(i) * 0.1, "z": 0.0} for i in range(n)]


def test_output_shape():
    result = extract(_make_landmarks())
    assert result.shape == (N_FEATURES,)


def test_output_dtype():
    result = extract(_make_landmarks())
    assert result.dtype == np.float32


def test_raw_coords_in_first_63():
    lms = [{"x": 1.0, "y": 2.0, "z": 3.0}] * 21
    result = extract(lms)
    expected_raw = np.array([1.0, 2.0, 3.0] * 21, dtype=np.float32)
    np.testing.assert_array_equal(result[:63], expected_raw)
