import numpy as np

from predict import landmarks_to_features


def _make_landmarks(n=21):
    return [{"x": float(i), "y": float(i) * 0.1, "z": 0.0} for i in range(n)]


def test_output_shape():
    result = landmarks_to_features(_make_landmarks())
    assert result.shape == (1, 63)


def test_output_dtype():
    result = landmarks_to_features(_make_landmarks())
    assert result.dtype == np.float32


def test_values_flattened_correctly():
    lms = [{"x": 1.0, "y": 2.0, "z": 3.0}] * 21
    result = landmarks_to_features(lms)
    expected = np.array([1.0, 2.0, 3.0] * 21, dtype=np.float32).reshape(1, -1)
    np.testing.assert_array_equal(result, expected)


def test_partial_landmarks_produce_partial_shape():
    # landmarks_to_features doesn't validate count — documents current behaviour
    result = landmarks_to_features(_make_landmarks(n=5))
    assert result.shape == (1, 15)
