"""Tests for the STFT in preprocessing.signals."""

import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing.signals import compute_stft  # noqa: E402


def test_compute_stft_shape_and_determinism():
    x = np.random.default_rng(0).standard_normal(2048)
    s1 = compute_stft(x, n_fft=256, hop_length=64)
    s2 = compute_stft(x, n_fft=256, hop_length=64)
    assert s1.shape == (129, (2048 - 256) // 64 + 1)
    assert np.iscomplexobj(s1)
    np.testing.assert_array_equal(s1, s2)


def test_compute_stft_tone_peak_bin():
    n_fft = 256
    k = 32
    x = np.sin(2 * np.pi * k * np.arange(2048) / n_fft)
    s = np.abs(compute_stft(x, n_fft=n_fft, hop_length=128))
    assert np.all(np.argmax(s, axis=0) == k)


def test_compute_stft_short_signal_is_padded():
    s = compute_stft(np.ones(100), n_fft=256, hop_length=64)
    assert s.shape == (129, 1)
