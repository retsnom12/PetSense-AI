"""Audio feature extraction utilities for PetSense-AI."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def extract_audio_features(file_path: str, n_mfcc: int = 40) -> np.ndarray:
    """Extract a fixed-length vocalization feature vector from an audio file.

    The signal is resampled to 22,050 Hz, trimmed for leading and trailing
    silence, and represented by MFCC means and standard deviations plus
    spectral centroid and zero-crossing-rate means.

    Args:
        file_path: Path to a readable audio file supported by librosa.
        n_mfcc: Number of MFCC coefficients to calculate.

    Returns:
        A one-dimensional NumPy array with ``2 * n_mfcc + 2`` values.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
        ValueError: If the input or ``n_mfcc`` is invalid, or no usable signal
            remains after trimming.
        RuntimeError: If librosa cannot decode or process the audio.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {file_path}")
    if n_mfcc <= 0:
        raise ValueError("n_mfcc must be a positive integer.")

    try:
        signal, sample_rate = librosa.load(path, sr=22050, mono=True)
        trimmed_signal, _ = librosa.effects.trim(signal, top_db=25)
        if trimmed_signal.size == 0:
            raise ValueError("Audio contains no usable signal after silence trimming.")

        mfcc = librosa.feature.mfcc(
            y=trimmed_signal,
            sr=sample_rate,
            n_mfcc=n_mfcc,
        )
        spectral_centroid = librosa.feature.spectral_centroid(
            y=trimmed_signal,
            sr=sample_rate,
        )
        zero_crossing_rate = librosa.feature.zero_crossing_rate(trimmed_signal)

        features = np.concatenate(
            [
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1),
                [float(np.mean(spectral_centroid))],
                [float(np.mean(zero_crossing_rate))],
            ]
        )
    except ValueError:
        raise
    except (OSError, RuntimeError, librosa.util.exceptions.ParameterError) as error:
        raise RuntimeError(f"Unable to process audio file '{file_path}': {error}") from error

    return np.asarray(features, dtype=np.float32)

