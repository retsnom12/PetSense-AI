"""Computer-vision metrics for pet facial expression analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class FacialExpressionAnalyzer:
    """Detect a frontal animal face and calculate interpretable proxy metrics."""

    def __init__(self, cascade_path: str | None = None) -> None:
        """Initialize the Haar cascade used for frontal animal-face detection."""
        selected_path = cascade_path or str(
            Path(cv2.data.haarcascades) / "haarcascade_frontalcatface_extended.xml"
        )
        self.cascade = cv2.CascadeClassifier(selected_path)
        if self.cascade.empty():
            raise RuntimeError(
                f"Unable to load the animal-face Haar cascade from '{selected_path}'."
            )

    def analyze(self, frame: np.ndarray) -> dict[str, Any]:
        """Analyze a BGR frame and return the largest detected face metrics.

        Args:
            frame: A non-empty BGR image as a NumPy array.

        Returns:
            A dictionary containing ``face_detected``, ``bbox``,
            ``aspect_ratio``, and ``whisker_tension_proxy``. If no face is
            detected, numeric metrics are zero and the bounding box is
            ``(0, 0, 0, 0)``.

        Raises:
            ValueError: If ``frame`` is empty or not a valid color image.
        """
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty NumPy array.")
        if frame.ndim not in (2, 3):
            raise ValueError("frame must be a two-dimensional grayscale or three-dimensional BGR image.")

        grayscale = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            grayscale,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        if len(faces) == 0:
            return {
                "face_detected": False,
                "bbox": (0, 0, 0, 0),
                "aspect_ratio": 0.0,
                "whisker_tension_proxy": 0.0,
            }

        x, y, width, height = max(faces, key=lambda box: int(box[2]) * int(box[3]))
        roi = grayscale[y : y + height, x : x + width]
        muzzle_region = roi[height // 2 :, :]
        edges = cv2.Canny(muzzle_region, threshold1=50, threshold2=150)
        whisker_tension_proxy = float(np.count_nonzero(edges) / edges.size) if edges.size else 0.0

        return {
            "face_detected": True,
            "bbox": (int(x), int(y), int(width), int(height)),
            "aspect_ratio": float(width / height),
            "whisker_tension_proxy": whisker_tension_proxy,
        }

