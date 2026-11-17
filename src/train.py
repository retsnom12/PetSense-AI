"""Train the PetSense-AI audio vocalization classifier."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.audio_features import extract_audio_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "models" / "petsense_audio_classifier.joblib"
SUPPORTED_EXTENSIONS = {".wav", ".mp3"}


def collect_dataset(data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Load all supported audio files and their parent directory labels."""
    feature_rows: list[np.ndarray] = []
    labels: list[str] = []
    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for audio_path in sorted(class_dir.iterdir()):
            if audio_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                feature_rows.append(extract_audio_features(str(audio_path)))
                labels.append(class_dir.name)
            except (FileNotFoundError, RuntimeError, ValueError) as error:
                print(f"Skipping '{audio_path}': {error}")

    if not feature_rows:
        return np.empty((0, 0), dtype=np.float32), np.asarray([], dtype=str)
    return np.vstack(feature_rows), np.asarray(labels)


def train_model() -> None:
    """Train, evaluate, and persist the audio classifier."""
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {DATA_DIR}")

    features, labels = collect_dataset()
    sample_count = len(labels)
    if sample_count < 10:
        print(
            f"At least 10 audio samples are required for reliable training; "
            f"found {sample_count}. Add more files under data/<class>/ and try again."
        )
        return
    if len(np.unique(labels)) < 2:
        print("At least two class directories with valid audio samples are required.")
        return

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    classifier = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(n_estimators=200, random_state=42),
            ),
        ]
    )
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    print(f"Overall accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(
        classification_report(
            y_test,
            predictions,
            labels=classifier.classes_,
            zero_division=0,
        )
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()

