# PetSense-AI

PetSense-AI is a multimodal, portfolio-ready machine-learning application for
classifying pet vocalizations and presenting interpretable facial-expression
proxies. The audio pipeline predicts four behavioral classes—`hungry`, `play`,
`stressed`, and `relaxed`—from WAV or MP3 recordings. The vision pipeline uses
an OpenCV frontal animal-face detector to estimate facial geometry and
muzzle-region edge density from an uploaded image.

> **Scope note:** These outputs are research and educational signals, not
> veterinary diagnoses. A qualified veterinary professional should assess any
> health or welfare concern.

## System architecture

```text
                         +----------------------+
                         |  Streamlit Web App   |
                         |      app.py          |
                         +----------+-----------+
                                    |
                  +-----------------+-----------------+
                  |                                   |
        +---------v---------+               +---------v---------+
        | Vocalization Tab  |               | Facial Metrics Tab|
        | WAV / MP3 upload  |               | JPG / PNG upload  |
        +---------+---------+               +---------+---------+
                  |                                   |
        +---------v---------+               +---------v---------+
        | librosa features |               | OpenCV Haar face  |
        | MFCC + spectrum  |               | + Canny muzzle ROI |
        +---------+---------+               +---------+---------+
                  |                                   |
        +---------v---------+               +---------v---------+
        | Scaler + Random  |               | Annotated image + |
        | Forest pipeline  |               | aspect/tension    |
        +---------+---------+               +-------------------+
                  |
        +---------v---------+
        | models/*.joblib   |
        +-------------------+
```

## Features

- Fixed-length audio features: MFCC means and standard deviations, spectral
  centroid mean, and zero-crossing-rate mean.
- Stratified train/test evaluation with a reproducible
  `StandardScaler` + `RandomForestClassifier` pipeline.
- Largest frontal animal-face detection and annotated image output.
- Streamlit tabs with upload validation, audio playback, prediction
  probabilities, metrics, and actionable missing-model messages.

## Setup

Python 3.10+ is recommended.

```bash
git clone <your-repository-url>
cd PetSense-AI

# Debian/Ubuntu: install the venv module if it is not already available.
# sudo apt install python3-venv
python3 -m venv venv
# macOS/Linux:
source venv/bin/activate
# Windows PowerShell:
# .\venv\Scripts\Activate.ps1

python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

If `python3 -m venv venv` reports that `ensurepip` is unavailable, install the
version-specific package on Debian/Ubuntu (for example,
`sudo apt install python3.12-venv`) and run the virtual-environment command
again.

## Dataset organization

Place labeled recordings under the matching class directory. Directory names
become the model labels, and only `.wav` and `.mp3` files are included.

```text
data/
├── hungry/
│   ├── cat_001.wav
│   └── cat_002.mp3
├── play/
├── stressed/
└── relaxed/
```

Use representative, consented recordings with consistent labeling. At least
10 valid audio files are required by the training script; a substantially
larger and balanced dataset is recommended for meaningful generalization.

## Train the audio classifier

Run training from the project root:

```bash
python3 -m src.train
```

The script prints overall accuracy and a classification report, then writes
the fitted pipeline to
`models/petsense_audio_classifier.joblib`. If fewer than 10 samples are
available, it prints a clear message and does not create a model.

## Launch the Streamlit application

```bash
streamlit run app.py
```

The **Vocalization Classifier** tab requires a trained model. The **Facial
Expression Metrics** tab can be used independently with a supported image.

## Sample evaluation report

The following is an illustrative output shape, not a benchmark:

```text
Overall accuracy: 0.7500

              precision    recall  f1-score   support

      hungry       0.80      0.67      0.73         3
         play       0.67      0.67      0.67         3
     relaxed       0.75      1.00      0.86         3
     stressed       0.80      0.67      0.73         3

    accuracy                           0.75        12
   macro avg       0.75      0.75      0.75        12
weighted avg       0.75      0.75      0.75        12
```

## Tech stack

- Python
- Librosa and SoundFile for audio loading and signal processing
- NumPy for numerical computation
- scikit-learn for preprocessing, classification, and evaluation
- OpenCV for image processing, Haar detection, and Canny edges
- MediaPipe is included for future multimodal landmark extensions
- Streamlit for the interactive web interface
- Joblib for model persistence

## Project structure

```text
PetSense-AI/
├── app.py
├── data/
├── models/
├── requirements.txt
└── src/
    ├── audio_features.py
    ├── train.py
    └── vision_metrics.py
```

## Future roadmap

1. Add more species, recording environments, and speaker-independent
   validation splits.
2. Introduce data augmentation and calibrated confidence estimates for noisy
   real-world audio.
3. Add MediaPipe landmarks and temporal video analysis alongside the Haar
   cascade.
4. Support batch inference, experiment tracking, and model version metadata.
5. Add automated tests, CI quality gates, and privacy-preserving dataset
   documentation.
