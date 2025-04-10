import os
import numpy as np
from pathlib import Path
from hmmlearn import hmm
import librosa
import logging

from utils.helpers import find_folder

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


def extract_features(file_path: str, n_mfcc=13, delta_n=2, sample_rate=44100):
    """
    Extract MFCC, delta MFCC, and delta-delta MFCC features from the audio file.
    Returns the features as a combined array of MFCCs and deltas.
    """
    # Load audio file
    y, sr = librosa.load(file_path, sr=sample_rate)

    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Extract delta MFCC
    delta = librosa.feature.delta(mfcc, order=delta_n)

    # Combine MFCCs and delta features (delta-delta will be created by chaining delta)
    delta_delta = librosa.feature.delta(delta, order=delta_n)

    # Stack features vertically (MFCC + delta + delta-delta)
    combined_features = np.vstack([mfcc, delta, delta_delta])

    return combined_features.T  # Return as a time-series of feature vectors


def train_hmm_model(features: np.ndarray, n_components: int = 5):
    """
    Train a Hidden Markov Model (HMM) using the given features.
    """
    # Create an HMM with a specified number of hidden states
    model = hmm.GaussianHMM(
        n_components=n_components, covariance_type="diag", n_iter=1000
    )

    # Fit the model to the data
    model.fit(features)

    return model


def create_speaker_model(training_data_folder: str, n_components: int = 5):
    """
    Train a model for a given speaker using the provided training data folder.
    """
    logger.info(f"Training model for speaker at {training_data_folder}...")

    features = []
    # List all sample files in the 'samples' directory for a character
    sample_files = [f for f in os.listdir(training_data_folder) if f.endswith(".wav")]

    logger.info(f"Found {len(sample_files)} audio samples for training.")
    for file_name in os.listdir(training_data_folder):
        if file_name.endswith(".wav"):  # Only process .wav files
            file_path = os.path.join(training_data_folder, file_name)
            file_features = extract_features(file_path)
            features.append(file_features)

    features = np.concatenate(features, axis=0)

    # Train the HMM model with the extracted features
    model = train_hmm_model(features, n_components=n_components)

    return model


def recognize_speaker(
    test_file: str, models: dict, speakers: list, n_components: int = 5
):
    """
    Recognize the speaker based on the test file by comparing it with models for all speakers.
    """
    test_features = extract_features(test_file)

    best_score = -np.inf
    recognized_speaker = None

    # Compare the test features against each speaker's model
    for speaker in speakers:
        score = models[speaker].score(test_features)

        if score > best_score:
            best_score = score
            recognized_speaker = speaker

    return recognized_speaker, best_score


# Example Usage
# Define the root directory containing the character data
root_dir = "data/sousou_no_frieren/characters"

# List of character names
speakers = ["FERN", "FRIEREN", "HIMMEL"]

# Create models for each speaker
models = {}
for speaker in speakers:
    speaker_folder = os.path.join(root_dir, speaker, "samples")
    models[speaker] = create_speaker_model(speaker_folder)

# Test the system by recognizing the speaker from a test file
table_name = "sousou_no_frieren"
char = "HIMMEL"
data_folder = find_folder(f"data/{table_name}/characters/{char}/samples")
test_files = [f for f in data_folder.iterdir() if f.is_file()]
matches = {}
scores = []
for file in test_files:
    recognized_speaker, score = recognize_speaker(file, models, speakers)
    # print(f"File: {file}: Recognized Speaker: {recognized_speaker} with score: {score}")
    if recognized_speaker not in matches:
        matches[recognized_speaker] = 1
    else:
        matches[recognized_speaker] += 1
    scores.append(score)
print(matches)
print(sum(scores) / len(scores))
