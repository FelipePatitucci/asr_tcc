import json
import logging
from pathlib import Path

import librosa
import numpy as np
from hmmlearn import hmm
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


def get_all_subfolders(folder_path: str) -> list[str]:
    """
    Get the names of all subfolders within a given folder.

    Parameters:
    - folder_path (str): The path to the folder where you want to list subfolders.

    Returns:
    - list[str]: A list of folder names inside the provided folder.
    """
    path = Path(folder_path)

    if not path.is_dir():
        raise ValueError(f"{folder_path} is not a valid directory.")

    return [folder.name for folder in path.iterdir() if folder.is_dir()]


def extract_features(file_path: str, n_mfcc=13, delta_n=2, sample_rate=44100):
    """
    Extract MFCC, delta MFCC, and delta-delta MFCC features from the audio file.
    Returns the features as a combined array of MFCCs and deltas.
    """
    y, sr = librosa.load(file_path, sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc, order=delta_n)
    # Combine MFCCs and delta features (delta-delta will be created by chaining delta)
    delta_delta = librosa.feature.delta(delta, order=delta_n)
    combined_features = np.vstack([mfcc, delta, delta_delta])

    return combined_features.T


def split_train_test(file_paths: list[Path], test_size=0.2):
    """
    Split the file paths into training and testing data.
    """
    logger.info(f"Splitting {len(file_paths)} files into training and testing...")
    train_paths, test_paths = train_test_split(
        file_paths, test_size=test_size, random_state=42
    )
    logger.info(f"Train size: {len(train_paths)}, Test size: {len(test_paths)}")

    return train_paths, test_paths


def train_hmm_model(
    features: np.ndarray, n_components: int = 5, n_iter: int = 1000
) -> hmm.GaussianHMM:
    """
    Train a Hidden Markov Model (HMM) using the given features.
    """
    # Create an HMM with a specified number of hidden states
    model = hmm.GaussianHMM(
        n_components=n_components, covariance_type="diag", n_iter=n_iter
    )
    model.fit(features)

    return model


def create_speaker_model(
    speaker: str, training_file_paths: str, n_components: int = 5, n_iter: int = 1000
) -> hmm.GaussianHMM:
    """
    Train a model for a given speaker using the provided training data folder.
    """
    logger.info(f"Training model for speaker {speaker}...")
    logger.info(f"{len(training_file_paths)} audio samples available.")
    features = []

    for file_path in training_file_paths:
        file_features = extract_features(file_path)
        features.append(file_features)

    features = np.concatenate(features, axis=0)
    # Train the HMM model with the extracted features
    model = train_hmm_model(features, n_components=n_components, n_iter=n_iter)

    return model


def recognize_speaker(
    test_file: str, models: dict[str, hmm.GaussianHMM], speakers: list[str]
) -> tuple[str, float]:
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


# vars
table_name = "sousou_no_frieren"
root_dir = "data/sousou_no_frieren/characters"
data_folder_name = "samples"
# speakers = ["FERN", "FRIEREN", "HIMMEL"]
speakers = get_all_subfolders(root_dir)
min_samples = 10  # minimum number of samples for a speaker to be considered
n_components = 5
n_iter = 1000
test_size = 0.2  # percentage in float
models = {}
data = {}

# training
for speaker in speakers:
    speaker_folder = Path.joinpath(Path(root_dir), speaker, data_folder_name)
    all_files = [f for f in speaker_folder.iterdir() if f.is_file()]
    if len(all_files) < min_samples:
        print(
            f"Insufficient samples ({len(all_files)}/{min_samples}) for speaker {speaker}!"
        )
        continue
    train_files, test_files = split_train_test(all_files, test_size)
    models[speaker] = create_speaker_model(speaker, train_files, n_components, n_iter)
    data[speaker] = test_files

# testing
matches = {
    speaker: {speaker: 0 for speaker in models.keys()} for speaker in models.keys()
}
for speaker, test_files in data.items():
    for file in test_files:
        recognized_speaker, score = recognize_speaker(file, models, list(models.keys()))
        matches[speaker][recognized_speaker] += 1
    matches[speaker]["accuracy"] = matches[speaker][speaker] / len(test_files)
with open("matches.json", "w") as f:
    json.dump(matches, f)
