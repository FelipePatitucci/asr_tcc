from pathlib import Path
from typing import Callable

import numpy as np
from scipy.io import wavfile

from utils.mfcc_zero import mfcc, delta
from utils.helpers import find_folder


def pad_signal(
    signal: np.ndarray, sample_rate: int, target_time: float, crop: bool = True
) -> np.ndarray:
    """
    Pad signal with zeros to have a target time length. If the signal is already
    longer than the target time, it will be cropped (controlled by crop arg).
    Args:
        signal (np.ndarray): Signal to be padded
        sample_rate (int): Sample rate of the signal
        target_time (float): Target time (in seconds) that the signal should have
        crop (bool, optional): Whether to crop the signal if it is longer than the target time.
          Defaults to True.
    Returns:
        np.ndarray: Padded signal
    """
    size = len(signal)
    target = int(target_time * sample_rate)

    if size < target:
        zeros_to_add = np.zeros(target - size)
        signal = np.concatenate((signal, zeros_to_add))
    elif size > target and crop:
        signal = signal[:target]

    return signal


def calculate_metrics_from_wav(
    wav_path: str,
    delta_n: int = 2,
    should_pad: bool = True,
    maximum_length: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    samplerate, signal = wavfile.read(wav_path)
    if len(signal.shape) > 1 and signal.shape[1] == 2:
        signal = signal.mean(axis=1)

    if should_pad:
        signal = pad_signal(signal, samplerate, maximum_length)

    mfcc_feat = mfcc(signal, samplerate)
    delta_mfcc = delta(mfcc_feat, delta_n)
    mfcc_feat, delta_mfcc = (
        np.mean(mfcc_feat, axis=0),
        np.mean(delta_mfcc, axis=0),
    )

    return mfcc_feat, delta_mfcc


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return np.square(np.subtract(a, b)).mean()


def get_mfcc_and_deltas_from_character(
    table_name: str,
    character: Path | str,
    samplerate: int = 44100,
    winlen: float = 0.02,
    winstep: float = 0.01,
    nfilt: int = 26,
    nfft: int = 1024,
    numcep: int = 13,
    lowfreq: int = 0,
    highfreq: int = None,
    preemph: float = 0.97,
    ceplifter: int = 0,
    append_energy: bool = True,
    delta_n: int = 2,
    winfunc: Callable[[np.ndarray], np.ndarray] = np.hamming,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the average of MFCC features and deltas from a character's audio files.

    Parameters:
    character (str | Path): The name of the character or the path to the character's sample folder.
        Note: the path should be absolute and end with "cleaned_samples". The string should be the name of the character.
    sample_rate (int): The sample rate of the audio file. Default is 44100.

    Returns:
    tuple(np.ndarray, np.ndarray): The average of MFCC features and deltas.
    Note: the first coefficient is the log of the energies.
    """
    mfcc_mean = np.zeros(shape=(numcep,), dtype=np.float32)
    delta_mean = np.zeros(shape=(numcep,), dtype=np.float32)
    data_dir = (
        character
        if isinstance(character, Path)
        else find_folder(f"data/{table_name}/characters/{character}/cleaned_samples")
    )
    files = [f for f in data_dir.iterdir() if f.is_file()]

    if not files:
        print(f"No files found for character {character}!")
        return mfcc_mean, delta_mean

    print(f"Found {len(files)} files for character {character}! Processing...")

    for idx, file in enumerate(files):
        _, signal = wavfile.read(file)

        if len(signal.shape) > 1 and signal.shape[1] == 2:
            signal = signal.mean(axis=1)

        mfcc_feat = mfcc(
            signal,
            samplerate,
            winlen,
            winstep,
            nfilt,
            nfft,
            numcep,
            lowfreq,
            highfreq,
            preemph,
            ceplifter,
            append_energy,
            winfunc,
        )
        delta_mfcc = delta(mfcc_feat, delta_n)

        # add to the summed array
        mfcc_mean += np.mean(mfcc_feat, axis=0)
        delta_mean += np.mean(delta_mfcc, axis=0)

    return mfcc_mean / len(files), delta_mean / len(files)


def process_metrics_from_anime(
    table_name: str,
    characters_path: str = None,
    delta_n: int = 2,
    log: bool = False,
) -> None:
    characters_path = characters_path or f"data/{table_name}/characters"
    folder_path = find_folder(characters_path)
    for idx, character in enumerate(folder_path.iterdir()):
        coefs, deltas = get_mfcc_and_deltas_from_character(
            table_name,
            character.name,
            delta_n=delta_n,
        )

        if np.all(coefs == 0) or np.all(deltas == 0):
            if log:
                print(f"Empty coefficients or deltas for character {character.name}!")
            continue

        metrics_path = Path.joinpath(character, "metrics")
        if not Path.exists(metrics_path):
            try:
                Path.mkdir(metrics_path)
            except OSError as e:
                print(f"Error creating folder {metrics_path}: {e}")
                continue

        # save the MFCC and deltas removing the first coefficient (log of the total energy)
        np.save(Path.joinpath(metrics_path, "mfcc.npy"), coefs[1:])
        np.save(Path.joinpath(metrics_path, "delta.npy"), deltas[1:])

        print(f"Character: {character.name} saved!")
        if log:
            print(f"MFCC: {coefs[1:]}")
            print(f"Deltas: {deltas[1:]}")


# table_name = "sousou_no_frieren"
# delta_n = 1
# process_metrics_from_anime(
#     table_name, characters_path="data/sousou_no_frieren/characters", delta_n=delta_n
# )

# owner = "FRIEREN"
# candidates = ["HIMMEL", "FERN", "FRIEREN"]
# base_path = Path(f"data/{table_name}/characters")
# mfccs, deltas = (
#     np.load(Path.joinpath(base_path, f"{owner}/metrics/mfcc.npy")),
#     np.load(Path.joinpath(base_path, f"{owner}/metrics/delta.npy")),
# )


# results = {}
# for candidate in candidates:
#     errors_mfcc = []
#     candidate_path = Path.joinpath(base_path, f"{candidate}/cleaned_samples")
#     files = [f for f in candidate_path.iterdir() if f.is_file()]
#     for file in files:
#         mfc, _ = calculate_metrics_from_wav(
#             wav_path=file, delta_n=delta_n, should_pad=False
#         )
#         error = mse(mfccs, mfc[1:])
#         errors_mfcc.append(error)

#     errors_mfcc = np.array(errors_mfcc)
#     results[candidate] = {
#         "ALL": errors_mfcc,
#         "MEAN": np.mean(errors_mfcc),
#         "MIN": (np.min(errors_mfcc), np.argmin(errors_mfcc)),
#         "MAX": (np.max(errors_mfcc), np.argmax(errors_mfcc)),
#     }

# passes = {char: 0 for char in candidates}
# print(results)
# for char, vals in results.items():
#     for sample in vals["ALL"]:
#         if sample < results[owner]["MEAN"]:
#             passes[char] += 1
# print(passes)
#     plt.plot(vals["ALL"])
#     plt.title(char)
#     plt.show()
