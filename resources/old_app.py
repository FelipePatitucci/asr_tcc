import os
import wave
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from moviepy.editor import AudioFileClip, VideoFileClip
from pydub import AudioSegment
from scipy.fft import irfft, rfft, rfftfreq
# import sounddevice as sd


def plot_frequency_spectrum(
    freqs: np.array, magnitudes: np.array, character: str = "", episode: str = "1"
) -> plt.figure:
    a = plt.figure(figsize=(12, 6))
    plt.plot(freqs, magnitudes)
    plt.title(f"Frequency Spectrum - {character} (Episode {episode})")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    return a


def frequency_spectrum(
    input_video_path: str, plot: bool = True
) -> tuple[np.array, np.array]:
    try:
        with VideoFileClip(input_video_path) as clip:
            audio = clip.audio
            fps = audio.fps
            audio_array = audio.to_soundarray(fps=fps, nbytes=2)
    except:
        print(f"Failed to parse {input_video_path} as video. Parsing as audio...")
        with AudioFileClip(input_video_path) as clip:
            fps = clip.fps
            audio_array = clip.to_soundarray(fps=fps, nbytes=2)

    # change to mono
    if len(audio_array.shape) > 1 and audio_array.shape[1] == 2:
        audio_array = audio_array.mean(axis=1)

    # compute real fast fourier transform to find coeficients
    yf = rfft(audio_array)
    xf = rfftfreq(len(audio_array), 1 / fps)

    # Compute the magnitude of the FFT
    magnitude = np.abs(yf)

    if plot:
        fig = plot_frequency_spectrum(xf, magnitude)
        plt.show()

    return (xf, magnitude)


def save_audio_to_wav(filename: str, audio_array: np.array, fps: int) -> None:
    with wave.open(filename, "w") as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # bytes per sample
        wav_file.setframerate(fps)
        # Write frames to WAV file
        wav_file.writeframes((audio_array * 32767).astype(np.int16).tobytes())


def filter_noise_from_video(
    input_video_path: str, output_video_path: str, threshold: int = 300
) -> None:
    video = VideoFileClip(input_video_path)
    audio = video.audio
    fps = audio.fps
    audio_array = audio.to_soundarray(fps=fps, nbytes=2)

    # Flatten stereo audio to mono by averaging left and right channels
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)

    # Fast Fourier Transform to get frequency domain
    yf = rfft(audio_array)
    xf = rfftfreq(len(audio_array), 1 / fps)

    # Filter to remove background noise (zeroing out all low frequencies (TEST))
    threshold_frequency = threshold  # frequency threshold in Hz
    yf[xf < threshold_frequency] = 0

    # Convert back to time domain
    new_audio_array = irfft(yf)

    # Temporary filename for the processed audio
    temp_audio_path = "temp_filtered_audio.wav"
    save_audio_to_wav(temp_audio_path, new_audio_array, fps)

    # Load the new audio as an AudioFileClip
    new_audio = AudioFileClip(temp_audio_path)

    # Set this new audio clip to the video
    video_with_filtered_audio = video.set_audio(new_audio)

    video_with_filtered_audio.write_videofile(output_video_path, codec="libx264")

    os.remove(temp_audio_path)


def parse_frequencies(
    frequencies: np.array,
    magnitudes: np.array,
    bin_width: int = 10,
    min_freq: int = 0,
    max_freq: int = 22000,
) -> tuple[np.array, np.array]:
    """
    This function groups frequency values into bins of a specified width and
    sums the corresponding magnitudes.

    Parameters:
    - frequencies (np.array): An array of positive, increasing float values representing frequencies.
    - magnitudes (np.array): An array of float values representing magnitudes corresponding each frequency.
    - bin_width (int, optional): The width of each frequency bin. Default is 10.

    Returns:
    - new_freq (np.array): An array of midpoints for each frequency bin.
    - new_magnitudes (np.array): An array of summed magnitudes for each frequency bin.

    Each frequency is assigned to a bin defined by the bin width, and the magnitudes for frequencies
    within the same bin are summed together.
    """
    bins = np.arange(min_freq, max_freq + bin_width, bin_width)

    # Use np.digitize to find out which bin each frequency belongs to
    indices = np.digitize(frequencies, bins)

    # Sum up magnitudes in each bin
    new_magnitudes = [np.sum(magnitudes[indices == i]) for i in range(1, len(bins))]

    # Filter out bins that don't have any frequencies assigned
    new_freq = [
        (bins[i] + bins[i - 1]) / 2
        for i in range(1, len(bins))
        # if np.sum(magnitudes[indices == i]) > 0
    ]

    return np.array(new_freq), np.array(new_magnitudes)


def average_arrays(arrays: list[np.array]) -> np.array:
    """
    Computes the element-wise average of multiple Numpy arrays.

    Parameters:
    - arrays (list of np.array): Variable number of Numpy arrays. All arrays must be of the same size.

    Returns:
    - np.array: A Numpy array containing the element-wise average of the input arrays.

    Raises:
    - ValueError: If the input arrays do not all have the same size.
    """

    if len(arrays) == 0:
        return np.array([])

    # Check if all arrays have the same size
    array_shape = arrays[0].shape
    if not all(arr.shape == array_shape for arr in arrays):
        raise ValueError("All arrays must have the same size.")

    # Sum all arrays element-wise
    sum_array = np.sum(arrays, axis=0)

    # Compute the average
    average_array = sum_array / len(arrays)

    return average_array


def generate_avg_magnitude(
    samples_folder: str,
    amount_to_use: int = 1000,
    character: str = "Sung Jinwoo",
    episode: str = "1",
    bin_width: int = 10,
    min_freq: int = 0,
    max_freq: int = 22000,
    normalize: bool = True,
) -> tuple[np.array, np.array]:
    mags = []
    freqs = []
    init = time()
    paths = os.listdir(samples_folder)
    if amount_to_use > len(paths):
        amount_to_use = len(paths)

    for file in paths[:amount_to_use]:
        teste = os.path.join(samples_folder, file)
        freq, mag = frequency_spectrum(teste, False)
        freq, mag = parse_frequencies(freq, mag, bin_width, min_freq, max_freq)
        freqs = freq
        mags.append(mag)

    average_mag = average_arrays(mags)

    if normalize:
        total = average_mag.sum()
        average_mag = average_mag / total

    img = plot_frequency_spectrum(freqs, average_mag, character, episode)
    img.savefig(f"imgs/name={character}_n={amount_to_use}_ep={episode}.png")
    end = time()
    print(f"Took {end - init: .2f} seconds.")

    return freqs, average_mag


def compare_magnitutes(baseline: np.array, character: str = None):
    path = f"data/{character}"
    files = os.listdir(path)
    error_per_file = {}
    for idx, file in enumerate(files):
        curr_path = path + f"/{file}"
        freq, mag = frequency_spectrum(curr_path, False)
        freq, mag = parse_frequencies(freq, mag, 10, 0, 22000)
        total = mag.sum()
        normalized_mag = mag / total
        diff = np.subtract(baseline, normalized_mag)
        error_per_file[idx] = np.sum(np.abs(diff))

    total_error = sum([val for val in error_per_file.values()])
    print(f"Total error: {total_error}")
    plt.figure(figsize=(12, 6))
    plt.plot(error_per_file.keys(), error_per_file.values())
    plt.title(f"Comparing with {character} (TE: {round(total_error,1)}) (AE)")
    plt.xlabel("index")
    plt.ylabel("sum of absolute error element-wise")
    plt.savefig(f"comparing_with_{character}.png")


def mp4_to_wav_with_sr(mp4_file: str, wav_file: str, sample_rate: int = 16000) -> None:
    # Load the video file
    video = VideoFileClip(mp4_file)

    # Extract the audio
    audio = video.audio

    # Save the audio temporarily to a .wav file
    temp_wav = "temp_audio.wav"
    audio.write_audiofile(temp_wav)

    # Load the temporary .wav file with pydub
    sound = AudioSegment.from_wav(temp_wav)

    # convert to mono
    sound = sound.set_channels(1)

    # Set the desired sampling rate
    sound = sound.set_frame_rate(sample_rate)

    # Export the audio with the new sampling rate
    sound.export(wav_file, format="wav")

    # delete the temporary .wav file
    os.remove(temp_wav)


character = "SungJinwoo"
samples_folder = f"data/{character}"
amount_to_use = 1000  # to use all, just put a large number here like 9999
test_sample = samples_folder + "/segment_0059.mp4"


# freqs, mags = generate_avg_magnitude(
#     samples_folder,
#     amount_to_use=amount_to_use,
#     character=character,
#     episode="1-2",
#     bin_width=10,
#     min_freq=0,
#     max_freq=22000,
#     normalize=True
# )
# np.save("freqs_jw.npy", freqs, allow_pickle=False)
# np.save("mags_jw.npy", mags, allow_pickle=False)
# jw_mags = np.load("mags_jw.npy")
# compare_magnitutes(jw_mags, "SungJinwoo")
# frequency_spectrum(test_sample, True)
# filter_noise_from_video(clip_path, filtered_clip, 100)
mp4_to_wav_with_sr("data/SungJinwoo/ep_1_segment_0084.mp4", "teste.wav", 16000)
