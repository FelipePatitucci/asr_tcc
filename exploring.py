import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from mfcc_zero import fbank, mfcc, delta


def plot_signal(
    ts,
    signal,
    title: str = "Amplitude x Time for .wav with sr of 44.1k",
    xlabel: str = "Time (s)",
    ylabel: str = "Amplitude",
):
    plt.figure(figsize=(10, 6))
    plt.plot(ts, signal, label="teste.wav", color="green")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


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


# first, we get the samples using another file,
# then pass through a model to remove background music
sample_rate, signal = wavfile.read("teste_before.wav")
print("SR:", sample_rate)
if len(signal.shape) > 1 and signal.shape[1] == 2:
    signal = signal.mean(axis=1)
print("Signal shape:", signal.shape)

# signal = pad_signal(signal, sample_rate, target_time=5)

feats, energy = fbank(
    signal,
    sample_rate,
    nfft=1024,
    winlen=0.02,
    winfunc=np.hamming,
)
mfcc_feat = mfcc(
    signal,
    sample_rate,
    nfft=1024,
    winlen=0.02,
    ceplifter=0,
    append_energy=True,
    winfunc=np.hamming,
)
teste = np.mean(mfcc_feat, axis=0)
np.testing.assert_array_equal(mfcc_feat[:, 0], np.log(energy))
assert np.isclose(np.mean(mfcc_feat[:, 0]), teste[0], atol=1e-5)
print(teste.shape, teste)
# d_mfcc_feat = delta(mfcc_feat, 2)
# # d_d_mfcc_feat = delta(d_mfcc_feat, 2)
# print(f"MFCC coefs of the first 3 frames: (shape {mfcc_feat.shape})")
# for idx, frame in enumerate(mfcc_feat[:3]):
#     print(f"Frame {idx + 1}:", frame)
# print("Delta MFCC coefs of the first 3 frames:")
# print(d_mfcc_feat[:3][1:])
# plt.figure(figsize=(10, 5))
# plt.imshow(np.log(d_mfcc_feat.T), aspect="auto", origin="lower", cmap="jet")
# plt.title("Log das energias")
# plt.xlabel("Tempo (s)")
# plt.ylabel("Índice dos Canais")
# plt.colorbar(label="Log Energia")
# plt.show()
