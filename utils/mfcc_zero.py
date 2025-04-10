from decimal import Decimal, ROUND_HALF_UP
from math import ceil
from typing import Callable

import numpy as np
from scipy.fftpack import dct


def hz2mel(hz: float | int | np.ndarray):
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1 + (hz / 700.0))


def mel2hz(mel: float | int | np.ndarray):
    """Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700 * (10 ** (mel / 2595.0) - 1)


def round_half_up(number):
    """
    Rounds a number to the nearest integer, rounding half up.

    :param number: The number to round.
    :returns: The rounded number.
    """
    return int(Decimal(number).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def preemphasis(signal: np.ndarray, coeff: float = 0.95) -> np.ndarray:
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def framesig(
    sig: np.ndarray,
    frame_len: float,
    frame_step: float,
    winfunc: Callable[[np.ndarray], np.ndarray] = lambda x: np.ones((x,)),
) -> np.ndarray:
    """Frame a signal into overlapping frames.

    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :returns: an array of frames. Shape is (NUMFRAMES, frame_len).
    """
    # get sizes
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))

    # NUMFRAMES
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(ceil((1.0 * slen - frame_len) / frame_step))

    # PADZEROS
    padlen = int((numframes - 1) * frame_step + frame_len)
    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((sig, zeros))

    # TILES
    indices = (
        np.tile(np.arange(0, frame_len), (numframes, 1))
        + np.tile(np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
    )
    indices = np.array(indices, dtype=np.int32)
    frames = padsignal[indices]
    win = np.tile(winfunc(frame_len), (numframes, 1))

    return frames * win


def magspec(frames: np.ndarray, nfft: int = 1024) -> np.ndarray:
    """Compute the magnitude spectrum of each frame in frames. If frames is an N x D matrix,
      output will be N x ((NFFT/2) + 1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an N x D matrix, output will be N x ((NFFT/2) + 1).
      Each row will be the magnitude spectrum of the corresponding frame.
    """
    if frames.shape[1] > nfft:
        print(
            "frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.",
            frames.shape[1],
            nfft,
        )
    complex_spec = np.fft.rfft(frames, nfft)
    return np.absolute(complex_spec)


def powspec(frames: np.ndarray, nfft: int = 1024) -> np.ndarray:
    """Compute the power spectrum density of each frame in frames. If frames is an N x D matrix,
      output will be N x ((NFFT/2) + 1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an N x D matrix, output will be N x ((NFFT/2) + 1).
      Each row will be the power spectrum density of the corresponding frame.
    """
    return (1.0 / nfft) * np.square(magspec(frames, nfft))


def get_filterbanks(
    nfilt: int = 26,
    nfft: int = 1024,
    samplerate: int = 44100,
    lowfreq: int = 0,
    highfreq: int = None,
) -> np.ndarray:
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * ((nfft/2) + 1)

    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 1024.
    :param samplerate: the samplerate of the signal we are working with, in Hz. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters. Default 0 Hz.
    :param highfreq: highest band edge of mel filters. Default samplerate/2.
    :returns: A numpy array of size nfilt * ((nfft/2) + 1) containing filterbank.
    Each row holds 1 filter.
    """
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)

    # our points are in Hz, but we use fft bins, so we have to convert from Hz to fft bin number
    # We don't have the frequency resolution required to put filters at the exact points calculated above,
    # so we need to round those frequencies to the nearest FFT bin
    bin = np.floor((nfft + 1) * mel2hz(melpoints) / samplerate)

    fbank = np.zeros([nfilt, nfft // 2 + 1])
    for j in range(0, nfilt):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
    return fbank


def fbank(
    signal: np.ndarray,
    samplerate: int = 44100,
    winlen: float = 0.02,
    winstep: float = 0.01, 
    nfilt: int = 26,
    nfft: int = 1024,
    lowfreq: int = 0,
    highfreq: int = None,
    preemph: float = 0.97,
    winfunc: Callable[[np.ndarray], np.ndarray] = lambda x: np.ones((x,)),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with (in Hz). Default is 44100.
    :param winlen: the length of the analysis window in seconds. Default is 0.02s (20 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 1024.
    :param lowfreq: lowest band edge of mel filters (in Hz). Default is 0.
    :param highfreq: highest band edge of mel filters (in Hz). Default is samplerate/2.
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
        second return value is the energy in each frame (total energy)
    """
    highfreq = highfreq or samplerate / 2
    signal = preemphasis(signal, preemph)
    frames = framesig(signal, winlen * samplerate, winstep * samplerate, winfunc)
    pspec = powspec(frames, nfft)
    energy = np.sum(pspec, 1)  # this stores the total energy in each frame
    mask_energy = np.abs(energy) < 1
    energy[mask_energy] = 1  # we do this to avoid problems with log later

    fb = get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)
    feat = np.dot(pspec, fb.T)  # compute the filterbank energies
    mask_feat = np.abs(feat) < 1
    feat[mask_feat] = 1  # we do this to avoid problems with log later

    return feat, energy


def lifter(cepstra: np.ndarray, liftering: int = 22) -> np.ndarray:
    """Apply a cepstral lifter to the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs
    
    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if liftering > 0:
        _, ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (liftering / 2.0) * np.sin(np.pi * n / liftering)
        return lift * cepstra
    else:
        return cepstra


def mfcc(
    signal: np.ndarray,
    samplerate: int = 44100,
    winlen: float = 0.025,
    winstep: float = 0.0125,
    nfilt: int = 40,
    nfft: int = 1024,
    numcep: int = 12,
    lowfreq: int = 0,
    highfreq: int = None, 
    preemph: float = 0.97,
    ceplifter: int = 0,
    append_energy: bool = True,
    winfunc: Callable[[np.ndarray], np.ndarray] = np.hamming,
) -> np.ndarray:
    """Compute MFCC features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with. 
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 40.
    :param nfft: the FFT size. Default is 1024.
    :param numcep: the number of cepstrum to return, default 12
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 0.
    :param append_energy: if this is true, the zeroth cepstral coefficient is replaced with the log of the
      total frame energy. Default is True.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
      You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """

    feat, energy = fbank(
        signal,
        samplerate,
        winlen,
        winstep,
        nfilt,
        nfft,
        lowfreq,
        highfreq,
        preemph,
        winfunc,
    )
    feat = np.log(feat)
    feat = dct(feat, type=2, axis=1, norm="ortho")[:, :numcep]
    feat = lifter(feat, ceplifter)

    if append_energy:
        feat[:, 0] = np.log(
            energy
        )  # replace first cepstral coefficient with log of frame energy
    
    feat = feat / np.linalg.norm(feat, axis = 1, keepdims = True)
    
    return feat


def delta(feat: np.ndarray, delta_n: int = 2) -> np.ndarray:
    """Compute delta features from a feature vector sequence.

    :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
    :param N: For each frame, calculate delta features based on preceding and following N frames
    :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
    """
    if delta_n < 1:
        raise ValueError("N must be an integer >= 1")

    # formula available here:
    # D:\Pasta Felipe\Faculdade\Periodos\TCC\VideoProcess\resources\correct_formula_delta.jpg
    num_frames = len(feat)
    denominator = 2 * sum([i * i for i in range(1, delta_n + 1)])
    delta_feat = np.empty_like(feat)

    # create delta_n rows before first row and after last row
    # for the first extra row(s), we use the first row as the reference
    # for the last extra row(s), we use the last row as the reference
    # e.g. for delta_n = 2, and feat = [[0, 1, 2, 3], [4, 5, 6, 7]] we have:
    # [[0, 1, 2, 3],
    # [0, 1, 2, 3],
    # [4, 5, 6, 7],
    # 4, 5, 6, 7]]
    padded = np.pad(feat, ((delta_n, delta_n), (0, 0)), mode="edge")

    # example for delta_n = 2, frame 1, first coeff:
    # c_1 = feat[2][0] - feat[0][0]
    # c_2 = 2 * (feat[3][0] - feat[0][0])
    # d_1_1 = (c_1 + c_2) / 10
    # we use padded, so the indexes are shifted by delta_n, but this is essentially what we do
    for t in range(num_frames):
        delta_feat[t] = (
            np.dot(np.arange(-delta_n, delta_n + 1), padded[t : t + 2 * delta_n + 1])
            / denominator
        )
    return delta_feat
