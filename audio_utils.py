# audio_utils.py

import numpy as np
import librosa
from scipy.signal import butter, filtfilt
import noisereduce as nr


# ---------------- Bandpass Filter ----------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')


def apply_bandpass_filter(data, sr, lowcut=1000, highcut=8000):
    b, a = butter_bandpass(lowcut, highcut, sr, order=4)
    filtered = filtfilt(b, a, data)
    return filtered


# ---------------- Noise Reduction ----------------
def apply_noise_reduction(audio, sr):
    noise_len = int(0.5 * sr)
    noise_sample = audio[:noise_len]

    reduced = nr.reduce_noise(
        y=audio,
        y_noise=noise_sample,
        sr=sr,
        prop_decrease=0.8,
    )
    return reduced


# ---------------- 统一封装：App 调用它即可 ----------------
def denoise_audio(audio, sr):
    """App 用的：带通滤波 → 降噪"""
    band = apply_bandpass_filter(audio, sr)
    denoised = apply_noise_reduction(band, sr)
    return denoised
