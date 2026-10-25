
from pathlib import Path
import sys

import joblib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import kurtosis, skew


BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "alzheimer_rf_binary.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

saved = joblib.load(MODEL_PATH)
model = saved["model"]
FEATURES = saved["features"]
LABELS = saved["label_mapping"]


def filter_bank_function(signal, sr):
    """Reproduce the notebook's 26-filter log filter-bank calculation."""
    signal = np.asarray(signal, dtype=float)
    if len(signal) == 0:
        raise ValueError("Audio is empty.")

    emphasized = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
    frame_length = max(1, int(round(0.025 * sr)))
    frame_step = max(1, int(round(0.010 * sr)))
    nfft = 512

    if len(emphasized) < frame_length:
        emphasized = np.pad(
            emphasized, (0, frame_length - len(emphasized))
        )

    n_frames = 1 + int(np.ceil(
        max(0, len(emphasized) - frame_length) / frame_step
    ))
    padded_length = (n_frames - 1) * frame_step + frame_length
    emphasized = np.pad(
        emphasized, (0, max(0, padded_length - len(emphasized)))
    )

    indices = (
        np.arange(frame_length)[None, :]
        + np.arange(n_frames)[:, None] * frame_step
    )
    frames = emphasized[indices] * np.hamming(frame_length)
    magnitude = np.abs(np.fft.rfft(frames, n=nfft, axis=1))
    power = magnitude ** 2 / nfft

    nfilt = 26
    mel_max = 2595 * np.log10(1 + (sr / 2) / 700)
    mel_points = np.linspace(0, mel_max, nfilt + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bins = np.floor((nfft + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, nfft // 2)

    bank = np.zeros((nfilt, nfft // 2 + 1))
    for m in range(1, nfilt + 1):
        left, center, right = bins[m - 1:m + 2]
        if center > left:
            for k in range(left, center):
                bank[m - 1, k] = (k - left) / (center - left)
        if right > center:
            for k in range(center, right):
                bank[m - 1, k] = (right - k) / (right - center)

    energies = power @ bank.T
    energies = 20 * np.log10(np.maximum(energies, np.finfo(float).eps))
    energies -= np.mean(energies, axis=0, keepdims=True) + 1e-8

    # Return filters x frames, matching the notebook's feature indexing.
    return energies.T


def approximate_voice_features(y, sr):
    """
    Approximate jitter, shimmer and HNR using librosa.
    These are NOT identical to the original surfboard definitions.
    """
    frame_length = max(256, int(0.040 * sr))
    hop = max(64, int(0.010 * sr))

    rms = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop
    )[0]
    rms = rms[rms > 1e-8]
    shimmer = (
        float(np.mean(np.abs(np.diff(rms))) / np.mean(rms))
        if len(rms) > 1 else 0.0
    )

    f0 = librosa.yin(y, fmin=60, fmax=min(400, sr / 2 - 1), sr=sr)
    f0 = f0[np.isfinite(f0) & (f0 > 0)]
    periods = 1.0 / f0
    jitter = (
        float(np.mean(np.abs(np.diff(periods))) / np.mean(periods))
        if len(periods) > 1 else 0.0
    )

    harmonic = librosa.effects.harmonic(y)
    residual = y[:len(harmonic)] - harmonic
    harmonic_power = float(np.mean(harmonic ** 2))
    residual_power = float(np.mean(residual ** 2))
    hnr = 10 * np.log10(
        (harmonic_power + 1e-12) / (residual_power + 1e-12)
    )

    return jitter, shimmer, float(hnr)


def extract_features(wav_path):
    y, sr = librosa.load(wav_path, sr=None, mono=True)

    if len(y) < int(sr * 1.0):
        raise ValueError("Please provide a recording at least 1 second long.")

    # Remove DC offset and trim leading/trailing silence.
    y = y - np.mean(y)
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    if len(y_trimmed) >= int(sr * 1.0):
        y = y_trimmed

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_fft=2048, hop_length=512, n_mfcc=42
    )
    filter_banks = filter_bank_function(y, sr)

    # Approximate silence statistics using 10 ms frames.
    frame_length = max(256, int(0.025 * sr))
    hop = max(64, int(0.010 * sr))
    rms = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop
    )[0]
    threshold = max(float(np.max(rms)) * 0.03, 1e-6)
    silent = rms <= threshold
    silence_percentage = float(np.mean(silent) * 100)

    # Count continuous silent runs and find the shortest run duration.
    changes = np.diff(np.r_[False, silent, False].astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    durations = (ends - starts) * hop / sr
    valid_durations = durations[durations >= 0.10]
    n_sil_segments = int(len(valid_durations))
    min_sil_len = (
        float(np.min(valid_durations)) if len(valid_durations) else 0.0
    )

    jitter, shimmer, hnr = approximate_voice_features(y, sr)

    # Use named features so column order matches the trained model.
    values = {
        "MFCC2": float(np.max(mfcc[1])),
        "kurt_MFCC30": float(kurtosis(mfcc[29])),
        "mean_MFCC30": float(np.mean(mfcc[29])),
        "skew_MFCC2": float(skew(mfcc[1])),
        "mean_MFCC16": float(np.mean(mfcc[15])),
        "flt_bnk_eng22": float(np.sum(filter_banks[21])),
        "MFCC30": float(np.max(mfcc[29])),
        "kurt_MFCC16": float(kurtosis(mfcc[15])),
        "flt_bnk_eng2": float(np.sum(filter_banks[1])),
        "flt_bnk_eng24": float(np.sum(filter_banks[23])),
        "MFCC1": float(np.max(mfcc[0])),
        "flt_bnk_eng15": float(np.sum(filter_banks[14])),
        "kurt_MFCC2": float(kurtosis(mfcc[1])),
        "flt_bnk_eng20": float(np.sum(filter_banks[19])),
        "flt_bnk_eng13": float(np.sum(filter_banks[12])),
        "n_sil_segments": n_sil_segments,
        "frac_silence": silence_percentage,
        "min_sil_len": min_sil_len,
        "jitter": jitter,
        "shimmer": shimmer,
        "HNR": hnr,
    }

    return values


def predict_audio(wav_path):
    features = extract_features(wav_path)
    row = pd.DataFrame([[features[name] for name in FEATURES]],
                       columns=FEATURES)
    prediction = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    classes = model.named_steps["classifier"].classes_

    return {
        "prediction": LABELS[prediction],
        "probabilities": {
            LABELS[int(c)]: round(float(p), 4)
            for c, p in zip(classes, probabilities)
        },
        "notice": "Experimental research output; not a medical diagnosis."
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python alzheimer_audio.py "path\\to\\recording.wav"')
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        sys.exit(1)

    try:
        result = predict_audio(str(audio_path))
        print("\n=== AUDIO PREDICTION ===")
        print("Prediction:", result["prediction"])
        print("Probabilities:", result["probabilities"])
        print(result["notice"])
    except Exception as exc:
        print(f"Could not process audio: {exc}")
        sys.exit(1)
