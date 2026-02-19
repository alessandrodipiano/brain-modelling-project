import numpy as np
from src.model import network


# ----------------------------
# Spectral analysis utilities
# ----------------------------
def compute_mean_power(lfp, dt_ms):
    """
    Compute mean (magnitude-squared) spectrum by splitting LFP into 1-second bins,
    mean-removing each bin, applying Hann window, FFT, then averaging power spectra.
    Returns (freqs, mean_power).
    """
    fs = 1.0 / (dt_ms * 1e-3)          # Hz
    n = int(round(fs))                # samples in ~1 second
    K = len(lfp) // n
    if K < 1:
        raise ValueError("LFP shorter than 1 second; cannot compute 1 s spectra.")

    x = np.asarray(lfp[:K * n]).reshape(K, n)
    x = x - x.mean(axis=1, keepdims=True)

    w = np.hanning(n)
    X = np.fft.rfft(x * w, axis=1)
    P = (np.abs(X) ** 2)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mean_power = P.mean(axis=0)
    return freqs, mean_power


def gamma_metrics_from_lfp(lfp, dt_ms, f_lo=20.0, f_hi=80.0, half_width=3.0):
    """
    Extract gamma peak frequency f0 (within [f_lo, f_hi]) and mean power in a
    +/- half_width band around the peak, restricted to gamma band.
    """
    freqs, mean_power = compute_mean_power(lfp, dt_ms)

    gamma = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(gamma):
        return np.nan, np.nan

    idx = np.argmax(mean_power[gamma])
    f0 = freqs[gamma][idx]

    band = gamma & (freqs >= f0 - half_width) & (freqs <= f0 + half_width)
    P0 = np.mean(mean_power[band]) if np.any(band) else np.nan
    return float(f0), float(P0)


# ----------------------------
# Parallel grid evaluation
# ----------------------------
def run_one_point(gg, IappI, dt_ms, T_ms, rng_seed, alpha_n_per_ms):
    """
    One simulation + gamma metrics. Returns (gg, IappI, f0, P0).
    """
    res = network(
        dt_ms=dt_ms,
        T_ms=T_ms,
        rng_seed=rng_seed,
        gNI_mS_cm2=float(gg),
        Iapp_I_uAcm2=float(IappI),
        alpha_n_per_ms=alpha_n_per_ms
    )

    lfp = res["lfp"]
    dt = res["params"]["dt_ms"]  # should equal dt_ms

    f0, P0 = gamma_metrics_from_lfp(lfp, dt)
    return float(gg), float(IappI), f0, P0, lfp