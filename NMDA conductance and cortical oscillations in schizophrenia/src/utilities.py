import numpy as np
from src.model import network
from scipy.ndimage import convolve1d


# ----------------------------
# Spectral analysis utilities
# ----------------------------

def gamma_metrics_from_lfp(
    lfp, dt_ms, f_lo=20.0, f_hi=80.0, half_width_hz=3.0,
    bin_s=1.0, window="hann", detrend="mean", summary="mean",
    return_diagnostics=False, psd_normalize=False, conv_mode="reflect"
):
    """
    - Split into 1-s bins
    - FFT per bin -> power spectrum P_k(f)
    - For each bin, compute a local ±half_width_hz mean in frequency: S_k(f)
    - Pick f0,k as argmax of S_k(f) within [f_lo, f_hi]
    - Define P0,k as S_k(f0,k) (i.e., mean power in ±half_width_hz around the chosen peak)
    - Aggregate across bins (mean or median)
    """
    fs = 1.0 / (dt_ms * 1e-3)
    n = int(round(fs * bin_s))
    K = len(lfp) // n
    if K < 1:
        raise ValueError(f"LFP shorter than {bin_s} s; cannot compute spectra.")

    x = np.asarray(lfp[:K * n]).reshape(K, n)

    if detrend == "mean":
        x = x - x.mean(axis=1, keepdims=True)
    elif detrend is not None:
        raise ValueError("detrend must be 'mean' or None")

    if window == "hann":
        w = np.hanning(n)
        x = x * w
        U = (w**2).sum()
    elif window is None:
        # rectangular window: sum(w^2) = n
        U = float(n)
    else:
        raise ValueError("window must be 'hann' or None")

    X = np.fft.rfft(x, axis=1)
    P = np.abs(X) ** 2

    if psd_normalize:
        P = P / (fs * U)

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    gamma = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(gamma):
        out = (np.nan, np.nan)
        return (*out, {}) if return_diagnostics else out

    df = freqs[1] - freqs[0] if len(freqs) > 1 else np.nan
    hw_bins = int(round(half_width_hz / df)) if np.isfinite(df) else 0
    hw_bins = max(hw_bins, 0)
    # Optional: ensure smoothing if half_width_hz > 0
    if half_width_hz > 0:
        hw_bins = max(hw_bins, 1)

    kernel = np.ones(2 * hw_bins + 1, dtype=float)
    kernel /= kernel.sum()

    # Smooth along frequency axis (per bin)
    S = convolve1d(P, kernel, axis=1, mode=conv_mode)

    # Peak pick based on locally averaged amplitude/power within gamma
    Sg = S[:, gamma]
    peak_idx = np.argmax(Sg, axis=1)
    f0_bins = freqs[gamma][peak_idx]
    P0_bins = Sg[np.arange(K), peak_idx]

    if summary == "mean":
        f0 = float(np.nanmean(f0_bins))
        P0 = float(np.nanmean(P0_bins))
    elif summary == "median":
        f0 = float(np.nanmedian(f0_bins))
        P0 = float(np.nanmedian(P0_bins))
    else:
        raise ValueError("summary must be 'mean' or 'median'")

    if not return_diagnostics:
        return f0, P0

    diagnostics = dict(
        f0_bins=f0_bins,
        P0_bins=P0_bins,
        f0_std=float(np.nanstd(f0_bins)),
        f0_iqr=float(np.nanpercentile(f0_bins, 75) - np.nanpercentile(f0_bins, 25)),
        df_hz=float(df),
        half_width_bins=int(hw_bins),
        n_bins=int(K),
        psd_normalize=bool(psd_normalize),
        window=window,
        detrend=detrend,
    )
    return f0, P0, diagnostics






# ---- spectral metric for Fig 5 ----
def power_at_frequency_band(lfp, dt_ms, target_freq_hz, half_width_hz=3.0, bin_s=1.0):
    fs = 1.0 / (dt_ms * 1e-3)
    n = int(round(fs * bin_s))
    K = len(lfp) // n
    if K < 1:
        raise ValueError("LFP too short for spectral binning.")

    x = np.asarray(lfp[:K*n]).reshape(K, n)
    x = x - x.mean(axis=1, keepdims=True)
    x = x * np.hanning(n)

    X = np.fft.rfft(x, axis=1)
    P = np.abs(X)**2
    freqs = np.fft.rfftfreq(n, d=1.0/fs)

    band = (freqs >= target_freq_hz - half_width_hz) & (freqs <= target_freq_hz + half_width_hz)
    return float(P[:, band].mean())

# ---- run_one_point (returns Pdrive in periodic mode) ----
def run_one_point(
    gNI, gNE, gEI, IappI,
    dt_ms, T_ms, rng_seed, alpha_n_per_ms,
    IappE=4.0,
    drive_freq_hz=0.0,
    return_spikes=False,
    normalize=False
):
    res = network(
        dt_ms=float(dt_ms),
        T_ms=float(T_ms),
        rng_seed=int(rng_seed),
        gNI_mS_cm2=float(gNI),
        gNE_mS_cm2=float(gNE),
        gEI_mS_cm2=float(gEI),
        Iapp_I_uAcm2=float(IappI),
        Iapp_E_uAcm2=float(IappE),
        alpha_n_per_ms=float(alpha_n_per_ms),
        normalize=bool(normalize),
        drive_freq_hz=float(drive_freq_hz),
    )

    lfp = res["lfp"]
    dt  = float(res["params"]["dt_ms"])

    # Always return the parameter point you simulated
    out = {
        "gNI": float(gNI),
        "gNE": float(gNE),
        "gEI": float(gEI),
        "IappI": float(IappI),
        "IappE": float(IappE),
        "dt_ms": float(dt_ms),
        "T_ms": float(T_ms),
        "rng_seed": int(rng_seed),
        "alpha_n_per_ms": float(alpha_n_per_ms),
        "normalize": bool(normalize),
        "drive_freq_hz": float(drive_freq_hz),
        "lfp": lfp,
    }

    if drive_freq_hz > 0.0:
        out["Pdrive"] = float(
            power_at_frequency_band(
                lfp, dt, drive_freq_hz,
                half_width_hz=3.0, bin_s=1.0
            )
        )
    else:
        f0, P0 = gamma_metrics_from_lfp(
            lfp, dt,
            f_lo=20.0, f_hi=80.0, half_width_hz=3.0,
            bin_s=1.0, window="hann", detrend="mean",
            summary="mean", psd_normalize=False,
            conv_mode="nearest"
        )
        out["f0"] = float(f0)
        out["P0"] = float(P0)

    if return_spikes:
        out.update({
            "spE_t": res["spE_t"], "spE_i": res["spE_i"],
            "spI_t": res["spI_t"], "spI_i": res["spI_i"],
        })

    return out

        
    


