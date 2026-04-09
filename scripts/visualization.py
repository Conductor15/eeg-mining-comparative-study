import mne
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch


def visualize_eeg(file_path, channel='EEG C3-M2', epoch_duration=5.0):
    """
    Visualize EEG signal including:
    - Time domain (raw vs filtered)
    - Power Spectral Density (PSD)
    - Band power comparison

    Parameters
    ----------
    file_path : str
        Path to EDF file.
    channel : str
        EEG channel name.
    epoch_duration : float
        Duration (seconds) for time-domain plotting.
    """

    # Load data
    raw = mne.io.read_raw_edf(file_path, preload=True)
    sfreq = raw.info['sfreq']

    # Get raw signal
    raw_data = raw.get_data(picks=channel)[0] * 1e6

    # Create filtered version (for comparison)
    filtered_raw = raw.copy()
    filtered_raw.filter(0.5, 30, picks=channel)

    filtered_data = filtered_raw.get_data(picks=channel)[0] * 1e6

    # Time vector
    n = int(epoch_duration * sfreq)
    times = np.arange(n) / sfreq

    # PSD
    nperseg = int(2 * sfreq)
    freqs_raw, psd_raw = welch(raw_data, fs=sfreq, nperseg=nperseg)
    freqs_filt, psd_filt = welch(filtered_data, fs=sfreq, nperseg=nperseg)

    # Band definition
    bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma": (30, 80)
    }

    def band_power(freqs, psd, band):
        idx = (freqs >= band[0]) & (freqs <= band[1])
        return np.trapz(psd[idx], freqs[idx])

    bp_raw = [band_power(freqs_raw, psd_raw, b) for b in bands.values()]
    bp_filt = [band_power(freqs_filt, psd_filt, b) for b in bands.values()]

    # ===== PLOT =====
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # --- Time domain ---
    axes[0].plot(times, raw_data[:n], label="Raw", alpha=0.6)
    axes[0].plot(times, filtered_data[:n], label="Filtered", linewidth=2)
    axes[0].set_title("EEG Signal (Time Domain)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude (µV)")
    axes[0].legend()
    axes[0].grid()

    # --- PSD ---
    axes[1].semilogy(freqs_raw, psd_raw, label="Raw")
    axes[1].semilogy(freqs_filt, psd_filt, label="Filtered")
    axes[1].set_title("Power Spectral Density (PSD)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power (µV²/Hz)")
    axes[1].set_xlim(0, 50)
    axes[1].legend()
    axes[1].grid()

    # --- Band power ---
    x = np.arange(len(bands))
    width = 0.35

    axes[2].bar(x - width/2, bp_raw, width, label="Raw")
    axes[2].bar(x + width/2, bp_filt, width, label="Filtered")

    axes[2].set_title("Power Distribution Across Frequency Bands")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(list(bands.keys()))
    axes[2].set_ylabel("Power (µV²)")
    axes[2].set_yscale("log")
    axes[2].legend()
    axes[2].grid()

    plt.tight_layout()
    plt.show()
    
    
visualize_eeg("data/raw/10003_26038.edf")