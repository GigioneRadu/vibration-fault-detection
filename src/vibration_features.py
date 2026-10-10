import numpy as np

from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import kurtosis


# ============================================================
# BEARING GEOMETRY
# ============================================================

NUMBER_OF_ROLLING_ELEMENTS = 8

BALL_DIAMETER_MM = 8.0
PITCH_DIAMETER_MM = 40.0

CONTACT_ANGLE_DEG = 0.0


# ============================================================
# FEATURE ORDER
# ============================================================

FEATURE_COLUMNS = [
    "rms",
    "peak",
    "peak_to_peak",
    "crest_factor",
    "kurtosis",
    "shaft_amplitude",
    "bpfo_raw",
    "bpfi_raw",
    "bsf_raw",
    "ftf_raw",
    "bpfo_envelope",
    "bpfi_envelope",
    "bsf_envelope",
    "ftf_envelope",
]


# ============================================================
# BEARING FAULT FREQUENCIES
# ============================================================

def calculate_bearing_frequencies(rpm):
    """
    Calculate shaft and characteristic bearing fault frequencies.

    Parameters
    ----------
    rpm : float
        Shaft rotational speed in revolutions per minute.

    Returns
    -------
    dict
        Dictionary containing:
        shaft, bpfo, bpfi, bsf, ftf
    """

    shaft_frequency = rpm / 60.0

    contact_angle_rad = np.deg2rad(
        CONTACT_ANGLE_DEG
    )

    ratio = (
        BALL_DIAMETER_MM
        / PITCH_DIAMETER_MM
    )

    cos_angle = np.cos(
        contact_angle_rad
    )

    bpfo = (
        NUMBER_OF_ROLLING_ELEMENTS
        / 2.0
        * shaft_frequency
        * (
            1.0
            - ratio * cos_angle
        )
    )

    bpfi = (
        NUMBER_OF_ROLLING_ELEMENTS
        / 2.0
        * shaft_frequency
        * (
            1.0
            + ratio * cos_angle
        )
    )

    bsf = (
        PITCH_DIAMETER_MM
        / (
            2.0
            * BALL_DIAMETER_MM
        )
        * shaft_frequency
        * (
            1.0
            - (
                ratio
                * cos_angle
            ) ** 2
        )
    )

    ftf = (
        shaft_frequency
        / 2.0
        * (
            1.0
            - ratio * cos_angle
        )
    )

    return {
        "shaft": shaft_frequency,
        "bpfo": bpfo,
        "bpfi": bpfi,
        "bsf": bsf,
        "ftf": ftf,
    }


# ============================================================
# FFT AMPLITUDE
# ============================================================

def get_fft_amplitude(
    signal,
    sampling_frequency,
    target_frequency,
):
    """
    Return the FFT amplitude nearest to a target frequency.
    """

    n = len(signal)

    fft_values = np.fft.rfft(
        signal
    )

    frequencies = np.fft.rfftfreq(
        n,
        d=1.0 / sampling_frequency,
    )

    amplitudes = (
        2.0
        / n
        * np.abs(
            fft_values
        )
    )

    nearest_index = np.argmin(
        np.abs(
            frequencies
            - target_frequency
        )
    )

    return amplitudes[
        nearest_index
    ]


# ============================================================
# ENVELOPE ANALYSIS
# ============================================================

def calculate_envelope(
    signal,
    sampling_frequency,
):
    """
    Band-pass filter the vibration signal and compute its
    Hilbert envelope.
    """

    low_frequency = 800.0
    high_frequency = 1600.0

    nyquist_frequency = (
        sampling_frequency
        / 2.0
    )

    if high_frequency >= nyquist_frequency:
        raise ValueError(
            "Sampling frequency is too low for "
            "the 800-1600 Hz envelope band."
        )

    low_normalized = (
        low_frequency
        / nyquist_frequency
    )

    high_normalized = (
        high_frequency
        / nyquist_frequency
    )

    b, a = butter(
        4,
        [
            low_normalized,
            high_normalized,
        ],
        btype="bandpass",
    )

    filtered_signal = filtfilt(
        b,
        a,
        signal,
    )

    analytic_signal = hilbert(
        filtered_signal
    )

    envelope = np.abs(
        analytic_signal
    )

    # Remove DC component before FFT.
    envelope = (
        envelope
        - np.mean(envelope)
    )

    return envelope


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_vibration_features(
    signal,
    rpm,
    sampling_frequency,
):
    """
    Extract the 14 features expected by the trained
    Random Forest model.
    """

    signal = np.asarray(
        signal,
        dtype=float,
    )

    if signal.ndim != 1:
        raise ValueError(
            "Signal must be one-dimensional."
        )

    if len(signal) == 0:
        raise ValueError(
            "Signal is empty."
        )

    if rpm <= 0:
        raise ValueError(
            "RPM must be greater than zero."
        )

    if sampling_frequency <= 0:
        raise ValueError(
            "Sampling frequency must be greater than zero."
        )

    bearing_frequencies = (
        calculate_bearing_frequencies(
            rpm
        )
    )


    # --------------------------------------------------------
    # TIME-DOMAIN FEATURES
    # --------------------------------------------------------

    rms_value = np.sqrt(
        np.mean(
            signal ** 2
        )
    )

    peak_value = np.max(
        np.abs(signal)
    )

    peak_to_peak_value = np.ptp(
        signal
    )

    if rms_value > 0:
        crest_factor_value = (
            peak_value
            / rms_value
        )
    else:
        crest_factor_value = 0.0

    kurtosis_value = kurtosis(
        signal,
        fisher=False,
    )


    # --------------------------------------------------------
    # RAW FFT FEATURES
    # --------------------------------------------------------

    shaft_amplitude = get_fft_amplitude(
        signal,
        sampling_frequency,
        bearing_frequencies[
            "shaft"
        ],
    )

    bpfo_raw = get_fft_amplitude(
        signal,
        sampling_frequency,
        bearing_frequencies[
            "bpfo"
        ],
    )

    bpfi_raw = get_fft_amplitude(
        signal,
        sampling_frequency,
        bearing_frequencies[
            "bpfi"
        ],
    )

    bsf_raw = get_fft_amplitude(
        signal,
        sampling_frequency,
        bearing_frequencies[
            "bsf"
        ],
    )

    ftf_raw = get_fft_amplitude(
        signal,
        sampling_frequency,
        bearing_frequencies[
            "ftf"
        ],
    )


    # --------------------------------------------------------
    # ENVELOPE FEATURES
    # --------------------------------------------------------

    envelope = calculate_envelope(
        signal,
        sampling_frequency,
    )

    bpfo_envelope = get_fft_amplitude(
        envelope,
        sampling_frequency,
        bearing_frequencies[
            "bpfo"
        ],
    )

    bpfi_envelope = get_fft_amplitude(
        envelope,
        sampling_frequency,
        bearing_frequencies[
            "bpfi"
        ],
    )

    bsf_envelope = get_fft_amplitude(
        envelope,
        sampling_frequency,
        bearing_frequencies[
            "bsf"
        ],
    )

    ftf_envelope = get_fft_amplitude(
        envelope,
        sampling_frequency,
        bearing_frequencies[
            "ftf"
        ],
    )


    # --------------------------------------------------------
    # FINAL FEATURE DICTIONARY
    # --------------------------------------------------------

    return {
        "rms": rms_value,
        "peak": peak_value,
        "peak_to_peak": peak_to_peak_value,
        "crest_factor": crest_factor_value,
        "kurtosis": kurtosis_value,
        "shaft_amplitude": shaft_amplitude,
        "bpfo_raw": bpfo_raw,
        "bpfi_raw": bpfi_raw,
        "bsf_raw": bsf_raw,
        "ftf_raw": ftf_raw,
        "bpfo_envelope": bpfo_envelope,
        "bpfi_envelope": bpfi_envelope,
        "bsf_envelope": bsf_envelope,
        "ftf_envelope": ftf_envelope,
    }