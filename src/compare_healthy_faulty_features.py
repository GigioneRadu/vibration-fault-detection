import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import kurtosis


# ---------------------------------------------------------
# BEARING / SHAFT PARAMETERS
# ---------------------------------------------------------

rpm = 1800.0
shaft_frequency_hz = rpm / 60.0

n = 8
d = 8.0
D = 40.0
contact_angle_deg = 0.0

contact_angle_rad = np.deg2rad(contact_angle_deg)


# ---------------------------------------------------------
# BEARING FAULT FREQUENCIES
# ---------------------------------------------------------

bpfo = (
    n / 2
    * shaft_frequency_hz
    * (
        1
        - (d / D) * np.cos(contact_angle_rad)
    )
)

bpfi = (
    n / 2
    * shaft_frequency_hz
    * (
        1
        + (d / D) * np.cos(contact_angle_rad)
    )
)

bsf = (
    D / (2 * d)
    * shaft_frequency_hz
    * (
        1
        - (
            (d / D)
            * np.cos(contact_angle_rad)
        ) ** 2
    )
)

ftf = (
    0.5
    * shaft_frequency_hz
    * (
        1
        - (d / D) * np.cos(contact_angle_rad)
    )
)


# ---------------------------------------------------------
# SIGNAL SETTINGS
# ---------------------------------------------------------

sampling_frequency = 5000.0
duration = 5.0

number_of_samples = int(
    sampling_frequency * duration
)

time = (
    np.arange(number_of_samples)
    / sampling_frequency
)


# ---------------------------------------------------------
# COMMON ROTATIONAL COMPONENT
# ---------------------------------------------------------

rotation_component = (
    0.20
    * np.sin(
        2
        * np.pi
        * shaft_frequency_hz
        * time
    )
)


# ---------------------------------------------------------
# NOISE
# ---------------------------------------------------------

rng = np.random.default_rng(seed=42)

noise_healthy = rng.normal(
    loc=0.0,
    scale=0.04,
    size=len(time),
)

noise_faulty = rng.normal(
    loc=0.0,
    scale=0.04,
    size=len(time),
)


# ---------------------------------------------------------
# HEALTHY SIGNAL
# ---------------------------------------------------------

healthy_signal = (
    rotation_component
    + noise_healthy
)


# ---------------------------------------------------------
# OUTER-RACE FAULT SIGNAL
# ---------------------------------------------------------

fault_signal = np.zeros_like(time)

fault_period = 1.0 / bpfo

impact_times = np.arange(
    0.0,
    duration,
    fault_period,
)

resonance_frequency_hz = 1200.0
decay_rate = 900.0
impact_amplitude = 1.0

ringing_duration = 0.015

ringing_samples = int(
    ringing_duration
    * sampling_frequency
)

ringing_time = (
    np.arange(ringing_samples)
    / sampling_frequency
)

ringing = (
    impact_amplitude
    * np.exp(
        -decay_rate * ringing_time
    )
    * np.sin(
        2
        * np.pi
        * resonance_frequency_hz
        * ringing_time
    )
)

for impact_time in impact_times:

    start_index = int(
        impact_time * sampling_frequency
    )

    end_index = min(
        start_index + ringing_samples,
        len(fault_signal),
    )

    valid_samples = (
        end_index - start_index
    )

    fault_signal[
        start_index:end_index
    ] += ringing[:valid_samples]


faulty_signal = (
    rotation_component
    + fault_signal
    + noise_faulty
)


# ---------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------

def extract_features(signal):
    """
    Extract time-domain, raw FFT, and envelope features.
    """

    # -----------------------------------------------------
    # TIME-DOMAIN FEATURES
    # -----------------------------------------------------

    rms = np.sqrt(
        np.mean(signal ** 2)
    )

    peak = np.max(
        np.abs(signal)
    )

    peak_to_peak = (
        np.max(signal)
        - np.min(signal)
    )

    crest_factor = (
        peak / rms
        if rms != 0
        else 0.0
    )

    signal_kurtosis = kurtosis(
        signal,
        fisher=False,
    )

    # -----------------------------------------------------
    # RAW FFT
    # -----------------------------------------------------

    centered_signal = (
        signal
        - np.mean(signal)
    )

    fft_values = np.fft.rfft(
        centered_signal
    )

    frequencies = np.fft.rfftfreq(
        len(centered_signal),
        d=1 / sampling_frequency,
    )

    fft_amplitude = (
        2.0
        / len(centered_signal)
        * np.abs(fft_values)
    )

    def amplitude_near_frequency(
        target_frequency,
        search_width_hz=1.0,
    ):
        mask = (
            (
                frequencies
                >= target_frequency - search_width_hz
            )
            & (
                frequencies
                <= target_frequency + search_width_hz
            )
        )

        if not np.any(mask):
            return 0.0

        return np.max(
            fft_amplitude[mask]
        )

    shaft_amplitude = amplitude_near_frequency(
        shaft_frequency_hz
    )

    bpfo_raw = amplitude_near_frequency(
        bpfo
    )

    bpfi_raw = amplitude_near_frequency(
        bpfi
    )

    bsf_raw = amplitude_near_frequency(
        bsf
    )

    ftf_raw = amplitude_near_frequency(
        ftf
    )

    # -----------------------------------------------------
    # ENVELOPE ANALYSIS
    # -----------------------------------------------------

    low_cut_hz = 800.0
    high_cut_hz = 1600.0

    sos = butter(
        N=4,
        Wn=[
            low_cut_hz,
            high_cut_hz,
        ],
        btype="bandpass",
        fs=sampling_frequency,
        output="sos",
    )

    filtered_signal = sosfiltfilt(
        sos,
        signal,
    )

    analytic_signal = hilbert(
        filtered_signal
    )

    envelope = np.abs(
        analytic_signal
    )

    envelope_centered = (
        envelope
        - np.mean(envelope)
    )

    envelope_fft = np.fft.rfft(
        envelope_centered
    )

    envelope_frequencies = np.fft.rfftfreq(
        len(envelope_centered),
        d=1 / sampling_frequency,
    )

    envelope_amplitude = (
        2.0
        / len(envelope_centered)
        * np.abs(envelope_fft)
    )

    def envelope_amplitude_near_frequency(
        target_frequency,
        search_width_hz=1.0,
    ):
        mask = (
            (
                envelope_frequencies
                >= target_frequency - search_width_hz
            )
            & (
                envelope_frequencies
                <= target_frequency + search_width_hz
            )
        )

        if not np.any(mask):
            return 0.0

        return np.max(
            envelope_amplitude[mask]
        )

    bpfo_envelope = envelope_amplitude_near_frequency(
        bpfo
    )

    bpfi_envelope = envelope_amplitude_near_frequency(
        bpfi
    )

    bsf_envelope = envelope_amplitude_near_frequency(
        bsf
    )

    ftf_envelope = envelope_amplitude_near_frequency(
        ftf
    )

    return {
        "rms": rms,
        "peak": peak,
        "peak_to_peak": peak_to_peak,
        "crest_factor": crest_factor,
        "kurtosis": signal_kurtosis,
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


# ---------------------------------------------------------
# EXTRACT BOTH STATES
# ---------------------------------------------------------

healthy_features = extract_features(
    healthy_signal
)

faulty_features = extract_features(
    faulty_signal
)


# ---------------------------------------------------------
# PRINT COMPARISON
# ---------------------------------------------------------

print("=" * 90)
print("HEALTHY VS OUTER-RACE FAULT FEATURE COMPARISON")
print("=" * 90)

print(
    f"{'Feature':<22}"
    f"{'Healthy':>16}"
    f"{'Outer Race Fault':>20}"
    f"{'Ratio Fault/Healthy':>24}"
)

print("-" * 90)

for feature_name in healthy_features:

    healthy_value = healthy_features[
        feature_name
    ]

    faulty_value = faulty_features[
        feature_name
    ]

    if healthy_value != 0:
        ratio = (
            faulty_value
            / healthy_value
        )
    else:
        ratio = np.nan

    print(
        f"{feature_name:<22}"
        f"{healthy_value:>16.6f}"
        f"{faulty_value:>20.6f}"
        f"{ratio:>24.2f}"
    )

print("=" * 90)


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print()
print("EXPECTED INTERPRETATION")
print("-" * 90)

print(
    "A strong increase in BPFO envelope amplitude "
    "is expected for the outer-race fault."
)

print(
    "Kurtosis and crest factor are also expected "
    "to increase because the fault introduces impulses."
)