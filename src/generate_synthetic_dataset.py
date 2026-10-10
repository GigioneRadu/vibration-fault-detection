from pathlib import Path
import csv

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import kurtosis


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_DIR / "synthetic_bearing_dataset.csv"


# ---------------------------------------------------------
# BEARING GEOMETRY
# ---------------------------------------------------------

NUMBER_OF_ROLLING_ELEMENTS = 8
ROLLING_ELEMENT_DIAMETER_MM = 8.0
PITCH_DIAMETER_MM = 40.0
CONTACT_ANGLE_DEG = 0.0

CONTACT_ANGLE_RAD = np.deg2rad(
    CONTACT_ANGLE_DEG
)


# ---------------------------------------------------------
# SIGNAL SETTINGS
# ---------------------------------------------------------

SAMPLING_FREQUENCY = 5000.0
WINDOW_DURATION = 1.0

NUMBER_OF_SAMPLES = int(
    SAMPLING_FREQUENCY * WINDOW_DURATION
)

TIME = (
    np.arange(NUMBER_OF_SAMPLES)
    / SAMPLING_FREQUENCY
)


# ---------------------------------------------------------
# DATASET SIZE
# ---------------------------------------------------------

HEALTHY_SAMPLES = 200
OUTER_RACE_SAMPLES = 200


# ---------------------------------------------------------
# RANDOM GENERATOR
# ---------------------------------------------------------

rng = np.random.default_rng(seed=42)


# ---------------------------------------------------------
# BEARING FREQUENCY CALCULATION
# ---------------------------------------------------------

def calculate_fault_frequencies(rpm):
    shaft_frequency_hz = rpm / 60.0

    n = NUMBER_OF_ROLLING_ELEMENTS
    d = ROLLING_ELEMENT_DIAMETER_MM
    D = PITCH_DIAMETER_MM
    theta = CONTACT_ANGLE_RAD

    bpfo = (
        n / 2
        * shaft_frequency_hz
        * (
            1
            - (d / D) * np.cos(theta)
        )
    )

    bpfi = (
        n / 2
        * shaft_frequency_hz
        * (
            1
            + (d / D) * np.cos(theta)
        )
    )

    bsf = (
        D / (2 * d)
        * shaft_frequency_hz
        * (
            1
            - (
                (d / D)
                * np.cos(theta)
            ) ** 2
        )
    )

    ftf = (
        0.5
        * shaft_frequency_hz
        * (
            1
            - (d / D) * np.cos(theta)
        )
    )

    return (
        shaft_frequency_hz,
        bpfo,
        bpfi,
        bsf,
        ftf,
    )


# ---------------------------------------------------------
# SYNTHETIC SIGNAL GENERATION
# ---------------------------------------------------------

def generate_signal(label):
    """
    Generate one synthetic vibration window.

    label:
        "healthy"
        "outer_race"
    """

    # Slightly different operating speed for every sample.
    rpm = rng.uniform(
        1700.0,
        1900.0,
    )

    (
        shaft_frequency_hz,
        bpfo,
        bpfi,
        bsf,
        ftf,
    ) = calculate_fault_frequencies(rpm)

    # Normal rotational vibration.
    rotation_amplitude = rng.uniform(
        0.15,
        0.25,
    )

    rotation_phase = rng.uniform(
        0.0,
        2 * np.pi,
    )

    rotation_component = (
        rotation_amplitude
        * np.sin(
            2
            * np.pi
            * shaft_frequency_hz
            * TIME
            + rotation_phase
        )
    )

    # Measurement noise also varies between samples.
    noise_std = rng.uniform(
        0.02,
        0.08,
    )

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=len(TIME),
    )

    signal = (
        rotation_component
        + noise
    )

    # -----------------------------------------------------
    # ADD OUTER-RACE FAULT
    # -----------------------------------------------------

    if label == "outer_race":

        fault_signal = np.zeros_like(
            TIME
        )

        resonance_frequency_hz = rng.uniform(
            1000.0,
            1400.0,
        )

        decay_rate = rng.uniform(
            700.0,
            1100.0,
        )

        impact_amplitude = rng.uniform(
            0.35,
            1.20,
        )

        ringing_duration = 0.015

        ringing_samples = int(
            ringing_duration
            * SAMPLING_FREQUENCY
        )

        ringing_time = (
            np.arange(ringing_samples)
            / SAMPLING_FREQUENCY
        )

        ringing = (
            impact_amplitude
            * np.exp(
                -decay_rate
                * ringing_time
            )
            * np.sin(
                2
                * np.pi
                * resonance_frequency_hz
                * ringing_time
            )
        )

        fault_period = 1.0 / bpfo

        # Random starting phase prevents every sample
        # from beginning with an impact at t = 0.
        first_impact = rng.uniform(
            0.0,
            fault_period,
        )

        impact_times = np.arange(
            first_impact,
            WINDOW_DURATION,
            fault_period,
        )

        for impact_time in impact_times:

            start_index = int(
                impact_time
                * SAMPLING_FREQUENCY
            )

            end_index = min(
                start_index
                + ringing_samples,
                len(fault_signal),
            )

            valid_samples = (
                end_index - start_index
            )

            fault_signal[
                start_index:end_index
            ] += ringing[:valid_samples]

        signal = (
            signal
            + fault_signal
        )

    return {
        "signal": signal,
        "rpm": rpm,
        "shaft_frequency_hz": shaft_frequency_hz,
        "bpfo": bpfo,
        "bpfi": bpfi,
        "bsf": bsf,
        "ftf": ftf,
    }


# ---------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------

def extract_features(signal_data):
    signal = signal_data["signal"]

    rpm = signal_data["rpm"]

    shaft_frequency_hz = (
        signal_data["shaft_frequency_hz"]
    )

    bpfo = signal_data["bpfo"]
    bpfi = signal_data["bpfi"]
    bsf = signal_data["bsf"]
    ftf = signal_data["ftf"]

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
        d=1 / SAMPLING_FREQUENCY,
    )

    fft_amplitude = (
        2.0
        / len(centered_signal)
        * np.abs(fft_values)
    )

    def amplitude_near_frequency(
        target_frequency,
        search_width_hz=2.0,
    ):
        mask = (
            (
                frequencies
                >= target_frequency
                - search_width_hz
            )
            & (
                frequencies
                <= target_frequency
                + search_width_hz
            )
        )

        if not np.any(mask):
            return 0.0

        return np.max(
            fft_amplitude[mask]
        )

    shaft_amplitude = (
        amplitude_near_frequency(
            shaft_frequency_hz
        )
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
        fs=SAMPLING_FREQUENCY,
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

    envelope_frequencies = (
        np.fft.rfftfreq(
            len(envelope_centered),
            d=1 / SAMPLING_FREQUENCY,
        )
    )

    envelope_amplitude = (
        2.0
        / len(envelope_centered)
        * np.abs(envelope_fft)
    )

    def envelope_amplitude_near_frequency(
        target_frequency,
        search_width_hz=2.0,
    ):
        mask = (
            (
                envelope_frequencies
                >= target_frequency
                - search_width_hz
            )
            & (
                envelope_frequencies
                <= target_frequency
                + search_width_hz
            )
        )

        if not np.any(mask):
            return 0.0

        return np.max(
            envelope_amplitude[mask]
        )

    bpfo_envelope = (
        envelope_amplitude_near_frequency(
            bpfo
        )
    )

    bpfi_envelope = (
        envelope_amplitude_near_frequency(
            bpfi
        )
    )

    bsf_envelope = (
        envelope_amplitude_near_frequency(
            bsf
        )
    )

    ftf_envelope = (
        envelope_amplitude_near_frequency(
            ftf
        )
    )

    return {
        "rpm": rpm,
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
# GENERATE DATASET
# ---------------------------------------------------------

dataset = []


def generate_class(
    label,
    number_of_samples,
):
    print(
        f"Generating {number_of_samples} "
        f"samples for class: {label}"
    )

    for sample_number in range(
        number_of_samples
    ):

        signal_data = generate_signal(
            label
        )

        features = extract_features(
            signal_data
        )

        features["label"] = label

        dataset.append(features)

        if (
            (sample_number + 1) % 50
            == 0
        ):
            print(
                f"  {sample_number + 1}"
                f"/{number_of_samples}"
            )


generate_class(
    "healthy",
    HEALTHY_SAMPLES,
)

generate_class(
    "outer_race",
    OUTER_RACE_SAMPLES,
)


# ---------------------------------------------------------
# SHUFFLE DATASET
# ---------------------------------------------------------

rng.shuffle(dataset)


# ---------------------------------------------------------
# SAVE CSV
# ---------------------------------------------------------

fieldnames = list(
    dataset[0].keys()
)

with OUTPUT_FILE.open(
    mode="w",
    newline="",
    encoding="utf-8",
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        dataset
    )


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

healthy_count = sum(
    row["label"] == "healthy"
    for row in dataset
)

fault_count = sum(
    row["label"] == "outer_race"
    for row in dataset
)

print()
print("=" * 70)
print("SYNTHETIC DATASET CREATED")
print("=" * 70)

print(
    f"Total samples:      "
    f"{len(dataset)}"
)

print(
    f"Healthy samples:    "
    f"{healthy_count}"
)

print(
    f"Outer-race samples: "
    f"{fault_count}"
)

print()

print("Saved to:")
print(OUTPUT_FILE)

print("=" * 70)