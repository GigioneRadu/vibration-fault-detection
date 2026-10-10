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

OUTPUT_FILE = (
    DATA_DIR
    / "synthetic_bearing_multiclass_v2.csv"
)


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
    SAMPLING_FREQUENCY
    * WINDOW_DURATION
)

TIME = (
    np.arange(NUMBER_OF_SAMPLES)
    / SAMPLING_FREQUENCY
)


# ---------------------------------------------------------
# DATASET SIZE
# ---------------------------------------------------------

SAMPLES_PER_CLASS = 300

CLASSES = [
    "healthy",
    "outer_race",
    "inner_race",
    "ball_fault",
]


# ---------------------------------------------------------
# RANDOM GENERATOR
# ---------------------------------------------------------

rng = np.random.default_rng(seed=42)


# ---------------------------------------------------------
# BEARING FAULT FREQUENCIES
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
# RINGING RESPONSE
# ---------------------------------------------------------

def create_ringing(
    resonance_frequency_hz,
    decay_rate,
    impact_amplitude,
):
    ringing_duration = 0.015

    ringing_samples = int(
        ringing_duration
        * SAMPLING_FREQUENCY
    )

    ringing_time = (
        np.arange(ringing_samples)
        / SAMPLING_FREQUENCY
    )

    return (
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


# ---------------------------------------------------------
# ADD IMPACT
# ---------------------------------------------------------

def add_impact(
    signal,
    impact_time,
    ringing,
):
    start_index = int(
        impact_time
        * SAMPLING_FREQUENCY
    )

    if start_index >= len(signal):
        return

    end_index = min(
        start_index + len(ringing),
        len(signal),
    )

    valid_samples = (
        end_index - start_index
    )

    signal[
        start_index:end_index
    ] += ringing[:valid_samples]


# ---------------------------------------------------------
# GENERATE FAULT IMPACT TRAIN
# ---------------------------------------------------------

def generate_fault_impacts(
    fault_frequency_hz,
    shaft_frequency_hz,
    label,
):
    fault_signal = np.zeros_like(
        TIME
    )

    base_resonance = rng.uniform(
        950.0,
        1500.0,
    )

    base_decay = rng.uniform(
        650.0,
        1300.0,
    )

    base_amplitude = rng.uniform(
        0.08,
        0.75,
    )

    fault_period = (
        1.0
        / fault_frequency_hz
    )

    first_impact = rng.uniform(
        0.0,
        fault_period,
    )

    current_impact_time = (
        first_impact
    )

    ball_modulation_phase = rng.uniform(
        0.0,
        2 * np.pi,
    )

    while (
        current_impact_time
        < WINDOW_DURATION
    ):

        # -------------------------------------------------
        # TIMING JITTER
        # -------------------------------------------------

        jitter_scale = 0.03

        if label == "ball_fault":
            jitter_scale = 0.06

        jitter = rng.normal(
            loc=0.0,
            scale=(
                jitter_scale
                * fault_period
            ),
        )

        actual_impact_time = (
            current_impact_time
            + jitter
        )

        if (
            0.0
            <= actual_impact_time
            < WINDOW_DURATION
        ):

            local_amplitude = (
                base_amplitude
                * rng.uniform(
                    0.65,
                    1.35,
                )
            )

            # -------------------------------------------------
            # INNER-RACE AMPLITUDE MODULATION
            # -------------------------------------------------

            if label == "inner_race":

                modulation_phase = (
                    2
                    * np.pi
                    * shaft_frequency_hz
                    * actual_impact_time
                )

                modulation = (
                    0.55
                    + 0.45
                    * (
                        0.5
                        * (
                            1
                            + np.cos(
                                modulation_phase
                            )
                        )
                    )
                )

                local_amplitude *= (
                    modulation
                )

            # -------------------------------------------------
            # BALL-FAULT AMPLITUDE MODULATION
            # -------------------------------------------------

            elif label == "ball_fault":

                modulation_frequency = (
                    0.5
                    * shaft_frequency_hz
                )

                modulation_phase = (
                    2
                    * np.pi
                    * modulation_frequency
                    * actual_impact_time
                    + ball_modulation_phase
                )

                modulation = (
                    0.35
                    + 0.65
                    * (
                        0.5
                        * (
                            1
                            + np.cos(
                                modulation_phase
                            )
                        )
                    )
                )

                local_amplitude *= (
                    modulation
                )

            # -------------------------------------------------
            # LOCAL STRUCTURAL VARIATION
            # -------------------------------------------------

            local_resonance = (
                base_resonance
                + rng.normal(
                    0.0,
                    35.0,
                )
            )

            local_decay = (
                base_decay
                * rng.uniform(
                    0.85,
                    1.15,
                )
            )

            ringing = create_ringing(
                resonance_frequency_hz=local_resonance,
                decay_rate=local_decay,
                impact_amplitude=local_amplitude,
            )

            add_impact(
                fault_signal,
                actual_impact_time,
                ringing,
            )

        current_impact_time += (
            fault_period
        )

    return fault_signal


# ---------------------------------------------------------
# GENERATE ONE SIGNAL
# ---------------------------------------------------------

def generate_signal(label):
    rpm = rng.uniform(
        1500.0,
        2100.0,
    )

    (
        shaft_frequency_hz,
        bpfo,
        bpfi,
        bsf,
        ftf,
    ) = calculate_fault_frequencies(
        rpm
    )

    # -----------------------------------------------------
    # ROTATIONAL COMPONENT
    # -----------------------------------------------------

    rotation_amplitude = rng.uniform(
        0.10,
        0.30,
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

    # -----------------------------------------------------
    # SECOND SHAFT HARMONIC
    # -----------------------------------------------------

    second_harmonic_amplitude = (
        rng.uniform(
            0.00,
            0.05,
        )
    )

    second_harmonic_phase = (
        rng.uniform(
            0.0,
            2 * np.pi,
        )
    )

    second_harmonic = (
        second_harmonic_amplitude
        * np.sin(
            2
            * np.pi
            * 2
            * shaft_frequency_hz
            * TIME
            + second_harmonic_phase
        )
    )

    # -----------------------------------------------------
    # MEASUREMENT NOISE
    # -----------------------------------------------------

    noise_std = rng.uniform(
        0.03,
        0.12,
    )

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=len(TIME),
    )

    signal = (
        rotation_component
        + second_harmonic
        + noise
    )

    # -----------------------------------------------------
    # ACCIDENTAL NON-FAULT IMPACTS
    # -----------------------------------------------------

    accidental_probability = (
        0.45
        if label == "healthy"
        else 0.25
    )

    if (
        rng.random()
        < accidental_probability
    ):

        number_of_impacts = (
            rng.integers(
                1,
                5,
            )
        )

        for _ in range(
            number_of_impacts
        ):

            resonance = rng.uniform(
                900.0,
                1500.0,
            )

            decay = rng.uniform(
                700.0,
                1300.0,
            )

            amplitude = rng.uniform(
                0.05,
                0.25,
            )

            ringing = create_ringing(
                resonance_frequency_hz=resonance,
                decay_rate=decay,
                impact_amplitude=amplitude,
            )

            impact_time = rng.uniform(
                0.0,
                WINDOW_DURATION,
            )

            add_impact(
                signal,
                impact_time,
                ringing,
            )

    # -----------------------------------------------------
    # OUTER-RACE FAULT
    # -----------------------------------------------------

    if label == "outer_race":

        signal += generate_fault_impacts(
            fault_frequency_hz=bpfo,
            shaft_frequency_hz=shaft_frequency_hz,
            label=label,
        )

    # -----------------------------------------------------
    # INNER-RACE FAULT
    # -----------------------------------------------------

    elif label == "inner_race":

        signal += generate_fault_impacts(
            fault_frequency_hz=bpfi,
            shaft_frequency_hz=shaft_frequency_hz,
            label=label,
        )

    # -----------------------------------------------------
    # BALL FAULT
    # -----------------------------------------------------

    elif label == "ball_fault":

        signal += generate_fault_impacts(
            fault_frequency_hz=bsf,
            shaft_frequency_hz=shaft_frequency_hz,
            label=label,
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
    shaft_frequency_hz = signal_data[
        "shaft_frequency_hz"
    ]

    bpfo = signal_data["bpfo"]
    bpfi = signal_data["bpfi"]
    bsf = signal_data["bsf"]
    ftf = signal_data["ftf"]

    # -----------------------------------------------------
    # TIME-DOMAIN FEATURES
    # -----------------------------------------------------

    rms = np.sqrt(
        np.mean(
            signal ** 2
        )
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

    frequencies = (
        np.fft.rfftfreq(
            len(centered_signal),
            d=1
            / SAMPLING_FREQUENCY,
        )
    )

    fft_amplitude = (
        2.0
        / len(centered_signal)
        * np.abs(fft_values)
    )

    def amplitude_near_frequency(
        target_frequency,
        search_width_hz=3.0,
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

    bpfo_raw = (
        amplitude_near_frequency(
            bpfo
        )
    )

    bpfi_raw = (
        amplitude_near_frequency(
            bpfi
        )
    )

    bsf_raw = (
        amplitude_near_frequency(
            bsf
        )
    )

    ftf_raw = (
        amplitude_near_frequency(
            ftf
        )
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
            d=1
            / SAMPLING_FREQUENCY,
        )
    )

    envelope_amplitude = (
        2.0
        / len(envelope_centered)
        * np.abs(envelope_fft)
    )

    def envelope_amplitude_near_frequency(
        target_frequency,
        search_width_hz=3.0,
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
        f"Generating "
        f"{number_of_samples} "
        f"samples for class: "
        f"{label}"
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

        dataset.append(
            features
        )

        if (
            (sample_number + 1)
            % 50
            == 0
        ):
            print(
                f"  "
                f"{sample_number + 1}"
                f"/"
                f"{number_of_samples}"
            )


for class_name in CLASSES:

    generate_class(
        class_name,
        SAMPLES_PER_CLASS,
    )


# ---------------------------------------------------------
# SHUFFLE DATASET
# ---------------------------------------------------------

rng.shuffle(
    dataset
)


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

print()
print("=" * 80)
print("MULTICLASS SYNTHETIC DATASET V2 CREATED")
print("=" * 80)

print(
    f"Total samples: "
    f"{len(dataset)}"
)

for class_name in CLASSES:

    count = sum(
        row["label"] == class_name
        for row in dataset
    )

    print(
        f"{class_name:<15}: "
        f"{count}"
    )

print()
print("Saved to:")
print(OUTPUT_FILE)

print("=" * 80)