from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# BEARING / SHAFT PARAMETERS
# ---------------------------------------------------------

rpm = 1800.0
shaft_frequency_hz = rpm / 60.0

# Bearing geometry
n = 8                      # Number of rolling elements
d = 8.0                    # Rolling element diameter [mm]
D = 40.0                   # Pitch diameter [mm]
contact_angle_deg = 0.0

contact_angle_rad = np.deg2rad(contact_angle_deg)


# ---------------------------------------------------------
# BPFO CALCULATION
# ---------------------------------------------------------

bpfo = (
    n / 2
    * shaft_frequency_hz
    * (
        1
        - (d / D) * np.cos(contact_angle_rad)
    )
)


# ---------------------------------------------------------
# SIGNAL SETTINGS
# ---------------------------------------------------------

sampling_frequency = 5000.0   # Hz
duration = 5.0                # seconds

number_of_samples = int(
    sampling_frequency * duration
)

time = np.arange(number_of_samples) / sampling_frequency


# ---------------------------------------------------------
# NORMAL ROTATIONAL VIBRATION
# ---------------------------------------------------------

rotation_component = (
    0.20
    * np.sin(
        2 * np.pi * shaft_frequency_hz * time
    )
)


# ---------------------------------------------------------
# OUTER-RACE FAULT IMPULSES
# ---------------------------------------------------------

fault_signal = np.zeros_like(time)

fault_period = 1.0 / bpfo

impact_times = np.arange(
    0.0,
    duration,
    fault_period,
)


# ---------------------------------------------------------
# STRUCTURAL RESONANCE
# ---------------------------------------------------------

# A bearing impact usually excites a much higher-frequency
# structural resonance rather than producing only a low-
# frequency sinusoid at BPFO.

resonance_frequency_hz = 1200.0

decay_rate = 900.0

impact_amplitude = 1.0

ringing_duration = 0.015

ringing_samples = int(
    ringing_duration * sampling_frequency
)

ringing_time = (
    np.arange(ringing_samples)
    / sampling_frequency
)

ringing = (
    impact_amplitude
    * np.exp(-decay_rate * ringing_time)
    * np.sin(
        2
        * np.pi
        * resonance_frequency_hz
        * ringing_time
    )
)


# ---------------------------------------------------------
# INSERT REPEATED IMPACTS
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# ADD MEASUREMENT NOISE
# ---------------------------------------------------------

rng = np.random.default_rng(seed=42)

noise = rng.normal(
    loc=0.0,
    scale=0.04,
    size=len(time),
)


# ---------------------------------------------------------
# COMPLETE VIBRATION SIGNAL
# ---------------------------------------------------------

vibration_signal = (
    rotation_component
    + fault_signal
    + noise
)


# ---------------------------------------------------------
# TIME-DOMAIN DISPLAY WINDOW
# ---------------------------------------------------------

display_duration = 0.15

display_samples = int(
    display_duration
    * sampling_frequency
)


# ---------------------------------------------------------
# FFT
# ---------------------------------------------------------

signal_for_fft = (
    vibration_signal
    - np.mean(vibration_signal)
)

fft_values = np.fft.rfft(
    signal_for_fft
)

frequencies = np.fft.rfftfreq(
    len(signal_for_fft),
    d=1 / sampling_frequency,
)

fft_amplitude = (
    2.0
    / len(signal_for_fft)
    * np.abs(fft_values)
)


# ---------------------------------------------------------
# FIND AMPLITUDES NEAR BPFO HARMONICS
# ---------------------------------------------------------

def amplitude_near_frequency(
    target_frequency,
    search_width_hz=1.0,
):
    mask = (
        (frequencies >= target_frequency - search_width_hz)
        & (frequencies <= target_frequency + search_width_hz)
    )

    if not np.any(mask):
        return 0.0

    return np.max(
        fft_amplitude[mask]
    )


bpfo_1x_amplitude = amplitude_near_frequency(
    bpfo
)

bpfo_2x_amplitude = amplitude_near_frequency(
    2 * bpfo
)

bpfo_3x_amplitude = amplitude_near_frequency(
    3 * bpfo
)


# ---------------------------------------------------------
# PRINT INFORMATION
# ---------------------------------------------------------

print("=" * 70)
print("SYNTHETIC OUTER-RACE BEARING FAULT")
print("=" * 70)

print(f"Shaft speed:                 {rpm:.1f} RPM")
print(
    f"Shaft rotational frequency: "
    f"{shaft_frequency_hz:.3f} Hz"
)

print()

print(
    f"BPFO:                        "
    f"{bpfo:.3f} Hz"
)

print(
    f"BPFO 2nd harmonic:           "
    f"{2 * bpfo:.3f} Hz"
)

print(
    f"BPFO 3rd harmonic:           "
    f"{3 * bpfo:.3f} Hz"
)

print()

print(
    f"Structural resonance:        "
    f"{resonance_frequency_hz:.1f} Hz"
)

print(
    f"Sampling frequency:          "
    f"{sampling_frequency:.1f} Hz"
)

print()

print(
    f"Amplitude near BPFO:         "
    f"{bpfo_1x_amplitude:.6f}"
)

print(
    f"Amplitude near 2 x BPFO:     "
    f"{bpfo_2x_amplitude:.6f}"
)

print(
    f"Amplitude near 3 x BPFO:     "
    f"{bpfo_3x_amplitude:.6f}"
)

print("=" * 70)


# ---------------------------------------------------------
# TIME-DOMAIN PLOT
# ---------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    time[:display_samples],
    vibration_signal[:display_samples],
    linewidth=1.0,
)

plt.title(
    "Synthetic Vibration Signal - Outer Race Fault"
)

plt.xlabel("Time [s]")
plt.ylabel("Acceleration [arbitrary units]")

plt.grid(True)
plt.tight_layout()

time_figure_path = (
    FIGURES_DIR
    / "outer_race_fault_time_signal.png"
)

plt.savefig(
    time_figure_path,
    dpi=300,
)


# ---------------------------------------------------------
# FFT PLOT - LOW FREQUENCY REGION
# ---------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    frequencies,
    fft_amplitude,
    linewidth=1.0,
)

plt.axvline(
    shaft_frequency_hz,
    linestyle="--",
    linewidth=1.0,
    label=(
        f"Shaft frequency = "
        f"{shaft_frequency_hz:.1f} Hz"
    ),
)

plt.axvline(
    bpfo,
    linestyle="--",
    linewidth=1.0,
    label=f"BPFO = {bpfo:.1f} Hz",
)

plt.axvline(
    2 * bpfo,
    linestyle="--",
    linewidth=1.0,
    label=f"2 x BPFO = {2 * bpfo:.1f} Hz",
)

plt.axvline(
    3 * bpfo,
    linestyle="--",
    linewidth=1.0,
    label=f"3 x BPFO = {3 * bpfo:.1f} Hz",
)

plt.xlim(
    0,
    400,
)

plt.title(
    "FFT Spectrum - Outer Race Fault"
)

plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")

plt.grid(True)
plt.legend()

plt.tight_layout()

fft_figure_path = (
    FIGURES_DIR
    / "outer_race_fault_fft.png"
)

plt.savefig(
    fft_figure_path,
    dpi=300,
)


# ---------------------------------------------------------
# HIGH-FREQUENCY RESONANCE PLOT
# ---------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    frequencies,
    fft_amplitude,
    linewidth=1.0,
)

plt.axvline(
    resonance_frequency_hz,
    linestyle="--",
    linewidth=1.0,
    label=(
        f"Structural resonance = "
        f"{resonance_frequency_hz:.0f} Hz"
    ),
)

plt.xlim(
    800,
    1600,
)

plt.title(
    "High-Frequency Resonance Excited by Bearing Impacts"
)

plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")

plt.grid(True)
plt.legend()

plt.tight_layout()

resonance_figure_path = (
    FIGURES_DIR
    / "outer_race_fault_resonance.png"
)

plt.savefig(
    resonance_figure_path,
    dpi=300,
)


# ---------------------------------------------------------
# SHOW ALL FIGURES
# ---------------------------------------------------------

plt.show()


# ---------------------------------------------------------
# SAVED FILES
# ---------------------------------------------------------

print()
print("Figures saved to:")

print(time_figure_path)
print(fft_figure_path)
print(resonance_figure_path)