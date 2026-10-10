from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt


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

n = 8                      # Number of rolling elements
d = 8.0                    # Rolling element diameter [mm]
D = 40.0                   # Pitch diameter [mm]
contact_angle_deg = 0.0

contact_angle_rad = np.deg2rad(contact_angle_deg)


# ---------------------------------------------------------
# BPFO
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
# NORMAL ROTATIONAL COMPONENT
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
# SYNTHETIC OUTER-RACE FAULT
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


# ---------------------------------------------------------
# INSERT IMPACTS
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
# BAND-PASS FILTER
# ---------------------------------------------------------

# We isolate the frequency region around the
# structural resonance excited by the impacts.

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
    vibration_signal,
)


# ---------------------------------------------------------
# HILBERT TRANSFORM
# ---------------------------------------------------------

# The analytic signal contains the instantaneous
# amplitude information of the filtered vibration.

analytic_signal = hilbert(
    filtered_signal
)

envelope = np.abs(
    analytic_signal
)


# ---------------------------------------------------------
# REMOVE ENVELOPE MEAN
# ---------------------------------------------------------

envelope_centered = (
    envelope
    - np.mean(envelope)
)


# ---------------------------------------------------------
# ENVELOPE FFT
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# DOMINANT ENVELOPE FREQUENCY
# ---------------------------------------------------------

search_mask = (
    (envelope_frequencies > 1.0)
    & (envelope_frequencies < 500.0)
)

search_frequencies = (
    envelope_frequencies[search_mask]
)

search_amplitudes = (
    envelope_amplitude[search_mask]
)

peak_index = np.argmax(
    search_amplitudes
)

dominant_frequency = (
    search_frequencies[peak_index]
)

dominant_amplitude = (
    search_amplitudes[peak_index]
)


# ---------------------------------------------------------
# AMPLITUDE NEAR TARGET FREQUENCY
# ---------------------------------------------------------

def amplitude_near_frequency(
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


bpfo_amplitude = amplitude_near_frequency(
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
print("BEARING ENVELOPE ANALYSIS")
print("=" * 70)

print(
    f"Shaft rotational frequency: "
    f"{shaft_frequency_hz:.3f} Hz"
)

print(
    f"BPFO:                       "
    f"{bpfo:.3f} Hz"
)

print()

print(
    f"Structural resonance:       "
    f"{resonance_frequency_hz:.1f} Hz"
)

print(
    f"Band-pass filter:           "
    f"{low_cut_hz:.0f} - {high_cut_hz:.0f} Hz"
)

print()

print(
    f"Dominant envelope frequency:"
    f" {dominant_frequency:.3f} Hz"
)

print(
    f"Dominant envelope amplitude:"
    f" {dominant_amplitude:.6f}"
)

print()

print(
    f"Envelope amplitude at BPFO: "
    f"{bpfo_amplitude:.6f}"
)

print(
    f"Envelope amplitude at 2xBPFO:"
    f" {bpfo_2x_amplitude:.6f}"
)

print(
    f"Envelope amplitude at 3xBPFO:"
    f" {bpfo_3x_amplitude:.6f}"
)

print("=" * 70)


# ---------------------------------------------------------
# DISPLAY WINDOW
# ---------------------------------------------------------

display_duration = 0.15

display_samples = int(
    display_duration
    * sampling_frequency
)


# ---------------------------------------------------------
# FILTERED SIGNAL + ENVELOPE
# ---------------------------------------------------------

plt.figure(
    figsize=(12, 7)
)

plt.plot(
    time[:display_samples],
    filtered_signal[:display_samples],
    linewidth=1.0,
    label="Band-pass filtered vibration",
)

plt.plot(
    time[:display_samples],
    envelope[:display_samples],
    linewidth=1.5,
    label="Envelope",
)

plt.title(
    "Filtered Bearing Vibration and Signal Envelope"
)

plt.xlabel("Time [s]")
plt.ylabel("Amplitude [arbitrary units]")

plt.grid(True)
plt.legend()

plt.tight_layout()

envelope_time_path = (
    FIGURES_DIR
    / "bearing_fault_envelope_time.png"
)

plt.savefig(
    envelope_time_path,
    dpi=300,
)


# ---------------------------------------------------------
# ENVELOPE SPECTRUM
# ---------------------------------------------------------

plt.figure(
    figsize=(12, 7)
)

plt.plot(
    envelope_frequencies,
    envelope_amplitude,
    linewidth=1.0,
)

plt.axvline(
    bpfo,
    linestyle="--",
    linewidth=1.2,
    label=f"BPFO = {bpfo:.1f} Hz",
)

plt.axvline(
    2 * bpfo,
    linestyle="--",
    linewidth=1.2,
    label=f"2 x BPFO = {2 * bpfo:.1f} Hz",
)

plt.axvline(
    3 * bpfo,
    linestyle="--",
    linewidth=1.2,
    label=f"3 x BPFO = {3 * bpfo:.1f} Hz",
)

plt.scatter(
    dominant_frequency,
    dominant_amplitude,
    zorder=3,
    label=(
        f"Dominant peak = "
        f"{dominant_frequency:.1f} Hz"
    ),
)

plt.xlim(
    0,
    400,
)

plt.title(
    "Envelope Spectrum - Outer Race Bearing Fault"
)

plt.xlabel("Frequency [Hz]")
plt.ylabel("Envelope amplitude")

plt.grid(True)
plt.legend()

plt.tight_layout()

envelope_spectrum_path = (
    FIGURES_DIR
    / "bearing_fault_envelope_spectrum.png"
)

plt.savefig(
    envelope_spectrum_path,
    dpi=300,
)


# ---------------------------------------------------------
# SHOW FIGURES
# ---------------------------------------------------------

plt.show()


# ---------------------------------------------------------
# SAVED FILES
# ---------------------------------------------------------

print()

print("Figures saved to:")

print(envelope_time_path)
print(envelope_spectrum_path)