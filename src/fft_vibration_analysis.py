from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# PHYSICAL PARAMETERS
# ---------------------------------------------------------

m = 1.0
c = 1.5
k = 100.0

F0 = 1.0


# ---------------------------------------------------------
# EXCITATION FREQUENCY
# ---------------------------------------------------------

omega_n = np.sqrt(k / m)
f_n = omega_n / (2 * np.pi)

excitation_frequency_hz = f_n


# ---------------------------------------------------------
# SIMULATION SETTINGS
# ---------------------------------------------------------

t_start = 0.0
t_end = 20.0

sampling_frequency = 500.0

number_of_samples = int(
    (t_end - t_start) * sampling_frequency
)

time = np.linspace(
    t_start,
    t_end,
    number_of_samples,
    endpoint=False,
)


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

def mass_spring_damper(t, state):
    x = state[0]
    velocity = state[1]

    omega = 2 * np.pi * excitation_frequency_hz

    force = F0 * np.sin(omega * t)

    acceleration = (
        force
        - c * velocity
        - k * x
    ) / m

    return [velocity, acceleration]


# ---------------------------------------------------------
# SOLVE SYSTEM
# ---------------------------------------------------------

solution = solve_ivp(
    fun=mass_spring_damper,
    t_span=(t_start, t_end),
    y0=[0.0, 0.0],
    t_eval=time,
)

displacement = solution.y[0]
velocity = solution.y[1]


# ---------------------------------------------------------
# CALCULATE ACCELERATION
# ---------------------------------------------------------

acceleration = np.array(
    [
        mass_spring_damper(t, [x, v])[1]
        for t, x, v in zip(
            time,
            displacement,
            velocity,
        )
    ]
)


# ---------------------------------------------------------
# REMOVE TRANSIENT REGION
# ---------------------------------------------------------

start_index = int(0.5 * len(time))

steady_time = time[start_index:]
steady_acceleration = acceleration[start_index:]


# ---------------------------------------------------------
# REMOVE MEAN
# ---------------------------------------------------------

steady_acceleration = (
    steady_acceleration
    - np.mean(steady_acceleration)
)


# ---------------------------------------------------------
# FFT
# ---------------------------------------------------------

fft_values = np.fft.rfft(
    steady_acceleration
)

frequencies = np.fft.rfftfreq(
    len(steady_acceleration),
    d=1 / sampling_frequency,
)


# ---------------------------------------------------------
# FFT AMPLITUDE
# ---------------------------------------------------------

fft_amplitude = (
    2.0
    / len(steady_acceleration)
    * np.abs(fft_values)
)


# ---------------------------------------------------------
# DOMINANT FREQUENCY
# ---------------------------------------------------------

peak_index = np.argmax(
    fft_amplitude[1:]
) + 1

dominant_frequency = frequencies[peak_index]
dominant_amplitude = fft_amplitude[peak_index]


# ---------------------------------------------------------
# PRINT INFORMATION
# ---------------------------------------------------------

print("=" * 65)
print("FFT VIBRATION ANALYSIS")
print("=" * 65)

print(f"Natural frequency:      {f_n:.3f} Hz")
print(
    f"Excitation frequency:   "
    f"{excitation_frequency_hz:.3f} Hz"
)

print(
    f"Sampling frequency:     "
    f"{sampling_frequency:.1f} Hz"
)

print()

print(
    f"Dominant FFT frequency: "
    f"{dominant_frequency:.3f} Hz"
)

print(
    f"Dominant FFT amplitude: "
    f"{dominant_amplitude:.6f}"
)

print("=" * 65)


# ---------------------------------------------------------
# TIME-DOMAIN PLOT
# ---------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    steady_time,
    steady_acceleration,
    linewidth=1.2,
)

plt.title(
    "Steady-State Acceleration Signal"
)

plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s²]")

plt.grid(True)

plt.tight_layout()

time_figure_path = (
    FIGURES_DIR
    / "steady_state_acceleration.png"
)

plt.savefig(
    time_figure_path,
    dpi=300,
)

plt.show()


# ---------------------------------------------------------
# FREQUENCY-DOMAIN PLOT
# ---------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    frequencies,
    fft_amplitude,
    linewidth=1.2,
)

plt.axvline(
    excitation_frequency_hz,
    linestyle="--",
    linewidth=1.2,
    label=(
        f"Excitation frequency = "
        f"{excitation_frequency_hz:.3f} Hz"
    ),
)

plt.scatter(
    dominant_frequency,
    dominant_amplitude,
    zorder=3,
    label=(
        f"Dominant FFT peak = "
        f"{dominant_frequency:.3f} Hz"
    ),
)

plt.xlim(
    0,
    10,
)

plt.title(
    "FFT Spectrum of the Vibration Signal"
)

plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")

plt.grid(True)
plt.legend()

plt.tight_layout()

fft_figure_path = (
    FIGURES_DIR
    / "fft_vibration_spectrum.png"
)

plt.savefig(
    fft_figure_path,
    dpi=300,
)

plt.show()