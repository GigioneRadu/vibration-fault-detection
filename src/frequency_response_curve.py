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

m = 1.0      # Mass [kg]
c = 1.5      # Damping coefficient [N*s/m]
k = 100.0    # Spring stiffness [N/m]

F0 = 1.0     # Force amplitude [N]


# ---------------------------------------------------------
# SYSTEM CHARACTERISTICS
# ---------------------------------------------------------

omega_n = np.sqrt(k / m)
f_n = omega_n / (2 * np.pi)

zeta = c / (2 * np.sqrt(k * m))


# ---------------------------------------------------------
# FREQUENCY SWEEP
# ---------------------------------------------------------

frequencies_hz = np.linspace(
    0.1,
    4.0,
    120,
)


# ---------------------------------------------------------
# SIMULATION SETTINGS
# ---------------------------------------------------------

t_start = 0.0
t_end = 20.0

t_eval = np.linspace(
    t_start,
    t_end,
    6000,
)


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

def mass_spring_damper(t, state, excitation_frequency_hz):
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
# RESPONSE AMPLITUDE
# ---------------------------------------------------------

def steady_state_amplitude(excitation_frequency_hz):
    """
    Simulate the system and estimate the displacement
    amplitude from the final part of the signal.
    """

    solution = solve_ivp(
        fun=lambda t, state: mass_spring_damper(
            t,
            state,
            excitation_frequency_hz,
        ),
        t_span=(t_start, t_end),
        y0=[0.0, 0.0],
        t_eval=t_eval,
    )

    displacement = solution.y[0]

    # Use only the last 25% of the signal.
    # This reduces the influence of the transient response.
    start_index = int(0.75 * len(displacement))

    steady_displacement = displacement[start_index:]

    amplitude = (
        np.max(steady_displacement)
        - np.min(steady_displacement)
    ) / 2

    return amplitude


# ---------------------------------------------------------
# RUN FREQUENCY SWEEP
# ---------------------------------------------------------

amplitudes = []

for frequency in frequencies_hz:
    amplitude = steady_state_amplitude(frequency)
    amplitudes.append(amplitude)

amplitudes = np.array(amplitudes)


# ---------------------------------------------------------
# FIND MAXIMUM RESPONSE
# ---------------------------------------------------------

peak_index = np.argmax(amplitudes)

peak_frequency = frequencies_hz[peak_index]
peak_amplitude = amplitudes[peak_index]


# ---------------------------------------------------------
# PRINT INFORMATION
# ---------------------------------------------------------

print("=" * 65)
print("FREQUENCY RESPONSE")
print("=" * 65)

print(f"Mass:                {m:.3f} kg")
print(f"Damping coefficient: {c:.3f} N*s/m")
print(f"Spring stiffness:    {k:.3f} N/m")
print(f"Force amplitude:     {F0:.3f} N")

print()

print(f"Natural frequency:   {f_n:.3f} Hz")
print(f"Damping ratio:       {zeta:.3f}")

print()

print(f"Maximum response:    {peak_amplitude:.6f} m")
print(f"Peak frequency:      {peak_frequency:.3f} Hz")

print("=" * 65)


# ---------------------------------------------------------
# PLOT
# ---------------------------------------------------------

plt.figure(figsize=(12, 7))

plt.plot(
    frequencies_hz,
    amplitudes,
    linewidth=1.8,
)

plt.axvline(
    x=f_n,
    linestyle="--",
    linewidth=1.2,
    label=f"Natural frequency = {f_n:.3f} Hz",
)

plt.scatter(
    peak_frequency,
    peak_amplitude,
    zorder=3,
    label=(
        f"Maximum response = "
        f"{peak_frequency:.3f} Hz"
    ),
)

plt.title(
    "Frequency Response of the Mass-Spring-Damper System"
)

plt.xlabel("Excitation Frequency [Hz]")
plt.ylabel("Steady-State Displacement Amplitude [m]")

plt.grid(True)
plt.legend()

plt.tight_layout()


# ---------------------------------------------------------
# SAVE FIGURE
# ---------------------------------------------------------

figure_path = FIGURES_DIR / "frequency_response_curve.png"

plt.savefig(
    figure_path,
    dpi=300,
)

print()
print("Figure saved to:")
print(figure_path)


# ---------------------------------------------------------
# DISPLAY FIGURE
# ---------------------------------------------------------

plt.show()