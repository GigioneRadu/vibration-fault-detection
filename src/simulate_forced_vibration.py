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
# EXCITATION FREQUENCIES
# ---------------------------------------------------------

# Frequencies below, near, and above resonance
excitation_frequencies_hz = [
    0.5 * f_n,
    1.0 * f_n,
    2.0 * f_n,
]


# ---------------------------------------------------------
# INITIAL CONDITIONS
# ---------------------------------------------------------

x0 = 0.0
v0 = 0.0


# ---------------------------------------------------------
# SIMULATION TIME
# ---------------------------------------------------------

t_start = 0.0
t_end = 20.0

t_eval = np.linspace(t_start, t_end, 10000)


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

def mass_spring_damper(t, state, excitation_frequency_hz):
    """
    Forced vibration of a single-degree-of-freedom
    mass-spring-damper system.

    Equation:

        m*x_ddot + c*x_dot + k*x = F0*sin(omega*t)
    """

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
# SIMULATION FUNCTION
# ---------------------------------------------------------

def simulate(excitation_frequency_hz):
    solution = solve_ivp(
        fun=lambda t, state: mass_spring_damper(
            t,
            state,
            excitation_frequency_hz,
        ),
        t_span=(t_start, t_end),
        y0=[x0, v0],
        t_eval=t_eval,
    )

    return solution.t, solution.y[0]


# ---------------------------------------------------------
# RUN SIMULATIONS
# ---------------------------------------------------------

results = []

for frequency in excitation_frequencies_hz:
    time, displacement = simulate(frequency)

    results.append(
        {
            "frequency": frequency,
            "time": time,
            "displacement": displacement,
        }
    )


# ---------------------------------------------------------
# PRINT SYSTEM INFORMATION
# ---------------------------------------------------------

print("=" * 65)
print("FORCED VIBRATION AND RESONANCE")
print("=" * 65)

print(f"Mass:                {m:.3f} kg")
print(f"Damping coefficient: {c:.3f} N*s/m")
print(f"Spring stiffness:    {k:.3f} N/m")
print(f"Force amplitude:     {F0:.3f} N")

print()

print(f"Natural angular frequency: {omega_n:.3f} rad/s")
print(f"Natural frequency:         {f_n:.3f} Hz")
print(f"Damping ratio:             {zeta:.3f}")

print()

for result in results:
    print(
        f"Excitation frequency: "
        f"{result['frequency']:.3f} Hz"
    )

print("=" * 65)


# ---------------------------------------------------------
# PLOT
# ---------------------------------------------------------

plt.figure(figsize=(12, 7))

for result in results:
    plt.plot(
        result["time"],
        result["displacement"],
        linewidth=1.3,
        label=(
            f"Excitation = "
            f"{result['frequency']:.3f} Hz"
        ),
    )

plt.axhline(
    y=0,
    linewidth=0.8,
)

plt.title(
    "Forced Vibration at Different Excitation Frequencies"
)

plt.xlabel("Time [s]")
plt.ylabel("Displacement [m]")

plt.grid(True)
plt.legend()

plt.tight_layout()


# ---------------------------------------------------------
# SAVE FIGURE
# ---------------------------------------------------------

figure_path = FIGURES_DIR / "forced_vibration_resonance.png"

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