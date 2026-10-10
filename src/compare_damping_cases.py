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
k = 100.0    # Spring stiffness [N/m]


# ---------------------------------------------------------
# INITIAL CONDITIONS
# ---------------------------------------------------------

x0 = 0.05    # Initial displacement [m]
v0 = 0.0     # Initial velocity [m/s]


# ---------------------------------------------------------
# SIMULATION TIME
# ---------------------------------------------------------

t_start = 0.0
t_end = 5.0

t_eval = np.linspace(t_start, t_end, 3000)


# ---------------------------------------------------------
# CRITICAL DAMPING
# ---------------------------------------------------------

c_critical = 2 * np.sqrt(k * m)

c_underdamped = 1.5
c_overdamped = 40.0


# ---------------------------------------------------------
# EXTERNAL FORCE
# ---------------------------------------------------------

def external_force(t):
    """
    Free vibration:
    no external force is applied.
    """
    return 0.0


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

def mass_spring_damper(t, state, c):
    """
    Mass-spring-damper equation:

        m*x_ddot + c*x_dot + k*x = F(t)

    The damping coefficient c is passed as an argument
    so the same model can be used for several damping cases.
    """

    x = state[0]
    velocity = state[1]

    force = external_force(t)

    acceleration = (
        force
        - c * velocity
        - k * x
    ) / m

    return [velocity, acceleration]


# ---------------------------------------------------------
# SIMULATION FUNCTION
# ---------------------------------------------------------

def simulate(c):
    """
    Solve the system for a given damping coefficient.
    """

    solution = solve_ivp(
        fun=lambda t, state: mass_spring_damper(t, state, c),
        t_span=(t_start, t_end),
        y0=[x0, v0],
        t_eval=t_eval,
    )

    return solution.t, solution.y[0]


# ---------------------------------------------------------
# RUN THE THREE CASES
# ---------------------------------------------------------

time_under, displacement_under = simulate(c_underdamped)

time_critical, displacement_critical = simulate(c_critical)

time_over, displacement_over = simulate(c_overdamped)


# ---------------------------------------------------------
# DAMPING RATIOS
# ---------------------------------------------------------

zeta_under = c_underdamped / (2 * np.sqrt(k * m))
zeta_critical = c_critical / (2 * np.sqrt(k * m))
zeta_over = c_overdamped / (2 * np.sqrt(k * m))


# ---------------------------------------------------------
# PRINT INFORMATION
# ---------------------------------------------------------

print("=" * 65)
print("DAMPING COMPARISON")
print("=" * 65)

print(f"Mass:             {m:.3f} kg")
print(f"Spring stiffness: {k:.3f} N/m")
print(f"Critical damping: {c_critical:.3f} N*s/m")

print()

print(
    f"Underdamped:      c = {c_underdamped:.3f} N*s/m"
    f" | zeta = {zeta_under:.3f}"
)

print(
    f"Critically damped: c = {c_critical:.3f} N*s/m"
    f" | zeta = {zeta_critical:.3f}"
)

print(
    f"Overdamped:       c = {c_overdamped:.3f} N*s/m"
    f" | zeta = {zeta_over:.3f}"
)

print("=" * 65)


# ---------------------------------------------------------
# PLOT
# ---------------------------------------------------------

plt.figure(figsize=(12, 7))

plt.plot(
    time_under,
    displacement_under,
    label=f"Underdamped (zeta = {zeta_under:.3f})",
    linewidth=1.5,
)

plt.plot(
    time_critical,
    displacement_critical,
    label=f"Critically damped (zeta = {zeta_critical:.1f})",
    linewidth=1.5,
)

plt.plot(
    time_over,
    displacement_over,
    label=f"Overdamped (zeta = {zeta_over:.1f})",
    linewidth=1.5,
)

plt.axhline(
    y=0,
    linewidth=0.8,
)

plt.title(
    "Comparison of Damping Regimes"
)

plt.xlabel("Time [s]")
plt.ylabel("Displacement [m]")

plt.grid(True)
plt.legend()

plt.tight_layout()


# ---------------------------------------------------------
# SAVE FIGURE
# ---------------------------------------------------------

figure_path = FIGURES_DIR / "damping_regimes_comparison.png"

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