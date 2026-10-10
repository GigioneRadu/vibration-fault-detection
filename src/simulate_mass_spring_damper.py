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

# Mass [kg]
m = 1.0

# Damping coefficient [N*s/m]
c = 1.5

# Spring stiffness [N/m]
k = 100.0


# ---------------------------------------------------------
# INITIAL CONDITIONS
# ---------------------------------------------------------

# Initial displacement [m]
x0 = 0.05

# Initial velocity [m/s]
v0 = 0.0


# ---------------------------------------------------------
# SIMULATION TIME
# ---------------------------------------------------------

t_start = 0.0
t_end = 10.0

t_eval = np.linspace(t_start, t_end, 5000)


# ---------------------------------------------------------
# EXTERNAL FORCE
# ---------------------------------------------------------

def external_force(t):
    """
    External force applied to the system.

    For the first simulation, the system is allowed
    to vibrate freely, so no external force is applied.
    """
    return 0.0


# ---------------------------------------------------------
# MASS-SPRING-DAMPER MODEL
# ---------------------------------------------------------

def mass_spring_damper(t, state):
    """
    Differential equation for a single-degree-of-freedom
    mass-spring-damper system.

    The physical equation is:

        m*x_ddot + c*x_dot + k*x = F(t)

    State variables:
        state[0] = displacement x
        state[1] = velocity x_dot
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
# SYSTEM CHARACTERISTICS
# ---------------------------------------------------------

omega_n = np.sqrt(k / m)

f_n = omega_n / (2 * np.pi)

zeta = c / (2 * np.sqrt(k * m))


# ---------------------------------------------------------
# NUMERICAL SOLUTION
# ---------------------------------------------------------

solution = solve_ivp(
    fun=mass_spring_damper,
    t_span=(t_start, t_end),
    y0=[x0, v0],
    t_eval=t_eval,
)


# ---------------------------------------------------------
# EXTRACT RESULTS
# ---------------------------------------------------------

time = solution.t
displacement = solution.y[0]
velocity = solution.y[1]

acceleration = np.array(
    [
        mass_spring_damper(t, [x, v])[1]
        for t, x, v in zip(time, displacement, velocity)
    ]
)


# ---------------------------------------------------------
# PRINT SYSTEM INFORMATION
# ---------------------------------------------------------

print("=" * 60)
print("MASS-SPRING-DAMPER SYSTEM")
print("=" * 60)

print(f"Mass:                {m:.3f} kg")
print(f"Damping coefficient: {c:.3f} N*s/m")
print(f"Spring stiffness:    {k:.3f} N/m")

print()

print(f"Natural angular frequency: {omega_n:.3f} rad/s")
print(f"Natural frequency:         {f_n:.3f} Hz")
print(f"Damping ratio:             {zeta:.3f}")

print()

if zeta < 1:
    print("System type: UNDERDAMPED")
elif np.isclose(zeta, 1.0):
    print("System type: CRITICALLY DAMPED")
else:
    print("System type: OVERDAMPED")

print("=" * 60)


# ---------------------------------------------------------
# PLOT DISPLACEMENT
# ---------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    time,
    displacement,
    linewidth=1.5,
)

plt.axhline(
    y=0,
    linewidth=0.8,
)

plt.title(
    "Free Vibration of a Mass-Spring-Damper System"
)

plt.xlabel("Time [s]")
plt.ylabel("Displacement [m]")

plt.grid(True)

plt.tight_layout()


# ---------------------------------------------------------
# SAVE FIGURE
# ---------------------------------------------------------

figure_path = FIGURES_DIR / "free_vibration_displacement.png"

plt.savefig(
    figure_path,
    dpi=300,
)

print()
print(f"Figure saved to:")
print(figure_path)


# ---------------------------------------------------------
# DISPLAY FIGURE
# ---------------------------------------------------------

plt.show()