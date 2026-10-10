import numpy as np


# ---------------------------------------------------------
# BEARING GEOMETRY
# ---------------------------------------------------------

# Number of rolling elements
n = 8

# Rolling element diameter [mm]
d = 8.0

# Pitch diameter [mm]
D = 40.0

# Contact angle [degrees]
contact_angle_deg = 0.0

# Shaft rotational speed [RPM]
rpm = 1800.0


# ---------------------------------------------------------
# CONVERSIONS
# ---------------------------------------------------------

shaft_frequency_hz = rpm / 60.0

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
# OUTPUT
# ---------------------------------------------------------

print("=" * 65)
print("BEARING FAULT FREQUENCIES")
print("=" * 65)

print(f"Shaft speed:               {rpm:.1f} RPM")
print(f"Shaft rotational frequency:{shaft_frequency_hz:8.3f} Hz")

print()

print(f"BPFO - Outer race:         {bpfo:8.3f} Hz")
print(f"BPFI - Inner race:         {bpfi:8.3f} Hz")
print(f"BSF  - Rolling element:    {bsf:8.3f} Hz")
print(f"FTF  - Cage frequency:     {ftf:8.3f} Hz")

print("=" * 65)