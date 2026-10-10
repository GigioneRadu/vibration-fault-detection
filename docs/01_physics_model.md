# Physical Model

## 1. Engineering Problem

Rotating machinery such as electric motors, pumps, fans, compressors, and gearboxes generates mechanical vibrations during normal operation.

These vibrations are not only a consequence of motion. They also contain information about the mechanical condition of the system.

Changes in vibration amplitude, frequency content, or statistical behavior may indicate developing mechanical faults such as:

- rotor imbalance;
- shaft misalignment;
- mechanical looseness;
- bearing defects;
- structural resonance;
- coupling problems;
- excessive friction;
- mounting problems.

The purpose of this project is to analyze vibration signals and determine whether useful indicators of mechanical faults can be extracted using signal processing and machine learning techniques.

The complete project follows the engineering pipeline:

**sensor measurement → vibration signal → signal processing → feature extraction → machine learning → fault diagnosis**

Before working with real vibration data, it is important to understand the physical system that produces the vibration.

For this reason, the project begins with a simplified mechanical model.

---

## 2. Mass-Spring-Damper System

A large class of mechanical vibration problems can be approximated using a mass-spring-damper system.

The model contains three basic physical elements:

- **Mass (m)** — represents the inertia of the mechanical system.
- **Spring (k)** — represents the stiffness of the system.
- **Damper (c)** — represents mechanisms that dissipate mechanical energy.

The displacement of the mass is represented by:

$$
x(t)
$$

where `x` represents displacement and `t` represents time.

The corresponding velocity is:

$$
\dot{x}(t)
$$

and the acceleration is:

$$
\ddot{x}(t)
$$

An external force acting on the system is represented by:

$$
F(t)
$$

This simplified model does not reproduce every detail of a real electric motor or bearing assembly.

Its purpose is to provide a physically meaningful foundation for understanding vibration behavior before analyzing real sensor data.

---

## 3. Equation of Motion

The equation of motion is obtained from Newton's second law.

Newton's second law states that the sum of all forces acting on a body is equal to its mass multiplied by its acceleration:

$$
\sum F = m\ddot{x}
$$

For the mass-spring-damper system, three forces are considered.

### Spring force

According to Hooke's law, the spring force is proportional to displacement:

$$
F_k = -kx
$$

The negative sign indicates that the spring force acts in the opposite direction to the displacement.

If the mass moves away from its equilibrium position, the spring tries to return it toward equilibrium.

---

### Damping force

The damping force is assumed to be proportional to velocity:

$$
F_c = -c\dot{x}
$$

The negative sign indicates that damping opposes motion.

The damper dissipates mechanical energy and reduces the amplitude of oscillation over time.

---

### External force

An external excitation may act on the mechanical system:

$$
F(t)
$$

Examples of external excitation include:

- rotating imbalance;
- periodic mechanical loading;
- impacts;
- electromagnetic forces;
- contact forces;
- external disturbances.

Applying Newton's second law gives:

$$
F(t) - c\dot{x} - kx = m\ddot{x}
$$

Rearranging the terms gives the standard vibration equation:

$$
m\ddot{x} + c\dot{x} + kx = F(t)
$$

This equation is the fundamental physical model used in the first stage of the project.

---

## 4. Physical Meaning of Each Term

The equation

$$
m\ddot{x} + c\dot{x} + kx = F(t)
$$

contains four important physical terms.

### Inertial term

$$
m\ddot{x}
$$

This term represents the resistance of the mass to changes in motion.

A larger mass generally requires a larger force to produce the same acceleration.

---

### Damping term

$$
c\dot{x}
$$

This term represents energy dissipation.

In real mechanical systems, energy can be lost through:

- friction;
- material deformation;
- lubrication;
- internal structural damping;
- contact between components.

Damping strongly influences how quickly oscillations decrease after a disturbance.

---

### Stiffness term

$$
kx
$$

This term represents the restoring force of the system.

A system with high stiffness resists deformation more strongly than a flexible system.

---

### External excitation

$$
F(t)
$$

This term represents the force that excites the mechanical system.

In rotating machinery, the excitation can be periodic.

For example, rotor imbalance may generate a force that changes sinusoidally with time.

A simplified harmonic excitation can be written as:

$$
F(t) = F_0 \sin(\omega t)
$$

where:

- `F0` is the force amplitude;
- `omega` is the angular excitation frequency.

---

## 5. Free and Forced Vibration

Mechanical vibration can be divided into two major categories:

- free vibration;
- forced vibration.

### Free vibration

Free vibration occurs when the system is displaced from equilibrium and then released without continuous external excitation.

In this case:

$$
F(t) = 0
$$

The equation becomes:

$$
m\ddot{x} + c\dot{x} + kx = 0
$$

The system then vibrates according to its own physical properties.

Its response depends mainly on:

- mass;
- stiffness;
- damping;
- initial displacement;
- initial velocity.

---

### Forced vibration

Forced vibration occurs when an external force continuously excites the system.

The equation becomes:

$$
m\ddot{x} + c\dot{x} + kx = F(t)
$$

In rotating machinery, forced vibration is especially important because rotating components can generate periodic forces.

Examples include:

- imbalance;
- misalignment;
- gear meshing;
- bearing defects;
- rotating eccentricity.

The frequency content of the excitation can therefore appear in the measured vibration signal.

This is one of the key reasons why frequency-domain analysis is useful for fault diagnosis.

---

## 6. Natural Frequency and Damping

Every mechanical system has characteristic dynamic properties.

One of the most important is the natural frequency.

For the ideal undamped system:

$$
m\ddot{x} + kx = 0
$$

the natural angular frequency is:

$$
\omega_n = \sqrt{\frac{k}{m}}
$$

The natural frequency in hertz is:

$$
f_n = \frac{\omega_n}{2\pi}
$$

or:

$$
f_n = \frac{1}{2\pi}\sqrt{\frac{k}{m}}
$$

This equation shows an important engineering relationship:

- increasing stiffness increases natural frequency;
- increasing mass decreases natural frequency.

---

### Damping ratio

The damping ratio is defined as:

$$
\zeta = \frac{c}{2\sqrt{km}}
$$

The value of the damping ratio determines how the system responds after a disturbance.

Three important cases exist.

#### Underdamped system

$$
\zeta < 1
$$

The system oscillates while the amplitude gradually decreases.

This behavior is common in many mechanical systems.

#### Critically damped system

$$
\zeta = 1
$$

The system returns to equilibrium as quickly as possible without oscillating.

#### Overdamped system

$$
\zeta > 1
$$

The system returns to equilibrium without oscillation, but more slowly than the critically damped case.

---

## 7. Resonance

Resonance occurs when the excitation frequency approaches one of the natural frequencies of the system.

If the excitation frequency is close to the natural frequency, vibration amplitude can increase significantly.

In simplified form:

$$
f_{excitation} \approx f_n
$$

This phenomenon is extremely important in engineering because excessive resonance can produce:

- high vibration levels;
- fatigue;
- structural damage;
- loosened connections;
- bearing overload;
- reduced equipment lifetime.

For condition monitoring, resonance can also amplify certain fault-related vibration components and make them easier to detect.

---

## 8. Connection to Rotating Machinery

The mass-spring-damper model is not intended to reproduce every mechanical component inside a motor.

Instead, it provides a simplified dynamic representation of how a mechanical structure responds to excitation.

A real motor contains many interacting elements, including:

- rotor;
- shaft;
- bearings;
- housing;
- couplings;
- mounting structure;
- electromagnetic components.

Each of these elements can contribute to the measured vibration signal.

The vibration measured on the motor housing is therefore the result of a dynamic response produced by multiple excitation sources.

Conceptually:

**mechanical fault → excitation force → structural response → vibration → accelerometer signal**

This relationship is central to the project.

---

## 9. Connection to Measured Acceleration

In this project, vibration will primarily be treated as an acceleration signal because accelerometers are widely used in machinery condition monitoring.

The acceleration is represented by:

$$
\ddot{x}(t)
$$

An accelerometer mounted on the machine converts mechanical acceleration into an electrical signal.

After signal conditioning and analog-to-digital conversion, the vibration becomes a digital time series that can be processed in Python.

The resulting signal can be represented as:

$$
a(t)
$$

where `a(t)` is the measured acceleration as a function of time.

A typical processing chain is:

**mechanical vibration → accelerometer → electrical signal → ADC → digital samples → Python analysis**

---

## 10. From Time Domain to Frequency Domain

A vibration signal can first be observed in the time domain.

The signal shows how acceleration changes with time:

$$
a(t)
$$

However, many mechanical faults produce characteristic frequency components.

For this reason, the same signal can also be analyzed in the frequency domain.

The Fourier Transform converts the time-domain signal into a frequency-domain representation:

$$
a(t) \rightarrow A(f)
$$

where:

- `a(t)` represents acceleration in the time domain;
- `A(f)` represents the frequency content of the signal.

In practice, the project will use the Fast Fourier Transform (FFT).

The FFT will help identify:

- dominant frequencies;
- harmonics;
- resonance frequencies;
- sidebands;
- frequency components associated with mechanical faults.

---

## 11. Why This Model Matters for Bearing Fault Detection

Bearing faults generate repeated mechanical interactions between damaged surfaces and rolling elements.

These interactions can create periodic impulses.

The impulses excite the mechanical structure and generate vibration components that can be measured by an accelerometer.

Different bearing components produce different characteristic fault frequencies.

Important examples include:

- **BPFO** — Ball Pass Frequency Outer race;
- **BPFI** — Ball Pass Frequency Inner race;
- **BSF** — Ball Spin Frequency;
- **FTF** — Fundamental Train Frequency.

These frequencies depend on:

- bearing geometry;
- number of rolling elements;
- shaft rotational speed;
- rolling element diameter;
- pitch diameter;
- contact angle.

Later stages of the project will compare theoretical bearing fault frequencies with peaks observed in measured vibration spectra.

This creates a connection between physical modeling and machine learning.

Instead of treating vibration data as an arbitrary set of numbers, the machine learning model will be trained using features that have physical meaning.

Examples may include:

- RMS vibration;
- peak amplitude;
- crest factor;
- kurtosis;
- dominant spectral frequencies;
- energy near BPFO;
- energy near BPFI;
- energy near BSF;
- energy near FTF.

This approach is closer to physics-informed machine learning than to blind classification.

---

## 12. Assumptions and Limitations

The mass-spring-damper model is intentionally simplified.

Several assumptions are made.

### Linear stiffness

The spring force is assumed to be proportional to displacement:

$$
F_k = -kx
$$

Real mechanical systems may exhibit nonlinear stiffness.

---

### Linear damping

The damping force is assumed to be proportional to velocity:

$$
F_c = -c\dot{x}
$$

Real damping mechanisms can be nonlinear and frequency dependent.

---

### Single degree of freedom

The initial model uses one displacement variable:

$$
x(t)
$$

A real motor is a multi-degree-of-freedom system with vibration in multiple directions.

Later versions may consider multiple axes or more complex structural models.

---

### Simplified excitation

The initial simulations may use ideal harmonic or impulse excitation.

Real mechanical faults produce more complex signals containing:

- noise;
- non-stationary behavior;
- multiple frequency components;
- modulation;
- impacts;
- changing rotational speed.

---

### Sensor effects

The mathematical model initially assumes ideal measurement.

Real sensors introduce additional factors such as:

- sensitivity;
- bandwidth;
- noise;
- mounting effects;
- calibration error;
- sampling limitations;
- analog filtering.

These effects will be addressed separately in the sensor and data acquisition documentation.

---

## 13. Role of the Physical Model in the Project

The physical model provides the foundation for the complete vibration analysis workflow.

It allows the project to move gradually from theoretical mechanics to real condition monitoring.

The planned progression is:

**physical model → numerical simulation → synthetic vibration signals → signal processing → real vibration data → feature extraction → machine learning → fault diagnosis**

The objective is not only to build a machine learning classifier.

The objective is to understand why vibration signals contain useful fault information and how that information can be extracted in a physically meaningful way.