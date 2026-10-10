import numpy as np

from vibration_features import calculate_bearing_frequencies


# ============================================================
# DEFAULT SIGNAL SETTINGS
# ============================================================

DEFAULT_SAMPLING_FREQUENCY = 5000
DEFAULT_DURATION = 1.0


# ============================================================
# SYNTHETIC SIGNAL GENERATOR
# ============================================================

def generate_synthetic_vibration_signal(
    fault_type,
    rpm,
    sampling_frequency=DEFAULT_SAMPLING_FREQUENCY,
    duration=DEFAULT_DURATION,
    random_seed=12345,
):
    """
    Generate a synthetic raw vibration signal.

    Parameters
    ----------
    fault_type : str
        One of:
        healthy
        outer_race
        inner_race
        ball_fault
        cage_fault

    rpm : float
        Shaft rotational speed.

    sampling_frequency : int or float
        Sampling frequency in Hz.

    duration : float
        Signal duration in seconds.

    random_seed : int
        Seed used for deterministic testing.

    Returns
    -------
    numpy.ndarray
        Raw vibration signal.
    """

    valid_fault_types = {
        "healthy",
        "outer_race",
        "inner_race",
        "ball_fault",
        "cage_fault",
    }

    if fault_type not in valid_fault_types:
        raise ValueError(
            f"Unknown fault type: {fault_type}"
        )

    if rpm <= 0:
        raise ValueError(
            "RPM must be greater than zero."
        )

    if sampling_frequency <= 0:
        raise ValueError(
            "Sampling frequency must be greater than zero."
        )

    if duration <= 0:
        raise ValueError(
            "Duration must be greater than zero."
        )


    # --------------------------------------------------------
    # RANDOM GENERATOR
    # --------------------------------------------------------

    rng = np.random.default_rng(
        random_seed
    )


    # --------------------------------------------------------
    # TIME VECTOR
    # --------------------------------------------------------

    number_of_samples = int(
        sampling_frequency
        * duration
    )

    time = (
        np.arange(
            number_of_samples
        )
        / sampling_frequency
    )


    # --------------------------------------------------------
    # BEARING FREQUENCIES
    # --------------------------------------------------------

    frequencies = (
        calculate_bearing_frequencies(
            rpm
        )
    )

    shaft_frequency = frequencies[
        "shaft"
    ]


    # --------------------------------------------------------
    # BASIC SHAFT VIBRATION
    # --------------------------------------------------------

    signal = (
        0.18
        * np.sin(
            2.0
            * np.pi
            * shaft_frequency
            * time
        )
    )


    # --------------------------------------------------------
    # SECOND SHAFT HARMONIC
    # --------------------------------------------------------

    signal += (
        0.02
        * np.sin(
            2.0
            * np.pi
            * 2.0
            * shaft_frequency
            * time
        )
    )


    # --------------------------------------------------------
    # RANDOM NOISE
    # --------------------------------------------------------

    signal += rng.normal(
        loc=0.0,
        scale=0.06,
        size=number_of_samples,
    )


    # --------------------------------------------------------
    # SELECT FAULT FREQUENCY
    # --------------------------------------------------------

    fault_frequency = None

    if fault_type == "outer_race":
        fault_frequency = frequencies[
            "bpfo"
        ]

    elif fault_type == "inner_race":
        fault_frequency = frequencies[
            "bpfi"
        ]

    elif fault_type == "ball_fault":
        fault_frequency = frequencies[
            "bsf"
        ]

    elif fault_type == "cage_fault":
        fault_frequency = frequencies[
            "ftf"
        ]


    # --------------------------------------------------------
    # FAULT IMPACT RESPONSE
    # --------------------------------------------------------

    if fault_frequency is not None:

        impact_interval = (
            1.0
            / fault_frequency
        )

        impact_times = np.arange(
            0.0,
            duration,
            impact_interval,
        )

        resonance_frequency = 1200.0
        decay_rate = 650.0

        for impact_time in impact_times:

            jitter = rng.normal(
                loc=0.0,
                scale=0.0002,
            )

            actual_impact_time = (
                impact_time
                + jitter
            )

            relative_time = (
                time
                - actual_impact_time
            )

            mask = (
                relative_time
                >= 0.0
            )

            impulse = np.zeros_like(
                time
            )

            impulse[
                mask
            ] = (
                0.20
                * np.exp(
                    -decay_rate
                    * relative_time[
                        mask
                    ]
                )
                * np.sin(
                    2.0
                    * np.pi
                    * resonance_frequency
                    * relative_time[
                        mask
                    ]
                )
            )

            signal += impulse


    # --------------------------------------------------------
    # CAGE FAULT COMPONENT
    # --------------------------------------------------------

    if fault_type == "cage_fault":

        signal += (
            0.015
            * np.sin(
                2.0
                * np.pi
                * frequencies["ftf"]
                * time
            )
        )


    return signal