from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

try:
    from synthetic_signal_generator import generate_synthetic_vibration_signal
except ImportError:
    from src.synthetic_signal_generator import generate_synthetic_vibration_signal


# ============================================================
# MEASUREMENT WINDOW
# ============================================================

@dataclass
class VibrationMeasurement:
    """
    Standard data structure returned by every vibration source.

    The rest of the diagnostic pipeline should not need to know
    whether the signal came from:

    - the synthetic simulator;
    - a real accelerometer;
    - a DAQ device;
    - an industrial edge device.

    It receives the same VibrationMeasurement object.
    """

    signal: np.ndarray
    rpm: float
    sampling_frequency: int
    timestamp: datetime
    source_name: str


# ============================================================
# BASE SIGNAL SOURCE
# ============================================================

class SignalSource(ABC):
    """
    Abstract base class for all vibration signal sources.

    Every future acquisition source must implement read().
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable source name.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def available(self) -> bool:
        """
        Indicates whether this source can currently be used.
        """
        raise NotImplementedError

    @abstractmethod
    def read(self) -> VibrationMeasurement:
        """
        Acquire one vibration measurement window.
        """
        raise NotImplementedError


# ============================================================
# SYNTHETIC SIGNAL SOURCE
# ============================================================

class SyntheticSignalSource(SignalSource):
    """
    Current working signal source.

    Generates synthetic vibration data using the existing
    synthetic_signal_generator module.
    """

    def __init__(
        self,
        fault_type: str,
        rpm: float,
        sampling_frequency: int = 5000,
        duration: float = 1.0,
        random_seed: Optional[int] = None,
    ):
        self.fault_type = fault_type
        self.rpm = rpm
        self.sampling_frequency = sampling_frequency
        self.duration = duration
        self.random_seed = random_seed

    @property
    def name(self) -> str:
        return "Synthetic simulation"

    @property
    def available(self) -> bool:
        return True

    def read(self) -> VibrationMeasurement:
        """
        Generate one synthetic vibration window.
        """

        signal = generate_synthetic_vibration_signal(
            fault_type=self.fault_type,
            rpm=self.rpm,
            sampling_frequency=self.sampling_frequency,
            duration=self.duration,
            random_seed=self.random_seed,
        )

        return VibrationMeasurement(
            signal=signal,
            rpm=self.rpm,
            sampling_frequency=self.sampling_frequency,
            timestamp=datetime.now(),
            source_name=self.name,
        )


# ============================================================
# REAL SENSOR SOURCE
# ============================================================

class RealSensorSource(SignalSource):
    """
    Placeholder for future real vibration acquisition.

    This class deliberately exists in the architecture now,
    but acquisition is disabled until real hardware support
    is implemented.
    """

    def __init__(
        self,
        sampling_frequency: int = 5000,
    ):
        self.sampling_frequency = sampling_frequency

    @property
    def name(self) -> str:
        return "Live sensor input"

    @property
    def available(self) -> bool:
        return False

    def read(self) -> VibrationMeasurement:
        """
        Real sensor acquisition is not implemented yet.
        """

        raise RuntimeError(
            "Real sensor input is not available yet. "
            "Connect and configure a supported vibration "
            "acquisition device before using this source."
        )


# ============================================================
# PLC / VFD RPM SOURCE
# ============================================================

class PLCRPMSource:
    """
    Placeholder for future PLC/VFD speed acquisition.

    In a real deployment this class could later communicate
    through protocols such as:

    - OPC UA
    - Modbus TCP
    - Profinet
    - EtherNet/IP

    No industrial protocol is implemented at this stage.
    """

    def __init__(self):
        self._available = False

    @property
    def name(self) -> str:
        return "PLC / VFD RPM"

    @property
    def available(self) -> bool:
        return self._available

    def read_rpm(self) -> float:
        """
        Read current shaft speed from PLC/VFD.

        Not implemented yet.
        """

        raise RuntimeError(
            "PLC / VFD RPM acquisition is not available yet."
        )