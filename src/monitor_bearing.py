from pathlib import Path
from time import sleep

import pandas as pd

try:
    from bearing_predictor import BearingPredictor
    from signal_sources import SyntheticSignalSource
    from vibration_features import extract_vibration_features
except ImportError:
    from src.bearing_predictor import BearingPredictor
    from src.signal_sources import SyntheticSignalSource
    from src.vibration_features import extract_vibration_features


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bearing_monitoring_history.csv"
)


# ============================================================
# MONITORING SETTINGS
# ============================================================

SAMPLING_FREQUENCY = 5000

WINDOW_DURATION = 1.0

NUMBER_OF_WINDOWS = 10

SLEEP_BETWEEN_WINDOWS = 1.0

CONFIDENCE_THRESHOLD = 0.60


# ============================================================
# PERSISTENT ALARM SETTINGS
# ============================================================

FAULT_CONFIRMATION_WINDOWS = 3

NORMAL_CONFIRMATION_WINDOWS = 2

CHECK_CONFIRMATION_WINDOWS = 2

FAULT_CLEAR_WINDOWS = 3


# ============================================================
# MACHINES
# ============================================================

MACHINES = [
    {
        "machine_id": "M01",
        "machine_name": "Pump 01",
        "rpm": 1800,
        "condition_schedule": [
            "healthy",
            "healthy",
            "healthy",
            "healthy",
            "healthy",
            "healthy",
            "healthy",
            "healthy",
            "healthy",
            "healthy",
        ],
    },
    {
        "machine_id": "M02",
        "machine_name": "Fan 02",
        "rpm": 1750,
        "condition_schedule": [
            "healthy",
            "healthy",
            "healthy",
            "healthy",
            "healthy",
            "healthy",
            "healthy",
            "healthy",
            "healthy",
            "healthy",
        ],
    },
    {
        "machine_id": "M03",
        "machine_name": "Motor 03",
        "rpm": 1800,
        "condition_schedule": [
            "healthy",
            "healthy",
            "healthy",
            "cage_fault",
            "cage_fault",
            "cage_fault",
            "cage_fault",
            "cage_fault",
            "cage_fault",
            "cage_fault",
        ],
    },
    {
        "machine_id": "M04",
        "machine_name": "Pump 04",
        "rpm": 1650,
        "condition_schedule": [
            "healthy",
            "healthy",
            "outer_race",
            "outer_race",
            "outer_race",
            "outer_race",
            "outer_race",
            "outer_race",
            "outer_race",
            "outer_race",
        ],
    },
    {
        "machine_id": "M05",
        "machine_name": "Fan 05",
        "rpm": 1950,
        "condition_schedule": [
            "healthy",
            "healthy",
            "healthy",
            "healthy",
            "ball_fault",
            "ball_fault",
            "ball_fault",
            "ball_fault",
            "ball_fault",
            "ball_fault",
        ],
    },
]


# ============================================================
# PERSISTENT STATE MEMORY
# ============================================================

machine_states = {}

for machine in MACHINES:

    machine_id = machine["machine_id"]

    machine_states[machine_id] = {
        "confirmed_state": "CHECK",
        "consecutive_fault": 0,
        "consecutive_check": 0,
        "consecutive_normal": 0,
    }


# ============================================================
# PERSISTENCE LOGIC
# ============================================================

def update_persistent_state(
    machine_id: str,
    model_state: str,
) -> dict:
    """
    Update persistent operational state for one machine.

    model_state:
        Result from the current one-second measurement window.

    confirmed_state:
        Persistent operational state based on multiple windows.
    """

    state_memory = machine_states[machine_id]

    confirmed_state = state_memory["confirmed_state"]

    consecutive_fault = state_memory["consecutive_fault"]
    consecutive_check = state_memory["consecutive_check"]
    consecutive_normal = state_memory["consecutive_normal"]


    # ========================================================
    # CURRENT WINDOW = FAULT
    # ========================================================

    if model_state == "FAULT":

        consecutive_fault += 1

        consecutive_check = 0
        consecutive_normal = 0


        if confirmed_state == "FAULT":

            new_state = "FAULT"

        elif consecutive_fault >= FAULT_CONFIRMATION_WINDOWS:

            new_state = "FAULT"

        else:

            new_state = "CHECK"


    # ========================================================
    # CURRENT WINDOW = NORMAL
    # ========================================================

    elif model_state == "NORMAL":

        consecutive_normal += 1

        consecutive_fault = 0
        consecutive_check = 0


        # ----------------------------------------------------
        # Clearing an already confirmed FAULT
        # ----------------------------------------------------

        if confirmed_state == "FAULT":

            if consecutive_normal >= FAULT_CLEAR_WINDOWS:

                new_state = "NORMAL"

            else:

                new_state = "FAULT"


        # ----------------------------------------------------
        # Confirming NORMAL from CHECK
        # ----------------------------------------------------

        elif confirmed_state == "CHECK":

            if consecutive_normal >= NORMAL_CONFIRMATION_WINDOWS:

                new_state = "NORMAL"

            else:

                new_state = "CHECK"


        # ----------------------------------------------------
        # Already NORMAL
        # ----------------------------------------------------

        else:

            new_state = "NORMAL"


    # ========================================================
    # CURRENT WINDOW = CHECK
    # ========================================================

    elif model_state == "CHECK":

        consecutive_check += 1

        consecutive_fault = 0
        consecutive_normal = 0


        # ----------------------------------------------------
        # A confirmed fault is not cleared by uncertain data
        # ----------------------------------------------------

        if confirmed_state == "FAULT":

            new_state = "FAULT"


        # ----------------------------------------------------
        # Already CHECK
        # ----------------------------------------------------

        elif confirmed_state == "CHECK":

            new_state = "CHECK"


        # ----------------------------------------------------
        # NORMAL requires persistent CHECK before downgrade
        # ----------------------------------------------------

        else:

            if consecutive_check >= CHECK_CONFIRMATION_WINDOWS:

                new_state = "CHECK"

            else:

                new_state = "NORMAL"


    else:

        raise ValueError(
            f"Unknown model_state: {model_state}"
        )


    # ========================================================
    # STORE UPDATED MEMORY
    # ========================================================

    state_memory["confirmed_state"] = new_state

    state_memory["consecutive_fault"] = consecutive_fault

    state_memory["consecutive_check"] = consecutive_check

    state_memory["consecutive_normal"] = consecutive_normal


    return {
        "state": new_state,
        "consecutive_fault": consecutive_fault,
        "consecutive_check": consecutive_check,
        "consecutive_normal": consecutive_normal,
    }


# ============================================================
# MONITORING
# ============================================================

def main():

    print()
    print("=" * 100)
    print("BEARING CONDITION MONITORING")
    print("=" * 100)

    print()
    print("Data source: Synthetic simulation")
    print(
        "Real sensor input: NOT IMPLEMENTED"
    )
    print(
        "PLC / VFD RPM input: NOT IMPLEMENTED"
    )

    print()
    print(
        f"Sampling frequency : "
        f"{SAMPLING_FREQUENCY} Hz"
    )

    print(
        f"Window duration    : "
        f"{WINDOW_DURATION:.1f} s"
    )

    print(
        f"Number of windows  : "
        f"{NUMBER_OF_WINDOWS}"
    )

    print(
        f"Confidence threshold: "
        f"{CONFIDENCE_THRESHOLD:.2f}"
    )


    # ========================================================
    # LOAD ML MODEL
    # ========================================================

    predictor = BearingPredictor(
        confidence_threshold=CONFIDENCE_THRESHOLD
    )


    # ========================================================
    # MONITORING HISTORY
    # ========================================================

    history = []


    # ========================================================
    # TABLE HEADER
    # ========================================================

    print()
    print(
        f"{'Time':<10}"
        f"{'Machine':<10}"
        f"{'Name':<14}"
        f"{'RPM':<8}"
        f"{'Prediction':<16}"
        f"{'Confidence':<12}"
        f"{'Model':<10}"
        f"{'Confirmed':<12}"
        f"{'F':<4}"
        f"{'C':<4}"
        f"{'N':<4}"
    )

    print("-" * 100)


    # ========================================================
    # MONITORING LOOP
    # ========================================================

    for window_index in range(NUMBER_OF_WINDOWS):

        for machine_index, machine in enumerate(MACHINES):

            machine_id = machine["machine_id"]

            machine_name = machine["machine_name"]

            rpm = machine["rpm"]

            true_condition = (
                machine[
                    "condition_schedule"
                ][window_index]
            )


            # =================================================
            # SIGNAL SOURCE
            # =================================================

            #
            # IMPORTANT:
            #
            # monitor_bearing.py no longer calls
            # generate_synthetic_vibration_signal() directly.
            #
            # It now works through a SignalSource object.
            #
            # Later, SyntheticSignalSource can be replaced by
            # RealSensorSource without changing the downstream
            # diagnostic pipeline.
            #

            seed = (
                10_000
                + window_index * 100
                + machine_index
            )

            source = SyntheticSignalSource(
                fault_type=true_condition,
                rpm=rpm,
                sampling_frequency=SAMPLING_FREQUENCY,
                duration=WINDOW_DURATION,
                random_seed=seed,
            )


            # =================================================
            # ACQUIRE ONE MEASUREMENT WINDOW
            # =================================================

            measurement = source.read()


            # =================================================
            # FEATURE EXTRACTION
            # =================================================

            features = extract_vibration_features(
                signal=measurement.signal,
                rpm=measurement.rpm,
                sampling_frequency=(
                    measurement.sampling_frequency
                ),
            )


            # =================================================
            # MACHINE-LEARNING INFERENCE
            # =================================================

            result = predictor.predict(
                features
            )

            prediction = result[
                "predicted_class"
            ]

            confidence = result[
                "confidence"
            ]

            model_state = result[
                "condition_state"
            ]


            # =================================================
            # PERSISTENT STATE UPDATE
            # =================================================

            persistent_result = (
                update_persistent_state(
                    machine_id=machine_id,
                    model_state=model_state,
                )
            )

            confirmed_state = (
                persistent_result[
                    "state"
                ]
            )

            consecutive_fault = (
                persistent_result[
                    "consecutive_fault"
                ]
            )

            consecutive_check = (
                persistent_result[
                    "consecutive_check"
                ]
            )

            consecutive_normal = (
                persistent_result[
                    "consecutive_normal"
                ]
            )


            # =================================================
            # HISTORY RECORD
            # =================================================

            record = {
                "machine_id": machine_id,
                "machine_name": machine_name,
                "timestamp": measurement.timestamp,
                "rpm": measurement.rpm,
                "source_name": measurement.source_name,
                "true_condition": true_condition,
                "prediction": prediction,
                "confidence": confidence,
                "model_state": model_state,
                "state": confirmed_state,
                "consecutive_fault": consecutive_fault,
                "consecutive_check": consecutive_check,
                "consecutive_normal": consecutive_normal,
            }


            # =================================================
            # ADD ALL FEATURES
            # =================================================

            record.update(
                features
            )


            history.append(
                record
            )


            # =================================================
            # TERMINAL OUTPUT
            # =================================================

            time_text = (
                measurement.timestamp
                .strftime(
                    "%H:%M:%S"
                )
            )

            print(
                f"{time_text:<10}"
                f"{machine_id:<10}"
                f"{machine_name:<14}"
                f"{measurement.rpm:<8.0f}"
                f"{prediction:<16}"
                f"{confidence:<12.3f}"
                f"{model_state:<10}"
                f"{confirmed_state:<12}"
                f"{consecutive_fault:<4}"
                f"{consecutive_check:<4}"
                f"{consecutive_normal:<4}"
            )


        # ====================================================
        # DELAY BETWEEN MONITORING WINDOWS
        # ====================================================

        if (
            window_index
            < NUMBER_OF_WINDOWS - 1
        ):

            sleep(
                SLEEP_BETWEEN_WINDOWS
            )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history_df = pd.DataFrame(
        history
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )


    # ========================================================
    # FINAL STATUS
    # ========================================================

    print()
    print("=" * 100)
    print("LATEST MACHINE STATUS")
    print("=" * 100)

    latest_status = (
        history_df
        .sort_values(
            by="timestamp"
        )
        .groupby(
            "machine_id",
            as_index=False,
        )
        .tail(1)
        .sort_values(
            by="machine_id"
        )
    )


    for _, row in latest_status.iterrows():

        print(
            f"{row['machine_id']} "
            f"{row['machine_name']} | "
            f"prediction={row['prediction']} | "
            f"confidence={row['confidence']:.3f} | "
            f"model={row['model_state']} | "
            f"state={row['state']} | "
            f"F={int(row['consecutive_fault'])} "
            f"C={int(row['consecutive_check'])} "
            f"N={int(row['consecutive_normal'])}"
        )


    # ========================================================
    # HALL SUMMARY
    # ========================================================

    normal_count = int(
        (
            latest_status[
                "state"
            ]
            == "NORMAL"
        ).sum()
    )

    check_count = int(
        (
            latest_status[
                "state"
            ]
            == "CHECK"
        ).sum()
    )

    fault_count = int(
        (
            latest_status[
                "state"
            ]
            == "FAULT"
        ).sum()
    )


    print()
    print("=" * 100)
    print("CURRENT HALL STATE")
    print("=" * 100)

    print(
        f"NORMAL : {normal_count}"
    )

    print(
        f"CHECK  : {check_count}"
    )

    print(
        f"FAULT  : {fault_count}"
    )


    # ========================================================
    # CONFIGURATION SUMMARY
    # ========================================================

    print()
    print("=" * 100)
    print("PERSISTENCE CONFIGURATION")
    print("=" * 100)

    print(
        "FAULT confirmation : "
        f"{FAULT_CONFIRMATION_WINDOWS} windows"
    )

    print(
        "CHECK confirmation : "
        f"{CHECK_CONFIRMATION_WINDOWS} windows"
    )

    print(
        "NORMAL confirmation: "
        f"{NORMAL_CONFIRMATION_WINDOWS} windows"
    )

    print(
        "FAULT clear        : "
        f"{FAULT_CLEAR_WINDOWS} normal windows"
    )


    # ========================================================
    # DATA SOURCE SUMMARY
    # ========================================================

    print()
    print("=" * 100)
    print("DATA SOURCE")
    print("=" * 100)

    print(
        "Active source      : Synthetic simulation"
    )

    print(
        "Live sensor input  : Disabled / not implemented"
    )

    print(
        "PLC / VFD RPM      : Disabled / not implemented"
    )


    # ========================================================
    # OUTPUT FILE
    # ========================================================

    print()
    print(
        f"Monitoring history saved to:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    main()