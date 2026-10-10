from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Bearing Condition Monitoring",
    page_icon="⚙️",
    layout="wide",
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

HISTORY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bearing_monitoring_history.csv"
)


# ============================================================
# FRIENDLY LABELS
# ============================================================

FAULT_LABELS = {
    "healthy": "Normal operation",
    "outer_race": "Outer race fault",
    "inner_race": "Inner race fault",
    "ball_fault": "Rolling element fault",
    "cage_fault": "Bearing cage fault",
}


STATE_SYMBOLS = {
    "NORMAL": "🟢",
    "CHECK": "🟡",
    "FAULT": "🔴",
}


SEVERITY_ORDER = {
    "FAULT": 0,
    "CHECK": 1,
    "NORMAL": 2,
}


# ============================================================
# DATA LOADING
# ============================================================

def load_monitoring_data():

    if not HISTORY_FILE.exists():
        return None

    df = pd.read_csv(
        HISTORY_FILE
    )

    if df.empty:
        return None

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = (
        df
        .sort_values(
            by="timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# LATEST RECORD PER MACHINE
# ============================================================

def get_latest_machine_status(df):

    latest = (
        df
        .sort_values(
            by="timestamp"
        )
        .groupby(
            "machine_id",
            as_index=False,
        )
        .tail(1)
        .copy()
    )

    latest["severity_rank"] = (
        latest[
            "state"
        ]
        .map(
            SEVERITY_ORDER
        )
    )

    latest = (
        latest
        .sort_values(
            by=[
                "severity_rank",
                "machine_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return latest


# ============================================================
# PAGE TITLE
# ============================================================

st.title(
    "⚙️ Bearing Condition Monitoring"
)

st.caption(
    "Multi-machine vibration condition monitoring "
    "using signal processing and machine learning."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Navigation"
)

page = st.sidebar.radio(
    "View",
    [
        "Hall Overview",
        "Machine Details",
    ],
)


# ============================================================
# DATA SOURCE SECTION
# ============================================================

st.sidebar.divider()

st.sidebar.subheader(
    "Data source"
)

st.sidebar.radio(
    "Active source",
    [
        "Synthetic simulation",
    ],
    index=0,
)


st.sidebar.button(
    "Live sensor input — Coming soon",
    disabled=True,
    use_container_width=True,
)

st.sidebar.button(
    "PLC / VFD integration — Coming soon",
    disabled=True,
    use_container_width=True,
)

st.sidebar.caption(
    "Real sensor and PLC/VFD acquisition "
    "are planned but not implemented yet."
)


# ============================================================
# DASHBOARD
# ============================================================

@st.fragment(run_every="2s")
def dashboard():

    df = load_monitoring_data()

    if df is None:

        st.error(
            "Monitoring history is not available."
        )

        st.code(
            "uv run python src/monitor_bearing.py"
        )

        return


    latest_status = (
        get_latest_machine_status(
            df
        )
    )


    # ========================================================
    # SOURCE INFORMATION
    # ========================================================

    if "source_name" in df.columns:

        source_name = (
            latest_status[
                "source_name"
            ]
            .iloc[0]
        )

    else:

        source_name = (
            "Synthetic simulation"
        )


    # ========================================================
    # HALL OVERVIEW
    # ========================================================

    if page == "Hall Overview":

        st.header(
            "Hall Overview"
        )

        st.caption(
            f"Active data source: {source_name}"
        )


        # ----------------------------------------------------
        # COUNTS
        # ----------------------------------------------------

        total_machines = len(
            latest_status
        )

        normal_count = int(
            (
                latest_status["state"]
                == "NORMAL"
            ).sum()
        )

        check_count = int(
            (
                latest_status["state"]
                == "CHECK"
            ).sum()
        )

        fault_count = int(
            (
                latest_status["state"]
                == "FAULT"
            ).sum()
        )


        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        col1, col2, col3, col4 = (
            st.columns(
                4
            )
        )

        with col1:

            st.metric(
                "Machines monitored",
                total_machines,
            )

        with col2:

            st.metric(
                "🟢 Normal",
                normal_count,
            )

        with col3:

            st.metric(
                "🟡 Check",
                check_count,
            )

        with col4:

            st.metric(
                "🔴 Fault",
                fault_count,
            )


        st.divider()


        # ----------------------------------------------------
        # MACHINE CARDS
        # ----------------------------------------------------

        st.subheader(
            "Current Machine Status"
        )

        st.caption(
            "Machines requiring attention "
            "are shown first."
        )


        machine_columns = (
            st.columns(
                3
            )
        )


        for index, row in (
            latest_status.iterrows()
        ):

            column = (
                machine_columns[
                    index % 3
                ]
            )

            machine_id = row[
                "machine_id"
            ]

            machine_name = row[
                "machine_name"
            ]

            state = row[
                "state"
            ]

            prediction = row[
                "prediction"
            ]

            confidence = float(
                row[
                    "confidence"
                ]
            )

            rpm = float(
                row[
                    "rpm"
                ]
            )

            last_update = row[
                "timestamp"
            ]

            state_symbol = (
                STATE_SYMBOLS.get(
                    state,
                    "⚪",
                )
            )

            friendly_prediction = (
                FAULT_LABELS.get(
                    prediction,
                    prediction,
                )
            )


            if state == "NORMAL":

                condition_text = (
                    "Machine operating normally"
                )

            elif state == "CHECK":

                condition_text = (
                    "Condition requires verification"
                )

            else:

                condition_text = (
                    friendly_prediction
                )


            with column:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        f"### {state_symbol} "
                        f"{machine_id}"
                    )

                    st.markdown(
                        f"**{machine_name}**"
                    )

                    st.write(
                        f"Status: **{state}**"
                    )

                    st.write(
                        condition_text
                    )

                    st.write(
                        f"Confidence: "
                        f"**{confidence:.1%}**"
                    )

                    st.write(
                        f"Speed: "
                        f"**{rpm:.0f} RPM**"
                    )

                    if state == "FAULT":

                        st.caption(
                            "Inspection recommended"
                        )

                    elif state == "CHECK":

                        st.caption(
                            "Continue monitoring"
                        )

                    else:

                        st.caption(
                            "No action required"
                        )

                    st.caption(
                        "Last update: "
                        + last_update.strftime(
                            "%H:%M:%S"
                        )
                    )


        # ----------------------------------------------------
        # ATTENTION LIST
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Machines Requiring Attention"
        )

        attention_df = (
            latest_status[
                latest_status[
                    "state"
                ]
                .isin(
                    [
                        "FAULT",
                        "CHECK",
                    ]
                )
            ]
            .copy()
        )


        if attention_df.empty:

            st.success(
                "No machines currently "
                "require attention."
            )

        else:

            attention_df[
                "condition"
            ] = (
                attention_df[
                    "prediction"
                ]
                .map(
                    lambda value:
                        FAULT_LABELS.get(
                            value,
                            value,
                        )
                )
            )

            attention_df[
                "confidence_display"
            ] = (
                attention_df[
                    "confidence"
                ]
                .map(
                    lambda value:
                        f"{value:.1%}"
                )
            )

            attention_display = (
                attention_df[
                    [
                        "machine_id",
                        "machine_name",
                        "state",
                        "condition",
                        "confidence_display",
                        "rpm",
                    ]
                ]
                .rename(
                    columns={
                        "machine_id":
                            "Machine",

                        "machine_name":
                            "Equipment",

                        "state":
                            "Status",

                        "condition":
                            "Condition",

                        "confidence_display":
                            "Confidence",

                        "rpm":
                            "RPM",
                    }
                )
            )

            st.dataframe(
                attention_display,
                use_container_width=True,
                hide_index=True,
            )


        # ----------------------------------------------------
        # STATE DISTRIBUTION
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Current Hall State Distribution"
        )

        state_counts = (
            latest_status[
                "state"
            ]
            .value_counts()
            .rename_axis(
                "state"
            )
            .reset_index(
                name="count"
            )
        )

        state_chart = (
            alt.Chart(
                state_counts
            )
            .mark_bar()
            .encode(
                x=alt.X(
                    "state:N",
                    title="Machine state",
                    sort=[
                        "FAULT",
                        "CHECK",
                        "NORMAL",
                    ],
                ),
                y=alt.Y(
                    "count:Q",
                    title="Machines",
                ),
                tooltip=[
                    alt.Tooltip(
                        "state:N",
                        title="State",
                    ),
                    alt.Tooltip(
                        "count:Q",
                        title="Machines",
                    ),
                ],
            )
            .properties(
                height=300
            )
        )

        st.altair_chart(
            state_chart,
            use_container_width=True,
        )


    # ========================================================
    # MACHINE DETAILS
    # ========================================================

    elif page == "Machine Details":

        st.header(
            "Machine Details"
        )

        st.caption(
            f"Active data source: {source_name}"
        )


        # ----------------------------------------------------
        # MACHINE SELECTOR
        # ----------------------------------------------------

        machine_options = (
            latest_status[
                [
                    "machine_id",
                    "machine_name",
                ]
            ]
            .drop_duplicates()
            .sort_values(
                by="machine_id"
            )
        )

        machine_labels = {

            row[
                "machine_id"
            ]:
            (
                f"{row['machine_id']} - "
                f"{row['machine_name']}"
            )

            for _, row
            in machine_options.iterrows()
        }


        selected_machine = (
            st.selectbox(
                "Select machine",
                options=list(
                    machine_labels.keys()
                ),
                format_func=lambda machine_id:
                    machine_labels[
                        machine_id
                    ],
            )
        )


        # ----------------------------------------------------
        # MACHINE DATA
        # ----------------------------------------------------

        machine_df = (
            df[
                df[
                    "machine_id"
                ]
                == selected_machine
            ]
            .copy()
            .sort_values(
                by="timestamp"
            )
        )

        latest = (
            machine_df.iloc[-1]
        )


        # ----------------------------------------------------
        # CURRENT VALUES
        # ----------------------------------------------------

        state = latest[
            "state"
        ]

        model_state = latest[
            "model_state"
        ]

        prediction = latest[
            "prediction"
        ]

        confidence = float(
            latest[
                "confidence"
            ]
        )

        rpm = float(
            latest[
                "rpm"
            ]
        )

        machine_name = latest[
            "machine_name"
        ]

        latest_timestamp = latest[
            "timestamp"
        ]

        friendly_prediction = (
            FAULT_LABELS.get(
                prediction,
                prediction,
            )
        )

        state_symbol = (
            STATE_SYMBOLS.get(
                state,
                "⚪",
            )
        )


        # ----------------------------------------------------
        # MACHINE TITLE
        # ----------------------------------------------------

        st.subheader(
            f"{selected_machine} - "
            f"{machine_name}"
        )

        st.caption(
            "Last measurement: "
            + latest_timestamp.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )


        # ----------------------------------------------------
        # OPERATIONAL STATUS
        # ----------------------------------------------------

        st.markdown(
            "### Operational Status"
        )

        status1, status2, status3 = (
            st.columns(
                3
            )
        )


        with status1:

            st.metric(
                "Confirmed state",
                f"{state_symbol} {state}",
            )


        with status2:

            st.metric(
                "Confidence",
                f"{confidence:.1%}",
            )


        with status3:

            st.metric(
                "Machine speed",
                f"{rpm:.0f} RPM",
            )


        # ----------------------------------------------------
        # DETECTED CONDITION
        # ----------------------------------------------------

        st.markdown(
            "#### Detected condition"
        )

        st.write(
            f"**{friendly_prediction}**"
        )


        # ----------------------------------------------------
        # OPERATOR MESSAGE
        # ----------------------------------------------------

        if state == "NORMAL":

            st.success(
                "Machine condition is stable. "
                "No immediate action is required."
            )

        elif state == "CHECK":

            st.warning(
                "The monitoring system has detected "
                "an uncertain or developing condition. "
                "Continue monitoring and verify the machine "
                "if the condition persists."
            )

        else:

            st.error(
                f"Confirmed fault condition: "
                f"{friendly_prediction}. "
                f"Inspection is recommended."
            )


        # ----------------------------------------------------
        # LATEST ANALYSIS
        # ----------------------------------------------------

        st.markdown(
            "### Latest Analysis"
        )

        reading1, reading2, reading3 = (
            st.columns(
                3
            )
        )


        with reading1:

            st.metric(
                "Current reading",
                model_state,
            )


        with reading2:

            if model_state == state:

                confirmation_text = (
                    "Confirmed"
                )

            else:

                confirmation_text = (
                    "Awaiting confirmation"
                )

            st.metric(
                "Reading status",
                confirmation_text,
            )


        with reading3:

            st.metric(
                "Model confidence",
                f"{confidence:.1%}",
            )


        st.caption(
            "The current reading represents one measurement window. "
            "The confirmed operational state uses multiple "
            "consecutive measurements to reduce false alarms."
        )


        # ----------------------------------------------------
        # CONFIDENCE TREND
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Condition Confidence Trend"
        )

        confidence_chart = (
            alt.Chart(
                machine_df
            )
            .mark_line(
                point=True
            )
            .encode(
                x=alt.X(
                    "timestamp:T",
                    title="Time",
                    axis=alt.Axis(
                        format="%H:%M:%S"
                    ),
                ),
                y=alt.Y(
                    "confidence:Q",
                    title="Confidence",
                    scale=alt.Scale(
                        domain=[
                            0,
                            1,
                        ]
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "timestamp:T",
                        title="Time",
                        format="%H:%M:%S",
                    ),
                    alt.Tooltip(
                        "prediction:N",
                        title="Detected condition",
                    ),
                    alt.Tooltip(
                        "confidence:Q",
                        title="Confidence",
                        format=".3f",
                    ),
                    alt.Tooltip(
                        "model_state:N",
                        title="Current reading",
                    ),
                    alt.Tooltip(
                        "state:N",
                        title="Confirmed state",
                    ),
                ],
            )
            .properties(
                height=300
            )
        )

        st.altair_chart(
            confidence_chart,
            use_container_width=True,
        )


        # ----------------------------------------------------
        # VIBRATION TREND
        # ----------------------------------------------------

        st.subheader(
            "Vibration Level Trend"
        )

        rms_chart = (
            alt.Chart(
                machine_df
            )
            .mark_line(
                point=True
            )
            .encode(
                x=alt.X(
                    "timestamp:T",
                    title="Time",
                    axis=alt.Axis(
                        format="%H:%M:%S"
                    ),
                ),
                y=alt.Y(
                    "rms:Q",
                    title="Vibration level",
                    scale=alt.Scale(
                        zero=False
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "timestamp:T",
                        title="Time",
                        format="%H:%M:%S",
                    ),
                    alt.Tooltip(
                        "rms:Q",
                        title="Vibration level",
                        format=".6f",
                    ),
                ],
            )
            .properties(
                height=300
            )
        )

        st.altair_chart(
            rms_chart,
            use_container_width=True,
        )


        # ----------------------------------------------------
        # OPERATOR HISTORY
        # ----------------------------------------------------

        st.subheader(
            "Recent Monitoring History"
        )

        display_df = (
            machine_df[
                [
                    "timestamp",
                    "prediction",
                    "confidence",
                    "model_state",
                    "state",
                    "rpm",
                ]
            ]
            .copy()
        )

        display_df[
            "timestamp"
        ] = (
            display_df[
                "timestamp"
            ]
            .dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        display_df[
            "prediction"
        ] = (
            display_df[
                "prediction"
            ]
            .map(
                lambda value:
                    FAULT_LABELS.get(
                        value,
                        value,
                    )
            )
        )

        display_df[
            "confidence"
        ] = (
            display_df[
                "confidence"
            ]
            .map(
                lambda value:
                    f"{value:.1%}"
            )
        )

        display_df = (
            display_df
            .rename(
                columns={
                    "timestamp":
                        "Time",

                    "prediction":
                        "Detected condition",

                    "confidence":
                        "Confidence",

                    "model_state":
                        "Current reading",

                    "state":
                        "Confirmed state",

                    "rpm":
                        "RPM",
                }
            )
            .sort_index(
                ascending=False
            )
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )


        # ----------------------------------------------------
        # ADVANCED DIAGNOSTICS
        # ----------------------------------------------------

        with st.expander(
            "Advanced diagnostics"
        ):

            st.caption(
                "Technical vibration information intended "
                "for maintenance and engineering personnel."
            )


            # -----------------------------------------------
            # DATA SOURCE
            # -----------------------------------------------

            st.markdown(
                "#### Acquisition information"
            )

            if "source_name" in latest.index:

                st.write(
                    "Signal source: "
                    f"**{latest['source_name']}**"
                )

            else:

                st.write(
                    "Signal source: "
                    "**Synthetic simulation**"
                )

            st.write(
                f"Sampling frequency: "
                f"**5000 Hz**"
            )

            st.write(
                f"Shaft speed: "
                f"**{rpm:.0f} RPM**"
            )

            st.caption(
                "In a real deployment, RPM should preferably "
                "be acquired automatically from the machine PLC "
                "or VFD rather than entered manually."
            )


            st.divider()


            # -----------------------------------------------
            # FAULT INDICATORS
            # -----------------------------------------------

            st.markdown(
                "#### Bearing fault indicators"
            )

            diag1, diag2, diag3, diag4 = (
                st.columns(
                    4
                )
            )


            with diag1:

                st.metric(
                    "Outer race",
                    f"{latest['bpfo_envelope']:.6f}",
                )

                st.caption(
                    "BPFO envelope"
                )


            with diag2:

                st.metric(
                    "Inner race",
                    f"{latest['bpfi_envelope']:.6f}",
                )

                st.caption(
                    "BPFI envelope"
                )


            with diag3:

                st.metric(
                    "Rolling element",
                    f"{latest['bsf_envelope']:.6f}",
                )

                st.caption(
                    "BSF envelope"
                )


            with diag4:

                st.metric(
                    "Bearing cage",
                    f"{latest['ftf_raw']:.6f}",
                )

                st.caption(
                    "FTF raw"
                )


            st.divider()


            # -----------------------------------------------
            # SIGNAL INDICATORS
            # -----------------------------------------------

            st.markdown(
                "#### Signal indicators"
            )

            adv1, adv2, adv3 = (
                st.columns(
                    3
                )
            )


            with adv1:

                st.metric(
                    "RMS",
                    f"{latest['rms']:.6f}",
                )


            with adv2:

                st.metric(
                    "Crest Factor",
                    f"{latest['crest_factor']:.6f}",
                )


            with adv3:

                st.metric(
                    "Kurtosis",
                    f"{latest['kurtosis']:.6f}",
                )


            st.divider()


            # -----------------------------------------------
            # PERSISTENCE DIAGNOSTICS
            # -----------------------------------------------

            st.markdown(
                "#### Alarm persistence"
            )

            persistence1, persistence2, persistence3 = (
                st.columns(
                    3
                )
            )


            with persistence1:

                st.metric(
                    "Consecutive fault windows",
                    int(
                        latest[
                            "consecutive_fault"
                        ]
                    ),
                )


            with persistence2:

                st.metric(
                    "Consecutive check windows",
                    int(
                        latest[
                            "consecutive_check"
                        ]
                    ),
                )


            with persistence3:

                st.metric(
                    "Consecutive normal windows",
                    int(
                        latest[
                            "consecutive_normal"
                        ]
                    ),
                )


            st.caption(
                "Current configuration: "
                "FAULT confirmation = 3 consecutive windows, "
                "CHECK confirmation = 2 consecutive windows, "
                "NORMAL confirmation = 2 consecutive windows."
            )


            st.divider()


            # -----------------------------------------------
            # COMPLETE TECHNICAL HISTORY
            # -----------------------------------------------

            st.markdown(
                "#### Technical Feature History"
            )

            technical_columns = [
                "timestamp",
                "rpm",
                "rms",
                "peak",
                "peak_to_peak",
                "crest_factor",
                "kurtosis",
                "shaft_amplitude",
                "bpfo_raw",
                "bpfi_raw",
                "bsf_raw",
                "ftf_raw",
                "bpfo_envelope",
                "bpfi_envelope",
                "bsf_envelope",
                "ftf_envelope",
            ]

            technical_df = (
                machine_df[
                    technical_columns
                ]
                .copy()
            )

            technical_df[
                "timestamp"
            ] = (
                technical_df[
                    "timestamp"
                ]
                .dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

            technical_df = (
                technical_df
                .sort_index(
                    ascending=False
                )
            )

            st.dataframe(
                technical_df,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# RUN DASHBOARD
# ============================================================

dashboard()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Synthetic proof-of-concept. "
    "The machine-learning model and alarm logic "
    "have not been validated for industrial diagnostic use."
)