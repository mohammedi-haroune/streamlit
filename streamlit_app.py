from typing import Callable
import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt
from relife.datasets import (
    load_circuit_breaker,
    load_insulator_string,
    load_power_transformer,
    LifetimeData,
)

from relife import (
    KaplanMeier,
    Weibull,
    Gompertz,
    Exponential,
    Gamma,
    LogLogistic,
)

st.header("ReLife App")
st.write("This app provides an easy to use interface to use ReLife package by RTE")


st.subheader("Input Data")

dataset_name = st.selectbox(
    "Dataset", options=["Circuit Breaker", "Power Transformer", "Insulator String"]
)

dataset_name_to_func: dict[str, Callable[[], LifetimeData]] = {
    "Circuit Breaker": load_circuit_breaker,
    "Power Transformer": load_power_transformer,
    "Insulator String": load_insulator_string,
}

dataset_func = dataset_name_to_func[dataset_name]

time, event, entry, *args = dataset_func().astuple()

df = pd.DataFrame({"time": time, "event": event, "entry": entry})

st.dataframe(df)


st.subheader("Survival Analysis")


strategy_name_to_class = {
    "KaplanMeier": KaplanMeier,
    "Weibull": Weibull,
    "Gompertz": Gompertz,
    "Exponential": Exponential,
    "Gamma": Gamma,
    "LogLogistic": LogLogistic,
}

strategies = st.multiselect(
    label="Survival strategy",
    options=[
        "KaplanMeier",
        "Weibull",
        "Gompertz",
        "Exponential",
        "Gamma",
        "LogLogistic",
    ],
    default=["KaplanMeier", "Weibull", "Gompertz"],
)

if not strategies:
    st.warning("Please choose at least one strategy")

else:
    fig = plt.figure()
    plt.xlabel("Age [year]")
    plt.ylabel("Survival probability")

    for strategy_name in strategies:
        strategy_class = strategy_name_to_class[strategy_name]
        strategy_class().fit(time, event, entry).plot()

    st.pyplot(fig)
