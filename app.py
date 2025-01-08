import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# GARCH
from arch import arch_model

# --------------------------------------------------------------
# Helper function: Plot ACF & PACF side by side
# --------------------------------------------------------------
def plot_acf_pacf(series, lags=40, title_prefix=""):
    """Plots ACF and PACF side by side for a given series."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_acf(series, lags=lags, ax=axes[0], title=f"{title_prefix} ACF")
    plot_pacf(series, lags=lags, ax=axes[1], title=f"{title_prefix} PACF")
    plt.tight_layout()
    return fig

# --------------------------------------------------------------
# Initialize Session State for storing forecasts, GARCH, and charts
# --------------------------------------------------------------
if "manual_forecast" not in st.session_state:
    st.session_state["manual_forecast"] = None
if "auto_forecast" not in st.session_state:
    st.session_state["auto_forecast"] = None

# We'll store model residuals for GARCH
if "manual_resid" not in st.session_state:
    st.session_state["manual_resid"] = None
if "auto_resid" not in st.session_state:
    st.session_state["auto_resid"] = None

# GARCH volatility forecasts
if "manual_garch_std" not in st.session_state:
    st.session_state["manual_garch_std"] = None
if "auto_garch_std" not in st.session_state:
    st.session_state["auto_garch_std"] = None

# We'll store all matplotlib figures to keep them displayed
if "charts" not in st.session_state:
    st.session_state["charts"] = []

# --------------------------------------------------------------
# Streamlit App Title
# --------------------------------------------------------------
st.set_page_config(layout="wide")
st.title("Housing Forecasting Time Series: Compare SARIMAX (Manual vs. Auto-ARIMA) + GARCH")

# --------------------------------------------------------------
# 1. File Upload or Default data.csv
# --------------------------------------------------------------
uploaded_file = st.file_uploader("Upload your housing time-series CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Using your uploaded file.")
else:
    if os.path.exists("data.csv"):
        st.write("### No file uploaded. Loading 'data.csv' as an example dataset.")
        df = pd.read_csv("data.csv")
    else:
        st.error("No file uploaded AND no 'data.csv' found!")
        st.stop()

st.write("### Data Preview")
st.dataframe(df.head())

# --------------------------------------------------------------
# 2. Date/Datetime Column
# --------------------------------------------------------------
date_column = st.selectbox("Select your datetime column:", df.columns)
df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
df.set_index(date_column, inplace=True)
df.sort_index(inplace=True)

# --------------------------------------------------------------
# 3. Target Column
# --------------------------------------------------------------
all_columns = list(df.columns)
target_column = st.selectbox("Select the target (housing price) column:", all_columns, index=0)

# --------------------------------------------------------------
# 4. Clean Target Column
# --------------------------------------------------------------
df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
df.dropna(subset=[target_column], inplace=True)

# --------------------------------------------------------------
# 5. User-Ranked Exogenous Features
# --------------------------------------------------------------
feature_candidates = [c for c in all_columns if c != target_column]
st.write("### Rank Exogenous Features by Importance")
ranked_features = []
remaining_features = feature_candidates.copy()

while remaining_features:
    next_feat = st.selectbox(
        "Pick next feature in rank order (or skip if done):",
        ["-- skip --"] + remaining_features,
        key=f"rank_{len(ranked_features)}"
    )
    if next_feat == "-- skip --":
        break
    ranked_features.append(next_feat)
    remaining_features.remove(next_feat)

st.write("**Your ranked features** (in order):", ranked_features)
exog_df = df[ranked_features].copy() if ranked_features else pd.DataFrame(index=df.index)

# --------------------------------------------------------------
# 6. Optional Scaling of Exogenous Features
# --------------------------------------------------------------
do_scale = st.checkbox("Scale exogenous features to [0,1]?", value=True)
if do_scale and not exog_df.empty:
    scaler = MinMaxScaler()
    scaled_vals = scaler.fit_transform(exog_df)
    exog_df = pd.DataFrame(scaled_vals, index=exog_df.index, columns=exog_df.columns)

# The main target series
y = df[target_column].astype(float)

# --------------------------------------------------------------
# 7. Train-Test Split
# --------------------------------------------------------------
st.write("### Train-Test Split")
split_ratio = st.slider("Train/Test Split (as % for training)", 50, 95, 80)
train_size = int(len(y) * (split_ratio / 100.0))

train_y = y.iloc[:train_size]
test_y  = y.iloc[train_size:]

train_exog = exog_df.iloc[:train_size] if not exog_df.empty else None
test_exog  = exog_df.iloc[train_size:] if not exog_df.empty else None

st.write(f"Training samples: {len(train_y)}, Test samples: {len(test_y)}")

# --------------------------------------------------------------
# 8. ACF/PACF (Raw or Differenced)
# --------------------------------------------------------------
st.write("## ACF/PACF Plots for Housing Price (Training Data)")
col1, col2 = st.columns(2)

with col1:
    if st.button("Plot ACF & PACF (RAW Training Target)"):
        fig_raw = plot_acf_pacf(train_y, lags=40, title_prefix="Target (Train, raw)")
        st.pyplot(fig_raw)
        st.session_state["charts"].append(fig_raw)

with col2:
    diff_order = st.slider("Difference order for plotting (0-2):", 0, 2, 1)
    if st.button("Plot ACF & PACF (DIFF Training Target)"):
        diff_series = train_y.diff(diff_order).dropna()
        fig_diff = plot_acf_pacf(diff_series, lags=40,
                                 title_prefix=f"Train Target (diff={diff_order})")
        st.pyplot(fig_diff)
        st.session_state["charts"].append(fig_diff)

# --------------------------------------------------------------
# 9. Manual SARIMAX
# --------------------------------------------------------------
st.write("## Manual SARIMAX Parameters")

p_ = st.number_input("p (AR order)", min_value=0, max_value=10, value=1)
d_ = st.number_input("d (Differencing)", min_value=0, max_value=3, value=1)
q_ = st.number_input("q (MA order)", min_value=0, max_value=10, value=1)

P_ = st.number_input("P (Seasonal AR)", min_value=0, max_value=10, value=0)
D_ = st.number_input("D (Seasonal diff)", min_value=0, max_value=3, value=0)
Q_ = st.number_input("Q (Seasonal MA)", min_value=0, max_value=10, value=0)
m_ = st.number_input("m (Seasonal period)", min_value=0, max_value=52, value=0)

if st.button("Fit Manual SARIMAX Model"):
    with st.spinner("Fitting manual SARIMAX..."):
        try:
            order_manual = (p_, d_, q_)
            seasonal_order_manual = (P_, D_, Q_, m_)
            manual_model = SARIMAX(
                endog=train_y,
                exog=train_exog,
                order=order_manual,
                seasonal_order=seasonal_order_manual,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            manual_results = manual_model.fit(disp=False)
            st.write("### Manual SARIMAX Model Summary")
            st.write(manual_results.summary())

            # Forecast
            horizon = len(test_y)
            if horizon < 1:
                st.warning("Not enough test data to forecast!")
            else:
                manual_fc = manual_results.get_forecast(steps=horizon, exog=test_exog)
                manual_pred = manual_fc.predicted_mean

                # Store forecast in session_state
                st.session_state["manual_forecast"] = manual_pred

                # Store residuals for GARCH
                st.session_state["manual_resid"] = manual_results.resid.dropna()

                st.success("Manual SARIMAX forecast + residuals stored!")

            # ACF/PACF of residuals
            if st.checkbox("Plot ACF & PACF (Manual SARIMAX Residuals)"):
                residuals_man = manual_results.resid.dropna()
                if len(residuals_man) < 2:
                    st.warning("Not enough residual data to plot.")
                else:
                    fig_mres = plot_acf_pacf(residuals_man, lags=40,
                                             title_prefix="Manual SARIMAX Residuals")
                    st.pyplot(fig_mres)
                    st.session_state["charts"].append(fig_mres)

        except Exception as e:
            st.error(f"Error fitting manual SARIMAX: {e}")

# --------------------------------------------------------------
# 10. Auto-ARIMA SARIMAX
# --------------------------------------------------------------
st.write("## Auto-ARIMA SARIMAX")

if st.button("Run Auto-ARIMA & Fit SARIMAX"):
    with st.spinner("Running auto_arima..."):
        try:
            auto_model = auto_arima(
                y=train_y,
                exogenous=train_exog,
                start_p=0, start_q=0,
                max_p=5, max_q=5,
                seasonal=True,
                m=12,  # adjust if known
                start_P=0, start_Q=0,
                max_P=5, max_Q=5,
                d=None, D=None,
                trace=True,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True,
                information_criterion='aic'
            )
            st.success("Auto-ARIMA found orders:")
            st.write(f"Order (p,d,q): {auto_model.order}")
            st.write(f"Seasonal order (P,D,Q,m): {auto_model.seasonal_order}")

            order_auto = auto_model.order
            seasonal_auto = auto_model.seasonal_order

            sarimax_model = SARIMAX(
                endog=train_y,
                exog=train_exog,
                order=order_auto,
                seasonal_order=seasonal_auto,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            sarimax_results = sarimax_model.fit(disp=False)

            st.write("### Auto-ARIMA SARIMAX Model Summary")
            st.write(sarimax_results.summary())

            horizon = len(test_y)
            if horizon < 1:
                st.warning("Not enough test data to forecast!")
            else:
                forecast_res = sarimax_results.get_forecast(steps=horizon, exog=test_exog)
                pred_mean = forecast_res.predicted_mean
                st.session_state["auto_forecast"] = pred_mean

                # Store residuals for GARCH
                st.session_state["auto_resid"] = sarimax_results.resid.dropna()

                st.success("Auto-ARIMA SARIMAX forecast + residuals stored!")

            # ACF/PACF of residuals
            if st.checkbox("Plot ACF & PACF (Auto-ARIMA SARIMAX Residuals)"):
                res_auto = sarimax_results.resid.dropna()
                if len(res_auto) < 2:
                    st.warning("Not enough residual data to plot.")
                else:
                    fig_ares = plot_acf_pacf(res_auto, lags=40,
                                             title_prefix="Auto-ARIMA SARIMAX Residuals")
                    st.pyplot(fig_ares)
                    st.session_state["charts"].append(fig_ares)

        except Exception as e:
            st.error(f"Error in Auto-ARIMA or SARIMAX: {e}")

# --------------------------------------------------------------
# 11. GARCH on SARIMAX Residuals
# --------------------------------------------------------------
st.write("## Optional: GARCH on SARIMAX Residuals")

resid_choice = st.radio(
    "Which SARIMAX residuals do you want to fit GARCH on?",
    ("None", "Manual SARIMAX Residuals", "Auto-ARIMA SARIMAX Residuals")
)

p_g = st.number_input("GARCH p", min_value=1, max_value=5, value=1)
q_g = st.number_input("GARCH q", min_value=1, max_value=5, value=1)
dist_g = st.selectbox("Distribution:", ["normal", "t", "studentst"], index=0)

if st.button("Fit GARCH on Selected Residuals"):
    if resid_choice == "None":
        st.warning("No residuals selected!")
    else:
        with st.spinner("Fitting GARCH..."):
            try:
                if resid_choice == "Manual SARIMAX Residuals":
                    if st.session_state["manual_resid"] is None or len(st.session_state["manual_resid"]) < 2:
                        st.error("No valid residuals from Manual SARIMAX. Fit the model first!")
                        st.stop()
                    chosen_resid = st.session_state["manual_resid"]
                else:  # "Auto-ARIMA SARIMAX Residuals"
                    if st.session_state["auto_resid"] is None or len(st.session_state["auto_resid"]) < 2:
                        st.error("No valid residuals from Auto-ARIMA SARIMAX. Fit the model first!")
                        st.stop()
                    chosen_resid = st.session_state["auto_resid"]

                garch_model = arch_model(
                    chosen_resid, vol='Garch', p=p_g, q=q_g, dist=dist_g
                )
                garch_res = garch_model.fit(disp='off')
                st.write("### GARCH Summary")
                st.write(garch_res.summary())

                # We'll do a naive multi-step GARCH forecast for test period length
                horizon = len(test_y)
                if horizon < 1:
                    st.warning("Not enough test data to forecast GARCH!")
                else:
                    garch_forecast = garch_res.forecast(horizon=horizon)
                    # arch_model can be tricky with indexing. We'll get last row of variance forecast.
                    var_df = garch_forecast.variance
                    last_idx = var_df.index[-1]
                    # This should be the variance forecast for out-of-sample steps
                    forecasted_var = var_df.loc[last_idx].values  # array of length `horizon`
                    forecasted_std = np.sqrt(forecasted_var)

                    if resid_choice == "Manual SARIMAX Residuals":
                        st.session_state["manual_garch_std"] = forecasted_std
                    else:
                        st.session_state["auto_garch_std"] = forecasted_std

                    st.success("GARCH volatility (std dev) forecast stored!")
                    st.write("First few volatility values:", forecasted_std[:5])

            except Exception as e:
                st.error(f"Error fitting GARCH: {e}")

# --------------------------------------------------------------
# 12. Final Comparison Chart: Actual, Manual SARIMAX, Auto SARIMAX, ±2 GARCH
# --------------------------------------------------------------
st.write("## Compare Forecasts & (Optional) GARCH Volatility")

if st.button("Plot Final Comparison Chart"):
    if len(test_y) < 1:
        st.warning("No test data to plot.")
    else:
        results_df = pd.DataFrame({"Actual": test_y}, index=test_y.index)

        # Manual forecast
        if st.session_state["manual_forecast"] is not None:
            man_pred = st.session_state["manual_forecast"].reindex(results_df.index, fill_value=np.nan)
            results_df["ManualSARIMAX"] = man_pred

        # Auto forecast
        if st.session_state["auto_forecast"] is not None:
            auto_pred = st.session_state["auto_forecast"].reindex(results_df.index, fill_value=np.nan)
            results_df["AutoARIMA_SARIMAX"] = auto_pred

        st.write("### Forecast Results (Test Set)")
        st.dataframe(results_df.head(10))

        # We'll let the user choose which forecast to apply GARCH band to
        forecast_choice = st.radio(
            "Apply GARCH volatility to which forecast?",
            ("None", "ManualSARIMAX", "AutoARIMA_SARIMAX")
        )

        # Build the plot
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(results_df.index, results_df["Actual"], label="Actual", color="blue")

        if "ManualSARIMAX" in results_df.columns:
            ax.plot(results_df.index, results_df["ManualSARIMAX"], label="Manual SARIMAX", color="green")

        if "AutoARIMA_SARIMAX" in results_df.columns:
            ax.plot(results_df.index, results_df["AutoARIMA_SARIMAX"], label="Auto-ARIMA SARIMAX", color="red")

        # Check GARCH
        if forecast_choice == "ManualSARIMAX":
            if st.session_state["manual_garch_std"] is not None and "ManualSARIMAX" in results_df.columns:
                std_array = st.session_state["manual_garch_std"]
                if len(std_array) == len(results_df):
                    upper = results_df["ManualSARIMAX"] + 2 * std_array
                    lower = results_df["ManualSARIMAX"] - 2 * std_array
                    ax.fill_between(results_df.index, lower, upper, color='gray', alpha=0.2,
                                    label="±2 Std (Manual GARCH)")
                else:
                    st.warning("GARCH std length does not match test horizon.")
            else:
                st.warning("No GARCH std for Manual SARIMAX is available. Fit GARCH first.")
        elif forecast_choice == "AutoARIMA_SARIMAX":
            if st.session_state["auto_garch_std"] is not None and "AutoARIMA_SARIMAX" in results_df.columns:
                std_array = st.session_state["auto_garch_std"]
                if len(std_array) == len(results_df):
                    upper = results_df["AutoARIMA_SARIMAX"] + 2 * std_array
                    lower = results_df["AutoARIMA_SARIMAX"] - 2 * std_array
                    ax.fill_between(results_df.index, lower, upper, color='gray', alpha=0.2,
                                    label="±2 Std (Auto GARCH)")
                else:
                    st.warning("GARCH std length does not match test horizon.")
            else:
                st.warning("No GARCH std for Auto-ARIMA SARIMAX is available. Fit GARCH first.")

        ax.set_title("Housing Forecast: Actual vs. Manual SARIMAX vs. Auto-ARIMA + (Optional GARCH Band)")
        ax.legend()
        st.pyplot(fig)
        st.session_state["charts"].append(fig)

# --------------------------------------------------------------
# 13. Show All Collected Charts
# --------------------------------------------------------------
st.write("## All Charts So Far")
for i, figure in enumerate(st.session_state["charts"]):
    st.write(f"**Chart {i+1}**")
    st.pyplot(figure)
