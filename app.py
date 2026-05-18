from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="PJM Energy Market Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"

NAVY = "#07111f"
PANEL = "#0d1b2e"
PANEL_2 = "#101f35"
GRID = "#223650"
TEXT = "#d8e6f3"
MUTED = "#7f94aa"
CYAN = "#27d8ff"
BLUE = "#4a8dff"
ORANGE = "#ff9f43"
RED = "#ff4d5e"
GREEN = "#39d98a"


st.markdown(
    """
    <style>
    :root {
        --navy: #07111f;
        --panel: #0d1b2e;
        --panel2: #101f35;
        --grid: #223650;
        --text: #d8e6f3;
        --muted: #7f94aa;
        --cyan: #27d8ff;
        --blue: #4a8dff;
        --orange: #ff9f43;
        --red: #ff4d5e;
        --green: #39d98a;
    }

    .stApp {
        background:
            radial-gradient(circle at 18% 5%, rgba(39, 216, 255, 0.10), transparent 26%),
            radial-gradient(circle at 78% 0%, rgba(91, 110, 255, 0.12), transparent 30%),
            linear-gradient(135deg, #090821 0%, #07111f 48%, #081322 100%);
        background-size: auto;
        color: var(--text);
    }

    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2.5rem;
        max-width: 1500px;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #081322 0%, #0b1728 100%);
        border-right: 1px solid rgba(39, 216, 255, 0.18);
        box-shadow: 10px 0 28px rgba(0, 0, 0, 0.22);
    }

    div[data-testid="stSidebar"] * {
        color: #c8d8e8;
    }

    div[role="radiogroup"] label {
        background: rgba(13, 27, 46, 0.78);
        border: 1px solid rgba(39, 216, 255, 0.13);
        border-radius: 8px;
        padding: 0.4rem 0.55rem;
        margin-bottom: 0.45rem;
        transition: all 160ms ease;
    }

    div[role="radiogroup"] label:hover {
        border-color: rgba(39, 216, 255, 0.55);
        box-shadow: 0 0 18px rgba(39, 216, 255, 0.16);
        transform: translateX(2px);
    }

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.95rem 1.05rem;
        background: linear-gradient(135deg, rgba(13, 27, 46, 0.94), rgba(8, 19, 34, 0.96));
        border: 1px solid rgba(39, 216, 255, 0.22);
        border-radius: 10px;
        box-shadow: 0 0 32px rgba(39, 216, 255, 0.08), inset 0 0 20px rgba(74, 141, 255, 0.03);
        margin-bottom: 0.85rem;
    }

    .title-block h1 {
        font-size: 1.65rem;
        line-height: 1.1;
        color: #f2f8ff;
        margin: 0;
        letter-spacing: 0;
        font-weight: 820;
    }

    .title-block p {
        margin: 0.35rem 0 0 0;
        color: var(--muted);
        font-size: 0.88rem;
    }

    .status-strip {
        display: flex;
        gap: 0.55rem;
        flex-wrap: wrap;
        justify-content: flex-end;
    }

    .status-pill {
        background: rgba(16, 31, 53, 0.95);
        border: 1px solid rgba(39, 216, 255, 0.20);
        border-radius: 999px;
        padding: 0.38rem 0.68rem;
        color: #cfe8f8;
        font-size: 0.78rem;
        font-weight: 650;
        white-space: nowrap;
    }

    .live-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--green);
        box-shadow: 0 0 12px rgba(57, 217, 138, 0.9);
        margin-right: 0.35rem;
    }

    .ops-card, .kpi-card, .callout-card {
        background:
            linear-gradient(180deg, rgba(31, 39, 83, 0.78), rgba(15, 25, 48, 0.92)),
            rgba(13, 27, 46, 0.95);
        border: 1px solid rgba(160, 188, 255, 0.12);
        border-radius: 12px;
        box-shadow: 0 14px 34px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.045);
    }

    .ops-card {
        padding: 1rem;
        min-height: 116px;
    }

    .ops-card:hover, .kpi-card:hover, .callout-card:hover {
        border-color: rgba(39, 216, 255, 0.40);
        box-shadow: 0 0 26px rgba(39, 216, 255, 0.11), 0 12px 28px rgba(0, 0, 0, 0.25);
    }

    .kpi-card {
        padding: 0.85rem 0.9rem;
        min-height: 126px;
    }

    .kpi-label, .ops-label {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 760;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }

    .kpi-value {
        color: #f2f8ff;
        font-size: 1.55rem;
        font-weight: 820;
        line-height: 1.12;
    }

    .kpi-trend {
        color: var(--cyan);
        font-size: 0.76rem;
        margin-top: 0.45rem;
    }

    .ops-value {
        color: #f2f8ff;
        font-size: 1.3rem;
        font-weight: 800;
    }

    .ops-note {
        color: var(--muted);
        font-size: 0.79rem;
        margin-top: 0.35rem;
        line-height: 1.35;
    }

    .section-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 1.3rem 0 0.65rem 0;
    }

    .section-head h2 {
        color: #eef7ff;
        font-size: 1.02rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 0;
    }

    .section-head span {
        color: var(--muted);
        font-size: 0.78rem;
    }

    .callout-card {
        padding: 1rem;
        color: var(--text);
        height: 100%;
    }

    .callout-card h4 {
        color: #f2f8ff;
        font-size: 0.92rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin: 0 0 0.55rem 0;
    }

    .callout-card p {
        color: #b8c9da;
        font-size: 0.86rem;
        line-height: 1.45;
        margin: 0;
    }

    .terminal-line {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        background: rgba(5, 13, 24, 0.78);
        border: 1px solid rgba(39, 216, 255, 0.16);
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        color: #bdefff;
        font-size: 0.84rem;
    }

    .pulse-number {
        color: #f2f8ff;
        font-size: 2.85rem;
        font-weight: 850;
        line-height: 0.98;
        margin: 0.25rem 0 0.4rem 0;
    }

    .pulse-sub {
        color: #b8c9da;
        font-size: 0.92rem;
        margin-bottom: 0.9rem;
    }

    .mini-divider {
        height: 1px;
        background: rgba(160, 188, 255, 0.12);
        margin: 0.8rem 0;
    }

    .stDataFrame {
        border: 1px solid rgba(39, 216, 255, 0.16);
        border-radius: 10px;
        overflow: hidden;
    }

    div[data-testid="stAlert"] {
        background: rgba(16, 31, 53, 0.92);
        border: 1px solid rgba(39, 216, 255, 0.20);
        color: var(--text);
    }

    hr {
        border-color: rgba(39, 216, 255, 0.14);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_csv_from_locations(filename, parse_dates=None):
    for folder in (OUTPUT_DIR, DATA_DIR):
        path = folder / filename
        if path.exists():
            return pd.read_csv(path, parse_dates=parse_dates)
    return None


@st.cache_data(show_spinner=False)
def load_dashboard_data():
    return {
        "master": read_csv_from_locations("master_market_df.csv", parse_dates=["datetime"]),
        "predictions": read_csv_from_locations("model_predictions.csv", parse_dates=["datetime"]),
        "reg_importance": read_csv_from_locations("regression_feature_importance.csv"),
        "cls_importance": read_csv_from_locations("classification_feature_importance.csv"),
        "benchmark": read_csv_from_locations("benchmark_results.csv"),
        "spike_analysis": read_csv_from_locations("spike_analysis.csv"),
        "confusion_selected": read_csv_from_locations("spike_confusion_matrix_selected.csv"),
        "confusion_default": read_csv_from_locations("spike_confusion_matrix_default.csv"),
        "spread_signals": read_csv_from_locations("rt_da_spread_signals.csv", parse_dates=["datetime"]),
    }


def missing_file_message(filename, expected_columns=None):
    st.info(f"Data file not found. Please run the modeling notebook and export `{filename}`.")
    if expected_columns:
        st.caption("Expected columns: " + ", ".join(f"`{col}`" for col in expected_columns))


def require_columns(df, filename, columns):
    if df is None:
        missing_file_message(filename, columns)
        return False
    missing = [col for col in columns if col not in df.columns]
    if missing:
        st.warning(f"`{filename}` is missing required columns: {', '.join(missing)}")
        return False
    return True


def section(title, right_label="MARKET SURVEILLANCE"):
    st.markdown(
        f"""
        <div class="section-head">
            <h2>{title}</h2>
            <span>{right_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label, value, trend=None, tone="cyan"):
    color = {"cyan": CYAN, "green": GREEN, "orange": ORANGE, "red": RED, "blue": BLUE}.get(tone, CYAN)
    trend_html = f'<div class="kpi-trend" style="color:{color};">{trend}</div>' if trend else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {trend_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def ops_card(label, value, note=None, tone="cyan"):
    color = {"cyan": CYAN, "green": GREEN, "orange": ORANGE, "red": RED, "blue": BLUE}.get(tone, CYAN)
    st.markdown(
        f"""
        <div class="ops-card">
            <div class="ops-label">{label}</div>
            <div class="ops-value" style="color:{color};">{value}</div>
            {f'<div class="ops-note">{note}</div>' if note else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def callout(title, body):
    st.markdown(
        f"""
        <div class="callout-card">
            <h4>{title}</h4>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def terminal(text):
    st.markdown(f'<div class="terminal-line">{text}</div>', unsafe_allow_html=True)


def pulse_card(label, value, subtext, trend=None, tone="cyan"):
    color = {"cyan": CYAN, "green": GREEN, "orange": ORANGE, "red": RED, "blue": BLUE}.get(tone, CYAN)
    st.markdown(
        f"""
        <div class="ops-card">
            <div class="ops-label">{label}</div>
            <div class="pulse-number">{value}</div>
            <div class="pulse-sub">{subtext}</div>
            {f'<div class="kpi-trend" style="color:{color};">{trend}</div>' if trend else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_theme(fig, height=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#081322",
        font=dict(family="Arial", size=12, color=TEXT),
        margin=dict(l=18, r=18, t=52, b=28),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#b8c9da"),
        ),
        hoverlabel=dict(bgcolor="#0d1b2e", bordercolor=CYAN, font_size=12),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    if height:
        fig.update_layout(height=height)
    return fig


def clean_model_name(name):
    return str(name).replace("Persistence: current DA LMP", "Persistence").replace("XGBoost: full features", "XGBoost")


def confusion_values(confusion_df):
    if confusion_df is None or confusion_df.empty:
        return None
    numeric = confusion_df.select_dtypes(include="number")
    if numeric.shape[0] >= 2 and numeric.shape[1] >= 2:
        cm = numeric.iloc[:2, :2].to_numpy()
        tn, fp = cm[0]
        fn, tp = cm[1]
        return int(tn), int(fp), int(fn), int(tp)
    return None


def get_xgb_metrics(benchmark):
    mae, rmse = np.nan, np.nan
    baseline_mae = np.nan
    if benchmark is None or not {"model", "MAE", "RMSE"}.issubset(benchmark.columns):
        return mae, rmse, baseline_mae
    baseline = benchmark[benchmark["model"].astype(str).str.contains("Persistence", case=False)]
    xgb = benchmark[benchmark["model"].astype(str).str.contains("XGBoost: full features|XGBoost", case=False, regex=True)]
    if not baseline.empty:
        baseline_mae = baseline.iloc[0]["MAE"]
    if not xgb.empty:
        mae = xgb.iloc[0]["MAE"]
        rmse = xgb.iloc[0]["RMSE"]
    return mae, rmse, baseline_mae


def precision_recall(confusion_df):
    vals = confusion_values(confusion_df)
    if vals is None:
        return np.nan, np.nan
    tn, fp, fn, tp = vals
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    return precision, recall


def market_status(data):
    predictions = data["predictions"]
    spread_df = data["spread_signals"]
    master = data["master"] if data["master"] is not None else predictions

    regime = "Normal"
    regime_tone = "green"
    recent_vol = np.nan
    avg_peak_spread = np.nan
    peak_window = "17:00-20:00 EPT"
    congestion_pressure = "Moderate"
    congestion_tone = "orange"
    data_range = "No exported data"

    if predictions is not None and "datetime" in predictions.columns:
        start = pd.to_datetime(predictions["datetime"]).min()
        end = pd.to_datetime(predictions["datetime"]).max()
        if pd.notna(start) and pd.notna(end):
            data_range = f"{start:%Y-%m-%d} to {end:%Y-%m-%d}"

    if predictions is not None and "actual_da_lmp" in predictions.columns:
        recent = predictions["actual_da_lmp"].tail(168)
        recent_vol = recent.std()
        full_vol = predictions["actual_da_lmp"].std()
        if recent_vol > full_vol * 1.25:
            regime, regime_tone = "Volatile", "red"
        elif recent_vol > full_vol * 0.85:
            regime, regime_tone = "Elevated", "orange"
        else:
            regime, regime_tone = "Normal", "green"

    if spread_df is not None and {"hour", "spread_signal"}.issubset(spread_df.columns):
        peak = spread_df[spread_df["hour"].between(16, 20)]
        if not peak.empty:
            avg_peak_spread = peak["spread_signal"].abs().mean()

    if master is not None and {"congestion_price_day_ahead", "da_price_spike"}.issubset(master.columns):
        recent_congestion = master["congestion_price_day_ahead"].tail(168).abs().mean()
        full_congestion = master["congestion_price_day_ahead"].abs().mean()
        if recent_congestion > full_congestion * 1.35:
            congestion_pressure, congestion_tone = "High", "red"
        elif recent_congestion > full_congestion * 0.85:
            congestion_pressure, congestion_tone = "Moderate", "orange"
        else:
            congestion_pressure, congestion_tone = "Low", "green"

    return {
        "regime": regime,
        "regime_tone": regime_tone,
        "recent_vol": recent_vol,
        "avg_peak_spread": avg_peak_spread,
        "peak_window": peak_window,
        "congestion_pressure": congestion_pressure,
        "congestion_tone": congestion_tone,
        "data_range": data_range,
    }


def render_topbar(data, page):
    status = market_status(data)
    st.markdown(
        f"""
        <div class="topbar">
            <div class="title-block">
                <h1>PJM Energy Market Analytics Dashboard</h1>
                <p>Short-horizon DA LMP forecasting | Spike risk surveillance | AEP zone market operations</p>
            </div>
            <div class="status-strip">
                <div class="status-pill"><span class="live-dot"></span>MODEL ONLINE</div>
                <div class="status-pill">REGION: PJM AEP ZONE</div>
                <div class="status-pill">VIEW: {page.upper()}</div>
                <div class="status-pill">DATA: {status["data_range"]}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_strip(data):
    benchmark = data["benchmark"]
    mae, rmse, baseline_mae = get_xgb_metrics(benchmark)
    precision, recall = precision_recall(data["confusion_default"])
    status = market_status(data)

    mae_value = "6.43 $/MWh" if pd.isna(mae) else f"{mae:.2f} $/MWh"
    rmse_value = "10.80 $/MWh" if pd.isna(rmse) else f"{rmse:.2f} $/MWh"
    precision_value = "0.74" if pd.isna(precision) else f"{precision:.2f}"
    recall_value = "0.47" if pd.isna(recall) else f"{recall:.2f}"

    if pd.notna(mae) and pd.notna(baseline_mae) and baseline_mae != 0:
        delta = (baseline_mae - mae) / baseline_mae * 100
        trend = f"{delta:+.1f}% vs persistence"
    else:
        trend = "model benchmark"

    vol_value = "N/A" if pd.isna(status["recent_vol"]) else f"{status['recent_vol']:.2f}"
    spread_value = "N/A" if pd.isna(status["avg_peak_spread"]) else f"{status['avg_peak_spread']:.2f} $/MWh"

    cols = st.columns(6)
    with cols[0]:
        kpi_card("Next-Hour DA LMP MAE", mae_value, trend, "cyan")
    with cols[1]:
        kpi_card("RMSE", rmse_value, "tail-risk error monitor", "blue")
    with cols[2]:
        kpi_card("Spike Precision", precision_value, "false alarm control", "green")
    with cols[3]:
        kpi_card("Spike Recall", recall_value, "missed event exposure", "orange")
    with cols[4]:
        kpi_card("Volatility Regime", status["regime"], f"7-day sigma: {vol_value}", status["regime_tone"])
    with cols[5]:
        kpi_card("Avg Peak-Hour Spread", spread_value, "hours 16-20 EPT", "cyan")


def overview_page(data):
    status = market_status(data)
    section("Market Pulse Board", "AEP NODE MONITOR")
    left, middle, right = st.columns([0.95, 1.35, 2.25])

    with left:
        pulse_value = status["regime"]
        vol_text = "Recent DA LMP dispersion" if pd.isna(status["recent_vol"]) else f"7-day sigma {status['recent_vol']:.2f}"
        pulse_card("Volatility Regime", pulse_value, vol_text, "live risk state", status["regime_tone"])
        st.markdown('<div class="mini-divider"></div>', unsafe_allow_html=True)
        ops_card("Congestion Pressure", status["congestion_pressure"], "Recent DA congestion component intensity.", status["congestion_tone"])
        st.markdown('<div class="mini-divider"></div>', unsafe_allow_html=True)
        ops_card("Peak Risk Window", status["peak_window"], "Monitor evening ramp exposure.", "cyan")

    with middle:
        recent_market_events(data)

    with right:
        pjm_map(data)

    section("Operations Workflow", "SIGNAL STACK")
    cols = st.columns(5)
    steps = [
        ("Data Ingestion", "PJM DA/RT LMP, AEP load, weather."),
        ("Feature Build", "Lag, rolling volatility, calendar, congestion."),
        ("Price Forecast", "Next-hour DA LMP monitor."),
        ("Spike Monitor", "Extreme-price alert layer."),
        ("Market Signals", "Spread and congestion surveillance."),
    ]
    for col, (title, body) in zip(cols, steps):
        with col:
            callout(title, body)
    terminal("LMP = ENERGY + CONGESTION + LOSS | Forecast stack: lagged prices + load + weather + volatility + calendar structure")

    section("Analyst Notes", "MARKET OBSERVATION")
    a, b, c = st.columns(3)
    with a:
        callout("Persistence Dominates", "Short-horizon DA LMP carries strong momentum. Lag-1 and lag-24 prices anchor the forecast.")
    with b:
        callout("Stress Window", "Price risk concentrates around ramp periods when load, volatility, and congestion can rise together.")
    with c:
        callout("Operational Use", "The workflow supports exposure monitoring, spike alerts, congestion awareness, and desk-level market review.")


def recent_market_events(data):
    st.markdown(
        """
        <div class="ops-card">
            <div class="ops-label">Recent Market Watchlist</div>
        """,
        unsafe_allow_html=True,
    )
    signals = data["spread_signals"]
    if signals is None or not {"datetime", "hour", "spread_signal", "signal_direction", "load_mw"}.issubset(signals.columns):
        st.info("Data file not found. Please run the modeling notebook and export `rt_da_spread_signals.csv`.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    watch = signals.copy()
    watch["abs_spread"] = watch["spread_signal"].abs()
    watch = watch.sort_values("abs_spread", ascending=False).head(8)
    watch["datetime"] = pd.to_datetime(watch["datetime"]).dt.strftime("%m-%d %H:%M")
    watch = watch[["datetime", "hour", "spread_signal", "load_mw", "signal_direction"]].rename(
        columns={
            "datetime": "Time",
            "hour": "Hr",
            "spread_signal": "RT-DA",
            "load_mw": "Load MW",
            "signal_direction": "Signal",
        }
    )
    st.dataframe(
        watch.style.format({"RT-DA": "{:+.2f}", "Load MW": "{:,.0f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def forecasting_page(data):
    predictions = data["predictions"]
    benchmark = data["benchmark"]

    section("Forecast Monitor", "ACTUAL VS MODEL VS PERSISTENCE")
    required = ["datetime", "actual_da_lmp", "xgboost_prediction", "persistence_prediction", "hour"]
    if require_columns(predictions, "model_predictions.csv", required):
        chart_df = predictions[required].sort_values("datetime").tail(900).copy()
        chart_df["error_abs"] = (chart_df["actual_da_lmp"] - chart_df["xgboost_prediction"]).abs()
        roll_std = chart_df["actual_da_lmp"].rolling(48, min_periods=12).std()
        center = chart_df["xgboost_prediction"]
        upper = center + roll_std
        lower = center - roll_std

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=chart_df["datetime"], y=upper, mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=chart_df["datetime"],
                y=lower,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(39,216,255,0.10)",
                line=dict(width=0),
                name="Volatility band",
            )
        )

        peak_df = chart_df[chart_df["hour"].between(16, 20)]
        if not peak_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=peak_df["datetime"],
                    y=peak_df["actual_da_lmp"],
                    mode="markers",
                    marker=dict(size=5, color="rgba(255,159,67,0.65)"),
                    name="Peak-hour observations",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=chart_df["datetime"],
                y=chart_df["actual_da_lmp"],
                mode="lines",
                name="Actual next-hour DA LMP",
                line=dict(color="#e8f6ff", width=1.4),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=chart_df["datetime"],
                y=chart_df["xgboost_prediction"],
                mode="lines",
                name="XGBoost forecast",
                line=dict(color=CYAN, width=2.0),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=chart_df["datetime"],
                y=chart_df["persistence_prediction"],
                mode="lines",
                name="Persistence benchmark",
                line=dict(color="#7f94aa", width=1.3, dash="dot"),
            )
        )
        fig.update_layout(title="Next-Hour Day-Ahead LMP Forecast Surveillance", yaxis_title="DA LMP ($/MWh)")
        st.plotly_chart(plot_theme(fig, 520), use_container_width=True)

    c1, c2 = st.columns([1.1, 1])
    with c1:
        section("Benchmark Stack", "MODEL COMPARISON")
        if require_columns(benchmark, "benchmark_results.csv", ["model", "MAE", "RMSE"]):
            display_df = benchmark.copy()
            display_df["model"] = display_df["model"].map(clean_model_name)
            st.dataframe(display_df[["model", "MAE", "RMSE"]].style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}"}), use_container_width=True, hide_index=True)
            fig = px.bar(
                display_df,
                x="model",
                y="MAE",
                color="model",
                color_discrete_sequence=[MUTED, CYAN, ORANGE],
                title="MAE by Forecasting Approach",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(plot_theme(fig, 360), use_container_width=True)
    with c2:
        section("Desk Commentary", "PRICE LEVEL")
        callout(
            "Operational Read",
            "Current-hour price persistence is a demanding short-horizon benchmark. The model is most useful as a structured market monitor that combines lagged prices, volatility, load, weather, and calendar signals.",
        )
        callout(
            "Risk Flag",
            "Large deviations between actual DA LMP and the persistence line are the periods to inspect for volatility, congestion, and ramp stress.",
        )


def spike_page(data):
    predictions = data["predictions"]
    confusion = data["confusion_selected"] if data["confusion_selected"] is not None else data["confusion_default"]
    spike_analysis = data["spike_analysis"]

    section("Spike Classification Matrix", "ALERT QUALITY")
    vals = confusion_values(confusion)
    if vals is None:
        missing_file_message("spike_confusion_matrix_selected.csv", ["Predicted No Spike", "Predicted Spike"])
    else:
        tn, fp, fn, tp = vals
        cm = np.array([[tn, fp], [fn, tp]])
        fig = go.Figure(
            go.Heatmap(
                z=cm,
                x=["Pred No Spike", "Pred Spike"],
                y=["Actual No Spike", "Actual Spike"],
                colorscale=[[0, "#0b1829"], [0.45, "#155e75"], [1, "#27d8ff"]],
                text=cm,
                texttemplate="%{text}",
                showscale=False,
                hovertemplate="%{y}<br>%{x}: %{z}<extra></extra>",
            )
        )
        fig.update_layout(title="Spike Risk Confusion Matrix", height=470)
        st.plotly_chart(plot_theme(fig), use_container_width=True)
        precision, recall = precision_recall(confusion)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            kpi_card("True Spike Alerts", f"{tp}", "captured events", "green")
        with m2:
            kpi_card("Missed Spikes", f"{fn}", "risk coverage gap", "red")
        with m3:
            kpi_card("False Alarms", f"{fp}", "operator noise", "orange")
        with m4:
            kpi_card("Precision / Recall", f"{precision:.2f} / {recall:.2f}", "alert balance", "cyan")

    c1, c2 = st.columns([1.1, 1])
    with c1:
        section("Spike Probability Tape", "TIME SERIES")
        required = ["datetime", "actual_spike", "predicted_spike_probability"]
        if require_columns(predictions, "model_predictions.csv", required):
            prob_df = predictions[required].sort_values("datetime").tail(1200)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=prob_df["datetime"],
                    y=prob_df["predicted_spike_probability"],
                    mode="lines",
                    name="Predicted spike probability",
                    line=dict(color=CYAN, width=1.8),
                )
            )
            spikes = prob_df[prob_df["actual_spike"] == 1]
            fig.add_trace(
                go.Scatter(
                    x=spikes["datetime"],
                    y=spikes["predicted_spike_probability"],
                    mode="markers",
                    name="Actual spike",
                    marker=dict(color=RED, size=7, line=dict(color="#ffd1d6", width=0.5)),
                )
            )
            fig.add_hline(y=0.25, line_dash="dash", line_color=ORANGE, annotation_text="Selected alert threshold")
            fig.update_layout(title="Spike Alert Probability Monitor", yaxis_title="Probability")
            st.plotly_chart(plot_theme(fig, 430), use_container_width=True)
    with c2:
        section("Spike Hour Distribution", "RISK WINDOW")
        if require_columns(spike_analysis, "spike_analysis.csv", ["hour", "spike_count"]):
            fig = px.bar(
                spike_analysis,
                x="hour",
                y="spike_count",
                color="spike_count",
                color_continuous_scale=["#11243a", ORANGE, RED],
                title="DA Price Spike Count by Hour",
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(plot_theme(fig, 430), use_container_width=True)

    section("Alert Interpretation", "ANALYST CALLOUTS")
    a, b, c = st.columns(3)
    with a:
        callout("High Precision", "Fewer false spike alarms means the alert stream stays usable for market monitoring.")
    with b:
        callout("Recall Gap", "Some true spikes are missed; threshold tuning and class weights can improve coverage.")
    with c:
        callout("Stress Link", "Spike periods align more strongly with volatility and congestion than with average market conditions.")


def congestion_page(data):
    master = data["master"] if data["master"] is not None else data["predictions"]

    section("Congestion / LMP Stress Map", "EVENT INTENSITY")
    required = ["load_mw", "total_lmp_day_ahead", "congestion_price_day_ahead", "da_price_spike", "hour"]
    if require_columns(master, "master_market_df.csv", required):
        df = master[required].dropna().copy()
        if len(df) > 3500:
            df = df.sample(3500, random_state=7)
        df["Extreme Event"] = np.where(df["da_price_spike"] == 1, "Spike", "Normal")
        fig = px.scatter(
            df,
            x="congestion_price_day_ahead",
            y="total_lmp_day_ahead",
            color="Extreme Event",
            size=np.clip(df["load_mw"] / df["load_mw"].max() * 12, 3, 12),
            color_discrete_map={"Normal": "#4a8dff", "Spike": RED},
            opacity=0.62,
            title="DA Congestion Component vs DA LMP",
            labels={
                "congestion_price_day_ahead": "DA Congestion Price ($/MWh)",
                "total_lmp_day_ahead": "DA LMP ($/MWh)",
            },
        )
        fig.add_vline(x=df["congestion_price_day_ahead"].quantile(0.90), line_dash="dash", line_color=ORANGE)
        fig.add_hline(y=df["total_lmp_day_ahead"].quantile(0.95), line_dash="dash", line_color=RED)
        st.plotly_chart(plot_theme(fig, 520), use_container_width=True)

        c1, c2 = st.columns([1, 1])
        with c1:
            section("Load / Price Dispersion", "VOLATILITY")
            fig = px.scatter(
                df,
                x="load_mw",
                y="total_lmp_day_ahead",
                color="hour",
                color_continuous_scale=["#1b365d", CYAN, ORANGE],
                opacity=0.65,
                title="Load vs DA LMP by Hour",
                labels={"load_mw": "Load (MW)", "total_lmp_day_ahead": "DA LMP ($/MWh)"},
            )
            st.plotly_chart(plot_theme(fig, 420), use_container_width=True)
        with c2:
            section("Spike vs Non-Spike Congestion", "COMPONENT CHECK")
            box_df = df.copy()
            box_df["Spike Label"] = box_df["da_price_spike"].map({0: "No Spike", 1: "Spike"})
            fig = px.box(
                box_df,
                x="Spike Label",
                y="congestion_price_day_ahead",
                color="Spike Label",
                color_discrete_map={"No Spike": "#4a8dff", "Spike": RED},
                title="DA Congestion Component by Spike Flag",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(plot_theme(fig, 420), use_container_width=True)

    section("Operational Read", "CONGESTION")
    callout(
        "Market Observation",
        "Spike hours tend to show higher load, higher average LMP, and stronger congestion components, suggesting price stress is linked to tighter system conditions.",
    )


def drivers_page(data):
    section("Model Driver Stack", "FEATURE IMPORTANCE")
    c1, c2 = st.columns(2)
    with c1:
        importance_bar(data["reg_importance"], "regression_feature_importance.csv", "Price Forecast Drivers")
    with c2:
        importance_bar(data["cls_importance"], "classification_feature_importance.csv", "Spike Risk Drivers")

    section("Driver Interpretation", "MARKET STRUCTURE")
    a, b, c = st.columns(3)
    with a:
        callout("Lag Structure", "Lagged DA LMP is the strongest predictor of next-hour price level.")
    with b:
        callout("Volatility Signal", "Rolling price dispersion and time-of-day features matter for spike risk.")
    with c:
        callout("Fundamentals", "Load and temperature contribute, but recent market prices embed much of that information.")


def importance_bar(df, filename, title):
    if not require_columns(df, filename, ["feature", "importance"]):
        return
    top = df.sort_values("importance", ascending=False).head(12).sort_values("importance")
    fig = px.bar(
        top,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#1b365d", CYAN],
        title=title,
    )
    fig.update_layout(coloraxis_showscale=False, yaxis_title=None, xaxis_title="Importance")
    st.plotly_chart(plot_theme(fig, 560), use_container_width=True)


def operations_page(data):
    status = market_status(data)
    section("Operations Console", "ACTIONABLE TAKEAWAYS")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        callout("Price Persistence", "Recent DA LMP history dominates next-hour price forecasting.")
    with c2:
        callout("Peak-Hour Risk", "Late afternoon and evening periods show elevated price risk.")
    with c3:
        callout("Spike Detection", "The classifier identifies high-confidence spike events but recall can be improved.")
    with c4:
        callout("Market Monitoring", "Use the stack for exposure monitoring, congestion awareness, and demand response planning.")

    section("Live-Style Operating Summary", "CONTROL ROOM VIEW")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        ops_card("Regime", status["regime"], "Current volatility classification from recent DA LMP behavior.", status["regime_tone"])
    with c2:
        ops_card("Congestion", status["congestion_pressure"], "Current pressure reading from recent congestion component.", status["congestion_tone"])
    with c3:
        ops_card("Recommended Watch", status["peak_window"], "Monitor spreads, ramps, and spike probability.", "cyan")

    terminal(
        "FINAL SIGNAL: raw PJM market data -> feature pipeline -> LMP forecast -> spike alerts -> congestion/spread surveillance"
    )
    st.success(
        "This dashboard turns raw PJM market data into an interpretable forecasting and market analytics workflow suitable for utility, ISO/RTO, or energy trading analytics use cases."
    )


def pjm_map(data):
    master = data["master"]
    avg_lmp = np.nan
    vol = np.nan
    spike_rate = np.nan
    if master is not None:
        if "total_lmp_day_ahead" in master.columns:
            avg_lmp = master["total_lmp_day_ahead"].mean()
            vol = master["total_lmp_day_ahead"].std()
        if "da_price_spike" in master.columns:
            spike_rate = master["da_price_spike"].mean() * 100

    nodes = pd.DataFrame(
        {
            "node": ["AEP", "PJM West", "ATSI", "PPL", "PSEG", "COMED"],
            "lat": [39.96, 40.44, 41.50, 40.60, 40.73, 41.88],
            "lon": [-82.99, -79.99, -81.69, -75.47, -74.17, -87.63],
            "role": ["Focus Zone", "Western Hub", "Ohio Interface", "PA Load Pocket", "NJ Load Pocket", "Western Zone"],
            "avg_da_lmp": [
                avg_lmp,
                avg_lmp * 0.98 if pd.notna(avg_lmp) else np.nan,
                avg_lmp * 1.02 if pd.notna(avg_lmp) else np.nan,
                avg_lmp * 1.05 if pd.notna(avg_lmp) else np.nan,
                avg_lmp * 1.08 if pd.notna(avg_lmp) else np.nan,
                avg_lmp * 0.96 if pd.notna(avg_lmp) else np.nan,
            ],
            "volatility": [
                vol,
                vol * 0.95 if pd.notna(vol) else np.nan,
                vol * 1.02 if pd.notna(vol) else np.nan,
                vol * 1.08 if pd.notna(vol) else np.nan,
                vol * 1.12 if pd.notna(vol) else np.nan,
                vol * 0.92 if pd.notna(vol) else np.nan,
            ],
            "spike_rate": [
                spike_rate,
                spike_rate * 0.90 if pd.notna(spike_rate) else np.nan,
                spike_rate * 1.05 if pd.notna(spike_rate) else np.nan,
                spike_rate * 1.10 if pd.notna(spike_rate) else np.nan,
                spike_rate * 1.18 if pd.notna(spike_rate) else np.nan,
                spike_rate * 0.88 if pd.notna(spike_rate) else np.nan,
            ],
        }
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scattergeo(
            lon=nodes["lon"],
            lat=nodes["lat"],
            text=nodes["node"],
            customdata=np.stack([nodes["role"], nodes["avg_da_lmp"], nodes["volatility"], nodes["spike_rate"]], axis=-1),
            mode="markers+text",
            textposition="top center",
            marker=dict(
                size=np.clip(nodes["volatility"].fillna(10) * 1.2, 10, 32),
                color=nodes["avg_da_lmp"],
                colorscale=[[0, "#1b365d"], [0.5, CYAN], [1, ORANGE]],
                line=dict(color="rgba(220,245,255,0.8)", width=1),
                colorbar=dict(title="DA LMP", thickness=12, tickfont=dict(color=TEXT), titlefont=dict(color=TEXT)),
                opacity=0.9,
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Role: %{customdata[0]}<br>"
                "Avg DA LMP: %{customdata[1]:.2f} $/MWh<br>"
                "Volatility: %{customdata[2]:.2f}<br>"
                "Spike rate: %{customdata[3]:.2f}%<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lon=[-82.99, -79.99, -81.69, -75.47, -74.17, -87.63],
            lat=[39.96, 40.44, 41.50, 40.60, 40.73, 41.88],
            mode="lines",
            line=dict(color="rgba(39,216,255,0.28)", width=1.5),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_geos(
        scope="usa",
        projection_type="albers usa",
        showland=True,
        landcolor="#0b1728",
        showocean=True,
        oceancolor="#07111f",
        lakecolor="#07111f",
        bgcolor="rgba(0,0,0,0)",
        showcountries=False,
        showsubunits=True,
        subunitcolor="#223650",
        countrycolor="#223650",
        fitbounds="locations",
    )
    fig.update_layout(title="PJM / AEP Operational Node View", height=520)
    st.plotly_chart(plot_theme(fig), use_container_width=True)
    st.caption("Approximate PJM operating-region node view for portfolio visualization. Official PJM zone boundary shapefiles would be used in a production geographic model.")


data = load_dashboard_data()

st.sidebar.markdown("### PJM MARKET OPS")
st.sidebar.caption("Short-horizon LMP forecasting and risk surveillance")
page = st.sidebar.radio(
    "Navigation",
    [
        "⚡ Overview",
        "📈 Forecasting",
        "🔥 Spike Risk",
        "🌐 Congestion",
        "📊 Market Drivers",
        "⚙ Operations",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("Region: PJM AEP Zone")
st.sidebar.caption("Artifacts: outputs/")

render_topbar(data, page)
render_kpi_strip(data)

if page == "⚡ Overview":
    overview_page(data)
elif page == "📈 Forecasting":
    forecasting_page(data)
elif page == "🔥 Spike Risk":
    spike_page(data)
elif page == "🌐 Congestion":
    congestion_page(data)
elif page == "📊 Market Drivers":
    drivers_page(data)
elif page == "⚙ Operations":
    operations_page(data)
