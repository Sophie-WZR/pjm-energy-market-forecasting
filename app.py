from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="PJM Energy Market Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"

COLORS = {
    "ink": "#02040a",
    "panel": "#0f172a",
    "grid": "rgba(148,163,184,0.12)",
    "text": "#dbeafe",
    "muted": "#94a3b8",
    "white": "#f8fbff",
    "cyan": "#38bdf8",
    "blue": "#60a5fa",
    "orange": "#fb923c",
    "red": "#f87171",
}


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(37, 74, 120, 0.28), transparent 34%),
            linear-gradient(145deg, #07101d 0%, #050a14 52%, #03060d 100%);
        color: #e8f1ff;
    }

    .block-container {
        padding-top: 0.65rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 1680px;
    }

    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    #MainMenu,
    footer {
        display: none !important;
        visibility: hidden !important;
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    [data-testid="collapsedControl"] {
        display: none;
    }

    .analytics-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 14px;
        padding: 11px 14px;
        margin-bottom: 7px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.18);
    }

    .analytics-title {
        color: #f8fbff;
        font-size: 22px;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.12;
    }

    .analytics-subtitle {
        margin-top: 3px;
        color: #a7b4c7;
        font-size: 13px;
        font-weight: 500;
        letter-spacing: 0.01em;
    }

    div[data-testid="stSegmentedControl"] {
        display: inline-block;
        width: fit-content;
        max-width: 100%;
        margin-bottom: 5px;
    }

    div[data-testid="stSegmentedControl"] [role="radiogroup"],
    div[data-testid="stButtonGroup"] {
        display: inline-flex !important;
        width: fit-content !important;
        max-width: 100%;
        gap: 12px;
        padding: 0;
        border: 0;
        border-radius: 0;
        background: transparent;
        box-shadow: none;
    }

    div[data-testid="stSegmentedControl"] button,
    div[data-testid="stButtonGroup"] button {
        min-height: 37px;
        padding-left: 13px;
        padding-right: 13px;
        margin-right: 8px !important;
        border-radius: 10px !important;
        border: 1px solid rgba(148, 163, 184, 0.16) !important;
        background: rgba(15, 23, 42, 0.58) !important;
        color: #cbd5e1 !important;
        font-weight: 650 !important;
        white-space: nowrap;
    }

    div[data-testid="stSegmentedControl"] button:last-child,
    div[data-testid="stButtonGroup"] button:last-child {
        margin-right: 0 !important;
    }

    div[data-testid="stSegmentedControl"] button:hover,
    div[data-testid="stButtonGroup"] button:hover {
        border-color: rgba(125, 177, 226, 0.34) !important;
        background: rgba(21, 35, 59, 0.84) !important;
    }

    div[data-testid="stSegmentedControl"] button[aria-checked="true"],
    div[data-testid="stButtonGroup"] button[aria-checked="true"],
    div[data-testid="stSegmentedControl"] button[aria-selected="true"],
    div[data-testid="stButtonGroup"] button[aria-selected="true"],
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
    div[data-testid="stButtonGroup"] button[aria-pressed="true"] {
        border: 1px solid rgba(96, 165, 250, 0.62) !important;
        background: rgba(25, 49, 81, 0.92) !important;
        color: #f8fbff !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
    }

    div[data-testid="stPopover"] button {
        min-height: 37px;
        border-radius: 10px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        background: rgba(15, 23, 42, 0.72);
        color: #dbeafe;
        font-weight: 650;
    }

    .page-head {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 16px;
        margin: 8px 0 10px;
        padding: 0 2px;
    }

    .page-title {
        font-size: 22px;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #f8fbff;
    }

    .page-subtitle {
        margin-top: 3px;
        color: #93a4bd;
        font-size: 13px;
        line-height: 1.35;
    }

    .date-label {
        padding: 4px 0;
        color: #a7b4c7;
        font-size: 12px;
        font-weight: 550;
        letter-spacing: 0.02em;
        white-space: nowrap;
    }

    .kpi-card {
        padding: 14px 15px;
        min-height: 100px;
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 0 8px 20px rgba(0,0,0,0.18);
    }

    .kpi-label {
        color: #94a3b8;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }

    .kpi-value {
        margin-top: 9px;
        color: #f8fafc;
        font-size: 27px;
        font-weight: 800;
        letter-spacing: -0.04em;
    }

    .kpi-note {
        margin-top: 6px;
        color: #38bdf8;
        font-size: 12px;
        font-weight: 700;
    }

    .kpi-warn {
        color: #fb923c;
    }

    .panel-title {
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #f1f5f9;
        margin: 11px 0 6px;
    }

    .panel-title:after {
        content: "";
        display: inline-block;
        width: 28px;
        height: 1px;
        margin-left: 10px;
        vertical-align: middle;
        background: rgba(148, 163, 184, 0.45);
    }

    .callout {
        padding: 16px 18px;
        border-radius: 16px;
        background: rgba(10, 17, 31, 0.72);
        border-left: 3px solid rgba(96, 165, 250, 0.72);
        color: #cbd5e1;
        font-size: 14px;
        line-height: 1.5;
        margin-bottom: 12px;
    }

    .warning-callout {
        border-left-color: #fb923c;
    }

    .risk-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-bottom: 14px;
    }

    .risk-cell {
        padding: 13px 14px;
        border-radius: 14px;
        background: rgba(8, 20, 38, 0.90);
        border: 1px solid rgba(148, 163, 184, 0.12);
    }

    .risk-label {
        color: #94a3b8;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }

    .risk-value {
        color: #f8fbff;
        font-size: 21px;
        font-weight: 800;
        margin-top: 7px;
    }

    .watch-table {
        border-collapse: collapse;
        width: 100%;
        overflow: hidden;
        border-radius: 14px;
        font-size: 12px;
    }

    .watch-table th {
        padding: 10px 11px;
        background: rgba(8, 20, 38, 0.96);
        border-bottom: 1px solid rgba(148, 163, 184, 0.16);
        color: #94a3b8;
        text-align: left;
        text-transform: uppercase;
        letter-spacing: 0.09em;
    }

    .watch-table td {
        padding: 10px 11px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.08);
        color: #dbeafe;
    }

    .watch-table tr:nth-child(even) td {
        background: rgba(148, 163, 184, 0.05);
    }

    .spread-up {
        color: #fb923c !important;
        font-weight: 800;
    }

    .spread-down {
        color: #38bdf8 !important;
        font-weight: 800;
    }

    [data-testid="stAlert"] {
        background: rgba(8, 20, 38, 0.92);
        border: 1px solid rgba(56, 189, 248, 0.22);
        color: #dbeafe;
    }

    div[data-testid="stPlotlyChart"] {
        border-radius: 16px;
    }

    hr {
        border-color: rgba(148, 163, 184, 0.16);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi_card(label, value, note="", warn=False):
    note_class = "kpi-note kpi-warn" if warn else "kpi-note"
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="{note_class}">{note}</div>
    </div>
    """


def panel_start(title):
    st.markdown(f'<div class="panel-title">{title}</div>', unsafe_allow_html=True)


def callout(text, warn=False):
    cls = "callout warning-callout" if warn else "callout"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def find_csv(name, parse_dates=None):
    for folder in (OUTPUT_DIR, DATA_DIR):
        path = folder / name
        if path.exists():
            return pd.read_csv(path, parse_dates=parse_dates)
    return None


def normalize_predictions(df):
    if df is None:
        return None
    rename = {
        "datetime_ending_ept": "datetime",
        "prediction": "xgboost_prediction",
        "pred_da_lmp": "xgboost_prediction",
        "actual": "actual_da_lmp",
        "target_da_lmp_next_hour": "actual_da_lmp",
        "baseline_prediction": "persistence_prediction",
        "spike_probability": "predicted_spike_probability",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns}).copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if "hour" not in df.columns and "datetime" in df.columns:
        df["hour"] = df["datetime"].dt.hour
    return df


def demo_bundle():
    idx = pd.date_range("2025-01-25", "2025-04-30 23:00", freq="h")
    rng = np.random.default_rng(129)
    hour = idx.hour
    wave = 9 * np.sin((hour - 7) / 24 * 2 * np.pi)
    stress = rng.normal(0, 8.5, len(idx))
    jump = (rng.random(len(idx)) < 0.045) * rng.gamma(4, 10, len(idx))
    actual = 34 + wave + stress + jump
    pred = pd.Series(actual).shift(1).bfill().to_numpy() + rng.normal(0, 4.2, len(idx))
    persistence = pd.Series(actual).shift(1).bfill().to_numpy()
    spike_threshold = np.quantile(actual, 0.95)
    spike_prob = np.clip((actual - spike_threshold + 22) / 45 + rng.normal(0, 0.08, len(idx)), 0.02, 0.98)
    load = 12100 + 1800 * np.sin((hour - 11) / 24 * 2 * np.pi) + rng.normal(0, 520, len(idx))
    congestion = rng.normal(0.8, 4.4, len(idx)) + jump * 0.16
    spread = rng.normal(-1.5, 12, len(idx)) + jump * 0.35
    master = pd.DataFrame(
        {
            "datetime": idx,
            "hour": hour,
            "load_mw": load,
            "temperature_2m": 44 + 15 * np.sin((hour - 13) / 24 * 2 * np.pi) + rng.normal(0, 5, len(idx)),
            "total_lmp_day_ahead": actual,
            "total_lmp_real_time": actual + spread,
            "congestion_price_day_ahead": congestion,
            "rt_da_spread": spread,
            "da_price_spike": (actual > spike_threshold).astype(int),
        }
    )
    prediction = master[
        [
            "datetime",
            "hour",
            "load_mw",
            "temperature_2m",
            "total_lmp_day_ahead",
            "total_lmp_real_time",
            "congestion_price_day_ahead",
            "rt_da_spread",
            "da_price_spike",
        ]
    ].copy()
    prediction["actual_da_lmp"] = actual
    prediction["actual_spike"] = master["da_price_spike"]
    prediction["xgboost_prediction"] = pred
    prediction["persistence_prediction"] = persistence
    prediction["predicted_spike_probability"] = spike_prob
    prediction["predicted_spike"] = (spike_prob >= 0.60).astype(int)
    spread_df = prediction[
        ["datetime", "hour", "total_lmp_day_ahead", "total_lmp_real_time", "rt_da_spread", "load_mw"]
    ].copy()
    spread_df["spread_signal"] = spread_df["rt_da_spread"]
    spread_df["signal_direction"] = np.where(spread_df["spread_signal"] > 0, "RT premium", "DA premium")
    importance = pd.DataFrame(
        {
            "feature": [
                "total_lmp_day_ahead_lag_1",
                "total_lmp_day_ahead_lag_24",
                "hour",
                "load_mw",
                "temperature_2m",
                "da_lmp_rolling_std_24",
            ],
            "importance": [0.60, 0.13, 0.08, 0.07, 0.06, 0.06],
        }
    )
    return {
        "master": master,
        "pred": prediction,
        "reg_imp": importance,
        "cls_imp": importance.sort_values("importance").reset_index(drop=True),
        "benchmark": pd.DataFrame(
            {
                "model": ["Persistence: current DA LMP", "XGBoost: full features"],
                "MAE": [7.52, 6.51],
                "RMSE": [12.48, 10.80],
            }
        ),
        "spike": master.groupby("hour", as_index=False)["da_price_spike"].sum().rename(columns={"da_price_spike": "spike_count"}),
        "cm_default": pd.DataFrame([[2031, 77], [78, 113]]),
        "cm_selected": pd.DataFrame([[2053, 55], [84, 107]]),
        "spread": spread_df,
    }


@st.cache_data(show_spinner=False)
def load_data():
    demo = demo_bundle()
    file_specs = {
        "master": ("master_market_df.csv", ["datetime"]),
        "pred": ("model_predictions.csv", ["datetime"]),
        "reg_imp": ("regression_feature_importance.csv", None),
        "cls_imp": ("classification_feature_importance.csv", None),
        "benchmark": ("benchmark_results.csv", None),
        "spike": ("spike_analysis.csv", None),
        "cm_default": ("spike_confusion_matrix_default.csv", None),
        "cm_selected": ("spike_confusion_matrix_selected.csv", None),
        "spread": ("rt_da_spread_signals.csv", ["datetime"]),
    }
    data, missing_files = {}, []
    for key, (name, parse_dates) in file_specs.items():
        df = find_csv(name, parse_dates=parse_dates)
        if df is None:
            data[key] = demo[key]
            missing_files.append(name)
        else:
            data[key] = df
    data["pred"] = normalize_predictions(data["pred"])
    for key in ["master", "spread"]:
        if "datetime" in data[key].columns:
            data[key]["datetime"] = pd.to_datetime(data[key]["datetime"], errors="coerce")
        if "hour" not in data[key].columns and "datetime" in data[key].columns:
            data[key]["hour"] = data[key]["datetime"].dt.hour
    data["missing_files"] = missing_files
    return data


def has_cols(df, cols):
    return df is not None and all(col in df.columns for col in cols)


def output_warning(data):
    if data["missing_files"]:
        names = ", ".join(data["missing_files"][:3])
        more = "..." if len(data["missing_files"]) > 3 else ""
        st.warning(f"Using demo data because exported model outputs were not found: {names}{more}")


def filter_market_data(data, start, end, hours):
    filtered = dict(data)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    for key in ["pred", "master", "spread"]:
        df = filtered[key].copy()
        if "datetime" not in df.columns:
            continue
        mask = df["datetime"].between(start_ts, end_ts)
        if "hour" in df.columns:
            mask &= df["hour"].between(hours[0], hours[1])
        filtered[key] = df.loc[mask].reset_index(drop=True)
    return filtered


def cm_values(df):
    if df is None:
        return None
    numeric = df.select_dtypes(include="number")
    if numeric.shape[0] < 2 or numeric.shape[1] < 2:
        return None
    tn, fp = numeric.iloc[0, :2]
    fn, tp = numeric.iloc[1, :2]
    return int(tn), int(fp), int(fn), int(tp)


def market_stats(data):
    pred, master, spread = data["pred"], data["master"], data["spread"]
    benchmark = data["benchmark"]
    mae, rmse, baseline_mae = 6.51, 10.80, np.nan
    if has_cols(benchmark, ["model", "MAE", "RMSE"]):
        models = benchmark["model"].astype(str)
        xgb = benchmark[models.str.contains("XGBoost", case=False)]
        baseline = benchmark[models.str.contains("Persistence", case=False)]
        if not xgb.empty:
            mae, rmse = float(xgb.iloc[0]["MAE"]), float(xgb.iloc[0]["RMSE"])
        if not baseline.empty:
            baseline_mae = float(baseline.iloc[0]["MAE"])
    precision, recall = 0.73, 0.48
    vals = cm_values(data["cm_default"])
    if vals:
        _, fp, fn, tp = vals
        precision = tp / (tp + fp) if tp + fp else precision
        recall = tp / (tp + fn) if tp + fn else recall
    vol_7d, regime = 16.89, "Elevated"
    if has_cols(pred, ["actual_da_lmp"]):
        recent = pred.sort_values("datetime")["actual_da_lmp"].tail(168)
        full = pred["actual_da_lmp"]
        if len(recent.dropna()) > 12:
            vol_7d = float(recent.std())
            ratio = vol_7d / full.std() if full.std() else 1
            regime = "Volatile" if ratio > 1.25 else "Elevated" if ratio > 0.85 else "Normal"
    peak_spread = 13.28
    if has_cols(spread, ["hour", "spread_signal"]):
        peak = spread[spread["hour"].between(16, 20)]["spread_signal"].abs()
        if not peak.empty:
            peak_spread = float(peak.mean())
    congestion_pressure = "Moderate"
    if has_cols(master, ["congestion_price_day_ahead"]):
        congestion = master["congestion_price_day_ahead"].abs()
        recent_congestion = congestion.tail(168).mean()
        ratio = recent_congestion / congestion.mean() if congestion.mean() else 1
        congestion_pressure = "High" if ratio > 1.35 else "Low" if ratio < 0.85 else "Moderate"
    improvement = np.nan
    if pd.notna(baseline_mae) and baseline_mae:
        improvement = (baseline_mae - mae) / baseline_mae * 100
    if "datetime" in pred.columns and not pred.empty:
        data_range = f"{pred['datetime'].min():%Y-%m-%d} to {pred['datetime'].max():%Y-%m-%d}"
    else:
        data_range = "No active range"
    return {
        "mae": mae,
        "rmse": rmse,
        "precision": precision,
        "recall": recall,
        "vol_7d": vol_7d,
        "regime": regime,
        "peak_spread": peak_spread,
        "congestion_pressure": congestion_pressure,
        "improvement": improvement,
        "data_range": data_range,
    }


def chart_theme(fig, height=430, hovermode="x unified", legend=True, compact_title_gap=False, legend_inside=False):
    title = fig.layout.title.text
    if legend_inside:
        top_margin = 6
    elif legend:
        top_margin = 18 if compact_title_gap or not title else 76
    else:
        top_margin = 8 if not title else 42
    legend_y = 0.995 if legend_inside else 1.005 if compact_title_gap else 1.045
    legend_anchor = "top" if legend_inside else "bottom"
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8, 20, 38, 0.55)",
        font=dict(color="#dbeafe", family="Inter"),
        margin=dict(l=20, r=20, t=top_margin, b=18),
        legend=dict(
            orientation="h",
            yanchor=legend_anchor,
            y=legend_y,
            xanchor="left",
            x=0.01 if legend_inside else 0,
            bgcolor="rgba(0,0,0,0)",
        ),
        height=height,
        hovermode=hovermode,
        showlegend=legend,
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(color="#f8fbff", family="Inter", size=16), x=0.01, xanchor="left", y=0.985))
    else:
        fig.update_layout(title_text="")
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)")
    return fig


def page_header(title, subtitle, stats):
    st.markdown(
        f"""
        <div class="page-head">
            <div>
                <div class="page-title">{title}</div>
                <div class="page-subtitle">{subtitle}</div>
            </div>
            <div class="date-label">Data range: {stats["data_range"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_strip(stats):
    improvement = f"{stats['improvement']:+.1f}% vs persistence" if pd.notna(stats["improvement"]) else "forecast error"
    cols = st.columns(6)
    cards = [
        ("Next-Hour DA LMP MAE", f"{stats['mae']:.2f} $/MWh", improvement, stats["improvement"] < 0 if pd.notna(stats["improvement"]) else False),
        ("RMSE", f"{stats['rmse']:.2f} $/MWh", "test-set error", False),
        ("Spike Precision", f"{stats['precision']:.2f}", "predicted spike quality", False),
        ("Spike Recall", f"{stats['recall']:.2f}", "spike event coverage", True),
        ("Recent Volatility", stats["regime"], f"7-day sigma: {stats['vol_7d']:.2f}", stats["regime"] != "Normal"),
        ("Avg Peak-Hour Spread", f"{stats['peak_spread']:.2f} $/MWh", "hours 16-20 EPT", False),
    ]
    for col, card in zip(cols, cards):
        with col:
            st.markdown(kpi_card(*card), unsafe_allow_html=True)


def forecast_chart(pred, height=468):
    need = ["datetime", "actual_da_lmp", "xgboost_prediction", "persistence_prediction"]
    if not has_cols(pred, need):
        st.info("Forecast output needs datetime, actual_da_lmp, xgboost_prediction, and persistence_prediction.")
        return
    df = pred.sort_values("datetime").tail(640)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["datetime"], y=df["actual_da_lmp"], name="Actual DA LMP", line=dict(color=COLORS["white"], width=2.1)))
    fig.add_trace(go.Scatter(x=df["datetime"], y=df["xgboost_prediction"], name="XGBoost forecast", line=dict(color=COLORS["cyan"], width=2)))
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["persistence_prediction"],
            name="Persistence",
            line=dict(color=COLORS["orange"], width=1.35, dash="dot"),
            opacity=0.80,
        )
    )
    if "predicted_spike_probability" in df.columns:
        high_risk = df[df["predicted_spike_probability"] >= df["predicted_spike_probability"].quantile(0.92)]
        fig.add_trace(
            go.Scatter(
                x=high_risk["datetime"],
                y=high_risk["actual_da_lmp"],
                mode="markers",
                name="High spike probability",
                marker=dict(color=COLORS["red"], size=7, line=dict(color=COLORS["white"], width=0.7)),
            )
        )
    fig.update_layout(title_text="", yaxis_title="$ / MWh")
    st.plotly_chart(chart_theme(fig, height=height, compact_title_gap=True, legend_inside=True), use_container_width=True)


def risk_panel(stats, pred):
    risk_window = "17:00-20:00"
    spike_prob = np.nan
    if has_cols(pred, ["predicted_spike_probability"]):
        spike_prob = pred["predicted_spike_probability"].tail(48).max()
    prob_text = f"{spike_prob:.2f}" if pd.notna(spike_prob) else "N/A"
    st.markdown(
        f"""
        <div class="risk-grid">
            <div class="risk-cell"><div class="risk-label">Volatility</div><div class="risk-value">{stats["regime"]}</div></div>
            <div class="risk-cell"><div class="risk-label">Congestion</div><div class="risk-value">{stats["congestion_pressure"]}</div></div>
            <div class="risk-cell"><div class="risk-label">Peak Window</div><div class="risk-value">{risk_window}</div></div>
            <div class="risk-cell"><div class="risk-label">48h Spike Prob Max</div><div class="risk-value">{prob_text}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    callout("Recent day-ahead price history anchors the short-horizon forecast. Volatility, spikes, and RT-DA spread add risk context.")
    callout("Recall is the active gap: high-confidence spike flags are useful, but some stress events remain unflagged.", warn=True)


def spread_watchlist(spread):
    if not has_cols(spread, ["datetime", "hour", "spread_signal", "load_mw"]):
        st.info("Spread observations need rt_da_spread_signals.csv from the modeling output export.")
        return
    df = spread.copy()
    if "signal_direction" not in df.columns:
        df["signal_direction"] = np.where(df["spread_signal"] > 0, "RT premium", "DA premium")
    df["abs_spread"] = df["spread_signal"].abs()
    rows = []
    for _, row in df.sort_values("abs_spread", ascending=False).head(8).iterrows():
        cls = "spread-up" if row["spread_signal"] >= 0 else "spread-down"
        rows.append(
            "<tr>"
            f"<td>{pd.to_datetime(row['datetime']):%m-%d %H:%M}</td>"
            f"<td>{int(row['hour']):02d}</td>"
            f"<td class='{cls}'>{row['spread_signal']:+.2f}</td>"
            f"<td>{row['load_mw']:,.0f}</td>"
            f"<td>{row['signal_direction']}</td>"
            "</tr>"
        )
    st.markdown(
        "<table class='watch-table'><thead><tr><th>Time</th><th>Hr</th><th>RT-DA</th><th>Load</th><th>State</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>",
        unsafe_allow_html=True,
    )


def aep_hourly_regime(master):
    need = ["hour", "total_lmp_day_ahead", "load_mw"]
    if not has_cols(master, need):
        st.info("AEP hourly regime view needs hour, load_mw, and total_lmp_day_ahead.")
        return
    df = master.dropna(subset=need).copy()
    if "da_price_spike" not in df.columns:
        df["da_price_spike"] = 0
    hourly = (
        df.groupby("hour", as_index=False)
        .agg(
            avg_da_lmp=("total_lmp_day_ahead", "mean"),
            median_load_mw=("load_mw", "median"),
            spike_rate=("da_price_spike", "mean"),
        )
    )
    hourly["spike_rate"] *= 100
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=hourly["hour"],
            y=hourly["avg_da_lmp"],
            name="Avg DA LMP",
            marker=dict(color="rgba(56,189,248,0.72)", line=dict(color=COLORS["cyan"], width=0.6)),
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hourly["hour"],
            y=hourly["spike_rate"],
            name="Spike rate",
            mode="lines+markers",
            line=dict(color=COLORS["orange"], width=2.4),
            marker=dict(size=7, color=COLORS["orange"]),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title_text="",
        xaxis_title="Hour EPT",
        yaxis=dict(title="Avg DA LMP ($ / MWh)"),
        yaxis2=dict(title="Spike rate (%)", overlaying="y", side="right", showgrid=False),
    )
    st.plotly_chart(
        chart_theme(fig, height=520, hovermode="x", legend=True, compact_title_gap=True, legend_inside=True),
        use_container_width=True,
        config={"displayModeBar": False},
    )


def aep_load_price_stress(master):
    need = ["hour", "load_mw", "total_lmp_day_ahead"]
    if not has_cols(master, need):
        st.info("AEP load-price view needs hour, load_mw, and total_lmp_day_ahead.")
        return
    df = master.dropna(subset=need).copy()
    if len(df) > 4500:
        df = df.sample(4500, random_state=129)
    if "da_price_spike" not in df.columns:
        df["da_price_spike"] = 0
    df["Price state"] = np.where(df["da_price_spike"] == 1, "Spike", "Normal")
    fig = px.scatter(
        df,
        x="load_mw",
        y="total_lmp_day_ahead",
        color="Price state",
        color_discrete_map={"Normal": COLORS["blue"], "Spike": COLORS["red"]},
        opacity=0.58,
        hover_data=["hour"],
    )
    fig.add_vline(x=df["load_mw"].quantile(0.90), line_color=COLORS["orange"], line_dash="dash")
    fig.add_hline(y=df["total_lmp_day_ahead"].quantile(0.95), line_color=COLORS["orange"], line_dash="dash")
    fig.update_layout(title_text="", xaxis_title="AEP load (MW)", yaxis_title="DA LMP ($ / MWh)")
    st.plotly_chart(chart_theme(fig, height=480, hovermode="closest", legend=True, legend_inside=True), use_container_width=True)


def rolling_volatility(pred):
    if not has_cols(pred, ["datetime", "actual_da_lmp"]):
        return
    df = pred.sort_values("datetime").copy()
    df["rolling_sigma"] = df["actual_da_lmp"].rolling(48, min_periods=12).std()
    fig = px.line(df, x="datetime", y="rolling_sigma")
    fig.update_traces(line=dict(color=COLORS["orange"], width=2))
    fig.update_layout(title_text="", yaxis_title="Std dev")
    st.plotly_chart(chart_theme(fig, height=315, legend=False), use_container_width=True)


def spread_timeline(spread):
    if not has_cols(spread, ["datetime", "spread_signal"]):
        return
    df = spread.sort_values("datetime").tail(900).copy()
    q = df["spread_signal"].abs().quantile(0.90)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["datetime"], y=df["spread_signal"], name="RT-DA spread", line=dict(color=COLORS["cyan"], width=1.6)))
    fig.add_hline(y=0, line=dict(color=COLORS["muted"], width=1))
    fig.add_hline(y=q, line=dict(color=COLORS["orange"], dash="dash", width=1))
    fig.add_hline(y=-q, line=dict(color=COLORS["orange"], dash="dash", width=1))
    fig.update_layout(title_text="", yaxis_title="$ / MWh")
    st.plotly_chart(chart_theme(fig, height=315, legend=False), use_container_width=True)


def benchmark_table(benchmark):
    if not has_cols(benchmark, ["model", "MAE", "RMSE"]):
        return
    show = benchmark[["model", "MAE", "RMSE"]].copy()
    show["MAE"] = show["MAE"].map(lambda x: f"{x:.2f}")
    show["RMSE"] = show["RMSE"].map(lambda x: f"{x:.2f}")
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Model", "MAE", "RMSE"],
                    fill_color="rgba(8,20,38,0.98)",
                    line_color="rgba(148,163,184,0.16)",
                    font=dict(color=COLORS["muted"], size=12),
                    align="left",
                    height=34,
                ),
                cells=dict(
                    values=[show["model"], show["MAE"], show["RMSE"]],
                    fill_color=[["rgba(15,23,42,0.86)", "rgba(30,41,78,0.68)"] * len(show)],
                    line_color="rgba(148,163,184,0.10)",
                    font=dict(color=COLORS["text"], size=12),
                    align="left",
                    height=34,
                ),
            )
        ]
    )
    st.plotly_chart(chart_theme(fig, height=210, hovermode=False, legend=False), use_container_width=True)


def confusion_heatmap(cm):
    vals = cm_values(cm)
    if not vals:
        st.info("Confusion matrix output is not available.")
        return
    tn, fp, fn, tp = vals
    z = np.array([[tn, fp], [fn, tp]])
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=["Pred no spike", "Pred spike"],
            y=["Actual no spike", "Actual spike"],
            text=z,
            texttemplate="%{text}",
            colorscale=[[0, "#07111f"], [0.55, "#075985"], [1, "#38bdf8"]],
            showscale=False,
        )
    )
    fig.update_layout(title_text="")
    st.plotly_chart(chart_theme(fig, height=420, hovermode="closest", legend=False), use_container_width=True)


def spike_probability_heatmap(pred):
    if not has_cols(pred, ["datetime", "hour", "predicted_spike_probability"]):
        return
    heat = pred.copy()
    heat["date"] = heat["datetime"].dt.date
    pivot = heat.groupby(["hour", "date"])["predicted_spike_probability"].mean().unstack().sort_index(ascending=False)
    pivot = pivot.iloc[:, -45:] if pivot.shape[1] > 45 else pivot
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=[str(day) for day in pivot.columns],
            y=pivot.index,
            colorscale=[[0, "#07111f"], [0.55, "#0ea5e9"], [1, "#f87171"]],
            colorbar=dict(title="Risk"),
        )
    )
    fig.update_layout(title_text="", yaxis_title="Hour")
    st.plotly_chart(chart_theme(fig, height=420, hovermode="closest", legend=False), use_container_width=True)


def spike_hour_chart(spike):
    if not has_cols(spike, ["hour", "spike_count"]):
        return
    fig = px.bar(spike, x="hour", y="spike_count", color="spike_count", color_continuous_scale=["#0f172a", "#fb923c", "#f87171"])
    fig.update_layout(title_text="", coloraxis_showscale=False)
    st.plotly_chart(chart_theme(fig, height=320, legend=False), use_container_width=True)


def feature_chart(df):
    if not has_cols(df, ["feature", "importance"]):
        st.info("Feature importance export is not available.")
        return
    top = df.sort_values("importance", ascending=False).head(12).sort_values("importance")
    fig = px.bar(top, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale=["#123254", "#38bdf8"])
    fig.update_layout(title_text="", yaxis_title=None, coloraxis_showscale=False)
    st.plotly_chart(chart_theme(fig, height=450, legend=False), use_container_width=True)


def congestion_scatter(master):
    need = ["congestion_price_day_ahead", "total_lmp_day_ahead", "load_mw"]
    if not has_cols(master, need):
        return
    df = master.dropna(subset=need).copy()
    if len(df) > 4000:
        df = df.sample(4000, random_state=129)
    if "da_price_spike" not in df.columns:
        df["da_price_spike"] = 0
    df["Market state"] = np.where(df["da_price_spike"] == 1, "Spike", "Normal")
    fig = px.scatter(
        df,
        x="congestion_price_day_ahead",
        y="total_lmp_day_ahead",
        color="Market state",
        size="load_mw",
        size_max=12,
        opacity=0.62,
        color_discrete_map={"Normal": COLORS["blue"], "Spike": COLORS["red"]},
    )
    fig.add_hline(y=df["total_lmp_day_ahead"].quantile(0.95), line_color=COLORS["orange"], line_dash="dash")
    fig.update_layout(title_text="", xaxis_title="DA congestion component", yaxis_title="DA LMP")
    st.plotly_chart(chart_theme(fig, height=445, hovermode="closest", legend=True, legend_inside=True), use_container_width=True)


def command_center(data):
    stats = market_stats(data)
    page_header(
        "Overview",
        "AEP short-horizon LMP forecasts, spike risk, congestion, and RT-DA spread analysis.",
        stats,
    )
    output_warning(data)
    metric_strip(stats)
    main, risk = st.columns([2.15, 1])
    with main:
        panel_start("Actual vs Predicted Next-Hour DA LMP")
        forecast_chart(data["pred"])
    with risk:
        panel_start("Forecast Context")
        risk_panel(stats, data["pred"])
    left, right = st.columns([1, 1.28])
    with left:
        panel_start("Extreme Spread Observations")
        spread_watchlist(data["spread"])
    with right:
        panel_start("AEP Hourly Price Regime")
        aep_hourly_regime(data["master"])


def forecast_monitor(data):
    stats = market_stats(data)
    page_header("Forecasting", "Forecast series, benchmarks, and recent price dispersion.", stats)
    output_warning(data)
    top, side = st.columns([2.2, 1])
    with top:
        panel_start("Actual vs Predicted Next-Hour DA LMP")
        forecast_chart(data["pred"], height=510)
    with side:
        panel_start("Benchmarks")
        benchmark_table(data["benchmark"])
        callout("Persistence is the bar to beat for next-hour LMP. The model adds context from lag structure, load, weather, and calendar signals.")
    a, b = st.columns(2)
    with a:
        panel_start("48-Hour Rolling LMP Volatility")
        rolling_volatility(data["pred"])
    with b:
        panel_start("RT-DA Spread Timeline")
        spread_timeline(data["spread"])


def spike_surveillance(data):
    stats = market_stats(data)
    page_header("Spike Risk", "Classification quality and hourly risk surface for next-hour DA price spikes.", stats)
    output_warning(data)
    a, b = st.columns([1, 1.25])
    with a:
        panel_start("Spike Classification Confusion Matrix")
        confusion_heatmap(data["cm_selected"])
    with b:
        panel_start("Spike Probability Heatmap")
        spike_probability_heatmap(data["pred"])
    c, d = st.columns([1, 1])
    with c:
        panel_start("Spike Events by Hour")
        spike_hour_chart(data["spike"])
    with d:
        panel_start("Spike Notes")
        callout("Precision keeps false spike alarms controlled during routine price movement.")
        callout("Recall remains the watch item when the desk cares more about missed stress events than extra alerts.", warn=True)


def market_drivers(data):
    stats = market_stats(data)
    page_header("Market Drivers", "Feature importance readout for price level and spike-risk models.", stats)
    output_warning(data)
    a, b = st.columns(2)
    with a:
        panel_start("Regression Feature Importance")
        feature_chart(data["reg_imp"])
    with b:
        panel_start("Spike Classification Feature Importance")
        feature_chart(data["cls_imp"])
    panel_start("Analyst Readout")
    callout("Lagged day-ahead LMP dominates short-horizon price levels; lag-24 captures daily market structure.")
    callout("Volatility, hour structure, load, and temperature remain useful for interpreting stress around spike windows.")


def congestion_watch(data):
    stats = market_stats(data)
    page_header(
        "Market Stress Analysis",
        "Congestion, load pressure, and RT-DA spread behavior during stressed AEP market conditions.",
        stats,
    )
    output_warning(data)
    a, b = st.columns([1.12, 1])
    with a:
        panel_start("Congestion vs Day-Ahead LMP")
        congestion_scatter(data["master"])
    with b:
        panel_start("Extreme RT-DA Spread Hours")
        spread_timeline(data["spread"])
        spread_watchlist(data["spread"])
    panel_start("Load vs Day-Ahead LMP Stress")
    aep_load_price_stress(data["master"])


def model_notes(data):
    stats = market_stats(data)
    page_header("Model Notes", "Pipeline context and analytical interpretation.", stats)
    output_warning(data)
    metric_strip(stats)
    a, b, c = st.columns(3)
    with a:
        panel_start("Market Problem")
        callout("Short-horizon PJM LMP can move with demand ramps, price persistence, congestion, and stressed operating hours.")
    with b:
        panel_start("Model Output")
        callout("The workflow forecasts next-hour day-ahead LMP, scores price spike risk, and benchmarks against persistence.")
    with c:
        panel_start("Analytical Use")
        callout("Use the outputs for exposure review, congestion interpretation, and short-horizon risk analysis; not as a standalone trading rule.")
    panel_start("Pipeline")
    st.markdown(
        '<div class="callout">PJM load + weather + DA/RT LMP &nbsp; -> &nbsp; Snowflake hourly mart &nbsp; -> &nbsp; time-series features &nbsp; -> &nbsp; forecasts + spike flags &nbsp; -> &nbsp; dashboard outputs</div>',
        unsafe_allow_html=True,
    )


raw = load_data()
date_source = raw["pred"]
if "datetime" in date_source.columns and not date_source["datetime"].dropna().empty:
    min_date = date_source["datetime"].min().date()
    max_date = date_source["datetime"].max().date()
else:
    min_date = max_date = pd.Timestamp.today().date()

st.markdown(
    """
    <div class="analytics-header">
        <div>
            <div class="analytics-title">PJM Energy Market Analytics Dashboard</div>
            <div class="analytics-subtitle">AEP Zone | Day-Ahead LMP Forecasting and Spike Risk Analysis</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

pages = [
    "Overview",
    "Forecasting",
    "Spike Risk",
    "Market Drivers",
    "Market Stress",
    "Model Notes",
]
nav, filter_drawer = st.columns([6.4, 1])
with nav:
    page = st.segmented_control("View", pages, default="Overview", label_visibility="collapsed")
start_date = min_date
end_date = max_date
hour_range = (0, 23)
with filter_drawer:
    with st.popover("Filters", use_container_width=True):
        st.markdown("#### Market Slice")
        start_date = st.date_input("Start date", value=start_date, min_value=min_date, max_value=max_date)
        end_date = st.date_input("End date", value=end_date, min_value=min_date, max_value=max_date)
        hour_range = st.slider("Hour of day", 0, 23, hour_range)
        st.caption("Filters apply across analysis views.")
if end_date < start_date:
    st.warning("End date is earlier than start date. Using the selected dates in chronological order.")
    start_date, end_date = end_date, start_date

data = filter_market_data(raw, start_date, end_date, hour_range)

if page == "Overview":
    command_center(data)
elif page == "Forecasting":
    forecast_monitor(data)
elif page == "Spike Risk":
    spike_surveillance(data)
elif page == "Market Drivers":
    market_drivers(data)
elif page == "Market Stress":
    congestion_watch(data)
elif page == "Model Notes":
    model_notes(data)
