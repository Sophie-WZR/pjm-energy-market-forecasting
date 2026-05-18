from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd

Path("/private/tmp/pjm_mpl_cache").mkdir(parents=True, exist_ok=True)
Path("/private/tmp/pjm_xdg_cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/pjm_mpl_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/pjm_xdg_cache")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier, XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def safe_mape(y_true, y_pred):
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    mask = np.abs(y_true) > 1
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan


def export_load_weather_outputs():
    load_weather_path = DATA_DIR / "aep_load_weather_hourly_dataset.csv"
    if not load_weather_path.exists():
        print(f"Skipping load/weather exports; missing {load_weather_path}")
        return

    df = pd.read_csv(load_weather_path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)

    df.to_csv(OUTPUT_DIR / "load_weather_hourly_dataset.csv", index=False)
    df[["load_mw", "temperature_2m"]].describe().T.to_csv(OUTPUT_DIR / "load_weather_summary.csv")

    hourly_profile = df.groupby(df["datetime"].dt.hour)["load_mw"].mean().reset_index()
    hourly_profile.columns = ["hour", "avg_load_mw"]
    hourly_profile.to_csv(OUTPUT_DIR / "hourly_load_profile.csv", index=False)

    monthly_profile = df.groupby(df["datetime"].dt.month)["load_mw"].mean().reset_index()
    monthly_profile.columns = ["month", "avg_load_mw"]
    monthly_profile.to_csv(OUTPUT_DIR / "monthly_load_profile.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=hourly_profile, x="hour", y="avg_load_mw", marker="o", ax=ax)
    ax.set_title("Average AEP Load by Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Load (MW)")
    savefig(OUTPUT_DIR / "hourly_load_profile.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df.sample(n=min(1200, len(df)), random_state=42),
        x="temperature_2m",
        y="load_mw",
        alpha=0.5,
        ax=ax,
    )
    ax.set_title("AEP Load vs Temperature")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Load (MW)")
    savefig(OUTPUT_DIR / "load_vs_temperature.png")


def engineer_load_features(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].between(7, 22).astype(int)
    df["temp_sq"] = df["temperature_2m"] ** 2
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["lag_1"] = df["load_mw"].shift(1)
    df["lag_24"] = df["load_mw"].shift(24)
    df["lag_168"] = df["load_mw"].shift(168)
    df["load_ramp_1"] = df["load_mw"] - df["lag_1"]
    df["rolling_mean_24"] = df["load_mw"].shift(1).rolling(24).mean()
    df["rolling_std_24"] = df["load_mw"].shift(1).rolling(24).std()
    df["target_load_mw"] = df["load_mw"].shift(-1)
    return df.dropna().reset_index(drop=True)


def export_load_model_outputs():
    load_weather_path = DATA_DIR / "aep_load_weather_hourly_dataset.csv"
    if not load_weather_path.exists():
        print(f"Skipping load model exports; missing {load_weather_path}")
        return

    market_df = engineer_load_features(pd.read_csv(load_weather_path))
    split_loc = int(len(market_df) * 0.8)
    train_df = market_df.iloc[:split_loc].copy()
    test_df = market_df.iloc[split_loc:].copy()

    feature_columns = [
        "temperature_2m",
        "temp_sq",
        "sin_hour",
        "cos_hour",
        "dayofweek",
        "month",
        "is_weekend",
        "is_peak_hour",
        "lag_1",
        "lag_24",
        "lag_168",
        "load_ramp_1",
        "rolling_mean_24",
        "rolling_std_24",
    ]
    X_train, X_test = train_df[feature_columns], test_df[feature_columns]
    y_train, y_test = train_df["target_load_mw"], test_df["target_load_mw"]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=0),
    }
    rows = []
    predictions = test_df[["datetime", "target_load_mw"]].rename(columns={"target_load_mw": "actual_load_mw"})
    predictions["persistence_prediction"] = test_df["load_mw"]

    rows.append(
        {
            "model": "Persistence",
            "MAE": mae(y_test, predictions["persistence_prediction"]),
            "RMSE": rmse(y_test, predictions["persistence_prediction"]),
            "MAPE": np.mean(np.abs((y_test - predictions["persistence_prediction"]) / y_test)) * 100,
        }
    )

    for name, model in models.items():
        model.fit(X_train, y_train)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pred = model.predict(X_test)
        predictions[f"{name.lower().replace(' ', '_')}_prediction"] = pred
        rows.append(
            {
                "model": name,
                "MAE": mae(y_test, pred),
                "RMSE": rmse(y_test, pred),
                "MAPE": np.mean(np.abs((y_test - pred) / y_test)) * 100,
            }
        )

    benchmark = pd.DataFrame(rows)
    baseline_mae = benchmark.loc[benchmark["model"] == "Persistence", "MAE"].iloc[0]
    benchmark["MAE_improvement_vs_baseline_pct"] = (baseline_mae - benchmark["MAE"]) / baseline_mae * 100
    benchmark.to_csv(OUTPUT_DIR / "load_forecast_benchmark_results.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "load_forecast_predictions.csv", index=False)

    rf_model = models["Random Forest"]
    xgb_model = models["XGBoost"]
    rf_importance = pd.DataFrame({"feature": feature_columns, "importance": rf_model.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    xgb_importance = pd.DataFrame({"feature": feature_columns, "importance": xgb_model.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    rf_importance.to_csv(OUTPUT_DIR / "load_rf_feature_importance.csv", index=False)
    xgb_importance.to_csv(OUTPUT_DIR / "load_xgb_feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    sample = predictions.tail(240)
    ax.plot(sample["datetime"], sample["actual_load_mw"], label="Actual", color="black", linewidth=1)
    ax.plot(sample["datetime"], sample["xgboost_prediction"], label="XGBoost", alpha=0.8)
    ax.plot(sample["datetime"], sample["persistence_prediction"], label="Persistence", alpha=0.6)
    ax.set_title("AEP Load Forecast: Actual vs Predicted")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Load (MW)")
    ax.legend()
    savefig(OUTPUT_DIR / "load_forecast_actual_vs_predicted.png")


def load_clean_lmp_data(paths):
    market_df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    market_df = market_df.rename(
        columns={
            "datetime_beginning_ept": "datetime",
            "total_lmp_da": "total_lmp_day_ahead",
            "total_lmp_rt": "total_lmp_real_time",
            "congestion_price_da": "congestion_price_day_ahead",
            "congestion_price_rt": "congestion_price_real_time",
            "marginal_loss_price_da": "marginal_loss_price_day_ahead",
            "marginal_loss_price_rt": "marginal_loss_price_real_time",
        }
    )
    market_df["datetime"] = pd.to_datetime(market_df["datetime"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    market_df = market_df.sort_values("datetime").reset_index(drop=True)
    keep_cols = [
        "datetime",
        "total_lmp_day_ahead",
        "total_lmp_real_time",
        "congestion_price_day_ahead",
        "congestion_price_real_time",
        "marginal_loss_price_day_ahead",
        "marginal_loss_price_real_time",
    ]
    return market_df[keep_cols].copy()


def engineer_market_features(df):
    df = df.copy()
    df["rt_da_spread"] = df["total_lmp_real_time"] - df["total_lmp_day_ahead"]
    df["congestion_share_da"] = df["congestion_price_day_ahead"] / df["total_lmp_day_ahead"]
    df["congestion_share_rt"] = df["congestion_price_real_time"] / df["total_lmp_real_time"]
    df[["congestion_share_da", "congestion_share_rt"]] = df[["congestion_share_da", "congestion_share_rt"]].replace(
        [np.inf, -np.inf], np.nan
    )
    df["da_price_spike"] = (df["total_lmp_day_ahead"] > df["total_lmp_day_ahead"].quantile(0.95)).astype(int)
    df["rt_price_spike"] = (df["total_lmp_real_time"] > df["total_lmp_real_time"].quantile(0.95)).astype(int)
    return df


def engineer_lmp_forecasting_features(df):
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    for var in ["load_mw", "total_lmp_day_ahead"]:
        for lag in [1, 24, 168]:
            df[f"{var}_lag_{lag}"] = df[var].shift(lag)
    df["load_mw_rolling_mean_24"] = df["load_mw"].shift(1).rolling(24).mean()
    df["load_mw_rolling_std_24"] = df["load_mw"].shift(1).rolling(24).std()
    df["da_lmp_rolling_mean_24"] = df["total_lmp_day_ahead"].shift(1).rolling(24).mean()
    df["da_lmp_rolling_std_24"] = df["total_lmp_day_ahead"].shift(1).rolling(24).std()
    df["congestion_share_da_lag_1"] = df["congestion_share_da"].shift(1)
    return df


def export_lmp_model_outputs():
    lmp_paths = [DATA_DIR / "rt_da_monthly_lmps_2024.csv", DATA_DIR / "rt_da_monthly_lmps_2025.csv"]
    load_weather_path = DATA_DIR / "aep_load_weather_hourly_dataset.csv"
    if not all(path.exists() for path in lmp_paths) or not load_weather_path.exists():
        print("Skipping LMP exports; missing LMP or load/weather data.")
        return

    market_df = engineer_market_features(load_clean_lmp_data(lmp_paths))
    load_weather_df = pd.read_csv(load_weather_path)
    load_weather_df["datetime"] = pd.to_datetime(load_weather_df["datetime"], errors="coerce")
    master_df = market_df.merge(load_weather_df[["datetime", "load_mw", "temperature_2m"]], on="datetime", how="inner")
    master_df = engineer_lmp_forecasting_features(master_df)
    master_df["target_da_lmp_next_hour"] = master_df["total_lmp_day_ahead"].shift(-1)
    master_df["target_da_spike_next_hour"] = master_df["da_price_spike"].shift(-1)

    regression_features = [
        "hour",
        "dayofweek",
        "month",
        "is_weekend",
        "sin_hour",
        "cos_hour",
        "temperature_2m",
        "load_mw",
        "load_mw_lag_1",
        "load_mw_lag_24",
        "load_mw_lag_168",
        "total_lmp_day_ahead_lag_1",
        "total_lmp_day_ahead_lag_24",
        "total_lmp_day_ahead_lag_168",
        "load_mw_rolling_mean_24",
        "load_mw_rolling_std_24",
        "da_lmp_rolling_mean_24",
        "da_lmp_rolling_std_24",
        "congestion_share_da_lag_1",
    ]
    master_df = master_df.dropna(subset=regression_features + ["target_da_lmp_next_hour", "target_da_spike_next_hour"])
    master_df = master_df.reset_index(drop=True)
    master_df.to_csv(OUTPUT_DIR / "master_market_df.csv", index=False)

    split_loc = int(len(master_df) * 0.8)
    train_df = master_df.iloc[:split_loc].copy()
    test_df = master_df.iloc[split_loc:].copy()
    X_train = train_df[regression_features]
    X_test = test_df[regression_features]
    y_train_reg = train_df["target_da_lmp_next_hour"]
    y_test_reg = test_df["target_da_lmp_next_hour"]
    y_train_cls = train_df["target_da_spike_next_hour"].astype(int)
    y_test_cls = test_df["target_da_spike_next_hour"].astype(int)

    reg_model = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=0)
    cls_model = XGBClassifier(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1, eval_metric="logloss", verbosity=0)
    reg_model.fit(X_train, y_train_reg)
    cls_model.fit(X_train, y_train_cls)
    y_pred_reg = reg_model.predict(X_test)
    y_prob_cls = cls_model.predict_proba(X_test)[:, 1]
    y_pred_cls = cls_model.predict(X_test)

    pred_df = test_df[
        [
            "datetime",
            "hour",
            "load_mw",
            "temperature_2m",
            "total_lmp_day_ahead",
            "total_lmp_real_time",
            "congestion_price_day_ahead",
            "marginal_loss_price_day_ahead",
            "rt_da_spread",
            "da_price_spike",
            "target_da_lmp_next_hour",
            "target_da_spike_next_hour",
        ]
    ].copy()
    pred_df = pred_df.rename(
        columns={
            "target_da_lmp_next_hour": "actual_da_lmp",
            "target_da_spike_next_hour": "actual_spike",
        }
    )
    pred_df["xgboost_prediction"] = y_pred_reg
    pred_df["persistence_prediction"] = pred_df["total_lmp_day_ahead"]
    pred_df["predicted_spike_probability"] = y_prob_cls
    pred_df["predicted_spike"] = y_pred_cls
    pred_df.to_csv(OUTPUT_DIR / "model_predictions.csv", index=False)

    reg_importance = pd.DataFrame({"feature": regression_features, "importance": reg_model.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    cls_importance = pd.DataFrame({"feature": regression_features, "importance": cls_model.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    reg_importance.to_csv(OUTPUT_DIR / "regression_feature_importance.csv", index=False)
    cls_importance.to_csv(OUTPUT_DIR / "classification_feature_importance.csv", index=False)

    features_no_lmp = [feature for feature in regression_features if "lmp" not in feature]
    reg_model_no_lmp = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=0)
    reg_model_no_lmp.fit(train_df[features_no_lmp], y_train_reg)
    pred_no_lmp = reg_model_no_lmp.predict(test_df[features_no_lmp])

    persistence_pred = test_df["total_lmp_day_ahead"]
    benchmark = pd.DataFrame(
        [
            {
                "model": "Persistence: current DA LMP",
                "MAE": mae(y_test_reg, persistence_pred),
                "RMSE": rmse(y_test_reg, persistence_pred),
                "Safe MAPE (abs > 1)": safe_mape(y_test_reg, persistence_pred),
            },
            {
                "model": "XGBoost: full features",
                "MAE": mae(y_test_reg, y_pred_reg),
                "RMSE": rmse(y_test_reg, y_pred_reg),
                "Safe MAPE (abs > 1)": safe_mape(y_test_reg, y_pred_reg),
            },
            {
                "model": "XGBoost: no LMP history",
                "MAE": mae(y_test_reg, pred_no_lmp),
                "RMSE": rmse(y_test_reg, pred_no_lmp),
                "Safe MAPE (abs > 1)": safe_mape(y_test_reg, pred_no_lmp),
            },
        ]
    )
    baseline_mae = benchmark.loc[benchmark["model"] == "Persistence: current DA LMP", "MAE"].iloc[0]
    benchmark["MAE_improvement_vs_persistence_pct"] = (baseline_mae - benchmark["MAE"]) / baseline_mae * 100
    benchmark.to_csv(OUTPUT_DIR / "benchmark_results.csv", index=False)

    cm = confusion_matrix(y_test_cls, y_pred_cls)
    pd.DataFrame(cm, index=["Actual No Spike", "Actual Spike"], columns=["Predicted No Spike", "Predicted Spike"]).to_csv(
        OUTPUT_DIR / "spike_confusion_matrix_default.csv"
    )

    def eval_spike(name, probs, threshold):
        pred = (probs >= threshold).astype(int)
        cm_local = confusion_matrix(y_test_cls, pred)
        tn, fp, fn, tp = cm_local.ravel()
        return {
            "model": name,
            "threshold": threshold,
            "accuracy": accuracy_score(y_test_cls, pred),
            "precision": precision_score(y_test_cls, pred, zero_division=0),
            "recall": recall_score(y_test_cls, pred, zero_division=0),
            "f1": f1_score(y_test_cls, pred, zero_division=0),
            "pr_auc": average_precision_score(y_test_cls, probs),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }

    threshold_rows = [eval_spike("Default XGBoost threshold search", y_prob_cls, threshold) for threshold in np.arange(0.05, 0.96, 0.01)]
    threshold_df = pd.DataFrame(threshold_rows)
    selected = threshold_df.loc[threshold_df["f1"].idxmax()]
    selected_pred = (y_prob_cls >= selected["threshold"]).astype(int)
    selected_cm = confusion_matrix(y_test_cls, selected_pred)
    pd.DataFrame(
        selected_cm,
        index=["Actual No Spike", "Actual Spike"],
        columns=["Predicted No Spike", "Predicted Spike"],
    ).to_csv(OUTPUT_DIR / "spike_confusion_matrix_selected.csv")
    threshold_df.to_csv(OUTPUT_DIR / "spike_threshold_tuning_results.csv", index=False)

    spike_by_hour = test_df[test_df["da_price_spike"] == 1]["hour"].value_counts().sort_index().reindex(range(24), fill_value=0)
    spike_analysis = spike_by_hour.reset_index()
    spike_analysis.columns = ["hour", "spike_count"]
    spike_analysis.to_csv(OUTPUT_DIR / "spike_analysis.csv", index=False)

    congestion_summary = test_df.groupby("da_price_spike")[
        [
            "total_lmp_day_ahead",
            "congestion_price_day_ahead",
            "marginal_loss_price_day_ahead",
            "rt_da_spread",
            "load_mw",
        ]
    ].agg(["mean", "median", "std"])
    congestion_summary.to_csv(OUTPUT_DIR / "congestion_stress_summary.csv")

    signal_df = pred_df[["datetime", "hour", "total_lmp_day_ahead", "total_lmp_real_time", "rt_da_spread", "load_mw"]].copy()
    signal_df["pred_next_hour_da_lmp"] = y_pred_reg
    signal_df["pred_spread"] = signal_df["pred_next_hour_da_lmp"] - signal_df["total_lmp_day_ahead"]
    signal_df["spread_signal"] = signal_df["rt_da_spread"]
    spread_threshold = signal_df["spread_signal"].abs().quantile(0.90)
    signal_df["signal_direction"] = np.select(
        [signal_df["spread_signal"] >= spread_threshold, signal_df["spread_signal"] <= -spread_threshold],
        ["RT premium / potential DA hedge", "DA premium / potential RT hedge"],
        default="Neutral",
    )
    signal_df.to_csv(OUTPUT_DIR / "rt_da_spread_signals.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    sample = pred_df.tail(500)
    ax.plot(sample["datetime"], sample["actual_da_lmp"], label="Actual Next-Hour DA LMP", color="black", linewidth=1)
    ax.plot(sample["datetime"], sample["xgboost_prediction"], label="XGBoost", alpha=0.8)
    ax.plot(sample["datetime"], sample["persistence_prediction"], label="Persistence", alpha=0.7)
    ax.set_title("Next-Hour DA LMP: Actual vs Predicted")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("DA LMP ($/MWh)")
    ax.legend()
    savefig(OUTPUT_DIR / "lmp_actual_vs_predicted.png")

    for importance_df, filename, title in [
        (reg_importance, "lmp_regression_feature_importance.png", "Regression Feature Importance"),
        (cls_importance, "lmp_classification_feature_importance.png", "Classification Feature Importance"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 6))
        top = importance_df.head(12).sort_values("importance")
        ax.barh(top["feature"], top["importance"], color="steelblue")
        ax.set_title(title)
        ax.set_xlabel("Importance")
        savefig(OUTPUT_DIR / filename)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=selected_cm, display_labels=["No Spike", "Spike"]).plot(
        cmap="Blues", values_format="d", ax=ax, colorbar=False
    )
    ax.set_title(f"Selected Spike Classifier Confusion Matrix\nDefault XGBoost threshold={selected['threshold']:.2f}")
    savefig(OUTPUT_DIR / "spike_confusion_matrix_selected.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=spike_analysis["hour"], y=spike_analysis["spike_count"], color="firebrick", ax=ax)
    ax.set_title("Day-Ahead Price Spike Count by Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Spike Count")
    savefig(OUTPUT_DIR / "spike_count_by_hour.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=test_df.sample(n=min(1000, len(test_df)), random_state=42),
        x="load_mw",
        y="total_lmp_day_ahead",
        hue="hour",
        palette="turbo",
        alpha=0.7,
        ax=ax,
    )
    ax.set_title("Load vs Day-Ahead LMP")
    ax.set_xlabel("Load (MW)")
    ax.set_ylabel("DA LMP ($/MWh)")
    savefig(OUTPUT_DIR / "load_vs_da_lmp.png")

    hourly_lmp = test_df.groupby("hour")["total_lmp_day_ahead"].mean().reset_index()
    hourly_lmp.to_csv(OUTPUT_DIR / "hourly_da_lmp_profile.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=hourly_lmp, x="hour", y="total_lmp_day_ahead", marker="o", ax=ax)
    ax.set_title("Hourly Average Day-Ahead LMP Profile")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average DA LMP ($/MWh)")
    savefig(OUTPUT_DIR / "hourly_da_lmp_profile.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=test_df, x="da_price_spike", y="congestion_price_day_ahead", color="steelblue", ax=ax)
    ax.set_title("DA Congestion Component: Spike vs Non-Spike Hours")
    ax.set_xlabel("DA Price Spike")
    ax.set_ylabel("DA Congestion Price ($/MWh)")
    savefig(OUTPUT_DIR / "congestion_spike_boxplot.png")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(signal_df["datetime"], signal_df["spread_signal"], label="RT - DA Spread", color="steelblue", linewidth=1)
    ax.axhline(spread_threshold, color="firebrick", linestyle="--", linewidth=1, label="90th percentile threshold")
    ax.axhline(-spread_threshold, color="firebrick", linestyle="--", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Real-Time vs Day-Ahead Spread Signal")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("RT - DA Spread ($/MWh)")
    ax.legend()
    savefig(OUTPUT_DIR / "rt_da_spread_signal.png")


def main():
    export_load_weather_outputs()
    export_load_model_outputs()
    export_lmp_model_outputs()
    print(f"Dashboard outputs exported to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
