# PJM Energy Market Forecasting

This project builds an applied energy market analytics and forecasting workflow using PJM AEP load, weather, and locational marginal price (LMP) data. The goal is to forecast short-horizon electricity demand and day-ahead prices, then interpret the market conditions that drive price risk, spikes, and real-time versus day-ahead spread behavior.

## Project Goals

- Build a reproducible hourly data pipeline for PJM AEP load, weather, and LMP data.
- Forecast next-hour AEP load using calendar, weather, lag, ramp, and rolling-window features.
- Forecast next-hour day-ahead LMP and classify day-ahead price spike risk.
- Interpret model behavior using feature importance and an LMP-lag ablation experiment.
- Extend the forecasting workflow into market analytics: spike timing analysis and RT-DA spread signal construction.

## Data Sources

The project uses public and locally downloaded hourly datasets:

- PJM AEP hourly load data for 2024 and 2025.
- Columbus hourly weather data, including temperature.
- PJM day-ahead and real-time LMP data from PJM Data Miner 2.
- Derived clean dataset: `data/aep_load_weather_hourly_dataset.csv`.

Core files are stored in `data/`.

## Notebook Workflow

Run the notebooks in this order:

1. `notebooks/01_pjm_aep_load_weather_pipeline_eda.ipynb`
   - Loads and cleans AEP hourly load data.
   - Prepares weather data.
   - Merges load and weather at hourly resolution.
   - Performs exploratory analysis on load, temperature, seasonality, and cyclical hour structure.
   - Exports the clean load-weather dataset.

2. `notebooks/02_pjm_aep_load_forecasting_models.ipynb`
   - Engineers next-hour load forecasting features.
   - Builds persistence, Linear Regression, Random Forest, and XGBoost models.
   - Evaluates models using MAE, RMSE, and MAPE.
   - Visualizes forecasts and inspects feature importance.

3. `notebooks/03_pjm_lmp_market_forecasting.ipynb`
   - Loads and cleans PJM day-ahead and real-time LMP data.
   - Merges LMP data with load and weather data.
   - Engineers lagged price, lagged load, rolling price, rolling load, time, and congestion features.
   - Trains XGBoost models for next-hour day-ahead LMP regression and spike classification.
   - Runs feature importance, ablation, diagnostics, spike timing, and RT-DA spread signal analysis.

Each notebook now ends with an `Export Dashboard Outputs` section that saves clean tables and presentation-ready figures to `outputs/`.

## Feature Engineering

The project uses time-series features designed for hourly electricity market data:

- Calendar features: hour, day of week, month, weekend indicator, peak-hour indicator.
- Cyclical features: sine and cosine transforms for hour of day.
- Weather features: temperature and squared temperature.
- Load features: lag-1, lag-24, lag-168, load ramp, rolling mean, and rolling standard deviation.
- LMP features: day-ahead LMP lags, rolling LMP mean and standard deviation.
- Market features: real-time minus day-ahead spread, congestion share, and lagged congestion share.

Train/test splits are chronological rather than random, preserving the time-series structure of the forecasting problem.

## Core Results

### AEP Load Forecasting

The load forecasting models substantially outperform the persistence baseline.

| Model | MAE | RMSE | MAPE | MAE Improvement vs Persistence |
|---|---:|---:|---:|---:|
| Persistence | 301.61 | 384.73 | 1.99% | 0.0% |
| Linear Regression | 165.22 | 215.58 | 1.08% | 45.2% |
| Random Forest | 149.16 | 196.18 | 0.97% | 50.5% |
| XGBoost | 127.88 | 165.62 | 0.84% | 57.6% |

Feature importance shows that next-hour load is strongly driven by short-term persistence. The previous-hour load feature (`lag_1`) is the dominant predictor, followed by short-term load ramp behavior.

### PJM LMP Forecasting

The next-hour day-ahead LMP XGBoost regression model achieves:

- MAE: `6.43`
- RMSE: `10.80`
- Safe MAPE: `13.12%`

The price spike classifier achieves:

- Accuracy: `0.94`
- Precision: `0.74`
- Recall: `0.47`
- F1: `0.57`

The LMP model is primarily driven by recent market price history. The most important regression feature is `total_lmp_day_ahead_lag_1`, followed by `total_lmp_day_ahead_lag_24`.

The project also includes a dedicated LMP benchmark table. A simple current-hour day-ahead LMP persistence benchmark is very strong for one-step-ahead price-level forecasting, with MAE around `5.73`. This is an important market result: short-horizon LMP has strong persistence, so complex models should be evaluated against naive but operationally meaningful baselines.

A confusion matrix is included for the spike classifier to separate false alarms from missed spike events. In the current test period, the model produces relatively few false positives but misses some true spike periods, suggesting that recall would be the main area to improve if the goal were risk-alert coverage.

### LMP Lag Ablation Experiment

To test whether the LMP model was overly dependent on recent prices, a second model was trained after removing all LMP-related lag and rolling features.

| Model | MAE | RMSE | Safe MAPE |
|---|---:|---:|---:|
| Model A: full features | 6.43 | 10.80 | 13.12% |
| Model B: no LMP history features | 14.65 | 19.46 | 30.75% |

Removing LMP history increased MAE by about `8.22`, or `127.9%`. This confirms that recent price history is highly informative for short-horizon day-ahead LMP forecasting, while the no-LMP model shifts toward fundamentals such as load, temperature, and calendar structure.

### Price Spike Analysis

Day-ahead price spikes are not evenly distributed across the day. In the test period, spikes cluster most strongly around morning and evening ramp periods, especially hours `6-7` and `19-20`. This aligns with operational intuition: price stress often appears during high-load or ramping periods when the system is tighter.

The notebook also compares congestion and spread behavior during spike versus non-spike periods. Spike hours have higher average day-ahead LMP, higher average congestion component, and higher load than non-spike hours. This provides a market-structure explanation layer beyond model metrics.

### RT-DA Spread Signal

The project also constructs a simple market signal using the real-time minus day-ahead spread:

```python
spread_signal = total_lmp_real_time - total_lmp_day_ahead
pred_spread = predicted_next_hour_da_lmp - current_day_ahead_lmp
```

Using a 90th-percentile absolute spread threshold of about `23.77 $/MWh`, the test set identifies:

- `143` RT premium / potential DA hedge signals.
- `87` DA premium / potential RT hedge signals.
- `2069` neutral periods.

This is not a production trading strategy, but it demonstrates how forecasting outputs and market spreads can be converted into operational hedging or arbitrage indicators.

## Key Conclusions

- Electricity load is highly persistent at the hourly horizon; lagged load and load ramp features dominate next-hour demand forecasting.
- Tree-based models, especially XGBoost, improve substantially over the persistence baseline for load forecasting.
- Day-ahead LMP is also highly persistent; lagged LMP features are the strongest predictors of next-hour price.
- Current-price persistence is a very strong LMP benchmark and can outperform more complex models on one-step-ahead price-level MAE.
- The LMP ablation experiment shows a clear tradeoff between a highly accurate price-history-driven model and a more fundamentals-driven model.
- Price spikes cluster around ramp and high-stress operating hours.
- Congestion and load are higher during spike periods, supporting a market-stress interpretation.
- RT-DA spread signals provide a practical bridge from forecasting to market risk monitoring and hedging analysis.

## How to Run

1. Install the core Python dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit plotly "snowflake-connector-python[pandas]" python-dotenv
```

2. Open Jupyter or VS Code notebooks from the project root.

3. Run notebooks in order:

```text
notebooks/01_pjm_aep_load_weather_pipeline_eda.ipynb
notebooks/02_pjm_aep_load_forecasting_models.ipynb
notebooks/03_pjm_lmp_market_forecasting.ipynb
```

4. Confirm that the expected CSV files exist in `data/` before running.

5. Optional: regenerate all dashboard and GitHub artifacts in one command:

```bash
python scripts/export_dashboard_outputs.py
```

This writes model predictions, benchmark tables, feature importance files, confusion matrices, spike/congestion summaries, spread signals, and PNG figures to `outputs/`.

6. Launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

The dashboard reads from `outputs/` and falls back gracefully if optional files are missing.

## Snowflake Data Layer

This project includes an optional Snowflake data layer that keeps the existing notebook workflow intact while adding a more industry-style analytics architecture.

- Snowflake stores raw PJM AEP load, Columbus weather, and PJM DA/RT LMP data in `PJM_MARKET_DB.ANALYTICS`.
- `sql/01_create_raw_tables.sql` creates raw tables that closely match the local CSV inputs:
  - `RAW_AEP_LOAD_HOURLY`
  - `RAW_WEATHER_HOURLY`
  - `RAW_LMP_HOURLY`
- `sql/02_create_market_mart.sql` builds `MART_AEP_MARKET_HOURLY`, a cleaned hourly market mart with load, weather, LMP, spread, congestion-share, spike, and calendar fields.
- The notebooks can either use local CSVs or Snowflake. Set `USE_SNOWFLAKE = True` near the top of a notebook to read from `MART_AEP_MARKET_HOURLY`; leave it as `False` to preserve the current local workflow.
- The Streamlit dashboard can later be pointed directly to Snowflake or continue reading exported files from `outputs/`.

Configure credentials through environment variables or a local `.env` file:

```bash
cp .env.example .env
```

Then fill in the Snowflake values. Never commit `.env` or hard-code passwords.

To create tables, upload local CSVs, and build the mart:

```bash
python scripts/upload_to_snowflake.py
```

To export the Snowflake mart back to a local CSV:

```bash
python scripts/query_market_mart.py
```

This saves `outputs/master_market_df_from_snowflake.csv` and prints row count, datetime range, and columns.

## Project Structure

```text
pjm-energy-market-forecasting/
├── data/
│   ├── aep_load_weather_hourly_dataset.csv
│   ├── columbus_hourly_weather_2024_2025.csv
│   ├── pjm_aep_hourly_load_2024.csv
│   ├── pjm_aep_hourly_load_2025.csv
│   ├── rt_da_monthly_lmps_2024.csv
│   └── rt_da_monthly_lmps_2025.csv
├── notebooks/
│   ├── 01_pjm_aep_load_weather_pipeline_eda.ipynb
│   ├── 02_pjm_aep_load_forecasting_models.ipynb
│   └── 03_pjm_lmp_market_forecasting.ipynb
├── app.py
├── outputs/
├── sql/
│   ├── 01_create_raw_tables.sql
│   └── 02_create_market_mart.sql
├── scripts/
│   ├── export_dashboard_outputs.py
│   ├── query_market_mart.py
│   └── upload_to_snowflake.py
└── README.md
```

## Limitations and Next Steps

- The current workflow focuses on the PJM AEP zone rather than all PJM zones.
- Public data does not fully capture outages, generation stack, transmission constraints, fuel prices, or renewable production.
- The LMP model relies heavily on recent price history, which is useful for short-horizon accuracy but may be less robust during structural market shifts.
- The spike classifier has stronger precision than recall, meaning it is better at confirming high-risk periods than capturing every spike.
- A production version should include walk-forward validation, multi-zone modeling, weather forecasts, outage data, transaction costs, and formal PnL backtesting for spread signals.
