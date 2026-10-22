# Databricks notebook source
# MAGIC %md
# MAGIC # RetailPulse — 04 Forecasting
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC This notebook develops the predictive layer of RetailPulse using the complete
# MAGIC historical period **2000–2024**.
# MAGIC
# MAGIC Business question:
# MAGIC
# MAGIC > **What might future market conditions look like for each country?**
# MAGIC
# MAGIC The notebook intentionally uses a lightweight, interpretable forecasting
# MAGIC framework rather than expensive per-country time-series models.
# MAGIC
# MAGIC ### Forecasting design
# MAGIC
# MAGIC - Historical data: **2000–2024**
# MAGIC - Training period: **2000–2019**
# MAGIC - Validation period: **2020–2024**
# MAGIC - Forecast horizon: **2025–2029**
# MAGIC - Targets:
# MAGIC   - GDP_USD
# MAGIC   - GDP_PER_CAPITA_USD
# MAGIC   - POPULATION
# MAGIC   - URBAN_POPULATION_PCT
# MAGIC   - INTERNET_USERS_PCT
# MAGIC - Candidate models:
# MAGIC   - Naive
# MAGIC   - Linear Trend
# MAGIC - Final model is selected **from validation performance**, not hard-coded.
# MAGIC
# MAGIC The forecasting output is intended for decision support, not as a guarantee of
# MAGIC future economic outcomes.
# MAGIC

# COMMAND ----------

# ============================================================
# 1. Setup and load only required data
# ============================================================

import numpy as np
import pandas as pd

from pyspark.sql import functions as F

SOURCE_TABLE = (
    "retailpulse_snowflake_catalog.modeled.country_economic_trends"
)

SEGMENT_TABLE = (
    "retailpulse_snowflake_catalog.modeled.country_market_segments"
)

FORECAST_TABLE = (
    "retailpulse_snowflake_catalog.modeled.market_forecasts"
)

FORECAST_TARGETS = [
    "GDP_USD",
    "GDP_PER_CAPITA_USD",
    "POPULATION",
    "URBAN_POPULATION_PCT",
    "INTERNET_USERS_PCT"
]

TRAIN_START_YEAR = 2000
TRAIN_END_YEAR = 2019
VALIDATION_START_YEAR = 2020
VALIDATION_END_YEAR = 2024

FORECAST_START_YEAR = 2025
FORECAST_END_YEAR = 2029
MIN_OBSERVATIONS = 20

# Only select columns required by this notebook.
forecast_pdf = (
    spark.table(SOURCE_TABLE)
    .select(
        "COUNTRY_ISO3",
        "COUNTRY_NAME",
        "YEAR",
        *FORECAST_TARGETS
    )
    .toPandas()
)

forecast_pdf["YEAR"] = forecast_pdf["YEAR"].astype(int)

print("Forecasting dataset loaded.")
print("Rows:", len(forecast_pdf))
print("Columns:", len(forecast_pdf.columns))
print("Historical years:", forecast_pdf["YEAR"].min(), "to", forecast_pdf["YEAR"].max())
print("Countries:", forecast_pdf["COUNTRY_ISO3"].nunique())


# COMMAND ----------

# ============================================================
# 2. Source and target validation
# ============================================================

assert forecast_pdf["YEAR"].min() == TRAIN_START_YEAR
assert forecast_pdf["YEAR"].max() == VALIDATION_END_YEAR

assert forecast_pdf[
    ["COUNTRY_ISO3", "YEAR"]
].duplicated().sum() == 0, "Duplicate country-year rows detected."

print("✓ Historical period:", TRAIN_START_YEAR, "–", VALIDATION_END_YEAR)
print("✓ Country-year grain validated.")
print("✓ Forecast targets:", ", ".join(FORECAST_TARGETS))

coverage = (
    forecast_pdf
    .groupby("COUNTRY_ISO3")[FORECAST_TARGETS]
    .count()
)

coverage_summary = pd.DataFrame({
    "INDICATOR": FORECAST_TARGETS,
    "COUNTRIES_WITH_20_PLUS_OBS": [
        int((coverage[target] >= MIN_OBSERVATIONS).sum())
        for target in FORECAST_TARGETS
    ],
    "TOTAL_COUNTRIES": len(coverage)
})

coverage_summary["EXCLUDED_COUNTRIES"] = (
    coverage_summary["TOTAL_COUNTRIES"]
    - coverage_summary["COUNTRIES_WITH_20_PLUS_OBS"]
)

display(coverage_summary)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Forecastability and eligibility
# MAGIC
# MAGIC A country-indicator series is eligible when it contains at least **20 observed
# MAGIC annual values between 2000 and 2024**.
# MAGIC
# MAGIC This is applied separately to each target, so a country may be forecast for one
# MAGIC indicator while being excluded for another.
# MAGIC
# MAGIC No imputation is performed for forecasting targets. Forecasts are based only on
# MAGIC the observed history of each eligible country-indicator series.

# COMMAND ----------

# ============================================================
# 4. Build long-format modeling data once
# ============================================================

long_pdf = forecast_pdf.melt(
    id_vars=["COUNTRY_ISO3", "COUNTRY_NAME", "YEAR"],
    value_vars=FORECAST_TARGETS,
    var_name="INDICATOR",
    value_name="VALUE"
)

series_counts = (
    long_pdf
    .dropna(subset=["VALUE"])
    .groupby(["COUNTRY_ISO3", "INDICATOR"])
    .size()
    .rename("OBSERVATIONS")
    .reset_index()
)

eligible_series = series_counts[
    series_counts["OBSERVATIONS"] >= MIN_OBSERVATIONS
].copy()

eligible_keys = set(
    zip(
        eligible_series["COUNTRY_ISO3"],
        eligible_series["INDICATOR"]
    )
)

print("Eligible country-indicator series:", len(eligible_series))

display(
    eligible_series
    .groupby("INDICATOR")
    .size()
    .reset_index(name="ELIGIBLE_COUNTRIES")
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Time-based train / validation strategy
# MAGIC
# MAGIC The validation period is kept completely after the training period.
# MAGIC
# MAGIC - **Training:** 2000–2019
# MAGIC - **Validation:** 2020–2024
# MAGIC - **Future forecast:** 2025–2029
# MAGIC
# MAGIC This prevents information from later years leaking into model training.
# MAGIC

# COMMAND ----------

# ============================================================
# 6. Fast forecasting functions
# ============================================================

def naive_predict(train_values, horizon):
    """Last observed training value carried forward."""
    return np.repeat(float(train_values[-1]), horizon)


def linear_trend_predict(train_years, train_values, future_years):
    """Simple linear trend using a closed-form regression."""
    x = np.asarray(train_years, dtype=float)
    y = np.asarray(train_values, dtype=float)
    xf = np.asarray(future_years, dtype=float)

    x_mean = x.mean()
    y_mean = y.mean()

    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        return np.repeat(y_mean, len(xf))

    slope = np.sum((x - x_mean) * (y - y_mean)) / denominator
    intercept = y_mean - slope * x_mean

    return intercept + slope * xf


def mae(actual, predicted):
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual, predicted):
    """MAPE with zero actual values excluded."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mask = actual != 0

    if not mask.any():
        return np.nan

    return float(
        np.mean(
            np.abs(
                (actual[mask] - predicted[mask])
                / actual[mask]
            )
        ) * 100
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Validation: Naive vs Linear Trend
# MAGIC
# MAGIC Each eligible country-indicator series is evaluated independently.
# MAGIC
# MAGIC The comparison uses:
# MAGIC
# MAGIC - MAE
# MAGIC - RMSE
# MAGIC - MAPE
# MAGIC
# MAGIC For selecting the model for an indicator, **median country MAPE** is used as
# MAGIC the primary metric because raw MAE is strongly affected by country size
# MAGIC differences, especially for GDP.
# MAGIC
# MAGIC Mean MAE and RMSE remain available for interpretation.
# MAGIC

# COMMAND ----------

# ============================================================
# 8. Evaluate both candidate models
# ============================================================

evaluation_rows = []

for (country, indicator), group in long_pdf.dropna(
    subset=["VALUE"]
).groupby(["COUNTRY_ISO3", "INDICATOR"], sort=False):

    if (country, indicator) not in eligible_keys:
        continue

    train = group[
        (group["YEAR"] >= TRAIN_START_YEAR) &
        (group["YEAR"] <= TRAIN_END_YEAR)
    ].sort_values("YEAR")

    validation = group[
        (group["YEAR"] >= VALIDATION_START_YEAR) &
        (group["YEAR"] <= VALIDATION_END_YEAR)
    ].sort_values("YEAR")

    if len(train) < 15 or len(validation) == 0:
        continue

    train_years = train["YEAR"].to_numpy()
    train_values = train["VALUE"].to_numpy(dtype=float)

    validation_years = validation["YEAR"].to_numpy()
    actual = validation["VALUE"].to_numpy(dtype=float)

    naive_pred = naive_predict(
        train_values,
        len(validation_years)
    )

    trend_pred = linear_trend_predict(
        train_years,
        train_values,
        validation_years
    )

    evaluation_rows.append({
        "COUNTRY_ISO3": country,
        "COUNTRY_NAME": train["COUNTRY_NAME"].iloc[0],
        "INDICATOR": indicator,
        "NAIVE_MAE": mae(actual, naive_pred),
        "NAIVE_RMSE": rmse(actual, naive_pred),
        "NAIVE_MAPE": mape(actual, naive_pred),
        "TREND_MAE": mae(actual, trend_pred),
        "TREND_RMSE": rmse(actual, trend_pred),
        "TREND_MAPE": mape(actual, trend_pred)
    })

evaluation_pdf = pd.DataFrame(evaluation_rows)

print("Country-indicator validation series:", len(evaluation_pdf))
display(evaluation_pdf.head())


# COMMAND ----------

# ============================================================
# 9. Select the best model for each indicator
# ============================================================

model_comparison_rows = []

for indicator in FORECAST_TARGETS:

    e = evaluation_pdf[
        evaluation_pdf["INDICATOR"] == indicator
    ].copy()

    if e.empty:
        continue

    naive_mape = e["NAIVE_MAPE"].median()
    trend_mape = e["TREND_MAPE"].median()

    naive_mae = e["NAIVE_MAE"].mean()
    trend_mae = e["TREND_MAE"].mean()

    naive_rmse = e["NAIVE_RMSE"].mean()
    trend_rmse = e["TREND_RMSE"].mean()

    naive_wins = int(
        (e["NAIVE_MAPE"] < e["TREND_MAPE"]).sum()
    )

    trend_wins = int(
        (e["TREND_MAPE"] < e["NAIVE_MAPE"]).sum()
    )

    selected_model = (
        "Linear Trend"
        if trend_mape < naive_mape
        else "Naive"
    )

    model_comparison_rows.append({
        "INDICATOR": indicator,
        "NAIVE_MEDIAN_MAPE": naive_mape,
        "TREND_MEDIAN_MAPE": trend_mape,
        "NAIVE_MEAN_MAE": naive_mae,
        "TREND_MEAN_MAE": trend_mae,
        "NAIVE_MEAN_RMSE": naive_rmse,
        "TREND_MEAN_RMSE": trend_rmse,
        "NAIVE_COUNTRY_WINS": naive_wins,
        "TREND_COUNTRY_WINS": trend_wins,
        "SELECTED_MODEL": selected_model
    })

model_comparison_pdf = pd.DataFrame(model_comparison_rows)

display(model_comparison_pdf)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Model selection interpretation
# MAGIC
# MAGIC The selected model is determined from the 2020–2024 holdout performance.
# MAGIC
# MAGIC A model is **not** selected because it is more sophisticated. If the naive
# MAGIC baseline performs better, it is retained.
# MAGIC
# MAGIC This makes the forecasting layer reproducible and defensible.
# MAGIC

# COMMAND ----------

# ============================================================
# 11. Generate 2025–2029 forecasts using selected models
# ============================================================

selected_models = dict(
    zip(
        model_comparison_pdf["INDICATOR"],
        model_comparison_pdf["SELECTED_MODEL"]
    )
)

forecast_years = np.arange(
    FORECAST_START_YEAR,
    FORECAST_END_YEAR + 1
)

forecast_rows = []

for (country, indicator), group in long_pdf.dropna(
    subset=["VALUE"]
).groupby(["COUNTRY_ISO3", "INDICATOR"], sort=False):

    if (country, indicator) not in eligible_keys:
        continue

    model_name = selected_models.get(indicator)

    if model_name is None:
        continue

    history = group.sort_values("YEAR")

    years = history["YEAR"].to_numpy(dtype=float)
    values = history["VALUE"].to_numpy(dtype=float)

    if model_name == "Naive":
        predictions = naive_predict(
            values,
            len(forecast_years)
        )
    else:
        predictions = linear_trend_predict(
            years,
            values,
            forecast_years
        )

    # All selected targets are non-negative level indicators.
    predictions = np.maximum(predictions, 0)

    country_name = history["COUNTRY_NAME"].iloc[0]

    for year, prediction in zip(forecast_years, predictions):
        forecast_rows.append({
            "COUNTRY_ISO3": country,
            "COUNTRY_NAME": country_name,
            "YEAR": int(year),
            "INDICATOR": indicator,
            "FORECAST_VALUE": float(prediction),
            "MODEL": model_name
        })

forecast_results_pdf = pd.DataFrame(forecast_rows)

print("Forecast rows:", len(forecast_results_pdf))
print("Forecast years:", sorted(forecast_results_pdf["YEAR"].unique()))
print("Countries:", forecast_results_pdf["COUNTRY_ISO3"].nunique())

display(forecast_results_pdf.head(20))


# COMMAND ----------

# ============================================================
# 12. Attach final market segments
# ============================================================

segments_pdf = (
    spark.table("workspace.modeled.market_segments")
    .select(
        "COUNTRY_ISO3",
        "CLUSTER",
        "SEGMENT"
    )
    .toPandas()
)

forecast_results_pdf = forecast_results_pdf.merge(
    segments_pdf,
    on="COUNTRY_ISO3",
    how="inner",
    validate="many_to_one"
)

assert forecast_results_pdf["SEGMENT"].notna().all()

print("✓ Forecasts linked to RetailPulse market segments.")

display(
    forecast_results_pdf[
        [
            "COUNTRY_ISO3",
            "COUNTRY_NAME",
            "CLUSTER",
            "SEGMENT",
            "YEAR",
            "INDICATOR",
            "FORECAST_VALUE",
            "MODEL"
        ]
    ]                   
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Forecast validation
# MAGIC
# MAGIC The final forecast dataset is checked before persistence.
# MAGIC
# MAGIC Required conditions:
# MAGIC
# MAGIC - Forecast years are exactly 2025–2029.
# MAGIC - No duplicate country-year-indicator combinations.
# MAGIC - No missing forecast values.
# MAGIC - All five target indicators are represented.
# MAGIC - Every forecast country has a valid segmentation label.
# MAGIC - Level forecasts are non-negative.

# COMMAND ----------

# ============================================================
# 14. Final forecast validation
# ============================================================

expected_years = set(range(2025, 2030))

assert set(forecast_results_pdf["YEAR"].unique()) == expected_years

assert (
    forecast_results_pdf
    .duplicated(
        ["COUNTRY_ISO3", "YEAR", "INDICATOR"]
    )
    .sum()
    == 0
)

assert forecast_results_pdf["FORECAST_VALUE"].notna().all()

assert set(
    forecast_results_pdf["INDICATOR"].unique()
) == set(FORECAST_TARGETS)

assert forecast_results_pdf["SEGMENT"].notna().all()

assert (
    forecast_results_pdf["FORECAST_VALUE"] >= 0
).all()

print("============================================")
print("FORECAST VALIDATION PASSED")
print("============================================")
print("Rows:", len(forecast_results_pdf))
print("Years:", sorted(forecast_results_pdf["YEAR"].unique()))
print("Indicators:", forecast_results_pdf["INDICATOR"].nunique())
print("Countries:", forecast_results_pdf["COUNTRY_ISO3"].nunique())

display(
    forecast_results_pdf
    .groupby(["INDICATOR", "MODEL"])
    .size()
    .reset_index(name="ROWS")
    .sort_values(["INDICATOR", "MODEL"])
)


# COMMAND ----------

# ============================================================
# 15. Persist final forecasting table
# ============================================================

FINAL_COLUMNS = [
    "COUNTRY_ISO3",
    "COUNTRY_NAME",
    "CLUSTER",
    "SEGMENT",
    "YEAR",
    "INDICATOR",
    "FORECAST_VALUE",
    "MODEL"
]

final_forecast_pdf = forecast_results_pdf[FINAL_COLUMNS].copy()

final_forecast_spark = spark.createDataFrame(
    final_forecast_pdf
)

(
    final_forecast_spark
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.modeled.market_forecasts")
)

print("✓ Forecast table persisted.")
print("Table: workspace.modeled.market_forecasts")
print("Rows:", len(final_forecast_pdf))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Forecasting conclusion
# MAGIC
# MAGIC RetailPulse now has a reproducible forecasting layer based on:
# MAGIC
# MAGIC - 2000–2024 historical data
# MAGIC - chronological 2000–2019 training
# MAGIC - 2020–2024 validation
# MAGIC - validation-driven model selection
# MAGIC - 2025–2029 forecasts
# MAGIC - country-level market segment linkage
# MAGIC
# MAGIC The final forecast table is ready for downstream analytical use and Power BI.
# MAGIC
# MAGIC Forecasts should be interpreted as **decision-support estimates**, not certain
# MAGIC predictions of future economic outcomes.

# COMMAND ----------

