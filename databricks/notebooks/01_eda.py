# Databricks notebook source
# MAGIC %md
# MAGIC # RetailPulse — Exploratory Data Analysis
# MAGIC
# MAGIC ## Market Expansion Intelligence Platform
# MAGIC
# MAGIC ### Objective
# MAGIC
# MAGIC This notebook performs the **Descriptive Analytics** stage of RetailPulse.
# MAGIC
# MAGIC The objective is to understand the structure, quality, distribution, and historical behavior of the economic and development indicators used to characterize global markets.
# MAGIC
# MAGIC The analysis covers **217 economies across 2000–2024** and examines 20 core indicators organized into seven market-expansion pillars.
# MAGIC
# MAGIC ### Data Science Hierarchy
# MAGIC
# MAGIC RetailPulse follows:
# MAGIC
# MAGIC 1. **Descriptive** — What happened?
# MAGIC 2. **Diagnostic** — What is associated with it?
# MAGIC 3. **Predictive** — What might happen?
# MAGIC 4. **Prescriptive** — What can we explore or use for decision support?
# MAGIC
# MAGIC This notebook focuses primarily on **Descriptive Analytics** while establishing the foundation for later diagnostic, predictive, and prescriptive analysis.
# MAGIC
# MAGIC ### Analytical Principles
# MAGIC
# MAGIC - Historical data from 2000–2024 will be used wherever appropriate.
# MAGIC - GDP and population will also be examined on a logarithmic scale because of their highly skewed distributions.
# MAGIC - Both Pearson and Spearman correlations will be considered.
# MAGIC - Relationships identified in this notebook represent **associations, not causal relationships**.
# MAGIC - No investment recommendation or ranking will be derived directly from these descriptive statistics.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# COMMAND ----------

trends_df = spark.table(
    "retailpulse_snowflake_catalog.modeled.country_economic_trends"
)

print("Rows:", trends_df.count())
print("Columns:", len(trends_df.columns))

# COMMAND ----------

display(trends_df.limit(10))

# COMMAND ----------

feature_cols = [
    "GDP_GROWTH_PCT",
    "GDP_PER_CAPITA_USD",
    "GDP_USD",
    "INFLATION_CPI_PCT",
    "UNEMPLOYMENT_PCT",
    "EXPORTS_PCT_GDP",
    "IMPORTS_PCT_GDP",
    "FDI_INFLOWS_PCT_GDP",
    "TRADE_PCT_GDP",
    "POPULATION",
    "POPULATION_GROWTH_PCT",
    "URBAN_POPULATION_PCT",
    "LIFE_EXPECTANCY_YEARS",
    "INTERNET_USERS_PCT",
    "ACCESS_TO_ELECTRICITY_PCT",
    "AGRICULTURE_VALUE_ADDED_PCT_GDP",
    "INDUSTRY_VALUE_ADDED_PCT_GDP",
    "SERVICES_VALUE_ADDED_PCT_GDP",
    "LABOR_FORCE_PARTICIPATION_PCT",
    "CO2_PER_CAPITA_TONNES"
]

print("Number of analytical features:", len(feature_cols))

# COMMAND ----------

eda_df = trends_df.select(
    "COUNTRY_ISO3",
    "COUNTRY_NAME",
    "YEAR",
    *feature_cols
)

print("EDA rows:", eda_df.count())
print("EDA columns:", len(eda_df.columns))

# COMMAND ----------

coverage = eda_df.agg(
    F.count("*").alias("rows"),
    F.countDistinct("COUNTRY_ISO3").alias("countries"),
    F.min("YEAR").alias("start_year"),
    F.max("YEAR").alias("end_year")
)

display(coverage)

# COMMAND ----------

year_coverage = (
    eda_df
    .groupBy("YEAR")
    .agg(
        F.countDistinct("COUNTRY_ISO3").alias("countries"),
        F.count("*").alias("observations")
    )
    .orderBy("YEAR")
)

display(year_coverage)

# COMMAND ----------

country_coverage = (
    eda_df
    .groupBy("COUNTRY_ISO3", "COUNTRY_NAME")
    .agg(
        F.countDistinct("YEAR").alias("years_available"),
        F.min("YEAR").alias("first_year"),
        F.max("YEAR").alias("last_year")
    )
    .orderBy(F.desc("years_available"))
)

display(country_coverage)

# COMMAND ----------

display(
    country_coverage
    .groupBy("years_available")
    .count()
    .orderBy("years_available")
)

# COMMAND ----------

# Now looks for data quality like missing values, outliers, etc
missing_exprs = [
    F.sum(
        F.when(F.col(c).isNull(), 1).otherwise(0)
    ).alias(c)
    for c in feature_cols
]

missing_counts = eda_df.agg(*missing_exprs)
display(missing_counts)

# COMMAND ----------

missing_summary = []

total_rows = eda_df.count()

for c in feature_cols:
    missing = eda_df.filter(F.col(c).isNull()).count()
    
    missing_summary.append({
        "feature": c,
        "missing_count": missing,
        "missing_pct": (missing / total_rows) * 100
    })

missing_pd = pd.DataFrame(missing_summary).sort_values(
    "missing_pct",
    ascending=False
)

display(missing_pd)

# COMMAND ----------

display(
    eda_df.select(*feature_cols).describe()
)

# COMMAND ----------

# find missingness by year
missing_by_year = []

for c in feature_cols:
    result = (
        eda_df
        .groupBy("YEAR")
        .agg(
            F.sum(
                F.when(F.col(c).isNull(), 1).otherwise(0)
            ).alias("missing_count")
        )
        .withColumn("feature", F.lit(c))
    )
    
    missing_by_year.append(result)

missing_year_df = missing_by_year[0]

for df_temp in missing_by_year[1:]:
    missing_year_df = missing_year_df.unionByName(df_temp)

display(
    missing_year_df
    .orderBy("feature", "YEAR")
)

# COMMAND ----------

missing_by_year_pct = (
    missing_year_df
    .join(
        eda_df.groupBy("YEAR").count().withColumnRenamed("count", "total_rows"),
        on="YEAR"
    )
    .withColumn(
        "missing_pct",
        F.round(
            F.col("missing_count") / F.col("total_rows") * 100,
            2
        )
    )
)

display(
    missing_by_year_pct
    .orderBy("feature", "YEAR")
) 

# COMMAND ----------

display(
    missing_by_year_pct
    .groupBy("YEAR")
    .agg(
        F.round(F.avg("missing_pct"), 2).alias("avg_missing_pct"),
        F.round(F.max("missing_pct"), 2).alias("max_missing_pct")
    )
    .orderBy("YEAR")
)

# COMMAND ----------

feature_missing_trend = (
    missing_by_year_pct
    .groupBy("feature")
    .agg(
        F.round(F.avg("missing_pct"), 2).alias("avg_missing_pct"),
        F.round(F.min("missing_pct"), 2).alias("min_missing_pct"),
        F.round(F.max("missing_pct"), 2).alias("max_missing_pct")
    )
    .orderBy(F.desc("avg_missing_pct"))
)

display(feature_missing_trend)

# COMMAND ----------

missing_country = []

for c in feature_cols:
    result = (
        eda_df
        .groupBy("COUNTRY_ISO3", "COUNTRY_NAME")
        .agg(
            F.sum(
                F.when(F.col(c).isNull(), 1).otherwise(0)
            ).alias("missing_count")
        )
        .withColumn("feature", F.lit(c))
    )
    
    missing_country.append(result)

missing_country_df = missing_country[0]

for df_temp in missing_country[1:]:
    missing_country_df = missing_country_df.unionByName(df_temp)

display(
    missing_country_df
    .orderBy(F.desc("missing_count"))
)

# COMMAND ----------

feature_missing_trend = (
    missing_by_year_pct
    .groupBy("feature")
    .agg(
        F.round(F.avg("missing_pct"), 2).alias("avg_missing_pct"),
        F.round(F.min("missing_pct"), 2).alias("min_missing_pct"),
        F.round(F.max("missing_pct"), 2).alias("max_missing_pct")
    )
    .orderBy(F.desc("avg_missing_pct"))
)

display(feature_missing_trend)

# COMMAND ----------

country_missingness = (
    eda_df
    .select(
        "COUNTRY_ISO3",
        "COUNTRY_NAME",
        *[
            F.when(F.col(c).isNull(), 1).otherwise(0).alias(c)
            for c in feature_cols
        ]
    )
    .groupBy("COUNTRY_ISO3", "COUNTRY_NAME")
    .agg(
        *[
            F.round(F.avg(c) * 100, 2).alias(c)
            for c in feature_cols
        ]
    )
)

display(country_missingness)

# COMMAND ----------

from pyspark.sql import functions as F

country_missing_summary = (
    eda_df
    .select(
        "COUNTRY_ISO3",
        "COUNTRY_NAME",
        *[
            F.when(F.col(c).isNull(), 1).otherwise(0).alias(c)
            for c in feature_cols
        ]
    )
    .groupBy("COUNTRY_ISO3", "COUNTRY_NAME")
    .agg(
        *[
            F.sum(c).alias(c)
            for c in feature_cols
        ]
    )
    .withColumn(
        "total_missing",
        sum(F.col(c) for c in feature_cols)
    )
    .withColumn(
        "total_possible",
        F.lit(len(feature_cols) * 25)
    )
    .withColumn(
        "missing_pct",
        F.round(
            F.col("total_missing") / F.col("total_possible") * 100,
            2
        )
    )
    .orderBy(F.desc("missing_pct"))
)

display(country_missing_summary.limit(20))

# COMMAND ----------

country_missing_summary.select(
    "COUNTRY_ISO3",
    "COUNTRY_NAME",
    "missing_pct"
).orderBy(
    F.desc("missing_pct")
).show(30, truncate=False)

# COMMAND ----------

country_missing_summary.select(
    F.count("*").alias("total_countries"),
    F.sum(
        F.when(F.col("missing_pct") == 0, 1).otherwise(0)
    ).alias("countries_with_no_missing"),
    F.sum(
        F.when(F.col("missing_pct") > 10, 1).otherwise(0)
    ).alias("countries_over_10pct_missing"),
    F.sum(
        F.when(F.col("missing_pct") > 25, 1).otherwise(0)
    ).alias("countries_over_25pct_missing"),
    F.sum(
        F.when(F.col("missing_pct") > 50, 1).otherwise(0)
    ).alias("countries_over_50pct_missing")
).show()

# COMMAND ----------

display(
    country_missing_summary
    .select(
        "COUNTRY_ISO3",
        "COUNTRY_NAME",
        "missing_pct"
    )
    .filter(F.col("missing_pct") > 50)
    .orderBy(F.desc("missing_pct"))
)

# COMMAND ----------

summary_stats = eda_df.select(
    *[
        F.round(F.mean(c), 2).alias(f"{c}_mean")
        for c in feature_cols
    ]
)

display(summary_stats)

# COMMAND ----------

display(
    eda_df.select(feature_cols).summary(
        "count",
        "mean",
        "stddev",
        "min",
        "25%",
        "50%",
        "75%",
        "max"
    )
)

# COMMAND ----------

display(
    eda_df
    .filter(F.col("GDP_GROWTH_PCT").isNotNull())
    .orderBy(F.col("GDP_GROWTH_PCT").asc())
    .select(
        "COUNTRY_ISO3",
        "COUNTRY_NAME",
        "YEAR",
        "GDP_GROWTH_PCT"
    )
    .limit(10)
)

# COMMAND ----------

display(
    eda_df
    .filter(F.col("GDP_GROWTH_PCT").isNotNull())
    .orderBy(F.col("GDP_GROWTH_PCT").desc())
    .select(
        "COUNTRY_ISO3",
        "COUNTRY_NAME",
        "YEAR",
        "GDP_GROWTH_PCT"
    )
    .limit(10)
)

# COMMAND ----------

display(
    eda_df
    .filter(F.col("INFLATION_CPI_PCT").isNotNull())
    .orderBy(F.col("INFLATION_CPI_PCT").asc())
    .select(
        "COUNTRY_ISO3",
        "COUNTRY_NAME",
        "YEAR",
        "INFLATION_CPI_PCT"
    )
    .limit(10)
)

# COMMAND ----------

display(
    eda_df
    .filter(F.col("INFLATION_CPI_PCT").isNotNull())
    .orderBy(F.col("INFLATION_CPI_PCT").desc())
    .select(
        "COUNTRY_ISO3",
        "COUNTRY_NAME",
        "YEAR",
        "INFLATION_CPI_PCT"
    )
    .limit(10)
)

# COMMAND ----------

feature_missing_detail = []

for feature in feature_cols:

    total = eda_df.count()

    missing = (
        eda_df
        .filter(F.col(feature).isNull())
        .count()
    )

    available = total - missing

    feature_missing_detail.append({
        "feature": feature,
        "total_rows": total,
        "available": available,
        "missing": missing,
        "missing_pct": round((missing / total) * 100, 2)
    })

feature_missing_detail_df = (
    pd.DataFrame(feature_missing_detail)
    .sort_values("missing_pct", ascending=False)
)

display(feature_missing_detail_df)

# COMMAND ----------

distribution_pd = (
    eda_df
    .select("GDP_USD", "POPULATION")
    .dropna()
    .toPandas()
)

print("Rows available for plotting:", len(distribution_pd))
display(distribution_pd.head())

# COMMAND ----------

distribution_pd["LOG_GDP_USD"] = np.log1p(distribution_pd["GDP_USD"])
distribution_pd["LOG_POPULATION"] = np.log1p(distribution_pd["POPULATION"])

display(distribution_pd.head())

# COMMAND ----------

plt.figure(figsize=(10, 5))

plt.hist(
    distribution_pd["GDP_USD"],
    bins=50
)

plt.title("GDP Distribution — Raw")
plt.xlabel("GDP (USD)")
plt.ylabel("Frequency")

plt.show()

# COMMAND ----------

plt.figure(figsize=(10, 5))

plt.hist(
    distribution_pd["LOG_GDP_USD"],
    bins=50
)

plt.title("GDP Distribution — Log Transformed")
plt.xlabel("log(1 + GDP)")
plt.ylabel("Frequency")

plt.show()

# COMMAND ----------

plt.figure(figsize=(10, 5))

plt.hist(
    distribution_pd["POPULATION"],
    bins=50
)

plt.title("Population Distribution — Raw")
plt.xlabel("Population")
plt.ylabel("Frequency")

plt.show()

# COMMAND ----------

plt.figure(figsize=(10, 5))

plt.hist(
    distribution_pd["LOG_POPULATION"],
    bins=50
)

plt.title("Population Distribution — Log Transformed")
plt.xlabel("log(1 + Population)")
plt.ylabel("Frequency")

plt.show()

# COMMAND ----------

outlier_summary = []

for feature in feature_cols:

    q1, median, q3 = (
        eda_df
        .select(
            F.expr(f"percentile_approx({feature}, 0.25)").alias("q1"),
            F.expr(f"percentile_approx({feature}, 0.50)").alias("median"),
            F.expr(f"percentile_approx({feature}, 0.75)").alias("q3")
        )
        .first()
    )

    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    total_non_null = (
        eda_df
        .filter(F.col(feature).isNotNull())
        .count()
    )

    outlier_count = (
        eda_df
        .filter(
            F.col(feature).isNotNull()
            & (
                (F.col(feature) < lower_bound)
                | (F.col(feature) > upper_bound)
            )
        )
        .count()
    )

    outlier_summary.append({
        "feature": feature,
        "q1": q1,
        "median": median,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": outlier_count,
        "outlier_pct": round(
            (outlier_count / total_non_null) * 100, 2
        )
    })

outlier_summary_df = (
    pd.DataFrame(outlier_summary)
    .sort_values("outlier_pct", ascending=False)
)

display(outlier_summary_df)

# COMMAND ----------

correlation_pd = (
    eda_df
    .select(feature_cols)
    .toPandas()
)

pearson_corr = correlation_pd.corr(
    method="pearson"
)

display(pearson_corr)

# COMMAND ----------

spearman_corr = (
    correlation_pd
    .corr(method="spearman")
)

display(spearman_corr)

# COMMAND ----------

relationships = [
    ("GDP_PER_CAPITA_USD", "INTERNET_USERS_PCT"),
    ("GDP_PER_CAPITA_USD", "ACCESS_TO_ELECTRICITY_PCT"),
    ("URBAN_POPULATION_PCT", "AGRICULTURE_VALUE_ADDED_PCT_GDP"),
    ("GDP_PER_CAPITA_USD", "LIFE_EXPECTANCY_YEARS")
]

for x, y in relationships:
    plot_df = (
        eda_df
        .select(x, y)
        .dropna()
        .toPandas()
    )

    plt.figure(figsize=(7, 5))
    plt.scatter(
        plot_df[x],
        plot_df[y],
        alpha=0.35
    )

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{x} vs {y}")
    plt.grid(alpha=0.2)
    plt.show()

# COMMAND ----------

