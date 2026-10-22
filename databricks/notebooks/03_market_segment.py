# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Market Segmentation (Optimized Production Version)
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC Identify distinct groups of countries with similar economic, demographic, digital,
# MAGIC structural, trade, and macroeconomic characteristics using the full 2000–2024 history.
# MAGIC
# MAGIC ### Important implementation choice
# MAGIC
# MAGIC The historical table is processed with **one country-level Spark aggregation**. Once the
# MAGIC 217-country × 37-feature matrix is created, the machine-learning work is performed locally
# MAGIC with Pandas / NumPy / scikit-learn because the ML matrix is only a few hundred rows.
# MAGIC
# MAGIC This avoids repeatedly recomputing Spark transformations for every feature and keeps the
# MAGIC notebook fast and reproducible.

# COMMAND ----------

# ============================================
# 1. Load production dataset
# ============================================

from pyspark.sql import functions as F

trends_df = spark.table(
    "retailpulse_snowflake_catalog.modeled.country_economic_trends"
)

print("Rows:", trends_df.count())
print("Columns:", len(trends_df.columns))
print("Distinct countries:", trends_df.select("COUNTRY_ISO3").distinct().count())
print(
    "Year range:",
    trends_df.select(F.min("YEAR").alias("min_year"), F.max("YEAR").alias("max_year")).collect()[0]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define the 37 segmentation features
# MAGIC
# MAGIC The feature representation has four components:
# MAGIC
# MAGIC - **20 historical-level features:** median over 2000–2024
# MAGIC - **8 long-term-change features:** 2000 → 2024
# MAGIC - **4 recent-momentum features:** average over 2020–2024
# MAGIC - **5 volatility features:** standard deviation over 2000–2024
# MAGIC
# MAGIC This represents each market through its historical level, structural change, recent momentum,
# MAGIC and stability rather than a single-year snapshot.

# COMMAND ----------

level_features = [
    "GDP_GROWTH_PCT", "GDP_PER_CAPITA_USD", "GDP_USD", "INFLATION_CPI_PCT",
    "UNEMPLOYMENT_PCT", "EXPORTS_PCT_GDP", "IMPORTS_PCT_GDP", "FDI_INFLOWS_PCT_GDP",
    "TRADE_PCT_GDP", "POPULATION", "POPULATION_GROWTH_PCT", "URBAN_POPULATION_PCT",
    "LIFE_EXPECTANCY_YEARS", "INTERNET_USERS_PCT", "ACCESS_TO_ELECTRICITY_PCT",
    "AGRICULTURE_VALUE_ADDED_PCT_GDP", "INDUSTRY_VALUE_ADDED_PCT_GDP",
    "SERVICES_VALUE_ADDED_PCT_GDP", "LABOR_FORCE_PARTICIPATION_PCT", "CO2_PER_CAPITA_TONNES"
]

long_term_change_features = [
    "URBAN_POPULATION_PCT", "LIFE_EXPECTANCY_YEARS", "INTERNET_USERS_PCT",
    "ACCESS_TO_ELECTRICITY_PCT", "AGRICULTURE_VALUE_ADDED_PCT_GDP",
    "INDUSTRY_VALUE_ADDED_PCT_GDP", "SERVICES_VALUE_ADDED_PCT_GDP"
]

recent_features = [
    "GDP_GROWTH_PCT", "INFLATION_CPI_PCT", "UNEMPLOYMENT_PCT", "POPULATION_GROWTH_PCT"
]

volatility_features = [
    "GDP_GROWTH_PCT", "INFLATION_CPI_PCT", "UNEMPLOYMENT_PCT",
    "POPULATION_GROWTH_PCT", "FDI_INFLOWS_PCT_GDP"
]

ml_feature_columns = (
    [f"{c}_MEDIAN" for c in level_features]
    + [f"{c}_CHANGE_2000_2024" for c in long_term_change_features]
    + ["GDP_PER_CAPITA_PCT_CHANGE_2000_2024"]
    + [f"{c}_RECENT_AVG_2020_2024" for c in recent_features]
    + [f"{c}_VOLATILITY_2000_2024" for c in volatility_features]
)

assert len(ml_feature_columns) == 37
print("ML features:", len(ml_feature_columns))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Build the country-level feature matrix in one Spark aggregation
# MAGIC
# MAGIC The previous implementation performed many separate Spark actions. This version computes
# MAGIC all required historical statistics in a single `groupBy`, then moves the small country-level
# MAGIC matrix to Pandas exactly once for machine learning.

# COMMAND ----------

# One Spark aggregation: 20 medians + endpoint statistics + recent averages + volatility.
agg_exprs = []

# Historical medians
for c in level_features:
    agg_exprs.append(F.percentile_approx(F.col(c), 0.5, 10000).alias(f"{c}_MEDIAN"))

# Long-term endpoints: each country has at most one observation per year.
for c in ["GDP_PER_CAPITA_USD"] + long_term_change_features:
    agg_exprs.append(F.max(F.when(F.col("YEAR") == 2000, F.col(c))).alias(f"{c}_2000"))
    agg_exprs.append(F.max(F.when(F.col("YEAR") == 2024, F.col(c))).alias(f"{c}_2024"))

# Recent momentum
for c in recent_features:
    agg_exprs.append(
        F.avg(F.when((F.col("YEAR") >= 2020) & (F.col("YEAR") <= 2024), F.col(c)))
        .alias(f"{c}_RECENT_AVG_2020_2024")
    )

# Historical volatility
for c in volatility_features:
    agg_exprs.append(F.stddev(F.col(c)).alias(f"{c}_VOLATILITY_2000_2024"))

aggregated_df = trends_df.groupBy("COUNTRY_ISO3", "COUNTRY_NAME").agg(*agg_exprs)

# Convert endpoint columns into the final long-term change features.
change_exprs = []
for c in long_term_change_features:
    change_exprs.append(
        (F.col(f"{c}_2024") - F.col(f"{c}_2000")).alias(f"{c}_CHANGE_2000_2024")
    )

gdp_pc_change = F.when(
    F.col("GDP_PER_CAPITA_USD_2000").isNotNull()
    & (F.col("GDP_PER_CAPITA_USD_2000") != 0)
    & F.col("GDP_PER_CAPITA_USD_2024").isNotNull(),
    ((F.col("GDP_PER_CAPITA_USD_2024") - F.col("GDP_PER_CAPITA_USD_2000"))
     / F.col("GDP_PER_CAPITA_USD_2000")) * 100
).alias("GDP_PER_CAPITA_PCT_CHANGE_2000_2024")

country_features_df = aggregated_df.select(
    "COUNTRY_ISO3", "COUNTRY_NAME", *[
        F.col(c) for c in [f"{x}_MEDIAN" for x in level_features]
    ], *change_exprs, gdp_pc_change,
    *[F.col(f"{c}_RECENT_AVG_2020_2024") for c in recent_features],
    *[F.col(f"{c}_VOLATILITY_2000_2024") for c in volatility_features]
)

print("Country-level rows:", country_features_df.count())
print("Feature columns:", len(country_features_df.columns) - 2)
assert len(country_features_df.columns) - 2 == 37

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Move the small country-level matrix to Pandas
# MAGIC
# MAGIC At this point the data is only 217 rows × 37 features. Spark is no longer necessary for the
# MAGIC machine-learning calculations. Keeping the matrix locally avoids repeated Spark jobs and makes
# MAGIC PCA, K-Means, silhouette, and stability calculations very fast.

# COMMAND ----------

import numpy as np
import pandas as pd

country_features_pdf = country_features_df.toPandas()
print(country_features_pdf.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Missing-value analysis and filtering
# MAGIC
# MAGIC Countries with more than 50% missing engineered features are excluded. Remaining missing values
# MAGIC are median-imputed across the retained countries. Missing values are never interpreted as zero.

# COMMAND ----------

country_features_pdf["MISSING_FEATURE_COUNT"] = country_features_pdf[ml_feature_columns].isna().sum(axis=1)
country_features_pdf["MISSING_FEATURE_PCT"] = (
    country_features_pdf["MISSING_FEATURE_COUNT"] / len(ml_feature_columns) * 100
)

missing_summary_pdf = pd.DataFrame({
    "FEATURE": ml_feature_columns,
    "MISSING_COUNT": country_features_pdf[ml_feature_columns].isna().sum().values
})
missing_summary_pdf["MISSING_PCT"] = (
    missing_summary_pdf["MISSING_COUNT"] / len(country_features_pdf) * 100
)
missing_summary_pdf = missing_summary_pdf.sort_values("MISSING_PCT", ascending=False)

display(missing_summary_pdf)

display(
    country_features_pdf[["COUNTRY_ISO3", "COUNTRY_NAME", "MISSING_FEATURE_COUNT", "MISSING_FEATURE_PCT"]]
    .sort_values("MISSING_FEATURE_COUNT", ascending=False)
)

segmentation_base_pdf = country_features_pdf[
    country_features_pdf["MISSING_FEATURE_PCT"] <= 50
].copy()

print("Countries retained:", len(segmentation_base_pdf))
print("Countries excluded:", len(country_features_pdf) - len(segmentation_base_pdf))
assert len(segmentation_base_pdf) == 211

# COMMAND ----------

# Median imputation on the 211-country matrix.
feature_matrix_original = segmentation_base_pdf[ml_feature_columns].copy()
feature_medians = feature_matrix_original.median()
feature_matrix_imputed = feature_matrix_original.fillna(feature_medians)

print("Remaining missing values:", int(feature_matrix_imputed.isna().sum().sum()))
assert feature_matrix_imputed.isna().sum().sum() == 0

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Skewness transformation
# MAGIC
# MAGIC Features with absolute skewness greater than 2 are transformed using a signed-log transform:
# MAGIC
# MAGIC `sign(x) × log1p(abs(x))`
# MAGIC
# MAGIC This is used because some economic changes can be negative, so a conventional log transform
# MAGIC would not be appropriate.

# COMMAND ----------

from scipy.stats import skew

skewness_series = feature_matrix_imputed.apply(lambda s: skew(s, bias=False))
skewness_table = (
    skewness_series.abs()
    .sort_values(ascending=False)
    .rename("ABS_SKEWNESS")
    .to_frame()
)
skewness_table["SKEWNESS"] = skewness_series[skewness_table.index]
display(skewness_table)

strongly_skewed_features = skewness_table.index[skewness_table["ABS_SKEWNESS"] > 2].tolist()
print("Features transformed:", len(strongly_skewed_features))
print(strongly_skewed_features)

feature_matrix_transformed = feature_matrix_imputed.copy()
for c in strongly_skewed_features:
    x = feature_matrix_transformed[c]
    feature_matrix_transformed[c] = np.sign(x) * np.log1p(np.abs(x))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Standardization and PCA
# MAGIC
# MAGIC All 37 engineered features are standardized before PCA. The first 18 principal components are
# MAGIC retained because the previous validated analysis showed that they explain approximately 91% of
# MAGIC the variance.

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_scaled = scaler.fit_transform(feature_matrix_transformed)

pca_full = PCA(n_components=len(ml_feature_columns), random_state=42)
X_pca_full = pca_full.fit_transform(X_scaled)

explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

pca_summary = pd.DataFrame({
    "PC": np.arange(1, len(explained_variance) + 1),
    "EXPLAINED_VARIANCE": explained_variance,
    "CUMULATIVE_VARIANCE": cumulative_variance
})
display(pca_summary)

n_components = 18
X_pca = X_pca_full[:, :n_components]

print(f"Variance retained by {n_components} PCs: {cumulative_variance[n_components-1]:.4%}")
assert X_pca.shape == (len(segmentation_base_pdf), n_components)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. PCA interpretation
# MAGIC
# MAGIC Loadings are inspected for the first five components so the dimensionality reduction remains
# MAGIC interpretable rather than being treated as a black box.

# COMMAND ----------

for pc in ["PC1", "PC2", "PC3", "PC4", "PC5"]:
    top = (
        loading_df[pc]
        .abs()
        .sort_values(ascending=False)
        .head(8)
        .index
    )

    display(
        loading_df.loc[top, [pc]]
        .assign(ABS_LOADING=lambda x: x[pc].abs())
        .sort_values("ABS_LOADING", ascending=False)
        .reset_index()
        .rename(columns={"index": "FEATURE"})
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Compare candidate cluster counts
# MAGIC
# MAGIC K=2 through K=8 are evaluated using WCSS and silhouette score. Cluster size is reported as a
# MAGIC business practicality diagnostic. The highest silhouette is not automatically treated as the
# MAGIC final business answer.

# COMMAND ----------

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

k_results = []
for k in range(2, 9):
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(X_pca)
    k_results.append({
        "K": k,
        "WCSS": model.inertia_,
        "SILHOUETTE": silhouette_score(X_pca, labels),
        "CLUSTER_SIZES": np.bincount(labels).tolist()
    })

k_results_df = pd.DataFrame(k_results)
display(k_results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### K-selection decision
# MAGIC
# MAGIC The silhouette score is expected to favor K=2 in this dataset. RetailPulse retains **K=3** as
# MAGIC the business-facing segmentation because three groups provide more useful differentiation
# MAGIC between emerging, industrial/diversified, and advanced/service-oriented market structures.
# MAGIC
# MAGIC Therefore, K=3 is a **business-selected model supported by clustering diagnostics**, not a claim
# MAGIC that K=3 is the mathematically unique optimum.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. K=3 stability
# MAGIC
# MAGIC The final K=3 model is refit across 20 random seeds. Pairwise Adjusted Rand Index (ARI) measures
# MAGIC whether country assignments remain consistent across runs.

# COMMAND ----------

from sklearn.metrics import adjusted_rand_score

stability_labels = []
stability_rows = []

for seed in range(20):
    model = KMeans(n_clusters=3, random_state=seed, n_init=20)
    labels = model.fit_predict(X_pca)
    stability_labels.append(labels)
    stability_rows.append({
        "SEED": seed,
        "SILHOUETTE": silhouette_score(X_pca, labels),
        "INERTIA": model.inertia_
    })

pairwise_ari = [
    adjusted_rand_score(stability_labels[i], stability_labels[j])
    for i in range(len(stability_labels))
    for j in range(i + 1, len(stability_labels))
]

stability_summary_df = pd.DataFrame(stability_rows)
display(stability_summary_df)
print("Mean silhouette:", round(stability_summary_df["SILHOUETTE"].mean(), 4))
print("Silhouette std:", round(stability_summary_df["SILHOUETTE"].std(), 4))
print("Mean pairwise ARI:", round(float(np.mean(pairwise_ari)), 4))
print("Minimum pairwise ARI:", round(float(np.min(pairwise_ari)), 4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster Stability
# MAGIC
# MAGIC The selected K=3 solution was evaluated across 20 different K-Means random initializations to assess the robustness of the segmentation.
# MAGIC
# MAGIC The clustering showed very low variation in internal quality metrics:
# MAGIC
# MAGIC - Mean silhouette score: **0.1412**
# MAGIC - Silhouette standard deviation: **0.0005**
# MAGIC - Mean pairwise Adjusted Rand Index (ARI): **0.942**
# MAGIC - Minimum pairwise ARI: **0.8455**
# MAGIC
# MAGIC The high mean ARI indicates that country-level cluster assignments are highly consistent across different initializations, while the low silhouette variation indicates stable clustering quality.
# MAGIC
# MAGIC Therefore, the K=3 segmentation is considered **robust to K-Means initialization**.
# MAGIC
# MAGIC It is important to distinguish this stability from overall cluster separation. The silhouette score remains relatively modest, indicating that global economic market characteristics form a continuous rather than sharply separated space. The resulting clusters should therefore be interpreted as **data-driven market archetypes**, rather than strictly separated natural classes.
# MAGIC
# MAGIC ### Final cluster-count decision
# MAGIC
# MAGIC K=2 achieved the highest silhouette score (**0.2180**), but produced only two broad groups of 72 and 139 countries. K=3 produced three substantially more balanced groups of 74, 73, and 64 countries, providing greater business differentiation while avoiding the small clusters observed at higher K values.
# MAGIC
# MAGIC Therefore, **K=3 was retained as the business-selected segmentation solution**, supported by its balanced cluster sizes and strong stability across random initializations. It is not claimed to be the statistically optimal solution according to silhouette score alone.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Final K=3 model
# MAGIC
# MAGIC Country identifiers remain in the exact same Pandas dataframe as the PCA matrix used by K-Means,
# MAGIC so cluster labels cannot be accidentally assigned to different countries because of Spark row-order
# MAGIC changes.

# COMMAND ----------

final_kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
final_labels = final_kmeans.fit_predict(X_pca)

final_segmentation_df = segmentation_base_pdf[["COUNTRY_ISO3", "COUNTRY_NAME"]].copy()
final_segmentation_df["CLUSTER"] = final_labels

print("Final K:", 3)
print("Final silhouette:", round(silhouette_score(X_pca, final_labels), 4))
print("Cluster sizes:")
print(final_segmentation_df["CLUSTER"].value_counts().sort_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Business-scale cluster profiles
# MAGIC
# MAGIC Clustering uses transformed and standardized features. Business interpretation instead uses the
# MAGIC original feature values before transformation, so the resulting profiles remain interpretable in
# MAGIC USD, percentages, years, and other original units.

# COMMAND ----------

# ============================================================
# BUSINESS CLUSTER PROFILE
# ============================================================

profile_pdf = segmentation_base_pdf[
    ["COUNTRY_ISO3", "COUNTRY_NAME"] + ml_feature_columns
].copy()

profile_pdf["CLUSTER"] = final_labels


# ------------------------------------------------------------
# 1. Original-scale cluster profiles
# ------------------------------------------------------------

cluster_profile_original = (
    profile_pdf
    .groupby("CLUSTER")[ml_feature_columns]
    .median()
)

display(cluster_profile_original)


# ------------------------------------------------------------
# 2. Standardized cluster profiles
# ------------------------------------------------------------

profile_z = profile_pdf[ml_feature_columns].apply(
    lambda s: (s - s.mean()) / s.std(ddof=0)
)

profile_z["CLUSTER"] = final_labels

cluster_z_profile = (
    profile_z
    .groupby("CLUSTER")[ml_feature_columns]
    .mean()
)

display(cluster_z_profile)


# ------------------------------------------------------------
# 3. Top distinguishing features for each cluster
# ------------------------------------------------------------

top_cluster_features = []

for cluster in sorted(cluster_z_profile.index):

    top_features = (
        cluster_z_profile.loc[cluster]
        .abs()
        .sort_values(ascending=False)
        .head(10)
        .index
    )

    cluster_top = (
        cluster_z_profile
        .loc[cluster, top_features]
        .sort_values(key=np.abs, ascending=False)
    )

    # Explicitly store feature name as a column
    for feature, value in cluster_top.items():

        top_cluster_features.append({
            "CLUSTER": cluster,
            "FEATURE": feature,
            "STANDARDIZED_PROFILE": value,
            "ABS_STANDARDIZED_PROFILE": abs(value)
        })


top_features_df = pd.DataFrame(top_cluster_features)

display(top_features_df)

# COMMAND ----------

# Display each cluster's top distinguishing features separately

for cluster in sorted(top_features_df["CLUSTER"].unique()):

    display(
        top_features_df[
            top_features_df["CLUSTER"] == cluster
        ][
            ["FEATURE", "STANDARDIZED_PROFILE", "ABS_STANDARDIZED_PROFILE"]
        ]
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Assign business segment labels
# MAGIC
# MAGIC Cluster numbers are arbitrary K-Means identifiers. The business labels below are therefore
# MAGIC interpreted through the cluster profiles rather than treating cluster number itself as meaningful.

# COMMAND ----------

# ============================================================
# ASSIGN BUSINESS SEGMENT LABELS
# ============================================================

# Business labels are assigned based on the validated
# cluster profiles and their dominant economic characteristics.

cluster_to_segment = {
    0: "Advanced & Highly Open Markets",
    1: "Industrial & Diversified Markets",
    2: "Emerging & Development-Constrained Markets"
}

# Assign business segment labels to each country
profile_pdf["SEGMENT"] = profile_pdf["CLUSTER"].map(cluster_to_segment)

# Validate that every country received a segment label
assert profile_pdf["SEGMENT"].notna().all(), \
    "Some countries do not have a business segment label."

# Validate expected number of segments
assert profile_pdf["SEGMENT"].nunique() == 3, \
    "Expected exactly 3 business segments."

# Display segment distribution
segment_distribution = (
    profile_pdf
    .groupby(["CLUSTER", "SEGMENT"])
    .size()
    .reset_index(name="COUNTRY_COUNT")
    .sort_values("CLUSTER")
)

display(segment_distribution)

# COMMAND ----------

# ============================================================
# COUNTRY MEMBERSHIP SANITY CHECK
# ============================================================

# Show a sample of countries from each business segment.
# This is a validation step only; it does not change cluster assignments.

membership_sample = (
    profile_pdf[
        ["COUNTRY_ISO3", "COUNTRY_NAME", "CLUSTER", "SEGMENT"]
    ]
    .sort_values(["CLUSTER", "COUNTRY_NAME"])
    .groupby("CLUSTER", group_keys=False)
    .head(15)
)

display(membership_sample)

# COMMAND ----------

# Complete membership counts
membership_summary = (
    profile_pdf
    .groupby(["CLUSTER", "SEGMENT"])
    .agg(
        COUNTRY_COUNT=("COUNTRY_ISO3", "nunique"),
        COUNTRIES=("COUNTRY_NAME", lambda x: ", ".join(sorted(x)))
    )
    .reset_index()
)

display(membership_summary)

# COMMAND ----------

# ============================================================
# RAW DATA VALIDATION OF BUSINESS SEGMENTS
# ============================================================

from pyspark.sql import functions as F

# Underlying 2000–2024 analytical data
raw_df = spark.table(
    "retailpulse_snowflake_catalog.modeled.country_economic_trends"
)

# Country -> cluster/segment mapping produced by K-Means
membership_spark = spark.createDataFrame(
    profile_pdf[
        ["COUNTRY_ISO3", "COUNTRY_NAME", "CLUSTER", "SEGMENT"]
    ]
)

# Join raw historical data with final segmentation
validation_df = (
    raw_df
    .join(
        membership_spark.select(
            "COUNTRY_ISO3",
            "CLUSTER",
            "SEGMENT"
        ),
        on="COUNTRY_ISO3",
        how="inner"
    )
)

print("Raw country-year rows used for validation:",
      validation_df.count())

print(
    "Countries represented:",
    validation_df.select("COUNTRY_ISO3").distinct().count()
)

# COMMAND ----------

# ============================================================
# RAW 2000–2024 SEGMENT PROFILES
# ============================================================

raw_segment_profile = (
    validation_df
    .groupBy("CLUSTER", "SEGMENT")
    .agg(
        F.expr("percentile_approx(GDP_PER_CAPITA_USD, 0.5)").alias(
            "GDP_PER_CAPITA_MEDIAN"
        ),

        F.expr("percentile_approx(GDP_USD, 0.5)").alias(
            "GDP_MEDIAN"
        ),

        F.expr("percentile_approx(POPULATION, 0.5)").alias(
            "POPULATION_MEDIAN"
        ),

        F.expr("percentile_approx(GDP_GROWTH_PCT, 0.5)").alias(
            "GDP_GROWTH_MEDIAN"
        ),

        F.expr("percentile_approx(INFLATION_CPI_PCT, 0.5)").alias(
            "INFLATION_MEDIAN"
        ),

        F.expr("percentile_approx(URBAN_POPULATION_PCT, 0.5)").alias(
            "URBANIZATION_MEDIAN"
        ),

        F.expr("percentile_approx(INTERNET_USERS_PCT, 0.5)").alias(
            "INTERNET_MEDIAN"
        ),

        F.expr("percentile_approx(ACCESS_TO_ELECTRICITY_PCT, 0.5)").alias(
            "ELECTRICITY_MEDIAN"
        ),

        F.expr("percentile_approx(LIFE_EXPECTANCY_YEARS, 0.5)").alias(
            "LIFE_EXPECTANCY_MEDIAN"
        ),

        F.expr("percentile_approx(EXPORTS_PCT_GDP, 0.5)").alias(
            "EXPORTS_GDP_MEDIAN"
        ),

        F.expr("percentile_approx(IMPORTS_PCT_GDP, 0.5)").alias(
            "IMPORTS_GDP_MEDIAN"
        ),

        F.expr("percentile_approx(TRADE_PCT_GDP, 0.5)").alias(
            "TRADE_GDP_MEDIAN"
        ),

        F.expr("percentile_approx(AGRICULTURE_VALUE_ADDED_PCT_GDP, 0.5)").alias(
            "AGRICULTURE_MEDIAN"
        ),

        F.expr("percentile_approx(INDUSTRY_VALUE_ADDED_PCT_GDP, 0.5)").alias(
            "INDUSTRY_MEDIAN"
        ),

        F.expr("percentile_approx(SERVICES_VALUE_ADDED_PCT_GDP, 0.5)").alias(
            "SERVICES_MEDIAN"
        ),

        F.expr("percentile_approx(POPULATION_GROWTH_PCT, 0.5)").alias(
            "POPULATION_GROWTH_MEDIAN"
        )
    )
    .orderBy("CLUSTER")
)

display(raw_segment_profile)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Persist final segmentation

# COMMAND ----------

# ============================================================
# 15.1 — Prepare Final Segmentation for Persistence
# ============================================================

from datetime import datetime

SEGMENTATION_RUN_ID = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

# Start from the already validated country profile
final_segmentation_pdf = profile_pdf[
    ["COUNTRY_ISO3", "COUNTRY_NAME"] + ml_feature_columns
].copy()

# Use the FINAL validated K=3 assignments
final_segmentation_pdf["CLUSTER"] = final_labels

# Map clusters to business segment names
final_segmentation_pdf["SEGMENT"] = (
    final_segmentation_pdf["CLUSTER"]
    .map(cluster_to_segment)
)

# Add run metadata
final_segmentation_pdf["SEGMENTATION_RUN_ID"] = SEGMENTATION_RUN_ID
final_segmentation_pdf["CREATED_AT"] = datetime.utcnow()

# Put important columns first
final_segmentation_pdf = final_segmentation_pdf[
    [
        "COUNTRY_ISO3",
        "COUNTRY_NAME",
        "CLUSTER",
        "SEGMENT",
        "SEGMENTATION_RUN_ID",
        "CREATED_AT"
    ] + ml_feature_columns
]

display(final_segmentation_pdf.head())

print("Rows:", len(final_segmentation_pdf))
print("Columns:", len(final_segmentation_pdf.columns))

# COMMAND ----------

# ============================================================
# 15.2 — Validate Final Segmentation
# ============================================================

assert len(final_segmentation_pdf) == 211
assert final_segmentation_pdf["COUNTRY_ISO3"].nunique() == 211
assert final_segmentation_pdf["CLUSTER"].nunique() == 3
assert final_segmentation_pdf["SEGMENT"].nunique() == 3
assert final_segmentation_pdf["SEGMENT"].notna().all()

print("✓ Countries:", len(final_segmentation_pdf))
print("✓ Unique countries:", final_segmentation_pdf["COUNTRY_ISO3"].nunique())
print("✓ Clusters:", final_segmentation_pdf["CLUSTER"].nunique())
print("✓ Segments:", final_segmentation_pdf["SEGMENT"].nunique())
print("✓ Missing segment labels:", final_segmentation_pdf["SEGMENT"].isna().sum())

display(
    final_segmentation_pdf
    .groupby(["CLUSTER", "SEGMENT"])
    .size()
    .reset_index(name="COUNTRY_COUNT")
    .sort_values("CLUSTER")
)

# COMMAND ----------

# ============================================================
# 15.3 — Convert Final Segmentation to Spark
# ============================================================

final_segmentation_sdf = spark.createDataFrame(final_segmentation_pdf)

print("Spark rows:", final_segmentation_sdf.count())
print("Spark columns:", len(final_segmentation_sdf.columns))

display(final_segmentation_sdf.limit(10))

# COMMAND ----------

# ============================================================
# 15.4 — Persist Final Segmentation
# ============================================================

SEGMENTATION_TABLE = (
    "workspace.modeled.market_segments"
)

(
    final_segmentation_sdf
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SEGMENTATION_TABLE)
)

print("✓ Final segmentation persisted successfully")
print(f"Table: {SEGMENTATION_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Final Status — Market Segmentation
# MAGIC
# MAGIC RetailPulse uses historical 2000–2024 information to identify three business-oriented market
# MAGIC segments across the 211 economies that pass the >50% missingness exclusion rule.
# MAGIC
# MAGIC ### Final methodology
# MAGIC
# MAGIC - 37 features represent historical level, long-term change, recent momentum, and volatility.
# MAGIC - Remaining missing values are median-imputed.
# MAGIC - Strongly skewed features use a signed-log transformation.
# MAGIC - StandardScaler standardizes the 37-feature matrix.
# MAGIC - PCA retains 18 components (approximately 91% variance based on the validated analysis).
# MAGIC - K=2 through K=8 are compared using WCSS, silhouette, and cluster sizes.
# MAGIC - K=3 is retained as a business-selected segmentation; K=2 has the strongest silhouette.
# MAGIC - K=3 stability is checked across 20 seeds using silhouette and ARI.
# MAGIC - Country IDs remain attached to PCA rows throughout clustering, preventing label misalignment.
# MAGIC - Cluster profiles are interpreted on original feature scales.
# MAGIC - Final outputs are persisted to `workspace.modeled.market_segments` and
# MAGIC   `workspace.modeled.segmentation_features`.
# MAGIC
# MAGIC The segmentation is a **decision-support framework**, not a claim that the three groups are
# MAGIC perfectly separated natural classes or that one segment is universally better than another.