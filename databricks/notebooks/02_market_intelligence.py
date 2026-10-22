# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

trends_df = spark.table(
    "retailpulse_snowflake_catalog.modeled.country_economic_trends"
)

print("Rows:", trends_df.count())
print("Columns:", len(trends_df.columns))

# COMMAND ----------

display(trends_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Economic Scale & Development
# MAGIC
# MAGIC This section examines the economic size and development characteristics of
# MAGIC countries using GDP, GDP per capita, population, and life expectancy.
# MAGIC
# MAGIC The analysis considers the full 2000–2024 historical period while using
# MAGIC 2024 as the latest available snapshot.
# MAGIC
# MAGIC ### Questions
# MAGIC
# MAGIC - How large are different economies?
# MAGIC - How does economic output per person differ across countries?
# MAGIC - How has GDP and GDP per capita evolved since 2000?
# MAGIC - How have population and life expectancy changed?
# MAGIC - Do large economies necessarily have high GDP per capita?
# MAGIC
# MAGIC ### Indicators
# MAGIC
# MAGIC - GDP
# MAGIC - GDP per capita
# MAGIC - Population
# MAGIC - Life expectancy
# MAGIC
# MAGIC Economic scale, population size, and development are treated as separate
# MAGIC dimensions because a large market is not necessarily a high-income market,
# MAGIC and a high-income market is not necessarily a large market.

# COMMAND ----------

year_summary = (
    trends_df
    .groupBy("year")
    .agg(
        F.countDistinct("country_iso3").alias("countries"),
        F.count("*").alias("country_year_rows")
    )
    .orderBy("year")
)

display(year_summary)

# COMMAND ----------

global_gdp_trend = (
    trends_df
    .groupBy("year")
    .agg(
        F.sum("GDP_USD").alias("total_global_gdp"),
        F.avg("GDP_USD").alias("average_country_gdp")
    )
    .orderBy("year")
)

display(global_gdp_trend)

# COMMAND ----------

global_gdp_pc_trend = (
    trends_df
    .groupBy("year")
    .agg(
        F.avg("GDP_PER_CAPITA_USD").alias("average_gdp_per_capita"),
        F.expr(
            "percentile_approx(GDP_PER_CAPITA_USD, 0.5)"
        ).alias("median_gdp_per_capita")
    )
    .orderBy("year")
)

display(global_gdp_pc_trend)

# COMMAND ----------

india_trend = (
    trends_df
    .filter(F.col("country_name") == "India")
    .select(
        "year",
        "GDP_USD",
        "GDP_PER_CAPITA_USD",
        "POPULATION",
        "LIFE_EXPECTANCY_YEARS"
    )
    .orderBy("year")
)

display(india_trend)

# COMMAND ----------

scale_evolution = (
    trends_df
    .filter(F.col("year").isin(2000, 2024))
    .groupBy("country_iso3", "country_name")
    .agg(
        F.max(
            F.when(F.col("year") == 2000, F.col("GDP_USD"))
        ).alias("GDP_2000"),

        F.max(
            F.when(F.col("year") == 2024, F.col("GDP_USD"))
        ).alias("GDP_2024"),

        F.max(
            F.when(F.col("year") == 2000, F.col("GDP_PER_CAPITA_USD"))
        ).alias("GDP_PC_2000"),

        F.max(
            F.when(F.col("year") == 2024, F.col("GDP_PER_CAPITA_USD"))
        ).alias("GDP_PC_2024"),

        F.max(
            F.when(F.col("year") == 2000, F.col("POPULATION"))
        ).alias("POPULATION_2000"),

        F.max(
            F.when(F.col("year") == 2024, F.col("POPULATION"))
        ).alias("POPULATION_2024"),

        F.max(
            F.when(F.col("year") == 2000, F.col("LIFE_EXPECTANCY_YEARS"))
        ).alias("LIFE_EXPECTANCY_2000"),

        F.max(
            F.when(F.col("year") == 2024, F.col("LIFE_EXPECTANCY_YEARS"))
        ).alias("LIFE_EXPECTANCY_2024")
    )
)

# COMMAND ----------

scale_evolution = (
    scale_evolution
    .withColumn(
        "GDP_CHANGE_PCT",
        ((F.col("GDP_2024") - F.col("GDP_2000"))
        / F.col("GDP_2000")) * 100
    )
    .withColumn(
        "GDP_PC_CHANGE_PCT",
        ((F.col("GDP_PC_2024") - F.col("GDP_PC_2000"))
        / F.col("GDP_PC_2000")) * 100
    )
    .withColumn(
        "POPULATION_CHANGE_PCT",
        ((F.col("POPULATION_2024") - F.col("POPULATION_2000"))
        / F.col("POPULATION_2000")) * 100
    )
    .withColumn(
        "LIFE_EXPECTANCY_CHANGE_YEARS",
        F.col("LIFE_EXPECTANCY_2024")
        - F.col("LIFE_EXPECTANCY_2000")
    )
)

display(scale_evolution)

# COMMAND ----------

display(
    scale_evolution.select(
        "GDP_CHANGE_PCT",
        "GDP_PC_CHANGE_PCT",
        "POPULATION_CHANGE_PCT",
        "LIFE_EXPECTANCY_CHANGE_YEARS"
    ).summary()
)

# COMMAND ----------

display(
    scale_evolution
    .select(
        "country_name",
        "GDP_CHANGE_PCT",
        "GDP_PC_CHANGE_PCT",
        "POPULATION_CHANGE_PCT",
        "LIFE_EXPECTANCY_CHANGE_YEARS"
    )
    .orderBy(F.col("GDP_PC_CHANGE_PCT").desc())
)

# COMMAND ----------

display(
    scale_evolution
    .select(
        "country_name",
        "GDP_2000",
        "GDP_2024",
        "GDP_CHANGE_PCT"
    )
    .orderBy(F.col("GDP_CHANGE_PCT").desc())
    .limit(15)
)

# COMMAND ----------

display(
    scale_evolution
    .select(
        "country_name",
        "GDP_PC_2000",
        "GDP_PC_2024",
        "GDP_PC_CHANGE_PCT"
    )
    .orderBy(F.col("GDP_PC_CHANGE_PCT").desc())
    .limit(15)
)

# COMMAND ----------

display(
    scale_evolution
    .select(
        "country_name",
        "POPULATION_2000",
        "POPULATION_2024",
        "POPULATION_CHANGE_PCT"
    )
    .orderBy(F.col("POPULATION_CHANGE_PCT").desc())
    .limit(5)
)

# COMMAND ----------

display(
    scale_evolution
    .select(
        "country_name",
        "LIFE_EXPECTANCY_2000",
        "LIFE_EXPECTANCY_2024",
        "LIFE_EXPECTANCY_CHANGE_YEARS"
    )
    .orderBy(F.col("LIFE_EXPECTANCY_CHANGE_YEARS").desc())
    .limit(5)
)

# COMMAND ----------

# global gdp growth over time
global_growth = (
    trends_df
    .groupBy("year")
    .agg(
        F.avg("GDP_GROWTH_PCT").alias("avg_gdp_growth"),
        F.expr(
            "percentile_approx(GDP_GROWTH_PCT, 0.5)"
        ).alias("median_gdp_growth")
    )
    .orderBy("year")
)

display(global_growth)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Economic Growth & Momentum
# MAGIC
# MAGIC This section examines how economic growth evolved across countries between 2000 and 2024.
# MAGIC
# MAGIC Unlike the previous endpoint analysis, which compared 2000 with 2024, this section uses annual GDP growth rates to understand the path of economic expansion, contraction, consistency, and recent momentum.
# MAGIC
# MAGIC The analysis focuses on:
# MAGIC - Global growth patterns over time
# MAGIC - Country-level average and median growth
# MAGIC - Growth volatility
# MAGIC - Negative-growth periods
# MAGIC - Recent growth momentum

# COMMAND ----------

# "Which countries experienced consistent, volatile, or frequently negative growth?"
country_growth_profile = (
    trends_df
    .groupBy("country_iso3", "country_name")
    .agg(
        F.avg("GDP_GROWTH_PCT").alias("AVG_GDP_GROWTH"),
        F.expr(
            "percentile_approx(GDP_GROWTH_PCT, 0.5)"
        ).alias("MEDIAN_GDP_GROWTH"),
        F.stddev("GDP_GROWTH_PCT").alias("GDP_GROWTH_VOLATILITY"), # this tells how much country gdp growth tends to fluctuate around its average.
        F.sum(
            F.when(F.col("GDP_GROWTH_PCT") < 0, 1).otherwise(0)
        ).alias("NEGATIVE_GROWTH_YEARS"),
        F.count("GDP_GROWTH_PCT").alias("GROWTH_YEARS")
    )
    .withColumn(
        "NEGATIVE_GROWTH_RATE",
        F.try_divide(
            F.col("NEGATIVE_GROWTH_YEARS"),
            F.col("GROWTH_YEARS")
        )
    )
    .orderBy(F.col("AVG_GDP_GROWTH").desc())
)

display(country_growth_profile)

# COMMAND ----------

recent_growth_profile = (
    trends_df
    .filter(F.col("year").between(2020, 2024))
    .groupBy("country_iso3", "country_name")
    .agg(
        F.avg("GDP_GROWTH_PCT").alias("AVG_GDP_GROWTH_2020_2024"),
        F.expr(
            "percentile_approx(GDP_GROWTH_PCT, 0.5)"
        ).alias("MEDIAN_GDP_GROWTH_2020_2024"),
        F.stddev("GDP_GROWTH_PCT").alias("GDP_GROWTH_VOLATILITY_2020_2024"),
        F.sum(
            F.when(F.col("GDP_GROWTH_PCT") < 0, 1).otherwise(0)
        ).alias("NEGATIVE_GROWTH_YEARS_2020_2024"),
        F.count("GDP_GROWTH_PCT").alias("GROWTH_YEARS_2020_2024"),
        F.max(
            F.when(
                F.col("year") == 2024,
                F.col("GDP_GROWTH_PCT")
            )
        ).alias("GDP_GROWTH_2024")
    )
    .withColumn(
        "NEGATIVE_GROWTH_RATE_2020_2024",
        F.try_divide(
            F.col("NEGATIVE_GROWTH_YEARS_2020_2024"),
            F.col("GROWTH_YEARS_2020_2024")
        )
    )
)

display(recent_growth_profile)

# COMMAND ----------

# MAGIC %md
# MAGIC Compare long term growth with recent momentum

# COMMAND ----------

growth_comparison = (
    country_growth_profile
    .select(
        "country_iso3",
        "country_name",
        "AVG_GDP_GROWTH",
        "GDP_GROWTH_VOLATILITY",
        "NEGATIVE_GROWTH_RATE"
    )
    .join(
        recent_growth_profile.select(
            "country_iso3",
            "AVG_GDP_GROWTH_2020_2024",
            "GDP_GROWTH_VOLATILITY_2020_2024",
            "NEGATIVE_GROWTH_RATE_2020_2024",
            "GDP_GROWTH_2024"
        ),
        on="country_iso3",
        how="left"
    )
    .withColumn(
        "RECENT_VS_LONG_TERM_GROWTH",
        F.col("AVG_GDP_GROWTH_2020_2024")
        - F.col("AVG_GDP_GROWTH")
    )
)

display(growth_comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC Growth Regime Analysis

# COMMAND ----------

growth_regime = (
    growth_comparison
    .withColumn(
        "LONG_TERM_GROWTH_GROUP",
        F.when(
            F.col("AVG_GDP_GROWTH") >=
            F.expr("percentile_approx(AVG_GDP_GROWTH, 0.5) OVER ()"),
            "Higher Long-Term Growth"
        ).otherwise("Lower Long-Term Growth")
    )
    .withColumn(
        "RECENT_MOMENTUM_GROUP",
        F.when(
            F.col("RECENT_VS_LONG_TERM_GROWTH") > 0,
            "Improving"
        ).otherwise("Moderating")
    )
)

display(
    growth_regime.groupBy(
        "LONG_TERM_GROWTH_GROUP",
        "RECENT_MOMENTUM_GROUP"
    )
    .count()
    .orderBy(
        "LONG_TERM_GROWTH_GROUP",
        "RECENT_MOMENTUM_GROUP"
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.5 Key Market Insights
# MAGIC
# MAGIC The economic growth analysis examined annual GDP growth across 217 economies over the 2000–2024 period, with a separate 2020–2024 window used to evaluate recent momentum.
# MAGIC
# MAGIC Several patterns emerged:
# MAGIC
# MAGIC - Global economic growth varied substantially over the 2000–2024 period, with pronounced contractions around major global shocks and subsequent recovery periods.
# MAGIC - Country-level growth differed not only in average magnitude but also in volatility and the frequency of contraction years.
# MAGIC - Recent momentum does not always follow a country's long-term growth pattern. Some countries experienced recent improvement relative to their historical growth, while others showed moderation.
# MAGIC - The long-term versus recent-growth classification produced four descriptive regimes:
# MAGIC   - Higher long-term growth + Improving: 24 countries
# MAGIC   - Higher long-term growth + Moderating: 84 countries
# MAGIC   - Lower long-term growth + Improving: 28 countries
# MAGIC   - Lower long-term growth + Moderating: 81 countries
# MAGIC - The largest group consisted of countries with higher long-term growth but moderating recent momentum, indicating that strong historical growth does not necessarily imply continued acceleration.
# MAGIC - These classifications are intended to support market investigation and comparison rather than establish a universal ranking of countries.
# MAGIC
# MAGIC Overall, the growth analysis shows that market conditions should be evaluated using both long-term economic trajectory and recent momentum rather than relying on a single growth measure.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Macroeconomic Stability
# MAGIC
# MAGIC This section evaluates the stability of macroeconomic conditions across countries using inflation and unemployment indicators.
# MAGIC
# MAGIC The analysis considers the full 2000–2024 historical period while separately examining recent conditions. The objective is to understand inflation persistence, labor-market conditions, and macroeconomic volatility across markets.

# COMMAND ----------

macro_profile = (
    trends_df
    .groupBy("country_iso3", "country_name")
    .agg(
        F.avg("INFLATION_CPI_PCT").alias("AVG_INFLATION"),
        F.expr(
            "percentile_approx(INFLATION_CPI_PCT, 0.5)"
        ).alias("MEDIAN_INFLATION"),
        F.stddev("INFLATION_CPI_PCT").alias("INFLATION_VOLATILITY"),

        F.avg("UNEMPLOYMENT_PCT").alias("AVG_UNEMPLOYMENT"),
        F.expr(
            "percentile_approx(UNEMPLOYMENT_PCT, 0.5)"
        ).alias("MEDIAN_UNEMPLOYMENT"),
        F.stddev("UNEMPLOYMENT_PCT").alias("UNEMPLOYMENT_VOLATILITY")
    )
)

display(macro_profile)

# COMMAND ----------

macro_coverage = (
    trends_df
    .groupBy("country_iso3", "country_name")
    .agg(
        F.count("INFLATION_CPI_PCT").alias("INFLATION_OBS"),
        F.count("UNEMPLOYMENT_PCT").alias("UNEMPLOYMENT_OBS"),
        F.count("*").alias("TOTAL_YEARS")
    )
    .withColumn(
        "INFLATION_COVERAGE_PCT",
        F.col("INFLATION_OBS") / F.col("TOTAL_YEARS") * 100
    )
    .withColumn(
        "UNEMPLOYMENT_COVERAGE_PCT",
        F.col("UNEMPLOYMENT_OBS") / F.col("TOTAL_YEARS") * 100
    )
    .orderBy("country_name")
)

display(macro_coverage)

# COMMAND ----------

inflation_df = (
    trends_df
    .join(
        macro_coverage.select(
            "country_iso3",
            "INFLATION_COVERAGE_PCT"
        ),
        on="country_iso3",
        how="left"
    )
    .filter(F.col("INFLATION_COVERAGE_PCT") >= 80)
)

print("Countries:", inflation_df.select("country_iso3").distinct().count())
print("Rows:", inflation_df.count())

# COMMAND ----------

unemployment_df = (
    trends_df
    .join(
        macro_coverage.select(
            "country_iso3",
            "UNEMPLOYMENT_COVERAGE_PCT"
        ),
        on="country_iso3",
        how="left"
    )
    .filter(F.col("UNEMPLOYMENT_COVERAGE_PCT") >= 80)
)

print("Countries:", unemployment_df.select("country_iso3").distinct().count())
print("Rows:", unemployment_df.count())

# COMMAND ----------

#Inflation Stability
inflation_profile = (
    inflation_df
    .groupBy("country_iso3", "country_name")
    .agg(
        F.avg("INFLATION_CPI_PCT").alias("AVG_INFLATION"),
        F.expr(
            "percentile_approx(INFLATION_CPI_PCT, 0.5)"
        ).alias("MEDIAN_INFLATION"),
        F.stddev("INFLATION_CPI_PCT").alias("INFLATION_VOLATILITY"),

        F.min("INFLATION_CPI_PCT").alias("MIN_INFLATION"),
        F.max("INFLATION_CPI_PCT").alias("MAX_INFLATION"),

        F.sum(
            F.when(
                F.col("INFLATION_CPI_PCT") > 10,
                1
            ).otherwise(0)
        ).alias("HIGH_INFLATION_YEARS"),

        F.count("INFLATION_CPI_PCT").alias("INFLATION_OBS")
    )
    .withColumn(
        "HIGH_INFLATION_RATE",
        F.col("HIGH_INFLATION_YEARS") /
        F.col("INFLATION_OBS")
    )
)

display(inflation_profile)

# COMMAND ----------

# Recent inflation conditions
recent_inflation_profile = (
    inflation_df
    .filter(F.col("year").between(2020, 2024))
    .groupBy("country_iso3", "country_name")
    .agg(
        F.avg("INFLATION_CPI_PCT").alias("AVG_INFLATION_2020_2024"),
        F.expr(
            "percentile_approx(INFLATION_CPI_PCT, 0.5)"
        ).alias("MEDIAN_INFLATION_2020_2024"),
        F.stddev("INFLATION_CPI_PCT").alias(
            "INFLATION_VOLATILITY_2020_2024"
        ),
        F.max(
            F.when(
                F.col("year") == 2024,
                F.col("INFLATION_CPI_PCT")
            )
        ).alias("INFLATION_2024"),
        F.sum(
            F.when(
                F.col("INFLATION_CPI_PCT") > 10,
                1
            ).otherwise(0)
        ).alias("HIGH_INFLATION_YEARS_2020_2024"),
        F.count("INFLATION_CPI_PCT").alias(
            "INFLATION_OBS_2020_2024"
        )
    )
    .withColumn(
        "HIGH_INFLATION_RATE_2020_2024",
        F.try_divide(
            F.col("HIGH_INFLATION_YEARS_2020_2024"),
            F.col("INFLATION_OBS_2020_2024")
        )
    )
)

display(recent_inflation_profile)

# COMMAND ----------

inflation_comparison = (
    inflation_profile
    .select(
        "country_iso3",
        "country_name",
        "AVG_INFLATION",
        "MEDIAN_INFLATION",
        "INFLATION_VOLATILITY"
    )
    .join(
        recent_inflation_profile.select(
            "country_iso3",
            "AVG_INFLATION_2020_2024",
            "INFLATION_VOLATILITY_2020_2024",
            "INFLATION_2024",
            "HIGH_INFLATION_RATE_2020_2024"
        ),
        on="country_iso3",
        how="left"
    )
    .withColumn(
        "RECENT_VS_LONG_TERM_INFLATION",
        F.col("AVG_INFLATION_2020_2024")
        - F.col("AVG_INFLATION")
    )
)

display(inflation_comparison.head(5))

# COMMAND ----------

# Now we will analyze unemployment
unemployment_profile = (
    unemployment_df
    .groupBy("country_iso3", "country_name")
    .agg(
        F.avg("UNEMPLOYMENT_PCT").alias("AVG_UNEMPLOYMENT"),
        F.expr(
            "percentile_approx(UNEMPLOYMENT_PCT, 0.5)"
        ).alias("MEDIAN_UNEMPLOYMENT"),
        F.stddev("UNEMPLOYMENT_PCT").alias(
            "UNEMPLOYMENT_VOLATILITY"
        ),
        F.min("UNEMPLOYMENT_PCT").alias("MIN_UNEMPLOYMENT"),
        F.max("UNEMPLOYMENT_PCT").alias("MAX_UNEMPLOYMENT"),
        F.count("UNEMPLOYMENT_PCT").alias("UNEMPLOYMENT_OBS")
    )
)

display(unemployment_profile)

# COMMAND ----------

recent_unemployment_profile = (
    unemployment_df
    .filter(F.col("year").between(2020, 2024))
    .groupBy("country_iso3", "country_name")
    .agg(
        F.avg("UNEMPLOYMENT_PCT").alias(
            "AVG_UNEMPLOYMENT_2020_2024"
        ),
        F.expr(
            "percentile_approx(UNEMPLOYMENT_PCT, 0.5)"
        ).alias(
            "MEDIAN_UNEMPLOYMENT_2020_2024"
        ),
        F.stddev("UNEMPLOYMENT_PCT").alias(
            "UNEMPLOYMENT_VOLATILITY_2020_2024"
        ),
        F.max(
            F.when(
                F.col("year") == 2024,
                F.col("UNEMPLOYMENT_PCT")
            )
        ).alias(
            "UNEMPLOYMENT_2024"
        ),
        F.count("UNEMPLOYMENT_PCT").alias(
            "UNEMPLOYMENT_OBS_2020_2024"
        )
    )
)

display(recent_unemployment_profile.head(5))

# COMMAND ----------

unemployment_comparison = (
    unemployment_profile
    .select(
        "country_iso3",
        "country_name",
        "AVG_UNEMPLOYMENT",
        "MEDIAN_UNEMPLOYMENT",
        "UNEMPLOYMENT_VOLATILITY"
    )
    .join(
        recent_unemployment_profile.select(
            "country_iso3",
            "AVG_UNEMPLOYMENT_2020_2024",
            "UNEMPLOYMENT_VOLATILITY_2020_2024",
            "UNEMPLOYMENT_2024"
        ),
        on="country_iso3",
        how="left"
    )
    .withColumn(
        "RECENT_VS_LONG_TERM_UNEMPLOYMENT",
        F.col("AVG_UNEMPLOYMENT_2020_2024")
        - F.col("AVG_UNEMPLOYMENT")
    )
)

display(unemployment_comparison)

# COMMAND ----------

# We are done with macroeconomic stability

# COMMAND ----------

# Section -4 (Digital and Infrastructure readiness)
# This will answer "How prepared are different markets in terms of digital adoption and basic infrastructure, and how has that changed over 2000–2024?"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Digital & Infrastructure Readiness
# MAGIC
# MAGIC This section evaluates digital connectivity and foundational infrastructure across countries using Internet usage and access to electricity.
# MAGIC
# MAGIC The analysis considers the full 2000–2024 historical period while using 2020–2024 as a separate recent-period lens.
# MAGIC
# MAGIC The analysis focuses on:
# MAGIC
# MAGIC - Historical Internet adoption
# MAGIC - Recent Internet adoption
# MAGIC - Historical electricity access
# MAGIC - Recent electricity access
# MAGIC - Changes in digital and infrastructure readiness over time
# MAGIC - Differences in development trajectories across markets
# MAGIC
# MAGIC The two indicators are analyzed separately because digital connectivity and electricity access represent different dimensions of market readiness.

# COMMAND ----------

digital_infra_coverage = (
    trends_df
    .groupBy("country_iso3", "country_name")
    .agg(
        F.count("INTERNET_USERS_PCT").alias("INTERNET_OBS"),
        F.count("ACCESS_TO_ELECTRICITY_PCT").alias("ELECTRICITY_OBS"),
        F.count("*").alias("TOTAL_YEARS")
    )
    .withColumn(
        "INTERNET_COVERAGE_PCT",
        F.col("INTERNET_OBS") / F.col("TOTAL_YEARS") * 100
    )
    .withColumn(
        "ELECTRICITY_COVERAGE_PCT",
        F.col("ELECTRICITY_OBS") / F.col("TOTAL_YEARS") * 100
    )
    .orderBy("country_name")
)

display(digital_infra_coverage)

# COMMAND ----------

display(
    digital_infra_coverage.select(
        F.expr(
            "percentile_approx(INTERNET_COVERAGE_PCT, 0.25)"
        ).alias("Internet_Q1"),
        F.expr(
            "percentile_approx(INTERNET_COVERAGE_PCT, 0.50)"
        ).alias("Internet_Median"),
        F.expr(
            "percentile_approx(INTERNET_COVERAGE_PCT, 0.75)"
        ).alias("Internet_Q3"),

        F.expr(
            "percentile_approx(ELECTRICITY_COVERAGE_PCT, 0.25)"
        ).alias("Electricity_Q1"),
        F.expr(
            "percentile_approx(ELECTRICITY_COVERAGE_PCT, 0.50)"
        ).alias("Electricity_Median"),
        F.expr(
            "percentile_approx(ELECTRICITY_COVERAGE_PCT, 0.75)"
        ).alias("Electricity_Q3")
    )
)

# COMMAND ----------

internet_df = (
    trends_df
    .join(
        digital_infra_coverage.select(
            "country_iso3",
            "INTERNET_COVERAGE_PCT"
        ),
        on="country_iso3",
        how="left"
    )
    .filter(F.col("INTERNET_COVERAGE_PCT") >= 80)
)

electricity_df = (
    trends_df
    .join(
        digital_infra_coverage.select(
            "country_iso3",
            "ELECTRICITY_COVERAGE_PCT"
        ),
        on="country_iso3",
        how="left"
    )
    .filter(F.col("ELECTRICITY_COVERAGE_PCT") >= 80)
)

print(
    "Internet countries:",
    internet_df.select("country_iso3").distinct().count()
)
print("Internet rows:", internet_df.count())

print(
    "Electricity countries:",
    electricity_df.select("country_iso3").distinct().count()
)
print("Electricity rows:", electricity_df.count())

# COMMAND ----------

internet_global_trend = (
    internet_df
    .groupBy("year")
    .agg(
        F.avg("INTERNET_USERS_PCT").alias(
            "AVG_INTERNET_USERS_PCT"
        ),
        F.expr(
            "percentile_approx(INTERNET_USERS_PCT, 0.5)"
        ).alias(
            "MEDIAN_INTERNET_USERS_PCT"
        ),
        F.count("INTERNET_USERS_PCT").alias(
            "INTERNET_OBS"
        )
    )
    .orderBy("year")
)

display(internet_global_trend)

# COMMAND ----------

internet_evolution = (
    internet_df
    .filter(F.col("year").isin(2000, 2024))
    .groupBy("country_iso3", "country_name")
    .agg(
        F.max(
            F.when(
                F.col("year") == 2000,
                F.col("INTERNET_USERS_PCT")
            )
        ).alias("INTERNET_2000"),

        F.max(
            F.when(
                F.col("year") == 2024,
                F.col("INTERNET_USERS_PCT")
            )
        ).alias("INTERNET_2024")
    )
    .withColumn(
        "INTERNET_CHANGE_PCT_POINTS",
        F.col("INTERNET_2024")
        - F.col("INTERNET_2000")
    )
)

display(internet_evolution)

# COMMAND ----------

display(
    internet_evolution
    .filter(
        F.col("INTERNET_2000").isNotNull() &
        F.col("INTERNET_2024").isNotNull()
    )
    .orderBy(
        F.col("INTERNET_CHANGE_PCT_POINTS").desc()
    )
    .limit(15)
)

# COMMAND ----------

display(
    internet_evolution
    .filter(
        F.col("INTERNET_2000").isNotNull() &
        F.col("INTERNET_2024").isNotNull()
    )
    .orderBy(
        F.col("INTERNET_CHANGE_PCT_POINTS").asc()
    )
    .limit(15)
)

# COMMAND ----------

internet_2024_distribution = (
    internet_df
    .filter(F.col("year") == 2024)
    .withColumn(
        "INTERNET_ADOPTION_GROUP",
        F.when(F.col("INTERNET_USERS_PCT") >= 80, "High (>=80%)")
        .when(F.col("INTERNET_USERS_PCT") >= 50, "Moderate (50-79%)")
        .otherwise("Low (<50%)")
    )
    .groupBy("INTERNET_ADOPTION_GROUP")
    .agg(
        F.count("country_iso3").alias("COUNTRIES"),
        F.round(
            F.avg("INTERNET_USERS_PCT"), 2
        ).alias("AVG_INTERNET_USERS_PCT")
    )
    .orderBy("INTERNET_ADOPTION_GROUP")
)

display(internet_2024_distribution)

# COMMAND ----------

#Electricity Access
electricity_global_trend = (
    electricity_df
    .groupBy("year")
    .agg(
        F.avg("ACCESS_TO_ELECTRICITY_PCT").alias("AVG_ELECTRICITY_ACCESS_PCT"),
        F.expr(
            "percentile_approx(ACCESS_TO_ELECTRICITY_PCT, 0.5)"
        ).alias("MEDIAN_ELECTRICITY_ACCESS_PCT"),
        F.count("ACCESS_TO_ELECTRICITY_PCT").alias("ELECTRICITY_OBS")
    )
    .orderBy("year")
)

display(electricity_global_trend)

# COMMAND ----------

electricity_evolution = (
    electricity_df
    .filter(F.col("year").isin(2000, 2024))
    .groupBy("country_iso3", "country_name")
    .agg(
        F.max(
            F.when(
                F.col("year") == 2000,
                F.col("ACCESS_TO_ELECTRICITY_PCT")
            )
        ).alias("ELECTRICITY_2000"),

        F.max(
            F.when(
                F.col("year") == 2024,
                F.col("ACCESS_TO_ELECTRICITY_PCT")
            )
        ).alias("ELECTRICITY_2024")
    )
    .withColumn(
        "ELECTRICITY_CHANGE_PCT_POINTS",
        F.col("ELECTRICITY_2024") - F.col("ELECTRICITY_2000")
    )
)

display(
    electricity_evolution
    .filter(
        F.col("ELECTRICITY_2000").isNotNull() &
        F.col("ELECTRICITY_2024").isNotNull()
    )
    .orderBy(
        F.col("ELECTRICITY_CHANGE_PCT_POINTS").desc()
    )
    .limit(15)
)

# COMMAND ----------

display(
    electricity_evolution
    .filter(
        F.col("ELECTRICITY_2000").isNotNull() &
        F.col("ELECTRICITY_2024").isNotNull()
    )
    .orderBy(
        F.col("ELECTRICITY_CHANGE_PCT_POINTS").asc()
    )
    .limit(15)
)

# COMMAND ----------

electricity_2024_distribution = (
    electricity_df
    .filter(F.col("year") == 2024)
    .withColumn(
        "ELECTRICITY_ACCESS_GROUP",
        F.when(
            F.col("ACCESS_TO_ELECTRICITY_PCT") >= 90,
            "High (>=90%)"
        )
        .when(
            F.col("ACCESS_TO_ELECTRICITY_PCT") >= 50,
            "Moderate (50-89%)"
        )
        .otherwise(
            "Low (<50%)"
        )
    )
    .groupBy("ELECTRICITY_ACCESS_GROUP")
    .agg(
        F.count("country_iso3").alias("COUNTRIES"),
        F.round(
            F.avg("ACCESS_TO_ELECTRICITY_PCT"), 2
        ).alias("AVG_ELECTRICITY_ACCESS_PCT")
    )
    .orderBy("ELECTRICITY_ACCESS_GROUP")
)

display(electricity_2024_distribution)

# COMMAND ----------

# Trade and Investments
# How integrated are different markets into international trade and investment, and how has that integration evolved over time?

# COMMAND ----------

trade_global_trend = (
    trends_df
    .groupBy("year")
    .agg(
        F.avg("EXPORTS_PCT_GDP").alias("AVG_EXPORTS_PCT_GDP"),
        F.expr(
            "percentile_approx(EXPORTS_PCT_GDP, 0.5)"
        ).alias("MEDIAN_EXPORTS_PCT_GDP"),
        F.avg("IMPORTS_PCT_GDP").alias("AVG_IMPORTS_PCT_GDP"),
        F.expr(
            "percentile_approx(IMPORTS_PCT_GDP, 0.5)"
        ).alias("MEDIAN_IMPORTS_PCT_GDP"),
        F.count("EXPORTS_PCT_GDP").alias("EXPORTS_OBS"),
        F.count("IMPORTS_PCT_GDP").alias("IMPORTS_OBS")
    )
    .orderBy("year")
)

display(trade_global_trend)

# COMMAND ----------

trade_investment_coverage = (
    trends_df
    .agg(
        F.count("*").alias("TOTAL_ROWS"),

        F.count("EXPORTS_PCT_GDP").alias("EXPORTS_NON_NULL"),
        F.count("IMPORTS_PCT_GDP").alias("IMPORTS_NON_NULL"),
        F.count("TRADE_PCT_GDP").alias("TRADE_NON_NULL"),
        F.count("FDI_INFLOWS_PCT_GDP").alias("FDI_NON_NULL")
    )
)

display(trade_investment_coverage)

# COMMAND ----------

trade_investment_country_coverage = (
    trends_df
    .groupBy("country_iso3", "country_name")
    .agg(
        F.count("EXPORTS_PCT_GDP").alias("EXPORTS_OBS"),
        F.count("IMPORTS_PCT_GDP").alias("IMPORTS_OBS"),
        F.count("TRADE_PCT_GDP").alias("TRADE_OBS"),
        F.count("FDI_INFLOWS_PCT_GDP").alias("FDI_OBS")
    )
    .withColumn(
        "EXPORTS_COVERAGE_PCT",
        F.round(F.col("EXPORTS_OBS") / 25 * 100, 1)
    )
    .withColumn(
        "IMPORTS_COVERAGE_PCT",
        F.round(F.col("IMPORTS_OBS") / 25 * 100, 1)
    )
    .withColumn(
        "TRADE_COVERAGE_PCT",
        F.round(F.col("TRADE_OBS") / 25 * 100, 1)
    )
    .withColumn(
        "FDI_COVERAGE_PCT",
        F.round(F.col("FDI_OBS") / 25 * 100, 1)
    )
)

display(
    trade_investment_country_coverage
    .orderBy("country_name")
)

# COMMAND ----------

# Countries meeting the 80% historical coverage threshold

exports_countries = (
    trade_investment_country_coverage
    .filter(F.col("EXPORTS_OBS") >= 20)
    .select("country_iso3")
)

imports_countries = (
    trade_investment_country_coverage
    .filter(F.col("IMPORTS_OBS") >= 20)
    .select("country_iso3")
)

trade_countries = (
    trade_investment_country_coverage
    .filter(F.col("TRADE_OBS") >= 20)
    .select("country_iso3")
)

fdi_countries = (
    trade_investment_country_coverage
    .filter(F.col("FDI_OBS") >= 20)
    .select("country_iso3")
)

# COMMAND ----------

exports_df = trends_df.join(
    exports_countries,
    on="country_iso3",
    how="inner"
)

imports_df = trends_df.join(
    imports_countries,
    on="country_iso3",
    how="inner"
)

trade_df = trends_df.join(
    trade_countries,
    on="country_iso3",
    how="inner"
)

fdi_df = trends_df.join(
    fdi_countries,
    on="country_iso3",
    how="inner"
)

# COMMAND ----------

print("Exports:", exports_df.select("country_iso3").distinct().count())
print("Imports:", imports_df.select("country_iso3").distinct().count())
print("Trade:", trade_df.select("country_iso3").distinct().count())
print("FDI:", fdi_df.select("country_iso3").distinct().count())

# COMMAND ----------

# Which market have been most economically integrated with international trade over 2020-2024 period.
trade_openness_profile = (
    trends_df
    .join(
        trade_investment_country_coverage
        .filter(F.col("TRADE_OBS") >= 20)
        .select("country_iso3"),
        on="country_iso3",
        how="inner"
    )
    .groupBy("country_iso3", "country_name")
    .agg(
        F.avg("TRADE_PCT_GDP").alias("AVG_TRADE_PCT_GDP"),

        F.expr(
            "percentile_approx(TRADE_PCT_GDP, 0.5)"
        ).alias("MEDIAN_TRADE_PCT_GDP"),

        F.stddev("TRADE_PCT_GDP").alias(
            "TRADE_VOLATILITY"
        ),

        F.min("TRADE_PCT_GDP").alias(
            "MIN_TRADE_PCT_GDP"
        ),

        F.max("TRADE_PCT_GDP").alias(
            "MAX_TRADE_PCT_GDP"
        ),

        F.count("TRADE_PCT_GDP").alias(
            "TRADE_OBS"
        )
    )
    .orderBy(
        F.col("AVG_TRADE_PCT_GDP").desc()
    )
)

display(trade_openness_profile.limit(15))

# COMMAND ----------

display(
    trade_openness_profile
    .orderBy(
        F.col("AVG_TRADE_PCT_GDP").asc()
    )
    .limit(15)
)

# COMMAND ----------

trade_2024 = (
    trends_df
    .filter(
        (F.col("year") == 2024) &
        F.col("TRADE_PCT_GDP").isNotNull()
    )
    .select(
        "country_iso3",
        "country_name",
        "TRADE_PCT_GDP"
    )
    .orderBy(
        F.col("TRADE_PCT_GDP").desc()
    )
)

display(trade_2024.limit(15))

# COMMAND ----------

# Which markets have attracted the most foreign direct investment relative to their economic size, and how stable has that investment been?


# COMMAND ----------

fdi_profile = (
    trends_df
    .join(
        trade_investment_country_coverage
        .filter(F.col("FDI_OBS") >= 20)
        .select("country_iso3"),
        on="country_iso3",
        how="inner"
    )
    .groupBy("country_iso3", "country_name")
    .agg(
        F.avg("FDI_INFLOWS_PCT_GDP").alias(
            "AVG_FDI_INFLOW_PCT_GDP"
        ),

        F.expr(
            "percentile_approx(FDI_INFLOWS_PCT_GDP, 0.5)"
        ).alias(
            "MEDIAN_FDI_INFLOW_PCT_GDP"
        ),

        F.stddev("FDI_INFLOWS_PCT_GDP").alias(
            "FDI_VOLATILITY"
        ),

        F.min("FDI_INFLOWS_PCT_GDP").alias(
            "MIN_FDI_INFLOW_PCT_GDP"
        ),

        F.max("FDI_INFLOWS_PCT_GDP").alias(
            "MAX_FDI_INFLOW_PCT_GDP"
        ),

        F.count("FDI_INFLOWS_PCT_GDP").alias(
            "FDI_OBS"
        )
    )
    .orderBy(
        F.col("AVG_FDI_INFLOW_PCT_GDP").desc()
    )
)

display(fdi_profile.limit(15))

# COMMAND ----------

display(
    fdi_profile
    .orderBy(
        F.col("AVG_FDI_INFLOW_PCT_GDP").asc()
    )
    .limit(15)
)

# COMMAND ----------

fdi_2024 = (
    trends_df
    .filter(
        (F.col("year") == 2024) &
        F.col("FDI_INFLOWS_PCT_GDP").isNotNull()
    )
    .select(
        "country_iso3",
        "country_name",
        "FDI_INFLOWS_PCT_GDP"
    )
    .orderBy(
        F.col("FDI_INFLOWS_PCT_GDP").desc()
    )
)

display(fdi_2024.limit(15))

# COMMAND ----------

# Section 6


# COMMAND ----------

economic_structure_coverage = (
    trends_df
    .groupBy("country_iso3", "country_name")
    .agg(
        F.count("AGRICULTURE_VALUE_ADDED_PCT_GDP").alias("AGRICULTURE_OBS"),
        F.count("INDUSTRY_VALUE_ADDED_PCT_GDP").alias("INDUSTRY_OBS"),
        F.count("SERVICES_VALUE_ADDED_PCT_GDP").alias("SERVICES_OBS")
    )
    .withColumn(
        "AGRICULTURE_COVERAGE_PCT",
        F.round(F.col("AGRICULTURE_OBS") / 25 * 100, 1)
    )
    .withColumn(
        "INDUSTRY_COVERAGE_PCT",
        F.round(F.col("INDUSTRY_OBS") / 25 * 100, 1)
    )
    .withColumn(
        "SERVICES_COVERAGE_PCT",
        F.round(F.col("SERVICES_OBS") / 25 * 100, 1)
    )
)
display(economic_structure_coverage)

# COMMAND ----------

# 6.1.1 — Global Economic Structure Trend (2000–2024)

economic_structure_global_trend = (
    trends_df
    .groupBy("year")
    .agg(
        # Agriculture
        F.avg("AGRICULTURE_VALUE_ADDED_PCT_GDP").alias(
            "AVG_AGRICULTURE_PCT_GDP"
        ),
        F.expr(
            "percentile_approx(AGRICULTURE_VALUE_ADDED_PCT_GDP, 0.5)"
        ).alias(
            "MEDIAN_AGRICULTURE_PCT_GDP"
        ),

        # Industry
        F.avg("INDUSTRY_VALUE_ADDED_PCT_GDP").alias(
            "AVG_INDUSTRY_PCT_GDP"
        ),
        F.expr(
            "percentile_approx(INDUSTRY_VALUE_ADDED_PCT_GDP, 0.5)"
        ).alias(
            "MEDIAN_INDUSTRY_PCT_GDP"
        ),

        # Services
        F.avg("SERVICES_VALUE_ADDED_PCT_GDP").alias(
            "AVG_SERVICES_PCT_GDP"
        ),
        F.expr(
            "percentile_approx(SERVICES_VALUE_ADDED_PCT_GDP, 0.5)"
        ).alias(
            "MEDIAN_SERVICES_PCT_GDP"
        ),

        # Number of countries contributing data
        F.count("AGRICULTURE_VALUE_ADDED_PCT_GDP").alias(
            "AGRICULTURE_OBS"
        ),
        F.count("INDUSTRY_VALUE_ADDED_PCT_GDP").alias(
            "INDUSTRY_OBS"
        ),
        F.count("SERVICES_VALUE_ADDED_PCT_GDP").alias(
            "SERVICES_OBS"
        )
    )
    .orderBy("year")
)

display(economic_structure_global_trend)

# COMMAND ----------

# 6.1.2 — Qualified Country-Level Structural Change
# Criteria:
# 1. At least 20 observations during 2000–2024
# 2. Non-null values in both 2000 and 2024

economic_structure_qualified = (
    trends_df
    .groupBy("country_iso3", "country_name")
    .agg(
        # Coverage
        F.count("AGRICULTURE_VALUE_ADDED_PCT_GDP").alias("AGRICULTURE_OBS"),
        F.count("INDUSTRY_VALUE_ADDED_PCT_GDP").alias("INDUSTRY_OBS"),
        F.count("SERVICES_VALUE_ADDED_PCT_GDP").alias("SERVICES_OBS"),

        # 2000 values
        F.max(
            F.when(
                F.col("year") == 2000,
                F.col("AGRICULTURE_VALUE_ADDED_PCT_GDP")
            )
        ).alias("AGRICULTURE_2000"),

        F.max(
            F.when(
                F.col("year") == 2000,
                F.col("INDUSTRY_VALUE_ADDED_PCT_GDP")
            )
        ).alias("INDUSTRY_2000"),

        F.max(
            F.when(
                F.col("year") == 2000,
                F.col("SERVICES_VALUE_ADDED_PCT_GDP")
            )
        ).alias("SERVICES_2000"),

        # 2024 values
        F.max(
            F.when(
                F.col("year") == 2024,
                F.col("AGRICULTURE_VALUE_ADDED_PCT_GDP")
            )
        ).alias("AGRICULTURE_2024"),

        F.max(
            F.when(
                F.col("year") == 2024,
                F.col("INDUSTRY_VALUE_ADDED_PCT_GDP")
            )
        ).alias("INDUSTRY_2024"),

        F.max(
            F.when(
                F.col("year") == 2024,
                F.col("SERVICES_VALUE_ADDED_PCT_GDP")
            )
        ).alias("SERVICES_2024")
    )
    .filter(
        (F.col("AGRICULTURE_OBS") >= 20) &
        (F.col("INDUSTRY_OBS") >= 20) &
        (F.col("SERVICES_OBS") >= 20) &
        F.col("AGRICULTURE_2000").isNotNull() &
        F.col("AGRICULTURE_2024").isNotNull() &
        F.col("INDUSTRY_2000").isNotNull() &
        F.col("INDUSTRY_2024").isNotNull() &
        F.col("SERVICES_2000").isNotNull() &
        F.col("SERVICES_2024").isNotNull()
    )
    .withColumn(
        "AGRICULTURE_CHANGE_PP",
        F.col("AGRICULTURE_2024") - F.col("AGRICULTURE_2000")
    )
    .withColumn(
        "INDUSTRY_CHANGE_PP",
        F.col("INDUSTRY_2024") - F.col("INDUSTRY_2000")
    )
    .withColumn(
        "SERVICES_CHANGE_PP",
        F.col("SERVICES_2024") - F.col("SERVICES_2000")
    )
)

print(
    "Qualified countries for 2000–2024 structural change:",
    economic_structure_qualified.count()
)

display(economic_structure_qualified)

# COMMAND ----------

# 6.1.3 — Largest Economic Structure Changes

print("Top Agriculture share increases:")
display(
    economic_structure_qualified
    .select(
        "country_iso3",
        "country_name",
        "AGRICULTURE_2000",
        "AGRICULTURE_2024",
        "AGRICULTURE_CHANGE_PP"
    )
    .orderBy(F.col("AGRICULTURE_CHANGE_PP").desc())
    .limit(15)
)

print("Top Agriculture share decreases:")
display(
    economic_structure_qualified
    .select(
        "country_iso3",
        "country_name",
        "AGRICULTURE_2000",
        "AGRICULTURE_2024",
        "AGRICULTURE_CHANGE_PP"
    )
    .orderBy(F.col("AGRICULTURE_CHANGE_PP").asc())
    .limit(15)
)

print("Top Industry share increases:")
display(
    economic_structure_qualified
    .select(
        "country_iso3",
        "country_name",
        "INDUSTRY_2000",
        "INDUSTRY_2024",
        "INDUSTRY_CHANGE_PP"
    )
    .orderBy(F.col("INDUSTRY_CHANGE_PP").desc())
    .limit(15)
)

print("Top Industry share decreases:")
display(
    economic_structure_qualified
    .select(
        "country_iso3",
        "country_name",
        "INDUSTRY_2000",
        "INDUSTRY_2024",
        "INDUSTRY_CHANGE_PP"
    )
    .orderBy(F.col("INDUSTRY_CHANGE_PP").asc())
    .limit(15)
)

print("Top Services share increases:")
display(
    economic_structure_qualified
    .select(
        "country_iso3",
        "country_name",
        "SERVICES_2000",
        "SERVICES_2024",
        "SERVICES_CHANGE_PP"
    )
    .orderBy(F.col("SERVICES_CHANGE_PP").desc())
    .limit(15)
)

print("Top Services share decreases:")
display(
    economic_structure_qualified
    .select(
        "country_iso3",
        "country_name",
        "SERVICES_2000",
        "SERVICES_2024",
        "SERVICES_CHANGE_PP"
    )
    .orderBy(F.col("SERVICES_CHANGE_PP").asc())
    .limit(15)
)

# COMMAND ----------

# 6.1.4 — 2024 Economic Structure Snapshot

economic_structure_2024 = (
    trends_df
    .filter(F.col("year") == 2024)
    .select(
        "country_iso3",
        "country_name",
        "AGRICULTURE_VALUE_ADDED_PCT_GDP",
        "INDUSTRY_VALUE_ADDED_PCT_GDP",
        "SERVICES_VALUE_ADDED_PCT_GDP"
    )
)

print("Top Agriculture Share — 2024")
display(
    economic_structure_2024
    .filter(F.col("AGRICULTURE_VALUE_ADDED_PCT_GDP").isNotNull())
    .orderBy(
        F.col("AGRICULTURE_VALUE_ADDED_PCT_GDP").desc()
    )
    .limit(15)
)

print("Top Industry Share — 2024")
display(
    economic_structure_2024
    .filter(F.col("INDUSTRY_VALUE_ADDED_PCT_GDP").isNotNull())
    .orderBy(
        F.col("INDUSTRY_VALUE_ADDED_PCT_GDP").desc()
    )
    .limit(15)
)

print("Top Services Share — 2024")
display(
    economic_structure_2024
    .filter(F.col("SERVICES_VALUE_ADDED_PCT_GDP").isNotNull())
    .orderBy(
        F.col("SERVICES_VALUE_ADDED_PCT_GDP").desc()
    )
    .limit(15)
)

# COMMAND ----------

# Labor Force Participation
labor_coverage = (
    trends_df
    .groupBy("country_iso3", "country_name")
    .agg(
        F.count("LABOR_FORCE_PARTICIPATION_PCT").alias("LABOR_OBS")
    )
    .withColumn(
        "LABOR_COVERAGE_PCT",
        F.round(F.col("LABOR_OBS") / 25 * 100, 1)
    )
)

display(labor_coverage)

# COMMAND ----------

# 6.2.2 — Historical Labour Force Participation Profile
# 80% coverage = at least 20 observations out of 25 years

labor_profile = (
    trends_df
    .groupBy("country_iso3", "country_name")
    .agg(
        F.avg("LABOR_FORCE_PARTICIPATION_PCT").alias(
            "AVG_LABOR_FORCE_PARTICIPATION"
        ),
        F.expr(
            "percentile_approx(LABOR_FORCE_PARTICIPATION_PCT, 0.5)"
        ).alias(
            "MEDIAN_LABOR_FORCE_PARTICIPATION"
        ),
        F.stddev("LABOR_FORCE_PARTICIPATION_PCT").alias(
            "LABOR_FORCE_VOLATILITY"
        ),
        F.min("LABOR_FORCE_PARTICIPATION_PCT").alias(
            "MIN_LABOR_FORCE_PARTICIPATION"
        ),
        F.max("LABOR_FORCE_PARTICIPATION_PCT").alias(
            "MAX_LABOR_FORCE_PARTICIPATION"
        ),
        F.count("LABOR_FORCE_PARTICIPATION_PCT").alias(
            "LABOR_OBS"
        )
    )
    .filter(F.col("LABOR_OBS") >= 20)
    .orderBy(
        F.col("AVG_LABOR_FORCE_PARTICIPATION").desc()
    )
)

print(
    "Countries meeting 80% historical coverage:",
    labor_profile.count()
)

display(labor_profile)

# COMMAND ----------

# 6.2.3 — 2024 Labour Force Participation Snapshot

labor_2024 = (
    trends_df
    .filter(
        (F.col("year") == 2024) &
        F.col("LABOR_FORCE_PARTICIPATION_PCT").isNotNull()
    )
    .select(
        "country_iso3",
        "country_name",
        "LABOR_FORCE_PARTICIPATION_PCT"
    )
    .orderBy(
        F.col("LABOR_FORCE_PARTICIPATION_PCT").desc()
    )
)

print(
    "Countries with valid 2024 labour-force participation:",
    labor_2024.count()
)

display(labor_2024.limit(15))

# COMMAND ----------

# 6.2.4 — Bottom 15 Labour Force Participation Countries, 2024

print("Bottom 15 Labour Force Participation — 2024")

display(
    labor_2024
    .orderBy(
        F.col("LABOR_FORCE_PARTICIPATION_PCT").asc()
    )
    .limit(15)
)

# COMMAND ----------

# 6.2.2 — Global Labour Force Participation Trend (2000–2024)

labor_global_trend = (
    trends_df
    .groupBy("year")
    .agg(
        F.avg("LABOR_FORCE_PARTICIPATION_PCT").alias(
            "AVG_LABOR_FORCE_PARTICIPATION"
        ),
        F.expr(
            "percentile_approx(LABOR_FORCE_PARTICIPATION_PCT, 0.5)"
        ).alias(
            "MEDIAN_LABOR_FORCE_PARTICIPATION"
        ),
        F.count("LABOR_FORCE_PARTICIPATION_PCT").alias(
            "LABOR_OBS"
        )
    )
    .orderBy("year")
)

display(labor_global_trend)

# COMMAND ----------

#CO2 emmison


# COMMAND ----------

# 6.3.1 — CO₂ per Capita Coverage

co2_coverage = (
    trends_df
    .groupBy("country_iso3", "country_name")
    .agg(
        F.count("CO2_PER_CAPITA_TONNES").alias("CO2_OBS")
    )
    .withColumn(
        "CO2_COVERAGE_PCT",
        F.round(F.col("CO2_OBS") / 25 * 100, 1)
    )
)

display(co2_coverage)

# COMMAND ----------

# 6.3.2 — Historical CO₂ per Capita Profile
# 80% coverage = at least 20 valid observations out of 25 years

co2_profile = (
    trends_df
    .groupBy("country_iso3", "country_name")
    .agg(
        F.avg("CO2_PER_CAPITA_TONNES").alias(
            "AVG_CO2_PER_CAPITA"
        ),
        F.expr(
            "percentile_approx(CO2_PER_CAPITA_TONNES, 0.5)"
        ).alias(
            "MEDIAN_CO2_PER_CAPITA"
        ),
        F.stddev("CO2_PER_CAPITA_TONNES").alias(
            "CO2_VOLATILITY"
        ),
        F.min("CO2_PER_CAPITA_TONNES").alias(
            "MIN_CO2_PER_CAPITA"
        ),
        F.max("CO2_PER_CAPITA_TONNES").alias(
            "MAX_CO2_PER_CAPITA"
        ),
        F.count("CO2_PER_CAPITA_TONNES").alias(
            "CO2_OBS"
        )
    )
    .filter(F.col("CO2_OBS") >= 20)
    .orderBy(
        F.col("AVG_CO2_PER_CAPITA").desc()
    )
)

print(
    "Countries meeting 80% historical coverage:",
    co2_profile.count()
)

display(co2_profile)

# COMMAND ----------

# 6.3.3 — Global CO₂ per Capita Trend (2000–2024)

co2_global_trend = (
    trends_df
    .groupBy("year")
    .agg(
        F.avg("CO2_PER_CAPITA_TONNES").alias(
            "AVG_CO2_PER_CAPITA"
        ),
        F.expr(
            "percentile_approx(CO2_PER_CAPITA_TONNES, 0.5)"
        ).alias(
            "MEDIAN_CO2_PER_CAPITA"
        ),
        F.count("CO2_PER_CAPITA_TONNES").alias(
            "CO2_OBS"
        )
    )
    .orderBy("year")
)

display(co2_global_trend)

# COMMAND ----------

# 6.3.4 — 2024 CO₂ per Capita Snapshot

co2_2024 = (
    trends_df
    .filter(
        (F.col("year") == 2024) &
        F.col("CO2_PER_CAPITA_TONNES").isNotNull()
    )
    .select(
        "country_iso3",
        "country_name",
        "CO2_PER_CAPITA_TONNES"
    )
)

print(
    "Countries with valid 2024 CO₂ data:",
    co2_2024.count()
)

print("Top 15 CO₂ emissions per capita — 2024")

display(
    co2_2024
    .orderBy(
        F.col("CO2_PER_CAPITA_TONNES").desc()
    )
    .limit(15)
)

print("Bottom 15 CO₂ emissions per capita — 2024")

display(
    co2_2024
    .orderBy(
        F.col("CO2_PER_CAPITA_TONNES").asc()
    )
    .limit(15)
)

# COMMAND ----------

#Section -7
#How large are the population bases of the markets in RetailPulse, and how have they changed from 2000–2024?

# COMMAND ----------

# 7.1.1 — Population Coverage

population_coverage = (
    trends_df
    .groupBy("country_iso3", "country_name")
    .agg(
        F.count("POPULATION").alias("POPULATION_OBS")
    )
    .withColumn(
        "POPULATION_COVERAGE_PCT",
        F.round(F.col("POPULATION_OBS") / 25 * 100, 1)
    )
    .orderBy(F.col("POPULATION_COVERAGE_PCT").desc())
)

display(population_coverage)

# COMMAND ----------

# 7.1.1 — Global Population Trend: 2000–2024

population_global_trend = (
    trends_df
    .groupBy("year")
    .agg(
        F.avg("POPULATION").alias("AVG_POPULATION"),
        F.expr("percentile_approx(POPULATION, 0.5)").alias("MEDIAN_POPULATION"),
        F.sum("POPULATION").alias("TOTAL_POPULATION"),
        F.count("POPULATION").alias("POPULATION_OBS")
    )
    .orderBy("year")
)

display(population_global_trend)

# COMMAND ----------

# 7.1.2 — Country-Level Population Change: 2000 → 2024

population_change = (
    trends_df
    .groupBy("country_iso3", "country_name")
    .agg(
        F.count("POPULATION").alias("POPULATION_OBS"),

        F.max(
            F.when(
                F.col("year") == 2000,
                F.col("POPULATION")
            )
        ).alias("POPULATION_2000"),

        F.max(
            F.when(
                F.col("year") == 2024,
                F.col("POPULATION")
            )
        ).alias("POPULATION_2024")
    )
    .filter(
        (F.col("POPULATION_OBS") >= 20) &
        F.col("POPULATION_2000").isNotNull() &
        F.col("POPULATION_2024").isNotNull()
    )
    .withColumn(
        "POPULATION_CHANGE",
        F.col("POPULATION_2024") - F.col("POPULATION_2000")
    )
    .withColumn(
        "POPULATION_CHANGE_PCT",
        F.round(
            (
                (F.col("POPULATION_2024") - F.col("POPULATION_2000"))
                / F.col("POPULATION_2000")
            ) * 100,
            2
        )
    )
)

display(
    population_change
    .orderBy(F.col("POPULATION_CHANGE_PCT").desc())
)

# COMMAND ----------

# 7.1.3 — 2024 Population Snapshot

population_2024 = (
    trends_df
    .filter(
        (F.col("year") == 2024) &
        F.col("POPULATION").isNotNull()
    )
    .select(
        "country_iso3",
        "country_name",
        "POPULATION"
    )
    .orderBy(F.col("POPULATION").desc())
)

display(population_2024)

# COMMAND ----------

# 7.1.3 — 2024 Population Snapshot

population_2024 = (
    trends_df
    .filter(
        (F.col("year") == 2024) &
        F.col("POPULATION").isNotNull()
    )
    .select(
        "country_iso3",
        "country_name",
        "POPULATION"
    )
    .orderBy(F.col("POPULATION").asc())
)

display(population_2024)

# COMMAND ----------

# Population Growth & Demographic Momentum
# 7.2.1 — Global Population Growth Trend: 2000–2024

population_growth_global_trend = (
    trends_df
    .groupBy("year")
    .agg(
        F.avg("POPULATION_GROWTH_PCT").alias("AVG_POPULATION_GROWTH_PCT"),
        F.expr(
            "percentile_approx(POPULATION_GROWTH_PCT, 0.5)"
        ).alias("MEDIAN_POPULATION_GROWTH_PCT"),
        F.count("POPULATION_GROWTH_PCT").alias("POPULATION_GROWTH_OBS")
    )
    .orderBy("year")
)

display(population_growth_global_trend)

# COMMAND ----------

# 7.2.2 — Historical Population Growth Profile

population_growth_profile = (
    trends_df
    .groupBy("country_iso3", "country_name")
    .agg(
        F.avg("POPULATION_GROWTH_PCT").alias("AVG_POPULATION_GROWTH_PCT"),
        F.expr(
            "percentile_approx(POPULATION_GROWTH_PCT, 0.5)"
        ).alias("MEDIAN_POPULATION_GROWTH_PCT"),
        F.stddev("POPULATION_GROWTH_PCT").alias("POPULATION_GROWTH_VOLATILITY"),
        F.min("POPULATION_GROWTH_PCT").alias("MIN_POPULATION_GROWTH_PCT"),
        F.max("POPULATION_GROWTH_PCT").alias("MAX_POPULATION_GROWTH_PCT"),
        F.count("POPULATION_GROWTH_PCT").alias("POPULATION_GROWTH_OBS")
    )
    .filter(F.col("POPULATION_GROWTH_OBS") >= 20)
    .orderBy(F.col("AVG_POPULATION_GROWTH_PCT").desc())
)

display(population_growth_profile)

# COMMAND ----------

# Urbanization
# 7.3.1 — Global Urbanization Trend: 2000–2024

urbanization_global_trend = (
    trends_df
    .groupBy("year")
    .agg(
        F.avg("URBAN_POPULATION_PCT").alias("AVG_URBAN_POPULATION_PCT"),
        F.expr(
            "percentile_approx(URBAN_POPULATION_PCT, 0.5)"
        ).alias("MEDIAN_URBAN_POPULATION_PCT"),
        F.count("URBAN_POPULATION_PCT").alias("URBANIZATION_OBS")
    )
    .orderBy("year")
)

display(urbanization_global_trend)

# COMMAND ----------

# 7.3.2 — Historical Urbanization Profile

urbanization_profile = (
    trends_df
    .groupBy("country_iso3", "country_name")
    .agg(
        F.avg("URBAN_POPULATION_PCT").alias("AVG_URBAN_POPULATION_PCT"),
        F.expr(
            "percentile_approx(URBAN_POPULATION_PCT, 0.5)"
        ).alias("MEDIAN_URBAN_POPULATION_PCT"),
        F.stddev("URBAN_POPULATION_PCT").alias("URBANIZATION_VOLATILITY"),
        F.min("URBAN_POPULATION_PCT").alias("MIN_URBAN_POPULATION_PCT"),
        F.max("URBAN_POPULATION_PCT").alias("MAX_URBAN_POPULATION_PCT"),
        F.count("URBAN_POPULATION_PCT").alias("URBANIZATION_OBS")
    )
    .filter(F.col("URBANIZATION_OBS") >= 20)
    .orderBy(F.col("AVG_URBAN_POPULATION_PCT").desc())
)

display(urbanization_profile)

# COMMAND ----------

# 7.3.3 — Urbanization Change: 2000 → 2024

urbanization_change = (
    trends_df
    .groupBy("country_iso3", "country_name")
    .agg(
        F.count("URBAN_POPULATION_PCT").alias("URBANIZATION_OBS"),

        F.max(
            F.when(
                F.col("year") == 2000,
                F.col("URBAN_POPULATION_PCT")
            )
        ).alias("URBANIZATION_2000"),

        F.max(
            F.when(
                F.col("year") == 2024,
                F.col("URBAN_POPULATION_PCT")
            )
        ).alias("URBANIZATION_2024")
    )
    .filter(
        (F.col("URBANIZATION_OBS") >= 20) &
        F.col("URBANIZATION_2000").isNotNull() &
        F.col("URBANIZATION_2024").isNotNull()
    )
    .withColumn(
        "URBANIZATION_CHANGE_PP",
        F.round(
            F.col("URBANIZATION_2024") -
            F.col("URBANIZATION_2000"),
            2
        )
    )
)

display(
    urbanization_change
    .orderBy(F.col("URBANIZATION_CHANGE_PP").desc())
)

# COMMAND ----------

# 7.3.4 — 2024 Urbanization Snapshot

urbanization_2024 = (
    trends_df
    .filter(
        (F.col("year") == 2024) &
        F.col("URBAN_POPULATION_PCT").isNotNull()
    )
    .select(
        "country_iso3",
        "country_name",
        "URBAN_POPULATION_PCT"
    )
    .orderBy(F.col("URBAN_POPULATION_PCT").desc())
)

display(urbanization_2024)

# COMMAND ----------

# 7.3.4 — 2024 Urbanization Snapshot

urbanization_2024 = (
    trends_df
    .filter(
        (F.col("year") == 2024) &
        F.col("URBAN_POPULATION_PCT").isNotNull()
    )
    .select(
        "country_iso3",
        "country_name",
        "URBAN_POPULATION_PCT"
    )
    .orderBy(F.col("URBAN_POPULATION_PCT").asc())
)

display(urbanization_2024)

# COMMAND ----------

# 7.4.1 — Global Life Expectancy Trend: 2000–2024

life_expectancy_global_trend = (
    trends_df
    .groupBy("year")
    .agg(
        F.avg("LIFE_EXPECTANCY_YEARS").alias("AVG_LIFE_EXPECTANCY_YEARS"),
        F.expr(
            "percentile_approx(LIFE_EXPECTANCY_YEARS, 0.5)"
        ).alias("MEDIAN_LIFE_EXPECTANCY_YEARS"),
        F.count("LIFE_EXPECTANCY_YEARS").alias("LIFE_EXPECTANCY_OBS")
    )
    .orderBy("year")
)

display(life_expectancy_global_trend)

# COMMAND ----------

# 7.4.1a — Life Expectancy Coverage

life_expectancy_coverage = (
    trends_df
    .groupBy("country_iso3", "country_name")
    .agg(
        F.count("LIFE_EXPECTANCY_YEARS").alias("LIFE_EXPECTANCY_OBS")
    )
    .withColumn(
        "LIFE_EXPECTANCY_COVERAGE_PCT",
        F.round(F.col("LIFE_EXPECTANCY_OBS") / 25 * 100, 1)
    )
    .orderBy(F.col("LIFE_EXPECTANCY_COVERAGE_PCT").desc())
)

display(life_expectancy_coverage)

# COMMAND ----------

# 7.4.2 — Historical Life Expectancy Profile

life_expectancy_profile = (
    trends_df
    .groupBy("country_iso3", "country_name")
    .agg(
        F.avg("LIFE_EXPECTANCY_YEARS").alias("AVG_LIFE_EXPECTANCY_YEARS"),
        F.expr(
            "percentile_approx(LIFE_EXPECTANCY_YEARS, 0.5)"
        ).alias("MEDIAN_LIFE_EXPECTANCY_YEARS"),
        F.stddev("LIFE_EXPECTANCY_YEARS").alias("LIFE_EXPECTANCY_VOLATILITY"),
        F.min("LIFE_EXPECTANCY_YEARS").alias("MIN_LIFE_EXPECTANCY_YEARS"),
        F.max("LIFE_EXPECTANCY_YEARS").alias("MAX_LIFE_EXPECTANCY_YEARS"),
        F.count("LIFE_EXPECTANCY_YEARS").alias("LIFE_EXPECTANCY_OBS")
    )
    .filter(F.col("LIFE_EXPECTANCY_OBS") >= 20)
    .orderBy(F.col("AVG_LIFE_EXPECTANCY_YEARS").desc())
)

display(life_expectancy_profile)

# COMMAND ----------

# 7.4.3 — Life Expectancy Change: 2000 → 2024

life_expectancy_change = (
    trends_df
    .groupBy("country_iso3", "country_name")
    .agg(
        F.count("LIFE_EXPECTANCY_YEARS").alias("LIFE_EXPECTANCY_OBS"),

        F.max(
            F.when(
                F.col("year") == 2000,
                F.col("LIFE_EXPECTANCY_YEARS")
            )
        ).alias("LIFE_EXPECTANCY_2000"),

        F.max(
            F.when(
                F.col("year") == 2024,
                F.col("LIFE_EXPECTANCY_YEARS")
            )
        ).alias("LIFE_EXPECTANCY_2024")
    )
    .filter(
        (F.col("LIFE_EXPECTANCY_OBS") >= 20) &
        F.col("LIFE_EXPECTANCY_2000").isNotNull() &
        F.col("LIFE_EXPECTANCY_2024").isNotNull()
    )
    .withColumn(
        "LIFE_EXPECTANCY_CHANGE_YEARS",
        F.round(
            F.col("LIFE_EXPECTANCY_2024") -
            F.col("LIFE_EXPECTANCY_2000"),
            2
        )
    )
)

display(
    life_expectancy_change
    .orderBy(F.col("LIFE_EXPECTANCY_CHANGE_YEARS").desc())
)

# COMMAND ----------

# 7.4.4 — 2024 Life Expectancy Snapshot

life_expectancy_2024 = (
    trends_df
    .filter(
        (F.col("year") == 2024) &
        F.col("LIFE_EXPECTANCY_YEARS").isNotNull()
    )
    .select(
        "country_iso3",
        "country_name",
        "LIFE_EXPECTANCY_YEARS"
    )
    .orderBy(F.col("LIFE_EXPECTANCY_YEARS").desc())
)

display(life_expectancy_2024)

# COMMAND ----------

# 7.4.4 — 2024 Life Expectancy Snapshot

life_expectancy_2024 = (
    trends_df
    .filter(
        (F.col("year") == 2024) &
        F.col("LIFE_EXPECTANCY_YEARS").isNotNull()
    )
    .select(
        "country_iso3",
        "country_name",
        "LIFE_EXPECTANCY_YEARS"
    )
    .orderBy(F.col("LIFE_EXPECTANCY_YEARS").asc())
)

display(life_expectancy_2024)

# COMMAND ----------

