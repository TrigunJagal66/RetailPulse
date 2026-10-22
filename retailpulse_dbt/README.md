# RetailPulse — Global Market Expansion Intelligence Platform

RetailPulse is a data engineering and data science project that transforms historical World Bank economic indicators into actionable insights for international market expansion.

## Project Objective

Analyze economic trends, segment economies with similar characteristics, and generate forecasts to support data-informed market research.

## Technology Stack

- **Data Source:** World Bank API
- **Data Ingestion:** Python
- **Data Warehouse:** Snowflake
- **Data Transformation:** dbt
- **Analytics and Machine Learning:** Databricks, PySpark, scikit-learn
- **Visualization:** Power BI

## Architecture

World Bank API → Python Ingestion → Snowflake RAW → dbt Transformation → Snowflake MODELED → Databricks Analytics and Machine Learning → Power BI

## Key Components

- **Data Engineering:** Ingest and transform economic indicator data.
- **Historical Market Intelligence:** Analyze economic changes over time.
- **Market Segmentation:** Engineer features and apply PCA and K-Means to group similar economies.
- **Forecasting:** Compare baseline forecasting approaches for selected economic indicators.
- **Business Intelligence:** Present findings through interactive Power BI dashboards.

## Historical Coverage

- **Source:** World Bank API
- **Historical period:** 2000–2024
- **Indicators:** 20 selected economic indicators
- **Analytical focus:** Market size, purchasing power, growth momentum, digital readiness, stability, openness, and economic structure.

## Repository Structure

- `ingestion/` — Python ingestion scripts
- `dbt/` — Data transformation models
- `databricks/` — Analytics, segmentation, and forecasting notebooks
- `powerbi/screenshots/` — Dashboard screenshots

## Limitations

Results depend on the availability and quality of the underlying economic data. Country segments and forecasts are analytical estimates, not guarantees of future performance.

## Author

Trigun Jagal