-- RetailPulse — Phase 1: Snowflake setup
-- Run this in a Snowsight worksheet, top to bottom, using the default
-- role/warehouse you get on trial signup (ACCOUNTADMIN / COMPUTE_WH).

-- 1. Warehouse (compute) — X-Small is enough and cheapest for this project.
CREATE WAREHOUSE IF NOT EXISTS retailpulse_wh
  WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60          -- suspends after 60s idle, saves trial credits
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE;

-- 2. Database
CREATE DATABASE IF NOT EXISTS retailpulse;

-- 3. Schemas — RAW holds data as-landed, MODELED holds dbt's cleaned output.
CREATE SCHEMA IF NOT EXISTS retailpulse.raw;
CREATE SCHEMA IF NOT EXISTS retailpulse.modeled;

USE WAREHOUSE retailpulse_wh;
USE DATABASE retailpulse;
USE SCHEMA raw;

-- 4. Raw table — column names/types match extract_world_bank_indicators.py's
-- CSV output exactly (country_name, country_iso3, indicator_code,
-- indicator_name, year, value).
CREATE TABLE IF NOT EXISTS raw.indicator_values (
    country_name     VARCHAR,
    country_iso3     VARCHAR(3),
    indicator_code   VARCHAR,
    indicator_name   VARCHAR,
    year             INTEGER,
    value            FLOAT,
    loaded_at        TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- 5. File format + stage, so we can load the CSV directly from Snowsight's
-- "Load Data" UI, or later via a script (Snowflake Python connector).
CREATE FILE FORMAT IF NOT EXISTS raw.csv_format
    TYPE = 'CSV'
    FIELD_DELIMITER = ','
    SKIP_HEADER = 1
    NULL_IF = ('', 'NULL')
    EMPTY_FIELD_AS_NULL = TRUE;

-- Next step (manual, first time): in Snowsight, go to
-- Data > Databases > RETAILPULSE > RAW > INDICATOR_VALUES > "Load Data",
-- and upload your worldbank_indicators_*.csv from data/raw/.
-- We'll script this load step later once the manual load is confirmed working.

-- Sanity check query to run after loading:
-- SELECT COUNT(*), COUNT(DISTINCT country_name), COUNT(DISTINCT indicator_name)
-- FROM raw.indicator_values;