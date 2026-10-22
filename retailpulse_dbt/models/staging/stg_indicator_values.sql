-- This model takes raw World Bank data from Snowflake,
-- cleans and standardizes it, removes invalid records,
-- and produces a consistent staging layer for downstream dbt models.

with source as (

    select *
    from {{ source('raw', 'indicator_values') }}

),

cleaned as (

    select

        -- --------------------------------------------------
        -- Economy
        -- --------------------------------------------------

        trim(country_iso3) as country_iso3,

        trim(country_name) as country_name,


        -- --------------------------------------------------
        -- Indicator
        -- --------------------------------------------------

        trim(indicator_code) as indicator_code,

        trim(indicator_name) as indicator_name,


        -- --------------------------------------------------
        -- Observation
        -- --------------------------------------------------

        cast(year as integer) as year,

        cast(value as float) as value,


        -- --------------------------------------------------
        -- Data lineage / ingestion metadata
        -- --------------------------------------------------

        run_id,

        trim(source) as source,

        extracted_at_utc,

        loaded_at


    from source


    -- ------------------------------------------------------
    -- Keep valid World Bank economy records.
    --
    -- World Bank aggregate regions are excluded during
    -- ingestion. This staging filter ensures that only
    -- records with a valid ISO3 code enter the modeled layer.
    -- ------------------------------------------------------

    where country_iso3 is not null

      and length(trim(country_iso3)) = 3

)

select *

from cleaned