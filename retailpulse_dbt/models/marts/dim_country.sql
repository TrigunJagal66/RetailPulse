with staged as (

    select * from {{ ref('stg_indicator_values') }}

),

distinct_countries as (

    select distinct
        country_iso3,
        country_name

    from staged

)

select * from distinct_countries
