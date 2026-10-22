with staged as (

    select * from {{ ref('stg_indicator_values') }}

),

distinct_indicators as (

    select distinct
        indicator_code,
        indicator_name

    from staged

)

select * from distinct_indicators
