with staged as (

    select *
    from {{ ref('stg_indicator_values') }}

),

deduplicated as (

    select
        country_iso3,
        indicator_code,
        year,
        value,
        run_id,
        source,
        extracted_at_utc,
        loaded_at,

        row_number() over (
            partition by
                country_iso3,
                indicator_code,
                year
            order by
                extracted_at_utc desc,
                loaded_at desc
        ) as row_num

    from staged

)

select
    country_iso3,
    indicator_code,
    year,
    value,
    run_id,
    source,
    extracted_at_utc,
    loaded_at

from deduplicated

where row_num = 1