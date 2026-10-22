with fact as (

    select
        f.country_iso3,
        c.country_name,
        f.year,
        f.indicator_code,
        f.value

    from {{ ref('fact_indicator_value') }} as f

    inner join {{ ref('dim_country') }} as c
        on f.country_iso3 = c.country_iso3

),

profile as (

    select

        country_iso3,
        country_name,
        year,

        -- --------------------------------------------------
        -- Macroeconomic indicators
        -- --------------------------------------------------

        max(case
            when indicator_code = 'NY.GDP.MKTP.KD.ZG'
            then value
        end) as gdp_growth_pct,

        max(case
            when indicator_code = 'NY.GDP.PCAP.CD'
            then value
        end) as gdp_per_capita_usd,

        max(case
            when indicator_code = 'NY.GDP.MKTP.CD'
            then value
        end) as gdp_usd,

        max(case
            when indicator_code = 'FP.CPI.TOTL.ZG'
            then value
        end) as inflation_cpi_pct,

        max(case
            when indicator_code = 'SL.UEM.TOTL.ZS'
            then value
        end) as unemployment_pct,

        -- --------------------------------------------------
        -- Trade and investment
        -- --------------------------------------------------

        max(case
            when indicator_code = 'NE.EXP.GNFS.ZS'
            then value
        end) as exports_pct_gdp,

        max(case
            when indicator_code = 'NE.IMP.GNFS.ZS'
            then value
        end) as imports_pct_gdp,

        max(case
            when indicator_code = 'BX.KLT.DINV.WD.GD.ZS'
            then value
        end) as fdi_inflows_pct_gdp,

        max(case
            when indicator_code = 'NE.TRD.GNFS.ZS'
            then value
        end) as trade_pct_gdp,

        -- --------------------------------------------------
        -- Demographics
        -- --------------------------------------------------

        max(case
            when indicator_code = 'SP.POP.TOTL'
            then value
        end) as population,

        max(case
            when indicator_code = 'SP.POP.GROW'
            then value
        end) as population_growth_pct,

        max(case
            when indicator_code = 'SP.URB.TOTL.IN.ZS'
            then value
        end) as urban_population_pct,

        -- --------------------------------------------------
        -- Social / development indicators
        -- --------------------------------------------------

        max(case
            when indicator_code = 'SP.DYN.LE00.IN'
            then value
        end) as life_expectancy_years,

        max(case
            when indicator_code = 'IT.NET.USER.ZS'
            then value
        end) as internet_users_pct,

        max(case
            when indicator_code = 'EG.ELC.ACCS.ZS'
            then value
        end) as access_to_electricity_pct,

        -- --------------------------------------------------
        -- Economic structure
        -- --------------------------------------------------

        max(case
            when indicator_code = 'NV.AGR.TOTL.ZS'
            then value
        end) as agriculture_value_added_pct_gdp,

        max(case
            when indicator_code = 'NV.IND.TOTL.ZS'
            then value
        end) as industry_value_added_pct_gdp,

        max(case
            when indicator_code = 'NV.SRV.TOTL.ZS'
            then value
        end) as services_value_added_pct_gdp,

        -- --------------------------------------------------
        -- Labour and environment
        -- --------------------------------------------------

        max(case
            when indicator_code = 'SL.TLF.CACT.ZS'
            then value
        end) as labor_force_participation_pct,

        max(case
            when indicator_code = 'EN.GHG.CO2.PC.CE.AR5'
            then value
        end) as co2_per_capita_tonnes

    from fact

    group by
        country_iso3,
        country_name,
        year

)

select
    country_iso3,
    country_name,
    year,

    -- Macroeconomics
    gdp_growth_pct,
    gdp_per_capita_usd,
    gdp_usd,
    inflation_cpi_pct,
    unemployment_pct,

    -- Trade and investment
    exports_pct_gdp,
    imports_pct_gdp,
    fdi_inflows_pct_gdp,
    trade_pct_gdp,

    -- Demographics
    population,
    population_growth_pct,
    urban_population_pct,

    -- Social / development
    life_expectancy_years,
    internet_users_pct,
    access_to_electricity_pct,

    -- Economic structure
    agriculture_value_added_pct_gdp,
    industry_value_added_pct_gdp,
    services_value_added_pct_gdp,

    -- Labour / environment
    labor_force_participation_pct,
    co2_per_capita_tonnes

from profile