with profile as (

    select *
    from {{ ref('country_economic_profile') }}

),

trends as (

    select

        country_iso3,
        country_name,
        year,

        -- --------------------------------------------------
        -- Macroeconomic indicators
        -- --------------------------------------------------

        gdp_growth_pct,
        lag(gdp_growth_pct) over (
            partition by country_iso3
            order by year
        ) as previous_gdp_growth_pct,

        gdp_per_capita_usd,
        lag(gdp_per_capita_usd) over (
            partition by country_iso3
            order by year
        ) as previous_gdp_per_capita_usd,

        gdp_usd,
        lag(gdp_usd) over (
            partition by country_iso3
            order by year
        ) as previous_gdp_usd,

        inflation_cpi_pct,
        lag(inflation_cpi_pct) over (
            partition by country_iso3
            order by year
        ) as previous_inflation_cpi_pct,

        unemployment_pct,
        lag(unemployment_pct) over (
            partition by country_iso3
            order by year
        ) as previous_unemployment_pct,


        -- --------------------------------------------------
        -- Trade and investment
        -- --------------------------------------------------

        exports_pct_gdp,
        lag(exports_pct_gdp) over (
            partition by country_iso3
            order by year
        ) as previous_exports_pct_gdp,

        imports_pct_gdp,
        lag(imports_pct_gdp) over (
            partition by country_iso3
            order by year
        ) as previous_imports_pct_gdp,

        fdi_inflows_pct_gdp,
        lag(fdi_inflows_pct_gdp) over (
            partition by country_iso3
            order by year
        ) as previous_fdi_inflows_pct_gdp,

        trade_pct_gdp,
        lag(trade_pct_gdp) over (
            partition by country_iso3
            order by year
        ) as previous_trade_pct_gdp,


        -- --------------------------------------------------
        -- Demographics
        -- --------------------------------------------------

        population,
        lag(population) over (
            partition by country_iso3
            order by year
        ) as previous_population,

        population_growth_pct,
        lag(population_growth_pct) over (
            partition by country_iso3
            order by year
        ) as previous_population_growth_pct,

        urban_population_pct,
        lag(urban_population_pct) over (
            partition by country_iso3
            order by year
        ) as previous_urban_population_pct,


        -- --------------------------------------------------
        -- Social / development
        -- --------------------------------------------------

        life_expectancy_years,
        lag(life_expectancy_years) over (
            partition by country_iso3
            order by year
        ) as previous_life_expectancy_years,

        internet_users_pct,
        lag(internet_users_pct) over (
            partition by country_iso3
            order by year
        ) as previous_internet_users_pct,

        access_to_electricity_pct,
        lag(access_to_electricity_pct) over (
            partition by country_iso3
            order by year
        ) as previous_access_to_electricity_pct,


        -- --------------------------------------------------
        -- Economic structure
        -- --------------------------------------------------

        agriculture_value_added_pct_gdp,
        lag(agriculture_value_added_pct_gdp) over (
            partition by country_iso3
            order by year
        ) as previous_agriculture_value_added_pct_gdp,

        industry_value_added_pct_gdp,
        lag(industry_value_added_pct_gdp) over (
            partition by country_iso3
            order by year
        ) as previous_industry_value_added_pct_gdp,

        services_value_added_pct_gdp,
        lag(services_value_added_pct_gdp) over (
            partition by country_iso3
            order by year
        ) as previous_services_value_added_pct_gdp,


        -- --------------------------------------------------
        -- Labour / environment
        -- --------------------------------------------------

        labor_force_participation_pct,
        lag(labor_force_participation_pct) over (
            partition by country_iso3
            order by year
        ) as previous_labor_force_participation_pct,

        co2_per_capita_tonnes,
        lag(co2_per_capita_tonnes) over (
            partition by country_iso3
            order by year
        ) as previous_co2_per_capita_tonnes

    from profile

)

select

    country_iso3,
    country_name,
    year,

    -- --------------------------------------------------
    -- Macroeconomic changes
    -- --------------------------------------------------

    gdp_growth_pct,
    previous_gdp_growth_pct,

    gdp_growth_pct - previous_gdp_growth_pct
        as gdp_growth_change_pct_points,

    gdp_per_capita_usd,
    previous_gdp_per_capita_usd,

    gdp_per_capita_usd - previous_gdp_per_capita_usd
        as gdp_per_capita_change_usd,

    gdp_usd,
    previous_gdp_usd,

    gdp_usd - previous_gdp_usd
        as gdp_change_usd,

    inflation_cpi_pct,
    previous_inflation_cpi_pct,

    inflation_cpi_pct - previous_inflation_cpi_pct
        as inflation_change_pct_points,

    unemployment_pct,
    previous_unemployment_pct,

    unemployment_pct - previous_unemployment_pct
        as unemployment_change_pct_points,


    -- --------------------------------------------------
    -- Trade and investment changes
    -- --------------------------------------------------

    exports_pct_gdp,
    previous_exports_pct_gdp,

    exports_pct_gdp - previous_exports_pct_gdp
        as exports_change_pct_points,

    imports_pct_gdp,
    previous_imports_pct_gdp,

    imports_pct_gdp - previous_imports_pct_gdp
        as imports_change_pct_points,

    fdi_inflows_pct_gdp,
    previous_fdi_inflows_pct_gdp,

    fdi_inflows_pct_gdp - previous_fdi_inflows_pct_gdp
        as fdi_inflows_change_pct_points,

    trade_pct_gdp,
    previous_trade_pct_gdp,

    trade_pct_gdp - previous_trade_pct_gdp
        as trade_change_pct_points,


    -- --------------------------------------------------
    -- Demographic changes
    -- --------------------------------------------------

    population,
    previous_population,

    population - previous_population
        as population_change,

    population_growth_pct,
    previous_population_growth_pct,

    population_growth_pct - previous_population_growth_pct
        as population_growth_change_pct_points,

    urban_population_pct,
    previous_urban_population_pct,

    urban_population_pct - previous_urban_population_pct
        as urban_population_change_pct_points,


    -- --------------------------------------------------
    -- Social / development changes
    -- --------------------------------------------------

    life_expectancy_years,
    previous_life_expectancy_years,

    life_expectancy_years - previous_life_expectancy_years
        as life_expectancy_change_years,

    internet_users_pct,
    previous_internet_users_pct,

    internet_users_pct - previous_internet_users_pct
        as internet_users_change_pct_points,

    access_to_electricity_pct,
    previous_access_to_electricity_pct,

    access_to_electricity_pct - previous_access_to_electricity_pct
        as access_to_electricity_change_pct_points,


    -- --------------------------------------------------
    -- Economic structure changes
    -- --------------------------------------------------

    agriculture_value_added_pct_gdp,
    previous_agriculture_value_added_pct_gdp,

    agriculture_value_added_pct_gdp
        - previous_agriculture_value_added_pct_gdp
        as agriculture_change_pct_points,

    industry_value_added_pct_gdp,
    previous_industry_value_added_pct_gdp,

    industry_value_added_pct_gdp
        - previous_industry_value_added_pct_gdp
        as industry_change_pct_points,

    services_value_added_pct_gdp,
    previous_services_value_added_pct_gdp,

    services_value_added_pct_gdp
        - previous_services_value_added_pct_gdp
        as services_change_pct_points,


    -- --------------------------------------------------
    -- Labour / environment changes
    -- --------------------------------------------------

    labor_force_participation_pct,
    previous_labor_force_participation_pct,

    labor_force_participation_pct
        - previous_labor_force_participation_pct
        as labor_force_participation_change_pct_points,

    co2_per_capita_tonnes,
    previous_co2_per_capita_tonnes,

    co2_per_capita_tonnes
        - previous_co2_per_capita_tonnes
        as co2_per_capita_change_tonnes

from trends