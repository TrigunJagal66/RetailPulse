import requests
import time


# ============================================================
# CONFIGURATION
# ============================================================

START_YEAR = 2000
END_YEAR = 2024

API_BASE = "https://api.worldbank.org/v2"

PER_PAGE = 1000

MAX_RETRIES = 5

RETRY_STATUS_CODES = {
    429,
    500,
    502,
    503,
    504,
}

REQUEST_TIMEOUT = 60

# We intentionally validate 22 indicators.
INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
    "NY.GDP.PCAP.CD": "GDP per capita (current US$)",
    "NY.GDP.MKTP.CD": "GDP (current US$)",
    "FP.CPI.TOTL.ZG": "Inflation, consumer prices (annual %)",
    "SL.UEM.TOTL.ZS": "Unemployment (% of labor force)",
    "NE.CON.PRVT.KD.ZG": "Household consumption growth",
    "NE.EXP.GNFS.ZS": "Exports (% of GDP)",
    "NE.IMP.GNFS.ZS": "Imports (% of GDP)",
    "BX.KLT.DINV.WD.GD.ZS": "FDI net inflows (% of GDP)",
    "SP.POP.TOTL": "Population, total",
    "SP.POP.GROW": "Population growth",
    "SP.URB.TOTL.IN.ZS": "Urban population (% of total)",
    "SP.DYN.LE00.IN": "Life expectancy at birth",
    "IT.NET.USER.ZS": "Individuals using the Internet (%)",
    "EG.ELC.ACCS.ZS": "Access to electricity (%)",
    "NE.GDI.TOTL.ZS": "Gross capital formation (% of GDP)",
    "NV.AGR.TOTL.ZS": "Agriculture value added (% of GDP)",
    "NV.IND.TOTL.ZS": "Industry value added (% of GDP)",
    "NV.SRV.TOTL.ZS": "Services value added (% of GDP)",
    "NE.TRD.GNFS.ZS": "Trade (% of GDP)",
    "SL.TLF.CACT.ZS": "Labor force participation rate (% of population)",
    "EN.GHG.CO2.PC.CE.AR5": "CO2 emissions per capita",
}


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent": "RetailPulse-WorldBank-Validator/2.0"
})


# ============================================================
# REQUEST WITH RETRIES
# ============================================================

def request_with_retry(url, params):
    """
    Make a World Bank API request with retry handling.

    Retries:
    - connection errors
    - timeouts
    - HTTP 429
    - HTTP 500/502/503/504
    """

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = session.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            # Retry temporary server/rate-limit errors.
            if response.status_code in RETRY_STATUS_CODES:

                if attempt == MAX_RETRIES:
                    response.raise_for_status()

                wait_time = 2 ** (attempt - 1)

                print(
                    f"    Temporary HTTP {response.status_code}. "
                    f"Retrying in {wait_time}s..."
                )

                time.sleep(wait_time)

                continue

            response.raise_for_status()

            return response

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ) as exc:

            if attempt == MAX_RETRIES:

                print(
                    f"    Request failed after {MAX_RETRIES} attempts: "
                    f"{exc}"
                )

                return None

            wait_time = 2 ** (attempt - 1)

            print(
                f"    Request error: {type(exc).__name__}. "
                f"Retrying in {wait_time}s..."
            )

            time.sleep(wait_time)

        except requests.RequestException as exc:

            print(
                f"    HTTP request failed: {exc}"
            )

            return None

    return None


# ============================================================
# FETCH ALL PAGES
# ============================================================

def fetch_all_pages(indicator_code):
    """
    Fetch every page returned by the World Bank API.

    Returns:
        list[dict]: all API records
        None: request/API failure
    """

    all_records = []

    page = 1

    total_pages = None
    expected_total_records = None

    url = (
        f"{API_BASE}/country/all/indicator/"
        f"{indicator_code}"
    )

    while True:

        params = {
            "format": "json",
            "source": 2,
            "date": f"{START_YEAR}:{END_YEAR}",
            "per_page": PER_PAGE,
            "page": page,
        }

        response = request_with_retry(
            url,
            params,
        )

        if response is None:
            return None

        try:
            payload = response.json()

        except ValueError as exc:

            print(
                f"    ERROR: invalid JSON response: {exc}"
            )

            return None

        # ----------------------------------------------------
        # Validate API response structure
        # ----------------------------------------------------

        if not isinstance(payload, list) or len(payload) < 2:

            print(
                "    ERROR: unexpected API response structure"
            )

            print(
                f"    Response: {payload}"
            )

            return None

        metadata = payload[0]

        records = payload[1] or []

        # ----------------------------------------------------
        # Pagination metadata
        # ----------------------------------------------------

        try:

            total_pages = int(
                metadata.get("pages", 1)
            )

            expected_total_records = int(
                metadata.get("total", 0)
            )

        except (TypeError, ValueError):

            print(
                "    ERROR: invalid pagination metadata"
            )

            return None

        all_records.extend(records)

        print(
            f"    page {page}/{total_pages} | "
            f"records collected: "
            f"{len(all_records)}/{expected_total_records}"
        )

        if page >= total_pages:
            break

        page += 1

        # Small delay to avoid hammering the API.
        time.sleep(0.10)

    return all_records


# ============================================================
# GET WORLD BANK ECONOMIES
# ============================================================

def fetch_economies():
    """
    Retrieve World Bank economies.

    We use the country endpoint rather than deriving the
    denominator independently for every indicator.

    This gives every indicator the same economy universe.
    """

    url = f"{API_BASE}/country"

    all_countries = []

    page = 1

    while True:

        params = {
            "format": "json",
            "per_page": 1000,
            "page": page,
        }

        response = request_with_retry(
            url,
            params,
        )

        if response is None:
            return None

        try:
            payload = response.json()

        except ValueError:

            print(
                "    ERROR: invalid JSON while fetching countries"
            )

            return None

        if not isinstance(payload, list) or len(payload) < 2:

            print(
                "    ERROR: unexpected country API response"
            )

            return None

        metadata = payload[0]

        records = payload[1] or []

        all_countries.extend(records)

        try:
            total_pages = int(
                metadata.get("pages", 1)
            )

        except (TypeError, ValueError):

            total_pages = 1

        if page >= total_pages:
            break

        page += 1

        time.sleep(0.10)

    # --------------------------------------------------------
    # Keep actual World Bank economies.
    #
    # World Bank country metadata has:
    #
    #     id
    #     iso2Code
    #     name
    #     region
    #     adminregion
    #     incomeLevel
    #     lendingType
    #
    # Regional aggregates have region.id == "NA".
    #
    # Actual economies have a real region ID.
    # --------------------------------------------------------

    economies = {}

    for record in all_countries:

        iso2 = record.get("iso2Code")

        region = record.get("region", {})

        region_id = region.get("id")

        if not iso2:
            continue

        if region_id == "NA":
            continue

        iso3 = record.get("id")

        if not iso3:
            continue

        if len(iso3) != 3:
            continue

        economies[iso3] = {
            "iso3": iso3,
            "name": record.get("name"),
        }

    return economies


# ============================================================
# VALIDATE ONE INDICATOR
# ============================================================

def validate_indicator(
    indicator_code,
    indicator_name,
    economies,
):

    records = fetch_all_pages(
        indicator_code
    )

    if records is None:

        return {
            "code": indicator_code,
            "name": indicator_name,
            "countries": len(economies),
            "country_years": len(economies)
            * (END_YEAR - START_YEAR + 1),
            "non_null": 0,
            "missing": 0,
            "coverage_pct": 0.0,
            "min_year": None,
            "max_year": None,
            "status": "API ERROR",
        }

    # --------------------------------------------------------
    # Create complete expected country/year grid.
    # --------------------------------------------------------

    expected_observations = {}

    for iso3 in economies:

        for year in range(
            START_YEAR,
            END_YEAR + 1,
        ):

            expected_observations[
                (iso3, year)
            ] = None

    # --------------------------------------------------------
    # Insert API observations.
    # --------------------------------------------------------

    for record in records:

        iso3 = record.get(
            "countryiso3code"
        )

        year = record.get("date")

        if not iso3:
            continue

        if iso3 not in economies:
            continue

        try:

            year_int = int(year)

        except (TypeError, ValueError):

            continue

        if not (
            START_YEAR
            <= year_int
            <= END_YEAR
        ):
            continue

        value = record.get("value")

        expected_observations[
            (iso3, year_int)
        ] = value

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    countries = len(economies)

    total_country_years = len(
        expected_observations
    )

    non_null = sum(
        1
        for value in expected_observations.values()
        if value is not None
    )

    missing = (
        total_country_years
        - non_null
    )

    if total_country_years > 0:

        coverage_pct = (
            non_null
            / total_country_years
        ) * 100

    else:

        coverage_pct = 0.0

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if non_null == 0:

        status = "NO VALUES"

    elif coverage_pct >= 80:

        status = "GOOD"

    elif coverage_pct >= 50:

        status = "PARTIAL"

    else:

        status = "LOW"

    return {
        "code": indicator_code,
        "name": indicator_name,
        "countries": countries,
        "country_years": total_country_years,
        "non_null": non_null,
        "missing": missing,
        "coverage_pct": round(
            coverage_pct,
            1,
        ),
        "min_year": START_YEAR,
        "max_year": END_YEAR,
        "status": status,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 120)
    print(
        "RetailPulse - World Bank Indicator Coverage Validation"
    )
    print("=" * 120)

    print()

    print(
        f"Period: {START_YEAR}-{END_YEAR}"
    )

    print(
        f"Indicators: {len(INDICATORS)}"
    )

    print()

    # --------------------------------------------------------
    # Fetch economy universe ONCE.
    # --------------------------------------------------------

    print(
        "Fetching World Bank economy list..."
    )

    economies = fetch_economies()

    if economies is None:

        print(
            "\nERROR: Could not retrieve World Bank economies."
        )

        return

    print(
        f"Economies detected: {len(economies)}"
    )

    expected_country_years = (
        len(economies)
        * (END_YEAR - START_YEAR + 1)
    )

    expected_total_rows = (
        expected_country_years
        * len(INDICATORS)
    )

    print(
        f"Expected country-year observations "
        f"per indicator: {expected_country_years:,}"
    )

    print(
        f"Maximum raw observations across all indicators: "
        f"{expected_total_rows:,}"
    )

    print()

    print(
        "The validator will retrieve ALL API pages."
    )

    print(
        "Temporary API failures will be retried."
    )

    print()

    # --------------------------------------------------------
    # Validate indicators.
    # --------------------------------------------------------

    results = []

    for index, (
        indicator_code,
        indicator_name,
    ) in enumerate(
        INDICATORS.items(),
        start=1,
    ):

        print()

        print(
            f"[{index:02d}/{len(INDICATORS)}] "
            f"{indicator_code}"
        )

        print(
            f"    {indicator_name}"
        )

        result = validate_indicator(
            indicator_code,
            indicator_name,
            economies,
        )

        results.append(result)

        time.sleep(0.25)

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print()
    print()

    print("=" * 145)

    print(
        "FINAL VALIDATION RESULTS"
    )

    print("=" * 145)

    print(
        f"{'CODE':<28}"
        f"{'COUNTRIES':>10}"
        f"{'OBS':>12}"
        f"{'NON_NULL':>12}"
        f"{'MISSING':>10}"
        f"{'COVERAGE':>12}"
        f"{'YEARS':>15}"
        f"{'STATUS':>12}"
    )

    print("-" * 145)

    for result in results:

        if (
            result["min_year"] is not None
            and result["max_year"] is not None
        ):

            year_range = (
                f"{result['min_year']}-"
                f"{result['max_year']}"
            )

        else:

            year_range = "N/A"

        print(
            f"{result['code']:<28}"
            f"{result['countries']:>10}"
            f"{result['country_years']:>12}"
            f"{result['non_null']:>12}"
            f"{result['missing']:>10}"
            f"{result['coverage_pct']:>11.1f}%"
            f"{year_range:>15}"
            f"{result['status']:>12}"
        )

    print("-" * 145)

    # ========================================================
    # SUMMARY
    # ========================================================

    good = sum(
        1
        for result in results
        if result["status"] == "GOOD"
    )

    partial = sum(
        1
        for result in results
        if result["status"] == "PARTIAL"
    )

    low = sum(
        1
        for result in results
        if result["status"] == "LOW"
    )

    no_values = sum(
        1
        for result in results
        if result["status"] == "NO VALUES"
    )

    api_errors = sum(
        1
        for result in results
        if result["status"] == "API ERROR"
    )

    print()

    print("=" * 60)

    print("SUMMARY")

    print("=" * 60)

    print(
        f"Total indicators: {len(results)}"
    )

    print(
        f"GOOD:             {good}"
    )

    print(
        f"PARTIAL:          {partial}"
    )

    print(
        f"LOW:              {low}"
    )

    print(
        f"NO VALUES:        {no_values}"
    )

    print(
        f"API ERRORS:       {api_errors}"
    )

    print("=" * 60)

    print()

    print(
        f"Economies: {len(economies)}"
    )

    print(
        f"Years:     {START_YEAR}-{END_YEAR}"
    )

    print(
        f"Indicators: {len(INDICATORS)}"
    )

    print(
        f"Maximum expected raw observations: "
        f"{expected_total_rows:,}"
    )

    print()

    print(
        "Validation complete."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()