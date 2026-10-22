import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests


# ============================================================
# CONFIGURATION
# ============================================================

START_YEAR = 2000
END_YEAR = 2024

API_BASE = "https://api.worldbank.org/v2"

PER_PAGE = 1000

REQUEST_TIMEOUT = 60

MAX_RETRIES = 5

RETRY_STATUS_CODES = {
    429,
    500,
    502,
    503,
    504,
}


# ============================================================
# FINAL VALIDATED INDICATORS
# ============================================================
#
# We removed the two PARTIAL indicators:
#
# NE.CON.PRVT.KD.ZG
# NE.GDI.TOTL.ZS
#
# Final dataset:
#
# 20 indicators
# ×
# 217 economies
# ×
# 25 years
# =
# 108,500 maximum observations
#
# ============================================================

INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
    "NY.GDP.PCAP.CD": "GDP per capita (current US$)",
    "NY.GDP.MKTP.CD": "GDP (current US$)",
    "FP.CPI.TOTL.ZG": "Inflation, consumer prices (annual %)",
    "SL.UEM.TOTL.ZS": "Unemployment (% of labor force)",
    "NE.EXP.GNFS.ZS": "Exports (% of GDP)",
    "NE.IMP.GNFS.ZS": "Imports (% of GDP)",
    "BX.KLT.DINV.WD.GD.ZS": "FDI net inflows (% of GDP)",
    "SP.POP.TOTL": "Population, total",
    "SP.POP.GROW": "Population growth",
    "SP.URB.TOTL.IN.ZS": "Urban population (% of total)",
    "SP.DYN.LE00.IN": "Life expectancy at birth",
    "IT.NET.USER.ZS": "Individuals using the Internet (%)",
    "EG.ELC.ACCS.ZS": "Access to electricity (%)",
    "NV.AGR.TOTL.ZS": "Agriculture value added (% of GDP)",
    "NV.IND.TOTL.ZS": "Industry value added (% of GDP)",
    "NV.SRV.TOTL.ZS": "Services value added (% of GDP)",
    "NE.TRD.GNFS.ZS": "Trade (% of GDP)",
    "SL.TLF.CACT.ZS": "Labor force participation rate (% of population)",
    "EN.GHG.CO2.PC.CE.AR5": "CO2 emissions per capita",
}


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

OUTPUT_FILE = RAW_DIR / "world_bank_indicators.csv"

STATE_FILE = METADATA_DIR / "ingestion_state.json"

LOG_DIR = PROJECT_ROOT / "logs"

LOG_FILE = LOG_DIR / "world_bank_extraction.log"


# ============================================================
# CREATE DIRECTORIES
# ============================================================

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

METADATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            LOG_FILE,
            encoding="utf-8",
        ),
    ],
)

logger = logging.getLogger(__name__)


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent": "RetailPulse-WorldBank-Ingestion/2.0"
})


# ============================================================
# REQUEST WITH RETRIES
# ============================================================

def request_with_retry(url, params):
    """
    Execute a GET request with retry handling.

    Retries temporary failures such as:
    - timeout
    - connection error
    - HTTP 429
    - HTTP 500
    - HTTP 502
    - HTTP 503
    - HTTP 504
    """

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            response = session.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code in RETRY_STATUS_CODES:

                if attempt == MAX_RETRIES:

                    response.raise_for_status()

                wait_time = 2 ** (attempt - 1)

                logger.warning(
                    "HTTP %s received. "
                    "Retrying in %s seconds...",
                    response.status_code,
                    wait_time,
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

                logger.error(
                    "Request failed after %s attempts: %s",
                    MAX_RETRIES,
                    exc,
                )

                return None

            wait_time = 2 ** (attempt - 1)

            logger.warning(
                "Request error (%s). "
                "Retrying in %s seconds...",
                type(exc).__name__,
                wait_time,
            )

            time.sleep(wait_time)

        except requests.RequestException as exc:

            logger.error(
                "HTTP request failed: %s",
                exc,
            )

            return None

    return None


# ============================================================
# FETCH WORLD BANK ECONOMIES
# ============================================================

def fetch_economies():
    """
    Retrieve the World Bank economy list.

    Regional/income-group aggregates are excluded.

    Returns:
        dict:
            {
                "IND": "India",
                "USA": "United States",
                ...
            }

    """

    logger.info(
        "Fetching World Bank economy list..."
    )

    url = f"{API_BASE}/country"

    all_records = []

    page = 1

    while True:

        params = {
            "format": "json",
            "per_page": PER_PAGE,
            "page": page,
        }

        response = request_with_retry(
            url,
            params,
        )

        if response is None:

            raise RuntimeError(
                "Unable to retrieve World Bank economy list."
            )

        try:

            payload = response.json()

        except ValueError as exc:

            raise RuntimeError(
                "World Bank returned invalid JSON "
                "for economy list."
            ) from exc

        if not isinstance(payload, list) or len(payload) < 2:

            raise RuntimeError(
                "Unexpected World Bank economy response."
            )

        metadata = payload[0]

        records = payload[1] or []

        all_records.extend(records)

        total_pages = int(
            metadata.get("pages", 1)
        )

        logger.info(
            "Economy page %s/%s | "
            "records collected: %s",
            page,
            total_pages,
            len(all_records),
        )

        if page >= total_pages:
            break

        page += 1

        time.sleep(0.1)

    economies = {}

    for record in all_records:

        iso3 = record.get("id")

        iso2 = record.get("iso2Code")

        region = record.get(
            "region",
            {},
        )

        region_id = region.get("id")

        # Skip invalid entries.
        if not iso3:
            continue

        if not iso2:
            continue

        if len(iso3) != 3:
            continue

        # "NA" represents aggregates such as
        # regions/income groups.
        if region_id == "NA":
            continue

        economies[iso3] = record.get(
            "name",
            iso3,
        )

    logger.info(
        "World Bank economies detected: %s",
        len(economies),
    )

    return economies


# ============================================================
# FETCH ONE INDICATOR
# ============================================================

def fetch_indicator(
    indicator_code,
    indicator_name,
):
    """
    Fetch all observations for one indicator.

    Returns a list of normalized records.
    """

    logger.info(
        "Fetching indicator: %s | %s",
        indicator_code,
        indicator_name,
    )

    url = (
        f"{API_BASE}/country/all/indicator/"
        f"{indicator_code}"
    )

    records = []

    page = 1

    total_pages = None

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

            raise RuntimeError(
                f"Failed to fetch indicator "
                f"{indicator_code}"
            )

        try:

            payload = response.json()

        except ValueError as exc:

            raise RuntimeError(
                f"Invalid JSON returned for "
                f"{indicator_code}"
            ) from exc

        if not isinstance(payload, list) or len(payload) < 2:

            raise RuntimeError(
                f"Unexpected API response for "
                f"{indicator_code}"
            )

        metadata = payload[0]

        page_records = payload[1] or []

        records.extend(page_records)

        total_pages = int(
            metadata.get("pages", 1)
        )

        total_records = int(
            metadata.get("total", 0)
        )

        logger.info(
            "%s | page %s/%s | "
            "records %s/%s",
            indicator_code,
            page,
            total_pages,
            len(records),
            total_records,
        )

        if page >= total_pages:
            break

        page += 1

        time.sleep(0.1)

    return records


# ============================================================
# NORMALIZE INDICATOR RECORDS
# ============================================================

def normalize_records(
    raw_records,
    economies,
    run_id,
    extracted_at,
):
    """
    Convert World Bank API records into the RetailPulse
    raw-table structure.

    Only validated economies are retained.
    """

    normalized = []

    for record in raw_records:

        iso3 = record.get(
            "countryiso3code"
        )

        year = record.get("date")

        value = record.get("value")

        indicator = record.get(
            "indicator",
            {},
        )

        indicator_code = indicator.get(
            "id"
        )

        indicator_name = indicator.get(
            "value"
        )

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

        normalized.append({
            "country_name": economies[iso3],
            "country_iso3": iso3,
            "indicator_code": indicator_code,
            "indicator_name": indicator_name,
            "year": year_int,
            "value": value,
            "run_id": run_id,
            "source": "World Bank API",
            "extracted_at_utc": extracted_at,
        })

    return normalized


# ============================================================
# DEDUPLICATE
# ============================================================

def deduplicate_records(records):
    """
    Ensure one row exists for:

        country_iso3
        indicator_code
        year
    """

    unique = {}

    for record in records:

        key = (
            record["country_iso3"],
            record["indicator_code"],
            record["year"],
        )

        unique[key] = record

    return list(
        unique.values()
    )


# ============================================================
# VALIDATE FINAL DATASET
# ============================================================

def validate_dataset(
    records,
    economies,
):
    """
    Validate the extracted dataset before writing it.

    Expected maximum:

        economies × indicators × years
    """

    expected_rows = (
        len(economies)
        * len(INDICATORS)
        * (END_YEAR - START_YEAR + 1)
    )

    logger.info(
        "Expected maximum observations: %s",
        f"{expected_rows:,}",
    )

    logger.info(
        "Actual extracted observations: %s",
        f"{len(records):,}",
    )

    # --------------------------------------------------------
    # Check for duplicate grain.
    # --------------------------------------------------------

    keys = [
        (
            record["country_iso3"],
            record["indicator_code"],
            record["year"],
        )
        for record in records
    ]

    if len(keys) != len(set(keys)):

        raise ValueError(
            "Duplicate country/indicator/year "
            "records detected."
        )

    # --------------------------------------------------------
    # Check indicators.
    # --------------------------------------------------------

    actual_indicators = {
        record["indicator_code"]
        for record in records
    }

    expected_indicators = set(
        INDICATORS.keys()
    )

    missing_indicators = (
        expected_indicators
        - actual_indicators
    )

    if missing_indicators:

        raise ValueError(
            "Missing indicators: "
            f"{sorted(missing_indicators)}"
        )

    # --------------------------------------------------------
    # Check economies.
    # --------------------------------------------------------

    actual_economies = {
        record["country_iso3"]
        for record in records
    }

    unexpected_economies = (
        actual_economies
        - set(economies.keys())
    )

    if unexpected_economies:

        raise ValueError(
            "Unexpected economies found: "
            f"{sorted(unexpected_economies)}"
        )

    # --------------------------------------------------------
    # Check years.
    # --------------------------------------------------------

    invalid_years = [
        record["year"]
        for record in records
        if not (
            START_YEAR
            <= record["year"]
            <= END_YEAR
        )
    ]

    if invalid_years:

        raise ValueError(
            "Records outside configured year range detected."
        )

    logger.info(
        "Dataset validation passed."
    )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(records):
    """
    Write normalized records to CSV.

    Uses Python's csv module to avoid an unnecessary
    pandas dependency in the ingestion layer.
    """

    import csv

    fieldnames = [
        "country_name",
        "country_iso3",
        "indicator_code",
        "indicator_name",
        "year",
        "value",
        "run_id",
        "source",
        "extracted_at_utc",
    ]

    temp_file = OUTPUT_FILE.with_suffix(
        ".tmp"
    )

    with open(
        temp_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(records)

    # Atomic replacement.
    temp_file.replace(
        OUTPUT_FILE
    )

    logger.info(
        "CSV written successfully: %s",
        OUTPUT_FILE,
    )


# ============================================================
# LOAD STATE
# ============================================================

def load_state():
    """
    Load previous ingestion state.

    Returns empty dictionary if no state exists.
    """

    if not STATE_FILE.exists():

        return {}

    try:

        with open(
            STATE_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except (
        json.JSONDecodeError,
        OSError,
    ):

        logger.warning(
            "Could not read ingestion state. "
            "Starting fresh."
        )

        return {}


# ============================================================
# SAVE STATE
# ============================================================

def save_state(state):

    temp_file = STATE_FILE.with_suffix(
        ".tmp"
    )

    with open(
        temp_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            state,
            file,
            indent=2,
        )

    temp_file.replace(
        STATE_FILE
    )


# ============================================================
# BUILD CONFIGURATION SIGNATURE
# ============================================================

def build_configuration_signature(
    economies,
):
    """
    Create a deterministic signature for the current
    extraction configuration.

    This lets the pipeline detect that the old
    16-country / 5-indicator dataset has been replaced
    by the new 217-economy / 20-indicator dataset.
    """

    configuration = {
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "indicators": sorted(
            INDICATORS.keys()
        ),
        "economies": sorted(
            economies.keys()
        ),
    }

    return json.dumps(
        configuration,
        sort_keys=True,
    )


# ============================================================
# MAIN EXTRACTION
# ============================================================

def main():

    logger.info(
        "=" * 80
    )

    logger.info(
        "RetailPulse World Bank Extraction"
    )

    logger.info(
        "=" * 80
    )

    run_id = str(
        uuid.uuid4()
    )

    extracted_at = datetime.now(
        timezone.utc
    ).isoformat()

    logger.info(
        "Run ID: %s",
        run_id,
    )

    logger.info(
        "Period: %s-%s",
        START_YEAR,
        END_YEAR,
    )

    logger.info(
        "Indicators: %s",
        len(INDICATORS),
    )

    # --------------------------------------------------------
    # Fetch economy universe.
    # --------------------------------------------------------

    economies = fetch_economies()

    logger.info(
        "Using %s economies.",
        len(economies),
    )

    # --------------------------------------------------------
    # Configuration change detection.
    # --------------------------------------------------------

    state = load_state()

    configuration_signature = (
        build_configuration_signature(
            economies
        )
    )

    previous_signature = state.get(
        "configuration_signature"
    )

    if previous_signature != configuration_signature:

        logger.info(
            "Extraction configuration changed."
        )

        logger.info(
            "A full refresh will be performed."
        )

    else:

        logger.info(
            "Extraction configuration matches "
            "previous successful run."
        )

        if (
            state.get("last_successful_year")
            == END_YEAR
            and OUTPUT_FILE.exists()
        ):

            logger.info(
                "No new years detected."
            )

            logger.info(
                "Existing dataset is already up to date."
            )

            return

    # --------------------------------------------------------
    # Extract all indicators.
    # --------------------------------------------------------

    all_records = []

    for index, (
        indicator_code,
        indicator_name,
    ) in enumerate(
        INDICATORS.items(),
        start=1,
    ):

        logger.info(
            ""
        )

        logger.info(
            "[%s/%s] Processing %s",
            index,
            len(INDICATORS),
            indicator_code,
        )

        raw_records = fetch_indicator(
            indicator_code,
            indicator_name,
        )

        normalized = normalize_records(
            raw_records,
            economies,
            run_id,
            extracted_at,
        )

        logger.info(
            "%s normalized records: %s",
            indicator_code,
            len(normalized),
        )

        all_records.extend(
            normalized
        )

        time.sleep(0.25)

    # --------------------------------------------------------
    # Deduplicate.
    # --------------------------------------------------------

    logger.info(
        ""
    )

    logger.info(
        "Deduplicating records..."
    )

    all_records = deduplicate_records(
        all_records
    )

    logger.info(
        "Records after deduplication: %s",
        f"{len(all_records):,}",
    )

    # --------------------------------------------------------
    # Sort data.
    # --------------------------------------------------------

    all_records.sort(
        key=lambda record: (
            record["country_iso3"],
            record["indicator_code"],
            record["year"],
        )
    )

    # --------------------------------------------------------
    # Validate.
    # --------------------------------------------------------

    validate_dataset(
        all_records,
        economies,
    )

    # --------------------------------------------------------
    # Save CSV.
    # --------------------------------------------------------

    save_csv(
        all_records
    )

    # --------------------------------------------------------
    # Save ingestion state.
    # --------------------------------------------------------

    state = {
        "last_successful_year": END_YEAR,
        "last_run_id": run_id,
        "last_run_at_utc": extracted_at,
        "records_extracted": len(
            all_records
        ),
        "economies": len(
            economies
        ),
        "indicators": len(
            INDICATORS
        ),
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "status": "SUCCESS",
        "source": "World Bank API",
        "configuration_signature": (
            configuration_signature
        ),
    }

    save_state(
        state
    )

    # --------------------------------------------------------
    # Final summary.
    # --------------------------------------------------------

    logger.info(
        ""
    )

    logger.info(
        "=" * 80
    )

    logger.info(
        "EXTRACTION COMPLETE"
    )

    logger.info(
        "=" * 80
    )

    logger.info(
        "Economies: %s",
        len(economies),
    )

    logger.info(
        "Indicators: %s",
        len(INDICATORS),
    )

    logger.info(
        "Years: %s-%s",
        START_YEAR,
        END_YEAR,
    )

    logger.info(
        "Records: %s",
        f"{len(all_records):,}",
    )

    logger.info(
        "Output: %s",
        OUTPUT_FILE,
    )

    logger.info(
        "State: %s",
        STATE_FILE,
    )

    logger.info(
        "Run ID: %s",
        run_id,
    )

    logger.info(
        "Status: SUCCESS"
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()