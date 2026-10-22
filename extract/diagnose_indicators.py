import requests

API_BASE = "https://api.worldbank.org/v2"

INDICATORS = [
    "SL.UEM.TOTL.ZS",
    "NE.CON.PRVT.KD.ZG",
    "NE.GDI.TOTL.ZS",
    "SL.TLF.CACT.ZS",
    "EN.GHG.CO2.PC.CE.AR5",
]

for code in INDICATORS:
    print("\n" + "=" * 80)
    print(f"INDICATOR: {code}")
    print("=" * 80)

    url = f"{API_BASE}/country/all/indicator/{code}"

    params = {
        "format": "json",
        "source": 2,
        "date": "2000:2024",
        "per_page": 5,
        "page": 1,
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=30,
        )

        print("HTTP STATUS:", response.status_code)
        print("FINAL URL:", response.url)

        data = response.json()

        if isinstance(data, list):
            print("RESPONSE TYPE: list")
            print("NUMBER OF TOP-LEVEL ITEMS:", len(data))

            if len(data) > 0:
                print("\nFIRST ITEM:")
                print(data[0])

            if len(data) > 1:
                print("\nFIRST RECORD:")
                print(data[1][0] if data[1] else "No records")

        else:
            print("UNEXPECTED RESPONSE:")
            print(data)

    except Exception as e:
        print("ERROR:", repr(e))