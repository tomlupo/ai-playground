#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "requests>=2.28",
# ]
# ///
"""
Download Fama-French factors and FRED data using direct API calls.
Workaround for pandas-datareader issues with Python 3.12+.

Usage:
    uv run scripts/download_factors_fred.py
"""

import io
import zipfile
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests

OUTPUT_DIR = Path(__file__).parent.parent / "data/samples"


def download_fama_french_5_factors() -> bool:
    """Download Fama-French 5 factors directly from Ken French's website."""
    print("  Downloading Fama-French 5 Factors (daily)...", end=" ", flush=True)
    try:
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Find the CSV file inside
            csv_name = [n for n in z.namelist() if n.endswith('.CSV') or n.endswith('.csv')][0]
            with z.open(csv_name) as f:
                # Skip header lines until we find the data
                lines = f.read().decode('utf-8').split('\n')

                # Find where data starts (after header lines)
                data_start = 0
                for i, line in enumerate(lines):
                    if line.strip() and line.strip()[0].isdigit():
                        data_start = i
                        break

                # Read the data
                data_lines = []
                for line in lines[data_start:]:
                    if line.strip() and line.strip()[0].isdigit() and len(line.strip().split(',')) >= 6:
                        data_lines.append(line.strip())

                # Create DataFrame
                df = pd.DataFrame([line.split(',') for line in data_lines])
                df.columns = ["Date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]

                # Convert date format (YYYYMMDD -> datetime)
                df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")

                # Convert values to float (they're in percent, divide by 100)
                for col in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]:
                    df[col] = pd.to_numeric(df[col], errors='coerce') / 100

                df = df.dropna()
                df.to_csv(OUTPUT_DIR / "FF5_factors.csv", index=False)
                print(f"OK ({len(df)} rows)")
                return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def download_fama_french_momentum() -> bool:
    """Download Fama-French momentum factor directly."""
    print("  Downloading Momentum Factor (daily)...", end=" ", flush=True)
    try:
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_name = [n for n in z.namelist() if n.endswith('.CSV') or n.endswith('.csv')][0]
            with z.open(csv_name) as f:
                lines = f.read().decode('utf-8').split('\n')

                # Find where data starts
                data_start = 0
                for i, line in enumerate(lines):
                    if line.strip() and line.strip()[0].isdigit():
                        data_start = i
                        break

                # Read the data
                data_lines = []
                for line in lines[data_start:]:
                    parts = line.strip().split(',')
                    if line.strip() and line.strip()[0].isdigit() and len(parts) >= 2:
                        data_lines.append(parts[:2])

                df = pd.DataFrame(data_lines, columns=["Date", "Mom"])
                df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
                df["Mom"] = pd.to_numeric(df["Mom"], errors='coerce') / 100
                df = df.dropna()
                df.to_csv(OUTPUT_DIR / "FF_momentum.csv", index=False)
                print(f"OK ({len(df)} rows)")
                return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def download_fred_series(series_id: str, description: str, start_date: str = "2000-01-01") -> bool:
    """Download FRED series directly via their JSON API."""
    print(f"  Downloading {series_id} ({description})...", end=" ", flush=True)
    try:
        # FRED's free JSON endpoint (no API key needed for basic access)
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start_date}"

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        df = pd.read_csv(io.StringIO(response.text))
        df.columns = ["Date", "Value"]

        # Handle missing values (FRED uses ".")
        df["Value"] = pd.to_numeric(df["Value"], errors='coerce')
        df = df.dropna()

        df.to_csv(OUTPUT_DIR / f"FRED_{series_id}.csv", index=False)
        print(f"OK ({len(df)} rows)")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    print("=" * 60)
    print("Downloading Fama-French Factors and FRED Data")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {"success": [], "failed": []}

    # Fama-French factors
    print("\n--- Fama-French Factors ---")
    if download_fama_french_5_factors():
        results["success"].append("FF5_factors")
    else:
        results["failed"].append("FF5_factors")

    if download_fama_french_momentum():
        results["success"].append("FF_momentum")
    else:
        results["failed"].append("FF_momentum")

    # FRED series
    print("\n--- FRED Economic Indicators ---")
    fred_series = {
        "GDP": "Gross Domestic Product",
        "CPIAUCSL": "Consumer Price Index",
        "UNRATE": "Unemployment Rate",
        "FEDFUNDS": "Federal Funds Rate",
        "T10Y2Y": "10Y-2Y Treasury Spread (recession indicator)",
        "DGS10": "10-Year Treasury Yield",
        "DGS2": "2-Year Treasury Yield",
    }

    for series_id, description in fred_series.items():
        if download_fred_series(series_id, description):
            results["success"].append(f"FRED_{series_id}")
        else:
            results["failed"].append(f"FRED_{series_id}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Successfully downloaded: {len(results['success'])} datasets")
    print(f"Failed: {len(results['failed'])} datasets")

    if results["failed"]:
        print(f"\nFailed: {', '.join(results['failed'])}")

    return 0 if not results["failed"] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
