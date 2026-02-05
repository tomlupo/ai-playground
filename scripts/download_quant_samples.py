#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "yfinance>=0.2.28",
#   "pandas-datareader>=0.10",
#   "requests>=2.28",
# ]
# ///
"""
Download comprehensive data samples for quant research.

Covers ~90% of typical quant investing research needs:
- US equities (large cap, small cap, tech)
- Fixed income (treasuries)
- Alternatives (gold, real estate, crypto)
- International (developed, emerging)
- Volatility (VIX)
- Sector ETFs
- Fama-French factors
- FRED economic indicators

Usage:
    uv run scripts/download_quant_samples.py
    uv run scripts/download_quant_samples.py --start 2005-01-01
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse

import pandas as pd
import yfinance as yf

# Add market-data-fetcher scripts to path
SKILL_PATH = Path(__file__).parent.parent / ".claude/skills/market-data-fetcher/scripts"
sys.path.insert(0, str(SKILL_PATH))

OUTPUT_DIR = Path(__file__).parent.parent / "data/samples"


# --- Configuration ---

# Tier 1: Essential (~70% coverage)
TIER1_TICKERS = {
    "SPY": "S&P 500 ETF (large cap US)",
    "QQQ": "Nasdaq 100 ETF (tech-heavy)",
    "TLT": "20+ Year Treasury ETF",
    "IEF": "7-10 Year Treasury ETF",
    "SHY": "1-3 Year Treasury ETF",
    "GLD": "Gold ETF",
}

# Tier 2: Diversification (~15% more)
TIER2_TICKERS = {
    "IWM": "Russell 2000 ETF (small cap)",
    "EFA": "Developed Markets ex-US ETF",
    "EEM": "Emerging Markets ETF",
    "VNQ": "Real Estate ETF",
    "DBC": "Commodities ETF",
    "BTC-USD": "Bitcoin",
}

# Tier 3: Sector rotation
SECTOR_TICKERS = {
    "XLF": "Financials",
    "XLE": "Energy",
    "XLK": "Technology",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
}

# Special: VIX (requires ^VIX format for Yahoo)
VIX_TICKER = "^VIX"

# FRED economic indicators
FRED_SERIES = {
    "GDP": "Gross Domestic Product",
    "CPIAUCSL": "Consumer Price Index",
    "UNRATE": "Unemployment Rate",
    "FEDFUNDS": "Federal Funds Rate",
    "T10Y2Y": "10Y-2Y Treasury Spread",
    "VIXCLS": "VIX (daily from FRED)",
}


def download_yahoo_ticker(ticker: str, start: str, end: str, output_dir: Path) -> bool:
    """Download a single ticker from Yahoo Finance."""
    print(f"  Downloading {ticker}...", end=" ", flush=True)
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            print("NO DATA")
            return False

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Standardize column names
        df = df.reset_index()
        df.columns = [c.replace(" ", "_") for c in df.columns]

        # Save
        filename = ticker.replace("^", "").replace("-", "_").replace("/", "_")
        filepath = output_dir / f"{filename}.csv"
        df.to_csv(filepath, index=False)
        print(f"OK ({len(df)} rows, {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')})")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def download_fama_french(output_dir: Path) -> bool:
    """Download Fama-French factors from pandas-datareader."""
    print("\n--- Fama-French Factors ---")
    try:
        import pandas_datareader.data as web

        # 5 Factors (daily)
        print("  Downloading Fama-French 5 Factors...", end=" ", flush=True)
        ff5 = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start="2000-01-01")[0]
        ff5 = ff5.reset_index()
        ff5.columns = ["Date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
        ff5.to_csv(output_dir / "FF5_factors.csv", index=False)
        print(f"OK ({len(ff5)} rows)")

        # Momentum factor (daily)
        print("  Downloading Momentum Factor...", end=" ", flush=True)
        mom = web.DataReader("F-F_Momentum_Factor_daily", "famafrench", start="2000-01-01")[0]
        mom = mom.reset_index()
        mom.columns = ["Date", "Mom"]
        mom.to_csv(output_dir / "FF_momentum.csv", index=False)
        print(f"OK ({len(mom)} rows)")

        return True
    except Exception as e:
        print(f"  ERROR downloading Fama-French: {e}")
        return False


def download_fred_series(output_dir: Path, start: str) -> bool:
    """Download FRED economic indicators."""
    print("\n--- FRED Economic Indicators ---")
    try:
        import pandas_datareader.data as web

        success_count = 0
        for series_id, description in FRED_SERIES.items():
            print(f"  Downloading {series_id} ({description})...", end=" ", flush=True)
            try:
                df = web.DataReader(series_id, "fred", start=start)
                df = df.reset_index()
                df.columns = ["Date", "Value"]
                df.to_csv(output_dir / f"FRED_{series_id}.csv", index=False)
                print(f"OK ({len(df)} rows)")
                success_count += 1
            except Exception as e:
                print(f"ERROR: {e}")

        return success_count > 0
    except ImportError:
        print("  pandas-datareader not available for FRED")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download quant research data samples")
    parser.add_argument("--start", default="2005-01-01", help="Start date (default: 2005-01-01)")
    parser.add_argument("--end", default=None, help="End date (default: today)")
    parser.add_argument("--tier", type=int, default=3, choices=[1, 2, 3],
                       help="Download tier: 1=essential, 2=+diversification, 3=+sectors")
    args = parser.parse_args()

    end_date = args.end or datetime.now().strftime("%Y-%m-%d")

    print(f"=" * 60)
    print(f"Downloading Quant Research Data Samples")
    print(f"Period: {args.start} to {end_date}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {"success": [], "failed": []}

    # Tier 1: Essential
    print("\n--- Tier 1: Essential Assets ---")
    for ticker, desc in TIER1_TICKERS.items():
        if download_yahoo_ticker(ticker, args.start, end_date, OUTPUT_DIR):
            results["success"].append(ticker)
        else:
            results["failed"].append(ticker)

    # VIX
    print("\n--- Volatility Index ---")
    if download_yahoo_ticker(VIX_TICKER, args.start, end_date, OUTPUT_DIR):
        results["success"].append("VIX")
    else:
        results["failed"].append("VIX")

    if args.tier >= 2:
        # Tier 2: Diversification
        print("\n--- Tier 2: Diversification Assets ---")
        for ticker, desc in TIER2_TICKERS.items():
            if download_yahoo_ticker(ticker, args.start, end_date, OUTPUT_DIR):
                results["success"].append(ticker)
            else:
                results["failed"].append(ticker)

    if args.tier >= 3:
        # Tier 3: Sectors
        print("\n--- Tier 3: Sector ETFs ---")
        for ticker, desc in SECTOR_TICKERS.items():
            if download_yahoo_ticker(ticker, args.start, end_date, OUTPUT_DIR):
                results["success"].append(ticker)
            else:
                results["failed"].append(ticker)

    # Fama-French factors
    if download_fama_french(OUTPUT_DIR):
        results["success"].extend(["FF5_factors", "FF_momentum"])
    else:
        results["failed"].extend(["FF5_factors", "FF_momentum"])

    # FRED economic indicators
    if download_fred_series(OUTPUT_DIR, args.start):
        results["success"].append("FRED_series")
    else:
        results["failed"].append("FRED_series")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Successfully downloaded: {len(results['success'])} datasets")
    print(f"Failed: {len(results['failed'])} datasets")

    if results["failed"]:
        print(f"\nFailed downloads: {', '.join(results['failed'])}")

    # List output files
    print(f"\nOutput files in {OUTPUT_DIR}:")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}: {size_kb:.1f} KB")

    return 0 if not results["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
