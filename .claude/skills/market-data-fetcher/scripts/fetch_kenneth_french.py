#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "requests>=2.28",
# ]
# ///
"""
Kenneth French Data Library fetcher.

Downloads Fama-French factor data and portfolio returns from:
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

No authentication required. Data updates monthly.

Usage:
    uv run fetch_kenneth_french.py FF5_daily 2020-01-01 2024-12-31
    uv run fetch_kenneth_french.py MOM_daily 2020-01-01
    uv run fetch_kenneth_french.py --list  # List available datasets
    uv run fetch_kenneth_french.py  # Run self-tests
"""

import io
import zipfile
from pathlib import Path
from typing import Optional, Literal
import pandas as pd
import requests

from utils import (
    get_cache_manager,
    get_rate_limiter,
    normalize_date,
    create_identifier,
    handle_api_error
)

# Base URL for Kenneth French data files
BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

# Dataset mappings: user-friendly name -> actual filename on server
DATASETS = {
    # Fama-French 5 Factors
    'FF5': 'F-F_Research_Data_5_Factors_2x3',
    'FF5_daily': 'F-F_Research_Data_5_Factors_2x3_daily',
    # Fama-French 3 Factors (classic)
    'FF3': 'F-F_Research_Data_Factors',
    'FF3_daily': 'F-F_Research_Data_Factors_daily',
    # Momentum Factor
    'MOM': 'F-F_Momentum_Factor',
    'MOM_daily': 'F-F_Momentum_Factor_daily',
    # Industry Portfolios
    'IND49': '49_Industry_Portfolios',
    'IND49_daily': '49_Industry_Portfolios_daily',
    'IND30': '30_Industry_Portfolios',
    'IND30_daily': '30_Industry_Portfolios_daily',
    'IND17': '17_Industry_Portfolios',
    'IND17_daily': '17_Industry_Portfolios_daily',
    'IND10': '10_Industry_Portfolios',
    'IND10_daily': '10_Industry_Portfolios_daily',
    'IND5': '5_Industry_Portfolios',
    'IND5_daily': '5_Industry_Portfolios_daily',
    # Size Portfolios
    'SIZE': 'Portfolios_Formed_on_ME',
    'SIZE_daily': 'Portfolios_Formed_on_ME_daily',
    # Value Portfolios
    'VALUE': 'Portfolios_Formed_on_BE-ME',
    'VALUE_daily': 'Portfolios_Formed_on_BE-ME_daily',
    # Size and Value (6 portfolios)
    'SIZE_VALUE': '6_Portfolios_2x3',
    'SIZE_VALUE_daily': '6_Portfolios_2x3_daily',
    # Size and Momentum
    'SIZE_MOM': '25_Portfolios_ME_Prior_12_2',
    'SIZE_MOM_daily': '25_Portfolios_ME_Prior_12_2_daily',
}

# Factor symbols that should route to Kenneth French
FACTOR_SYMBOLS = {'MKT-RF', 'MKT', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'MOM', 'UMD', 'WML'}


def is_factor_symbol(symbol: str) -> bool:
    """
    Check if symbol is a Fama-French factor or dataset.

    Args:
        symbol: Symbol to check

    Returns:
        True if symbol should route to Kenneth French
    """
    symbol_upper = symbol.upper()
    return (
        symbol_upper in FACTOR_SYMBOLS
        or symbol_upper in DATASETS
        or symbol_upper.startswith(('FF5', 'FF3', 'FF_', 'IND', 'SIZE', 'VALUE', 'MOM'))
    )


def parse_kf_date(date_str: str) -> pd.Timestamp:
    """
    Parse Kenneth French date format.

    Args:
        date_str: Date string in YYYYMMDD (daily) or YYYYMM (monthly) format

    Returns:
        Parsed timestamp
    """
    date_str = str(date_str).strip()
    if len(date_str) == 8:  # YYYYMMDD (daily)
        return pd.to_datetime(date_str, format='%Y%m%d')
    elif len(date_str) == 6:  # YYYYMM (monthly)
        return pd.to_datetime(date_str + '01', format='%Y%m%d')
    raise ValueError(f"Unknown date format: {date_str}")


class KennethFrenchFetcher:
    """Fetcher for Kenneth French Data Library."""

    def __init__(self, use_cache: bool = True, cache_hours: int = 168):
        """
        Initialize Kenneth French fetcher.

        Args:
            use_cache: Whether to use caching
            cache_hours: Cache validity in hours (default: 168 = 1 week)
        """
        self.use_cache = use_cache
        self.cache_hours = cache_hours
        self.cache = get_cache_manager()
        self.rate_limiter = get_rate_limiter()

    def fetch(self, dataset: str, start_date: Optional[str] = None,
              end_date: Optional[str] = None, frequency: str = 'auto') -> pd.DataFrame:
        """
        Fetch factor or portfolio data from Kenneth French Data Library.

        Args:
            dataset: Dataset name (e.g., 'FF5_daily', 'MOM', 'IND49')
                    Or factor symbol (e.g., 'SMB', 'HML', 'MKT-RF')
            start_date: Start date (YYYY-MM-DD or YYYYMMDD)
            end_date: End date (YYYY-MM-DD or YYYYMMDD)
            frequency: 'daily', 'monthly', or 'auto' (infer from dataset name)

        Returns:
            DataFrame with factor/portfolio returns (values in decimal form)

        Raises:
            ValueError: If dataset not found or no data returned
        """
        # Handle factor symbol requests (route to appropriate dataset)
        dataset_upper = dataset.upper()
        if dataset_upper in FACTOR_SYMBOLS:
            # Route factor symbols to FF5 dataset
            is_daily = frequency == 'daily'
            actual_dataset = 'FF5_daily' if is_daily else 'FF5'
            return self._fetch_and_filter_factor(actual_dataset, dataset_upper, start_date, end_date)

        # Resolve dataset name
        resolved_dataset = self._resolve_dataset(dataset)

        # Create cache identifier
        start_norm = normalize_date(start_date) if start_date else None
        end_norm = normalize_date(end_date) if end_date else None
        cache_id = create_identifier(resolved_dataset, start_norm, end_norm)

        # Try cache first
        if self.use_cache:
            cached = self.cache.get('kenneth_french', cache_id, max_age_hours=self.cache_hours)
            if cached is not None:
                print(f"[Kenneth French] Using cached data for {dataset}")
                # Apply date filtering to cached data
                return self._filter_by_date(cached, start_date, end_date)

        # Download and parse
        df = self._download_dataset(resolved_dataset)

        if df.empty:
            raise ValueError(f"No data returned for dataset: {dataset}")

        # Cache the full dataset
        if self.use_cache:
            self.cache.set('kenneth_french', cache_id, df)

        # Filter by date
        return self._filter_by_date(df, start_date, end_date)

    def _fetch_and_filter_factor(self, dataset: str, factor: str,
                                  start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        """
        Fetch a dataset and filter to specific factor column.

        Args:
            dataset: Full dataset name
            factor: Factor symbol to extract
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with Date and factor value columns
        """
        df = self.fetch(dataset, start_date, end_date)

        # Normalize factor name for column matching
        factor_upper = factor.upper()
        factor_map = {
            'MKT': 'Mkt-RF',
            'MKT-RF': 'Mkt-RF',
            'MOM': 'Mom',
            'UMD': 'Mom',
            'WML': 'Mom',
        }
        col_name = factor_map.get(factor_upper, factor_upper)

        # Find matching column (case-insensitive)
        matching_cols = [c for c in df.columns if c.upper() == col_name.upper() or c == col_name]
        if not matching_cols:
            available = [c for c in df.columns if c != 'Date']
            raise ValueError(f"Factor '{factor}' not found. Available: {available}")

        result = df[['Date', matching_cols[0]]].copy()
        result.columns = ['Date', factor_upper]
        return result

    def _resolve_dataset(self, dataset: str) -> str:
        """
        Resolve user-friendly dataset name to actual filename.

        Args:
            dataset: User-provided dataset name

        Returns:
            Actual filename on server
        """
        dataset_upper = dataset.upper()

        # Check exact match in mapping
        if dataset_upper in DATASETS:
            return DATASETS[dataset_upper]

        # Check if already a valid filename
        if dataset in DATASETS.values():
            return dataset

        # Try case-insensitive match
        for key, value in DATASETS.items():
            if key.upper() == dataset_upper:
                return value

        # Assume it's a direct filename
        return dataset

    def _download_dataset(self, dataset: str) -> pd.DataFrame:
        """
        Download and parse dataset from Kenneth French website.

        Args:
            dataset: Dataset filename (without _CSV.zip extension)

        Returns:
            Parsed DataFrame
        """
        url = f"{BASE_URL}/{dataset}_CSV.zip"

        try:
            self.rate_limiter.wait('kenneth_french')

            print(f"[Kenneth French] Downloading {dataset}...")
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            # Extract CSV from ZIP
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                # Find the CSV file in the archive
                csv_files = [f for f in zf.namelist() if f.endswith('.CSV') or f.endswith('.csv')]
                if not csv_files:
                    raise ValueError(f"No CSV file found in archive for {dataset}")

                csv_name = csv_files[0]
                with zf.open(csv_name) as csv_file:
                    return self._parse_csv(csv_file, dataset)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Dataset not found: {dataset}. Use --list to see available datasets.")
            handle_api_error('Kenneth French', e, dataset)
            raise
        except Exception as e:
            handle_api_error('Kenneth French', e, dataset)
            raise

    def _parse_csv(self, csv_file, dataset: str) -> pd.DataFrame:
        """
        Parse Kenneth French CSV format.

        The format has:
        - Header rows to skip (vary by file)
        - Date in first column (YYYYMMDD or YYYYMM)
        - Values as percentages (need to divide by 100)
        - May have multiple tables (we take the first one)

        Args:
            csv_file: File-like object
            dataset: Dataset name for context

        Returns:
            Parsed DataFrame with Date column and values in decimal form
        """
        # Read raw content
        content = csv_file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='ignore')

        lines = content.strip().split('\n')

        # Find header row (first row with column names)
        header_idx = None
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            # Skip empty lines and description lines
            if not line_stripped or line_stripped.startswith('This file'):
                continue
            # Look for numeric data start (first column should be date-like)
            parts = [p.strip() for p in line_stripped.split(',')]
            if len(parts) >= 2:
                first_part = parts[0].strip()
                # Check if first part looks like a date (all digits, 6 or 8 chars)
                if first_part.isdigit() and len(first_part) in [6, 8]:
                    # Previous non-empty line is the header
                    for j in range(i - 1, -1, -1):
                        if lines[j].strip():
                            header_idx = j
                            break
                    break

        if header_idx is None:
            # Fallback: try pandas auto-detection
            csv_file.seek(0) if hasattr(csv_file, 'seek') else None
            try:
                df = pd.read_csv(io.StringIO(content), skiprows=0)
                if 'Date' in df.columns or df.columns[0].isdigit():
                    return self._standardize_df(df)
            except Exception:
                pass
            raise ValueError(f"Could not parse CSV format for {dataset}")

        # Read with identified header
        df = pd.read_csv(
            io.StringIO('\n'.join(lines[header_idx:])),
            skipinitialspace=True,
            on_bad_lines='skip'
        )

        return self._standardize_df(df)

    def _standardize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize DataFrame format.

        - Parse dates
        - Convert percentages to decimals
        - Clean column names
        - Remove rows with invalid data

        Args:
            df: Raw DataFrame

        Returns:
            Standardized DataFrame
        """
        # Identify date column (usually first column or named 'Date', 'Unnamed: 0')
        date_col = None
        for col in df.columns:
            if col.lower() in ['date', 'unnamed: 0', '']:
                date_col = col
                break

        if date_col is None:
            date_col = df.columns[0]

        # Parse dates
        dates = []
        valid_rows = []
        for idx, val in df[date_col].items():
            try:
                date_val = str(val).strip()
                # Skip non-numeric values (headers, section breaks)
                if not date_val.replace('.', '').replace('-', '').isdigit():
                    continue
                # Handle potential float representation
                if '.' in date_val:
                    date_val = date_val.split('.')[0]
                parsed_date = parse_kf_date(date_val)
                dates.append(parsed_date)
                valid_rows.append(idx)
            except (ValueError, TypeError):
                continue

        if not dates:
            return pd.DataFrame()

        # Filter to valid rows
        df = df.loc[valid_rows].copy()
        df['Date'] = dates

        # Get value columns (all except date column)
        value_cols = [c for c in df.columns if c != date_col and c != 'Date']

        # Convert to numeric and divide by 100 (percentages to decimals)
        for col in value_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce') / 100

        # Clean column names
        df = df[['Date'] + value_cols].copy()

        # Remove rows where all values are NaN
        df.dropna(how='all', subset=value_cols, inplace=True)

        # Sort by date
        df.sort_values('Date', inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    def _filter_by_date(self, df: pd.DataFrame, start_date: Optional[str],
                        end_date: Optional[str]) -> pd.DataFrame:
        """
        Filter DataFrame by date range.

        Args:
            df: DataFrame with Date column
            start_date: Start date
            end_date: End date

        Returns:
            Filtered DataFrame
        """
        if df.empty or 'Date' not in df.columns:
            return df

        result = df.copy()

        if start_date:
            start_dt = pd.to_datetime(normalize_date(start_date), format='%Y%m%d')
            result = result[result['Date'] >= start_dt]

        if end_date:
            end_dt = pd.to_datetime(normalize_date(end_date), format='%Y%m%d')
            result = result[result['Date'] <= end_dt]

        return result.reset_index(drop=True)

    def list_datasets(self) -> list[str]:
        """
        List available dataset names.

        Returns:
            List of dataset identifiers
        """
        return sorted(DATASETS.keys())


def fetch_kenneth_french(dataset: str, start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         use_cache: bool = True) -> pd.DataFrame:
    """
    Convenience function to fetch Kenneth French data.

    Args:
        dataset: Dataset name or factor symbol
        start_date: Start date
        end_date: End date
        use_cache: Whether to use caching

    Returns:
        DataFrame with factor/portfolio data
    """
    fetcher = KennethFrenchFetcher(use_cache=use_cache)
    return fetcher.fetch(dataset, start_date, end_date)


if __name__ == '__main__':
    import sys

    # Handle --list flag
    if len(sys.argv) > 1 and sys.argv[1] == '--list':
        print("Available Kenneth French datasets:\n")
        fetcher = KennethFrenchFetcher()
        for ds in fetcher.list_datasets():
            actual = DATASETS.get(ds, ds)
            print(f"  {ds:20s} -> {actual}")
        print("\nFactor symbols (route to FF5):")
        for sym in sorted(FACTOR_SYMBOLS):
            print(f"  {sym}")
        sys.exit(0)

    # Handle CLI arguments: dataset [start_date] [end_date]
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
        start_date = sys.argv[2] if len(sys.argv) > 2 else '2020-01-01'
        end_date = sys.argv[3] if len(sys.argv) > 3 else None

        print(f"Fetching {dataset} from {start_date} to {end_date or 'latest'}...\n")
        df = fetch_kenneth_french(dataset, start_date, end_date)
        print(f"Retrieved {len(df)} rows")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nLast 5 rows:")
        print(df.tail())
        sys.exit(0)

    # Self-tests
    print("Testing Kenneth French fetcher...\n")

    fetcher = KennethFrenchFetcher()

    # Test 1: List datasets
    print("1. Available datasets:")
    datasets = fetcher.list_datasets()
    print(f"   {len(datasets)} datasets available")
    print(f"   Examples: {datasets[:5]}")

    # Test 2: FF5 daily factors
    print("\n2. Fetching FF5_daily (5 factors, daily):")
    try:
        df = fetcher.fetch('FF5_daily', start_date='2024-01-01', end_date='2024-01-31')
        print(f"   Retrieved {len(df)} rows")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Sample values (should be decimals ~0.01, not percentages ~1.0):")
        print(df.head(3).to_string(index=False))
    except Exception as e:
        print(f"   Error: {e}")

    # Test 3: Momentum factor
    print("\n3. Fetching MOM_daily (momentum factor):")
    try:
        df = fetcher.fetch('MOM_daily', start_date='2024-01-01', end_date='2024-01-31')
        print(f"   Retrieved {len(df)} rows")
        print(f"   Columns: {list(df.columns)}")
        print(df.head(3).to_string(index=False))
    except Exception as e:
        print(f"   Error: {e}")

    # Test 4: Single factor symbol
    print("\n4. Fetching SMB (size factor from FF5):")
    try:
        df = fetcher.fetch('SMB', start_date='2024-01-01', end_date='2024-01-31')
        print(f"   Retrieved {len(df)} rows")
        print(f"   Columns: {list(df.columns)}")
        print(df.head(3).to_string(index=False))
    except Exception as e:
        print(f"   Error: {e}")

    # Test 5: Monthly data
    print("\n5. Fetching FF3 (3 factors, monthly):")
    try:
        df = fetcher.fetch('FF3', start_date='2020-01-01', end_date='2024-12-31')
        print(f"   Retrieved {len(df)} rows (monthly)")
        print(f"   Columns: {list(df.columns)}")
        print(df.tail(3).to_string(index=False))
    except Exception as e:
        print(f"   Error: {e}")

    # Test 6: Industry portfolios
    print("\n6. Fetching IND10_daily (10 industry portfolios):")
    try:
        df = fetcher.fetch('IND10_daily', start_date='2024-01-01', end_date='2024-01-15')
        print(f"   Retrieved {len(df)} rows")
        print(f"   Columns: {list(df.columns)}")
        print(df.head(3).to_string(index=False))
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "="*50)
    print("Kenneth French fetcher tests complete!")
