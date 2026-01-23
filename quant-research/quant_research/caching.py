"""
Caching module for expensive operations.

Provides disk and memory caching for:
- Data fetching
- Indicator calculations
- ML model training
- Portfolio optimization
"""

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable
import pickle

import pandas as pd
import numpy as np
from diskcache import Cache
from joblib import Memory


# Default cache directory
CACHE_DIR = Path.home() / ".cache" / "quant_research"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Initialize caches
_disk_cache = Cache(str(CACHE_DIR / "diskcache"))
_joblib_memory = Memory(str(CACHE_DIR / "joblib"), verbose=0)


def get_cache_key(*args, **kwargs) -> str:
    """Generate a unique cache key from arguments."""
    key_data = {
        "args": [_serialize_arg(a) for a in args],
        "kwargs": {k: _serialize_arg(v) for k, v in sorted(kwargs.items())},
    }
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_str.encode()).hexdigest()


def _serialize_arg(arg: Any) -> Any:
    """Serialize an argument for hashing."""
    if isinstance(arg, pd.DataFrame):
        return f"df:{arg.shape}:{hash(str(arg.columns.tolist()))}:{hash(str(arg.index[:5].tolist()))}"
    elif isinstance(arg, pd.Series):
        return f"series:{len(arg)}:{arg.name}:{hash(str(arg.index[:5].tolist()))}"
    elif isinstance(arg, np.ndarray):
        return f"array:{arg.shape}:{arg.dtype}"
    elif isinstance(arg, (list, tuple)):
        return [_serialize_arg(a) for a in arg]
    elif isinstance(arg, dict):
        return {k: _serialize_arg(v) for k, v in arg.items()}
    else:
        return arg


class CacheManager:
    """
    Centralized cache management.

    Supports multiple caching backends and TTL.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        default_ttl: int = 3600,  # 1 hour
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl

        self._disk_cache = Cache(str(self.cache_dir / "disk"))
        self._memory_cache: dict[str, tuple[Any, datetime]] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        # Try memory first
        if key in self._memory_cache:
            value, expires = self._memory_cache[key]
            if expires > datetime.now():
                return value
            else:
                del self._memory_cache[key]

        # Try disk
        if key in self._disk_cache:
            return self._disk_cache[key]

        return default

    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        memory_only: bool = False,
    ) -> None:
        """Set value in cache."""
        ttl = ttl or self.default_ttl
        expires = datetime.now() + timedelta(seconds=ttl)

        # Always store in memory
        self._memory_cache[key] = (value, expires)

        # Store in disk unless memory_only
        if not memory_only:
            self._disk_cache.set(key, value, expire=ttl)

    def delete(self, key: str) -> None:
        """Delete from cache."""
        if key in self._memory_cache:
            del self._memory_cache[key]
        if key in self._disk_cache:
            del self._disk_cache[key]

    def clear(self) -> None:
        """Clear all caches."""
        self._memory_cache.clear()
        self._disk_cache.clear()

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "memory_entries": len(self._memory_cache),
            "disk_entries": len(self._disk_cache),
            "disk_size_mb": self._disk_cache.volume() / (1024 * 1024),
        }


# Global cache manager
_cache_manager = CacheManager()


def cached(
    ttl: int = 3600,
    key_prefix: str = "",
    memory_only: bool = False,
):
    """
    Decorator for caching function results.

    Args:
        ttl: Time-to-live in seconds
        key_prefix: Prefix for cache key
        memory_only: Only use memory cache
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            base_key = get_cache_key(*args, **kwargs)
            cache_key = f"{key_prefix}:{func.__name__}:{base_key}"

            # Try to get from cache
            result = _cache_manager.get(cache_key)
            if result is not None:
                return result

            # Execute function
            result = func(*args, **kwargs)

            # Cache result
            _cache_manager.set(cache_key, result, ttl=ttl, memory_only=memory_only)

            return result

        return wrapper

    return decorator


def cache_dataframe(
    ttl: int = 3600,
    key_prefix: str = "df",
):
    """
    Decorator specifically for DataFrame-returning functions.

    Stores DataFrames efficiently using parquet format.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            base_key = get_cache_key(*args, **kwargs)
            cache_key = f"{key_prefix}:{func.__name__}:{base_key}"

            # Check file cache
            cache_file = CACHE_DIR / "dataframes" / f"{cache_key}.parquet"
            if cache_file.exists():
                # Check TTL
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if datetime.now() - mtime < timedelta(seconds=ttl):
                    return pd.read_parquet(cache_file)

            # Execute function
            result = func(*args, **kwargs)

            # Cache as parquet
            if isinstance(result, pd.DataFrame):
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                result.to_parquet(cache_file)

            return result

        return wrapper

    return decorator


class DataCache:
    """
    Specialized cache for market data.

    Handles incremental updates and data validation.
    """

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR / "market_data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, symbol: str, interval: str = "1d") -> Path:
        """Get cache file path for symbol."""
        return self.cache_dir / f"{symbol}_{interval}.parquet"

    def get(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame | None:
        """Get cached data for symbol."""
        cache_path = self._get_path(symbol, interval)

        if not cache_path.exists():
            return None

        df = pd.read_parquet(cache_path)

        # Filter by date range
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        return df if not df.empty else None

    def set(
        self,
        symbol: str,
        data: pd.DataFrame,
        interval: str = "1d",
    ) -> None:
        """Cache data for symbol."""
        cache_path = self._get_path(symbol, interval)

        # Merge with existing data if present
        if cache_path.exists():
            existing = pd.read_parquet(cache_path)
            data = pd.concat([existing, data])
            data = data[~data.index.duplicated(keep="last")]
            data = data.sort_index()

        data.to_parquet(cache_path)

    def get_last_date(self, symbol: str, interval: str = "1d") -> datetime | None:
        """Get the last cached date for symbol."""
        cache_path = self._get_path(symbol, interval)

        if not cache_path.exists():
            return None

        df = pd.read_parquet(cache_path)
        return df.index[-1] if not df.empty else None

    def clear(self, symbol: str | None = None) -> None:
        """Clear cache for symbol or all symbols."""
        if symbol:
            for path in self.cache_dir.glob(f"{symbol}_*.parquet"):
                path.unlink()
        else:
            for path in self.cache_dir.glob("*.parquet"):
                path.unlink()


class IndicatorCache:
    """
    Cache for computed indicators.

    Stores indicators indexed by symbol and parameters.
    """

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR / "indicators"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, pd.Series] = {}

    def _get_key(
        self, symbol: str, indicator: str, params: dict
    ) -> str:
        """Generate cache key for indicator."""
        params_str = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
        return f"{symbol}_{indicator}_{params_str}"

    def get(
        self,
        symbol: str,
        indicator: str,
        params: dict,
    ) -> pd.Series | None:
        """Get cached indicator."""
        key = self._get_key(symbol, indicator, params)

        # Memory cache
        if key in self._memory:
            return self._memory[key]

        # Disk cache
        cache_path = self.cache_dir / f"{key}.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            series = df.iloc[:, 0]
            self._memory[key] = series
            return series

        return None

    def set(
        self,
        symbol: str,
        indicator: str,
        params: dict,
        data: pd.Series,
    ) -> None:
        """Cache indicator."""
        key = self._get_key(symbol, indicator, params)

        # Memory
        self._memory[key] = data

        # Disk
        cache_path = self.cache_dir / f"{key}.parquet"
        df = data.to_frame()
        df.to_parquet(cache_path)

    def clear(self) -> None:
        """Clear all cached indicators."""
        self._memory.clear()
        for path in self.cache_dir.glob("*.parquet"):
            path.unlink()


class ModelCache:
    """
    Cache for trained ML models.

    Uses joblib for efficient model serialization.
    """

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, model_name: str, params_hash: str) -> Path:
        """Get cache path for model."""
        return self.cache_dir / f"{model_name}_{params_hash}.joblib"

    def get(
        self,
        model_name: str,
        data_hash: str,
        params: dict,
    ) -> Any | None:
        """Get cached model."""
        params_hash = get_cache_key(data_hash, params)
        cache_path = self._get_path(model_name, params_hash)

        if cache_path.exists():
            import joblib
            return joblib.load(cache_path)

        return None

    def set(
        self,
        model_name: str,
        data_hash: str,
        params: dict,
        model: Any,
    ) -> None:
        """Cache trained model."""
        import joblib
        params_hash = get_cache_key(data_hash, params)
        cache_path = self._get_path(model_name, params_hash)
        joblib.dump(model, cache_path)

    def clear(self) -> None:
        """Clear all cached models."""
        for path in self.cache_dir.glob("*.joblib"):
            path.unlink()


def clear_all_caches() -> None:
    """Clear all caches."""
    _cache_manager.clear()
    DataCache().clear()
    IndicatorCache().clear()
    ModelCache().clear()


def get_cache_info() -> dict:
    """Get information about all caches."""
    return {
        "general": _cache_manager.get_stats(),
        "cache_dir": str(CACHE_DIR),
        "total_size_mb": sum(
            f.stat().st_size for f in CACHE_DIR.rglob("*") if f.is_file()
        )
        / (1024 * 1024),
    }
