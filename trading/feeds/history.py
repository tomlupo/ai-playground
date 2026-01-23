"""
Historical data storage and retrieval.

Provides multiple storage backends:
- SQLite: Simple, file-based, good for development
- CSV: Human-readable, portable
- Parquet: Efficient, compressed, good for large datasets
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Optional, Iterator
import json
import sqlite3
from contextlib import contextmanager

from trading.core.models import Tick, OHLCV
from trading.core.enums import DataResolution


class HistoryStore(ABC):
    """
    Abstract base class for historical data storage.

    Provides interface for storing and retrieving market data.
    """

    @abstractmethod
    def store_tick(self, tick: Tick) -> None:
        """Store a single tick."""
        pass

    @abstractmethod
    def store_ticks(self, ticks: list[Tick]) -> None:
        """Store multiple ticks in batch."""
        pass

    @abstractmethod
    def store_bar(self, bar: OHLCV, resolution: DataResolution) -> None:
        """Store a single bar."""
        pass

    @abstractmethod
    def store_bars(self, bars: list[OHLCV], resolution: DataResolution) -> None:
        """Store multiple bars in batch."""
        pass

    @abstractmethod
    def get_ticks(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[Tick]:
        """Retrieve ticks for a symbol in a time range."""
        pass

    @abstractmethod
    def get_bars(
        self,
        symbol: str,
        resolution: DataResolution,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[OHLCV]:
        """Retrieve bars for a symbol and resolution in a time range."""
        pass

    @abstractmethod
    def get_latest_tick(self, symbol: str) -> Optional[Tick]:
        """Get the most recent tick for a symbol."""
        pass

    @abstractmethod
    def get_latest_bar(
        self,
        symbol: str,
        resolution: DataResolution
    ) -> Optional[OHLCV]:
        """Get the most recent bar for a symbol and resolution."""
        pass

    @abstractmethod
    def get_symbols(self) -> list[str]:
        """Get list of all stored symbols."""
        pass

    @abstractmethod
    def get_time_range(
        self,
        symbol: str,
        resolution: Optional[DataResolution] = None
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        """Get the time range of stored data for a symbol."""
        pass

    @abstractmethod
    def delete_old_data(self, before: datetime) -> int:
        """Delete data older than a timestamp. Returns count deleted."""
        pass

    def close(self) -> None:
        """Close the store and release resources."""
        pass


class SQLiteHistoryStore(HistoryStore):
    """
    SQLite-based historical data storage.

    Efficient for development and moderate data volumes.
    Stores ticks and bars in separate tables.

    Usage:
        store = SQLiteHistoryStore("./data/history.db")

        # Store data
        store.store_tick(tick)
        store.store_bars(bars, DataResolution.MINUTE_1)

        # Retrieve data
        bars = store.get_bars(
            "BTCUSDT",
            DataResolution.MINUTE_1,
            start=datetime(2024, 1, 1),
        )

        store.close()
    """

    def __init__(self, db_path: str, create_if_missing: bool = True):
        self.db_path = Path(db_path)

        if create_if_missing:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()

        conn.executescript("""
            -- Ticks table
            CREATE TABLE IF NOT EXISTS ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                bid REAL NOT NULL,
                ask REAL NOT NULL,
                bid_size REAL,
                ask_size REAL,
                last_price REAL,
                last_size REAL,
                volume REAL,
                UNIQUE(symbol, timestamp)
            );

            CREATE INDEX IF NOT EXISTS idx_ticks_symbol_time
                ON ticks(symbol, timestamp);

            -- Bars table
            CREATE TABLE IF NOT EXISTS bars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                resolution TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                trades INTEGER,
                vwap REAL,
                UNIQUE(symbol, resolution, timestamp)
            );

            CREATE INDEX IF NOT EXISTS idx_bars_symbol_res_time
                ON bars(symbol, resolution, timestamp);
        """)

        conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            self._conn.row_factory = sqlite3.Row

        yield self._conn

    def store_tick(self, tick: Tick) -> None:
        """Store a single tick."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO ticks
                (symbol, timestamp, bid, ask, bid_size, ask_size, last_price, last_size, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tick.symbol,
                tick.timestamp.isoformat(),
                float(tick.bid),
                float(tick.ask),
                float(tick.bid_size) if tick.bid_size else None,
                float(tick.ask_size) if tick.ask_size else None,
                float(tick.last_price) if tick.last_price else None,
                float(tick.last_size) if tick.last_size else None,
                float(tick.volume) if tick.volume else None,
            ))
            conn.commit()

    def store_ticks(self, ticks: list[Tick]) -> None:
        """Store multiple ticks in batch."""
        if not ticks:
            return

        with self._get_connection() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO ticks
                (symbol, timestamp, bid, ask, bid_size, ask_size, last_price, last_size, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    tick.symbol,
                    tick.timestamp.isoformat(),
                    float(tick.bid),
                    float(tick.ask),
                    float(tick.bid_size) if tick.bid_size else None,
                    float(tick.ask_size) if tick.ask_size else None,
                    float(tick.last_price) if tick.last_price else None,
                    float(tick.last_size) if tick.last_size else None,
                    float(tick.volume) if tick.volume else None,
                )
                for tick in ticks
            ])
            conn.commit()

    def store_bar(self, bar: OHLCV, resolution: DataResolution) -> None:
        """Store a single bar."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO bars
                (symbol, resolution, timestamp, open, high, low, close, volume, trades, vwap)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                bar.symbol,
                resolution.name,
                bar.timestamp.isoformat(),
                float(bar.open),
                float(bar.high),
                float(bar.low),
                float(bar.close),
                float(bar.volume),
                bar.trades,
                float(bar.vwap) if bar.vwap else None,
            ))
            conn.commit()

    def store_bars(self, bars: list[OHLCV], resolution: DataResolution) -> None:
        """Store multiple bars in batch."""
        if not bars:
            return

        with self._get_connection() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO bars
                (symbol, resolution, timestamp, open, high, low, close, volume, trades, vwap)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    bar.symbol,
                    resolution.name,
                    bar.timestamp.isoformat(),
                    float(bar.open),
                    float(bar.high),
                    float(bar.low),
                    float(bar.close),
                    float(bar.volume),
                    bar.trades,
                    float(bar.vwap) if bar.vwap else None,
                )
                for bar in bars
            ])
            conn.commit()

    def get_ticks(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[Tick]:
        """Retrieve ticks for a symbol."""
        end = end or datetime.utcnow()

        query = """
            SELECT * FROM ticks
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
        """
        params = [symbol, start.isoformat(), end.isoformat()]

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_tick(row) for row in rows]

    def get_bars(
        self,
        symbol: str,
        resolution: DataResolution,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[OHLCV]:
        """Retrieve bars for a symbol and resolution."""
        end = end or datetime.utcnow()

        query = """
            SELECT * FROM bars
            WHERE symbol = ? AND resolution = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
        """
        params = [symbol, resolution.name, start.isoformat(), end.isoformat()]

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_bar(row) for row in rows]

    def get_latest_tick(self, symbol: str) -> Optional[Tick]:
        """Get most recent tick for symbol."""
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM ticks
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (symbol,)).fetchone()

        return self._row_to_tick(row) if row else None

    def get_latest_bar(
        self,
        symbol: str,
        resolution: DataResolution
    ) -> Optional[OHLCV]:
        """Get most recent bar for symbol and resolution."""
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM bars
                WHERE symbol = ? AND resolution = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (symbol, resolution.name)).fetchone()

        return self._row_to_bar(row) if row else None

    def get_symbols(self) -> list[str]:
        """Get list of all stored symbols."""
        with self._get_connection() as conn:
            # Get symbols from both tables
            tick_symbols = conn.execute(
                "SELECT DISTINCT symbol FROM ticks"
            ).fetchall()
            bar_symbols = conn.execute(
                "SELECT DISTINCT symbol FROM bars"
            ).fetchall()

        symbols = set()
        for row in tick_symbols:
            symbols.add(row[0])
        for row in bar_symbols:
            symbols.add(row[0])

        return sorted(symbols)

    def get_time_range(
        self,
        symbol: str,
        resolution: Optional[DataResolution] = None
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        """Get time range of stored data."""
        with self._get_connection() as conn:
            if resolution:
                row = conn.execute("""
                    SELECT MIN(timestamp), MAX(timestamp) FROM bars
                    WHERE symbol = ? AND resolution = ?
                """, (symbol, resolution.name)).fetchone()
            else:
                # Check ticks first
                row = conn.execute("""
                    SELECT MIN(timestamp), MAX(timestamp) FROM ticks
                    WHERE symbol = ?
                """, (symbol,)).fetchone()

                if not row[0]:
                    row = conn.execute("""
                        SELECT MIN(timestamp), MAX(timestamp) FROM bars
                        WHERE symbol = ?
                    """, (symbol,)).fetchone()

        if row and row[0]:
            return (
                datetime.fromisoformat(row[0]),
                datetime.fromisoformat(row[1]),
            )

        return None, None

    def delete_old_data(self, before: datetime) -> int:
        """Delete data older than timestamp."""
        before_str = before.isoformat()

        with self._get_connection() as conn:
            tick_result = conn.execute(
                "DELETE FROM ticks WHERE timestamp < ?", (before_str,)
            )
            bar_result = conn.execute(
                "DELETE FROM bars WHERE timestamp < ?", (before_str,)
            )
            conn.commit()

        return tick_result.rowcount + bar_result.rowcount

    def _row_to_tick(self, row: sqlite3.Row) -> Tick:
        """Convert database row to Tick."""
        return Tick(
            symbol=row["symbol"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            bid=Decimal(str(row["bid"])),
            ask=Decimal(str(row["ask"])),
            bid_size=Decimal(str(row["bid_size"])) if row["bid_size"] else Decimal("0"),
            ask_size=Decimal(str(row["ask_size"])) if row["ask_size"] else Decimal("0"),
            last_price=Decimal(str(row["last_price"])) if row["last_price"] else None,
            last_size=Decimal(str(row["last_size"])) if row["last_size"] else None,
            volume=Decimal(str(row["volume"])) if row["volume"] else None,
        )

    def _row_to_bar(self, row: sqlite3.Row) -> OHLCV:
        """Convert database row to OHLCV."""
        return OHLCV(
            symbol=row["symbol"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            open=Decimal(str(row["open"])),
            high=Decimal(str(row["high"])),
            low=Decimal(str(row["low"])),
            close=Decimal(str(row["close"])),
            volume=Decimal(str(row["volume"])),
            trades=row["trades"] or 0,
            vwap=Decimal(str(row["vwap"])) if row["vwap"] else None,
        )

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


class CSVHistoryStore(HistoryStore):
    """
    CSV-based historical data storage.

    Human-readable and portable format.
    Stores each symbol/resolution combination in separate files.

    Usage:
        store = CSVHistoryStore("./data/csv")
        store.store_bars(bars, DataResolution.MINUTE_1)

        # Files created:
        # ./data/csv/bars/BTCUSDT_MINUTE_1.csv
        # ./data/csv/ticks/BTCUSDT.csv
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.ticks_dir = self.data_dir / "ticks"
        self.bars_dir = self.data_dir / "bars"

        self.ticks_dir.mkdir(parents=True, exist_ok=True)
        self.bars_dir.mkdir(parents=True, exist_ok=True)

    def _tick_file(self, symbol: str) -> Path:
        return self.ticks_dir / f"{symbol}.csv"

    def _bar_file(self, symbol: str, resolution: DataResolution) -> Path:
        return self.bars_dir / f"{symbol}_{resolution.name}.csv"

    def store_tick(self, tick: Tick) -> None:
        """Store a single tick."""
        self.store_ticks([tick])

    def store_ticks(self, ticks: list[Tick]) -> None:
        """Store multiple ticks."""
        if not ticks:
            return

        # Group by symbol
        by_symbol: dict[str, list[Tick]] = {}
        for tick in ticks:
            if tick.symbol not in by_symbol:
                by_symbol[tick.symbol] = []
            by_symbol[tick.symbol].append(tick)

        for symbol, symbol_ticks in by_symbol.items():
            file_path = self._tick_file(symbol)
            write_header = not file_path.exists()

            with open(file_path, "a") as f:
                if write_header:
                    f.write("timestamp,bid,ask,bid_size,ask_size,last_price,last_size,volume\n")

                for tick in symbol_ticks:
                    f.write(f"{tick.timestamp.isoformat()},"
                            f"{tick.bid},{tick.ask},"
                            f"{tick.bid_size or ''},"
                            f"{tick.ask_size or ''},"
                            f"{tick.last_price or ''},"
                            f"{tick.last_size or ''},"
                            f"{tick.volume or ''}\n")

    def store_bar(self, bar: OHLCV, resolution: DataResolution) -> None:
        """Store a single bar."""
        self.store_bars([bar], resolution)

    def store_bars(self, bars: list[OHLCV], resolution: DataResolution) -> None:
        """Store multiple bars."""
        if not bars:
            return

        # Group by symbol
        by_symbol: dict[str, list[OHLCV]] = {}
        for bar in bars:
            if bar.symbol not in by_symbol:
                by_symbol[bar.symbol] = []
            by_symbol[bar.symbol].append(bar)

        for symbol, symbol_bars in by_symbol.items():
            file_path = self._bar_file(symbol, resolution)
            write_header = not file_path.exists()

            with open(file_path, "a") as f:
                if write_header:
                    f.write("timestamp,open,high,low,close,volume,trades,vwap\n")

                for bar in symbol_bars:
                    f.write(f"{bar.timestamp.isoformat()},"
                            f"{bar.open},{bar.high},{bar.low},{bar.close},"
                            f"{bar.volume},{bar.trades or ''},"
                            f"{bar.vwap or ''}\n")

    def get_ticks(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[Tick]:
        """Retrieve ticks from CSV."""
        file_path = self._tick_file(symbol)
        if not file_path.exists():
            return []

        end = end or datetime.utcnow()
        ticks = []

        with open(file_path, "r") as f:
            header = f.readline()  # Skip header
            for line in f:
                parts = line.strip().split(",")
                timestamp = datetime.fromisoformat(parts[0])

                if timestamp < start:
                    continue
                if timestamp > end:
                    break

                tick = Tick(
                    symbol=symbol,
                    timestamp=timestamp,
                    bid=Decimal(parts[1]),
                    ask=Decimal(parts[2]),
                    bid_size=Decimal(parts[3]) if parts[3] else Decimal("0"),
                    ask_size=Decimal(parts[4]) if parts[4] else Decimal("0"),
                    last_price=Decimal(parts[5]) if parts[5] else None,
                    last_size=Decimal(parts[6]) if parts[6] else None,
                    volume=Decimal(parts[7]) if parts[7] else None,
                )
                ticks.append(tick)

                if limit and len(ticks) >= limit:
                    break

        return ticks

    def get_bars(
        self,
        symbol: str,
        resolution: DataResolution,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[OHLCV]:
        """Retrieve bars from CSV."""
        file_path = self._bar_file(symbol, resolution)
        if not file_path.exists():
            return []

        end = end or datetime.utcnow()
        bars = []

        with open(file_path, "r") as f:
            header = f.readline()  # Skip header
            for line in f:
                parts = line.strip().split(",")
                timestamp = datetime.fromisoformat(parts[0])

                if timestamp < start:
                    continue
                if timestamp > end:
                    break

                bar = OHLCV(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=Decimal(parts[1]),
                    high=Decimal(parts[2]),
                    low=Decimal(parts[3]),
                    close=Decimal(parts[4]),
                    volume=Decimal(parts[5]),
                    trades=int(parts[6]) if parts[6] else 0,
                    vwap=Decimal(parts[7]) if parts[7] else None,
                )
                bars.append(bar)

                if limit and len(bars) >= limit:
                    break

        return bars

    def get_latest_tick(self, symbol: str) -> Optional[Tick]:
        """Get most recent tick."""
        file_path = self._tick_file(symbol)
        if not file_path.exists():
            return None

        # Read last line (inefficient but simple)
        with open(file_path, "r") as f:
            lines = f.readlines()
            if len(lines) <= 1:
                return None

            parts = lines[-1].strip().split(",")
            return Tick(
                symbol=symbol,
                timestamp=datetime.fromisoformat(parts[0]),
                bid=Decimal(parts[1]),
                ask=Decimal(parts[2]),
                bid_size=Decimal(parts[3]) if parts[3] else Decimal("0"),
                ask_size=Decimal(parts[4]) if parts[4] else Decimal("0"),
                last_price=Decimal(parts[5]) if parts[5] else None,
                last_size=Decimal(parts[6]) if parts[6] else None,
                volume=Decimal(parts[7]) if parts[7] else None,
            )

    def get_latest_bar(
        self,
        symbol: str,
        resolution: DataResolution
    ) -> Optional[OHLCV]:
        """Get most recent bar."""
        file_path = self._bar_file(symbol, resolution)
        if not file_path.exists():
            return None

        with open(file_path, "r") as f:
            lines = f.readlines()
            if len(lines) <= 1:
                return None

            parts = lines[-1].strip().split(",")
            return OHLCV(
                symbol=symbol,
                timestamp=datetime.fromisoformat(parts[0]),
                open=Decimal(parts[1]),
                high=Decimal(parts[2]),
                low=Decimal(parts[3]),
                close=Decimal(parts[4]),
                volume=Decimal(parts[5]),
                trades=int(parts[6]) if parts[6] else 0,
                vwap=Decimal(parts[7]) if parts[7] else None,
            )

    def get_symbols(self) -> list[str]:
        """Get list of all stored symbols."""
        symbols = set()

        for file in self.ticks_dir.glob("*.csv"):
            symbols.add(file.stem)

        for file in self.bars_dir.glob("*.csv"):
            # Remove resolution suffix
            symbol = file.stem.rsplit("_", 1)[0]
            symbols.add(symbol)

        return sorted(symbols)

    def get_time_range(
        self,
        symbol: str,
        resolution: Optional[DataResolution] = None
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        """Get time range of stored data."""
        if resolution:
            file_path = self._bar_file(symbol, resolution)
        else:
            file_path = self._tick_file(symbol)

        if not file_path.exists():
            return None, None

        with open(file_path, "r") as f:
            lines = f.readlines()
            if len(lines) <= 1:
                return None, None

            first_parts = lines[1].strip().split(",")
            last_parts = lines[-1].strip().split(",")

            return (
                datetime.fromisoformat(first_parts[0]),
                datetime.fromisoformat(last_parts[0]),
            )

    def delete_old_data(self, before: datetime) -> int:
        """Delete old data by rewriting files."""
        count = 0

        # Process tick files
        for file in self.ticks_dir.glob("*.csv"):
            symbol = file.stem
            ticks = self.get_ticks(symbol, before, datetime.max)
            if ticks:
                file.unlink()
                self.store_ticks(ticks)
                count += 1

        # Process bar files
        for file in self.bars_dir.glob("*.csv"):
            parts = file.stem.rsplit("_", 1)
            symbol = parts[0]
            resolution = DataResolution[parts[1]]
            bars = self.get_bars(symbol, resolution, before, datetime.max)
            if bars:
                file.unlink()
                self.store_bars(bars, resolution)
                count += 1

        return count


class ParquetHistoryStore(HistoryStore):
    """
    Parquet-based historical data storage.

    Efficient, compressed columnar format for large datasets.
    Requires pyarrow or fastparquet library.

    Usage:
        store = ParquetHistoryStore("./data/parquet")
        store.store_bars(bars, DataResolution.MINUTE_1)
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        try:
            import pyarrow.parquet as pq
            import pyarrow as pa
            self._pq = pq
            self._pa = pa
        except ImportError:
            raise ImportError("pyarrow required: pip install pyarrow")

    def _tick_file(self, symbol: str) -> Path:
        return self.data_dir / "ticks" / f"{symbol}.parquet"

    def _bar_file(self, symbol: str, resolution: DataResolution) -> Path:
        return self.data_dir / "bars" / f"{symbol}_{resolution.name}.parquet"

    def store_tick(self, tick: Tick) -> None:
        self.store_ticks([tick])

    def store_ticks(self, ticks: list[Tick]) -> None:
        if not ticks:
            return

        # Group by symbol
        by_symbol: dict[str, list[dict]] = {}
        for tick in ticks:
            if tick.symbol not in by_symbol:
                by_symbol[tick.symbol] = []
            by_symbol[tick.symbol].append({
                "timestamp": tick.timestamp,
                "bid": float(tick.bid),
                "ask": float(tick.ask),
                "bid_size": float(tick.bid_size) if tick.bid_size else None,
                "ask_size": float(tick.ask_size) if tick.ask_size else None,
                "last_price": float(tick.last_price) if tick.last_price else None,
                "last_size": float(tick.last_size) if tick.last_size else None,
                "volume": float(tick.volume) if tick.volume else None,
            })

        for symbol, rows in by_symbol.items():
            file_path = self._tick_file(symbol)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            table = self._pa.Table.from_pydict({
                "timestamp": [r["timestamp"] for r in rows],
                "bid": [r["bid"] for r in rows],
                "ask": [r["ask"] for r in rows],
                "bid_size": [r["bid_size"] for r in rows],
                "ask_size": [r["ask_size"] for r in rows],
                "last_price": [r["last_price"] for r in rows],
                "last_size": [r["last_size"] for r in rows],
                "volume": [r["volume"] for r in rows],
            })

            if file_path.exists():
                existing = self._pq.read_table(file_path)
                table = self._pa.concat_tables([existing, table])

            self._pq.write_table(table, file_path, compression="snappy")

    def store_bar(self, bar: OHLCV, resolution: DataResolution) -> None:
        self.store_bars([bar], resolution)

    def store_bars(self, bars: list[OHLCV], resolution: DataResolution) -> None:
        if not bars:
            return

        # Group by symbol
        by_symbol: dict[str, list[dict]] = {}
        for bar in bars:
            if bar.symbol not in by_symbol:
                by_symbol[bar.symbol] = []
            by_symbol[bar.symbol].append({
                "timestamp": bar.timestamp,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
                "trades": bar.trades,
                "vwap": float(bar.vwap) if bar.vwap else None,
            })

        for symbol, rows in by_symbol.items():
            file_path = self._bar_file(symbol, resolution)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            table = self._pa.Table.from_pydict({
                "timestamp": [r["timestamp"] for r in rows],
                "open": [r["open"] for r in rows],
                "high": [r["high"] for r in rows],
                "low": [r["low"] for r in rows],
                "close": [r["close"] for r in rows],
                "volume": [r["volume"] for r in rows],
                "trades": [r["trades"] for r in rows],
                "vwap": [r["vwap"] for r in rows],
            })

            if file_path.exists():
                existing = self._pq.read_table(file_path)
                table = self._pa.concat_tables([existing, table])

            self._pq.write_table(table, file_path, compression="snappy")

    def get_ticks(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[Tick]:
        file_path = self._tick_file(symbol)
        if not file_path.exists():
            return []

        end = end or datetime.utcnow()

        table = self._pq.read_table(
            file_path,
            filters=[
                ("timestamp", ">=", start),
                ("timestamp", "<=", end),
            ],
        )

        ticks = []
        for i in range(len(table)):
            tick = Tick(
                symbol=symbol,
                timestamp=table["timestamp"][i].as_py(),
                bid=Decimal(str(table["bid"][i].as_py())),
                ask=Decimal(str(table["ask"][i].as_py())),
                bid_size=Decimal(str(table["bid_size"][i].as_py())) if table["bid_size"][i].as_py() else Decimal("0"),
                ask_size=Decimal(str(table["ask_size"][i].as_py())) if table["ask_size"][i].as_py() else Decimal("0"),
                last_price=Decimal(str(table["last_price"][i].as_py())) if table["last_price"][i].as_py() else None,
                last_size=Decimal(str(table["last_size"][i].as_py())) if table["last_size"][i].as_py() else None,
                volume=Decimal(str(table["volume"][i].as_py())) if table["volume"][i].as_py() else None,
            )
            ticks.append(tick)

            if limit and len(ticks) >= limit:
                break

        return ticks

    def get_bars(
        self,
        symbol: str,
        resolution: DataResolution,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[OHLCV]:
        file_path = self._bar_file(symbol, resolution)
        if not file_path.exists():
            return []

        end = end or datetime.utcnow()

        table = self._pq.read_table(
            file_path,
            filters=[
                ("timestamp", ">=", start),
                ("timestamp", "<=", end),
            ],
        )

        bars = []
        for i in range(len(table)):
            bar = OHLCV(
                symbol=symbol,
                timestamp=table["timestamp"][i].as_py(),
                open=Decimal(str(table["open"][i].as_py())),
                high=Decimal(str(table["high"][i].as_py())),
                low=Decimal(str(table["low"][i].as_py())),
                close=Decimal(str(table["close"][i].as_py())),
                volume=Decimal(str(table["volume"][i].as_py())),
                trades=table["trades"][i].as_py() or 0,
                vwap=Decimal(str(table["vwap"][i].as_py())) if table["vwap"][i].as_py() else None,
            )
            bars.append(bar)

            if limit and len(bars) >= limit:
                break

        return bars

    def get_latest_tick(self, symbol: str) -> Optional[Tick]:
        ticks = self.get_ticks(symbol, datetime.min, limit=1)
        return ticks[-1] if ticks else None

    def get_latest_bar(self, symbol: str, resolution: DataResolution) -> Optional[OHLCV]:
        bars = self.get_bars(symbol, resolution, datetime.min, limit=1)
        return bars[-1] if bars else None

    def get_symbols(self) -> list[str]:
        symbols = set()

        ticks_dir = self.data_dir / "ticks"
        if ticks_dir.exists():
            for file in ticks_dir.glob("*.parquet"):
                symbols.add(file.stem)

        bars_dir = self.data_dir / "bars"
        if bars_dir.exists():
            for file in bars_dir.glob("*.parquet"):
                symbol = file.stem.rsplit("_", 1)[0]
                symbols.add(symbol)

        return sorted(symbols)

    def get_time_range(
        self,
        symbol: str,
        resolution: Optional[DataResolution] = None
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        if resolution:
            file_path = self._bar_file(symbol, resolution)
        else:
            file_path = self._tick_file(symbol)

        if not file_path.exists():
            return None, None

        table = self._pq.read_table(file_path, columns=["timestamp"])
        if len(table) == 0:
            return None, None

        timestamps = table["timestamp"].to_pylist()
        return min(timestamps), max(timestamps)

    def delete_old_data(self, before: datetime) -> int:
        # Parquet doesn't support in-place deletion
        # Need to rewrite files
        count = 0

        for symbol in self.get_symbols():
            # Ticks
            tick_file = self._tick_file(symbol)
            if tick_file.exists():
                ticks = self.get_ticks(symbol, before, datetime.max)
                if ticks:
                    tick_file.unlink()
                    self.store_ticks(ticks)
                    count += 1

            # Bars for all resolutions
            bars_dir = self.data_dir / "bars"
            if bars_dir.exists():
                for file in bars_dir.glob(f"{symbol}_*.parquet"):
                    parts = file.stem.rsplit("_", 1)
                    resolution = DataResolution[parts[1]]
                    bars = self.get_bars(symbol, resolution, before, datetime.max)
                    if bars:
                        file.unlink()
                        self.store_bars(bars, resolution)
                        count += 1

        return count
