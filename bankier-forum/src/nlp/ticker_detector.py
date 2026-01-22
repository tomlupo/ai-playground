"""WSE (Warsaw Stock Exchange) ticker detector for Polish financial text."""

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class WSETickerDetector:
    """
    Detector for WSE ticker symbols in Polish financial text.

    Supports both explicit ticker mentions (e.g., CDR, PKO) and
    company name aliases (e.g., "cd projekt", "allegro").
    """

    # Major WSE tickers and their company names
    DEFAULT_TICKERS = {
        # WIG20 components
        "ALE": "Allegro",
        "ACP": "Asseco Poland",
        "CDR": "CD Projekt",
        "CPS": "Cyfrowy Polsat",
        "DNP": "Dino Polska",
        "JSW": "JSW",
        "KGH": "KGHM Polska Miedź",
        "KRU": "Kruk",
        "KTY": "Kęty",
        "LPP": "LPP",
        "MBK": "mBank",
        "OPL": "Orange Polska",
        "PEO": "Bank Pekao",
        "PGE": "PGE",
        "PGN": "PGNiG",
        "PKN": "PKN Orlen",
        "PKO": "PKO Bank Polski",
        "PZU": "PZU",
        "SPL": "Santander Bank Polska",
        # Other popular tickers
        "11B": "11 bit studios",
        "AMC": "Amica",
        "ASB": "Asseco Business Solutions",
        "BDX": "Budimex",
        "BFT": "Benefit Systems",
        "BHW": "Bank Handlowy",
        "CAR": "Inter Cars",
        "CCC": "CCC",
        "CIG": "Cigames",
        "COM": "Comarch",
        "DAT": "Datawalk",
        "DOM": "Dom Development",
        "EAT": "AmRest",
        "ENA": "Enea",
        "EUR": "Eurocash",
        "GPW": "GPW",
        "GTC": "GTC",
        "HUG": "Huuuge",
        "ING": "ING Bank Śląski",
        "KER": "Kernel",
        "KRK": "Krka",
        "LBW": "Lubawa",
        "LWB": "Bogdanka",
        "MIL": "Bank Millennium",
        "MRC": "Mercator Medical",
        "NEU": "Neuca",
        "PCO": "Pepco",
        "PLW": "Playway",
        "TEN": "Ten Square Games",
        "TPE": "Tauron",
        "TXT": "Text",
        "VRG": "VRG",
        "WPL": "Wirtualna Polska",
        "XTB": "XTB",
        "ZAB": "Żabka",
    }

    # Common aliases for company names
    DEFAULT_ALIASES = {
        # CD Projekt variations
        "cd projekt": "CDR",
        "cdprojekt": "CDR",
        "cd project": "CDR",
        "cdp": "CDR",
        "redzi": "CDR",  # Polish nickname
        # Banks
        "allegro": "ALE",
        "orlen": "PKN",
        "pko": "PKO",
        "pko bp": "PKO",
        "pekao": "PEO",
        "santander": "SPL",
        "millennium": "MIL",
        "mbank": "MBK",
        "ing": "ING",
        # Mining/Energy
        "kghm": "KGH",
        "polska miedź": "KGH",
        "jsw": "JSW",
        "bogdanka": "LWB",
        "tauron": "TPE",
        "pge": "PGE",
        "pgnig": "PGN",
        # Retail
        "dino": "DNP",
        "dino polska": "DNP",
        "pepco": "PCO",
        "zabka": "ZAB",
        "żabka": "ZAB",
        "ccc": "CCC",
        "lpp": "LPP",
        "reserved": "LPP",
        "cyfrowy polsat": "CPS",
        "polsat": "CPS",
        # Gaming
        "11bit": "11B",
        "11 bit": "11B",
        "playway": "PLW",
        "ten square": "TEN",
        "huuuge": "HUG",
        "cigames": "CIG",
        # Insurance
        "pzu": "PZU",
        # Telecom
        "orange": "OPL",
        "orange polska": "OPL",
        # Others
        "amrest": "EAT",
        "kruk": "KRU",
        "wirtualna polska": "WPL",
        "wp": "WPL",
        "xtb": "XTB",
        "asseco": "ACP",
        "comarch": "COM",
        "budimex": "BDX",
    }

    def __init__(
        self,
        tickers_file: Optional[Path] = None,
        aliases_file: Optional[Path] = None,
    ):
        """
        Initialize ticker detector.

        Args:
            tickers_file: Optional JSON file with ticker -> company mapping
            aliases_file: Optional JSON file with alias -> ticker mapping
        """
        # Load default tickers
        self.ticker_to_company = dict(self.DEFAULT_TICKERS)
        self.aliases = dict(self.DEFAULT_ALIASES)

        # Load from files if provided
        if tickers_file and tickers_file.exists():
            with open(tickers_file) as f:
                self.ticker_to_company.update(json.load(f))

        if aliases_file and aliases_file.exists():
            with open(aliases_file) as f:
                self.aliases.update(json.load(f))

        # Build reverse mapping for company name to ticker
        self.company_to_ticker = {v.lower(): k for k, v in self.ticker_to_company.items()}

        # Regex for explicit tickers (2-5 uppercase letters/numbers)
        self.ticker_pattern = re.compile(r"\b([A-Z0-9]{2,5})\b")

        # Compile alias patterns for efficient matching
        self._alias_patterns = [
            (re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE), ticker)
            for alias, ticker in sorted(self.aliases.items(), key=lambda x: -len(x[0]))
        ]

    def detect(self, text: str) -> list[tuple[str, str, str]]:
        """
        Detect ticker mentions in text.

        Args:
            text: Text to analyze

        Returns:
            List of (ticker, company_name, mention_type) tuples
            - mention_type is 'explicit' for ticker symbols, 'inferred' for company names
        """
        if not text:
            return []

        mentions: list[tuple[str, str, str]] = []
        seen_tickers: set[str] = set()

        # Check aliases first (longer matches preferred)
        for pattern, ticker in self._alias_patterns:
            if pattern.search(text):
                if ticker not in seen_tickers:
                    seen_tickers.add(ticker)
                    mentions.append((
                        ticker,
                        self.ticker_to_company.get(ticker, ""),
                        "inferred",
                    ))

        # Check explicit tickers
        for match in self.ticker_pattern.finditer(text):
            ticker = match.group(1)
            if ticker in self.ticker_to_company and ticker not in seen_tickers:
                seen_tickers.add(ticker)
                mentions.append((
                    ticker,
                    self.ticker_to_company[ticker],
                    "explicit",
                ))

        return mentions

    def get_context(self, text: str, ticker: str, window: int = 50) -> str:
        """
        Extract context around a ticker mention.

        Args:
            text: Full text
            ticker: Ticker to find
            window: Characters before and after to include

        Returns:
            Context snippet around the mention
        """
        if not text:
            return ""

        text_lower = text.lower()

        # First try to find the ticker symbol
        pos = text.find(ticker)
        if pos == -1:
            pos = text_lower.find(ticker.lower())

        # Then try aliases
        if pos == -1:
            for alias, t in self.aliases.items():
                if t == ticker and alias in text_lower:
                    pos = text_lower.find(alias)
                    break

        # Try company name
        if pos == -1:
            company = self.ticker_to_company.get(ticker, "").lower()
            if company and company in text_lower:
                pos = text_lower.find(company)

        if pos == -1:
            # No match found, return beginning of text
            return text[: window * 2]

        start = max(0, pos - window)
        end = min(len(text), pos + len(ticker) + window)
        return text[start:end]

    def add_ticker(self, ticker: str, company_name: str, aliases: Optional[list[str]] = None) -> None:
        """
        Add a new ticker to the detector.

        Args:
            ticker: Ticker symbol
            company_name: Company name
            aliases: Optional list of aliases
        """
        self.ticker_to_company[ticker] = company_name
        self.company_to_ticker[company_name.lower()] = ticker

        if aliases:
            for alias in aliases:
                self.aliases[alias.lower()] = ticker
                self._alias_patterns.append((
                    re.compile(rf"\b{re.escape(alias.lower())}\b", re.IGNORECASE),
                    ticker,
                ))

    def get_all_tickers(self) -> list[str]:
        """Get list of all known tickers."""
        return list(self.ticker_to_company.keys())
