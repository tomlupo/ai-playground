# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.27",
#     "beautifulsoup4>=4.12",
#     "rich>=13.0",
#     "pandas>=2.0",
# ]
# ///
"""
Bankier.pl Forum Rotation Signal Scanner
=========================================
Scrapes Bankier.pl stock forum threads, extracts ticker mentions,
themes, and sentiment to identify capital rotation signals before
mainstream analysts and media catch on.

Strategy:
- Monitor Bankier.pl "Giełda" forum (most active WSE discussion board)
- Extract ticker mentions from thread titles
- Classify threads into macro themes (sectors, catalysts)
- Score rotation signals: mention velocity, sentiment skew, cross-theme momentum
- Surface top 10 themes + high-potential tickers

Usage:
    uv run tools/forum-rotation-signals/main.py
    uv run tools/forum-rotation-signals/main.py --pages 15
    uv run tools/forum-rotation-signals/main.py --output outputs/rotation_signals.md
"""

from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import pandas as pd

console = Console()

# ---------------------------------------------------------------------------
# WSE Ticker & Sector Master
# ---------------------------------------------------------------------------

# Comprehensive Polish WSE ticker -> sector mapping
TICKER_SECTOR: dict[str, str] = {
    # Mining & Metals
    "KGHM": "Mining & Metals", "JSW": "Mining & Metals", "GREENX": "Mining & Metals",
    "COGNOR": "Mining & Metals", "MENNICA": "Mining & Metals", "BUMECH": "Mining & Metals",
    "FAMUR": "Mining & Metals", "BOGDANKA": "Mining & Metals",
    # Energy
    "ORLEN": "Energy", "ENERGA": "Energy", "PGE": "Energy", "TAURON": "Energy",
    "COLUMBUS": "Energy", "POLENERGIA": "Energy", "ZEPAK": "Energy",
    "ML SYSTEM": "Energy", "PHOTON": "Energy",
    # Gaming & Tech
    "CDPROJEKT": "Gaming & Tech", "11BIT": "Gaming & Tech", "TSGAMES": "Gaming & Tech",
    "CREEPYJAR": "Gaming & Tech", "PLAYWAY": "Gaming & Tech", "PCF": "Gaming & Tech",
    "TECHLAND": "Gaming & Tech", "IRONWOLF": "Gaming & Tech", "OML": "Gaming & Tech",
    "BLOOBER": "Gaming & Tech", "TEQUILA": "Gaming & Tech",
    # Fintech & Brokerage
    "XTB": "Fintech & Brokerage", "CAPITALPARTNERS": "Fintech & Brokerage",
    # Banks
    "PKOBP": "Banks", "PEKAO": "Banks", "MBANK": "Banks", "MILLENNIUM": "Banks",
    "INGBSK": "Banks", "BNP": "Banks", "ALIOR": "Banks", "GETIN": "Banks",
    "SANTANDER": "Banks", "HANDLOWY": "Banks",
    # Retail & E-commerce
    "ALLEGRO": "Retail & E-commerce", "CCC": "Retail & E-commerce",
    "PEPCO": "Retail & E-commerce", "LPP": "Retail & E-commerce",
    "DINO": "Retail & E-commerce", "KOMPUTRONIK": "Retail & E-commerce",
    "EUROCASH": "Retail & E-commerce",
    # Healthcare & Biotech
    "MEDINICE": "Healthcare & Biotech", "MOLECURE": "Healthcare & Biotech",
    "PURE": "Healthcare & Biotech", "CELON": "Healthcare & Biotech",
    "MABION": "Healthcare & Biotech", "BIOMED": "Healthcare & Biotech",
    "RYVU": "Healthcare & Biotech", "SELVITA": "Healthcare & Biotech",
    "INVENTIONMED": "Healthcare & Biotech", "VINCIGEN": "Healthcare & Biotech",
    "DIAGNOSTYKA": "Healthcare & Biotech",
    # Defense & Industrial
    "LUBAWA": "Defense & Industrial", "PGZ": "Defense & Industrial",
    "DEFENCEH": "Defense & Industrial", "WB ELECTRONICS": "Defense & Industrial",
    "CREOTECH": "Defense & Industrial",
    # Real Estate & Construction
    "DEVELIA": "Real Estate", "ECHO": "Real Estate", "ATAL": "Real Estate",
    "DOMDEV": "Real Estate", "ARCHICOM": "Real Estate", "RONSON": "Real Estate",
    # Transport & Logistics
    "PKPCARGO": "Transport & Logistics", "OTLOGISTICS": "Transport & Logistics",
    "INPOST": "Transport & Logistics",
    # IT & Software
    "ASSECO": "IT & Software", "COMARCH": "IT & Software", "SYGNITY": "IT & Software",
    "WIRTUALNA": "IT & Software", "TEXT": "IT & Software", "LIVECHAT": "IT & Software",
    # Steel & Manufacturing
    "STALEXPORT": "Steel & Manufacturing", "RAFAKO": "Steel & Manufacturing",
    "GRENEVIA": "Steel & Manufacturing",
    # Food & Agriculture
    "MFOOD": "Food & Agriculture", "KERNEL": "Food & Agriculture",
    "ASTARTA": "Food & Agriculture",
    # Telco
    "ORANGE": "Telco", "CYFROWY": "Telco", "PLAY": "Telco",
    # Additional WSE companies
    "DATAWALK": "IT & Software", "NANOGROUP": "Healthcare & Biotech",
    "PYRAMID": "Gaming & Tech", "NEWAG": "Transport & Logistics",
    "ASBIS": "IT & Software", "CIECH": "Steel & Manufacturing",
    "MIRBUD": "Real Estate", "BENEFIT": "Retail & E-commerce",
    "VOXEL": "Healthcare & Biotech", "INTERCARS": "Retail & E-commerce",
    "AMREST": "Retail & E-commerce", "KETY": "Steel & Manufacturing",
    "AUTOPARTNER": "Retail & E-commerce", "WIELTON": "Steel & Manufacturing",
    "MERCATOR": "Healthcare & Biotech", "XTPL": "IT & Software",
    "HUUUGE": "Gaming & Tech", "BORYSZEW": "Steel & Manufacturing",
    "BUDIMEX": "Real Estate", "GPW": "Fintech & Brokerage",
    "AMBRA": "Food & Agriculture", "KRUK": "Fintech & Brokerage",
    "PZU": "Banks", "MARVIPOL": "Real Estate",
    "STRONGPOINT": "IT & Software",
}

# Aliases: common forum shorthand -> canonical ticker
TICKER_ALIASES: dict[str, str] = {
    "CD PROJEKT": "CDPROJEKT", "CDP": "CDPROJEKT", "CD": "CDPROJEKT",
    "CDR": "CDPROJEKT", "PROJEKT RED": "CDPROJEKT", "RED": "CDPROJEKT",
    "GENIE": "CDPROJEKT",
    "TEN SQUARE": "TSGAMES", "TSG": "TSGAMES",
    "JASTRZĘBSKA": "JSW", "JASTRZEBSKA": "JSW",
    "PKO": "PKOBP", "PKO BP": "PKOBP",
    "MIEDZ": "KGHM", "MIEDZI": "KGHM", "MIEDŹ": "KGHM",
    "SILVER": "KGHM",  # context-dependent but KGHM is primary silver producer
    "SREBRO": "KGHM",
    "OML": "OML", "ONE MORE LEVEL": "OML",
    "CREEPY JAR": "CREEPYJAR", "CREEPY": "CREEPYJAR",
    "11 BIT": "11BIT", "11BIT STUDIOS": "11BIT",
    "DEFENCE HUB": "DEFENCEH", "DEFENSE HUB": "DEFENCEH",
    "IRON WOLF": "IRONWOLF",
    "PURE BIOLOGICS": "PURE",
    "CELON PHARMA": "CELON",
    "CAPITAL PARTNERS": "CAPITALPARTNERS",
    "PKP CARGO": "PKPCARGO", "PKP": "PKPCARGO",
    "PEPCO GROUP": "PEPCO",
    "GREENX METALS": "GREENX", "GREEN X": "GREENX",
    "COLUMBUS ENERGY": "COLUMBUS",
    "OT LOGISTICS": "OTLOGISTICS",
    "NG2": "CCC",
    "RANK PROGRESS": "ECHO",
    "ENEL-MED": "DIAGNOSTYKA",
    "PIPKU": "PEPCO", "PIPKO": "PEPCO",
    "RAFAKO": "RAFAKO",
    "STALEXPORT": "STALEXPORT",
    # Common Polish forum names for companies
    "ORLEN": "ORLEN", "PKN": "ORLEN", "PKN ORLEN": "ORLEN",
    "ZAMEK": "TSGAMES",  # Ten Square Games forum nickname
    "BLOOBER TEAM": "BLOOBER",
    "MEDINICE": "MEDINICE", "ATRI CLAMP": "MEDINICE",
    "MOLECURE": "MOLECURE",
    "LUBAWA": "LUBAWA",
    "XTB": "XTB",
    "ALLEGRO": "ALLEGRO",
    "INPOST": "INPOST",
    "TEXT": "TEXT", "LIVECHAT": "LIVECHAT",
    "ASSECO": "ASSECO",
    "COMARCH": "COMARCH",
    "KERNEL": "KERNEL",
    "PGE": "PGE",
    "TAURON": "TAURON",
    "ALIOR": "ALIOR",
    "MILLENNIUM": "MILLENNIUM",
    "MABION": "MABION",
    "RYVU": "RYVU",
    "SELVITA": "SELVITA",
    "DINO": "DINO",
    "DOMKA": "DOMDEV", "DOM DEV": "DOMDEV", "DOM DEVELOPMENT": "DOMDEV",
    "ATAL": "ATAL",
    "RONSON": "RONSON",
    "WB ELECTRONICS": "WB ELECTRONICS",
    "CREOTECH": "CREOTECH",
    "JHM DEVELOPMENT": "DEVELIA", "JHM": "DEVELIA",
    "CELON": "CELON",
    "DIAGNOSTYKA": "DIAGNOSTYKA",
    "SYNTHAVERSE": "PURE",
    "GRENEVIA": "GRENEVIA",
    "COGNOR": "COGNOR",
    "KOMPUTRONIK": "KOMPUTRONIK",
    "INVENTIONMED": "INVENTIONMED",
    "VINCIGEN": "VINCIGEN",
    "POLENERGIA": "POLENERGIA",
    "FAMUR": "FAMUR",
    "BOGDANKA": "BOGDANKA", "LW BOGDANKA": "BOGDANKA",
    "EUROCASH": "EUROCASH",
    "INGBSK": "INGBSK", "ING": "INGBSK",
    "PEKAO": "PEKAO",
    "GETIN": "GETIN",
    "HANDLOWY": "HANDLOWY", "BHW": "HANDLOWY",
    "CYFROWY": "CYFROWY", "CYFROWY POLSAT": "CYFROWY",
    "ORANGE": "ORANGE", "ORANGE POLSKA": "ORANGE",
    "PHOTON": "PHOTON", "PHOTON ENERGY": "PHOTON",
    "BUMECH": "BUMECH",
    "MENNICA": "MENNICA", "MENNICA POLSKA": "MENNICA",
    "ECHO": "ECHO", "ECHO INVESTMENT": "ECHO",
    "ARCHICOM": "ECHO",
    # Forum threadGroup company names -> canonical ticker
    "DATAWALK": "DATAWALK", "DATAWALK SA": "DATAWALK",
    "NANOGROUP": "NANOGROUP", "NANOGROUP SA": "NANOGROUP",
    "PYRAMID GAMES": "PYRAMID", "PYRAMID GAMES SA": "PYRAMID",
    "NEWAG": "NEWAG",
    "ASBIS": "ASBIS",
    "CIECH": "CIECH",
    "MIRBUD": "MIRBUD",
    "BENEFIT SYSTEMS": "BENEFIT", "BENEFIT": "BENEFIT",
    "VOXEL": "VOXEL",
    "INTER CARS": "INTERCARS", "INTERCARS": "INTERCARS",
    "AMREST": "AMREST",
    "KETY": "KETY", "GRUPA KETY": "KETY",
    "AUTOPARTNER": "AUTOPARTNER",
    "WIRTUALNA POLSKA": "WIRTUALNA",
    "WIELTON": "WIELTON",
    "MERCATOR": "MERCATOR", "MERCATOR MEDICAL": "MERCATOR",
    "STRONG POINT": "STRONGPOINT",
    "XTPL": "XTPL",
    "PCF GROUP": "PCF",
    "HUUUGE": "HUUUGE",
    "PEPCO": "PEPCO", "PEPCO GROUP NV": "PEPCO",
    "BORYSZEW": "BORYSZEW",
    "BUDIMEX": "BUDIMEX",
    "GPW": "GPW",
    "AMBRA": "AMBRA",
    "KRUK": "KRUK",
    "PZU": "PZU",
    "MARVIPOL": "MARVIPOL",
}

# Map threadGroup text to canonical ticker for reliable company identification
COMPANY_TO_TICKER: dict[str, str] = {
    "Orlen": "ORLEN", "PKN Orlen": "ORLEN",
    "KGHM": "KGHM",
    "CD Projekt": "CDPROJEKT", "CD PROJEKT": "CDPROJEKT",
    "Jastrzębska Spółka Węglowa": "JSW", "JSW": "JSW",
    "NG2 / CCC": "CCC", "CCC": "CCC",
    "OT Logistics": "OTLOGISTICS",
    "GREENX METALS LIMITED": "GREENX", "GreenX Metals": "GREENX",
    "Molecure SA": "MOLECURE", "Molecure": "MOLECURE",
    "DATAWALK": "DATAWALK", "Datawalk": "DATAWALK",
    "NanoGroup SA": "NANOGROUP", "NanoGroup": "NANOGROUP",
    "Lubawa": "LUBAWA",
    "PYRAMID GAMES SA": "PYRAMID", "Pyramid Games": "PYRAMID",
    "XTB": "XTB", "XTB S.A.": "XTB",
    "Allegro": "ALLEGRO", "ALLEGRO": "ALLEGRO",
    "Pepco Group NV": "PEPCO", "PEPCO": "PEPCO",
    "InPost": "INPOST", "INPOST": "INPOST",
    "PKO BP": "PKOBP", "PKO Bank Polski": "PKOBP",
    "PEKAO": "PEKAO", "Bank Pekao": "PEKAO",
    "mBank": "MBANK", "MBANK": "MBANK",
    "Millennium": "MILLENNIUM",
    "ING BSK": "INGBSK", "ING Bank Śląski": "INGBSK",
    "Alior Bank": "ALIOR", "ALIOR": "ALIOR",
    "Santander Bank Polska": "SANTANDER", "SANTANDER": "SANTANDER",
    "Bank Handlowy": "HANDLOWY",
    "PGE": "PGE", "PGE Polska Grupa Energetyczna": "PGE",
    "Energa": "ENERGA", "ENERGA": "ENERGA",
    "Tauron": "TAURON", "TAURON": "TAURON",
    "PZU": "PZU",
    "LPP": "LPP",
    "Dino Polska": "DINO", "DINO": "DINO",
    "Asseco Poland": "ASSECO", "ASSECO": "ASSECO",
    "Comarch": "COMARCH", "COMARCH": "COMARCH",
    "Sygnity": "SYGNITY", "SYGNITY": "SYGNITY",
    "Text": "TEXT",
    "LiveChat Software": "LIVECHAT",
    "11 bit studios": "11BIT", "11BIT": "11BIT",
    "Ten Square Games": "TSGAMES",
    "Creepy Jar": "CREEPYJAR",
    "PlayWay": "PLAYWAY", "Playway": "PLAYWAY",
    "Bloober Team": "BLOOBER",
    "Iron Wolf Studio SA": "IRONWOLF",
    "One More Level": "OML",
    "PCF Group": "PCF",
    "Medinice S.A.": "MEDINICE", "Medinice": "MEDINICE",
    "Pure Biologics": "PURE",
    "Celon Pharma": "CELON",
    "Mabion": "MABION",
    "Ryvu Therapeutics": "RYVU",
    "Selvita": "SELVITA",
    "Develia": "DEVELIA", "DEVELIA": "DEVELIA",
    "Echo Investment": "ECHO",
    "Atal": "ATAL",
    "Dom Development": "DOMDEV",
    "Ronson Development": "RONSON",
    "PKP Cargo": "PKPCARGO", "PKP CARGO": "PKPCARGO",
    "Columbus Energy": "COLUMBUS",
    "Polenergia": "POLENERGIA",
    "Stalexport Autostrady": "STALEXPORT", "Stalexport": "STALEXPORT",
    "Rafako": "RAFAKO", "RAFAKO": "RAFAKO",
    "Grenevia": "GRENEVIA", "GRENEVIA": "GRENEVIA",
    "Cognor": "COGNOR", "Cognor S.A.": "COGNOR",
    "Famur": "FAMUR",
    "Bogdanka": "BOGDANKA", "LW Bogdanka": "BOGDANKA",
    "Eurocash": "EUROCASH",
    "Kernel": "KERNEL",
    "Cyfrowy Polsat": "CYFROWY",
    "Orange Polska": "ORANGE",
    "Komputronik": "KOMPUTRONIK",
    "Defence Hub": "DEFENCEH",
    "Creotech": "CREOTECH", "Creotech Instruments": "CREOTECH",
    "Mennica Polska": "MENNICA",
    "Capital Partners": "CAPITALPARTNERS",
    "Budimex": "BUDIMEX",
    "GPW": "GPW",
    "Kruk": "KRUK", "KRUK": "KRUK",
    "Inter Cars": "INTERCARS",
    "Amrest": "AMREST", "AmRest": "AMREST",
    "Benefit Systems": "BENEFIT",
    "Wielton": "WIELTON",
    "Boryszew": "BORYSZEW",
    "Ciech": "CIECH",
    "Newag": "NEWAG",
    "Asbis": "ASBIS",
    "Huuuge": "HUUUGE",
    "Voxel": "VOXEL",
    "Grupa Kęty": "KETY", "Kety": "KETY",
    "Mercator Medical": "MERCATOR",
    "Marvipol": "MARVIPOL",
    "InventionMed SA": "INVENTIONMED",
    "Vincigen": "VINCIGEN",
}

# ---------------------------------------------------------------------------
# Sentiment lexicon (Polish financial forum slang)
# ---------------------------------------------------------------------------

BULLISH_WORDS: set[str] = {
    "wzrost", "wzrosty", "rośnie", "rosnie", "rakieta", "moon", "kupuj", "buy",
    "okazja", "perła", "perla", "czarny koń", "czarny kon", "low risk",
    "super", "wyniki", "zysk", "dywidenda", "rekomendacja", "target",
    "przebicie", "wybicie", "zakupy", "mocne", "dno", "odbicie", "squeeze",
    "short squeeze", "long", "kupować", "kupowac", "świeca zielona",
    "historyczny", "peak", "rekord", "pozytywne", "byczo", "rajd",
    "gratulowac", "pogratulowac", "brawa", "hurra", "zielono",
    "ruszamy", "startujemy", "czas na wzrosty", "potencjal", "potencjał",
    "szansa", "kupuję", "kupuje", "dobrze", "bomba", "petarda",
    "espi", "rdr", "poprawa", "strong buy", "niedowartościowana",
    "tania spółka", "tani", "obroty", "wejście", "wejscie",
}

BEARISH_WORDS: set[str] = {
    "spadek", "spadki", "spada", "krach", "crash", "bessa", "sell", "sprzedaj",
    "ucieka", "rzeź", "rzez", "katastrofa", "dramat", "dno", "bankrut",
    "short", "szort", "szorty", "czerwono", "strata", "straty",
    "odwołani", "odwolani", "skandal", "afera", "pozew", "sąd", "sad",
    "bolesny", "boli", "traci", "krwawy", "masakra", "tragedia",
    "uwalony", "przelew", "przecena", "uciekaj", "uciekajcie",
    "sprzedawać", "sprzedawac", "leszcze", "bagholder", "pułapka",
    "pulapka", "manipulacja", "cuchnie", "szkoda", "żałuję", "zaluje",
    "stracony", "strach", "panika", "runął", "runal", "leci",
    "odleszczyli", "blokujcie", "shortom",
}

# ---------------------------------------------------------------------------
# Theme classification keywords
# ---------------------------------------------------------------------------

THEME_KEYWORDS: dict[str, list[str]] = {
    "Management Shakeup & Governance": [
        "prezes", "wiceprezes", "odwołan", "zarząd", "zarzad", "rezygnacja",
        "rada nadzorcza", "walne", "governance", "skandal", "afera",
    ],
    "Commodity Supercycle / Metals": [
        "miedź", "miedz", "srebro", "złoto", "zloto", "surowc", "metal",
        "copper", "silver", "gold", "commodity", "ruda", "kopalnia",
    ],
    "Defense & Geopolitics": [
        "wojsko", "obronny", "obronna", "defence", "defense", "nato", "armia",
        "military", "zbroj", "lubawa", "pgz", "geopolit",
    ],
    "AI & Tech Disruption": [
        "ai", "sztuczna inteligencja", "deepseek", "gpt", "model", "veo",
        "robot", "tech", "digital", "neural", "llm", "machine learning",
    ],
    "Gaming Cycle": [
        "gra", "gier", "game", "gaming", "dlc", "premiera", "release",
        "patch", "steam", "epic", "konsol", "studio", "playstation",
    ],
    "Biotech & Pharma Catalysts": [
        "lek", "pharma", "biotech", "fda", "badani", "kliniczn",
        "molecul", "terapia", "onkolog", "pipeline", "trial",
    ],
    "Rate Cycle & Banking": [
        "stopy", "stopa", "rpp", "nbp", "inflacja", "deflacja",
        "bank", "kredyt", "hipoteczn", "rata", "obligacj",
    ],
    "Green Energy Transition": [
        "oze", "solar", "fotowoltaik", "wiatr", "turbina", "wodór",
        "hydrogen", "ev", "electric", "zielona", "climate", "carbon",
        "energetyka", "transformacja",
    ],
    "E-commerce & Retail Rotation": [
        "ecommerce", "e-commerce", "allegro", "sklep", "retail",
        "online", "marketplace", "sprzedaż", "detaliczn",
    ],
    "Short Squeeze & Retail Momentum": [
        "squeeze", "short", "szort", "szorty", "gamma", "blokuj",
        "wallstreetbets", "wsb", "retail", "ape", "diamond",
        "obroty", "miliard obrotu", "wolumen",
    ],
    "Real Estate Cycle": [
        "nieruchomo", "mieszkan", "deweloper", "development", "develia",
        "atal", "dom", "apart", "grunt", "dzialka",
    ],
    "Earnings Momentum": [
        "wynik", "raport", "szacunkow", "przychod", "ebitda", "zysk",
        "strata", "q1", "q2", "q3", "q4", "rdr", "kwartaln",
        "roczn", "sprawozdanie", "szacunki",
    ],
    "IPO & Capital Events": [
        "ipo", "spo", "emisja", "prawo poboru", "nowa emisja",
        "debiut", "wezwanie", "squeeze out", "delist", "skup",
    ],
    "Institutional Flow": [
        "norges", "fundusz", "ofe", "tfi", "blackrock", "vanguard",
        "instytucjonaln", "aviva", "nn", "pzu tfi", "anchor",
    ],
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ForumThread:
    title: str
    url: str
    company_group: str = ""  # From threadGroup column
    replies: int = 0
    tickers: list[str] = field(default_factory=list)
    sentiment_score: float = 0.0  # -1 (bearish) to +1 (bullish)
    themes: list[str] = field(default_factory=list)


@dataclass
class RotationSignal:
    theme: str
    score: float
    mention_count: int
    unique_tickers: list[str]
    sentiment_avg: float
    thread_count: int
    top_threads: list[str]
    signal_strength: str  # "Strong", "Moderate", "Emerging"
    rotation_direction: str  # "Inflow", "Outflow", "Churning"


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class BankierForumScraper:
    """Scrapes Bankier.pl stock exchange forum threads."""

    BASE_URL = "https://www.bankier.pl/forum/forum_gielda,6,{page}.html"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "pl-PL,pl;q=0.9,en;q=0.8",
    }

    def __init__(self, pages: int = 10, delay: float = 1.5) -> None:
        self.pages = pages
        self.delay = delay
        self.client = httpx.Client(
            headers=self.HEADERS,
            timeout=30.0,
            follow_redirects=True,
        )

    def scrape(self) -> list[ForumThread]:
        """Scrape forum pages and return structured threads."""
        all_threads: list[ForumThread] = []
        for page in range(1, self.pages + 1):
            url = self.BASE_URL.format(page=page)
            console.print(f"  [dim]Fetching page {page}/{self.pages}...[/dim]", end=" ")
            # Retry with exponential backoff on 503
            for attempt in range(3):
                try:
                    resp = self.client.get(url)
                    resp.raise_for_status()
                    threads = self._parse_page(resp.text)
                    all_threads.extend(threads)
                    console.print(f"[green]{len(threads)} threads[/green]")
                    break
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 503 and attempt < 2:
                        wait = (attempt + 1) * 2
                        console.print(f"[yellow]503, retry in {wait}s...[/yellow]", end=" ")
                        time.sleep(wait)
                    else:
                        console.print(f"[red]HTTP {e.response.status_code}[/red]")
                        break
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                    break
            time.sleep(self.delay)
        return all_threads

    def _parse_page(self, html: str) -> list[ForumThread]:
        """Parse a single forum page into threads."""
        soup = BeautifulSoup(html, "html.parser")
        threads: list[ForumThread] = []

        # Bankier uses <table class="threadsList"> with <tr> rows
        # Each row: threadTitle <td> has <a href="temat_...">
        # threadCount <td> has reply count, createDate has timestamp
        thread_table = soup.find("table", class_="threadsList")
        if thread_table:
            for row in thread_table.find_all("tr"):
                title_td = row.find("td", class_="threadTitle")
                if not title_td:
                    continue
                link = title_td.find("a", href=True)
                if not link:
                    continue
                href = link.get("href", "")
                if "temat_" not in href:
                    continue

                title = link.get_text(strip=True)
                if not title or len(title) < 3:
                    continue

                # Skip editorial / sticky threads
                if any(skip in title.lower() for skip in [
                    "giełdowy pit", "gieldowy pit", "informacja od redakcji",
                    "regulamin", "zasady",
                ]):
                    continue

                # Build full URL
                if href.startswith("http"):
                    full_url = href
                elif href.startswith("/"):
                    full_url = f"https://www.bankier.pl{href}"
                else:
                    full_url = f"https://www.bankier.pl/forum/{href}"

                # Extract company group from threadGroup td
                company_group = ""
                group_td = row.find("td", class_="threadGroup")
                if group_td:
                    company_group = group_td.get_text(strip=True)

                # Extract reply count from threadCount td
                replies = 0
                count_td = row.find("td", class_="threadCount")
                if count_td:
                    span = count_td.find("span")
                    if span:
                        try:
                            replies = int(re.sub(r"\D", "", span.get_text()))
                        except ValueError:
                            pass

                thread = ForumThread(
                    title=title, url=full_url,
                    company_group=company_group, replies=replies,
                )
                threads.append(thread)
        else:
            # Fallback: scan all links for temat_ pattern
            for link in soup.find_all("a", href=True):
                href = link.get("href", "")
                if "temat_" not in href:
                    continue
                title = link.get_text(strip=True)
                if not title or len(title) < 3:
                    continue
                if any(skip in title.lower() for skip in [
                    "giełdowy pit", "gieldowy pit", "informacja od redakcji",
                    "regulamin", "zasady",
                ]):
                    continue
                if href.startswith("http"):
                    full_url = href
                elif href.startswith("/"):
                    full_url = f"https://www.bankier.pl{href}"
                else:
                    full_url = f"https://www.bankier.pl/forum/{href}"
                thread = ForumThread(title=title, url=full_url)
                threads.append(thread)

        # Deduplicate by URL
        seen: set[str] = set()
        unique: list[ForumThread] = []
        for t in threads:
            if t.url not in seen:
                seen.add(t.url)
                unique.append(t)

        return unique

    def close(self) -> None:
        self.client.close()


# ---------------------------------------------------------------------------
# NLP analysis engine
# ---------------------------------------------------------------------------

class RotationAnalyzer:
    """Analyzes forum threads for capital rotation signals."""

    def __init__(self) -> None:
        # Build regex patterns for ticker extraction
        self._ticker_pattern = self._build_ticker_regex()

    def _build_ticker_regex(self) -> re.Pattern:
        """Build a regex that matches known tickers and aliases in text."""
        # Combine canonical tickers and aliases
        all_names = list(TICKER_SECTOR.keys()) + list(TICKER_ALIASES.keys())
        # Sort by length (longest first) to avoid partial matches
        all_names.sort(key=len, reverse=True)
        escaped = [re.escape(name) for name in all_names]
        pattern = r"\b(" + "|".join(escaped) + r")\b"
        return re.compile(pattern, re.IGNORECASE)

    def extract_tickers(self, text: str) -> list[str]:
        """Extract canonical tickers from text."""
        matches = self._ticker_pattern.findall(text.upper())
        canonical: list[str] = []
        for m in matches:
            m_upper = m.upper()
            if m_upper in TICKER_ALIASES:
                canonical.append(TICKER_ALIASES[m_upper])
            elif m_upper in TICKER_SECTOR:
                canonical.append(m_upper)
        return list(set(canonical))

    def score_sentiment(self, text: str) -> float:
        """Score sentiment from -1 (bearish) to +1 (bullish)."""
        text_lower = text.lower()
        bull_count = sum(1 for w in BULLISH_WORDS if w in text_lower)
        bear_count = sum(1 for w in BEARISH_WORDS if w in text_lower)
        total = bull_count + bear_count
        if total == 0:
            return 0.0
        return (bull_count - bear_count) / total

    def classify_themes(self, text: str) -> list[str]:
        """Classify text into themes based on keyword matching."""
        text_lower = text.lower()
        matched: list[str] = []
        for theme, keywords in THEME_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                matched.append(theme)
        return matched if matched else ["General Market Chatter"]

    def resolve_company_group(self, group: str) -> Optional[str]:
        """Resolve a threadGroup company name to a canonical ticker."""
        if not group:
            return None
        # Exact match first
        if group in COMPANY_TO_TICKER:
            return COMPANY_TO_TICKER[group]
        # Case-insensitive match
        group_lower = group.lower()
        for name, ticker in COMPANY_TO_TICKER.items():
            if name.lower() == group_lower:
                return ticker
        # Partial match (group name contains known company)
        for name, ticker in COMPANY_TO_TICKER.items():
            if name.lower() in group_lower or group_lower in name.lower():
                return ticker
        return None

    def analyze_threads(self, threads: list[ForumThread]) -> list[ForumThread]:
        """Enrich threads with tickers, sentiment, and themes."""
        for thread in threads:
            # Start with tickers from title text
            title_tickers = self.extract_tickers(thread.title)

            # Add ticker from company group (most reliable source)
            group_ticker = self.resolve_company_group(thread.company_group)
            if group_ticker:
                if group_ticker not in title_tickers:
                    title_tickers.append(group_ticker)

            thread.tickers = title_tickers
            thread.sentiment_score = self.score_sentiment(thread.title)
            thread.themes = self.classify_themes(thread.title)
        return threads

    def compute_rotation_signals(self, threads: list[ForumThread]) -> list[RotationSignal]:
        """Aggregate thread-level data into rotation signals."""
        theme_data: dict[str, dict] = defaultdict(lambda: {
            "tickers": Counter(),
            "sentiments": [],
            "thread_count": 0,
            "thread_titles": [],
        })

        for thread in threads:
            for theme in thread.themes:
                data = theme_data[theme]
                data["thread_count"] += 1
                data["sentiments"].append(thread.sentiment_score)
                data["thread_titles"].append(thread.title)
                for ticker in thread.tickers:
                    data["tickers"][ticker] += 1

        signals: list[RotationSignal] = []
        for theme, data in theme_data.items():
            if theme == "General Market Chatter":
                continue  # filter noise

            mention_count = sum(data["tickers"].values())
            unique_tickers = [t for t, _ in data["tickers"].most_common(10)]
            sentiments = data["sentiments"]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

            # Composite score: weighted combination of volume, breadth, and sentiment
            volume_score = min(mention_count / 5, 3.0)  # cap at 3
            breadth_score = min(len(unique_tickers) / 3, 3.0)  # cap at 3
            activity_score = min(data["thread_count"] / 5, 2.0)  # cap at 2
            sentiment_bonus = abs(avg_sentiment) * 2  # strong sentiment = signal

            composite = volume_score + breadth_score + activity_score + sentiment_bonus

            # Classify signal strength
            if composite >= 5.0:
                strength = "Strong"
            elif composite >= 3.0:
                strength = "Moderate"
            else:
                strength = "Emerging"

            # Rotation direction based on sentiment
            if avg_sentiment > 0.15:
                direction = "Inflow"
            elif avg_sentiment < -0.15:
                direction = "Outflow"
            else:
                direction = "Churning"

            signals.append(RotationSignal(
                theme=theme,
                score=round(composite, 2),
                mention_count=mention_count,
                unique_tickers=unique_tickers,
                sentiment_avg=round(avg_sentiment, 3),
                thread_count=data["thread_count"],
                top_threads=data["thread_titles"][:5],
                signal_strength=strength,
                rotation_direction=direction,
            ))

        # Sort by composite score descending
        signals.sort(key=lambda s: s.score, reverse=True)
        return signals

    def extract_high_potential_tickers(
        self, threads: list[ForumThread], signals: list[RotationSignal]
    ) -> pd.DataFrame:
        """Identify high-potential tickers from rotation signals."""
        ticker_stats: dict[str, dict] = defaultdict(lambda: {
            "mentions": 0,
            "sentiments": [],
            "themes": set(),
            "thread_count": 0,
        })

        for thread in threads:
            for ticker in thread.tickers:
                stats = ticker_stats[ticker]
                stats["mentions"] += 1
                stats["sentiments"].append(thread.sentiment_score)
                stats["themes"].update(thread.themes)
                stats["thread_count"] += 1

        rows = []
        for ticker, stats in ticker_stats.items():
            avg_sent = sum(stats["sentiments"]) / len(stats["sentiments"]) if stats["sentiments"] else 0
            sector = TICKER_SECTOR.get(ticker, "Unknown")
            themes = [t for t in stats["themes"] if t != "General Market Chatter"]

            # High-potential score: mentions * (1 + sentiment) * theme_breadth
            theme_breadth = max(len(themes), 1)
            hp_score = stats["mentions"] * (1 + avg_sent) * (1 + 0.3 * theme_breadth)

            rows.append({
                "Ticker": ticker,
                "Sector": sector,
                "Mentions": stats["mentions"],
                "Sentiment": round(avg_sent, 3),
                "Themes": ", ".join(themes[:3]) if themes else "General",
                "HP_Score": round(hp_score, 2),
                "Signal": "BUY WATCH" if avg_sent > 0.1 and stats["mentions"] >= 2
                    else "MONITOR" if avg_sent > -0.1
                    else "AVOID",
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("HP_Score", ascending=False).reset_index(drop=True)
        return df


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def display_results(
    signals: list[RotationSignal],
    ticker_df: pd.DataFrame,
    thread_count: int,
    output_path: Optional[Path] = None,
) -> None:
    """Display rotation signals and tickers using rich tables."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Header
    console.print()
    console.print(Panel(
        f"[bold]Bankier.pl Forum Rotation Signal Scanner[/bold]\n"
        f"[dim]{timestamp} | {thread_count} threads analyzed[/dim]",
        border_style="blue",
        expand=False,
    ))

    # Top 10 Themes table
    console.print()
    console.print("[bold cyan]TOP 10 ROTATION THEMES[/bold cyan]")
    console.print()

    theme_table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold white on dark_blue",
        border_style="blue",
        expand=True,
    )
    theme_table.add_column("#", style="dim", width=3)
    theme_table.add_column("Theme", style="bold", min_width=30)
    theme_table.add_column("Score", justify="right", width=7)
    theme_table.add_column("Strength", width=10)
    theme_table.add_column("Direction", width=10)
    theme_table.add_column("Threads", justify="right", width=8)
    theme_table.add_column("Mentions", justify="right", width=9)
    theme_table.add_column("Sentiment", justify="right", width=10)
    theme_table.add_column("Key Tickers", min_width=25)

    for i, sig in enumerate(signals[:10], 1):
        # Color-code signal strength
        if sig.signal_strength == "Strong":
            strength_style = "[bold green]Strong[/bold green]"
        elif sig.signal_strength == "Moderate":
            strength_style = "[yellow]Moderate[/yellow]"
        else:
            strength_style = "[dim]Emerging[/dim]"

        # Color-code direction
        if sig.rotation_direction == "Inflow":
            dir_style = "[bold green]Inflow[/bold green]"
        elif sig.rotation_direction == "Outflow":
            dir_style = "[bold red]Outflow[/bold red]"
        else:
            dir_style = "[yellow]Churning[/yellow]"

        # Color-code sentiment
        if sig.sentiment_avg > 0.1:
            sent_style = f"[green]+{sig.sentiment_avg:.3f}[/green]"
        elif sig.sentiment_avg < -0.1:
            sent_style = f"[red]{sig.sentiment_avg:.3f}[/red]"
        else:
            sent_style = f"[yellow]{sig.sentiment_avg:.3f}[/yellow]"

        tickers = ", ".join(sig.unique_tickers[:5])

        theme_table.add_row(
            str(i),
            sig.theme,
            f"{sig.score:.1f}",
            strength_style,
            dir_style,
            str(sig.thread_count),
            str(sig.mention_count),
            sent_style,
            tickers,
        )

    console.print(theme_table)

    # High-Potential Tickers table
    console.print()
    console.print("[bold cyan]HIGH POTENTIAL TICKERS[/bold cyan]")
    console.print()

    ticker_table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold white on dark_green",
        border_style="green",
        expand=True,
    )
    ticker_table.add_column("#", style="dim", width=3)
    ticker_table.add_column("Ticker", style="bold", width=14)
    ticker_table.add_column("Sector", min_width=20)
    ticker_table.add_column("Mentions", justify="right", width=9)
    ticker_table.add_column("Sentiment", justify="right", width=10)
    ticker_table.add_column("HP Score", justify="right", width=9)
    ticker_table.add_column("Signal", width=12)
    ticker_table.add_column("Themes", min_width=30)

    top_tickers = ticker_df.head(15)
    for i, (_, row) in enumerate(top_tickers.iterrows(), 1):
        # Color-code signal
        if row["Signal"] == "BUY WATCH":
            sig_style = "[bold green]BUY WATCH[/bold green]"
        elif row["Signal"] == "MONITOR":
            sig_style = "[yellow]MONITOR[/yellow]"
        else:
            sig_style = "[red]AVOID[/red]"

        # Color-code sentiment
        if row["Sentiment"] > 0.1:
            sent_style = f"[green]+{row['Sentiment']:.3f}[/green]"
        elif row["Sentiment"] < -0.1:
            sent_style = f"[red]{row['Sentiment']:.3f}[/red]"
        else:
            sent_style = f"[yellow]{row['Sentiment']:.3f}[/yellow]"

        ticker_table.add_row(
            str(i),
            row["Ticker"],
            row["Sector"],
            str(row["Mentions"]),
            sent_style,
            f"{row['HP_Score']:.1f}",
            sig_style,
            row["Themes"],
        )

    console.print(ticker_table)

    # Signal narratives
    console.print()
    console.print("[bold cyan]ROTATION SIGNAL NARRATIVES[/bold cyan]")
    console.print()

    for i, sig in enumerate(signals[:5], 1):
        sentiment_word = "bullish" if sig.sentiment_avg > 0 else "bearish" if sig.sentiment_avg < 0 else "neutral"
        tickers_str = ", ".join(sig.unique_tickers[:5])

        narrative = (
            f"[bold]{i}. {sig.theme}[/bold] "
            f"({sig.signal_strength} signal, {sig.rotation_direction})\n"
            f"   Forum chatter ({sig.thread_count} threads, {sentiment_word} bias) "
            f"suggests capital is {'flowing into' if sig.rotation_direction == 'Inflow' else 'rotating out of' if sig.rotation_direction == 'Outflow' else 'actively churning in'} "
            f"this theme. Key tickers: [bold]{tickers_str}[/bold]\n"
            f"   [dim]Sample threads: {'; '.join(sig.top_threads[:3])}[/dim]"
        )
        console.print(narrative)
        console.print()

    # Disclaimer
    console.print(Panel(
        "[dim]This is a social sentiment analysis tool. Forum chatter is noisy and "
        "contains manipulation, pump & dump schemes, and misinformation. "
        "Always validate signals against fundamentals, technicals, and risk management "
        "before any investment decision. Not financial advice.[/dim]",
        title="[yellow]Disclaimer[/yellow]",
        border_style="yellow",
        expand=False,
    ))

    # Save report
    if output_path:
        _save_markdown_report(signals, ticker_df, thread_count, output_path)


def _save_markdown_report(
    signals: list[RotationSignal],
    ticker_df: pd.DataFrame,
    thread_count: int,
    path: Path,
) -> None:
    """Save results as a Markdown report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Bankier.pl Forum Rotation Signals",
        f"",
        f"**Generated:** {timestamp}",
        f"**Threads analyzed:** {thread_count}",
        f"",
        f"---",
        f"",
        f"## Top 10 Rotation Themes",
        f"",
        f"| # | Theme | Score | Strength | Direction | Threads | Mentions | Sentiment | Key Tickers |",
        f"|---|-------|-------|----------|-----------|---------|----------|-----------|-------------|",
    ]

    for i, sig in enumerate(signals[:10], 1):
        tickers = ", ".join(sig.unique_tickers[:5])
        lines.append(
            f"| {i} | {sig.theme} | {sig.score:.1f} | {sig.signal_strength} | "
            f"{sig.rotation_direction} | {sig.thread_count} | {sig.mention_count} | "
            f"{sig.sentiment_avg:+.3f} | {tickers} |"
        )

    lines.extend([
        f"",
        f"## High Potential Tickers",
        f"",
        f"| # | Ticker | Sector | Mentions | Sentiment | HP Score | Signal | Themes |",
        f"|---|--------|--------|----------|-----------|----------|--------|--------|",
    ])

    for i, (_, row) in enumerate(ticker_df.head(15).iterrows(), 1):
        lines.append(
            f"| {i} | {row['Ticker']} | {row['Sector']} | {row['Mentions']} | "
            f"{row['Sentiment']:+.3f} | {row['HP_Score']:.1f} | {row['Signal']} | {row['Themes']} |"
        )

    lines.extend([
        f"",
        f"## Signal Narratives",
        f"",
    ])

    for i, sig in enumerate(signals[:5], 1):
        tickers_str = ", ".join(sig.unique_tickers[:5])
        sentiment_word = "bullish" if sig.sentiment_avg > 0 else "bearish" if sig.sentiment_avg < 0 else "neutral"
        lines.append(
            f"### {i}. {sig.theme} ({sig.signal_strength}, {sig.rotation_direction})")
        lines.append(f"")
        lines.append(
            f"Forum chatter ({sig.thread_count} threads, {sentiment_word} bias) "
            f"suggests capital is {'flowing into' if sig.rotation_direction == 'Inflow' else 'rotating out of' if sig.rotation_direction == 'Outflow' else 'actively churning in'} "
            f"this theme. Key tickers: **{tickers_str}**"
        )
        lines.append(f"")
        lines.append(f"Sample threads:")
        for t in sig.top_threads[:3]:
            lines.append(f"- {t}")
        lines.append(f"")

    lines.extend([
        f"---",
        f"",
        f"*Disclaimer: Social sentiment analysis tool. Not financial advice. "
        f"Always validate against fundamentals and proper risk management.*",
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"\n[green]Report saved to:[/green] {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(pages: int = 10, output: Optional[str] = None) -> None:
    """Run the Bankier forum rotation signal scanner."""
    import argparse

    parser = argparse.ArgumentParser(description="Bankier.pl Forum Rotation Signal Scanner")
    parser.add_argument("--pages", type=int, default=pages, help="Number of forum pages to scrape (default: 10)")
    parser.add_argument("--output", type=str, default=output, help="Output path for markdown report")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else Path(
        f"outputs/rotation_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )

    console.print("[bold blue]Bankier.pl Forum Rotation Signal Scanner[/bold blue]")
    console.print("[dim]Scanning for capital rotation signals in WSE forum chatter...[/dim]")
    console.print()

    # Step 1: Scrape
    console.print("[bold]Step 1/3:[/bold] Scraping forum threads...")
    scraper = BankierForumScraper(pages=args.pages, delay=1.5)
    try:
        threads = scraper.scrape()
    finally:
        scraper.close()

    console.print(f"  [green]Collected {len(threads)} threads[/green]")
    console.print()

    if not threads:
        console.print("[red]No threads found. The forum may be blocking requests.[/red]")
        return

    # Step 2: Analyze
    console.print("[bold]Step 2/3:[/bold] Analyzing tickers, sentiment & themes...")
    analyzer = RotationAnalyzer()
    threads = analyzer.analyze_threads(threads)

    # Stats
    threads_with_tickers = [t for t in threads if t.tickers]
    all_tickers = [t for thread in threads for t in thread.tickers]
    console.print(f"  Threads with ticker matches: {len(threads_with_tickers)}/{len(threads)}")
    console.print(f"  Total ticker mentions: {len(all_tickers)}")
    console.print(f"  Unique tickers: {len(set(all_tickers))}")
    console.print()

    # Step 3: Compute signals
    console.print("[bold]Step 3/3:[/bold] Computing rotation signals...")
    signals = analyzer.compute_rotation_signals(threads)
    ticker_df = analyzer.extract_high_potential_tickers(threads, signals)

    console.print(f"  Rotation themes detected: {len(signals)}")
    console.print()

    # Display
    display_results(signals, ticker_df, len(threads), output_path)


if __name__ == "__main__":
    main()
