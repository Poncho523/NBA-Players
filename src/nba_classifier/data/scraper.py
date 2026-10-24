import argparse
import logging
import time
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup, Comment
import cloudscraper

logger = logging.getLogger(__name__)
BASE_URL = "https://www.basketball-reference.com/leagues/NBA_{season}_{kind}.html"
TABLE_IDS = {"totals": "totals_stats", "advanced": "advanced"}
KINDS = tuple(TABLE_IDS)
RAW_DIR = Path("data/raw")
REQUEST_DELAY = 3.0
MAX_RETRIES = 5

COLUMN_RENAMES = {"name_display": "player_name", "team_name_abbr": "team"}
DROP_COLS = ["ranker"]

def _session(): 
    return cloudscraper.create_scraper()

def _cache_path(season: int, kind: str) -> Path:
    return RAW_DIR / f"NBA_{season}_{kind}.html"

def fetch_season_html(session, season: int, kind: str, *, force: bool = False) -> str:
    cache = _cache_path(season, kind)
    if cache.exists() and not force:
        return cache.read_text(encoding="utf-8")

    url = BASE_URL.format(season=season, kind=kind)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"}
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=30, headers=headers)
            if resp.status_code == 429:
                wait = 2 ** attempt
                logger.warning("429 en %s, espero %ss", url, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            cache.write_text(resp.text, encoding="utf-8")
            time.sleep(REQUEST_DELAY)
            return resp.text
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = 2 ** attempt
            logger.warning("Fallo %s (%s), reintento en %ss", url, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"No se pudo obtener {url}")

def _find_table(soup: BeautifulSoup, table_id: str) -> BeautifulSoup | None:
    table = soup.find("table", id=table_id)
    if table is not None:
        return table
    for node in soup.find_all(string=lambda t: isinstance(t, Comment)):
        inner = BeautifulSoup(node, "lxml")
        table = inner.find("table", id=table_id)
        if table is not None:
            return table
    return None

def _to_number(value: str) -> float | None:
    v = value.strip()
    if v == "" or v in {"-", "—"}:
        return None
    try:
        return float(v)
    except ValueError:
        return None

def parse_table(html: str, kind: str, season: int) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    table = _find_table(soup, TABLE_IDS[kind])
    if table is None:
        raise ValueError(f"No se encontró la tabla {TABLE_IDS[kind]} (season={season})")

    header_cells = table.select("thead tr th")
    stats = [th.get("data-stat") for th in header_cells]
    
    rows: list[dict] = []
    for tr in table.select("tbody tr"):
        classes = tr.get("class", []) or []
        if "thead" in classes:
            continue
        cells = tr.find_all(["th", "td"])
        if len(cells) != len(stats):
            continue

        record: dict = {"season": season}
        for cell, stat in zip(cells, stats):
            if stat == "ranker":
                continue
            if stat == "name_display":
                link = cell.find("a")
                record["player_name"] = (link.get_text(strip=True) if link else cell.get_text(strip=True))
                record["player_id"] = cell.get("data-append-csv") or None
            elif stat == "team_name_abbr":
                record["team"] = cell.get_text(strip=True)
            elif stat == "awards":
                record["awards_raw"] = cell.get_text(" ", strip=True)
            else:
                record[COLUMN_RENAMES.get(stat, stat)] = _to_number(cell.get_text(strip=True))
        if record.get("player_id") is None:
            continue
        rows.append(record)

    df = pd.DataFrame(rows)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df["kind"] = kind
    return df

def scrape_season(season: int, *, force: bool = False) -> dict[str, pd.DataFrame]:
    session = _session()
    out: dict[str, pd.DataFrame] = {}
    for kind in KINDS:
        logger.info("Season %s / %s", season, kind)
        html = fetch_season_html(session, season, kind, force=force)
        out[kind] = parse_table(html, kind, season)
    return out

def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Scraper BBR")
    p.add_argument("--seasons", type=int, nargs="+", required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    for s in args.seasons:
        dfs = scrape_season(s, force=args.force)
        print(f"Season {s}: totals={dfs['totals'].shape}, advanced={dfs['advanced'].shape}")

if __name__ == "__main__":
    _main()