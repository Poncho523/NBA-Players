"""Scraper de Basketball-Reference: tablas Totals y Advanced por temporada.

Uso:
    python -m nba_classifier.data.scraper --seasons 2015 2025

Devuelve un dict {"totals": DataFrame, "advanced": DataFrame} por temporada.
No se encarga de limpiar ni de fusionar: eso es de ingestion/cleaning.
"""
from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

logger = logging.getLogger(__name__)

BASE_URL = "https://www.basketball-reference.com/leagues/NBA_{season}_{kind}.html"
TABLE_IDS = {"totals": "totals_stats", "advanced": "advanced"}
KINDS = tuple(TABLE_IDS)

RAW_DIR = Path("data/raw")
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
REQUEST_DELAY = 3.0   # segundos entre requests a la misma fuente
MAX_RETRIES = 5

COLUMN_RENAMES = {"name_display": "player_name", "team_name_abbr": "team"}
DROP_COLS = ["ranker"]


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #
def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _cache_path(season: int, kind: str) -> Path:
    return RAW_DIR / f"NBA_{season}_{kind}.html"


def fetch_season_html(session: requests.Session, season: int, kind: str,
                      *, force: bool = False) -> str:
    """Descarga (o lee de caché) el HTML de una temporada/tabla."""
    cache = _cache_path(season, kind)
    if cache.exists() and not force:
        return cache.read_text(encoding="utf-8")

    url = BASE_URL.format(season=season, kind=kind)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=30)
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
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = 2 ** attempt
            logger.warning("Fallo %s (%s), reintento en %ss", url, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"No se pudo obtener {url}")


# --------------------------------------------------------------------------- #
# Parse
# --------------------------------------------------------------------------- #
def _find_table(soup: BeautifulSoup, table_id: str) -> BeautifulSoup | None:
    """BBR comenta algunas tablas; buscamos también dentro de comentarios."""
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
    if v == "":
        return None
    # BBR usa a veces "-" para vacío
    if v in {"-", "—"}:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def parse_table(html: str, kind: str, season: int) -> pd.DataFrame:
    """Convierte el HTML de una temporada/tabla en un DataFrame 'largo'."""
    soup = BeautifulSoup(html, "lxml")
    table = _find_table(soup, TABLE_IDS[kind])
    if table is None:
        raise ValueError(f"No se encontró la tabla {TABLE_IDS[kind]} (season={season})")

    header_cells = table.select("thead tr th")
    stats = [th.get("data-stat") for th in header_cells]
    if not stats or None in stats:
        raise ValueError("Cabecera incompleta en la tabla")

    rows: list[dict] = []
    for tr in table.select("tbody tr"):
        classes = tr.get("class", []) or []
        if "thead" in classes:
            continue  # BBR a veces inserta cabeceras en medio del tbody
        cells = tr.find_all(["th", "td"])
        if len(cells) != len(stats):
            continue

        record: dict = {"season": season}
        for cell, stat in zip(cells, stats):
            if stat == "ranker":
                continue
            if stat == "name_display":
                link = cell.find("a")
                record["player_name"] = (link.get_text(strip=True) if link
                                         else cell.get_text(strip=True))
                record["player_id"] = cell.get("data-append-csv") or None
            elif stat == "team_name_abbr":
                record["team"] = cell.get_text(strip=True)
            elif stat == "awards":
                record["awards_raw"] = cell.get_text(" ", strip=True)
            else:
                record[COLUMN_RENAMES.get(stat, stat)] = _to_number(
                    cell.get_text(strip=True)
                )
        if record.get("player_id") is None:
            continue  # fila de relleno
        rows.append(record)

    df = pd.DataFrame(rows)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df["kind"] = kind
    return df


def scrape_season(season: int, *, force: bool = False) -> dict[str, pd.DataFrame]:
    """Devuelve {'totals': df, 'advanced': df} para una temporada."""
    session = _session()
    out: dict[str, pd.DataFrame] = {}
    for kind in KINDS:
        logger.info("Season %s / %s", season, kind)
        html = fetch_season_html(session, season, kind, force=force)
        out[kind] = parse_table(html, kind, season)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seasons", type=int, nargs="+", required=True)
    p.add_argument("--force", action="store_true", help="Re-descargar aunque exista caché")
    args = p.parse_args()
    for s in args.seasons:
        dfs = scrape_season(s, force=args.force)
        print(f"Season {s}: totals={dfs['totals'].shape}, advanced={dfs['advanced'].shape}")


if __name__ == "__main__":
    _main()