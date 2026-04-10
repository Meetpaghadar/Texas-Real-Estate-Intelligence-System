import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("scrape_real_sources")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}


def _to_number(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value)
    s = re.sub(r"[^\d.]", "", s)
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _fetch(url: str, timeout: int = 40) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _fetch_selenium(url: str, timeout: int = 40) -> str:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument(f"--user-agent={USER_AGENT}")
    driver = webdriver.Chrome(options=options)
    try:
        driver.set_page_load_timeout(timeout)
        driver.get(url)
        time.sleep(2.0)
        return driver.page_source
    finally:
        driver.quit()


def _fetch_with_fallback(url: str, timeout: int = 40) -> str:
    try:
        return _fetch(url, timeout=timeout)
    except Exception as exc:
        logger.warning("Requests fetch failed for %s, trying Selenium: %s", url, exc)
        return _fetch_selenium(url, timeout=timeout)


def scrape_realtor(max_pages: int = 120, sleep_s: float = 0.8) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        url = f"https://www.realtor.com/realestateandhomes-search/Houston_TX/pg-{page}"
        try:
            html = _fetch_with_fallback(url)
            soup = BeautifulSoup(html, "html.parser")
            scripts = soup.find_all("script", type="application/ld+json")
            before = len(rows)
            for s in scripts:
                try:
                    payload = json.loads(s.get_text(strip=True))
                except Exception:
                    continue
                items = payload if isinstance(payload, list) else [payload]
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    if item.get("@type") not in ("SingleFamilyResidence", "Apartment", "House", "Residence"):
                        continue
                    address_obj = item.get("address", {}) or {}
                    geo = item.get("geo", {}) or {}
                    offer = item.get("offers", {}) or {}
                    rows.append(
                        {
                            "price": _to_number(offer.get("price") or item.get("price")),
                            "bedrooms": _to_number(item.get("numberOfRooms") or item.get("numberOfBedrooms")),
                            "bathrooms": _to_number(item.get("numberOfBathroomsTotal") or item.get("numberOfBathrooms")),
                            "built_up_area": _to_number(item.get("floorSize", {}).get("value")),
                            "property_type": item.get("@type"),
                            "address": address_obj.get("streetAddress"),
                            "locality": address_obj.get("addressLocality") or "Houston",
                            "latitude": _to_number(geo.get("latitude")),
                            "longitude": _to_number(geo.get("longitude")),
                            "listing_url": item.get("url"),
                            "source": "realtor",
                        }
                    )
            logger.info("Realtor page %s parsed, added %s rows", page, len(rows) - before)
            time.sleep(sleep_s)
        except Exception as exc:
            logger.warning("Realtor page %s failed: %s", page, exc)
            continue
    return pd.DataFrame(rows)


def scrape_apartments(max_pages: int = 120, sleep_s: float = 0.8) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        url = "https://www.apartments.com/houston-tx/" if page == 1 else f"https://www.apartments.com/houston-tx/{page}/"
        try:
            html = _fetch_with_fallback(url)
            soup = BeautifulSoup(html, "html.parser")
            scripts = soup.find_all("script", type="application/ld+json")
            before = len(rows)
            for s in scripts:
                try:
                    payload = json.loads(s.get_text(strip=True))
                except Exception:
                    continue
                if isinstance(payload, dict) and payload.get("@type") == "ItemList":
                    for entry in payload.get("itemListElement", []):
                        item = entry.get("item", {}) if isinstance(entry, dict) else {}
                        if not isinstance(item, dict):
                            continue
                        addr = item.get("address", {}) or {}
                        geo = item.get("geo", {}) or {}
                        rows.append(
                            {
                                "price": _to_number(item.get("offers", {}).get("price")),
                                "bedrooms": _to_number(item.get("numberOfRooms")),
                                "bathrooms": _to_number(item.get("numberOfBathroomsTotal")),
                                "built_up_area": _to_number(item.get("floorSize", {}).get("value")),
                                "property_type": item.get("@type") or "Apartment",
                                "address": addr.get("streetAddress") or item.get("name"),
                                "locality": addr.get("addressLocality") or "Houston",
                                "latitude": _to_number(geo.get("latitude")),
                                "longitude": _to_number(geo.get("longitude")),
                                "listing_url": item.get("url"),
                                "source": "apartments",
                            }
                        )
            logger.info("Apartments page %s parsed, added %s rows", page, len(rows) - before)
            time.sleep(sleep_s)
        except Exception as exc:
            logger.warning("Apartments page %s failed: %s", page, exc)
            continue
    return pd.DataFrame(rows)


def clean_scraped(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [
        "price",
        "bedrooms",
        "bathrooms",
        "built_up_area",
        "property_type",
        "address",
        "locality",
        "latitude",
        "longitude",
        "listing_url",
        "source",
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = None

    for c in ["price", "bedrooms", "bathrooms", "built_up_area", "latitude", "longitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["locality"] = df["locality"].fillna("Houston").astype(str).str.strip()
    df = df[df["locality"].str.contains("houston|katy|sugar land|bellaire|pearland|cypress|spring", case=False, regex=True)]
    df = df[df["price"].notna()]
    df = df[(df["price"] > 300) & (df["price"] < 25_000_000)]
    df = df.drop_duplicates(subset=["listing_url", "address", "price"], keep="first")
    return df.reset_index(drop=True)


def run_scraping(
    output_csv: str = "data/raw/texas_houston_scraped.csv",
    min_rows: int = 10_000,
    max_pages_per_source: int = 120,
) -> Path:
    root = Path(__file__).resolve().parents[1]
    output_path = root / output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)

    realtor_df = scrape_realtor(max_pages=max_pages_per_source)
    apartments_df = scrape_apartments(max_pages=max_pages_per_source)
    all_df = clean_scraped(pd.concat([realtor_df, apartments_df], ignore_index=True))

    source_counts = all_df["source"].value_counts().to_dict() if not all_df.empty else {}
    logger.info("Scrape source counts: %s", source_counts)
    if len(source_counts) < 2:
        raise RuntimeError("Scraping succeeded for fewer than 2 sources. Check bot blocking/network limits.")
    if len(all_df) < min_rows:
        raise RuntimeError(
            f"Scraped rows below required minimum ({len(all_df)} < {min_rows}). "
            "Increase pages or run with Selenium-backed browser automation."
        )

    all_df.to_csv(output_path, index=False)
    logger.info("Saved scraped dataset to %s with %s rows.", output_path, len(all_df))
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Houston/Texas real estate from real listing sources.")
    parser.add_argument("--output_csv", type=str, default="data/raw/texas_houston_scraped.csv")
    parser.add_argument("--min_rows", type=int, default=10_000)
    parser.add_argument("--max_pages_per_source", type=int, default=120)
    args = parser.parse_args()
    run_scraping(
        output_csv=args.output_csv,
        min_rows=args.min_rows,
        max_pages_per_source=args.max_pages_per_source,
    )
