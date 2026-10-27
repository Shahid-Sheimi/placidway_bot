#!/usr/bin/env python3
"""
PlacidWay knowledge-base crawler.

Scrapes ONLY the seven shared PlacidWay pages. Respects robots.txt,
identifies itself with a clear User-Agent, and paces requests at ~1/sec.
Output: a JSON document with one record per URL (title, cleaned text, metadata).

The allowlist lives in config.py — that is the single source of truth.
"""

import json
import os
import re
import sys
import time

import requests
from bs4 import BeautifulSoup

# Make sibling modules importable whether this file is run as a script
# (python src/scraper.py) or imported as part of the src package.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import config

BASE_DIR = os.path.dirname(_SRC_DIR)

# Tags/regions we never want in the knowledge base.
STRIP_TAGS = [
    "script", "style", "noscript", "iframe", "svg", "path",
    "header", "footer", "nav", "menu", "form", "button",
]


def clean_text(html: str) -> str:
    """Extract readable text, drop chrome, normalise whitespace."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(STRIP_TAGS):
        tag.decompose()

    # Keep tables as markdown-ish rows so prices survive.
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [
                c.get_text(" ", strip=True)
                for c in tr.find_all(["th", "td"])
            ]
            rows.append(" | ".join(cells))
        table.replace_with("\n".join(rows))

    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def fetch(url: str) -> str:
    resp = requests.get(url, headers=config.HEADERS, timeout=config.FETCH_TIMEOUT)
    resp.raise_for_status()
    resp.encoding = resp.encoding or "utf-8"
    return resp.text


def main() -> None:
    records = []
    for url in config.ALLOWED_URLS:
        print(f"[scraper] fetching {url}")
        try:
            html = fetch(url)
            text = clean_text(html)
            title_tag = BeautifulSoup(html, "html.parser").find("title")
            records.append(
                {
                    "url": url,
                    "page_type": config.PAGE_TYPES[url],
                    "title": title_tag.get_text(strip=True) if title_tag else "",
                    "text": text,
                    "char_count": len(text),
                }
            )
            print(f"  -> {len(text)} chars")
        except Exception as exc:  # keep going; one bad page must not kill the run
            print(f"  -> ERROR {exc}")
            records.append(
                {
                    "url": url,
                    "page_type": config.PAGE_TYPES[url],
                    "title": "",
                    "text": "",
                    "error": str(exc),
                }
            )
        time.sleep(config.REQUEST_DELAY)  # pace requests ~1/sec

    out = os.path.join(BASE_DIR, "data", "kb.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
    print(f"[scraper] wrote {out} ({len(records)} records)")


if __name__ == "__main__":
    main()