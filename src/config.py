#!/usr/bin/env python3
"""Shared configuration and boundary rules for the PlacidWay crawler.

ALLOWED_URLS is the single source of truth for what the bot may read.
Nothing outside this list is ever fetched, followed, or cited.
"""

import re
from urllib.parse import urlparse, urlunparse

USER_AGENT = (
    "PlacidWayChatbot/1.0 (+https://placidway.com; "
    "contact: research@placidway.com)"
)
HEADERS = {"User-Agent": USER_AGENT}

# The exact pages the bot is allowed to read. No crawling beyond these.
ALLOWED_URLS = [
    "https://www.placidway.com/profile/5/Alternative-Cancer-Treatment-by-ITC-Immunity-Therapy-Center",
    "https://www.placidway.com/package/7903/Alternative-Adenocarcinoma-Cancer-Treatment-Package-in-Tijuana-Mexico-by-ITC",
    "https://www.placidway.com/package/7902/Alternative-Adrenal-Cancer-Treatment-Package-in-Tijuana-Mexico-by-ITC",
    "https://www.placidway.com/package/7901/Alternative-Anal-Cancer-Treatment-Package-in-Tijuana-Mexico-by-ITC",
    "https://www.placidway.com/profile-videos/5/Alternative-Cancer-Treatment-by-ITC-Immunity-Therapy-Center",
    "https://www.placidway.com/price-list/5/Alternative-Cancer-Treatment-by-ITC-Immunity-Therapy-Center",
    "https://www.placidway.com/search-medical-pricings/Alternative-Medicine/All/1",
]

PAGE_TYPES = {
    ALLOWED_URLS[0]: "clinic_profile",
    ALLOWED_URLS[1]: "treatment_package",
    ALLOWED_URLS[2]: "treatment_package",
    ALLOWED_URLS[3]: "treatment_package",
    ALLOWED_URLS[4]: "clinic_videos",
    ALLOWED_URLS[5]: "price_list",
    ALLOWED_URLS[6]: "price_comparison",
}

# Query params that carry no content value and are stripped before comparison.
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "msclkid", "yclid", "dclid", "mc_cid", "mc_eid",
    "ref", "source", "cmpid", "igshid",
}

# How long to wait between requests (seconds).
REQUEST_DELAY = 1.0
# Hard cap on redirects we will follow.
MAX_REDIRECTS = 5
# Fetch timeout (seconds).
FETCH_TIMEOUT = 45


def normalize_url(url: str) -> str:
    """Normalize a URL for allowlist comparison.

    - lowercase the host
    - strip the fragment
    - strip tracking query params
    - strip the trailing slash
    """
    parsed = urlparse(url.strip())
    host = (parsed.netloc or "").lower()
    # drop default ports
    if host.endswith(":80"):
        host = host[:-3]
    elif host.endswith(":443"):
        host = host[:-4]
    path = parsed.path or "/"
    # collapse trailing slash (keep root as "/")
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/") or "/"
    query_pairs = parsed.query.split("&") if parsed.query else []
    kept = [
        pair for pair in query_pairs
        if pair and pair.split("=", 1)[0].lower() not in TRACKING_PARAMS
    ]
    query = "&".join(kept)
    return urlunparse(
        (parsed.scheme.lower() or "https", host, path, parsed.params, query, "")
    )


# Pre-computed allowlist (normalized form).
ALLOWED_NORMALIZED = {normalize_url(u) for u in ALLOWED_URLS}


def is_allowed(url: str) -> bool:
    """Return True only if `url` (after normalization) is in ALLOWED_URLS."""
    return normalize_url(url) in ALLOWED_NORMALIZED


def url_host(url: str) -> str:
    """Return the lowercased host of a URL."""
    return (urlparse(url).netloc or "").lower()