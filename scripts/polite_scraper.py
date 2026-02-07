#!/usr/bin/env python3
"""
polite_scraper.py

A reusable, polite crawler + downloader for *publicly linked* documents.

What it does
- Crawls a starting URL (and internal links it discovers, within the same host).
- Collects links to files matching extensions you choose (default: pdf).
- Downloads them into an output folder, optionally mirroring a simple path structure.
- Optional: handles a common “age verification gate” pattern by auto-submitting a form
  when a redirect lands you on a URL containing a configurable substring.

What it does NOT do
- Does not bypass auth, brute force, or scrape restricted areas.
- Only follows links it can fetch normally as a public user.

Example:
Crawl only under a path prefix:
   polite_scraper.py \
     --start "https://example.com/resources/index.html" \
     --out "example-downloads" \
     --ext "pdf,png,zip" \
     --scope "path" \
     --path-prefix "/resources/"

Requirements:
  pip install requests beautifulsoup4 tqdm
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


# ----------------------------
# Helpers
# ----------------------------

def safe_name(s: str, max_len: int = 160) -> str:
    """Make a filesystem-safe filename/folder name."""
    s = (s or "").strip()
    s = unquote(s)  # decode %20 etc
    s = re.sub(r"[\/\\:\*\?\"<>\|]+", "_", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    if len(s) > max_len:
        s = s[:max_len].rstrip()
    return s or "untitled"


def normalize_url(u: str) -> str:
    """Normalize for deduping: strip fragment."""
    p = urlparse(u)
    return p._replace(fragment="").geturl()


def is_html_response(resp: requests.Response) -> bool:
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "text/html" in ctype:
        return True
    # Some servers omit content-type; lightly sniff
    try:
        head = resp.content[:256].lower()
        return b"<html" in head or b"<!doctype html" in head
    except Exception:
        return False


def default_user_agent() -> str:
    return "Mozilla/5.0 (compatible; generic_site_downloader/1.0)"

def looks_like_age_gate_html(html_text: str) -> bool:
    s = html_text.lower()
    return ("are you 18 years of age or older" in s) or ("/age-verify" in s)


# ----------------------------
# Data structures
# ----------------------------

@dataclass(frozen=True)
class FileItem:
    url: str
    rel_path: str   # where to save relative to out dir
    filename: str   # filename to save


# ----------------------------
# Scope rules
# ----------------------------

def in_scope(candidate_url: str, start_url: str, scope: str, path_prefix: str | None) -> bool:
    """
    scope:
      - "host": same hostname only
      - "path": same hostname AND must start with path_prefix (or start_url path's directory if not provided)
    """
    try:
        cand = urlparse(candidate_url)
        start = urlparse(start_url)
    except Exception:
        return False

    if cand.scheme not in ("http", "https"):
        return False

    if cand.netloc != start.netloc:
        return False

    if scope == "host":
        return True

    # scope == "path"
    if path_prefix:
        return cand.path.startswith(path_prefix)

    # default: constrain to start URL directory
    start_dir = start.path
    if not start_dir.endswith("/"):
        start_dir = start_dir.rsplit("/", 1)[0] + "/"
    return cand.path.startswith(start_dir)


# ----------------------------
# Age gate handler (optional)
# ----------------------------

def pass_age_gate(
    session: requests.Session,
    gate_url: str,
    sleep: float,
    timeout: int,
    submit_value: str = "Yes",
    user_agent: str = "",
) -> None:
    """
    Robust age-gate handler for justice.gov:
    - GET the age-verify page
    - Find the *correct* form (not the header search form)
      by looking for:
        - action containing 'age-verify', OR
        - an input named 'destination'
    - POST the form with op=Yes (or equivalent)
    """
    import time
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin

    headers = {"User-Agent": user_agent or "Mozilla/5.0"}
    time.sleep(sleep)
    r = session.get(gate_url, timeout=timeout, headers=headers, allow_redirects=True)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    forms = soup.find_all("form")

    # Pick the right form: prefer one that targets age-verify or has the 'destination' field.
    chosen = None
    for f in forms:
        action = (f.get("action") or "").lower()
        has_destination = f.find("input", {"name": "destination"}) is not None
        if "age-verify" in action or has_destination:
            chosen = f
            break

    if not chosen:
        # Fallback: some templates use links/buttons; try anchor containing "Yes"
        yes_link = soup.find("a", string=lambda s: s and s.strip().lower() == "yes")
        if yes_link and yes_link.get("href"):
            time.sleep(sleep)
            session.get(urljoin(gate_url, yes_link["href"]), timeout=timeout, headers=headers, allow_redirects=True).raise_for_status()
            return
        raise RuntimeError(f"Could not find age-verify form on: {gate_url}")

    action = chosen.get("action") or gate_url
    post_url = urljoin(gate_url, action)

    data = {}
    for inp in chosen.find_all("input"):
        name = inp.get("name")
        if not name:
            continue
        data[name] = inp.get("value", "")

    # Common pattern: Drupal form uses 'op' for the clicked button text.
    if "op" in data:
        data["op"] = submit_value
    else:
        data["op"] = submit_value  # safe default

    time.sleep(sleep)
    pr = session.post(post_url, data=data, timeout=timeout, headers=headers, allow_redirects=True)
    pr.raise_for_status()



# ----------------------------
# Crawler + collector
# ----------------------------

def get_soup(session: requests.Session, url: str, sleep: float, timeout: int, user_agent: str) -> BeautifulSoup:
    time.sleep(sleep)
    r = session.get(url, timeout=timeout, headers={"User-Agent": user_agent})
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def extract_links(soup: BeautifulSoup, base_url: str) -> List[str]:
    links: List[str] = []
    for a in soup.find_all("a", href=True):
        links.append(urljoin(base_url, a["href"]))
    return links


def build_save_location(file_url: str, start_url: str, mirror: str) -> Tuple[str, str]:
    """
    Decide relative path + filename.

    mirror:
      - "flat": save everything into out_dir/
      - "path": mirror URL path directories under out_dir/ (host omitted)
      - "hostpath": mirror host + path under out_dir/ (useful if you crawl multiple domains)
    """
    p = urlparse(file_url)
    host = safe_name(p.netloc)
    path = p.path

    filename = safe_name(os.path.basename(path)) or "download"
    # Ensure we keep a filename even if URL ends in /
    if path.endswith("/") or not os.path.splitext(filename)[1]:
        # attempt name from last non-empty segment
        segs = [s for s in path.split("/") if s]
        filename = safe_name(segs[-1] if segs else "download")

    # Mirror directories
    dir_parts = [s for s in path.split("/")[:-1] if s]
    dir_parts = [safe_name(x) for x in dir_parts]

    if mirror == "flat":
        rel_dir = ""
    elif mirror == "path":
        rel_dir = os.path.join(*dir_parts) if dir_parts else ""
    else:  # hostpath
        rel_dir = os.path.join(host, *dir_parts) if dir_parts else host

    return rel_dir, filename


def matches_extension(url: str, exts: Set[str]) -> bool:
    path = urlparse(url).path
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    return ext in exts


def crawl_and_collect(
    session: requests.Session,
    start_url: str,
    exts: Set[str],
    scope: str,
    path_prefix: Optional[str],
    sleep: float,
    timeout: int,
    user_agent: str,
    max_pages: int,
    ignore_query: bool,
    ignore_paths_regex: Optional[re.Pattern],
) -> Tuple[Set[str], List[FileItem]]:
    seen_pages: Set[str] = set()
    to_crawl: List[str] = [start_url]
    file_items: Dict[str, FileItem] = {}

    while to_crawl and len(seen_pages) < max_pages:
        page_url = to_crawl.pop(0)
        page_url_norm = normalize_url(page_url)

        if page_url_norm in seen_pages:
            continue
        seen_pages.add(page_url_norm)

        try:
            soup = get_soup(session, page_url, sleep=sleep, timeout=timeout, user_agent=user_agent)
        except Exception:
            continue

        for link in extract_links(soup, page_url):
            link = normalize_url(link)

            if not in_scope(link, start_url, scope, path_prefix):
                continue

            # Optional: ignore some paths
            if ignore_paths_regex and ignore_paths_regex.search(urlparse(link).path):
                continue

            # Dedupe by removing query if requested
            if ignore_query:
                parsed = urlparse(link)
                link = parsed._replace(query="").geturl()

            # File link?
            if matches_extension(link, exts):
                rel_dir, filename = build_save_location(link, start_url, mirror="path")
                file_items.setdefault(link, FileItem(url=link, rel_path=rel_dir, filename=filename))
                continue

            # Otherwise, if it looks like an HTML page, queue it
            # (Heuristic: no file extension OR endswith / OR .html/.htm)
            path = urlparse(link).path
            ext = os.path.splitext(path)[1].lower()
            if ext in ("", ".html", ".htm") or path.endswith("/"):
                if link not in to_crawl and link not in seen_pages:
                    to_crawl.append(link)

    return seen_pages, list(file_items.values())


# ----------------------------
# Downloader
# ----------------------------

def download_one(
    session: requests.Session,
    item: FileItem,
    out_dir: Path,
    sleep: float,
    timeout: int,
    user_agent: str,
    age_gate_substring: Optional[str],
    age_submit_value: str,
) -> bool:
    dst_dir = out_dir / item.rel_path if item.rel_path else out_dir
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Preserve extension if missing (rare, but possible)
    ext = os.path.splitext(urlparse(item.url).path)[1]
    filename = item.filename
    if ext and not filename.lower().endswith(ext.lower()):
        filename += ext

    dst = dst_dir / filename
    if dst.exists() and dst.stat().st_size > 0:
        return True

    try:
        # ------------------------------------------------------------
        # 1) PROBE REQUEST (stream=False so we can read HTML easily)
        # ------------------------------------------------------------
        time.sleep(sleep)
        r = session.get(
            item.url,
            timeout=timeout,
            stream=False,
            allow_redirects=True,
            headers={"User-Agent": user_agent},
        )
        r.raise_for_status()

        # ------------------------------------------------------------
        # 2) AGE-GATE HANDLING
        #    A) redirect to /age-verify (your original behavior)
        #    B) gate HTML served at the media URL itself (new behavior)
        # ------------------------------------------------------------
        hit_gate = False

        # A) redirected URL contains known substring (e.g., "/age-verify")
        if age_gate_substring and age_gate_substring in r.url:
            hit_gate = True

        # B) HTML response that looks like the DOJ gate page
        #    (even when URL is still the original media URL)
        if is_html_response(r):
            # Lightweight heuristic; adjust if needed
            snippet = (r.text or "")[:8000].lower()
            if ("are you 18 years of age or older" in snippet) or ("/age-verify" in snippet):
                hit_gate = True

        if hit_gate:
            pass_age_gate(
                session=session,
                gate_url=r.url,  # important: use the page we actually got (contains form tokens)
                sleep=sleep,
                timeout=timeout,
                submit_value=age_submit_value,
                user_agent=user_agent,
            )

            # Re-fetch the original media URL after verification (still probe)
            time.sleep(sleep)
            r = session.get(
                item.url,
                timeout=timeout,
                stream=False,
                allow_redirects=True,
                headers={"User-Agent": user_agent},
            )
            r.raise_for_status()

        # Still HTML after attempting gate? Don't save.
        if is_html_response(r):
            return False

        # ------------------------------------------------------------
        # 3) STREAMING DOWNLOAD (now that we expect binary)
        # ------------------------------------------------------------
        time.sleep(sleep)
        r = session.get(
            item.url,
            timeout=timeout,
            stream=True,
            allow_redirects=True,
            headers={"User-Agent": user_agent},
        )
        r.raise_for_status()

        # Final guard: never save HTML
        if is_html_response(r):
            return False

        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
        return True

    except Exception as e:
        print(f"FAIL: {item.url} ({type(e).__name__}: {e})", flush=True)
    return False




# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Reusable crawler/downloader for publicly linked files.")
    ap.add_argument("--start", required=True, help="Starting URL to crawl from")
    ap.add_argument("--out", default="downloads", help="Output directory")
    ap.add_argument("--ext", default="pdf", help="Comma-separated extensions to download (e.g. pdf,zip,png)")
    ap.add_argument("--scope", choices=["host", "path"], default="host", help="Crawl scope")
    ap.add_argument("--path-prefix", default=None, help="When --scope path, only crawl URLs whose path starts with this")
    ap.add_argument("--sleep", type=float, default=0.25, help="Delay between requests (seconds)")
    ap.add_argument("--timeout", type=int, default=60, help="Request timeout (seconds)")
    ap.add_argument("--max-pages", type=int, default=300, help="Safety cap for number of pages crawled")
    ap.add_argument("--ignore-query", action="store_true", help="Ignore URL querystring when deduping/collecting files")
    ap.add_argument("--ignore-paths", default=None, help=r"Regex of URL paths to ignore, e.g. '/tag/|/login'")
    ap.add_argument("--user-agent", default=default_user_agent(), help="Custom User-Agent string")

    
    # Optional age-gate automation
    ap.add_argument("--age-gate-substring", default=None, help="If a redirect URL contains this substring, try to pass gate form")
    ap.add_argument("--age-submit-value", default="Yes", help="Value to submit on the age gate form (default: Yes)")

    args = ap.parse_args()

    start_url = args.start
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {e.strip().lower().lstrip(".") for e in args.ext.split(",") if e.strip()}
    if not exts:
        raise SystemExit("No extensions provided. Use --ext pdf or --ext pdf,zip,...")

    ignore_re = re.compile(args.ignore_paths) if args.ignore_paths else None

    session = requests.Session()
    user_agent = args.user_agent  # already parsed by argparse

    # Optional: warm up DOJ age verification once so cookies stick
    if args.age_gate_substring:
        try:
            pass_age_gate(
                session=session,
                gate_url="https://www.justice.gov/age-verify?destination=/epstein/doj-disclosures",
                sleep=args.sleep,
                timeout=args.timeout,
                submit_value=args.age_submit_value,
                user_agent=user_agent,
            )
            print("Age gate warm-up successful", flush=True)
        except Exception as e:
            print(f"Age gate warm-up failed (continuing anyway): {e}", flush=True)
    seen_pages, items = crawl_and_collect(
        session=session,
        start_url=start_url,
        exts=exts,
        scope=args.scope,
        path_prefix=args.path_prefix,
        sleep=args.sleep,
        timeout=args.timeout,
        user_agent=args.user_agent,
        max_pages=args.max_pages,
        ignore_query=args.ignore_query,
        ignore_paths_regex=ignore_re,
    )

    print(f"Crawled {len(seen_pages)} page(s). Found {len(items)} file link(s) matching: {sorted(exts)}")

    ok = 0
    fail = 0

    # Choose a mirroring strategy here if you want:
    # - currently crawl collector builds rel_path using "path" mirroring logic.
    for item in tqdm(items, desc="Downloading"):
        if download_one(
            session=session,
            item=item,
            out_dir=out_dir,
            sleep=args.sleep,
            timeout=args.timeout,
            user_agent=args.user_agent,
            age_gate_substring=args.age_gate_substring,
            age_submit_value=args.age_submit_value,
        ):
            ok += 1
        else:
            fail += 1

    print(f"\nDone. Success: {ok}, Failed: {fail}")
    print(f"Saved under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
