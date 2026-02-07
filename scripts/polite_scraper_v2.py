#!/usr/bin/env python3
"""
scrape_specific_pages.py

Scrape ONLY the page URLs you provide for file links and download them.
Optional: follow in-page pagination (?page=...) for those same pages.

Install:
  pip install requests beautifulsoup4 tqdm
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, List, Dict
from urllib.parse import urljoin, urlparse, unquote, quote, parse_qs

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def default_user_agent() -> str:
    return "Mozilla/5.0 (compatible; scrape_specific_pages/1.0)"


def safe_name(s: str, max_len: int = 160) -> str:
    s = (s or "").strip()
    s = unquote(s)
    s = re.sub(r"[\/\\:\*\?\"<>\|]+", "_", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    if len(s) > max_len:
        s = s[:max_len].rstrip()
    return s or "untitled"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def req_timeout(connect_s: int, read_s: int):
    return (connect_s, read_s)


def is_html(resp: requests.Response) -> bool:
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "text/html" in ctype:
        return True
    # light sniff
    try:
        head = (resp.content or b"")[:256].lower()
        return b"<html" in head or b"<!doctype html" in head
    except Exception:
        return False


def looks_like_age_gate_html(html: str) -> bool:
    s = (html or "").lower()
    return ("are you 18 years of age or older" in s) or ("/age-verify" in s)


def make_gate_url(file_url: str) -> str:
    p = urlparse(file_url)
    return f"https://www.justice.gov/age-verify?destination={quote(p.path)}"


def pass_age_gate(session: requests.Session, destination_url: str, sleep: float, timeout_s: int, user_agent: str) -> None:
    """
    Tries multiple strategies to pass DOJ age verification:
    1) GET /age-verify?destination=... and click the best "Yes" link if present.
    2) If a form exists, POST it (Drupal-ish).
    """
    headers = {"User-Agent": user_agent}
    gate_url = make_gate_url(destination_url)

    time.sleep(sleep)
    r = session.get(gate_url, timeout=req_timeout(15, timeout_s), headers=headers, allow_redirects=True)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # Strategy 1: Find a "Yes" anchor with href
    yes_hrefs: List[str] = []
    for a in soup.find_all("a", href=True):
        txt = (a.get_text(" ", strip=True) or "").lower()
        if txt == "yes":
            yes_hrefs.append(a["href"])

    def score(href: str) -> int:
        h = href.lower()
        s = 0
        if "search.justice.gov" in h:
            s -= 1000
        if "age-verify" in h:
            s += 25
        if "destination=" in h:
            s += 10
        return s

    if yes_hrefs:
        best = sorted(yes_hrefs, key=score, reverse=True)[0]
        yes_url = urljoin(gate_url, best)
        time.sleep(sleep)
        rr = session.get(yes_url, timeout=req_timeout(15, timeout_s), headers=headers, allow_redirects=True)
        rr.raise_for_status()
        return

    # Strategy 2: Look for a form and POST it
    form = soup.find("form")
    if form:
        action = form.get("action") or gate_url
        post_url = urljoin(gate_url, action)
        data: Dict[str, str] = {}
        for inp in form.find_all("input"):
            name = inp.get("name")
            if name:
                data[name] = inp.get("value", "")
        # Common button field
        data["op"] = "Yes"
        time.sleep(sleep)
        pr = session.post(post_url, data=data, timeout=req_timeout(15, timeout_s), headers=headers, allow_redirects=True)
        pr.raise_for_status()
        return

    raise RuntimeError("Could not locate a usable YES link or form on age-verify page.")


def load_urls(urls_file: Optional[str], urls: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    if urls:
        out.extend(urls)
    if urls_file:
        with open(urls_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    out.append(s)
    # dedupe preserve order
    seen = set()
    dedup = []
    for u in out:
        if u not in seen:
            seen.add(u)
            dedup.append(u)
    return dedup


def extract_file_links(page_url: str, html: str, exts: Set[str], file_path_prefix: Optional[str]) -> Set[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: Set[str] = set()

    for a in soup.find_all("a", href=True):
        u = urljoin(page_url, a["href"])
        p = urlparse(u)
        if p.scheme not in ("http", "https"):
            continue
        ext = os.path.splitext(p.path)[1].lower().lstrip(".")
        if not ext or ext not in exts:
            continue
        if file_path_prefix and not p.path.startswith(file_path_prefix):
            continue
        links.add(p._replace(fragment="").geturl())

    return links


def extract_pager_next(page_url: str, html: str) -> Optional[str]:
    """
    Returns a URL for the next page if the current page has a pager 'Next' link.
    Only returns links that look like the same page with ?page=N.
    """
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        txt = (a.get_text(" ", strip=True) or "").lower()
        if txt == "next":
            u = urljoin(page_url, a["href"])
            # guard: must have a page= query param
            q = parse_qs(urlparse(u).query)
            if "page" in q:
                return u
    return None


@dataclass(frozen=True)
class Result:
    url: str
    ok: bool
    saved_path: str
    status: str
    size_bytes: int
    sha256: str


def download_one(session: requests.Session, file_url: str, out_dir: Path, sleep: float, timeout_s: int,
                 user_agent: str, age_gate: bool, do_hash: bool) -> Result:
    headers = {"User-Agent": user_agent}
    p = urlparse(file_url)

    rel_dir = Path(*[safe_name(x) for x in p.path.split("/")[:-1] if x])
    filename = safe_name(os.path.basename(p.path)) or "download"
    dst_dir = out_dir / rel_dir
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / filename

    if dst.exists() and dst.stat().st_size > 0:
        digest = sha256_file(dst) if do_hash else ""
        return Result(file_url, True, str(dst), "exists", dst.stat().st_size, digest)

    try:
        # probe
        time.sleep(sleep)
        r = session.get(file_url, timeout=req_timeout(15, timeout_s), headers=headers, allow_redirects=True, stream=False)
        r.raise_for_status()

        if age_gate and (("age-verify" in r.url) or (is_html(r) and looks_like_age_gate_html(r.text[:12000]))):
            pass_age_gate(session, file_url, sleep=sleep, timeout_s=timeout_s, user_agent=user_agent)
            time.sleep(sleep)
            r = session.get(file_url, timeout=req_timeout(15, timeout_s), headers=headers, allow_redirects=True, stream=False)
            r.raise_for_status()

        if is_html(r):
            return Result(file_url, False, "", "html_not_file_after_gate", 0, "")

        # stream
        time.sleep(sleep)
        r = session.get(file_url, timeout=req_timeout(15, timeout_s), headers=headers, allow_redirects=True, stream=True)
        r.raise_for_status()

        if is_html(r):
            return Result(file_url, False, "", "html_not_file_stream", 0, "")

        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

        size = dst.stat().st_size
        digest = sha256_file(dst) if do_hash else ""
        return Result(file_url, True, str(dst), "downloaded", size, digest)

    except Exception as e:
        return Result(file_url, False, "", f"{type(e).__name__}: {e}", 0, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls-file", required=True, help="Text file with one page URL per line")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--ext", default="pdf", help="Comma-separated extensions to download")
    ap.add_argument("--file-path-prefix", default=None, help="Only download files whose URL path starts with this")
    ap.add_argument("--follow-pager", action="store_true", help="Follow ?page= pagination on each provided URL")
    ap.add_argument("--pager-max", type=int, default=5000, help="Safety cap on pager pages per start URL")
    ap.add_argument("--sleep", type=float, default=0.75, help="Delay between requests")
    ap.add_argument("--timeout", type=int, default=120, help="Read timeout seconds")
    ap.add_argument("--user-agent", default=default_user_agent())
    ap.add_argument("--age-gate", action="store_true")
    ap.add_argument("--hash", action="store_true")
    ap.add_argument("--manifest", default="download_manifest.csv")
    args = ap.parse_args()

    page_urls = load_urls(args.urls_file, None)
    if not page_urls:
        raise SystemExit("urls file had zero usable URLs")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {e.strip().lower().lstrip(".") for e in args.ext.split(",") if e.strip()}
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = out_dir / manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    headers = {"User-Agent": args.user_agent}

    all_files: Set[str] = set()
    print(f"Scraping {len(page_urls)} start page(s)...", flush=True)

    for start in page_urls:
        seen_pages = 0
        next_url: Optional[str] = start

        while next_url:
            seen_pages += 1
            if seen_pages > args.pager_max:
                print(f"  pager cap hit at {args.pager_max} pages for {start}", flush=True)
                break

            time.sleep(args.sleep)
            r = session.get(next_url, timeout=req_timeout(15, args.timeout), headers=headers, allow_redirects=True)
            r.raise_for_status()

            found = extract_file_links(next_url, r.text, exts, args.file_path_prefix)
            all_files |= found

            if not args.follow_pager:
                break

            nxt = extract_pager_next(next_url, r.text)
            next_url = nxt

        print(f"  {start} -> pages={seen_pages}, total_files_so_far={len(all_files)}", flush=True)

    files_list = sorted(all_files)
    print(f"Total unique file URLs: {len(files_list)}", flush=True)

    ok = 0
    fail = 0
    results: List[Result] = []

    for u in tqdm(files_list, desc="Downloading"):
        res = download_one(session, u, out_dir, args.sleep, args.timeout, args.user_agent, args.age_gate, args.hash)
        results.append(res)
        if res.ok:
            ok += 1
        else:
            fail += 1

    file_exists = manifest_path.exists()
    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["url", "ok", "saved_path", "status", "size_bytes", "sha256"])
        if not file_exists or manifest_path.stat().st_size == 0:
            w.writeheader()
        for r in results:
            w.writerow(
                dict(
                    url=r.url,
                    ok=r.ok,
                    saved_path=r.saved_path,
                    status=r.status,
                    size_bytes=r.size_bytes,
                    sha256=r.sha256,
                )
            )

    print(f"\nDone. Success: {ok}, Failed: {fail}")
    print(f"Saved under: {out_dir.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
