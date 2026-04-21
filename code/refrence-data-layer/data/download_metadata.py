#!/usr/bin/env python3
"""
Download all metadata files from the UDIVA v0.5 dataset
for train, test, and val splits.

The metadata directory listing is scraped recursively, so nested
sub-folders (e.g. metadata/annotations/, metadata/sessions/) are
downloaded preserving their structure.

Usage:
    1. Fill in UDIVA_USER and UDIVA_PASSWORD in the .env file next to this script.
    2. python download_metadata.py [--output_dir <DIR>]

    You can also pass --user / --password on the command line to override .env.
"""

import argparse
import os
import re
import requests
from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import urljoin

# Load .env from the same directory as this script
load_dotenv(Path(__file__).resolve().parent / ".env")

BASE_URL = "https://data.chalearnlap.cvc.uab.es/UDIVA_Protected/UDIVA_v0.5"
SPLITS = ["train", "test", "val"]

# File extensions we expect for metadata; empty set means download everything
# that is not a directory link.
DIRECTORY_LINK_RE = re.compile(r'href="([^"?]+/)"')
FILE_LINK_RE = re.compile(r'href="([^"?][^"]*\.[^"/]+)"')


def parse_listing(session: requests.Session, url: str) -> tuple[list[str], list[str]]:
    """Fetch a directory listing and return (file_urls, subdirectory_urls)."""
    resp = session.get(url)
    resp.raise_for_status()
    base = url if url.endswith("/") else url + "/"

    filenames = FILE_LINK_RE.findall(resp.text)
    file_urls = [
        urljoin(base, f)
        for f in filenames
        if not f.startswith("/") and not f.startswith("?")
    ]

    dirs = DIRECTORY_LINK_RE.findall(resp.text)
    dir_urls = [
        urljoin(base, d)
        for d in dirs
        if not d.startswith("/") and not d.startswith("?") and d != "../"
    ]

    return file_urls, dir_urls


def crawl_files(session: requests.Session, url: str) -> list[str]:
    """Recursively crawl a directory listing and collect all file URLs."""
    file_urls, dir_urls = parse_listing(session, url)
    for sub_url in dir_urls:
        file_urls.extend(crawl_files(session, sub_url))
    return file_urls


def download_file(session: requests.Session, url: str, dest: Path) -> None:
    """Download a single file with a streaming GET, showing progress."""
    filename = dest.name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {filename} already exists")
        return

    print(f"  [downloading] {filename} ...", end=" ", flush=True)
    with session.get(url, stream=True) as resp:
        resp.raise_for_status()
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
        size_mb = downloaded / (1024 * 1024)
        print(f"{size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Download UDIVA v0.5 metadata files for all splits."
    )
    parser.add_argument("--user", default=None, help="HTTP Basic Auth username (default: UDIVA_USER env var)")
    parser.add_argument("--password", default=None, help="HTTP Basic Auth password (default: UDIVA_PASSWORD env var)")
    parser.add_argument(
        "--output_dir",
        default="metadata",
        help="Root directory to save downloads (default: metadata/)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=SPLITS,
        help=f"Splits to download (default: {SPLITS})",
    )
    args = parser.parse_args()

    user = args.user or os.environ.get("UDIVA_USER")
    password = args.password or os.environ.get("UDIVA_PASSWORD")

    if not user or not password:
        parser.error(
            "Credentials required. Set UDIVA_USER / UDIVA_PASSWORD in .env "
            "or pass --user / --password."
        )

    session = requests.Session()
    session.auth = (user, password)

    for split in args.splits:
        split_url = f"{BASE_URL}/{split}/metadata/"
        out_root = Path(args.output_dir) / split

        print(f"\n{'='*60}")
        print(f"Split: {split}")
        print(f"URL:   {split_url}")
        print(f"Dest:  {out_root}")
        print(f"{'='*60}")

        try:
            file_urls = crawl_files(session, split_url)
        except requests.HTTPError as e:
            print(f"  [error] Could not list files: {e}")
            continue

        if not file_urls:
            print("  [warn] No files found on the page.")
            continue

        print(f"  Found {len(file_urls)} file(s)")
        for file_url in file_urls:
            # Preserve sub-directory structure relative to the split metadata URL
            rel = file_url.replace(split_url, "")
            dest = out_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)

            try:
                download_file(session, file_url, dest)
            except requests.HTTPError as e:
                print(f"  [error] Failed to download {file_url}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
