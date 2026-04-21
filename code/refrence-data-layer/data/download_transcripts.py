#!/usr/bin/env python3
"""
Download all transcript zip files from the UDIVA v0.5 dataset
for train, test, and val splits.

Usage:
    1. Fill in UDIVA_USER and UDIVA_PASSWORD in the .env file next to this script.
    2. python download_transcripts.py [--output_dir <DIR>]

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


def get_zip_links(session: requests.Session, url: str) -> list[str]:
    """Scrape the directory listing page and return all .zip file URLs."""
    resp = session.get(url)
    resp.raise_for_status()
    # Match href attributes pointing to .zip files
    filenames = re.findall(r'href="([^"]+\.zip)"', resp.text)
    return [urljoin(url if url.endswith("/") else url + "/", f) for f in filenames]


def download_file(session: requests.Session, url: str, dest: Path) -> None:
    """Download a single file with a streaming GET, showing progress."""
    filename = url.split("/")[-1]
    if dest.exists():
        print(f"  [skip] {filename} already exists")
        return

    print(f"  [downloading] {filename} ...", end=" ", flush=True)
    with session.get(url, stream=True) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
        size_mb = downloaded / (1024 * 1024)
        print(f"{size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Download UDIVA v0.5 transcript zip files for all splits."
    )
    parser.add_argument("--user", default=None, help="HTTP Basic Auth username (default: UDIVA_USER env var)")
    parser.add_argument("--password", default=None, help="HTTP Basic Auth password (default: UDIVA_PASSWORD env var)")
    parser.add_argument(
        "--output_dir",
        default="transcriptions",
        help="Root directory to save downloads (default: transcriptions/)",
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
        split_url = f"{BASE_URL}/{split}/transcriptions/"
        out_dir = Path(args.output_dir) / split
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Split: {split}")
        print(f"URL:   {split_url}")
        print(f"Dest:  {out_dir}")
        print(f"{'='*60}")

        try:
            zip_urls = get_zip_links(session, split_url)
        except requests.HTTPError as e:
            print(f"  [error] Could not list files: {e}")
            continue

        if not zip_urls:
            print("  [warn] No .zip files found on the page.")
            continue

        print(f"  Found {len(zip_urls)} zip file(s)")
        for zip_url in zip_urls:
            dest = out_dir / zip_url.split("/")[-1]
            try:
                download_file(session, zip_url, dest)
            except requests.HTTPError as e:
                print(f"  [error] Failed to download {zip_url}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
