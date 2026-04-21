#!/usr/bin/env python3
"""
Download all files under the mosaics/ prefix from the UDIVA MinIO S3 console.

Usage:
    1. Fill in MINIO_USER and MINIO_PASSWORD in the .env file.
    2. python download_mosaics.py [--output_dir <DIR>] [--prefix <PREFIX>]

    You can also pass --user / --password on the command line to override .env.
"""

import argparse
import os
import re
import requests
from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import quote

# Load .env from the same directory as this script
load_dotenv(Path(__file__).resolve().parent / ".env")

CONSOLE_URL = "https://s3-console.data.chalearnlap.cvc.uab.es"
BUCKET = "udiva"
DEFAULT_PREFIX = "mosaics/"


def console_login(session: requests.Session, user: str, password: str) -> None:
    """Authenticate with the MinIO Console API and store the session cookie."""
    resp = session.post(
        f"{CONSOLE_URL}/api/v1/login",
        json={"accessKey": user, "secretKey": password},
    )
    resp.raise_for_status()
    # MinIO Console sets the token as a cookie (204 No Content response)
    if "token" not in session.cookies:
        raise RuntimeError(f"Login failed — no token cookie received (status {resp.status_code})")
    print("Logged in successfully.")


def list_objects(session: requests.Session, prefix: str, recursive: bool = True) -> list[dict]:
    """List all objects under a given prefix in the bucket."""
    objects = []
    params = {
        "prefix": prefix,
        "recursive": str(recursive).lower(),
        "withVersions": "false",
    }
    # MinIO Console may paginate; keep fetching until done
    resp = session.get(
        f"{CONSOLE_URL}/api/v1/buckets/{BUCKET}/objects",
        params=params,
    )
    resp.raise_for_status()
    data = resp.json()
    if "objects" in data and data["objects"]:
        objects.extend(data["objects"])
    return objects


def download_object(session: requests.Session, object_name: str, dest: Path) -> None:
    """Download a single object from MinIO Console."""
    filename = object_name.split("/")[-1]
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {object_name} already exists")
        return

    print(f"  [downloading] {object_name} ...", end=" ", flush=True)
    encoded_prefix = quote(object_name, safe="")
    url = f"{CONSOLE_URL}/api/v1/buckets/{BUCKET}/objects/download"
    with session.get(url, params={"prefix": object_name}, stream=True) as resp:
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
        description="Download files from UDIVA MinIO S3 Console (mosaics/)."
    )
    parser.add_argument("--user", default=None, help="MinIO Console username (default: MINIO_USER env var)")
    parser.add_argument("--password", default=None, help="MinIO Console password (default: MINIO_PASSWORD env var)")
    parser.add_argument(
        "--output_dir",
        default="mosaics",
        help="Root directory to save downloads (default: mosaics/)",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help=f"S3 prefix to download (default: {DEFAULT_PREFIX})",
    )
    args = parser.parse_args()

    user = args.user or os.environ.get("MINIO_USER")
    password = args.password or os.environ.get("MINIO_PASSWORD")

    if not user or not password:
        parser.error(
            "Credentials required. Set MINIO_USER / MINIO_PASSWORD in .env "
            "or pass --user / --password."
        )

    session = requests.Session()
    console_login(session, user, password)

    print(f"\nListing objects under '{args.prefix}' ...")
    objects = list_objects(session, args.prefix)

    if not objects:
        print("No objects found.")
        return

    # Filter out "directory" markers (size 0 with trailing /)
    files = [o for o in objects if not o.get("name", "").endswith("/")]
    print(f"Found {len(files)} file(s) to download.\n")

    output_dir = Path(args.output_dir)
    for obj in files:
        name = obj.get("name", "")
        # Preserve the sub-directory structure relative to the prefix
        rel_path = name
        if rel_path.startswith(args.prefix):
            rel_path = rel_path[len(args.prefix):]

        dest = output_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            download_object(session, name, dest)
        except requests.HTTPError as e:
            print(f"  [error] Failed to download {name}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
