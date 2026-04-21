#!/usr/bin/env python3
"""
Download all annotation files from the lego-annotations bucket on the
UDIVA MinIO S3 console.

Usage:
    1. Fill in MINIO_USER and MINIO_PASSWORD in the .env file.
    2. python download_annotations.py [--output_dir <DIR>]

Notes:
    - Files ending with '_sp' are training annotations.
    - Files without '_sp' are test annotations.
"""

import argparse
import os
import requests
from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import quote

# Load .env from the same directory as this script
load_dotenv(Path(__file__).resolve().parent / ".env")

CONSOLE_URL = "https://s3-console.data.chalearnlap.cvc.uab.es"
BUCKET = "lego-annotations"
DEFAULT_PREFIX = ""


def console_login(session: requests.Session, user: str, password: str) -> None:
    """Authenticate with the MinIO Console API and store the session cookie."""
    resp = session.post(
        f"{CONSOLE_URL}/api/v1/login",
        json={"accessKey": user, "secretKey": password},
    )
    resp.raise_for_status()
    if "token" not in session.cookies:
        raise RuntimeError(
            f"Login failed — no token cookie received (status {resp.status_code})"
        )
    print("Logged in successfully.")


def list_objects(
    session: requests.Session, prefix: str = "", recursive: bool = True
) -> list[dict]:
    """List all objects under a given prefix in the bucket."""
    objects = []
    last_name = ""
    while True:
        params = {
            "prefix": prefix,
            "recursive": str(recursive).lower(),
            "withVersions": "false",
            "limit": "1000",
        }
        if last_name:
            params["after"] = last_name
        resp = session.get(
            f"{CONSOLE_URL}/api/v1/buckets/{BUCKET}/objects",
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()
        page = data.get("objects", None)
        if not page:
            break
        objects.extend(page)
        last_name = page[-1].get("name", "")
        if len(page) < 1000:
            break
    return objects


def download_object(
    session: requests.Session, object_name: str, dest: Path
) -> None:
    """Download a single object from the MinIO Console."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {object_name} already exists")
        return

    print(f"  [downloading] {object_name} ...", end=" ", flush=True)
    url = f"{CONSOLE_URL}/api/v1/buckets/{BUCKET}/objects/download"
    with session.get(url, params={"prefix": object_name}, stream=True) as resp:
        resp.raise_for_status()
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
        size_kb = downloaded / 1024
        print(f"{size_kb:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Download annotation files from the lego-annotations MinIO bucket."
    )
    parser.add_argument(
        "--user",
        default=None,
        help="MinIO Console username (default: MINIO_USER env var)",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="MinIO Console password (default: MINIO_PASSWORD env var)",
    )
    parser.add_argument(
        "--output_dir",
        default="annotations",
        help="Root directory to save downloads (default: annotations/)",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="S3 prefix to filter objects (default: all)",
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

    prefix = args.prefix
    print(f"\nListing objects in bucket '{BUCKET}' (prefix='{prefix}') ...")
    objects = list_objects(session, prefix)

    if not objects:
        print("No objects found.")
        return

    # Filter out directory markers
    files = [o for o in objects if not o.get("name", "").endswith("/")]
    print(f"Found {len(files)} file(s) to download.\n")

    # Classify by split
    train_files = [f for f in files if "_sp" in Path(f["name"]).stem]
    test_files = [f for f in files if "_sp" not in Path(f["name"]).stem]
    print(f"  Training files (_sp): {len(train_files)}")
    print(f"  Test files (no _sp):  {len(test_files)}\n")

    output_dir = Path(args.output_dir)
    for obj in files:
        name = obj.get("name", "")
        # Strip prefix to get relative path
        rel_path = name
        if prefix and rel_path.startswith(prefix):
            rel_path = rel_path[len(prefix) :]

        dest = output_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            download_object(session, name, dest)
        except requests.HTTPError as e:
            print(f"  [error] Failed to download {name}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
