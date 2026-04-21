#!/usr/bin/env python3
"""
Decrypt and extract all downloaded UDIVA zip files using 7z.

The decryption keys are embedded in this script (from the UDIVA dataset docs).
Each zip is extracted into the same directory where it resides.

Usage:
    python unzip_all.py [--dry-run] [--root <DIR>]
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Decryption-key mapping
#
# For transcript files with <batch_number>, the key applies to ALL batches
# (e.g. talk_transcripts1_train.zip … talk_transcripts7_train.zip).
# We store the pattern WITHOUT the batch number and match by prefix+suffix.
# ---------------------------------------------------------------------------

# Exact filename -> key
EXACT_KEYS: dict[str, str] = {
    # ── TRAIN: metadata ───────────────────────────────────────────────
    "metadata_train.zip":                       r"UX4%cC{<n4c9^zMk",
    # ── TRAIN: annotations ────────────────────────────────────────────
    "annotations_talk_train.zip":               r"_#.8JE7)Phd%v>+W",
    "annotations_animals_train.zip":            r"L2kX3zEq]4yH<4Vy",
    "annotations_ghost_train.zip":              r"R+!wu<TJQ3mv3~SC",
    "annotations_lego_train.zip":               r"`7p8{fpFgg]TWuBa",
    # ── VAL (masked): metadata ────────────────────────────────────────
    "metadata_val.zip":                         r"^pGu2L7d@gWkh\+@",
    # ── VAL (masked): annotations ─────────────────────────────────────
    "talk_annotations_val_masked.zip":          r'pJd9f3teL`Zz<b"4',
    "animals_annotations_val.zip":              r"E*2fMJ\p7=^d@P-<",
    "ghost_annotations_val.zip":                r"p&tVJ]@_L<aau2(N",
    "lego_annotations_val.zip":                 r"MQ&$\tVyvH>A}4%]",
    # ── VAL (unmasked) / TEST (masked): metadata ─────────────────────
    "metadata_val_unmasked.zip":                r'h^7fuG8wN:"&;BFp',
    "metadata_test.zip":                        r"7jbfn]7}g}3Y-/Z{",
    # ── VAL (unmasked): annotations ───────────────────────────────────
    "talk_annotations_val.zip":                 r"wt^2(LhSmKqP^USy",
    # ── TEST (masked): annotations ────────────────────────────────────
    "talk_annotations_test_masked.zip":         r"4}YZMX5jFY`5`%Nz",
    "animals_annotations_test.zip":             r'xquP"AH:(3.erLC5',
    "ghost_annotations_test.zip":               r";2qLT]6SYATM>@#}",
    "lego_annotations_test.zip":                r'avyaK7:gmFD2\/")',
    # ── TEST (unmasked): annotations ──────────────────────────────────
    "talk_annotations_test.zip":                r"Bs6D5{y#D&)HJC~g",
    # ── TEST (unmasked): metadata ─────────────────────────────────────
    "metadata_test_unmasked.zip":               ":q&.Skg!cgQ\"E'7M",
}

# Transcript patterns: (prefix, suffix) -> key
# Matches files like  <prefix><digits><suffix>
TRANSCRIPT_KEYS: list[tuple[str, str, str]] = [
    # ── TRAIN ─────────────────────────────────────────────────────────
    ("talk_transcripts",       "_train.zip",        r"Sn2pu`m6RGd&M7N@"),
    ("animals_transcripts",    "_train.zip",        r"ndMq%Y8r37zY&*wj"),
    ("ghost_transcripts",      "_train.zip",        r"U~Z@7{Qa#.7^*q/4"),
    ("lego_transcripts",       "_train.zip",        r":y9<5L)eFf}+Lvz@"),
    # ── VAL (masked) ─────────────────────────────────────────────────
    ("talk_transcripts",       "_val_masked.zip",   r"BDArK236!sJ7N$\a"),
    ("animals_transcripts",    "_val.zip",          r'A]]"+@;GWV9^R6sQ'),
    ("ghost_transcripts",      "_val.zip",          r"EP-?n<q`3Mm3_YT*"),
    ("lego_transcripts",       "_val.zip",          r'nK[D:g9JqB"S"-UT'),
    # ── VAL (unmasked) ───────────────────────────────────────────────
    ("talk_transcripts",       "_val.zip",          r'D+DtKDxLpXv+ge3"'),
    # ── TEST (masked) ─────────────────────────────────────────────────
    ("talk_transcripts",       "_test_masked.zip",  r'xA2-6nM"Be*B-\YA'),
    ("animals_transcripts",    "_test.zip",         r"k@]jXsD?3[QDWgwX"),
    ("ghost_transcripts",      "_test.zip",         r"wh(DDkd.J~D9\Fem"),
    ("lego_transcripts",       "_test.zip",         r"5/M5/vNc)_nJf}G5"),
    # ── TEST (unmasked) ──────────────────────────────────────────────
    ("talk_transcripts",       "_test.zip",         r"Z-t[bX2>UST`t`Yx"),
]

# Recording patterns (multi-part .z01, .z02, … archives)
# The key applies to the set; we extract only from the .zip master file.
RECORDING_KEYS: dict[str, str] = {
    # ── TRAIN ─────────────────────────────────────────────────────────
    "recordings_talk_train":        r"t@6Gwvm%M^-M6M5-",
    "recordings_animals_train":     r"?7W3WmJu{fNPVg<u",
    "recordings_ghost_train":       r"#/WCDf6x+8}Ex%PR",
    "recordings_lego_train":        r"7>epBX>WRjG3_]9p",
    # ── VAL (masked) ─────────────────────────────────────────────────
    "talk_recordings_val_masked":   r"k(F>WJb!;z_w&q8m",
    "animals_recordings_val":       r"Jw5Q^'2y9N+<Qj>S",
    "ghost_recordings_val":         r"5E5s?e?,N^;}_w(}",
    "lego_recordings_val":          r"Dd,e;G3!<YY6yjCE",
    # ── VAL (unmasked) ───────────────────────────────────────────────
    "talk_recordings_val":          r"R2P5Jj2'*N32/M3=",
    # ── TEST (masked) ────────────────────────────────────────────────
    "talk_recordings_test_masked":  r'B)y"_`WFL6d/?>:]',
    "animals_recordings_test":      r'-[6C7"bFm{*GaF<B',
    "ghost_recordings_test":        r"9jqb`Z{!Ld'N3p`7",
    "lego_recordings_test":         r'@J#3`P6#-Wrc"LLf',
    # ── TEST (unmasked) ──────────────────────────────────────────────
    "talk_recordings_test":         r"fR@fpJY.A<6R<Uzj",
}


def lookup_key(filename: str) -> str | None:
    """Return the decryption key for *filename*, or None if unknown."""
    # 1. Exact match
    if filename in EXACT_KEYS:
        return EXACT_KEYS[filename]

    # 2. Transcript pattern match (prefix + digits + suffix)
    #    More-specific suffixes are tried first (e.g. _val_masked before _val)
    sorted_patterns = sorted(TRANSCRIPT_KEYS, key=lambda t: -len(t[1]))
    for prefix, suffix, key in sorted_patterns:
        if filename.startswith(prefix) and filename.endswith(suffix):
            middle = filename[len(prefix):-len(suffix)]
            if middle.isdigit():
                return key

    # 3. Recording multi-part archive (.zip master)
    stem = Path(filename).stem          # e.g. "recordings_talk_train"
    if stem in RECORDING_KEYS:
        return RECORDING_KEYS[stem]

    return None


def extract(zip_path: Path, key: str, dry_run: bool = False) -> bool:
    """Extract *zip_path* into its parent directory using 7z with *key*."""
    out_dir = zip_path.parent
    cmd = [
        "7z", "x",
        f"-p{key}",
        "-aoa",              # overwrite all
        f"-o{out_dir}",
        str(zip_path),
    ]
    if dry_run:
        safe = " ".join(cmd[:3]) + " <key> " + " ".join(cmd[3:])
        print(f"  [dry-run] {safe}")
        return True

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [FAIL] 7z returned {result.returncode}")
        if result.stderr:
            print(f"         {result.stderr.strip()}")
        if result.stdout:
            # 7z often prints errors to stdout
            for line in result.stdout.splitlines():
                if "ERROR" in line or "Wrong password" in line:
                    print(f"         {line.strip()}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Decrypt & extract all UDIVA zip files.")
    parser.add_argument(
        "--root",
        default=".",
        help="Root data directory containing metadata/, transcriptions/, etc. (default: .)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    args = parser.parse_args()

    root = Path(args.root).resolve()

    # Collect all .zip files
    zips = sorted(root.rglob("*.zip"))
    if not zips:
        print("No .zip files found.")
        return

    print(f"Found {len(zips)} zip file(s) under {root}\n")

    ok, skipped, failed = 0, 0, 0
    for zp in zips:
        fname = zp.name
        key = lookup_key(fname)
        if key is None:
            print(f"  [skip] {zp.relative_to(root)}  (no key found)")
            skipped += 1
            continue

        print(f"  [extract] {zp.relative_to(root)}")
        if extract(zp, key, dry_run=args.dry_run):
            ok += 1
        else:
            failed += 1

    print(f"\nDone: {ok} extracted, {skipped} skipped, {failed} failed.")


if __name__ == "__main__":
    main()
