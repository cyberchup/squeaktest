"""Verify the Hugging Face mirror of ASVspoof 2019 LA eval against the official release.

The official LA.zip (7.6 GB, Edinburgh DataShare) downloads at about 40 KB/s, so squeaktest
uses a mirror (Bisher/ASVspoof_2019_LA). The mirror is repackaged as parquet, so its checksum
can't be compared with the official one. Instead, using HTTP range requests into the official
zip (a few MB in total), this checks:

1. every one of the 71,237 eval rows (file, speaker, attack system, label) against the
   official protocol file, and
2. a random sample of clips byte for byte against the official FLAC files.

    uv run --group eval python scripts/verify_asvspoof_mirror.py MIRROR_TEST.parquet
"""

from __future__ import annotations

import random
import struct
import sys
import urllib.request
import zlib
from pathlib import Path

import pyarrow.parquet as pq

OFFICIAL_ZIP = (
    "https://datashare.ed.ac.uk/server/api/core/bitstreams/"
    "a9f87c35-f055-4015-80e2-2fdff0d46269/content"
)
ZIP_SIZE = 7_640_952_520
PROTOCOL = "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
AUDIO_PREFIX = "LA/ASVspoof2019_LA_eval/flac/"


def fetch(start: int, end: int) -> bytes:
    """Bytes [start, end] of the official zip."""
    request = urllib.request.Request(OFFICIAL_ZIP, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=600) as response:  # noqa: S310 - fixed https URL
        if response.status != 206:
            raise RuntimeError(f"server ignored the range request (HTTP {response.status})")
        return response.read()


def central_directory() -> dict[str, tuple[int, int, int]]:
    """Map member name -> (compression method, compressed size, local header offset)."""
    tail = fetch(ZIP_SIZE - 65536, ZIP_SIZE - 1)
    i = tail.rfind(b"PK\x06\x06")  # zip64 end of central directory
    if i < 0:
        raise RuntimeError("zip64 end of central directory not found")
    cd_size, cd_offset = struct.unpack_from("<QQ", tail, i + 40)
    print(f"Reading the official zip's central directory ({cd_size / 1e6:.1f} MB)...")
    cd = fetch(cd_offset, cd_offset + cd_size - 1)
    members, pos = {}, 0
    while pos < len(cd) and cd[pos : pos + 4] == b"PK\x01\x02":
        (method,) = struct.unpack_from("<H", cd, pos + 10)
        csize, usize = struct.unpack_from("<II", cd, pos + 20)
        name_len, extra_len, comment_len = struct.unpack_from("<HHH", cd, pos + 28)
        (offset,) = struct.unpack_from("<I", cd, pos + 42)
        name = cd[pos + 46 : pos + 46 + name_len].decode()
        extra = cd[pos + 46 + name_len : pos + 46 + name_len + extra_len]
        csize, offset = _zip64_values(extra, usize, csize, offset)
        members[name] = (method, csize, offset)
        pos += 46 + name_len + extra_len + comment_len
    return members


def _zip64_values(extra: bytes, usize: int, csize: int, offset: int) -> tuple[int, int]:
    """Read 64-bit sizes and offsets from the zip64 extra field where the 32-bit ones overflow."""
    pos = 0
    while pos + 4 <= len(extra):
        tag, size = struct.unpack_from("<HH", extra, pos)
        if tag == 0x0001:
            values = iter(struct.unpack_from(f"<{size // 8}Q", extra, pos + 4))
            if usize == 0xFFFFFFFF:
                next(values)
            if csize == 0xFFFFFFFF:
                csize = next(values)
            if offset == 0xFFFFFFFF:
                offset = next(values)
        pos += 4 + size
    return csize, offset


def read_member(members: dict, name: str) -> bytes:
    method, csize, offset = members[name]
    header = fetch(offset, offset + 29)
    name_len, extra_len = struct.unpack_from("<HH", header, 26)
    start = offset + 30 + name_len + extra_len
    data = fetch(start, start + csize - 1)
    if method == 0:
        return data
    if method == 8:
        return zlib.decompress(data, -15)
    raise RuntimeError(f"unsupported compression method {method}")


def main(mirror: Path, sample: int = 30, seed: int = 0) -> int:
    table = pq.read_table(mirror, columns=["audio_file_name", "speaker_id", "system_id", "key"])
    rows = table.to_pylist()
    print(f"Mirror: {len(rows):,} rows")

    members = central_directory()
    official = {}
    for line in read_member(members, PROTOCOL).decode().splitlines():
        speaker, file_id, _, system, key = line.split()
        official[file_id] = (speaker, system, key)
    print(f"Official protocol: {len(official):,} rows")

    keys = ["bonafide", "spoof"]

    def mirror_entry(r: dict) -> tuple[str, str, str]:
        key = keys[r["key"]] if isinstance(r["key"], int) else r["key"]
        return r["speaker_id"], r["system_id"], key

    mismatches = [
        r["audio_file_name"]
        for r in rows
        if official.get(r["audio_file_name"].removesuffix(".flac")) != mirror_entry(r)
    ]
    missing = set(official) - {r["audio_file_name"].removesuffix(".flac") for r in rows}
    print(f"Label check: {len(mismatches)} mismatched rows, {len(missing)} official files missing")

    picked = random.Random(seed).sample(range(len(rows)), sample)
    audio_table = pq.read_table(mirror, columns=["audio_file_name", "audio"])
    names = audio_table.column("audio_file_name").to_pylist()
    audios = audio_table.column("audio")
    identical = 0
    for i in picked:
        name = names[i].removesuffix(".flac")
        mirror_bytes = audios[i].as_py()["bytes"]
        official_bytes = read_member(members, f"{AUDIO_PREFIX}{name}.flac")
        identical += mirror_bytes == official_bytes
    print(f"Audio check: {identical} of {sample} sampled clips byte-identical to the official FLAC")

    ok = not mismatches and not missing and identical == sample
    print("VERIFIED" if ok else "NOT VERIFIED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
