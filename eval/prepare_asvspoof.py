"""Unpack the ASVspoof 2019 LA eval mirror into FLAC files plus a metadata CSV.

    uv run --group eval python -m eval.prepare_asvspoof

Reads the mirror's parquet file (verified against the official release by
scripts/verify_asvspoof_mirror.py) and writes, under SQUEAKTEST_DATA_DIR:

    datasets/asvspoof2019/LA_eval/flac/LA_E_*.flac   original FLAC bytes, unchanged
    datasets/asvspoof2019/LA_eval/meta.csv           file, speaker, system, label

so the evaluation reads ASVspoof clips through squeaktest's normal loader, like any upload.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pyarrow.parquet as pq

from squeaktest.config import Settings

MIRROR = Path("datasets/asvspoof2019/mirror/data/test-00000-of-00001.parquet")
OUTPUT = Path("datasets/asvspoof2019/LA_eval")


def main() -> int:
    data_dir = Settings.from_env().data_dir
    out = data_dir / OUTPUT
    (out / "flac").mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(data_dir / MIRROR)
    labels = ["bonafide", "spoof"]
    count = 0
    with (out / "meta.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "speaker", "system", "label"])
        for batch in parquet.iter_batches(batch_size=1000):
            for row in batch.to_pylist():
                name = row["audio_file_name"].removesuffix(".flac") + ".flac"
                (out / "flac" / name).write_bytes(row["audio"]["bytes"])
                writer.writerow([name, row["speaker_id"], row["system_id"], labels[row["key"]]])
                count += 1
    print(f"Wrote {count:,} clips to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
