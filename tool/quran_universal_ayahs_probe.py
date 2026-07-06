from __future__ import annotations

import argparse
import json
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DATASET_ID = "hetchyy/quranic-universal-ayahs"
HF_API_URL = f"https://huggingface.co/api/datasets/{DATASET_ID}"
HF_ROWS_API_URL = "https://datasets-server.huggingface.co/first-rows"


def _get_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "Qrn-ASR-audit/1.0"})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description=(
            "Inspect the Quranic Universal Ayahs Hugging Face dataset without "
            "downloading the full audio corpus."
        )
    )
    parser.add_argument(
        "--config",
        default="mishary_rashid_al_afasy_mp3quran",
        help="Dataset config/subset to sample.",
    )
    parser.add_argument("--split", default="train", help="Dataset split to sample.")
    parser.add_argument("--rows", type=int, default=3, help="Sample row count.")
    args = parser.parse_args()

    metadata = _get_json(HF_API_URL)
    siblings = metadata.get("siblings", [])
    card_data = metadata.get("cardData", {}) or {}
    configs = [
        item.get("config_name")
        for item in card_data.get("configs", [])
        if item.get("config_name")
    ]

    print("Dataset:", DATASET_ID)
    print("License:", metadata.get("license") or card_data.get("license") or "unknown")
    print("Last modified:", metadata.get("lastModified", "unknown"))
    print("Configs:", len(configs))
    if configs:
        print("First configs:", ", ".join(configs[:10]))
    print("Files:", len(siblings))

    query = urlencode({
        "dataset": DATASET_ID,
        "config": args.config,
        "split": args.split,
    })
    rows_payload = _get_json(f"{HF_ROWS_API_URL}?{query}")
    rows = rows_payload.get("rows", [])[: max(0, args.rows)]

    print()
    print(f"Sample rows from {args.config}/{args.split}:")
    for row in rows:
        data = row.get("row", {})
        summary = {
            "surah": data.get("surah"),
            "ayah": data.get("ayah"),
            "duration_ms": data.get("duration_ms"),
            "text_uthmani": data.get("text_uthmani"),
            "segments_count": len(data.get("segments") or []),
            "word_timestamps_count": len(data.get("word_timestamps") or []),
            "has_letter_timestamps": bool(data.get("letter_timestamps")),
            "source_url": data.get("source_url"),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
