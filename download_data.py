#!/usr/bin/env python3
"""
Download the arxiv-for-FANNS dataset (small split) from HuggingFace.

Usage:
    python download_data.py              # downloads to data/
    python download_data.py --out-dir /path/to/data
"""
import argparse
import os
import sys

import logging
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download arxiv-for-FANNS dataset")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="Output directory (default: data/)",
    )
    parser.add_argument(
        "--split",
        choices=["small", "medium"],
        default="medium",
        help="Dataset split to download (default: medium)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.info("ERROR: huggingface_hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    repo_id = "SPCL/arxiv-for-fanns-medium"
    suffix = "_small" if args.split == "small" else ""

    files = [
        f"database_vectors{suffix}.fvecs",
        f"database_attributes{suffix}.jsonl",
    ]

    for filename in files:
        dest = os.path.join(args.out_dir, filename)
        if os.path.exists(dest):
            logger.info(f"  Already exists: {dest}")
            continue
        logger.info(f"  Downloading {filename}...")
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=args.out_dir,
            )
            logger.info(f"  Saved: {path}")
        except Exception as e:
            logger.info(f"  ERROR downloading {filename}: {e}")
            logger.info(f"  Try manually: https://huggingface.co/datasets/{repo_id}")
            sys.exit(1)

    logger.info(f"\nDataset ready in {args.out_dir}/")
    logger.info("Next: python run_all.sh  OR  python benchmarks/evaluate_all.py --data-dir data/")


if __name__ == "__main__":
    main()
