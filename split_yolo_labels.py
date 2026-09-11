#!/usr/bin/env python3

import os
import shutil
import argparse


def split_yolo_labels(labels_dir, output_dir, batch_size=10000):
    """
    Split YOLO .txt label files into sorted batches.

    - Files are sorted by filename.
    - Each split contains up to batch_size .txt files.
    - Original files are never modified or moved.
    - No file is duplicated between splits.
    """

    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    os.makedirs(output_dir, exist_ok=True)

    print("Scanning YOLO TXT files...")

    txt_files = [
        entry.name
        for entry in os.scandir(labels_dir)
        if entry.is_file() and entry.name.lower().endswith(".txt")
    ]

    txt_files.sort()

    total = len(txt_files)

    print(f"Found {total:,} YOLO TXT files.")
    print(f"Batch size: {batch_size:,}")

    if total == 0:
        print("No .txt files found.")
        return

    split_number = 1

    for start in range(0, total, batch_size):
        batch = txt_files[start:start + batch_size]

        split_dir = os.path.join(
            output_dir, f"split_{split_number:02d}", "labels"
        )
        os.makedirs(split_dir, exist_ok=True)

        print(
            f"Creating split_{split_number:02d}: "
            f"{len(batch):,} files "
            f"({start + 1:,} - {start + len(batch):,})"
        )

        for filename in batch:
            src = os.path.join(labels_dir, filename)
            dst = os.path.join(split_dir, filename)
            shutil.copy2(src, dst)

        split_number += 1

    total_splits = split_number - 1

    print("\nDone!")
    print(f"Total files : {total:,}")
    print(f"Total splits: {total_splits}")
    print(f"Output      : {output_dir}")
    print("Original labels were not modified.")


def main():
    parser = argparse.ArgumentParser(
        description="Split YOLO TXT labels into sorted batches."
    )

    parser.add_argument(
        "--labels_dir",
        required=True,
        help="Source directory containing YOLO .txt files"
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for split folders"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Number of TXT files per split (default: 10000)"
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch_size must be greater than 0")

    split_yolo_labels(
        args.labels_dir,
        args.output_dir,
        args.batch_size
    )


if __name__ == "__main__":
    main()
