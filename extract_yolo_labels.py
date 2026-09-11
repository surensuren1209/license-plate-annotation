#!/usr/bin/env python3
"""
extract_yolo_labels.py

Extracts YOLO-format .txt annotation files based on filenames listed in an input CSV.

Usage Example:
    python3 extract_yolo_labels.py \
        --csv input.csv \
        --source /path/to/yolo_labels \
        --output /path/to/output \
        --column filename
"""

import argparse
import csv
import os
import pathlib
import shutil
import sys


def parse_arguments() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract YOLO .txt annotation files based on filenames in an input CSV."
    )
    parser.add_argument(
        "--csv",
        "-c",
        required=True,
        type=str,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--source",
        "-s",
        required=True,
        type=str,
        help="Path to the source folder containing YOLO .txt annotation files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=str,
        help="Path to the output folder where extracted .txt files will be saved.",
    )
    parser.add_argument(
        "--column",
        "-col",
        default="filename",
        type=str,
        help="Header name of the CSV column containing filenames (default: 'filename').",
    )
    return parser.parse_args()


def index_source_annotations(source_dir: pathlib.Path) -> dict:
    """
    Recursively scans the source directory for all .txt annotation files
    and maps each filename to its absolute file path.

    Returns:
        dict: A mapping of {annotation_filename: absolute_filepath}
    """
    annotation_map = {}
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(".txt"):
                # Store the first occurrence encountered
                if file not in annotation_map:
                    annotation_map[file] = pathlib.Path(root) / file
    return annotation_map


def main() -> None:
    args = parse_arguments()

    csv_path = pathlib.Path(args.csv).resolve()
    source_dir = pathlib.Path(args.source).resolve()
    output_dir = pathlib.Path(args.output).resolve()
    column_name = args.column

    # -------------------------------------------------------------------------
    # 1. Error Handling: Verify file and folder paths
    # -------------------------------------------------------------------------
    if not csv_path.exists():
        print(f"Error: Input CSV file does not exist: '{csv_path}'", file=sys.stderr)
        sys.exit(1)

    if not csv_path.is_file():
        print(f"Error: Specified CSV path is not a file: '{csv_path}'", file=sys.stderr)
        sys.exit(1)

    if not source_dir.exists():
        print(f"Error: Source annotation folder does not exist: '{source_dir}'", file=sys.stderr)
        sys.exit(1)

    if not source_dir.is_dir():
        print(f"Error: Specified source path is not a directory: '{source_dir}'", file=sys.stderr)
        sys.exit(1)

    # Create the output directory if it does not exist
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error: Failed to create output folder '{output_dir}': {e}", file=sys.stderr)
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 2. Index Source Directory Recursively
    # -------------------------------------------------------------------------
    print(f"Indexing YOLO annotation files recursively in: {source_dir}")
    annotation_index = index_source_annotations(source_dir)
    print(f"Found {len(annotation_index)} total .txt annotation file(s) in source directory.")

    # -------------------------------------------------------------------------
    # 3. Read CSV and Process Filenames
    # -------------------------------------------------------------------------
    total_csv_entries = 0
    processed_stems = set()  # Tracks unique stems to prevent duplicate processing
    found_count = 0
    missing_count = 0
    copied_count = 0
    missing_filenames = []

    try:
        with open(csv_path, mode="r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)

            # Error Handling: Empty CSV or missing header
            if reader.fieldnames is None:
                print(f"Error: CSV file '{csv_path}' is empty or invalid.", file=sys.stderr)
                sys.exit(1)

            # Error Handling: Missing specified column
            if column_name not in reader.fieldnames:
                available_cols = ", ".join(f"'{c}'" for c in reader.fieldnames)
                print(
                    f"Error: Column '{column_name}' not found in CSV header.\n"
                    f"Available columns: {available_cols}",
                    file=sys.stderr,
                )
                sys.exit(1)

            for row in reader:
                total_csv_entries += 1
                raw_filename = row.get(column_name, "").strip()

                if not raw_filename:
                    continue

                # Strip image extension (e.g. .jpg, .png, .jpeg, .bmp) to get base name
                file_path_obj = pathlib.Path(raw_filename)
                target_txt_name = f"{file_path_obj.stem}.txt"

                # Error Handling & Optimization: Avoid duplicate processing for identical filenames
                if target_txt_name in processed_stems:
                    continue

                processed_stems.add(target_txt_name)

                # Search indexed source folder for the target .txt file
                if target_txt_name in annotation_index:
                    found_count += 1
                    src_file = annotation_index[target_txt_name]
                    dest_file = output_dir / target_txt_name

                    try:
                        # Copy while preserving metadata and exact contents
                        shutil.copy2(src_file, dest_file)
                        copied_count += 1
                    except Exception as e:
                        print(
                            f"Warning: Failed to copy '{src_file}' to '{dest_file}': {e}",
                            file=sys.stderr,
                        )
                else:
                    missing_count += 1
                    missing_filenames.append(raw_filename)

    except csv.Error as e:
        print(f"Error parsing CSV file '{csv_path}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error while reading CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    # Error Handling: Empty CSV with no rows
    if total_csv_entries == 0:
        print(f"Warning: CSV file '{csv_path}' has no data rows.", file=sys.stderr)

    # -------------------------------------------------------------------------
    # 4. Generate missing_files.txt Report
    # -------------------------------------------------------------------------
    missing_report_path = output_dir / "missing_files.txt"
    try:
        with open(missing_report_path, mode="w", encoding="utf-8") as report_file:
            for fname in missing_filenames:
                report_file.write(f"{fname}\n")
    except Exception as e:
        print(f"Warning: Could not write missing files report: {e}", file=sys.stderr)

    # -------------------------------------------------------------------------
    # 5. Output Extraction Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 48)
    print("             EXTRACTION SUMMARY             ")
    print("=" * 48)
    print(f" Total CSV entries processed : {total_csv_entries}")
    print(f" Unique filenames            : {len(processed_stems)}")
    print(f" .txt files found            : {found_count}")
    print(f" .txt files missing          : {missing_count}")
    print(f" .txt files successfully copied: {copied_count}")
    print(f" Missing files report saved to: {missing_report_path}")
    print("=" * 48)


if __name__ == "__main__":
    main()
