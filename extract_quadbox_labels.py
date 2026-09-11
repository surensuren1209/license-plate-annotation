#!/usr/bin/env python3
"""
extract_quadbox_labels.py

Extracts 4-corner quadrilateral / quadbox (Oriented Bounding Box) coordinates
from all_annotations.csv and saves them into YOLO OBB format .txt label files.

Output Format per line in .txt file:
    <class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>

Corners (Clockwise):
    (x1, y1) = Top-Left     (vertex_lu_x, vertex_lu_y)
    (x2, y2) = Top-Right    (vertex_ru_x, vertex_ru_y)
    (x3, y3) = Bottom-Right (vertex_rb_x, vertex_rb_y)
    (x4, y4) = Bottom-Left  (vertex_lb_x, vertex_lb_y)
"""

import argparse
import csv
import pathlib
import sys
import time
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract 4-corner quadbox (YOLO OBB) coordinates from CSV into .txt label files."
    )
    parser.add_argument(
        "--csv",
        "-c",
        default="/home/sxr23/snap/antigravity/5/.gemini/antigravity/scratch/all_annotations.csv",
        type=str,
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="/home/sxr23/snap/antigravity/5/.gemini/antigravity/scratch/quadbox_labels",
        type=str,
        help="Output directory to save extracted .txt files.",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="Class ID to assign for objects (default: 0).",
    )
    parser.add_argument(
        "--pixel-coords",
        action="store_true",
        help="Save raw pixel coordinates instead of normalized 0..1 coordinates.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = pathlib.Path(args.csv).resolve()
    output_dir = pathlib.Path(args.output).resolve()

    if not csv_path.exists():
        print(f"Error: CSV file '{csv_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory '{output_dir}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading CSV: {csv_path}")
    print(f"Output directory: {output_dir}")

    start_time = time.time()

    image_annotations = defaultdict(list)
    total_rows = 0
    parsed_boxes = 0

    with open(csv_path, mode="r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            total_rows += 1
            filename = row.get("filename", "").strip()
            if not filename:
                continue

            stem = pathlib.Path(filename).stem
            txt_filename = f"{stem}.txt"

            try:
                # 4-corner pixel coordinates from CCPD / quad annotations
                lu_x = float(row["vertex_lu_x"])
                lu_y = float(row["vertex_lu_y"])
                ru_x = float(row["vertex_ru_x"])
                ru_y = float(row["vertex_ru_y"])
                rb_x = float(row["vertex_rb_x"])
                rb_y = float(row["vertex_rb_y"])
                lb_x = float(row["vertex_lb_x"])
                lb_y = float(row["vertex_lb_y"])

                if args.pixel_coords:
                    line = f"{args.class_id} {lu_x:.1f} {lu_y:.1f} {ru_x:.1f} {ru_y:.1f} {rb_x:.1f} {rb_y:.1f} {lb_x:.1f} {lb_y:.1f}"
                else:
                    img_w = float(row["img_width"])
                    img_h = float(row["img_height"])

                    x1, y1 = lu_x / img_w, lu_y / img_h
                    x2, y2 = ru_x / img_w, ru_y / img_h
                    x3, y3 = rb_x / img_w, rb_y / img_h
                    x4, y4 = lb_x / img_w, lb_y / img_h

                    line = f"{args.class_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"

                image_annotations[txt_filename].append(line)
                parsed_boxes += 1
            except (KeyError, ValueError) as e:
                continue

    print(f"Writing {len(image_annotations)} .txt label files...")
    files_created = 0
    for txt_name, lines in image_annotations.items():
        out_file = output_dir / txt_name
        with open(out_file, mode="w", encoding="utf-8") as out_f:
            for line in lines:
                out_f.write(f"{line}\n")
        files_created += 1

    elapsed = time.time() - start_time

    print("\n" + "=" * 50)
    print("           QUADBOX EXTRACTION SUMMARY           ")
    print("=" * 50)
    print(f" Total CSV rows processed   : {total_rows}")
    print(f" Total .txt files created   : {files_created}")
    print(f" Total quadboxes extracted  : {parsed_boxes}")
    print(f" Format saved               : {'Pixel (x y ...)' if args.pixel_coords else 'Normalized YOLO OBB (0..1)'}")
    print(f" Output folder              : {output_dir}")
    print(f" Time elapsed               : {elapsed:.2f} seconds")
    print("=" * 50)


if __name__ == "__main__":
    main()
