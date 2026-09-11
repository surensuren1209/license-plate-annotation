#!/usr/bin/env python3
"""
csv_to_yolo_labels.py

Converts CSV bounding box annotations [x1, y1, x2, y2] to standard normalized
YOLO format annotation text files: <class_id> <center_x> <center_y> <width> <height>

Usage:
    python3 csv_to_yolo_labels.py -h
    python3 csv_to_yolo_labels.py --csv input.csv --output /path/to/output_dir
"""

import argparse
import csv
import pathlib
import sys
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        prog="csv_to_yolo_labels.py",
        description="Convert CSV bounding boxes [x1, y1, x2, y2] to standard normalized YOLO .txt format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with normalized [x1, y1, x2, y2] coordinates:
  python3 csv_to_yolo_labels.py -i input.csv -o ./labels

  # Usage with absolute pixel coordinates and fixed image size (640x640):
  python3 csv_to_yolo_labels.py --csv input.csv --output ./labels --img-width 640 --img-height 640

  # Display help menu:
  python3 csv_to_yolo_labels.py -h
        """,
    )
    parser.add_argument(
        "-i",
        "--csv",
        "--input",
        required=True,
        type=str,
        dest="csv",
        help="Path to the input CSV file containing image annotations. (Required)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        dest="output",
        help="Path to the destination output directory for .txt label files. (Required)",
    )
    parser.add_argument(
        "-w",
        "--img-width",
        type=float,
        default=None,
        help="Image width in pixels (required if x1, y1, x2, y2 are pixel values and CSV has no width column).",
    )
    parser.add_argument(
        "-H",
        "--img-height",
        type=float,
        default=None,
        help="Image height in pixels (required if x1, y1, x2, y2 are pixel values and CSV has no height column).",
    )
    return parser.parse_args()


def detect_columns(fieldnames):
    """Detects filename, class_id, and box coordinate columns in CSV header."""
    headers = [f.strip() for f in fieldnames]
    lower_headers = [f.lower() for f in headers]

    def get_col(candidates):
        for c in candidates:
            if c in lower_headers:
                return headers[lower_headers.index(c)]
        return None

    # Filename
    fn_col = get_col(["filename", "file", "image", "img_name", "image_name", "image_id", "img", "path"]) or headers[0]

    # Class ID / Label
    class_col = get_col(["class_id", "class", "label", "category_id", "cls", "name", "category"])

    # Separate box coordinate columns: x1, y1, x2, y2 or xmin, ymin, xmax, ymax
    x1_col = get_col(["x1", "xmin", "left", "x_min", "box_x1"])
    y1_col = get_col(["y1", "ymin", "top", "y_min", "box_y1"])
    x2_col = get_col(["x2", "xmax", "right", "x_max", "box_x2"])
    y2_col = get_col(["y2", "ymax", "bottom", "y_max", "box_y2"])

    # Pre-normalized YOLO format: center_x, center_y, width, height
    xc_col = get_col(["x_center", "xc", "center_x"])
    yc_col = get_col(["y_center", "yc", "center_y"])
    w_col = get_col(["width", "w", "box_w", "box_width"])
    h_col = get_col(["height", "h", "box_h", "box_height"])

    # Single bbox column (e.g., "[x1, y1, x2, y2]" or "x1,y1,x2,y2")
    bbox_str_col = get_col(["bbox", "box", "location", "annotations", "yolo"])

    # Image dimensions
    img_w_col = get_col(["img_width", "image_width", "width_img", "img_w"])
    img_h_col = get_col(["img_height", "image_height", "height_img", "img_h"])

    return {
        "filename": fn_col,
        "class": class_col,
        "x1": x1_col,
        "y1": y1_col,
        "x2": x2_col,
        "y2": y2_col,
        "xc": xc_col,
        "yc": yc_col,
        "w": w_col,
        "h": h_col,
        "bbox_str": bbox_str_col,
        "img_w": img_w_col,
        "img_h": img_h_col,
    }


def parse_bbox_string(val_str):
    """Parse string representations like '[100, 150, 200, 250]'."""
    if not val_str:
        return None
    val_str = val_str.strip().strip("[]()").replace(",", " ")
    parts = [float(p) for p in val_str.split() if p]
    return parts if len(parts) >= 4 else None


def convert_to_yolo(row, cols, default_img_w=None, default_img_h=None):
    """
    Converts [x1, y1, x2, y2] to standard YOLO format:
    <class_id> <center_x> <center_y> <width> <height>
    """
    # 1. Extract class_id (default to 0 if absent)
    class_id = "0"
    if cols["class"] and row.get(cols["class"]) is not None:
        class_id = str(row[cols["class"]]).strip()

    x1, y1, x2, y2 = None, None, None, None

    # Single bbox string column parsing
    if cols["bbox_str"] and row.get(cols["bbox_str"]):
        parts = parse_bbox_string(row[cols["bbox_str"]])
        if parts:
            if len(parts) == 5:
                class_id = str(int(parts[0]))
                x1, y1, x2, y2 = parts[1:5]
            elif len(parts) == 4:
                x1, y1, x2, y2 = parts[:4]

    # Individual x1, y1, x2, y2 column parsing
    if x1 is None and all(cols[k] and row.get(cols[k]) is not None for k in ["x1", "y1", "x2", "y2"]):
        x1 = float(row[cols["x1"]])
        y1 = float(row[cols["y1"]])
        x2 = float(row[cols["x2"]])
        y2 = float(row[cols["y2"]])

    # If CSV already contains center_x, center_y, width, height
    if x1 is None and all(cols[k] and row.get(cols[k]) is not None for k in ["xc", "yc", "w", "h"]):
        xc = float(row[cols["xc"]])
        yc = float(row[cols["yc"]])
        w = float(row[cols["w"]])
        h = float(row[cols["h"]])
        return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"

    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None

    # Retrieve image dimensions if present
    img_w = default_img_w
    if cols["img_w"] and row.get(cols["img_w"]):
        img_w = float(row[cols["img_w"]])

    img_h = default_img_h
    if cols["img_h"] and row.get(cols["img_h"]):
        img_h = float(row[cols["img_h"]])

    # Determine if coordinates are already normalized (0.0 to 1.0)
    is_normalized = max(x1, y1, x2, y2) <= 1.0 and min(x1, y1, x2, y2) >= 0.0

    if is_normalized:
        # Convert normalized [x1, y1, x2, y2] -> normalized YOLO
        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0
        w = abs(x2 - x1)
        h = abs(y2 - y1)
    else:
        # Convert pixel [x1, y1, x2, y2] -> normalized YOLO
        if not img_w or not img_h:
            raise ValueError(
                f"Absolute pixel coordinates found ({x1}, {y1}, {x2}, {y2}), "
                f"but image width/height were not provided in CSV or arguments."
            )
        xc = ((x1 + x2) / 2.0) / img_w
        yc = ((y1 + y2) / 2.0) / img_h
        w = abs(x2 - x1) / img_w
        h = abs(y2 - y1) / img_h

    # Clamp values between 0.0 and 1.0
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


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

    image_annotations = defaultdict(list)
    total_rows = 0
    parsed_boxes = 0

    with open(csv_path, mode="r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            print(f"Error: CSV file '{csv_path}' is empty or invalid.", file=sys.stderr)
            sys.exit(1)

        cols = detect_columns(reader.fieldnames)
        fn_col = cols["filename"]

        print(f"Reading CSV: '{csv_path}'")
        print(f"Detected Filename column : '{fn_col}'")
        print(f"Detected Class column    : '{cols['class']}'")

        for row_num, row in enumerate(reader, start=1):
            total_rows += 1
            raw_filename = row.get(fn_col, "").strip()

            if not raw_filename:
                continue

            stem = pathlib.Path(raw_filename).stem
            txt_filename = f"{stem}.txt"

            try:
                yolo_line = convert_to_yolo(
                    row,
                    cols,
                    default_img_w=args.img_width,
                    default_img_h=args.img_height,
                )
                if yolo_line:
                    image_annotations[txt_filename].append(yolo_line)
                    parsed_boxes += 1
                else:
                    if txt_filename not in image_annotations:
                        image_annotations[txt_filename] = []
            except Exception as e:
                print(f"Warning (Row {row_num}): {e}", file=sys.stderr)

    # Write YOLO format .txt files
    files_created = 0
    for txt_name, lines in image_annotations.items():
        out_path = output_dir / txt_name
        with open(out_path, mode="w", encoding="utf-8") as out_f:
            for line in lines:
                out_f.write(f"{line}\n")
        files_created += 1

    print("\n" + "=" * 48)
    print("             EXTRACTION SUMMARY             ")
    print("=" * 48)
    print(f" Total CSV rows processed   : {total_rows}")
    print(f" Total .txt files created   : {files_created}")
    print(f" Total YOLO boxes saved     : {parsed_boxes}")
    print(f" Saved format               : <class_id> <xc> <yc> <w> <h>")
    print(f" Output directory           : {output_dir}")
    print("=" * 48)


if __name__ == "__main__":
    main()
