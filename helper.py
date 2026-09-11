#!/usr/bin/env python3
import os
import argparse
import logging
import cv2
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def parse_label_file(label_path):
    results = []
    if not os.path.exists(label_path):
        return results
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            coords = [float(p) for p in parts[1:]]

            if len(coords) == 4:
                cx, cy, w, h = coords
                x_min, x_max = cx - w / 2, cx + w / 2
                y_min, y_max = cy - h / 2, cy + h / 2
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                xs, ys = coords[0::2], coords[1::2]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
            else:
                continue

            results.append({'class_id': class_id, 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max})
    return results

def order_quad_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[s.argmin()]  # TL
    rect[2] = pts[s.argmax()]  # BR
    d = np.diff(pts, axis=1)
    rect[1] = pts[d.argmin()]  # TR
    rect[3] = pts[d.argmax()]  # BL
    return rect

def refine_plate_contour(crop_bgr):
    h_crop, w_crop = crop_bgr.shape[:2]
    crop_area = h_crop * w_crop

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Contrast enhancement via CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # 2. Edge detection with lowered threshold
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    valid_boxes = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        # Relaxed area threshold: 3% minimum
        if area < crop_area * 0.03:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        if w == 0 or h == 0:
            continue

        aspect_ratio = max(w, h) / min(w, h)
        # Relaxed aspect ratio range: 1.0 to 7.0
        if 1.0 <= aspect_ratio <= 7.0:
            box = cv2.boxPoints(rect)
            valid_boxes.append((area, box))

    if not valid_boxes:
        return None

    valid_boxes.sort(key=lambda x: x[0], reverse=True)
    return order_quad_points(valid_boxes[0][1])

def process_dataset(images_dir, labels_dir, output_dir, crop_scale=1.2, out_format="quadbox"):
    images_dir, labels_dir, output_dir = Path(images_dir), Path(labels_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_dir.glob("*.txt"))

    for lf in label_files:
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
            p = images_dir / f"{lf.stem}{ext}"
            if p.is_file():
                img_path = p
                break
        if not img_path:
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        H, W = img_bgr.shape[:2]
        labels = parse_label_file(lf)
        out_lines = []

        for li, lb in enumerate(labels, 1):
            cid = lb['class_id']
            x1p, y1p = lb['x_min'] * W, lb['y_min'] * H
            x2p, y2p = lb['x_max'] * W, lb['y_max'] * H
            bw, bh = max(1.0, x2p - x1p), max(1.0, y2p - y1p)
            cx, cy = (x1p + x2p) / 2, (y1p + y2p) / 2

            cw, ch = bw * crop_scale, bh * crop_scale
            rx1, ry1 = max(0, int(cx - cw / 2)), max(0, int(cy - ch / 2))
            rx2, ry2 = min(W, int(cx + cw / 2)), min(H, int(cy + ch / 2))

            crop = img_bgr[ry1:ry2, rx1:rx2]
            if crop.shape[0] < 5 or crop.shape[1] < 5:
                continue

            refined_box_crop = refine_plate_contour(crop)

            # FALLBACK: If edge fitting fails, fall back to loose box relative to crop
            if refined_box_crop is None:
                bx1, by1 = x1p - rx1, y1p - ry1
                bx2, by2 = x2p - rx1, y2p - ry1
                refined_box_crop = np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]], dtype=np.float32)

            coords = []
            if out_format == "quadbox":
                for px, py in refined_box_crop:
                    coords.extend([
                        round(min(1.0, max(0.0, (px + rx1) / W)), 6),
                        round(min(1.0, max(0.0, (py + ry1) / H)), 6)
                    ])
            
            out_lines.append(f"{cid} " + " ".join(map(str, coords)))

            vis = crop.copy()
            q = refined_box_crop.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [q], True, (0, 255, 0), 2)
            cv2.imwrite(str(crops_dir / f"{lf.stem}_obj{li}.jpg"), vis)

        with open(output_dir / lf.name, "w") as f:
            f.write("\n".join(out_lines) + ("\n" if out_lines else ""))

    print(f"Done! Check results in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", default="images")
    parser.add_argument("-l", "--labels", default="labels")
    parser.add_argument("-o", "--output", default="valitate")
    parser.add_argument("-f", "--format", default="quadbox")
    parser.add_argument("-cs", "--crop-scale", type=float, default=1.2)
    args = parser.parse_args()
    process_dataset(args.images, args.labels, args.output, args.crop_scale, args.format)
