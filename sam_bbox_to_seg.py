#!/usr/bin/env python3
"""
Automated SAM License Plate Segmentation & QuadBox Export.

Handles tricky cases:
  - Loose input bboxes that include holder/mount area above the plate
  - Rotated plates (bow-tie / crossed-lines fix)
  - Over-segmentation into vehicle body

Key improvements:
  --box-shrink     : Shrinks the SAM box prompt inward by a fraction (e.g. 0.08)
                     so SAM focuses inside the plate, not the holder above.
  --neg-points     : Auto negative points at top-edge midpoint & corners to
                     push SAM away from areas outside the actual plate.
  --white-filter   : Post-process mask to keep only the largest white/bright
                     connected region (great for white license plates).

Input label formats supported (auto-detected):
  1. Standard YOLO BBox  (5 tokens) : class_id x_center y_center width height
  2. QuadBox             (9 tokens) : class_id x1 y1 x2 y2 x3 y3 x4 y4
  3. Multi-point Polygon (7+ tokens): class_id x1 y1 x2 y2 ... xN yN

Output formats:
  - quadbox  : 4 corner points (class_id x1 y1 x2 y2 x3 y3 x4 y4) [Default]
  - bbox     : Refined bounding box (class_id x_center y_center width height)
  - polygon  : Multi-point segmentation polygon
"""

import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("SAM_LP_Seg")


# ---------------------------------------------------------------------------
# SAM setup
# ---------------------------------------------------------------------------

def add_sam_to_sys_path():
    script_dir = Path(__file__).resolve().parent
    search_paths = [
        script_dir / "segment-anything-main",
        script_dir.parent / "segment-anything-main",
        Path("/home/sxr23/ultralytics_quadbox/segment-anything-main"),
        Path("/home/sxr23/Documents/segment-anything-main"),
    ]
    for p in search_paths:
        if p.exists() and str(p) not in sys.path:
            sys.path.append(str(p))


def _patch_torchvision_compat():
    """
    Fix: torchvision 0.15/0.16 uses torch.library.register_fake which was
    renamed / added in PyTorch 2.1+. If the installed torch is older,
    we monkey-patch a no-op so torchvision can import without crashing.

    Root cause:
      torchvision/__init__.py → _meta_registrations.py line 163:
        @torch.library.register_fake("torchvision::nms")
      AttributeError: module 'torch.library' has no attribute 'register_fake'

    Fix: install matching versions (see below), OR let this patch handle it.
    Matching version pairs:
      torch 2.0.x  → torchvision 0.15.x
      torch 2.1.x  → torchvision 0.16.x
      torch 2.2.x  → torchvision 0.17.x
      torch 2.3.x  → torchvision 0.18.x
      torch 2.4.x  → torchvision 0.19.x
    """
    try:
        import torch
        if not hasattr(torch.library, 'register_fake'):
            # Provide a no-op decorator so torchvision imports cleanly
            def _noop_register_fake(op_name):
                def decorator(fn):
                    return fn
                return decorator
            torch.library.register_fake = _noop_register_fake
            logger.warning(
                "Applied torch.library.register_fake compatibility patch "
                f"(torch {torch.__version__} + torchvision version mismatch). "
                "To fix properly, run:\n"
                "  pip install --upgrade torch torchvision --index-url "
                "https://download.pytorch.org/whl/cu118  # (or cu121/cu124)"
            )
    except Exception:
        pass


def load_sam_predictor(checkpoint_path, model_type="vit_h", device="auto"):
    add_sam_to_sys_path()
    try:
        import torch
    except ImportError:
        logger.error("PyTorch (torch) not installed.")
        sys.exit(1)

    # Apply compatibility patch BEFORE importing torchvision (via segment_anything)
    _patch_torchvision_compat()

    try:
        from segment_anything import sam_model_registry, SamPredictor
    except AttributeError as e:
        logger.error(
            f"Import error: {e}\n"
            "This is a torch vs torchvision version mismatch.\n"
            "Fix with (choose your CUDA version):\n"
            "  pip install --upgrade torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu118\n"
            "  pip install --upgrade torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu121\n"
            "  pip install --upgrade torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu124"
        )
        sys.exit(1)
    except ImportError:
        logger.error("segment_anything not found. Add segment-anything-main to path.")
        sys.exit(1)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        script_dir = Path(__file__).resolve().parent
        for cand in [
            script_dir / checkpoint_path,
            script_dir / "segment-anything-main" / checkpoint_path.name,
            Path("/home/sxr23/ultralytics_quadbox/segment-anything-main") / checkpoint_path.name,
        ]:
            if cand.is_file():
                checkpoint_path = cand
                break
        else:
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading SAM ({model_type}) from {checkpoint_path}")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device=device)
    return SamPredictor(sam)


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def parse_label_file(label_path):
    """
    Parse any YOLO label format:
      - 5 tokens  → standard bbox: class cx cy w h
      - 9 tokens  → quadbox: class x1y1 x2y2 x3y3 x4y4
      - 2N+1 toks → polygon: class x1 y1 ... xN yN
    Returns list of dicts with bbox extents (x_min/max, y_min/max in 0..1).
    """
    results = []
    if not os.path.exists(label_path):
        return results
    with open(label_path, "r") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(float(parts[0]))
                coords = [float(p) for p in parts[1:]]

                if len(coords) == 4:
                    # standard bbox: cx cy w h
                    cx, cy, w, h = coords
                    x_min, x_max = cx - w / 2, cx + w / 2
                    y_min, y_max = cy - h / 2, cy + h / 2
                elif len(coords) >= 6 and len(coords) % 2 == 0:
                    # quadbox or polygon: x1 y1 x2 y2 ...
                    xs = coords[0::2]
                    ys = coords[1::2]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    cx = (x_min + x_max) / 2
                    cy = (y_min + y_max) / 2
                    w = x_max - x_min
                    h = y_max - y_min
                else:
                    logger.warning(f"Line {ln}: unsupported coord count {len(coords)}")
                    continue

                results.append(dict(
                    class_id=class_id,
                    cx=cx, cy=cy, w=w, h=h,
                    x_min=x_min, x_max=x_max,
                    y_min=y_min, y_max=y_max,
                ))
            except ValueError as e:
                logger.warning(f"Line {ln} parse error: {e}")
    return results


# ---------------------------------------------------------------------------
# SAM inference
# ---------------------------------------------------------------------------

def run_sam_on_crop(predictor, crop_rgb, box_in_crop, center_in_crop,
                    box_shrink=0.0, use_neg_points=True):
    """
    Run SAM inside a cropped patch.

    box_shrink   : shrink SAM box prompt inward by this fraction on each side.
                   E.g. 0.08 → 8% inward on all 4 sides. Prevents SAM from
                   grabbing the holder area above the plate.
    use_neg_points: Add negative SAM points at midpoints of top & side edges
                   to push SAM away from regions outside the actual plate.
    """
    import numpy as np

    x1, y1, x2, y2 = box_in_crop
    bw = x2 - x1
    bh = y2 - y1

    # Shrink box prompt inward so SAM focuses inside the plate
    pad_x = bw * box_shrink
    pad_y = bh * box_shrink
    sx1 = x1 + pad_x
    sy1 = y1 + pad_y
    sx2 = x2 - pad_x
    sy2 = y2 - pad_y

    # Guard against degenerate box after shrinking
    if sx2 - sx1 < 2 or sy2 - sy1 < 2:
        sx1, sy1, sx2, sy2 = x1, y1, x2, y2

    box_prompt = np.array([sx1, sy1, sx2, sy2], dtype=np.float32)

    # Positive: plate center
    pos_pts = [center_in_crop]
    pos_labels = [1]

    # Negative: midpoints of the 4 edges of the ORIGINAL (un-shrunk) box
    # → top-center, bottom-center, left-center, right-center
    # These lie just outside the actual plate and suppress the holder/mount area
    if use_neg_points:
        h_act, w_act = crop_rgb.shape[:2]
        neg_offset = max(4, int(min(bw, bh) * 0.08))  # a few pixels outside box

        neg_candidates = [
            # top-edge midpoint (most important: holder above plate)
            ((x1 + x2) / 2, max(0, y1 - neg_offset)),
            # bottom-edge midpoint
            ((x1 + x2) / 2, min(h_act - 1, y2 + neg_offset)),
            # left-edge midpoint
            (max(0, x1 - neg_offset), (y1 + y2) / 2),
            # right-edge midpoint
            (min(w_act - 1, x2 + neg_offset), (y1 + y2) / 2),
        ]
        for px, py in neg_candidates:
            pos_pts.append((px, py))
            pos_labels.append(0)  # 0 = negative

    point_coords = np.array(pos_pts, dtype=np.float32)
    point_labels = np.array(pos_labels, dtype=np.int32)

    predictor.set_image(crop_rgb)
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box_prompt,
        multimask_output=True,
    )
    best = int(scores.argmax())
    return masks[best], float(scores[best])


def refine_mask_white_filter(mask, crop_rgb, white_threshold=180):
    """
    Post-process SAM mask:
    Intersect SAM mask with the brightest (white/light) pixels in the crop.
    Keeps only the largest connected component that is both:
      - inside SAM mask
      - sufficiently bright (white plate surface)

    This eliminates the dark motorcycle body / holder that may leak in.
    """
    import cv2
    import numpy as np

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    _, white_mask = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)

    # Intersection: bright pixels AND SAM mask
    combined = (mask.astype(np.uint8) * 255) & white_mask

    # Find largest connected component in intersection
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    if num_labels <= 1:
        # Fallback: return original SAM mask if no bright region found
        return mask

    # Ignore background (label 0), pick largest foreground component
    largest_label = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    refined = (labels == largest_label).astype(bool)

    # Safety: if refined is too small vs original, fall back to SAM mask
    if refined.sum() < mask.sum() * 0.1:
        return mask

    return refined


def erode_mask(mask, erode_px):
    """
    Erode (shrink inward) the binary SAM mask by `erode_px` pixels.

    WHY: SAM mask edges tend to bleed 1-3 pixels outside the actual plate edge.
    Eroding the mask pulls the contour inward so that when we fit the QuadBox,
    all 4 corners land INSIDE the plate boundary — not on the holder/frame.

    erode_px = 2  → safe default, removes ~2px bleeding on each side
    erode_px = 4  → use when corners still go outside
    erode_px = 0  → disabled
    """
    import cv2
    import numpy as np

    if erode_px <= 0:
        return mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1)
    )
    m8 = (mask.astype(np.uint8)) * 255
    eroded = cv2.erode(m8, kernel, iterations=1)
    result = eroded > 0
    # Safety: if erosion killed the mask entirely, return original
    if result.sum() < mask.sum() * 0.1:
        return mask
    return result


# ---------------------------------------------------------------------------
# Contour extraction
# ---------------------------------------------------------------------------

def order_quad_points(pts):
    """
    Clockwise order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    Uses sum/diff trick — no bow-tie / X-shape bugs on rotated plates.
    """
    import numpy as np
    pts = pts.astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[s.argmin()]   # TL
    rect[2] = pts[s.argmax()]   # BR
    d = np.diff(pts, axis=1)
    rect[1] = pts[d.argmin()]   # TR
    rect[3] = pts[d.argmax()]   # BL
    return rect


def mask_to_quadbox(mask):
    """Fit a 4-corner quadrilateral to the mask contour."""
    import cv2
    import numpy as np

    m8 = (mask.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 1.0:
        return None

    hull = cv2.convexHull(cnt)
    eps = 0.015 * cv2.arcLength(hull, True)
    approx = None
    for _ in range(30):
        approx = cv2.approxPolyDP(hull, eps, True)
        if len(approx) == 4:
            break
        eps *= (0.8 if len(approx) < 4 else 1.2)

    if approx is None or len(approx) != 4:
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect).astype(np.float32)
    else:
        box = approx.reshape(-1, 2).astype(np.float32)

    return order_quad_points(box)


def mask_to_polygon(mask, simplify_eps=0.001):
    """Extract simplified polygon contour from mask."""
    import cv2

    m8 = (mask.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 1.0:
        return None
    if simplify_eps > 0:
        eps = simplify_eps * cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, eps, True)
    if len(cnt) < 3:
        return None
    return [pt[0] for pt in cnt]


# ---------------------------------------------------------------------------
# Image lookup
# ---------------------------------------------------------------------------

def find_image(stem, images_dir):
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif',
                '.JPG', '.JPEG', '.PNG']:
        p = images_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_dataset(
    images_dir, labels_dir, output_dir,
    sam_checkpoint, model_type="vit_h", device="auto",
    out_format="quadbox",
    crop_scale=2.5,
    box_shrink=0.08,
    use_neg_points=True,
    white_filter=True,
    white_threshold=180,
    mask_erode=0,
    save_crops=True,
    crops_dir=None,
    simplify_eps=0.001,
):
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.error("OpenCV not installed: pip install opencv-python")
        sys.exit(1)

    images_dir  = Path(images_dir)
    labels_dir  = Path(labels_dir)
    output_dir  = Path(output_dir)
    if crops_dir is None:
        crops_dir = output_dir / "crops"
    crops_dir = Path(crops_dir)

    for d in [output_dir]:
        d.mkdir(parents=True, exist_ok=True)
    if save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Images  : {images_dir}")
    logger.info(f"Labels  : {labels_dir}")
    logger.info(f"Output  : {output_dir}")
    logger.info(f"Format  : {out_format}")
    logger.info(f"Crop ROI: {crop_scale*100:.0f}%  BoxShrink: {box_shrink*100:.0f}%  "
                f"NegPoints: {use_neg_points}  WhiteFilter: {white_filter}  "
                f"MaskErode: {'OFF' if mask_erode == 0 else str(mask_erode)+'px'}")

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        logger.warning(f"No .txt files in {labels_dir}")
        return

    logger.info(f"Found {len(label_files)} label files")
    predictor = load_sam_predictor(sam_checkpoint, model_type, device)

    n_imgs = n_boxes = n_masks = n_skip = 0

    for fi, lf in enumerate(label_files, 1):
        stem = lf.stem
        img_path = find_image(stem, images_dir)
        if img_path is None:
            logger.warning(f"[{fi}] No image for {lf.name}")
            n_skip += 1
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            logger.warning(f"[{fi}] Cannot read {img_path.name}")
            n_skip += 1
            continue

        H, W = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        labels = parse_label_file(lf)
        out_lines = []

        for li, lb in enumerate(labels, 1):
            n_boxes += 1
            cid = lb['class_id']

            # Pixel bbox
            x1p = lb['x_min'] * W
            y1p = lb['y_min'] * H
            x2p = lb['x_max'] * W
            y2p = lb['y_max'] * H
            bw = max(1.0, x2p - x1p)
            bh = max(1.0, y2p - y1p)
            cx = (x1p + x2p) / 2
            cy = (y1p + y2p) / 2

            # Crop ROI (crop_scale × plate size)
            cw = bw * crop_scale
            ch = bh * crop_scale
            rx1 = max(0, int(round(cx - cw / 2)))
            ry1 = max(0, int(round(cy - ch / 2)))
            rx2 = min(W, int(round(cx + cw / 2)))
            ry2 = min(H, int(round(cy + ch / 2)))

            crop = img_rgb[ry1:ry2, rx1:rx2]
            ch_act, cw_act = crop.shape[:2]
            if ch_act < 4 or cw_act < 4:
                logger.warning(f"  [{fi}/{len(label_files)}] crop too small, skipping object {li}")
                continue

            # LP box & center relative to crop
            def clamp(v, lo, hi): return max(lo, min(hi, v))
            bx1 = clamp(x1p - rx1, 0, cw_act - 1)
            by1 = clamp(y1p - ry1, 0, ch_act - 1)
            bx2 = clamp(x2p - rx1, 0, cw_act - 1)
            by2 = clamp(y2p - ry1, 0, ch_act - 1)
            bcx = clamp(cx - rx1, 0, cw_act - 1)
            bcy = clamp(cy - ry1, 0, ch_act - 1)

            try:
                mask, score = run_sam_on_crop(
                    predictor, crop,
                    box_in_crop=(bx1, by1, bx2, by2),
                    center_in_crop=(bcx, bcy),
                    box_shrink=box_shrink,
                    use_neg_points=use_neg_points,
                )
            except Exception as e:
                logger.error(f"  SAM error on {img_path.name} obj {li}: {e}")
                continue

            # Optional: white-plate brightness filter
            if white_filter:
                mask = refine_mask_white_filter(mask, crop, white_threshold)

            # Erode mask inward to pull corners inside the plate boundary
            # This is the key fix for corners going slightly outside the plate
            if mask_erode > 0:
                mask = erode_mask(mask, mask_erode)

            # Extract coordinates
            coords = []
            pts_crop = None

            if out_format == "quadbox":
                pts_crop = mask_to_quadbox(mask)
                if pts_crop is None:
                    pts_crop = np.array([[bx1,by1],[bx2,by1],[bx2,by2],[bx1,by2]], dtype=np.float32)
                pts_crop = order_quad_points(pts_crop)

                # Map crop-space corners back to full image space and normalize to 0..1
                # We do NOT clip to the input bbox — SAM mask defines the plate boundary.
                # Clipping to bbox caused corners to go INSIDE the plate on tight labels.
                for px, py in pts_crop:
                    coords += [
                        round(clamp((px + rx1) / W, 0, 1), 6),
                        round(clamp((py + ry1) / H, 0, 1), 6),
                    ]

            elif out_format == "bbox":
                m8 = (mask.astype(np.uint8)) * 255
                cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    bx, by, bww, bhh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
                    coords = [
                        round(clamp((bx + rx1 + bww/2) / W, 0, 1), 6),
                        round(clamp((by + ry1 + bhh/2) / H, 0, 1), 6),
                        round(clamp(bww / W, 0, 1), 6),
                        round(clamp(bhh / H, 0, 1), 6),
                    ]
                else:
                    coords = [lb['cx'], lb['cy'], lb['w'], lb['h']]

            else:  # polygon
                pts_crop = mask_to_polygon(mask, simplify_eps)
                if pts_crop:
                    for px, py in pts_crop:
                        coords += [
                            round(clamp((px + rx1) / W, 0, 1), 6),
                            round(clamp((py + ry1) / H, 0, 1), 6),
                        ]

            if not coords:
                logger.warning(f"  [{fi}] No coords for obj {li} in {img_path.name}")
                continue

            out_lines.append(f"{cid} " + " ".join(str(c) for c in coords))
            n_masks += 1

            # Save crop visualization
            if save_crops:
                vis = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                # Draw SAM mask overlay
                mask_colored = np.zeros_like(vis)
                mask_colored[mask] = (0, 120, 255)  # orange tint for mask
                vis = cv2.addWeighted(vis, 0.75, mask_colored, 0.25, 0)

                if pts_crop is not None and out_format == "quadbox":
                    q = pts_crop.astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(vis, [q], True, (0, 255, 0), 2)
                    for pt in pts_crop:
                        cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
                elif pts_crop is not None and out_format == "polygon":
                    poly = np.array(pts_crop, np.int32).reshape(-1, 1, 2)
                    cv2.polylines(vis, [poly], True, (0, 255, 0), 2)

                cv2.imwrite(str(crops_dir / f"{stem}_obj{li}.jpg"), vis)

        out_txt = output_dir / lf.name
        with open(out_txt, "w") as f:
            f.write("\n".join(out_lines) + ("\n" if out_lines else ""))

        n_imgs += 1
        logger.info(f"[{fi}/{len(label_files)}] {img_path.name} → {len(out_lines)}/{len(labels)} labels written")

    logger.info("=" * 60)
    logger.info(f"Done.  Images: {n_imgs}  Boxes: {n_boxes}  Masks: {n_masks}  Skipped: {n_skip}")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="SAM LP Segmentation — handles loose bboxes, holders, rotated plates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("-i", "--images",  default="images",      help="Images folder")
    ap.add_argument("-l", "--labels",  default="labels_bbox", help="Input label folder (any YOLO format)")
    ap.add_argument("-o", "--output",  default="labels_seg",  help="Output label folder")
    ap.add_argument("-f", "--format",  default="quadbox",
                    choices=["quadbox", "bbox", "polygon", "segmentation"],
                    help="Output label format")
    ap.add_argument("-c", "--checkpoint",
                    default="segment-anything-main/sam_vit_h_4b8939.pth",
                    help="SAM checkpoint path")
    ap.add_argument("-m", "--model-type", default="vit_h",
                    choices=["vit_h", "vit_l", "vit_b"])
    ap.add_argument("-d", "--device",  default="auto",
                    help="cuda / cpu / auto")

    # Crop ROI
    ap.add_argument("-cs","--crop-scale",   type=float, default=2.5,
                    help="ROI crop scale around LP bbox (2.5 = 250 percent)")

    # Tight-fit options  ← KEY FIXES for the loose-bbox / holder problem
    ap.add_argument("-bs","--box-shrink",   type=float, default=0.08,
                    help="Shrink SAM box prompt inward by this fraction. "
                         "Use 0.08 to 0.20 range (NOT 10 or 20 — those are fractions, not percent!). "
                         "E.g. -bs 0.10 = 10 percent inward shrink. "
                         "Prevents SAM from grabbing holder/mount above the plate.")
    ap.add_argument("--no-neg-points", action="store_false", dest="use_neg_points",
                    help="Disable automatic negative SAM points at bbox edges")
    ap.add_argument("-nwf","--no-white-filter", action="store_false", dest="white_filter",
                    help="Disable white-brightness post-filter")
    ap.add_argument("-wt","--white-threshold", type=int, default=180,
                    help="Pixel brightness threshold for white-filter (0-255)")
    ap.add_argument("-me","--mask-erode", type=int, default=0,
                    help="Erode SAM mask inward by N pixels before fitting QuadBox. "
                         "Default 0 (OFF). Use only if corners go outside the plate "
                         "(e.g. -me 2). Do NOT use if corners are already inside the plate.")

    # Crop image saving
    ap.add_argument("-sc","--save-crops",   action="store_true", default=True)
    ap.add_argument("-nsv","--no-save-crops", action="store_false", dest="save_crops")
    ap.add_argument("-cd","--crops-dir", "--crop-dir", dest="crops_dir", default=None,
                    help="Folder for crop visualisation images (default: <output>/crops)")

    ap.add_argument("--simplify-epsilon", type=float, default=0.001,
                    help="Polygon simplification epsilon (0 = off)")

    args = ap.parse_args()
    fmt = "polygon" if args.format == "segmentation" else args.format

    # Auto-fix: if user passed -bs 10 (meaning 10%) instead of -bs 0.10
    # clamp box_shrink to valid range [0.0, 0.5]
    if args.box_shrink > 0.5:
        original_bs = args.box_shrink
        args.box_shrink = min(args.box_shrink / 100.0, 0.5)
        logger.warning(
            f"--box-shrink value {original_bs} is too large (>0.5). "
            f"Treating as percentage: {original_bs:.0f}% → {args.box_shrink:.2f}. "
            "Use 0.08 to 0.20 for best results (e.g. -bs 0.10 for 10%)."
        )

    # Auto-fix: if crop_scale > 10, likely entered as percent (e.g. 120 instead of 1.2)
    if args.crop_scale > 10:
        original_cs = args.crop_scale
        args.crop_scale = args.crop_scale / 100.0
        logger.warning(
            f"--crop-scale value {original_cs} is very large. "
            f"Treating as percentage: {original_cs:.0f}% → {args.crop_scale:.2f}. "
            "Use 1.2 to 3.0 range (e.g. -cs 1.2 for 120%)."
        )

    process_dataset(
        images_dir=args.images,
        labels_dir=args.labels,
        output_dir=args.output,
        sam_checkpoint=args.checkpoint,
        model_type=args.model_type,
        device=args.device,
        out_format=fmt,
        crop_scale=args.crop_scale,
        box_shrink=args.box_shrink,
        use_neg_points=args.use_neg_points,
        white_filter=args.white_filter,
        white_threshold=args.white_threshold,
        mask_erode=args.mask_erode,
        save_crops=args.save_crops,
        crops_dir=args.crops_dir,
        simplify_eps=args.simplify_epsilon,
    )


if __name__ == "__main__":
    main()
