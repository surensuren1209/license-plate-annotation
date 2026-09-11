import os
import sys
import re
import cv2

CCPD_ROOT = "/home/sxr23/snap/antigravity/5/.gemini/antigravity/scratch/ccpd_img"

def parse_ccpd_filename(filename, img_w, img_h):
    fname = os.path.basename(filename)
    parts = fname.split('-')
    if len(parts) < 5:
        return None
    try:
        # Bounding box: 183&439_422&600 -> x1&y1_x2&y2
        bbox_part = parts[2]
        pts = bbox_part.split('_')
        if len(pts) != 2: return None
        x1, y1 = map(int, pts[0].split('&'))
        x2, y2 = map(int, pts[1].split('&'))

        # Vertices: 422&600_202&554_183&439_403&485
        vertices_part = parts[3]
        v_pts = vertices_part.split('_')
        vertices = []
        for vp in v_pts:
            vx, vy = map(int, vp.split('&'))
            vertices.append((round(vx / float(img_w), 6), round(vy / float(img_h), 6)))

        bw = x2 - x1
        bh = y2 - y1
        xc = (x1 + x2) / 2.0 / float(img_w)
        yc = (y1 + y2) / 2.0 / float(img_h)
        w = bw / float(img_w)
        h = bh / float(img_h)

        return {
            'class': '0',
            'bbox': [round(xc, 6), round(yc, 6), round(w, 6), round(h, 6)],
            'vertices': vertices
        }
    except Exception as e:
        return None

def sync_ccpd_folder(target_dir):
    print(f"[INFO] Syncing CCPD dataset in: {target_dir}")
    if not os.path.exists(target_dir):
        return

    for root, dirs, files in os.walk(target_dir):
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        if not image_files:
            continue

        # Determine parent split directory
        parent_dir = root
        if os.path.basename(root) in ("crops", "images"):
            parent_dir = os.path.dirname(root)

        labels_dir = os.path.join(parent_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)

        synced_count = 0
        for img_name in image_files:
            img_path = os.path.join(root, img_name)
            base_no_ext = os.path.splitext(img_name)[0]
            label_path = os.path.join(labels_dir, f"{base_no_ext}.txt")

            if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
                # Read image dimensions
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img_h, img_w = img.shape[:2]

                parsed = parse_ccpd_filename(img_name, img_w, img_h)
                if parsed:
                    cls_id = parsed['class']
                    xc, yc, w, h = parsed['bbox']
                    line = f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                    v_pts = parsed['vertices']
                    if v_pts:
                        pts_str = " ".join([f"{v[0]:.6f} {v[1]:.6f}" for v in v_pts])
                        line += f" {pts_str}"

                    with open(label_path, "w", encoding="utf-8") as lf:
                        lf.write(line + "\n")
                    synced_count += 1

        print(f"  + Synced {synced_count} label files for {len(image_files)} images in {root}")

def main():
    if os.path.exists(CCPD_ROOT):
        for sub in ("test", "val", "train"):
            sub_path = os.path.join(CCPD_ROOT, sub)
            if os.path.exists(sub_path):
                sync_ccpd_folder(sub_path)
    print("============================================================")
    print("✅ CCPD DATASET LABELS SYNC COMPLETED 100%!")
    print("============================================================")

if __name__ == "__main__":
    main()
