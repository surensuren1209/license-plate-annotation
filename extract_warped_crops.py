#!/usr/bin/env python3
import os
import sys
import argparse
import cv2
import numpy as np

def order_points(pts):
    """
    Orders 4 coordinates clockwise starting from Top-Left:
    [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp_quadrilateral(image, pts):
    """
    Performs perspective warping on a 4-point quadrilateral to produce
    a flat, deskewed rectangular crop.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Calculate widths and heights
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Destination points for top-down view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
        
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def parse_yolo_segmentation(label_path, img_width, img_height):
    """
    Parses YOLO label files (supports both segmentation polygons and standard bounding boxes).
    """
    annotations = []
    if not os.path.exists(label_path):
        return annotations
        
    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
            
        label = parts[0]
        if len(parts) == 5:
            # Standard YOLO cx, cy, w, h format
            try:
                cx, cy, w, h = map(float, parts[1:5])
            except ValueError:
                continue
            x1 = (cx - w / 2.0) * img_width
            y1 = (cy - h / 2.0) * img_height
            x2 = (cx + w / 2.0) * img_width
            y2 = (cy - h / 2.0) * img_height
            x3 = (cx + w / 2.0) * img_width
            y3 = (cy + h / 2.0) * img_height
            x4 = (cx - w / 2.0) * img_width
            y4 = (cy + h / 2.0) * img_height
            points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            annotations.append({"label": label, "points": points})
        elif len(parts) >= 7:
            # YOLO segmentation / polygon format (x1 y1 x2 y2 ... xN yN)
            try:
                coords = list(map(float, parts[1:]))
            except ValueError:
                continue
                
            points = []
            for i in range(0, len(coords), 2):
                px = coords[i] * img_width
                py = coords[i+1] * img_height
                points.append((px, py))
                
            annotations.append({"label": label, "points": points})
            
    return annotations

def process_file(image_path, label_path, output_dir):
    """
    Processes a single image-label pair to extract and save crops.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
        
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to read image: {image_path}")
        return
        
    h, w = image.shape[:2]
    annotations = parse_yolo_segmentation(label_path, w, h)
    
    if not annotations:
        print(f"No valid 4-point polygon annotations found in: {label_path}")
        return
        
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, ann in enumerate(annotations):
        pts = ann["points"]
        label = ann["label"]
        
        # Enforce 4 points for perspective warp
        if len(pts) != 4:
            print(f"Skipping annotation {idx} in {os.path.basename(label_path)}: Has {len(pts)} points instead of 4.")
            continue
            
        pts_np = np.array(pts, dtype="float32")
        warped = warp_quadrilateral(image, pts_np)
        
        # Save output crop
        out_filename = f"{img_name}_warp_{idx}_{label}.png"
        out_path = os.path.join(output_dir, out_filename)
        cv2.imwrite(out_path, warped)
        print(f"Saved warped crop: {out_path}")

def process_directory(parent_dir):
    """
    Scans the directory for image-label pairs.
    Handles:
    1. Nested structures: "crops" folder mapped to sibling "labels" folder.
    2. Flat structures: image and label file with same name side-by-side in any directory.
    """
    supported_imgs = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    count = 0
    
    for root, _, filenames in os.walk(parent_dir):
        # Check if the current root directory is inside a "crops" folder structure
        in_crops = False
        crops_root = None
        current = root
        while current and current != os.path.dirname(parent_dir):
            base = os.path.basename(current)
            if base.lower() == "crops":
                in_crops = True
                crops_root = current
                break
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
            
        if in_crops:
            crops_parent = os.path.dirname(crops_root)
            labels_dir = None
            try:
                siblings = os.listdir(crops_parent)
            except OSError:
                siblings = []
            for sibling in siblings:
                if sibling.lower() == "labels":
                    sibling_path = os.path.join(crops_parent, sibling)
                    if os.path.isdir(sibling_path):
                        labels_dir = sibling_path
                        break
            if not labels_dir:
                labels_dir = os.path.join(crops_parent, "labels")
                
            output_dir = os.path.join(crops_parent, "warped_crops")
            
            for f in filenames:
                ext = os.path.splitext(f)[1].lower()
                if ext in supported_imgs:
                    img_path = os.path.join(root, f)
                    rel_path = os.path.relpath(img_path, crops_root)
                    rel_no_ext = os.path.splitext(rel_path)[0]
                    label_path = os.path.join(labels_dir, rel_no_ext + ".txt")
                    
                    if os.path.exists(label_path):
                        process_file(img_path, label_path, output_dir)
                        count += 1
        else:
            output_dir = os.path.join(root, "warped_crops")
            for f in filenames:
                ext = os.path.splitext(f)[1].lower()
                if ext in supported_imgs:
                    img_path = os.path.join(root, f)
                    label_path = os.path.splitext(img_path)[0] + ".txt"
                    if os.path.exists(label_path):
                        process_file(img_path, label_path, output_dir)
                        count += 1
                        
    print(f"\nCompleted! Processed {count} image-label sets.")

def process_directories_pair(image_dir, label_dir, output_dir):
    """
    Processes a pair of directories: matches images from image_dir
    with labels from label_dir using relative path name.
    """
    supported_imgs = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    count = 0
    
    for root, _, filenames in os.walk(image_dir):
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in supported_imgs:
                img_path = os.path.join(root, f)
                
                # Find matching label in label_dir
                rel_path = os.path.relpath(img_path, image_dir)
                rel_no_ext = os.path.splitext(rel_path)[0]
                label_path = os.path.join(label_dir, rel_no_ext + ".txt")
                
                if os.path.exists(label_path):
                    process_file(img_path, label_path, output_dir)
                    count += 1
                else:
                    print(f"Skipping: No label file found for {img_path} at {label_path}")
                    
    print(f"\nCompleted! Processed {count} image-label sets from directories pair.")

def main():
    parser = argparse.ArgumentParser(description="Perspective Warped Crop Extractor for YOLO Segmentation Labels")
    parser.add_argument("-d", "--input_dir", type=str, help="Parent folder path to scan recursively (looks for crops/ and labels/ subfolders)")
    parser.add_argument("-i", "--image", type=str, help="Specific image file path or directory containing images to process")
    parser.add_argument("-l", "--label", type=str, help="Specific YOLO segment label file path or directory containing label files to process")
    parser.add_argument("-o", "--output_dir", type=str, default="warped_crops", help="Output directory to save warped crops")
    
    args = parser.parse_args()
    
    if args.image and args.label:
        is_img_dir = os.path.isdir(args.image)
        is_lbl_dir = os.path.isdir(args.label)
        
        if is_img_dir and is_lbl_dir:
            process_directories_pair(args.image, args.label, args.output_dir)
        elif not is_img_dir and not is_lbl_dir:
            process_file(args.image, args.label, args.output_dir)
        else:
            print("Error: Both -i/--image and -l/--label must be directories, or both must be files.")
            sys.exit(1)
    elif args.input_dir:
        process_directory(args.input_dir)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
