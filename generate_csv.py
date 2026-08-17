#!/usr/bin/env python3
import os
import sys
import argparse
import csv
import subprocess

def parse_sftp_url(url):
    if not url.startswith("sftp://"):
        return None, None
    url_nopref = url[7:]
    parts = url_nopref.split("/", 1)
    if len(parts) != 2:
        return url_nopref, "/"
    user_host, path = parts
    if not path.startswith("/"):
        path = "/" + path
    return user_host, path

def list_remote_files(user_host, remote_dir):
    cmd = ["ssh", "-o", "BatchMode=yes", user_host, f"find '{remote_dir}' -type f"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        if res.returncode == 0:
            lines = res.stdout.strip().split("\n")
            return [l.strip() for l in lines if l.strip()]
        else:
            print(f"SSH list failed on {user_host} (Exit Code: {res.returncode})")
            print(f"Stderr: {res.stderr.strip()}")
    except Exception as e:
        print(f"Error executing SSH command: {e}")
    return []

def process_local_directories_pair(image_dir, label_dir):
    supported_imgs = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    rows = []
    for root, _, filenames in os.walk(image_dir):
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in supported_imgs:
                img_path = os.path.join(root, f)
                rel_path = os.path.relpath(img_path, image_dir)
                rel_no_ext = os.path.splitext(rel_path)[0]
                
                label_path = None
                for label_ext in (".txt", ".xml", ".json", ".csv"):
                    p = os.path.join(label_dir, rel_no_ext + label_ext)
                    if os.path.exists(p):
                        label_path = p
                        break
                        
                if label_path:
                    rows.append([
                        os.path.abspath(img_path),
                        os.path.abspath(label_path)
                    ])
    return rows

def process_remote_directories_pair(image_url, label_url):
    user_host_img, path_img = parse_sftp_url(image_url)
    user_host_lbl, path_lbl = parse_sftp_url(label_url)
    rows = []
    
    print("Listing remote images directory...")
    img_files = list_remote_files(user_host_img, path_img)
    print(f"Found {len(img_files)} remote images.")
    
    print("Listing remote labels directory...")
    lbl_files = list_remote_files(user_host_lbl, path_lbl)
    print(f"Found {len(lbl_files)} remote labels.")
    
    lbl_lookup = {}
    for l_path in lbl_files:
        rel = os.path.relpath(l_path, path_lbl)
        rel_no_ext = os.path.splitext(rel)[0].lower()
        lbl_lookup[rel_no_ext] = l_path
        
    supported_imgs = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    for img_path in img_files:
        ext = os.path.splitext(img_path)[1].lower()
        if ext in supported_imgs:
            rel = os.path.relpath(img_path, path_img)
            rel_no_ext = os.path.splitext(rel)[0].lower()
            
            label_path = lbl_lookup.get(rel_no_ext)
            if label_path:
                rel_img = os.path.relpath(img_path, path_img)
                rel_lbl = os.path.relpath(label_path, path_lbl)
                
                full_img_url = image_url.rstrip("/") + "/" + rel_img
                full_lbl_url = label_url.rstrip("/") + "/" + rel_lbl
                
                rows.append([
                    full_img_url,
                    full_lbl_url
                ])
    return rows

def main():
    parser = argparse.ArgumentParser(description="Global CSV Dataset Generator from Crops & Labels")
    parser.add_argument("-d", "--input_dir", type=str, help="Parent folder path containing both crops/ and labels/ subfolders to scan recursively")
    parser.add_argument("-i", "--image_dir", type=str, help="Directory containing crop/image files (local path or sftp:// URL)")
    parser.add_argument("-l", "--label_dir", type=str, help="Directory containing matching label files (local path or sftp:// URL)")
    parser.add_argument("-o", "--output_file", type=str, default="dataset_annotations.csv", help="Output CSV filepath (or output directory path)")
    
    args = parser.parse_args()
    
    if args.output_file and os.path.isdir(args.output_file):
        args.output_file = os.path.join(args.output_file, "dataset_annotations.csv")
        
    rows = []
    
    if args.image_dir and args.label_dir:
        is_remote = args.image_dir.startswith("sftp://") and args.label_dir.startswith("sftp://")
        if is_remote:
            rows = process_remote_directories_pair(args.image_dir, args.label_dir)
        else:
            rows = process_local_directories_pair(args.image_dir, args.label_dir)
            
    elif args.input_dir:
        supported_imgs = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for root, _, filenames in os.walk(args.input_dir):
            in_crops = False
            crops_root = None
            current = root
            while current and current != os.path.dirname(args.input_dir):
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
                    
                for f in filenames:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in supported_imgs:
                        img_path = os.path.join(root, f)
                        rel_path = os.path.relpath(img_path, crops_root)
                        rel_no_ext = os.path.splitext(rel_path)[0]
                        
                        label_path = None
                        for label_ext in (".txt", ".xml", ".json", ".csv"):
                            p = os.path.join(labels_dir, rel_no_ext + label_ext)
                            if os.path.exists(p):
                                label_path = p
                                break
                                
                        if label_path:
                            rows.append([
                                os.path.abspath(img_path),
                                os.path.abspath(label_path)
                            ])
            else:
                for f in filenames:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in supported_imgs:
                        img_path = os.path.join(root, f)
                        base_path = os.path.splitext(img_path)[0]
                        
                        label_path = None
                        for label_ext in (".txt", ".xml", ".json", ".csv"):
                            p = base_path + label_ext
                            if os.path.exists(p):
                                label_path = p
                                break
                                
                        if label_path:
                            rows.append([
                                os.path.abspath(img_path),
                                os.path.abspath(label_path)
                            ])
    else:
        print("Error: You must provide either -d/--input_dir OR both -i/--image_dir and -l/--label_dir.")
        parser.print_help()
        sys.exit(1)
        
    with open(args.output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label_path"])
        writer.writerows(rows)
        
    print(f"Generated CSV with {len(rows)} entries at: {args.output_file}")

if __name__ == "__main__":
    main()
