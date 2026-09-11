import os
import shutil
import argparse

def build_label_lookup(labels_dir, label_ext=None):
    """Build a dict of stem -> full label filename (single pass, fast)."""
    lookup = {}
    with os.scandir(labels_dir) as it:
        for entry in it:
            if not entry.is_file():
                continue
            name = entry.name
            stem, ext = os.path.splitext(name)
            if label_ext and ext != label_ext:
                continue
            lookup[stem] = name
    return lookup

def copy_batch(images_dir, labels_dir, output_dir, offset, batch_size, label_ext=None):
    out_images = os.path.join(output_dir, "images")
    out_labels = os.path.join(output_dir, "labels")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    print("Building label lookup...")
    label_lookup = build_label_lookup(labels_dir, label_ext)
    print(f"Found {len(label_lookup)} labels.")

    matched = 0   # count of valid pairs seen so far (for offset skipping)
    copied = 0

    print("Scanning images...")
    with os.scandir(images_dir) as it:
        # scandir gives arbitrary order; sort by name for reproducible batches
        entries = sorted(it, key=lambda e: e.name)

    for entry in entries:
        if copied >= batch_size:
            break
        if not entry.is_file():
            continue

        stem = os.path.splitext(entry.name)[0]
        label_name = label_lookup.get(stem)
        if label_name is None:
            continue  # no matching label, skip (doesn't count toward offset)

        matched += 1
        if matched <= offset:
            continue  # skip until we reach the offset

        img_src = os.path.join(images_dir, entry.name)
        lbl_src = os.path.join(labels_dir, label_name)
        shutil.copy2(img_src, os.path.join(out_images, entry.name))
        shutil.copy2(lbl_src, os.path.join(out_labels, label_name))
        copied += 1

    print(f"Copied {copied} pairs (offset={offset}, batch_size={batch_size}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy batches of matching image/label pairs.")
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--labels_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--offset", type=int, default=0, help="Number of matched pairs to skip")
    parser.add_argument("--batch_size", type=int, default=10000, help="Number of pairs to copy")
    parser.add_argument("--label_ext", default=None, help="e.g. .txt (optional, speeds up matching)")
    args = parser.parse_args()

    copy_batch(
        args.images_dir,
        args.labels_dir,
        args.output_dir,
        args.offset,
        args.batch_size,
        args.label_ext,
    )
