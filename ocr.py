#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Annotation Editor
======================

A professional desktop application for REVIEWING, VERIFYING and CORRECTING
existing OCR text-extraction results (text + bounding box) that were produced
by an external / upstream OCR-detection system.

IMPORTANT - SCOPE OF THIS APPLICATION
--------------------------------------
This application performs NO object detection and NO AI inference of any
kind. License-plate / text detection has already been carried out upstream.
This tool only loads the image + the already-existing annotation file,
displays the boxes, and lets a human operator correct the text and/or the
bounding-box geometry, then writes the corrected annotation back to disk in
the same file format it was read from (with optional JSON export).

Architecture (MVC / SOLID)
---------------------------
    Data / Domain layer   -> BoundingBox, Annotation                (Model data)
    Persistence layer     -> AnnotationParser (ABC) + concrete
                              parsers + ParserRegistry               (Model I/O - Open/Closed)
    Application state     -> AnnotationModel (QObject, signal bus)   (Model)
    Undo / Redo            -> QUndoCommand subclasses                (Command pattern)
    Presentation           -> BBoxGraphicsItem, ImageViewer,
                              PropertiesPanel, AnnotationListPanel,
                              FileNavigatorPanel                     (View)
    Orchestration           -> MainWindow                            (Controller)

Everything lives in a single file per project requirements, but the code is
internally partitioned exactly along these responsibilities so that, e.g., a
new annotation format can be added by writing one new AnnotationParser
subclass and registering it - no other class needs to change (Open/Closed
Principle). Similarly the parser layer knows nothing about Qt, and the Qt
widgets know nothing about file formats.

Dependencies: Python 3.10+, PyQt6, OpenCV-Python, NumPy
"""

from __future__ import annotations

import sys
import re
import os
import json
import uuid
import csv
import io
import traceback
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
from datetime import datetime

from PyQt6.QtCore import (
    Qt, QRect, QRectF, QPointF, QSizeF, QSize, pyqtSignal, QObject, QSettings,
    QEvent, QTimer, QModelIndex,
)
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QBrush, QColor, QFont, QCursor,
    QKeySequence, QAction, QUndoStack, QUndoCommand, QIcon, QWheelEvent,
    QDragEnterEvent, QDropEvent, QTransform, QMouseEvent, QPolygonF,
    QFontMetricsF,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QGraphicsPixmapItem, QGraphicsItem, QVBoxLayout,
    QHBoxLayout, QFormLayout, QLabel, QLineEdit, QDoubleSpinBox, QPushButton,
    QListWidget, QListWidgetItem, QToolBar, QStatusBar, QFileDialog,
    QMessageBox, QDockWidget, QCheckBox, QSplitter, QMenu, QSizePolicy,
    QAbstractItemView, QStyleOptionGraphicsItem, QToolButton, QFrame,
    QGraphicsSceneMouseEvent, QGraphicsSceneHoverEvent, QGroupBox,
    QComboBox, QSpinBox, QStyle, QDialog,
)


APP_NAME = "OCR Annotation Editor"
APP_VERSION = "1.10.0"
# 1.10.0 - Doubt marking & CSV persistence: Right-click any crop/image in
#          the left Files list to mark/unmark as Doubt with optional reason
#          notes or quick presets ("Blurry", "Cut-off Plate", "Wrong OCR",
#          etc.). All doubts automatically persist to `doubts.csv` with
#          filename, full path, label path, OCR text, reason, and timestamp.
#          Features visual orange doubt badges ([❓ DOUBT]), quick filter
#          toggle (All vs Doubts only), Ctrl+Shift+D shortcut, doubt
#          navigation (Ctrl+Alt+Left/Right), and one-click CSV export.
# 1.9.0 - Frame-to-frame copy-paste workflow (Ctrl+Shift+V / P): copy
#         previous frame boxes directly into empty or active frames with
#         new UUIDs and preserved metadata (label, text, class ID,
#         confidence, extra). Multi-box clipboard copy (Ctrl+C), paste
#         (Ctrl+V), and paste with custom X/Y offset dialog (Ctrl+Alt+V).
#         Smooth mouse-wheel zoom centered at cursor, Middle-drag and
#         Space+Left-drag panning with guaranteed bounding-box isolation,
#         and single-step undo/redo for all group moves and paste actions.
# 1.8.0 - Removed the Character Editor dock from the main UI, added a
#         toggleable auto-save feature, and made the window restore/resize
#         logic monitor-aware so it behaves consistently on multi-display
#         setups.
# 1.7.0 - Ctrl+A (Select All) now also opens a new "Edit All Text" dock:
#         every selected box's recognized text listed as one editable row
#         per box, in SEQUENTIAL READING ORDER (top-to-bottom rows,
#         left-to-right within a row) instead of file order, for a full
#         correction pass over a whole image's OCR output. "Apply
#         Corrections" commits every changed row as a single undo/redo
#         step; "Apply && Save" does the same and immediately writes the
#         annotation file. A Refresh button re-scans the current
#         selection (e.g. after narrowing it) without losing edits in
#         rows that are still shown.
# 1.6.0 - Multi-select: Shift/Ctrl-click a box to add/remove it from the
#         selection, Shift-drag an empty area to rubber-band select many
#         boxes at once, Ctrl+A selects all boxes on the image, Escape
#         clears the selection. Dragging any selected box now moves the
#         WHOLE selection together, and Delete removes every selected box
#         -- both as a single undo/redo step. New Character Editor dock:
#         shows the selected annotation's recognized text broken into one
#         editable cell per character (read AND write each glyph), a
#         per-character delete button, and a one-click "Strip Quote
#         Characters" cleanup for stray " ' ` characters OCR sometimes
#         injects around/inside recognized text.
# 1.5.0 - Files panel reworked: Prev/Next moved to the bottom with a
#         position counter ("3 / 18"); new Labels list mirroring the
#         Images list (shows each image's matched annotation file,
#         flags missing labels in red, click opens the pair); Left/Right
#         arrow keys added as Prev/Next shortcuts; new Save & Next
#         action (Ctrl+Enter) for the fast validation loop
# 1.4.0 - added YoloPlateOCRTextParser ('class cx cy w h' + 'OCR: <text>'
#         line pairs, whole-plate granularity); images/ + labels/ sibling
#         folder auto-matching; ReferenceZoomPanel (magnifier dock that
#         follows the selected box, live-updates during drag, wheel zoom,
#         click-to-reset); Help > Keyboard Shortcuts dialog (F1); fixed
#         unused PropertiesPanel.update_geometry_only being dead code
# 1.3.0 - added YoloCharOCRAnnotationParser (normalized 'class cx cy w h
#         conf' per-character format, OCR_CLASSES 0-9/A-Z); parser
#         load()/save() now take an optional image_size for
#         normalized-coordinate formats; content-sniffing added to
#         disambiguate parsers sharing an extension; fixed a Save-As bug
#         where extension ambiguity could override the user's explicitly
#         chosen format filter; fixed a signature mismatch that made
#         JSON saves raise TypeError
# 1.2.0 - added `--selftest` CLI mode (headless install/regression check)
# 1.1.0 - Help>About dialog, Width/Height spin-box floor, dead-code cleanup
APP_ORG = "LocalTools"
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ======================================================================
# 1. DOMAIN MODEL  (pure Python, no Qt / no I/O)
# ======================================================================

@dataclass
class BoundingBox:
    """Axis-aligned bounding box in image-pixel coordinates."""
    x: float
    y: float
    width: float
    height: float

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0

    def clone(self) -> "BoundingBox":
        return BoundingBox(self.x, self.y, self.width, self.height)

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.width, self.height)


@dataclass
class Annotation:
    """One OCR text region: label + recognized text + geometry."""
    id: str
    label: str
    text: str
    bbox: BoundingBox
    locked: bool = False
    # Anything format-specific we don't explicitly model (confidence,
    # original polygon points, custom attributes, ...) is preserved here so
    # round-tripping through the editor does not silently drop data.
    extra: Dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "Annotation":
        return Annotation(
            id=self.id,
            label=self.label,
            text=self.text,
            bbox=self.bbox.clone(),
            locked=self.locked,
            extra=dict(self.extra),
        )


def new_annotation_id() -> str:
    return uuid.uuid4().hex[:8]


# ======================================================================
# 2. PERSISTENCE LAYER  -  modular parser architecture
# ======================================================================
#
# To add a new annotation file format later:
#   1. Subclass AnnotationParser
#   2. Implement `extensions`, `name`, `load()`, `save()`
#   3. Call ParserRegistry.register(YourParser()) once, near the bottom of
#      this section.
# No other part of the application needs to change.
# ======================================================================

class AnnotationParseError(Exception):
    pass


class AnnotationParser(ABC):
    """Strategy interface for reading/writing one annotation file format."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human readable format name, shown in Save-As dialogs."""

    @property
    @abstractmethod
    def extensions(self) -> List[str]:
        """File extensions this parser handles, e.g. ['.json']."""

    @abstractmethod
    def load(self, path: str, image_size: Optional[Tuple[float, float]] = None) -> List[Annotation]:
        """
        Read `path` and return a list of Annotation objects.

        `image_size`, if given, is the paired image's (width, height) in
        pixels. Pixel-space formats (JSON, delimited text) ignore it;
        normalized-coordinate formats (e.g. YOLO-style) require it to
        convert into the editor's pixel-space BoundingBox.
        """

    @abstractmethod
    def save(self, path: str, annotations: List[Annotation],
              image_size: Optional[Tuple[float, float]] = None) -> None:
        """Write `annotations` back to `path` in this format. See `load()`
        for what `image_size` is for."""

    def sniff(self, path: str) -> bool:
        """
        Optional content-based detection, used only to disambiguate two
        registered parsers that claim the same file extension (e.g. two
        different '.txt' conventions). Return True if this parser
        recognizes `path`'s content. Default: not applicable.
        """
        return False

    def file_filter(self) -> str:
        pattern = " ".join(f"*{e}" for e in self.extensions)
        return f"{self.name} ({pattern})"


class JSONAnnotationParser(AnnotationParser):
    """
    Native JSON format:

    {
      "annotations": [
        {"id": "a1b2c3d4", "label": "plate", "text": "ABC1234",
         "bbox": {"x": 10.0, "y": 20.0, "width": 120.0, "height": 40.0},
         "locked": false, "extra": {}}
      ]
    }
    """

    @property
    def name(self) -> str:
        return "JSON Annotation"

    @property
    def extensions(self) -> List[str]:
        return [".json"]

    def load(self, path: str, image_size: Optional[Tuple[float, float]] = None) -> List[Annotation]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_list = data.get("annotations", data if isinstance(data, list) else [])
        results: List[Annotation] = []
        for item in raw_list:
            bbox_data = item.get("bbox", {})
            bbox = BoundingBox(
                x=float(bbox_data.get("x", 0)),
                y=float(bbox_data.get("y", 0)),
                width=float(bbox_data.get("width", 0)),
                height=float(bbox_data.get("height", 0)),
            )
            results.append(Annotation(
                id=str(item.get("id") or new_annotation_id()),
                label=str(item.get("label", "")),
                text=str(item.get("text", "")),
                bbox=bbox,
                locked=bool(item.get("locked", False)),
                extra=dict(item.get("extra", {})),
            ))
        return results

    def save(self, path: str, annotations: List[Annotation],
              image_size: Optional[Tuple[float, float]] = None) -> None:
        payload = {
            "annotations": [
                {
                    "id": a.id,
                    "label": a.label,
                    "text": a.text,
                    "bbox": {
                        "x": round(a.bbox.x, 3),
                        "y": round(a.bbox.y, 3),
                        "width": round(a.bbox.width, 3),
                        "height": round(a.bbox.height, 3),
                    },
                    "locked": a.locked,
                    "extra": a.extra,
                }
                for a in annotations
            ]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


class DelimitedTextAnnotationParser(AnnotationParser):
    """
    Simple, common OCR annotation line format (one region per line):

        label,x,y,width,height,text

    - Coordinates are top-left x/y plus width/height, in pixels.
    - `text` is everything after the 5th comma, so OCR text may itself
      contain commas.
    - Lines starting with '#' are treated as comments and ignored on load,
      preserved is NOT guaranteed (comments are not required by the format).
    """

    @property
    def name(self) -> str:
        return "Delimited Text Annotation"

    @property
    def extensions(self) -> List[str]:
        return [".txt", ".csv"]

    def sniff(self, path: str) -> bool:
        # Recognize our own 'label,x,y,w,h,text' shape (>=4 commas before
        # any trailing free-text). A YOLO-style line ('class cx cy w h
        # conf') has zero commas, so this cleanly rejects it.
        try:
            with open(path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    return line.count(",") >= 4
            return False
        except Exception:
            return False

    def load(self, path: str, image_size: Optional[Tuple[float, float]] = None) -> List[Annotation]:
        results: List[Annotation] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.rstrip("\n").rstrip("\r")
                if not line.strip() or line.strip().startswith("#"):
                    continue
                parts = line.split(",", 5)
                if len(parts) < 5:
                    raise AnnotationParseError(
                        f"{path}:{line_no}: expected at least 5 fields "
                        f"'label,x,y,width,height[,text]', got: {line!r}"
                    )
                label = parts[0].strip()
                try:
                    x, y, w, h = (float(p.strip()) for p in parts[1:5])
                except ValueError as e:
                    raise AnnotationParseError(f"{path}:{line_no}: bad numeric field ({e})")
                text = parts[5] if len(parts) > 5 else ""
                results.append(Annotation(
                    id=new_annotation_id(),
                    label=label,
                    text=text,
                    bbox=BoundingBox(x, y, w, h),
                ))
        return results

    def save(self, path: str, annotations: List[Annotation],
              image_size: Optional[Tuple[float, float]] = None) -> None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            for a in annotations:
                fields = [
                    a.label,
                    f"{a.bbox.x:.3f}",
                    f"{a.bbox.y:.3f}",
                    f"{a.bbox.width:.3f}",
                    f"{a.bbox.height:.3f}",
                    a.text,
                ]
                f.write(",".join(fields) + "\n")


class YoloCharOCRAnnotationParser(AnnotationParser):
    """
    Per-character OCR detection format (YOLO-style, normalized, one
    detected character per line):

        class_id cx cy width height confidence

    - `class_id` indexes into OCR_CLASSES (0-9 digits, then A-Z):
          0-9   -> '0'-'9'
          10-35 -> 'A'-'Z'
    - cx, cy, width, height are normalized to [0, 1] relative to the
      paired image's pixel width/height (YOLO convention: box CENTER +
      size, not top-left).
    - confidence is the upstream detector's per-character confidence
      score. It is preserved through edits in Annotation.extra and
      re-emitted on save; a box with no known confidence (e.g. one drawn
      new in the editor) is saved with confidence 1.0 (manually
      verified).

    Because coordinates are normalized, this format REQUIRES the paired
    image's pixel size to convert to/from the editor's pixel-space
    BoundingBox - both load() and save() raise AnnotationParseError if
    `image_size` isn't supplied (the app always supplies it when an image
    is open; see MainWindow._current_image_size()).

    On save, each annotation's `text` (case-insensitively, first
    character) must be one of OCR_CLASSES so it can be mapped back to a
    class_id. This format has no representation for missing/unknown text,
    so an empty or unrecognized text field raises AnnotationParseError
    rather than silently writing something else — fix the text or delete
    the box first.

    Shares the '.txt' extension with DelimitedTextAnnotationParser; see
    `sniff()` for how the two are told apart automatically.
    """

    OCR_CLASSES: List[str] = list("0123456789") + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    @property
    def name(self) -> str:
        return "YOLO Per-Character OCR (class cx cy w h conf)"

    @property
    def extensions(self) -> List[str]:
        return [".txt"]

    def sniff(self, path: str) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) != 6:
                        return False
                    class_id = int(parts[0])
                    cx, cy, w, h, _conf = (float(p) for p in parts[1:])
                    if not (0 <= class_id < len(self.OCR_CLASSES)):
                        return False
                    return all(0.0 <= v <= 1.0 for v in (cx, cy, w, h))
            return False
        except Exception:
            return False

    def load(self, path: str, image_size: Optional[Tuple[float, float]] = None) -> List[Annotation]:
        if not image_size or image_size[0] <= 0 or image_size[1] <= 0:
            raise AnnotationParseError(
                f"{path}: this format stores normalized coordinates and requires the "
                f"paired image's pixel dimensions to load; none were available. "
                f"Open the matching image first."
            )
        img_w, img_h = image_size
        results: List[Annotation] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 6:
                    raise AnnotationParseError(
                        f"{path}:{line_no}: expected 'class_id cx cy width height confidence' "
                        f"(6 whitespace-separated fields), got: {line!r}"
                    )
                try:
                    class_id = int(parts[0])
                    cx, cy, w, h, conf = (float(p) for p in parts[1:])
                except ValueError as e:
                    raise AnnotationParseError(f"{path}:{line_no}: bad numeric field ({e})")
                if not (0 <= class_id < len(self.OCR_CLASSES)):
                    raise AnnotationParseError(
                        f"{path}:{line_no}: class_id {class_id} out of range "
                        f"(expected 0-{len(self.OCR_CLASSES) - 1})"
                    )
                ch = self.OCR_CLASSES[class_id]
                px_w = w * img_w
                px_h = h * img_h
                px_x = cx * img_w - px_w / 2.0
                px_y = cy * img_h - px_h / 2.0
                results.append(Annotation(
                    id=new_annotation_id(),
                    label="character",
                    text=ch,
                    bbox=BoundingBox(px_x, px_y, px_w, px_h),
                    extra={"class_id": class_id, "confidence": conf},
                ))
        # Sort left-to-right by box center for a natural reading order in
        # the UI. Line order in the source file is detector output order,
        # not reading order, and carries no other meaning, so reordering
        # here loses nothing and reads far more sensibly in the Annotation
        # List panel.
        results.sort(key=lambda a: a.bbox.center_x)
        return results

    def save(self, path: str, annotations: List[Annotation],
              image_size: Optional[Tuple[float, float]] = None) -> None:
        if not image_size or image_size[0] <= 0 or image_size[1] <= 0:
            raise AnnotationParseError(
                f"{path}: this format stores normalized coordinates and requires the "
                f"paired image's pixel dimensions to save; none were available."
            )
        img_w, img_h = image_size
        lines = []
        for a in annotations:
            ch = (a.text or "").strip().upper()[:1]
            if ch and ch in self.OCR_CLASSES:
                class_id = self.OCR_CLASSES.index(ch)
            else:
                raise AnnotationParseError(
                    f"Annotation {a.id!r} has text {a.text!r}, which is not one of the "
                    f"{len(self.OCR_CLASSES)} supported OCR_CLASSES characters (0-9, A-Z). "
                    f"This format has no way to represent missing/unknown text -- fix the "
                    f"text field or delete this box before saving to this format."
                )
            cx = (a.bbox.x + a.bbox.width / 2.0) / img_w
            cy = (a.bbox.y + a.bbox.height / 2.0) / img_h
            w = a.bbox.width / img_w
            h = a.bbox.height / img_h
            conf = float(a.extra.get("confidence", 1.0))
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f}")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))


class YoloBoxAnnotationParser(AnnotationParser):
    """
    YOLO box-only format with normalized coordinates and no text/label:

        class_id cx cy width height

    This is useful for object-detection style label files that carry
    only class ID + normalized center/size coordinates. If the class ID
    maps into the OCR_CLASSES set, the parser preserves it as the
    annotation text; otherwise the class ID is stored as the label.
    """

    OCR_CLASSES: List[str] = list("0123456789") + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    @property
    def name(self) -> str:
        return "YOLO Box Only (class cx cy w h)"

    @property
    def extensions(self) -> List[str]:
        return [".txt"]

    def sniff(self, path: str) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        return False
                    class_id = int(float(parts[0]))
                    cx, cy, w, h = (float(p) for p in parts[1:])
                    return True
            return False
        except Exception:
            return False

    def load(self, path: str, image_size: Optional[Tuple[float, float]] = None) -> List[Annotation]:
        if not image_size or image_size[0] <= 0 or image_size[1] <= 0:
            raise AnnotationParseError(
                f"{path}: this format stores normalized coordinates and requires the "
                f"paired image's pixel dimensions to load; none were available. "
                f"Open the matching image first."
            )
        img_w, img_h = image_size
        results: List[Annotation] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 5:
                    raise AnnotationParseError(
                        f"{path}:{line_no}: expected 'class_id cx cy width height' "
                        f"(5 whitespace-separated fields), got: {line!r}"
                    )
                try:
                    class_id = int(float(parts[0]))
                    cx, cy, w, h = (float(p) for p in parts[1:])
                except ValueError as e:
                    raise AnnotationParseError(f"{path}:{line_no}: bad numeric field ({e})")
                px_w = w * img_w
                px_h = h * img_h
                px_x = cx * img_w - px_w / 2.0
                px_y = cy * img_h - px_h / 2.0
                if 0 <= class_id < len(self.OCR_CLASSES):
                    text = self.OCR_CLASSES[class_id]
                    label = "character"
                else:
                    text = ""
                    label = str(class_id)
                results.append(Annotation(
                    id=new_annotation_id(),
                    label=label,
                    text=text,
                    bbox=BoundingBox(px_x, px_y, px_w, px_h),
                    extra={"class_id": class_id},
                ))
        results.sort(key=lambda a: a.bbox.center_x)
        return results

    def save(self, path: str, annotations: List[Annotation],
              image_size: Optional[Tuple[float, float]] = None) -> None:
        if not image_size or image_size[0] <= 0 or image_size[1] <= 0:
            raise AnnotationParseError(
                f"{path}: this format stores normalized coordinates and requires the "
                f"paired image's pixel dimensions to save; none were available."
            )
        img_w, img_h = image_size
        lines = []
        for a in annotations:
            class_id = None
            if a.text and a.text.strip().upper()[:1] in self.OCR_CLASSES:
                class_id = self.OCR_CLASSES.index(a.text.strip().upper()[:1])
            elif "class_id" in a.extra:
                class_id = a.extra.get("class_id")
            elif a.label:
                try:
                    class_id = int(float(a.label))
                except ValueError:
                    class_id = 0
            try:
                class_id = int(float(class_id if class_id is not None else 0))
            except (TypeError, ValueError):
                class_id = 0
            cx = (a.bbox.x + a.bbox.width / 2.0) / img_w
            cy = (a.bbox.y + a.bbox.height / 2.0) / img_h
            w = a.bbox.width / img_w
            h = a.bbox.height / img_h
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))


class YoloPlateOCRTextParser(AnnotationParser):
    """
    Whole-plate YOLO-style box paired with a separate recognized-text
    line, one plate block per pair of lines:

        class_id cx cy width height
        OCR: <recognized text>

    - `class_id cx cy width height` (5 whitespace-separated fields, no
      confidence) is a single normalized [0, 1] bounding box in YOLO
      center+size convention, same as YoloCharOCRAnnotationParser but at
      whole-plate granularity instead of per-character, and with no
      trailing confidence value.
    - The following non-blank line must start with `OCR:` (case
      insensitive) and gives the recognized plate text verbatim after the
      prefix.
    - A file may contain multiple plate blocks (repeated box-line +
      OCR-line pairs) for multi-plate images; each pair becomes one
      Annotation.

    Example (`car_001.txt`)::

        0 0.512 0.431 0.182 0.064
        OCR: TN38AB1234

    Like YoloCharOCRAnnotationParser, this format stores normalized
    coordinates, so load()/save() require the paired image's pixel size
    (raises AnnotationParseError without it).

    Shares the '.txt' extension with the other two text-based parsers;
    `sniff()` distinguishes it by the presence of an `OCR:` line, which
    is unique to this format.
    """

    _OCR_LINE_RE = re.compile(r"^\s*OCR\s*:\s*(.*)$", re.IGNORECASE)
    _BOX_LINE_RE = re.compile(
        r"^\s*(-?\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*$"
    )

    def __init__(self) -> None:
        self._ocr_line_re = self._OCR_LINE_RE
        self._box_line_re = self._BOX_LINE_RE

    @property
    def name(self) -> str:
        return "YOLO Plate Box + OCR Text (txt)"

    @property
    def extensions(self) -> List[str]:
        return [".txt"]

    def sniff(self, path: str) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    if self._ocr_line_re.match(raw_line):
                        return True
            return False
        except Exception:
            return False

    def load(self, path: str, image_size: Optional[Tuple[float, float]] = None) -> List[Annotation]:
        if not image_size or image_size[0] <= 0 or image_size[1] <= 0:
            raise AnnotationParseError(
                f"{path}: this format stores normalized coordinates and requires the "
                f"paired image's pixel dimensions to load; none were available. "
                f"Open the matching image first."
            )
        img_w, img_h = image_size

        with open(path, "r", encoding="utf-8") as f:
            raw_lines = [l.rstrip("\n").rstrip("\r") for l in f]

        results: List[Annotation] = []
        pending_box: Optional[Tuple[int, float, float, float, float]] = None
        pending_line_no = 0

        for line_no, line in enumerate(raw_lines, start=1):
            if not line.strip() or line.strip().startswith("#"):
                continue

            ocr_match = self._ocr_line_re.match(line)
            if ocr_match:
                if pending_box is None:
                    raise AnnotationParseError(
                        f"{path}:{line_no}: found an 'OCR:' line with no preceding "
                        f"'class_id cx cy width height' box line."
                    )
                class_id, cx, cy, w, h = pending_box
                text = ocr_match.group(1).strip()
                px_w = w * img_w
                px_h = h * img_h
                px_x = cx * img_w - px_w / 2.0
                px_y = cy * img_h - px_h / 2.0
                results.append(Annotation(
                    id=new_annotation_id(),
                    label="plate",
                    text=text,
                    bbox=BoundingBox(px_x, px_y, px_w, px_h),
                    extra={"class_id": class_id},
                ))
                pending_box = None
                continue

            box_match = self._box_line_re.match(line)
            if box_match:
                if pending_box is not None:
                    raise AnnotationParseError(
                        f"{path}:{pending_line_no}: box line has no following 'OCR:' "
                        f"text line before the next box at line {line_no}."
                    )
                class_id = int(box_match.group(1))
                cx, cy, w, h = (float(box_match.group(i)) for i in (2, 3, 4, 5))
                pending_box = (class_id, cx, cy, w, h)
                pending_line_no = line_no
                continue

            raise AnnotationParseError(
                f"{path}:{line_no}: expected either 'class_id cx cy width height' or "
                f"'OCR: <text>', got: {line!r}"
            )

        if pending_box is not None:
            raise AnnotationParseError(
                f"{path}:{pending_line_no}: box line has no following 'OCR:' text line "
                f"(reached end of file)."
            )

        results.sort(key=lambda a: a.bbox.center_x)
        return results

    def save(self, path: str, annotations: List[Annotation],
              image_size: Optional[Tuple[float, float]] = None) -> None:
        if not image_size or image_size[0] <= 0 or image_size[1] <= 0:
            raise AnnotationParseError(
                f"{path}: this format stores normalized coordinates and requires the "
                f"paired image's pixel dimensions to save; none were available."
            )
        img_w, img_h = image_size
        lines = []
        for a in annotations:
            class_id = a.extra.get("class_id", 0)
            try:
                class_id = int(class_id)
            except (TypeError, ValueError):
                class_id = 0
            cx = (a.bbox.x + a.bbox.width / 2.0) / img_w
            cy = (a.bbox.y + a.bbox.height / 2.0) / img_h
            w = a.bbox.width / img_w
            h = a.bbox.height / img_h
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            lines.append(f"OCR: {a.text}")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))


class ParserRegistry:
    """Central lookup so the rest of the app never hard-codes a format."""

    _parsers: List[AnnotationParser] = []

    @classmethod
    def register(cls, parser: AnnotationParser) -> None:
        cls._parsers.append(parser)

    @classmethod
    def all_parsers(cls) -> List[AnnotationParser]:
        return list(cls._parsers)

    @classmethod
    def for_extension(cls, ext: str) -> Optional[AnnotationParser]:
        ext = ext.lower()
        for p in cls._parsers:
            if ext in p.extensions:
                return p
        return None

    @classmethod
    def for_path(cls, path: str) -> Optional[AnnotationParser]:
        ext = Path(path).suffix.lower()
        candidates = [p for p in cls._parsers if ext in p.extensions]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        # Multiple registered parsers share this extension (e.g. two
        # different '.txt' conventions) -- use content sniffing to pick
        # the right one. Falls back to the first-registered candidate if
        # none of them recognize the content (or the file doesn't exist
        # yet, e.g. mid Save-As).
        for p in candidates:
            try:
                if p.sniff(path):
                    return p
            except Exception:
                continue
        return candidates[0]

    @classmethod
    def save_dialog_filter(cls) -> str:
        return ";;".join(p.file_filter() for p in cls._parsers)


# Register built-in formats. Add new parsers here as they are written.
ParserRegistry.register(JSONAnnotationParser())
ParserRegistry.register(YoloPlateOCRTextParser())
ParserRegistry.register(YoloCharOCRAnnotationParser())
ParserRegistry.register(YoloBoxAnnotationParser())
ParserRegistry.register(DelimitedTextAnnotationParser())


# ======================================================================
# 3. APPLICATION STATE  (Model, Qt signal bus)
# ======================================================================

class AnnotationModel(QObject):
    """
    Holds the in-memory list of annotations for the currently open image and
    broadcasts fine-grained change signals so the View layer can stay in
    sync without full rebuilds. All mutation goes through this class so
    QUndoCommand objects have one place to call into.
    """

    annotationsReset = pyqtSignal()
    annotationAdded = pyqtSignal(str)          # id
    annotationRemoved = pyqtSignal(str)        # id
    annotationUpdated = pyqtSignal(str)        # id (geometry or fields)
    selectionChanged = pyqtSignal(object)       # id or None
    dirtyChanged = pyqtSignal(bool)

    def __init__(self) -> None:
        super().__init__()
        self._annotations: Dict[str, Annotation] = {}
        self._order: List[str] = []
        self._selected_id: Optional[str] = None
        self._dirty = False

        self.source_path: Optional[str] = None
        self.parser: Optional[AnnotationParser] = None

    # -- bulk ------------------------------------------------------
    def reset(self, annotations: List[Annotation], source_path: Optional[str],
              parser: Optional[AnnotationParser]) -> None:
        self._annotations = {a.id: a for a in annotations}
        self._order = [a.id for a in annotations]
        self._selected_id = None
        self.source_path = source_path
        self.parser = parser
        self._set_dirty(False)
        self.annotationsReset.emit()

    def clear(self) -> None:
        self.reset([], None, None)

    # -- queries -----------------------------------------------------
    def get(self, ann_id: str) -> Optional[Annotation]:
        return self._annotations.get(ann_id)

    def all(self) -> List[Annotation]:
        return [self._annotations[i] for i in self._order if i in self._annotations]

    def selected_id(self) -> Optional[str]:
        return self._selected_id

    def selected(self) -> Optional[Annotation]:
        return self.get(self._selected_id) if self._selected_id else None

    def is_dirty(self) -> bool:
        return self._dirty

    # -- mutation (called by QUndoCommand subclasses) -----------------
    def add(self, annotation: Annotation, index: Optional[int] = None) -> None:
        self._annotations[annotation.id] = annotation
        if index is None or index >= len(self._order):
            self._order.append(annotation.id)
        else:
            self._order.insert(index, annotation.id)
        self._set_dirty(True)
        self.annotationAdded.emit(annotation.id)

    def remove(self, ann_id: str) -> None:
        if ann_id in self._annotations:
            del self._annotations[ann_id]
            self._order.remove(ann_id)
            if self._selected_id == ann_id:
                self.select(None)
            self._set_dirty(True)
            self.annotationRemoved.emit(ann_id)

    def index_of(self, ann_id: str) -> int:
        return self._order.index(ann_id)

    def update_bbox(self, ann_id: str, bbox: BoundingBox) -> None:
        ann = self._annotations.get(ann_id)
        if ann is None:
            return
        ann.bbox = bbox.clone()
        self._set_dirty(True)
        self.annotationUpdated.emit(ann_id)

    def update_field(self, ann_id: str, field_name: str, value: Any) -> None:
        ann = self._annotations.get(ann_id)
        if ann is None:
            return
        setattr(ann, field_name, value)
        self._set_dirty(True)
        self.annotationUpdated.emit(ann_id)

    def select(self, ann_id: Optional[str]) -> None:
        if ann_id != self._selected_id:
            self._selected_id = ann_id
            self.selectionChanged.emit(ann_id)

    def mark_saved(self) -> None:
        self._set_dirty(False)

    def _set_dirty(self, value: bool) -> None:
        if self._dirty != value:
            self._dirty = value
            self.dirtyChanged.emit(value)


# ======================================================================
# 4. UNDO / REDO COMMANDS
# ======================================================================

class AddAnnotationCommand(QUndoCommand):
    def __init__(self, model: AnnotationModel, annotation: Annotation,
                 index: Optional[int] = None):
        super().__init__(f"Add box '{annotation.label or annotation.text}'")
        self.model = model
        self.annotation = annotation
        self.index = index

    def redo(self) -> None:
        self.model.add(self.annotation, self.index)
        self.model.select(self.annotation.id)

    def undo(self) -> None:
        self.model.remove(self.annotation.id)


class DeleteAnnotationCommand(QUndoCommand):
    def __init__(self, model: AnnotationModel, annotation: Annotation, index: int):
        super().__init__(f"Delete box '{annotation.label or annotation.text}'")
        self.model = model
        self.annotation = annotation
        self.index = index

    def redo(self) -> None:
        self.model.remove(self.annotation.id)

    def undo(self) -> None:
        self.model.add(self.annotation, self.index)
        self.model.select(self.annotation.id)


class GeometryChangeCommand(QUndoCommand):
    def __init__(self, model: AnnotationModel, ann_id: str,
                 old_bbox: BoundingBox, new_bbox: BoundingBox):
        super().__init__("Edit box geometry")
        self.model = model
        self.ann_id = ann_id
        self.old_bbox = old_bbox.clone()
        self.new_bbox = new_bbox.clone()

    def redo(self) -> None:
        self.model.update_bbox(self.ann_id, self.new_bbox)

    def undo(self) -> None:
        self.model.update_bbox(self.ann_id, self.old_bbox)


class FieldChangeCommand(QUndoCommand):
    def __init__(self, model: AnnotationModel, ann_id: str, field_name: str,
                 old_value: Any, new_value: Any):
        super().__init__(f"Edit {field_name}")
        self.model = model
        self.ann_id = ann_id
        self.field_name = field_name
        self.old_value = old_value
        self.new_value = new_value

    def redo(self) -> None:
        self.model.update_field(self.ann_id, self.field_name, self.new_value)

    def undo(self) -> None:
        self.model.update_field(self.ann_id, self.field_name, self.old_value)


# ======================================================================
# 5. GRAPHICS ITEM  -  interactive bounding box (View)
# ======================================================================

class ResizeHandle:
    NONE, TL, T, TR, R, BR, B, BL, L = range(9)

    CORNERS = (TL, TR, BR, BL)
    EDGES = (T, R, B, L)


class BBoxGraphicsItem(QGraphicsRectItem):
    """
    One editable bounding box overlay. Position (item.pos()) always equals
    the box's top-left corner in image-pixel/scene coordinates; rect() is
    always (0, 0, width, height) in the item's local coordinates. This keeps
    geometry math simple for both move and resize interactions.
    """

    BASE_HANDLE_PX = 8.0

    def __init__(self, annotation: Annotation, controller: "AnnotationController"):
        super().__init__(0, 0, max(annotation.bbox.width, 1.0), max(annotation.bbox.height, 1.0))
        self.setPos(annotation.bbox.x, annotation.bbox.y)
        self.annotation_id = annotation.id
        self.controller = controller
        self.locked = annotation.locked

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(1.0)

        self._mode: Optional[str] = None
        self._active_handle = ResizeHandle.NONE
        self._press_scene_pos: Optional[QPointF] = None
        self._press_bbox: Optional[BoundingBox] = None
        # -- group-move bookkeeping (multi-select drag) ------------------
        self._group_press_bboxes: Dict[str, BoundingBox] = {}

    # -- helpers -------------------------------------------------------
    def set_locked(self, locked: bool) -> None:
        self.locked = locked
        self.update()

    def current_bbox(self) -> BoundingBox:
        return BoundingBox(self.pos().x(), self.pos().y(),
                            self.rect().width(), self.rect().height())

    def sync_from_model(self, bbox: BoundingBox) -> None:
        """Called by the controller to reflect model state (e.g. undo/redo)
        without generating another undo command."""
        self.prepareGeometryChange()
        self.setPos(bbox.x, bbox.y)
        self.setRect(0, 0, max(bbox.width, 1.0), max(bbox.height, 1.0))
        self.update()

    def _handle_px(self) -> float:
        scale = 1.0
        if self.scene() and self.scene().views():
            scale = self.scene().views()[0].transform().m11() or 1.0
        return max(self.BASE_HANDLE_PX / scale, 3.0)

    def _handle_at(self, local: QPointF) -> int:
        if self.locked:
            return ResizeHandle.NONE
        hs = self._handle_px()
        r = self.rect()
        pts = {
            ResizeHandle.TL: r.topLeft(), ResizeHandle.TR: r.topRight(),
            ResizeHandle.BL: r.bottomLeft(), ResizeHandle.BR: r.bottomRight(),
            ResizeHandle.T: QPointF(r.center().x(), r.top()),
            ResizeHandle.B: QPointF(r.center().x(), r.bottom()),
            ResizeHandle.L: QPointF(r.left(), r.center().y()),
            ResizeHandle.R: QPointF(r.right(), r.center().y()),
        }
        for handle, p in pts.items():
            if (abs(local.x() - p.x()) <= hs) and (abs(local.y() - p.y()) <= hs):
                return handle
        return ResizeHandle.NONE

    # -- painting --------------------------------------------------------
    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget=None) -> None:
        selected = self.isSelected()
        if self.locked:
            color = QColor(160, 160, 160)
        elif selected:
            color = QColor(255, 140, 0)
        else:
            color = QColor(0, 200, 120)

        pen = QPen(color, 2 if not selected else 2.5)
        pen.setCosmetic(True)
        painter.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(35 if not selected else 60)
        painter.setBrush(QBrush(fill))
        painter.drawRect(self.rect())

        # label / OCR text badge at the top side corner of the box
        ann = self.controller.model.get(self.annotation_id)
        if ann is not None:
            if ann.text and ann.label and ann.label.lower() not in ("character", "plate", "") and ann.label != ann.text:
                label_text = f"{ann.label}: {ann.text}"
            elif ann.text:
                label_text = ann.text
            elif ann.label:
                label_text = ann.label
            else:
                label_text = "(unlabeled)"

            # View scale to convert between item coordinates and screen pixels
            scale = 1.0
            if self.scene() and self.scene().views():
                scale = abs(self.scene().views()[0].transform().m11()) or 1.0
            elif painter.worldTransform():
                scale = abs(painter.worldTransform().m11()) or 1.0

            painter.save()

            # Anchor at the top-left of the bounding box in item coordinates
            top_left = self.rect().topLeft()
            painter.translate(top_left)

            # Invert the view zoom scale so we draw in exact screen pixel dimensions (fixed larger size)
            painter.scale(1.0 / scale, 1.0 / scale)

            # Fixed larger bold font for prominent visibility regardless of zoom level
            font = QFont("Segoe UI", 11)
            font.setStyleHint(QFont.StyleHint.SansSerif)
            font.setBold(True)
            painter.setFont(font)
            fm = QFontMetricsF(font)

            pad_x = 8.0   # screen pixels padding
            pad_y = 4.0   # screen pixels padding
            tw = fm.horizontalAdvance(label_text)
            th = fm.height()
            badge_w = tw + 2.0 * pad_x
            badge_h = th + 2.0 * pad_y

            # Place above the top-left corner (or inside the top if at the very ceiling of scene)
            top_margin_px = 3.0
            if (self.pos().y() + top_left.y()) * scale >= (badge_h + top_margin_px):
                badge_y = -badge_h - top_margin_px
            else:
                badge_y = top_margin_px

            badge_rect = QRectF(0.0, badge_y, badge_w, badge_h)

            # Draw badge background pill with high contrast
            painter.setPen(QPen(QColor(0, 0, 0, 90), 1.0))
            bg_color = QColor(color)
            bg_color.setAlpha(235)
            painter.setBrush(QBrush(bg_color))
            painter.drawRoundedRect(badge_rect, 4.0, 4.0)

            # Draw crisp readable text
            painter.setPen(QPen(QColor(255, 255, 255)))
            text_rect = QRectF(pad_x, badge_y + pad_y, tw, th)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, label_text)

            painter.restore()

        # resize handles
        if selected and not self.locked:
            hs = self._handle_px()
            painter.setPen(QPen(QColor(30, 30, 30), 1))
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            r = self.rect()
            handle_points = [
                r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight(),
                QPointF(r.center().x(), r.top()), QPointF(r.center().x(), r.bottom()),
                QPointF(r.left(), r.center().y()), QPointF(r.right(), r.center().y()),
            ]
            for p in handle_points:
                painter.drawRect(QRectF(p.x() - hs / 2, p.y() - hs / 2, hs, hs))

    def boundingRect(self) -> QRectF:
        scale = 1.0
        if self.scene() and self.scene().views():
            scale = abs(self.scene().views()[0].transform().m11()) or 1.0
        hs = max(self.BASE_HANDLE_PX / scale, 3.0)
        badge_h_scene = 45.0 / scale
        badge_w_scene = 400.0 / scale
        r = self.rect()
        left = min(r.left() - hs, r.left())
        top = min(r.top() - hs - badge_h_scene, r.top() - hs)
        right = max(r.right() + hs, r.left() + badge_w_scene)
        bottom = r.bottom() + hs
        return QRectF(left, top, right - left, bottom - top)

    # -- interaction -------------------------------------------------------
    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        if self.scene() and self.scene().views() and getattr(self.scene().views()[0], "_space_pan", False):
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            return super().hoverMoveEvent(event)
        if self.locked:
            self.setCursor(Qt.CursorShape.ForbiddenCursor)
            return super().hoverMoveEvent(event)
        handle = self._handle_at(event.pos())
        cursor_map = {
            ResizeHandle.TL: Qt.CursorShape.SizeFDiagCursor,
            ResizeHandle.BR: Qt.CursorShape.SizeFDiagCursor,
            ResizeHandle.TR: Qt.CursorShape.SizeBDiagCursor,
            ResizeHandle.BL: Qt.CursorShape.SizeBDiagCursor,
            ResizeHandle.T: Qt.CursorShape.SizeVerCursor,
            ResizeHandle.B: Qt.CursorShape.SizeVerCursor,
            ResizeHandle.L: Qt.CursorShape.SizeHorCursor,
            ResizeHandle.R: Qt.CursorShape.SizeHorCursor,
            ResizeHandle.NONE: Qt.CursorShape.SizeAllCursor,
        }
        self.setCursor(cursor_map[handle])
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        # Panning must NEVER move the bounding boxes
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return
        if self.scene() and self.scene().views() and getattr(self.scene().views()[0], "_space_pan", False):
            event.ignore()
            return

        # Shift/Ctrl-click toggles this box's membership in the current
        # multi-selection instead of collapsing the selection down to just
        # this box; a plain click on an already-selected box (as part of a
        # multi-selection) preserves the group so the drag below can move
        # everything together.
        additive = bool(event.modifiers() & (
            Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier
        ))
        if additive:
            self.setSelected(not self.isSelected())
            if not self.isSelected() and self.controller.model.selected_id() == self.annotation_id:
                remaining = [it.annotation_id for it in self.scene().selectedItems() if isinstance(it, BBoxGraphicsItem)]
                self.controller.select_annotation(remaining[0] if remaining else None)
            elif self.isSelected():
                self.controller.select_annotation(self.annotation_id)
        elif not self.isSelected():
            self.scene().clearSelection()
            self.setSelected(True)
            self.controller.select_annotation(self.annotation_id)
        else:
            self.controller.select_annotation(self.annotation_id)

        if self.locked:
            event.accept()
            return
        local = event.pos()
        self._active_handle = self._handle_at(local)
        self._press_scene_pos = event.scenePos()
        self._press_bbox = self.current_bbox()
        self._mode = "resize" if self._active_handle != ResizeHandle.NONE else "move"

        # Snapshot every OTHER selected, unlocked box's starting geometry so
        # a "move" drag can translate the whole group together, not just
        # the box the mouse actually landed on.
        self._group_press_bboxes = {}
        if self._mode == "move" and self.scene() is not None:
            for item in self.scene().selectedItems():
                if (isinstance(item, BBoxGraphicsItem) and item is not self
                        and not item.locked):
                    self._group_press_bboxes[item.annotation_id] = item.current_bbox()

        event.accept()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self.locked or self._mode is None or self._press_bbox is None:
            return
        delta = event.scenePos() - self._press_scene_pos
        b = self._press_bbox
        min_size = 4.0

        if self._mode == "move":
            self.setPos(b.x + delta.x(), b.y + delta.y())
            self.controller.live_geometry_preview(self.annotation_id, self.current_bbox())
            # Drag every other selected box by the same delta so the whole
            # group moves together.
            if self._group_press_bboxes:
                for ann_id, start_bbox in self._group_press_bboxes.items():
                    sibling = self.controller.bbox_item_for(ann_id)
                    if sibling is None:
                        continue
                    sibling.setPos(start_bbox.x + delta.x(), start_bbox.y + delta.y())
            return

        x, y, w, h = b.x, b.y, b.width, b.height
        dx, dy = delta.x(), delta.y()
        h_ = self._active_handle

        new_x, new_w = x, w
        new_y, new_h = y, h

        if h_ in (ResizeHandle.TL, ResizeHandle.L, ResizeHandle.BL):
            new_x = x + dx
            new_w = w - dx
        elif h_ in (ResizeHandle.TR, ResizeHandle.R, ResizeHandle.BR):
            new_w = w + dx

        if h_ in (ResizeHandle.TL, ResizeHandle.T, ResizeHandle.TR):
            new_y = y + dy
            new_h = h - dy
        elif h_ in (ResizeHandle.BL, ResizeHandle.B, ResizeHandle.BR):
            new_h = h + dy

        if new_w < min_size:
            if h_ in (ResizeHandle.TL, ResizeHandle.L, ResizeHandle.BL):
                new_x = x + w - min_size
            new_w = min_size
        if new_h < min_size:
            if h_ in (ResizeHandle.TL, ResizeHandle.T, ResizeHandle.TR):
                new_y = y + h - min_size
            new_h = min_size

        self.prepareGeometryChange()
        self.setPos(new_x, new_y)
        self.setRect(0, 0, new_w, new_h)
        self.controller.live_geometry_preview(self.annotation_id, self.current_bbox())

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self.locked or self._mode is None or self._press_bbox is None:
            self._mode = None
            self._group_press_bboxes = {}
            return
        new_bbox = self.current_bbox()
        old_bbox = self._press_bbox
        changed = (
            abs(new_bbox.x - old_bbox.x) > 0.01 or abs(new_bbox.y - old_bbox.y) > 0.01
            or abs(new_bbox.width - old_bbox.width) > 0.01
            or abs(new_bbox.height - old_bbox.height) > 0.01
        )

        # Build the list of (ann_id, old_bbox, new_bbox) for this box plus
        # any group-move siblings, so the whole drag becomes one undo step.
        changes: List[Tuple[str, BoundingBox, BoundingBox]] = []
        if changed:
            changes.append((self.annotation_id, old_bbox, new_bbox))
        for ann_id, start_bbox in self._group_press_bboxes.items():
            sibling = self.controller.bbox_item_for(ann_id)
            if sibling is None:
                continue
            sib_new = sibling.current_bbox()
            if (abs(sib_new.x - start_bbox.x) > 0.01 or abs(sib_new.y - start_bbox.y) > 0.01
                    or abs(sib_new.width - start_bbox.width) > 0.01
                    or abs(sib_new.height - start_bbox.height) > 0.01):
                changes.append((ann_id, start_bbox, sib_new))

        if changes:
            self.controller.commit_group_geometry_change(changes)

        self._mode = None
        self._active_handle = ResizeHandle.NONE
        self._press_bbox = None
        self._group_press_bboxes = {}


# ======================================================================
# 6. IMAGE VIEWER  (zoom / pan / fit / box-creation rubber band)
# ======================================================================

class ImageViewer(QGraphicsView):
    zoomChanged = pyqtSignal(float)
    boxCreated = pyqtSignal(QRectF)

    MIN_ZOOM, MAX_ZOOM = 0.02, 40.0

    def __init__(self, scene: QGraphicsScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
            | QPainter.RenderHint.TextAntialiasing
        )
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QBrush(QColor(45, 45, 48)))
        self.setRubberBandSelectionMode(Qt.ItemSelectionMode.IntersectsItemShape)

        self._zoom = 1.0
        self._space_pan = False
        self._panning = False
        self._pan_start = QPointF()
        self._shift_multiselect = False

        self._create_mode = False
        self._draw_start: Optional[QPointF] = None
        self._rubber_item: Optional[QGraphicsRectItem] = None

    # -- public API -----------------------------------------------------
    def set_create_mode(self, enabled: bool) -> None:
        self._create_mode = enabled
        if enabled:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.viewport().setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor
        )

    def fit_to_image(self, rect: QRectF) -> None:
        if rect.isEmpty():
            return
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = self.transform().m11()
        self.zoomChanged.emit(self._zoom)

    def reset_zoom(self) -> None:
        self.setTransform(QTransform())
        self._zoom = 1.0
        self.zoomChanged.emit(self._zoom)

    def set_zoom(self, factor: float, anchor_viewport_pos=None) -> None:
        factor = max(self.MIN_ZOOM, min(factor, self.MAX_ZOOM))
        ratio = factor / self._zoom if self._zoom else 1.0
        self.scale(ratio, ratio)
        self._zoom = factor
        self.zoomChanged.emit(self._zoom)

    # -- events ----------------------------------------------------------
    def wheelEvent(self, event: QWheelEvent) -> None:
        angle = event.angleDelta().y()
        if angle == 0:
            return
        factor = 1.15 ** (angle / 120.0)
        self.set_zoom(self._zoom * factor)
        event.accept()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan = True
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        # Holding Shift (with nothing else down) arms rubber-band
        # multi-select: drag over empty canvas to lasso every box that
        # intersects the rectangle, in addition to whatever is already
        # selected. Released automatically on key-up (see below).
        if event.key() == Qt.Key.Key_Shift and not event.isAutoRepeat() and not self._create_mode:
            self._shift_multiselect = True
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        # QGraphicsView normally consumes arrow keys to scroll. We give
        # Left/Right to the window-level Prev/Next Image shortcuts instead
        # (scrolling stays available via wheel, space-drag, and middle
        # drag). Up/Down still scroll as usual.
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            event.ignore()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan = False
            self.viewport().setCursor(
                Qt.CursorShape.CrossCursor if self._create_mode else Qt.CursorShape.ArrowCursor
            )
        if event.key() == Qt.Key.Key_Shift and not event.isAutoRepeat():
            self._shift_multiselect = False
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._create_mode and event.button() == Qt.MouseButton.LeftButton:
            self._draw_start = self.mapToScene(event.pos())
            self._rubber_item = QGraphicsRectItem(QRectF(self._draw_start, self._draw_start))
            pen = QPen(QColor(255, 60, 60), 2, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self._rubber_item.setPen(pen)
            self._rubber_item.setZValue(1000)
            self.scene().addItem(self._rubber_item)
            event.accept()
            return

        if event.button() == Qt.MouseButton.MiddleButton or (
            self._space_pan and event.button() == Qt.MouseButton.LeftButton
        ):
            self._panning = True
            self._pan_start = event.pos()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        # Shift-drag over empty canvas: let Qt's built-in rubber band
        # selection do the multi-select (dragMode was switched to
        # RubberBandDrag in keyPressEvent above).
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._create_mode and self._rubber_item is not None:
            cur = self.mapToScene(event.pos())
            rect = QRectF(self._draw_start, cur).normalized()
            self._rubber_item.setRect(rect)
            event.accept()
            return

        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._create_mode and self._rubber_item is not None and event.button() == Qt.MouseButton.LeftButton:
            rect = self._rubber_item.rect()
            self.scene().removeItem(self._rubber_item)
            self._rubber_item = None
            self._draw_start = None
            if rect.width() > 3 and rect.height() > 3:
                self.boxCreated.emit(rect)
            event.accept()
            return

        if self._panning and event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
            self._panning = False
            self.viewport().setCursor(
                Qt.CursorShape.OpenHandCursor if self._space_pan else
                (Qt.CursorShape.CrossCursor if self._create_mode else Qt.CursorShape.ArrowCursor)
            )
            event.accept()
            return

        super().mouseReleaseEvent(event)


class PasteOffsetDialog(QDialog):
    """Modal dialog to specify an optional X/Y pixel offset when pasting annotations."""

    def __init__(self, parent: Optional[QWidget], num_boxes: int, source_name: str,
                 default_x: float = 0.0, default_y: float = 0.0):
        super().__init__(parent)
        self.setWindowTitle("Paste Annotations with Offset")
        self.setMinimumWidth(340)

        layout = QVBoxLayout(self)

        info_lbl = QLabel(f"<b>Pasting:</b> {num_boxes} box(es) from {source_name}")
        info_lbl.setWordWrap(True)
        layout.addWidget(info_lbl)

        form = QFormLayout()
        self.spin_x = QDoubleSpinBox()
        self.spin_x.setRange(-9999.0, 9999.0)
        self.spin_x.setSingleStep(1.0)
        self.spin_x.setDecimals(1)
        self.spin_x.setValue(default_x)
        self.spin_x.setSuffix(" px")
        form.addRow("X Offset (horizontal):", self.spin_x)

        self.spin_y = QDoubleSpinBox()
        self.spin_y.setRange(-9999.0, 9999.0)
        self.spin_y.setSingleStep(1.0)
        self.spin_y.setDecimals(1)
        self.spin_y.setValue(default_y)
        self.spin_y.setSuffix(" px")
        form.addRow("Y Offset (vertical):", self.spin_y)
        layout.addLayout(form)

        preset_box = QGroupBox("Quick Presets")
        preset_layout = QHBoxLayout(preset_box)
        presets = [
            ("0, 0", 0.0, 0.0),
            ("+15, +15", 15.0, 15.0),
            ("Left (-10)", -10.0, 0.0),
            ("Right (+10)", 10.0, 0.0),
            ("Up (-10)", 0.0, -10.0),
            ("Down (+10)", 0.0, 10.0),
        ]
        for label, px, py in presets:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked=False, x=px, y=py: self._set_preset(x, y))
            preset_layout.addWidget(btn)
        layout.addWidget(preset_box)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok = QPushButton("Paste Boxes")
        self.btn_ok.setDefault(True)
        self.btn_ok.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_ok)
        layout.addLayout(btn_layout)

    def _set_preset(self, x: float, y: float) -> None:
        self.spin_x.setValue(x)
        self.spin_y.setValue(y)

    def offset(self) -> Tuple[float, float]:
        return (self.spin_x.value(), self.spin_y.value())


# ======================================================================
# 7. PROPERTIES PANEL
# ======================================================================

class PropertiesPanel(QWidget):
    """Shows/edits Label, Text, X, Y, Width, Height (editable) and
    Center X / Center Y (read-only, auto-derived)."""

    labelChanged = pyqtSignal(str)
    textChanged = pyqtSignal(str)
    geometryChanged = pyqtSignal(float, float, float, float)  # x, y, w, h
    lockToggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        box = QGroupBox("Selected Annotation")
        form = QFormLayout(box)

        self.label_edit = QLineEdit()
        self.text_edit = QLineEdit()
        self.text_edit.setPlaceholderText("Recognized OCR text")

        self.x_spin = self._make_spin()
        self.y_spin = self._make_spin()
        self.w_spin = self._make_spin(minimum=1.0)
        self.h_spin = self._make_spin(minimum=1.0)
        self.cx_spin = self._make_spin(read_only=True)
        self.cy_spin = self._make_spin(read_only=True)

        self.locked_check = QCheckBox("Locked")

        form.addRow("Label:", self.label_edit)
        form.addRow("Text:", self.text_edit)
        form.addRow("X:", self.x_spin)
        form.addRow("Y:", self.y_spin)
        form.addRow("Width:", self.w_spin)
        form.addRow("Height:", self.h_spin)
        form.addRow("Center X:", self.cx_spin)
        form.addRow("Center Y:", self.cy_spin)
        form.addRow("", self.locked_check)

        layout.addWidget(box)
        layout.addStretch(1)

        self.setEnabled(False)

        self.label_edit.editingFinished.connect(self._on_label_edited)
        self.text_edit.editingFinished.connect(self._on_text_edited)
        for spin in (self.x_spin, self.y_spin, self.w_spin, self.h_spin):
            spin.editingFinished.connect(self._on_geometry_edited)
        self.locked_check.toggled.connect(self._on_lock_toggled)

    @staticmethod
    def _make_spin(read_only: bool = False, minimum: float = -1_000_000) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, 1_000_000)
        spin.setDecimals(2)
        spin.setSingleStep(1.0)
        if read_only:
            spin.setReadOnly(True)
            spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
            spin.setStyleSheet("background-color: #eeeeee;")
        return spin

    # -- external API --------------------------------------------------
    def show_annotation(self, ann: Optional[Annotation]) -> None:
        self._updating = True
        try:
            if ann is None:
                self.setEnabled(False)
                for w in (self.label_edit, self.text_edit):
                    w.clear()
                for s in (self.x_spin, self.y_spin, self.w_spin, self.h_spin, self.cx_spin, self.cy_spin):
                    s.setValue(0)
                self.locked_check.setChecked(False)
                return
            self.setEnabled(True)
            self.label_edit.setText(ann.label)
            self.text_edit.setText(ann.text)
            self.x_spin.setValue(ann.bbox.x)
            self.y_spin.setValue(ann.bbox.y)
            self.w_spin.setValue(ann.bbox.width)
            self.h_spin.setValue(ann.bbox.height)
            self.cx_spin.setValue(ann.bbox.center_x)
            self.cy_spin.setValue(ann.bbox.center_y)
            self.locked_check.setChecked(ann.locked)
            editable = not ann.locked
            for w in (self.label_edit, self.text_edit, self.x_spin, self.y_spin, self.w_spin, self.h_spin):
                w.setEnabled(editable)
        finally:
            self._updating = False

    def update_geometry_only(self, bbox: BoundingBox) -> None:
        """Refresh geometry fields without touching label/text/lock (used
        during live drag so users can watch numbers update)."""
        self._updating = True
        try:
            self.x_spin.setValue(bbox.x)
            self.y_spin.setValue(bbox.y)
            self.w_spin.setValue(bbox.width)
            self.h_spin.setValue(bbox.height)
            self.cx_spin.setValue(bbox.center_x)
            self.cy_spin.setValue(bbox.center_y)
        finally:
            self._updating = False

    # -- internal slots --------------------------------------------------
    def _on_label_edited(self) -> None:
        if not self._updating:
            self.labelChanged.emit(self.label_edit.text())

    def _on_text_edited(self) -> None:
        if not self._updating:
            self.textChanged.emit(self.text_edit.text())

    def _on_geometry_edited(self) -> None:
        if not self._updating:
            self.geometryChanged.emit(
                self.x_spin.value(), self.y_spin.value(),
                self.w_spin.value(), self.h_spin.value(),
            )

    def _on_lock_toggled(self, checked: bool) -> None:
        if not self._updating:
            self.lockToggled.emit(checked)


# ======================================================================
# 7b. CHARACTER EDIT PANEL
# ======================================================================

class CharacterEditPanel(QWidget):
    """
    Per-character view/editor for the selected annotation's recognized
    text. Complements PropertiesPanel's single Text field: instead of
    editing the whole string at once, this shows the text broken out into
    one small box per character, each individually READ (displayed) and
    WRITEABLE (editable in place), plus a per-character delete ("x")
    button so a stray/garbage glyph can be removed without retyping the
    whole word.

    Also provides one-click cleanup for the most common OCR artifact:
    stray quote characters (" ' ` and their curly Unicode cousins) that
    upstream OCR sometimes wraps around or injects into recognized text
    (e.g. an upstream engine returning `"ABC1234"` instead of `ABC1234`).
    "Strip Quotes" removes every such character from the text; each
    character can also be removed individually via its own [x] button
    below the letter, for any other stray/incorrect character.
    """

    textChanged = pyqtSignal(str)

    # Characters considered "stray quote" noise for the one-click cleanup.
    QUOTE_CHARS = set('"\'`\u201c\u201d\u2018\u2019\u00ab\u00bb')

    CELL_WIDTH = 30

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False
        self._current_text = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        box = QGroupBox("Character Editor")
        v = QVBoxLayout(box)

        hint = QLabel(
            "Each box below is one character of the recognized text -- "
            "edit a letter directly, or use \u00d7 to delete just that "
            "character."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 10px;")
        v.addWidget(hint)

        # Scrollable strip of per-character cells, built dynamically.
        self.strip_container = QWidget()
        self.strip_layout = QHBoxLayout(self.strip_container)
        self.strip_layout.setContentsMargins(0, 4, 0, 4)
        self.strip_layout.setSpacing(4)
        self.strip_layout.addStretch(1)
        v.addWidget(self.strip_container)

        btn_row = QHBoxLayout()
        self.strip_quotes_btn = QPushButton('Strip Quote Characters ( " \' )')
        self.strip_quotes_btn.setToolTip(
            "Remove every \" ' ` \u201c \u201d \u2018 \u2019 character from the recognized text"
        )
        self.strip_quotes_btn.clicked.connect(self._on_strip_quotes)
        btn_row.addWidget(self.strip_quotes_btn)

        self.add_char_btn = QPushButton("+ Add Character")
        self.add_char_btn.clicked.connect(self._on_add_character)
        btn_row.addWidget(self.add_char_btn)
        v.addLayout(btn_row)

        self.preview_label = QLabel("")
        self.preview_label.setStyleSheet("color: gray; font-size: 10px;")
        v.addWidget(self.preview_label)

        outer.addWidget(box)
        outer.addStretch(1)
        self.setEnabled(False)

    # -- external API --------------------------------------------------
    def show_text(self, text: Optional[str]) -> None:
        self._updating = True
        try:
            self._current_text = text or ""
            self.setEnabled(text is not None)
            self._rebuild_cells()
            self._update_preview()
        finally:
            self._updating = False

    # -- internal --------------------------------------------------------
    def _clear_cells(self) -> None:
        while self.strip_layout.count() > 1:  # keep the trailing stretch
            item = self.strip_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _rebuild_cells(self) -> None:
        self._clear_cells()
        for i, ch in enumerate(self._current_text):
            cell = self._make_cell(i, ch)
            self.strip_layout.insertWidget(self.strip_layout.count() - 1, cell)

    def _make_cell(self, index: int, ch: str) -> QWidget:
        cell = QWidget()
        col = QVBoxLayout(cell)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)

        edit = QLineEdit(ch)
        edit.setFixedWidth(self.CELL_WIDTH)
        edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        edit.setMaxLength(1)
        if ch in self.QUOTE_CHARS:
            edit.setStyleSheet("background-color: #ffe0e0;")
            edit.setToolTip("Stray quote character -- likely an OCR artifact")
        edit.editingFinished.connect(lambda idx=index, e=edit: self._on_char_edited(idx, e))
        col.addWidget(edit)

        del_btn = QToolButton()
        del_btn.setText("\u00d7")
        del_btn.setFixedWidth(self.CELL_WIDTH)
        del_btn.setToolTip("Delete this character")
        del_btn.clicked.connect(lambda checked=False, idx=index: self._on_char_deleted(idx))
        col.addWidget(del_btn)

        return cell

    def _update_preview(self) -> None:
        self.preview_label.setText(f'Text: "{self._current_text}"  ({len(self._current_text)} chars)')

    def _emit_change(self, new_text: str) -> None:
        self._current_text = new_text
        self._update_preview()
        if not self._updating:
            self.textChanged.emit(new_text)

    def _on_char_edited(self, index: int, edit: QLineEdit) -> None:
        if self._updating or index >= len(self._current_text):
            return
        new_char = edit.text()
        if len(new_char) == 0:
            # Emptied via backspace -- treat like a delete.
            self._on_char_deleted(index)
            return
        chars = list(self._current_text)
        chars[index] = new_char[0]
        self._emit_change("".join(chars))
        self._rebuild_cells()

    def _on_char_deleted(self, index: int) -> None:
        if index >= len(self._current_text):
            return
        chars = list(self._current_text)
        del chars[index]
        self._emit_change("".join(chars))
        self._rebuild_cells()

    def _on_add_character(self) -> None:
        self._emit_change(self._current_text + " ")
        self._rebuild_cells()

    def _on_strip_quotes(self) -> None:
        cleaned = "".join(c for c in self._current_text if c not in self.QUOTE_CHARS)
        if cleaned != self._current_text:
            self._emit_change(cleaned)
            self._rebuild_cells()


# ======================================================================
# 7c. BATCH TEXT EDIT PANEL  (Ctrl+A -> review/correct every text field
#     in one place, in left-to-right / top-to-bottom reading order)
# ======================================================================

class BatchTextEditPanel(QWidget):
    """
    Lists the recognized text of every currently-selected annotation
    (normally "all of them", via Ctrl+A / Select All) as one editable row
    per box, in SEQUENTIAL reading order rather than file/creation order,
    so a human doing a full correction pass over a plate/label can work
    top-to-bottom the way they'd read the image.

    This is a batch workflow, distinct from PropertiesPanel/
    CharacterEditPanel (which only ever show the ONE currently selected
    box): edit as many rows as needed, then either "Apply Corrections"
    (commits every changed row as a single undo/redo step) or "Apply &&
    Save" (same, then immediately writes the annotation file).
    """

    applyRequested = pyqtSignal(dict)   # {ann_id: new_text} for CHANGED rows only
    saveRequested = pyqtSignal(dict)    # same, but caller should save() after applying
    refreshRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries: List[Tuple[str, QLineEdit, str]] = []  # (ann_id, edit, original_text)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        box = QGroupBox("Edit All Text (reading order)")
        v = QVBoxLayout(box)

        hint = QLabel(
            "Every selected box's recognized text, in sequential reading "
            "order (top-to-bottom, left-to-right). Correct as many rows "
            "as you like, then apply."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 10px;")
        v.addWidget(hint)

        header_row = QHBoxLayout()
        self.count_label = QLabel("0 boxes")
        self.count_label.setStyleSheet("color: gray; font-size: 10px;")
        header_row.addWidget(self.count_label)
        header_row.addStretch(1)
        refresh_btn = QToolButton()
        refresh_btn.setText("Refresh")
        refresh_btn.setToolTip("Re-scan the current selection in reading order")
        refresh_btn.clicked.connect(self.refreshRequested.emit)
        header_row.addWidget(refresh_btn)
        v.addLayout(header_row)

        self._rows_container = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 4, 0, 4)
        self._rows_layout.setSpacing(4)
        self._rows_layout.addStretch(1)
        v.addWidget(self._rows_container, 1)

        btn_row = QHBoxLayout()
        self.apply_btn = QPushButton("Apply Corrections")
        self.apply_btn.setToolTip("Commit every changed row as one undo step")
        self.apply_btn.clicked.connect(self._on_apply)
        btn_row.addWidget(self.apply_btn)

        self.apply_save_btn = QPushButton("Apply && Save")
        self.apply_save_btn.setToolTip("Commit changes, then write the annotation file")
        self.apply_save_btn.clicked.connect(self._on_apply_and_save)
        btn_row.addWidget(self.apply_save_btn)
        v.addLayout(btn_row)

        outer.addWidget(box)
        self.setEnabled(False)

    # -- external API --------------------------------------------------
    def show_annotations(self, annotations: List[Annotation]) -> None:
        """`annotations` must already be sorted into the desired
        (reading) order by the caller -- this panel just renders rows."""
        self._clear_rows()
        self._entries = []
        for i, ann in enumerate(annotations, start=1):
            row, edit = self._make_row(i, ann)
            self._rows_layout.insertWidget(self._rows_layout.count() - 1, row)
            self._entries.append((ann.id, edit, ann.text))
        self.count_label.setText(f"{len(annotations)} box{'es' if len(annotations) != 1 else ''}")
        self.setEnabled(len(annotations) > 0)

    # -- internal --------------------------------------------------------
    def _clear_rows(self) -> None:
        while self._rows_layout.count() > 1:  # keep the trailing stretch
            item = self._rows_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _make_row(self, order_index: int, ann: Annotation) -> Tuple[QWidget, QLineEdit]:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)

        order_label = QLabel(f"{order_index}.")
        order_label.setFixedWidth(22)
        order_label.setStyleSheet("color: gray;")
        h.addWidget(order_label)

        tag_label = QLabel(ann.label or "(no label)")
        tag_label.setFixedWidth(70)
        tag_label.setStyleSheet("color: gray; font-size: 11px;")
        tag_label.setToolTip(f"Annotation id: {ann.id}")
        h.addWidget(tag_label)

        edit = QLineEdit(ann.text)
        if ann.locked:
            edit.setEnabled(False)
            edit.setToolTip("Locked -- unlock this box to edit its text")
        h.addWidget(edit, 1)

        return row, edit

    def _collect_changes(self) -> Dict[str, str]:
        changes: Dict[str, str] = {}
        for ann_id, edit, original_text in self._entries:
            new_text = edit.text()
            if new_text != original_text:
                changes[ann_id] = new_text
        return changes

    def _on_apply(self) -> None:
        changes = self._collect_changes()
        if changes:
            self.applyRequested.emit(changes)

    def _on_apply_and_save(self) -> None:
        changes = self._collect_changes()
        # Even with no text changes, "Apply && Save" should still save --
        # the user may just want to persist other edits made elsewhere.
        self.saveRequested.emit(changes)


# ======================================================================
# 8. ANNOTATION LIST PANEL
# ======================================================================

class AnnotationListPanel(QWidget):
    selectRequested = pyqtSignal(str)
    deleteRequested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(QLabel("Annotations"))
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._context_menu)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.list_widget)

    def refresh(self, annotations: List[Annotation], selected_id: Optional[str]) -> None:
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for ann in annotations:
            lock_mark = " [locked]" if ann.locked else ""
            item = QListWidgetItem(f"{ann.label or '(no label)'}: \"{ann.text}\"{lock_mark}")
            item.setData(Qt.ItemDataRole.UserRole, ann.id)
            if ann.id == selected_id:
                item.setSelected(True)
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        self.selectRequested.emit(item.data(Qt.ItemDataRole.UserRole))

    def _context_menu(self, pos) -> None:
        item = self.list_widget.itemAt(pos)
        if item is None:
            return
        ann_id = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        delete_action = menu.addAction("Delete")
        chosen = menu.exec(self.list_widget.mapToGlobal(pos))
        if chosen is delete_action:
            self.deleteRequested.emit(ann_id)


# ======================================================================
# 9. DOUBT NOTE DIALOG & FILE NAVIGATOR PANEL (folder mode)
# ======================================================================

class DoubtNoteDialog(QDialog):
    """Dialog to enter or edit a reason/note when marking a crop as doubt."""

    def __init__(self, parent: Optional[QWidget], filename: str, current_reason: str = "Doubt"):
        super().__init__(parent)
        self.setWindowTitle("Mark Crop as Doubt")
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)

        info_lbl = QLabel(f"Mark doubt for: <b>{filename}</b>")
        info_lbl.setWordWrap(True)
        layout.addWidget(info_lbl)

        layout.addWidget(QLabel("Doubt Reason / Note:"))
        self.edit_reason = QLineEdit(current_reason or "Doubt")
        layout.addWidget(self.edit_reason)

        preset_box = QGroupBox("Quick Reasons")
        grid = QVBoxLayout(preset_box)
        presets = [
            ("Blurry Crop", "Blurry Crop"),
            ("Cut-off Plate", "Cut-off Plate"),
            ("Unclear Chars", "Unclear Characters"),
            ("Wrong OCR", "Wrong OCR Text"),
            ("Bad Box", "Incorrect Box Boundary"),
            ("No Plate Visible", "No Plate Visible"),
        ]
        row1 = QHBoxLayout()
        row2 = QHBoxLayout()
        for label, val in presets[:3]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked=False, txt=val: self.edit_reason.setText(txt))
            row1.addWidget(btn)
        for label, val in presets[3:]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked=False, txt=val: self.edit_reason.setText(txt))
            row2.addWidget(btn)
        grid.addLayout(row1)
        grid.addLayout(row2)
        layout.addWidget(preset_box)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok = QPushButton("Save Doubt")
        self.btn_ok.setDefault(True)
        self.btn_ok.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_ok)
        layout.addLayout(btn_layout)

    def reason(self) -> str:
        return self.edit_reason.text().strip() or "Doubt"


class FileNavigatorPanel(QWidget):
    """
    Left-hand dock: lists the images/crops in the current folder AND their
    matched label files. Supports right-click context menu to mark/unmark
    crops as Doubt, edit doubt reasons, filter to Doubts only, and save/export
    doubts directly to a CSV file.
    """
    imageSelected = pyqtSignal(str)
    doubtToggled = pyqtSignal(str, bool, str)  # image_path, is_doubt, reason
    exportDoubtsRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._all_paths: List[str] = []
        self._current_path: Optional[str] = None
        self._label_lookup = None
        self._doubt_map: Dict[str, Dict[str, str]] = {}
        self._filter_doubts_only: bool = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header with Filter Toggle: All vs Doubts
        header_row = QHBoxLayout()
        self.lbl_title = QLabel("<b>Crops / Images</b>")
        header_row.addWidget(self.lbl_title)
        header_row.addStretch()

        self.btn_filter_all = QPushButton("All (0)")
        self.btn_filter_all.setCheckable(True)
        self.btn_filter_all.setChecked(True)
        self.btn_filter_all.clicked.connect(lambda: self._set_doubt_filter(False))

        self.btn_filter_doubts = QPushButton("❓ Doubts (0)")
        self.btn_filter_doubts.setCheckable(True)
        self.btn_filter_doubts.setChecked(False)
        self.btn_filter_doubts.clicked.connect(lambda: self._set_doubt_filter(True))

        header_row.addWidget(self.btn_filter_all)
        header_row.addWidget(self.btn_filter_doubts)
        layout.addLayout(header_row)

        self.list_widget = QListWidget()
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.list_widget, 3)

        layout.addWidget(QLabel("<b>Labels</b>"))
        self.labels_list = QListWidget()
        self.labels_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.labels_list.customContextMenuRequested.connect(self._show_context_menu_from_labels)
        self.labels_list.itemClicked.connect(self._on_label_item_clicked)
        layout.addWidget(self.labels_list, 2)

        # Quick action row: Toggle Doubt button + Export CSV button
        doubt_row = QHBoxLayout()
        self.btn_quick_doubt = QPushButton("❓ Mark Doubt")
        self.btn_quick_doubt.setToolTip("Toggle Doubt for current crop (Ctrl+Shift+D)")
        self.btn_quick_doubt.clicked.connect(self._on_quick_doubt_clicked)

        self.btn_export_csv = QPushButton("💾 Doubts CSV")
        self.btn_export_csv.setToolTip("Export/View Doubts CSV file")
        self.btn_export_csv.clicked.connect(self.exportDoubtsRequested.emit)

        doubt_row.addWidget(self.btn_quick_doubt)
        doubt_row.addWidget(self.btn_export_csv)
        layout.addLayout(doubt_row)

        # Prev / Next at the BOTTOM, with position counter
        nav_row = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Prev")
        self.counter_label = QLabel("- / -")
        self.counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.next_btn = QPushButton("Next ▶")
        nav_row.addWidget(self.prev_btn)
        nav_row.addWidget(self.counter_label, 1)
        nav_row.addWidget(self.next_btn)
        layout.addLayout(nav_row)

    def _set_doubt_filter(self, doubts_only: bool) -> None:
        self._filter_doubts_only = doubts_only
        self.btn_filter_all.setChecked(not doubts_only)
        self.btn_filter_doubts.setChecked(doubts_only)
        self._refresh_list()

    def set_doubts(self, doubt_map: Dict[str, Dict[str, str]]) -> None:
        self._doubt_map = doubt_map
        self._refresh_list()

    def _is_path_doubt(self, path: str) -> bool:
        if path in self._doubt_map:
            return True
        basename = os.path.basename(path)
        for k, v in self._doubt_map.items():
            if k == basename or v.get("filename") == basename:
                return True
        return False

    def _get_doubt_info(self, path: str) -> Optional[Dict[str, str]]:
        if path in self._doubt_map:
            return self._doubt_map[path]
        basename = os.path.basename(path)
        for k, v in self._doubt_map.items():
            if k == basename or v.get("filename") == basename:
                return v
        return None

    def set_files(self, paths: List[str], current: Optional[str] = None,
                  label_lookup=None, doubt_map: Optional[Dict[str, Dict[str, str]]] = None) -> None:
        self._all_paths = paths
        self._current_path = current
        self._label_lookup = label_lookup
        if doubt_map is not None:
            self._doubt_map = doubt_map
        self._refresh_list()

    def _refresh_list(self) -> None:
        self.list_widget.clear()
        self.labels_list.clear()

        displayed_paths = [
            p for p in self._all_paths
            if not self._filter_doubts_only or self._is_path_doubt(p)
        ]

        num_doubts = sum(1 for p in self._all_paths if self._is_path_doubt(p))
        self.btn_filter_all.setText(f"All ({len(self._all_paths)})")
        self.btn_filter_doubts.setText(f"❓ Doubts ({num_doubts})")

        current_idx = -1
        if self._current_path:
            is_cur_doubt = self._is_path_doubt(self._current_path)
            self.btn_quick_doubt.setText("✅ Unmark Doubt" if is_cur_doubt else "❓ Mark Doubt")

        for i, p in enumerate(displayed_paths):
            basename = os.path.basename(p)
            doubt_info = self._get_doubt_info(p)
            is_doubt = doubt_info is not None

            if is_doubt:
                reason = doubt_info.get("reason", "Doubt")
                item_text = f"❓ [DOUBT] {i + 1}. {basename} ({reason})"
            else:
                item_text = f"{i + 1}. {basename}"

            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, p)
            if is_doubt:
                item.setForeground(QColor(255, 140, 0))  # vibrant orange
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                reason = doubt_info.get("reason", "Doubt")
                ts = doubt_info.get("timestamp", "")
                item.setToolTip(f"⚠️ DOUBT: {reason}\nMarked: {ts}\nPath: {p}")
            else:
                item.setToolTip(p)
            self.list_widget.addItem(item)

            label_path = self._label_lookup(p) if self._label_lookup else None
            label_text = os.path.basename(label_path) if label_path else "--  (no label found)"
            if is_doubt:
                label_item_text = f"❓ {i + 1}. {label_text}"
            else:
                label_item_text = f"{i + 1}. {label_text}"

            label_item = QListWidgetItem(label_item_text)
            label_item.setData(Qt.ItemDataRole.UserRole, p)
            if is_doubt:
                label_item.setForeground(QColor(255, 140, 0))
            elif label_path is None:
                label_item.setForeground(QColor(200, 80, 80))
            self.labels_list.addItem(label_item)

            if p == self._current_path:
                current_idx = i
                item.setSelected(True)
                self.list_widget.setCurrentItem(item)
                label_item.setSelected(True)
                self.labels_list.setCurrentItem(label_item)

        if displayed_paths and current_idx >= 0:
            self.counter_label.setText(f"{current_idx + 1} / {len(displayed_paths)}")
        elif displayed_paths:
            self.counter_label.setText(f"- / {len(displayed_paths)}")
        else:
            self.counter_label.setText("- / -")

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        p = item.data(Qt.ItemDataRole.UserRole)
        self._current_path = p
        self._update_quick_doubt_btn()
        self.imageSelected.emit(p)

    def _on_label_item_clicked(self, item: QListWidgetItem) -> None:
        p = item.data(Qt.ItemDataRole.UserRole)
        self._current_path = p
        self._update_quick_doubt_btn()
        self.imageSelected.emit(p)

    def _update_quick_doubt_btn(self) -> None:
        if self._current_path:
            is_doubt = self._is_path_doubt(self._current_path)
            self.btn_quick_doubt.setText("✅ Unmark Doubt" if is_doubt else "❓ Mark Doubt")

    def _on_quick_doubt_clicked(self) -> None:
        if not self._current_path:
            return
        is_doubt = self._is_path_doubt(self._current_path)
        if is_doubt:
            self.doubtToggled.emit(self._current_path, False, "")
        else:
            filename = os.path.basename(self._current_path)
            dlg = DoubtNoteDialog(self, filename, "Doubt")
            if dlg.exec() == QDialog.DialogCode.Accepted:
                self.doubtToggled.emit(self._current_path, True, dlg.reason())

    def _show_context_menu_from_labels(self, pos) -> None:
        item = self.labels_list.itemAt(pos)
        self._show_context_menu_for_item(item, self.labels_list.mapToGlobal(pos))

    def _show_context_menu(self, pos) -> None:
        item = self.list_widget.itemAt(pos)
        self._show_context_menu_for_item(item, self.list_widget.mapToGlobal(pos))

    def _show_context_menu_for_item(self, item: Optional[QListWidgetItem], global_pos) -> None:
        if item is None:
            item = self.list_widget.currentItem()
        if item is None:
            return

        path = item.data(Qt.ItemDataRole.UserRole)
        if not path:
            return

        filename = os.path.basename(path)
        doubt_info = self._get_doubt_info(path)
        is_doubt = doubt_info is not None

        menu = QMenu(self)

        if is_doubt:
            cur_reason = doubt_info.get("reason", "Doubt")
            act_unmark = menu.addAction("✅ Unmark Doubt (Remove from CSV)")
            act_edit = menu.addAction(f"✏️ Edit Doubt Reason (current: '{cur_reason}')...")
            menu.addSeparator()
            act_mark = None
            act_mark_note = None
        else:
            act_mark = menu.addAction("❓ Mark as Doubt")
            act_mark_note = menu.addAction("📝 Mark as Doubt with Note/Reason...")
            menu.addSeparator()
            act_unmark = None
            act_edit = None

        act_copy_name = menu.addAction("📋 Copy Filename")
        act_copy_path = menu.addAction("📄 Copy Full Path")
        menu.addSeparator()
        act_export = menu.addAction("💾 Export Doubts CSV...")

        chosen = menu.exec(global_pos)
        if not chosen:
            return

        if act_mark is not None and chosen == act_mark:
            self.doubtToggled.emit(path, True, "Doubt")
        elif act_mark_note is not None and chosen == act_mark_note:
            dlg = DoubtNoteDialog(self, filename, "Doubt")
            if dlg.exec() == QDialog.DialogCode.Accepted:
                self.doubtToggled.emit(path, True, dlg.reason())
        elif act_unmark is not None and chosen == act_unmark:
            self.doubtToggled.emit(path, False, "")
        elif act_edit is not None and chosen == act_edit:
            cur_reason = doubt_info.get("reason", "Doubt") if doubt_info else "Doubt"
            dlg = DoubtNoteDialog(self, filename, cur_reason)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                self.doubtToggled.emit(path, True, dlg.reason())
        elif chosen == act_copy_name:
            QApplication.clipboard().setText(filename)
        elif chosen == act_copy_path:
            QApplication.clipboard().setText(path)
        elif chosen == act_export:
            self.exportDoubtsRequested.emit()


class ReferenceZoomPanel(QWidget):
    """
    A magnifier panel: shows a cropped, zoomed-in view of the region
    around whichever box is selected (with a little context padding),
    for reading tiny/blurry license-plate text that's hard to make out at
    normal viewer zoom. Independent of the main ImageViewer's zoom/pan.

    - Automatically follows the current selection.
    - Scroll wheel over the panel zooms in/out.
    - Click resets to the default zoom level.
    - +/- buttons and a Fit button are also provided for accessibility.
    """

    DEFAULT_ZOOM = 6.0
    MIN_ZOOM, MAX_ZOOM = 1.0, 40.0
    CONTEXT_PADDING_PX = 14.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source_pixmap: Optional[QPixmap] = None
        self._crop_rect = QRectF()
        self._zoom = self.DEFAULT_ZOOM

        self.setMinimumHeight(200)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QHBoxLayout()
        header.addWidget(QLabel("Reference Zoom"))
        header.addStretch(1)
        self.zoom_label = QLabel(f"{self._zoom:.1f}x")
        self.zoom_label.setMinimumWidth(38)
        header.addWidget(self.zoom_label)

        zoom_out_btn = QToolButton()
        zoom_out_btn.setText("−")
        zoom_out_btn.setToolTip("Zoom out (scroll down over the image)")
        zoom_out_btn.clicked.connect(lambda: self._set_zoom(self._zoom / 1.3))
        header.addWidget(zoom_out_btn)

        zoom_in_btn = QToolButton()
        zoom_in_btn.setText("+")
        zoom_in_btn.setToolTip("Zoom in (scroll up over the image)")
        zoom_in_btn.clicked.connect(lambda: self._set_zoom(self._zoom * 1.3))
        header.addWidget(zoom_in_btn)

        fit_btn = QToolButton()
        fit_btn.setText("Reset")
        fit_btn.setToolTip("Reset to default zoom (or just click the image)")
        fit_btn.clicked.connect(self.reset_zoom)
        header.addWidget(fit_btn)

        layout.addLayout(header)

        self.canvas = QLabel()
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas.setStyleSheet("background-color: #1a1a1a; border: 1px solid #444;")
        self.canvas.setMinimumHeight(160)
        self.canvas.setCursor(Qt.CursorShape.PointingHandCursor)
        self.canvas.setToolTip("Scroll to zoom \u00b7 click to reset zoom")
        self.canvas.installEventFilter(self)
        layout.addWidget(self.canvas, 1)

        hint = QLabel("Follows the selected box \u00b7 scroll to zoom \u00b7 click to reset")
        hint.setStyleSheet("color: gray; font-size: 10px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

    # -- external API -----------------------------------------------------
    def set_source_pixmap(self, pixmap: Optional[QPixmap]) -> None:
        self._source_pixmap = pixmap
        self._render()

    def show_region(self, bbox: Optional[BoundingBox]) -> None:
        """Point the magnifier at `bbox` (with context padding), or at the
        whole image if `bbox` is None (e.g. nothing selected)."""
        if self._source_pixmap is None or self._source_pixmap.isNull():
            self._crop_rect = QRectF()
            self._render()
            return
        if bbox is None:
            self._crop_rect = QRectF(0, 0, self._source_pixmap.width(), self._source_pixmap.height())
        else:
            pad = self.CONTEXT_PADDING_PX
            self._crop_rect = QRectF(bbox.x - pad, bbox.y - pad,
                                      bbox.width + 2 * pad, bbox.height + 2 * pad)
        self._render()

    def reset_zoom(self) -> None:
        self._set_zoom(self.DEFAULT_ZOOM)

    # -- internal -----------------------------------------------------------
    def _set_zoom(self, value: float) -> None:
        self._zoom = max(self.MIN_ZOOM, min(value, self.MAX_ZOOM))
        self.zoom_label.setText(f"{self._zoom:.1f}x")
        self._render()

    def eventFilter(self, obj, event) -> bool:
        if obj is self.canvas:
            if event.type() == QEvent.Type.Wheel:
                factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
                self._set_zoom(self._zoom * factor)
                return True
            if event.type() == QEvent.Type.MouseButtonPress:
                self.reset_zoom()
                return True
        return super().eventFilter(obj, event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render()

    def _render(self) -> None:
        if (self._source_pixmap is None or self._source_pixmap.isNull()
                or self._crop_rect.isEmpty() or self._crop_rect.width() <= 0
                or self._crop_rect.height() <= 0):
            self.canvas.setPixmap(QPixmap())
            self.canvas.setText("No image loaded" if self._source_pixmap is None else "")
            return

        img_rect = QRectF(0, 0, self._source_pixmap.width(), self._source_pixmap.height())
        crop = self._crop_rect.intersected(img_rect)
        if crop.width() <= 0 or crop.height() <= 0:
            return
        cropped = self._source_pixmap.copy(crop.toRect())

        target_w = max(int(cropped.width() * self._zoom), 1)
        target_h = max(int(cropped.height() * self._zoom), 1)
        scaled = cropped.scaled(
            target_w, target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        # Don't let a huge zoom blow past the available canvas space --
        # cap the displayed pixmap to the canvas viewport, the user can
        # still scroll-zoom further and the label re-centers each time.
        avail = self.canvas.size()
        if scaled.width() > avail.width() or scaled.height() > avail.height():
            scaled = scaled.scaled(avail, Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
        self.canvas.setPixmap(scaled)


# ======================================================================
# 10. IMAGE I/O HELPERS (OpenCV / NumPy -> QPixmap)
# ======================================================================

def cv_imread_unicode(path: str) -> np.ndarray:
    """Robustly load an image (including non-ASCII paths) with OpenCV."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"OpenCV could not decode image: {path}")
    return img


def cv_to_qpixmap(img: np.ndarray) -> QPixmap:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = np.ascontiguousarray(img_rgb)
    h, w, ch = img_rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


# ======================================================================
# 11. CONTROLLER GLUE  (protocol used by BBoxGraphicsItem)
# ======================================================================

class AnnotationController:
    """Small interface the graphics items call into; implemented by
    MainWindow. Kept separate so BBoxGraphicsItem doesn't depend on the
    concrete MainWindow class (Dependency Inversion)."""

    model: AnnotationModel

    def select_annotation(self, ann_id: str) -> None:
        raise NotImplementedError

    def commit_geometry_change(self, ann_id: str, old_bbox: BoundingBox, new_bbox: BoundingBox) -> None:
        raise NotImplementedError

    def commit_group_geometry_change(self, changes: List[Tuple[str, BoundingBox, BoundingBox]]) -> None:
        """Commit a batch of (ann_id, old_bbox, new_bbox) changes -- e.g.
        a multi-select drag that moved several boxes at once -- as a
        SINGLE undo/redo step."""
        raise NotImplementedError

    def bbox_item_for(self, ann_id: str) -> Optional["BBoxGraphicsItem"]:
        raise NotImplementedError

    def live_geometry_preview(self, ann_id: str, bbox: BoundingBox) -> None:
        """Called continuously while a box is being dragged/resized (before
        the change is committed to the undo stack), so live-feedback views
        like the Reference Zoom panel can track it in real time."""
        raise NotImplementedError


# ======================================================================
# 12. MAIN WINDOW  (Controller / Orchestration)
# ======================================================================

class MainWindow(QMainWindow, AnnotationController):
    MAX_RECENT = 10

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setAcceptDrops(True)

        self.settings = QSettings(APP_ORG, APP_NAME)
        self._restore_window_geometry()

        # -- state -------------------------------------------------------
        self.model = AnnotationModel()
        self.undo_stack = QUndoStack(self)
        self.bbox_items: Dict[str, BBoxGraphicsItem] = {}
        self.clipboard_annotations: List[Annotation] = []
        self.clipboard_source_image: Optional[str] = None
        self.previous_frame_annotations: List[Annotation] = []
        self.previous_frame_parser: Optional[AnnotationParser] = None
        self.previous_frame_path: Optional[str] = None
        self.previous_frame_image_size: Optional[Tuple[float, float]] = None
        self.paste_offset_x: float = float(self.settings.value("paste_offset_x", 0.0))
        self.paste_offset_y: float = float(self.settings.value("paste_offset_y", 0.0))

        self.current_image_path: Optional[str] = None
        self.current_annotation_path: Optional[str] = None
        self.current_annotation_source: Optional[str] = None
        self.current_label_dir: Optional[str] = None
        self.current_label2_dir: Optional[str] = None
        self.folder_images: List[str] = []
        self.doubts: Dict[str, Dict[str, str]] = {}
        self.doubts_csv_path: Optional[str] = None
        self.auto_save_timer = QTimer(self)
        self.auto_save_timer.setSingleShot(True)
        self.auto_save_timer.setInterval(1000)
        self.auto_save_timer.timeout.connect(self._auto_save_if_needed)

        # -- scene / view --------------------------------------------------
        self.scene = QGraphicsScene(self)
        self.pixmap_item = QGraphicsPixmapItem()
        self.pixmap_item.setZValue(-1)
        self.scene.addItem(self.pixmap_item)
        self.view = ImageViewer(self.scene, self)
        self.setCentralWidget(self.view)

        self.scene.selectionChanged.connect(self._on_scene_selection_changed)
        self.view.boxCreated.connect(self._on_box_created)

        # -- side panels -----------------------------------------------------
        self.properties_panel = PropertiesPanel()
        self._add_dock("Properties", self.properties_panel, Qt.DockWidgetArea.RightDockWidgetArea)

        self.batch_text_panel = BatchTextEditPanel()
        self.batch_text_dock = self._add_dock(
            "Edit All Text", self.batch_text_panel, Qt.DockWidgetArea.RightDockWidgetArea
        )

        self.list_panel = AnnotationListPanel()
        self._add_dock("Annotation List", self.list_panel, Qt.DockWidgetArea.RightDockWidgetArea)

        self.nav_panel = FileNavigatorPanel()
        self._add_dock("Files", self.nav_panel, Qt.DockWidgetArea.LeftDockWidgetArea)

        self.reference_panel = ReferenceZoomPanel()
        self._add_dock("Reference Zoom", self.reference_panel, Qt.DockWidgetArea.RightDockWidgetArea)

        self.properties_panel.labelChanged.connect(self._on_label_changed)
        self.properties_panel.textChanged.connect(self._on_text_changed)
        self.properties_panel.geometryChanged.connect(self._on_geometry_field_changed)
        self.properties_panel.lockToggled.connect(self._on_lock_toggled)

        self.batch_text_panel.applyRequested.connect(self._apply_batch_text_edits)
        self.batch_text_panel.saveRequested.connect(self._apply_batch_text_edits_and_save)
        self.batch_text_panel.refreshRequested.connect(self._refresh_batch_text_panel)

        self.list_panel.selectRequested.connect(self.select_annotation)
        self.list_panel.deleteRequested.connect(self._delete_annotation_by_id)

        self.nav_panel.imageSelected.connect(self._open_image_path)
        self.nav_panel.prev_btn.clicked.connect(self.prev_image)
        self.nav_panel.next_btn.clicked.connect(self.next_image)
        self.nav_panel.doubtToggled.connect(self._on_doubt_toggled)
        self.nav_panel.exportDoubtsRequested.connect(self._export_doubts_csv)

        # -- model signal wiring -------------------------------------------
        self.model.annotationsReset.connect(self._on_model_reset)
        self.model.annotationAdded.connect(self._on_model_annotation_added)
        self.model.annotationRemoved.connect(self._on_model_annotation_removed)
        self.model.annotationUpdated.connect(self._on_model_annotation_updated)
        self.model.selectionChanged.connect(self._on_model_selection_changed)
        self.model.dirtyChanged.connect(self._on_dirty_changed)

        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self._update_title()
        self._suppress_selection_echo = False
        self._scene_originated_selection = False

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _add_dock(self, title: str, widget: QWidget, area) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(area, dock)
        return dock

    def _build_actions(self) -> None:
        self.act_open_image = QAction("Open Image...", self, shortcut=QKeySequence("Ctrl+O"))
        self.act_open_image.triggered.connect(self.open_image_dialog)

        self.act_open_folder = QAction("Open Folder...", self)
        self.act_open_folder.triggered.connect(self.open_folder_dialog)

        self.act_load_annotation = QAction("Load Annotation...", self, shortcut=QKeySequence("Ctrl+L"))
        self.act_load_annotation.triggered.connect(self.load_annotation_dialog)

        self.act_load_labels2_annotation = QAction("Toggle labels / labels2 Annotation", self, shortcut=QKeySequence("M"))
        self.act_load_labels2_annotation.triggered.connect(self.load_labels2_annotation)

        self.act_reload_annotation = QAction("Reload Annotation", self)
        self.act_reload_annotation.triggered.connect(self.reload_annotation)

        self.act_save = QAction("Save", self, shortcut=QKeySequence("Ctrl+S"))
        self.act_save.triggered.connect(self.save_annotation)

        self.act_save_as = QAction("Save As...", self, shortcut=QKeySequence("Ctrl+Shift+S"))
        self.act_save_as.triggered.connect(self.save_annotation_as)

        self.act_auto_save = QAction("Auto Save (every 1s)", self, checkable=True)
        self.act_auto_save.setChecked(self.settings.value("auto_save", True, type=bool))
        self.act_auto_save.triggered.connect(self._on_auto_save_toggled)

        self.act_export_json = QAction("Export JSON...", self)
        self.act_export_json.triggered.connect(self.export_json)

        self.act_undo = self.undo_stack.createUndoAction(self, "Undo")
        self.act_undo.setShortcut(QKeySequence("Ctrl+Z"))
        self.act_redo = self.undo_stack.createRedoAction(self, "Redo")
        self.act_redo.setShortcut(QKeySequence("Ctrl+Y"))

        self.act_delete = QAction("Delete Box", self, shortcut=QKeySequence("Delete"))
        self.act_delete.triggered.connect(self._delete_selected)

        self.act_copy = QAction("Copy Box(es)", self, shortcut=QKeySequence("Ctrl+C"))
        self.act_copy.triggered.connect(self._copy_selected)

        self.act_paste = QAction("Paste Box(es)", self, shortcut=QKeySequence("Ctrl+V"))
        self.act_paste.triggered.connect(lambda checked=False: self._paste_clipboard())

        self.act_paste_offset = QAction("Paste with Offset...", self, shortcut=QKeySequence("Ctrl+Alt+V"))
        self.act_paste_offset.triggered.connect(self._paste_with_offset_dialog)

        self.act_copy_previous_frame = QAction("Copy from Previous Frame", self)
        self.act_copy_previous_frame.setShortcuts([QKeySequence("Ctrl+Shift+V"), QKeySequence("P")])
        self.act_copy_previous_frame.triggered.connect(lambda checked=False: self._copy_from_previous_frame())

        self.act_duplicate = QAction("Duplicate Box", self, shortcut=QKeySequence("Ctrl+D"))
        self.act_duplicate.triggered.connect(self._duplicate_selected)

        self.act_toggle_lock = QAction("Lock/Unlock Box", self, shortcut=QKeySequence("Ctrl+K"))
        self.act_toggle_lock.triggered.connect(self._toggle_lock_selected)

        self.act_edit_selected_text = QAction("Edit Selected Box Text", self, shortcut=QKeySequence("U"))
        self.act_edit_selected_text.triggered.connect(self._focus_selected_text_editor)

        self.act_new_box = QAction("New Bounding Box", self, checkable=True, shortcut=QKeySequence("N"))
        self.act_new_box.triggered.connect(self._on_new_box_toggled)

        self.act_select_all = QAction("Select All Boxes", self, shortcut=QKeySequence("Ctrl+A"))
        self.act_select_all.triggered.connect(self._select_all_boxes)

        self.act_deselect_all = QAction("Deselect All", self, shortcut=QKeySequence("Escape"))
        self.act_deselect_all.triggered.connect(self._deselect_all_boxes)

        self.act_fit = QAction("Fit Image", self, shortcut=QKeySequence("F"))
        self.act_fit.triggered.connect(self.fit_image)

        self.act_reset_zoom = QAction("Reset Zoom", self, shortcut=QKeySequence("R"))
        self.act_reset_zoom.triggered.connect(self.view.reset_zoom)

        self.act_prev_image = QAction("Previous Image", self)
        self.act_prev_image.setShortcuts([QKeySequence("PgUp"), QKeySequence("Left")])
        self.act_prev_image.triggered.connect(self.prev_image)

        self.act_next_image = QAction("Next Image", self)
        self.act_next_image.setShortcuts([QKeySequence("PgDown"), QKeySequence("Right")])
        self.act_next_image.triggered.connect(self.next_image)

        self.act_save_next = QAction("Save && Next", self, shortcut=QKeySequence("Ctrl+Return"))
        self.act_save_next.triggered.connect(self.save_and_next)

        self.act_toggle_doubt = QAction("Mark / Unmark Doubt", self, shortcut=QKeySequence("Ctrl+Shift+D"))
        self.act_toggle_doubt.triggered.connect(self._toggle_current_doubt)

        self.act_export_doubts = QAction("Export Doubts CSV...", self)
        self.act_export_doubts.triggered.connect(self._export_doubts_csv)

        self.act_next_doubt = QAction("Next Doubt Crop", self, shortcut=QKeySequence("Ctrl+Alt+Right"))
        self.act_next_doubt.triggered.connect(self.next_doubt_image)

        self.act_prev_doubt = QAction("Previous Doubt Crop", self, shortcut=QKeySequence("Ctrl+Alt+Left"))
        self.act_prev_doubt.triggered.connect(self.prev_doubt_image)

        self.act_about = QAction("About", self)
        self.act_about.triggered.connect(self.show_about_dialog)

        self.act_shortcuts = QAction("Keyboard Shortcuts", self, shortcut=QKeySequence("F1"))
        self.act_shortcuts.triggered.connect(self.show_shortcuts_dialog)

        self.recent_actions: List[QAction] = []

    def _build_menus(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.act_open_image)
        file_menu.addAction(self.act_open_folder)
        file_menu.addAction(self.act_load_annotation)
        file_menu.addAction(self.act_load_labels2_annotation)
        file_menu.addAction(self.act_reload_annotation)
        file_menu.addSeparator()
        file_menu.addAction(self.act_save)
        file_menu.addAction(self.act_save_as)
        file_menu.addAction(self.act_auto_save)
        file_menu.addAction(self.act_export_json)
        file_menu.addAction(self.act_export_doubts)
        file_menu.addSeparator()
        self.recent_menu = file_menu.addMenu("Recent Files")
        self._refresh_recent_menu()
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction(self.act_undo)
        edit_menu.addAction(self.act_redo)
        edit_menu.addSeparator()
        edit_menu.addAction(self.act_copy)
        edit_menu.addAction(self.act_paste)
        edit_menu.addAction(self.act_paste_offset)
        edit_menu.addAction(self.act_copy_previous_frame)
        edit_menu.addAction(self.act_duplicate)
        edit_menu.addAction(self.act_delete)
        edit_menu.addAction(self.act_toggle_lock)
        edit_menu.addAction(self.act_edit_selected_text)
        edit_menu.addSeparator()
        edit_menu.addAction(self.act_select_all)
        edit_menu.addAction(self.act_deselect_all)

        box_menu = menubar.addMenu("&Box")
        box_menu.addAction(self.act_new_box)

        view_menu = menubar.addMenu("&View")
        view_menu.addAction(self.act_fit)
        view_menu.addAction(self.act_reset_zoom)

        nav_menu = menubar.addMenu("&Navigate")
        nav_menu.addAction(self.act_prev_image)
        nav_menu.addAction(self.act_next_image)
        nav_menu.addAction(self.act_save_next)
        nav_menu.addSeparator()
        nav_menu.addAction(self.act_toggle_doubt)
        nav_menu.addAction(self.act_prev_doubt)
        nav_menu.addAction(self.act_next_doubt)

        help_menu = menubar.addMenu("&Help")
        help_menu.addAction(self.act_about)
        help_menu.addAction(self.act_shortcuts)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        for action in (self.act_open_image, self.act_open_folder, self.act_load_annotation,
                       self.act_save, self.act_save_as, self.act_auto_save):
            toolbar.addAction(action)
        toolbar.addSeparator()
        for action in (self.act_undo, self.act_redo):
            toolbar.addAction(action)
        toolbar.addSeparator()
        for action in (self.act_new_box, self.act_delete, self.act_duplicate,
                       self.act_copy, self.act_paste, self.act_copy_previous_frame, self.act_toggle_lock,
                       self.act_toggle_doubt,
                       self.act_select_all, self.act_deselect_all):
            toolbar.addAction(action)
        toolbar.addSeparator()
        for action in (self.act_fit, self.act_reset_zoom, self.act_prev_image, self.act_next_image):
            toolbar.addAction(action)

    # ------------------------------------------------------------------
    # File: image loading
    # ------------------------------------------------------------------
    def open_image_dialog(self) -> None:
        start_dir = self.settings.value("last_dir", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", start_dir,
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)"
        )
        if path:
            self._open_image_path(path)

    def open_folder_dialog(self) -> None:
        start_dir = self.settings.value("last_dir", str(Path.home()))
        folder = QFileDialog.getExistingDirectory(self, "Open Folder", start_dir)
        if folder:
            self._load_folder(folder)

    def _resolve_image_and_label_dirs(self, folder: str) -> Tuple[str, Optional[str], Optional[str]]:
        base = Path(folder)
        if base.is_file():
            base = base.parent

        candidates: List[Tuple[Path, Optional[Path], Optional[Path]]] = []
        for image_dir in (base, base / "crops", base / "images"):
            if image_dir.exists() and image_dir.is_dir():
                label_dir = None
                label2_dir = None
                if (base / "labels").is_dir():
                    label_dir = base / "labels"
                elif (image_dir.parent / "labels").is_dir():
                    label_dir = image_dir.parent / "labels"
                if (base / "labels2").is_dir():
                    label2_dir = base / "labels2"
                elif (image_dir.parent / "labels2").is_dir():
                    label2_dir = image_dir.parent / "labels2"
                candidates.append((image_dir, label_dir, label2_dir))

        if not candidates:
            return str(base), None, None

        for image_dir, label_dir, label2_dir in candidates:
            if image_dir.name.lower() in {"crops", "images"}:
                return (
                    str(image_dir),
                    str(label_dir) if label_dir else None,
                    str(label2_dir) if label2_dir else None,
                )

        image_dir, label_dir, label2_dir = candidates[0]
        return (
            str(image_dir),
            str(label_dir) if label_dir else None,
            str(label2_dir) if label2_dir else None,
        )

    def _load_folder(self, folder: str) -> None:
        image_dir, label_dir, label2_dir = self._resolve_image_and_label_dirs(folder)
        self.current_label_dir = label_dir
        self.current_label2_dir = label2_dir

        self.doubts_csv_path = self._resolve_doubts_csv_path(folder)
        self._load_doubts_csv(self.doubts_csv_path)

        files = sorted(
            str(p) for p in Path(image_dir).iterdir()
            if p.suffix.lower() in SUPPORTED_IMAGE_EXTS
        )
        self.folder_images = files
        self.nav_panel.set_files(files, self.current_image_path,
                                  label_lookup=self._find_matching_annotation,
                                  doubt_map=self.doubts)
        if files and not self.current_image_path:
            self._open_image_path(files[0])

    def _open_image_path(self, path: str) -> None:
        if not self._confirm_discard_changes():
            return

        # Snapshot previous frame annotations before switching image
        if self.model.all():
            self.previous_frame_annotations = [a.clone() for a in self.model.all()]
            self.previous_frame_parser = self.model.parser
            self.previous_frame_path = self.current_image_path
            self.previous_frame_image_size = self._current_image_size()

        try:
            img = cv_imread_unicode(path)
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Image", str(e))
            return

        pixmap = cv_to_qpixmap(img)
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(QRectF(0, 0, pixmap.width(), pixmap.height()))
        self.reference_panel.set_source_pixmap(pixmap)
        self.current_image_path = path
        self.settings.setValue("last_dir", str(Path(path).parent))
        self._add_recent_file(path)

        folder = str(Path(path).parent)
        if not self.folder_images or Path(self.folder_images[0]).parent != Path(folder):
            self._load_folder(folder)
        else:
            if not self.doubts_csv_path:
                self.doubts_csv_path = self._resolve_doubts_csv_path(folder)
                self._load_doubts_csv(self.doubts_csv_path)
            self.nav_panel.set_files(self.folder_images, path,
                                      label_lookup=self._find_matching_annotation,
                                      doubt_map=self.doubts)

        self.model.clear()
        self.undo_stack.clear()
        self.view.fit_to_image(self.scene.sceneRect())

        auto_path = self._find_matching_annotation(path, self.current_label_dir)
        if auto_path:
            self._load_annotation_path(auto_path)
        else:
            self.current_annotation_path = None
            if self.previous_frame_parser is not None:
                self.model.parser = self.previous_frame_parser

        self._update_title()
        if self.model.all():
            self.status.showMessage(f"Loaded image: {path} ({len(self.model.all())} box(es))", 4000)
        elif self.previous_frame_annotations:
            self.status.showMessage(
                f"Loaded image: {path} (Empty - press Ctrl+Shift+V or P to copy {len(self.previous_frame_annotations)} box(es) from previous frame)",
                5000
            )
        else:
            self.status.showMessage(f"Loaded image: {path}", 4000)

    @staticmethod
    def _find_matching_annotation(
        image_path: str, label_dir: Optional[str] = None,
        only_label_dir: bool = False
    ) -> Optional[str]:
        """
        Look for an annotation file with the same filename stem as the
        image.

        If `only_label_dir` is True, only the explicitly supplied `label_dir`
        is searched. Otherwise, we search:
          1. The same folder as the image.
          2. A sibling `labels/` folder for `images/` or `crops/` layouts.
          3. An explicitly resolved label directory, if one was supplied.

        This supports parent-folder openings where the images live under a
        `crops/` or `images/` subfolder and the labels live under a sibling
        `labels/` directory.
        """
        stem = Path(image_path).stem
        image_dir = Path(image_path).parent
        if only_label_dir:
            search_dirs = []
            if label_dir:
                resolved_label_dir = Path(label_dir)
                if resolved_label_dir.is_dir():
                    search_dirs.append(resolved_label_dir)
        else:
            search_dirs = [image_dir]
            if image_dir.name.lower() in {"images", "crops"}:
                sibling_labels = image_dir.parent / "labels"
                if sibling_labels.is_dir():
                    search_dirs.append(sibling_labels)
            if label_dir:
                resolved_label_dir = Path(label_dir)
                if resolved_label_dir.is_dir():
                    search_dirs.append(resolved_label_dir)

        seen_dirs = set()
        for directory in search_dirs:
            if directory in seen_dirs:
                continue
            seen_dirs.add(directory)
            for parser in ParserRegistry.all_parsers():
                for ext in parser.extensions:
                    candidate = directory / (stem + ext)
                    if candidate.is_file():
                        return str(candidate)
        return None

    # ------------------------------------------------------------------
    # File: annotation loading / saving
    # ------------------------------------------------------------------
    def load_annotation_dialog(self) -> None:
        start_dir = str(Path(self.current_image_path).parent) if self.current_image_path else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Annotation", start_dir, ParserRegistry.save_dialog_filter()
        )
        if path:
            self._load_annotation_path(path)

    def load_labels2_annotation(self) -> None:
        if not self.current_image_path:
            QMessageBox.information(self, "Toggle labels / labels2 Annotation", "Open an image first.")
            return

        labels2_path = None
        if self.current_label2_dir and Path(self.current_label2_dir).is_dir():
            labels2_path = self._find_matching_annotation(
                self.current_image_path, self.current_label2_dir, only_label_dir=True
            )

        labels_path = None
        if self.current_label_dir and Path(self.current_label_dir).is_dir():
            labels_path = self._find_matching_annotation(
                self.current_image_path, self.current_label_dir, only_label_dir=True
            )

        if self.current_annotation_source == "labels2":
            if labels_path:
                self._load_annotation_path(labels_path)
                self.status.showMessage("Loaded labels annotation for current image.", 4000)
                return
            if labels2_path:
                self._load_annotation_path(labels2_path)
                self.status.showMessage("Loaded labels2 annotation for current image.", 4000)
                return
        else:
            if labels2_path:
                self._load_annotation_path(labels2_path)
                self.status.showMessage("Loaded labels2 annotation for current image.", 4000)
                return
            if labels_path:
                self._load_annotation_path(labels_path)
                self.status.showMessage("Loaded labels annotation for current image.", 4000)
                return

        if not self.current_label2_dir or not Path(self.current_label2_dir).is_dir():
            if not labels_path:
                QMessageBox.information(
                    self, "Toggle labels / labels2 Annotation",
                    "No labels2 or labels folder was found for the current image folder."
                )
                return
            self._load_annotation_path(labels_path)
            self.status.showMessage("Loaded labels annotation for current image.", 4000)
            return

        if labels2_path is None and labels_path is None:
            QMessageBox.information(
                self, "Toggle labels / labels2 Annotation",
                "No matching annotation file was found in labels2 or labels for this image."
            )
            return

        if labels2_path:
            self._load_annotation_path(labels2_path)
            self.status.showMessage("Loaded labels2 annotation for current image.", 4000)
            return
        if labels_path:
            self._load_annotation_path(labels_path)
            self.status.showMessage("Loaded labels annotation for current image.", 4000)
            return

    def _current_image_size(self) -> Optional[Tuple[float, float]]:
        """(width, height) in pixels of the currently loaded image, or
        None if no image is open. Needed by normalized-coordinate parsers
        (e.g. YoloCharOCRAnnotationParser) to convert to/from the
        editor's pixel-space BoundingBox."""
        pixmap = self.pixmap_item.pixmap()
        if pixmap.isNull():
            return None
        return (float(pixmap.width()), float(pixmap.height()))

    def _load_annotation_path(self, path: str) -> None:
        parser = ParserRegistry.for_path(path)
        if parser is None:
            QMessageBox.warning(self, "Unsupported Format",
                                 f"No parser registered for extension '{Path(path).suffix}'.")
            return
        try:
            annotations = parser.load(path, image_size=self._current_image_size())
        except AnnotationParseError as e:
            QMessageBox.critical(self, "Annotation Parse Error", str(e))
            return
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Annotation", f"{e}\n\n{traceback.format_exc()}")
            return

        self.model.reset(annotations, path, parser)
        self.undo_stack.clear()
        self.current_annotation_path = path
        if self.current_label2_dir and Path(path).parent == Path(self.current_label2_dir):
            self.current_annotation_source = "labels2"
        elif self.current_label_dir and Path(path).parent == Path(self.current_label_dir):
            self.current_annotation_source = "labels"
        else:
            self.current_annotation_source = None
        self._update_title()
        self.status.showMessage(f"Loaded {len(annotations)} annotation(s) from {path}", 4000)

    def reload_annotation(self) -> None:
        if not self.current_annotation_path:
            QMessageBox.information(self, "Reload Annotation", "No annotation file is currently loaded.")
            return
        if not self._confirm_discard_changes():
            return
        self._load_annotation_path(self.current_annotation_path)

    def save_annotation(self) -> None:
        if self.model.parser is None:
            self.save_annotation_as()
            return

        target_path = self._target_annotation_path()
        if target_path is None:
            self.save_annotation_as()
            return

        self._write_annotation(target_path, self.model.parser)

    def _annotation_path_in_labels(self) -> Optional[str]:
        if not self.current_image_path or not self.current_label_dir or self.model.parser is None:
            return None
        image_stem = Path(self.current_image_path).stem
        suffix = Path(self.current_annotation_path).suffix if self.current_annotation_path else ""
        if not suffix and self.model.parser.extensions:
            suffix = self.model.parser.extensions[0]
        if not suffix:
            suffix = ".json"
        target = Path(self.current_label_dir) / f"{image_stem}{suffix}"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return str(target)

    def _target_annotation_path(self) -> Optional[str]:
        if self.current_annotation_source == "labels2":
            labels_target = self._annotation_path_in_labels()
            if labels_target:
                return labels_target
        if self.model.source_path:
            return self.model.source_path
        return self._default_annotation_path_for_current_image()

    def save_annotation_as(self) -> None:
        start_dir = str(Path(self.current_image_path).parent) if self.current_image_path else str(Path.home())
        default_name = str(Path(start_dir) / (Path(self.current_image_path).stem if self.current_image_path else "annotation"))
        path, chosen_filter = QFileDialog.getSaveFileName(
            self, "Save Annotation As", default_name, ParserRegistry.save_dialog_filter()
        )
        if not path:
            return
        # The dialog's chosen filter unambiguously tells us the format the
        # user picked, so it takes priority. This matters when two parsers
        # share an extension (e.g. two '.txt' conventions): content-sniffing
        # in ParserRegistry.for_path() can't help here since the target file
        # doesn't exist yet, so we resolve by filter text first and only
        # fall back to extension-based lookup if that fails.
        parser = None
        for p in ParserRegistry.all_parsers():
            if p.file_filter() == chosen_filter:
                parser = p
                break
        if parser is None:
            parser = ParserRegistry.for_path(path)
        if parser is None:
            QMessageBox.warning(self, "Unsupported Format", "Could not determine annotation format for that path.")
            return
        if not any(path.lower().endswith(ext) for ext in parser.extensions):
            path += parser.extensions[0]
        self._write_annotation(path, parser)

    def _write_annotation(self, path: str, parser: AnnotationParser) -> None:
        try:
            parser.save(path, self.model.all(), image_size=self._current_image_size())
        except Exception as e:
            QMessageBox.critical(self, "Error Saving Annotation", f"{e}\n\n{traceback.format_exc()}")
            return
        self.model.source_path = path
        self.model.parser = parser
        self.current_annotation_path = path
        if self.current_label2_dir and Path(path).parent == Path(self.current_label2_dir):
            self.current_annotation_source = "labels2"
        elif self.current_label_dir and Path(path).parent == Path(self.current_label_dir):
            self.current_annotation_source = "labels"
        else:
            self.current_annotation_source = None
        self.model.mark_saved()
        self._update_title()
        self.status.showMessage(f"Saved {len(self.model.all())} annotation(s) to {path}", 4000)

    def _on_auto_save_toggled(self, checked: bool) -> None:
        self.settings.setValue("auto_save", checked)
        if checked and self.model.is_dirty():
            self.auto_save_timer.start()
        else:
            self.auto_save_timer.stop()
        self.status.showMessage(f"Auto-save {'enabled (every 1s)' if checked else 'disabled'}", 2500)

    def _auto_save_if_needed(self) -> None:
        if not self.act_auto_save.isChecked() or not self.model.is_dirty():
            return
        if self.model.parser is None:
            if self.previous_frame_parser is not None:
                self.model.parser = self.previous_frame_parser
            else:
                self.model.parser = ParserRegistry.for_extension(".txt") or ParserRegistry.for_extension(".json")
        target_path = self._target_annotation_path()
        parser = self.model.parser
        if not target_path or parser is None:
            return
        self._write_annotation(target_path, parser)

    def _default_annotation_path_for_current_image(self) -> Optional[str]:
        if not self.current_image_path:
            return None
        parser = self.model.parser or ParserRegistry.for_extension(".json")
        if parser is None:
            return None
        ext = parser.extensions[0] if parser.extensions else ".json"
        return str(Path(self.current_image_path).with_suffix(ext))

    def export_json(self) -> None:
        start_dir = str(Path(self.current_image_path).parent) if self.current_image_path else str(Path.home())
        default_name = str(Path(start_dir) / (Path(self.current_image_path).stem if self.current_image_path else "annotation")) + ".json"
        path, _ = QFileDialog.getSaveFileName(self, "Export JSON", default_name, "JSON (*.json)")
        if not path:
            return
        json_parser = ParserRegistry.for_extension(".json")
        try:
            json_parser.save(path, self.model.all(), image_size=self._current_image_size())
        except Exception as e:
            QMessageBox.critical(self, "Error Exporting JSON", str(e))
            return
        self.status.showMessage(f"Exported JSON to {path}", 4000)

    def _confirm_discard_changes(self) -> bool:
        if not self.model.is_dirty():
            return True
        resp = QMessageBox.question(
            self, "Unsaved Changes",
            "The current annotation has unsaved changes. Discard them?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
        )
        if resp == QMessageBox.StandardButton.Save:
            self.save_annotation()
            return True
        return resp == QMessageBox.StandardButton.Discard

    # ------------------------------------------------------------------
    # Doubt management & CSV persistence
    # ------------------------------------------------------------------
    def _resolve_doubts_csv_path(self, folder: str) -> str:
        base = Path(folder)
        candidates = [
            base / "doubts.csv",
            base.parent / "doubts.csv",
            base / "crops" / "doubts.csv",
        ]
        if self.current_label_dir:
            candidates.append(Path(self.current_label_dir).parent / "doubts.csv")
        for c in candidates:
            if c.is_file():
                return str(c)
        return str(base / "doubts.csv")

    def _load_doubts_csv(self, path: Optional[str]) -> None:
        self.doubts = {}
        if not path or not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_path = row.get("image_path") or ""
                    img_name = row.get("filename") or row.get("image_name") or ""
                    key = img_path if img_path else img_name
                    if key:
                        self.doubts[key] = {
                            "filename": img_name or os.path.basename(img_path),
                            "image_path": img_path,
                            "label_path": row.get("label_path", ""),
                            "ocr_text": row.get("ocr_text", ""),
                            "status": row.get("status", "Doubt"),
                            "reason": row.get("reason", "Doubt"),
                            "timestamp": row.get("timestamp", ""),
                        }
        except Exception as e:
            print(f"Error loading doubts CSV {path}: {e}")

    def _save_doubts_csv(self, path: Optional[str] = None) -> None:
        target = path or self.doubts_csv_path
        if not target:
            if self.current_image_path:
                target = str(Path(self.current_image_path).parent / "doubts.csv")
                self.doubts_csv_path = target
            else:
                return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
            fieldnames = ["filename", "image_path", "label_path", "ocr_text", "status", "reason", "timestamp"]
            with open(target, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for d in self.doubts.values():
                    writer.writerow({
                        "filename": d.get("filename", ""),
                        "image_path": d.get("image_path", ""),
                        "label_path": d.get("label_path", ""),
                        "ocr_text": d.get("ocr_text", ""),
                        "status": d.get("status", "Doubt"),
                        "reason": d.get("reason", "Doubt"),
                        "timestamp": d.get("timestamp", ""),
                    })
        except Exception as e:
            QMessageBox.warning(self, "Error Saving Doubts CSV", f"Could not save {target}: {e}")

    def _on_doubt_toggled(self, image_path: str, is_doubt: bool, reason: str = "Doubt") -> None:
        filename = os.path.basename(image_path)
        label_path = self._find_matching_annotation(image_path, self.current_label_dir) or ""

        if is_doubt:
            ocr_text = ""
            if image_path == self.current_image_path:
                ocr_text = " ".join(a.text for a in self.model.all() if a.text) or " ".join(a.label for a in self.model.all())
            elif label_path and os.path.isfile(label_path):
                try:
                    p = ParserRegistry.for_path(label_path)
                    if p:
                        anns = p.load(label_path)
                        ocr_text = " ".join(a.text for a in anns if a.text) or " ".join(a.label for a in anns)
                except Exception:
                    pass

            self.doubts[image_path] = {
                "filename": filename,
                "image_path": image_path,
                "label_path": label_path,
                "ocr_text": ocr_text,
                "status": "Doubt",
                "reason": reason or "Doubt",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._save_doubts_csv()
            self.nav_panel.set_doubts(self.doubts)
            csv_name = Path(self.doubts_csv_path).name if self.doubts_csv_path else "doubts.csv"
            self.status.showMessage(f"Marked '{filename}' as Doubt: '{reason}' (Saved to {csv_name})", 4000)
        else:
            self.doubts.pop(image_path, None)
            keys_to_remove = [k for k, v in self.doubts.items() if k == filename or v.get("filename") == filename]
            for k in keys_to_remove:
                self.doubts.pop(k, None)
            self._save_doubts_csv()
            self.nav_panel.set_doubts(self.doubts)
            csv_name = Path(self.doubts_csv_path).name if self.doubts_csv_path else "doubts.csv"
            self.status.showMessage(f"Removed Doubt for '{filename}' (Updated {csv_name})", 3500)

    def _toggle_current_doubt(self) -> None:
        if not self.current_image_path:
            self.status.showMessage("No image currently open", 2000)
            return
        is_doubt = self.nav_panel._is_path_doubt(self.current_image_path)
        filename = os.path.basename(self.current_image_path)
        if is_doubt:
            self._on_doubt_toggled(self.current_image_path, False, "")
        else:
            dlg = DoubtNoteDialog(self, filename, "Doubt")
            if dlg.exec() == QDialog.DialogCode.Accepted:
                self._on_doubt_toggled(self.current_image_path, True, dlg.reason())

    def _export_doubts_csv(self) -> None:
        start_dir = str(Path(self.doubts_csv_path).parent) if self.doubts_csv_path else (
            str(Path(self.current_image_path).parent) if self.current_image_path else str(Path.home())
        )
        default_path = str(Path(start_dir) / "doubts.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Export Doubts CSV", default_path, "CSV Files (*.csv)")
        if not path:
            return
        self._save_doubts_csv(path)
        self.status.showMessage(f"Exported {len(self.doubts)} doubt item(s) to {path}", 4000)

    def next_doubt_image(self) -> None:
        if not self.folder_images:
            return
        doubt_paths = [p for p in self.folder_images if self.nav_panel._is_path_doubt(p)]
        if not doubt_paths:
            self.status.showMessage("No doubt images found in folder", 2500)
            return
        if self.current_image_path in doubt_paths:
            cur_idx = doubt_paths.index(self.current_image_path)
            next_idx = (cur_idx + 1) % len(doubt_paths)
        else:
            next_idx = 0
        self._open_image_path(doubt_paths[next_idx])

    def prev_doubt_image(self) -> None:
        if not self.folder_images:
            return
        doubt_paths = [p for p in self.folder_images if self.nav_panel._is_path_doubt(p)]
        if not doubt_paths:
            self.status.showMessage("No doubt images found in folder", 2500)
            return
        if self.current_image_path in doubt_paths:
            cur_idx = doubt_paths.index(self.current_image_path)
            prev_idx = (cur_idx - 1) % len(doubt_paths)
        else:
            prev_idx = len(doubt_paths) - 1
        self._open_image_path(doubt_paths[prev_idx])

    # ------------------------------------------------------------------
    # Recent files
    # ------------------------------------------------------------------
    def _add_recent_file(self, path: str) -> None:
        recents = self.settings.value("recent_files", [], type=list)
        recents = [r for r in recents if r != path]
        recents.insert(0, path)
        recents = recents[: self.MAX_RECENT]
        self.settings.setValue("recent_files", recents)
        self._refresh_recent_menu()

    def _refresh_recent_menu(self) -> None:
        self.recent_menu.clear()
        recents = self.settings.value("recent_files", [], type=list)
        if not recents:
            empty = self.recent_menu.addAction("(none)")
            empty.setEnabled(False)
            return
        for path in recents:
            action = QAction(path, self)
            action.triggered.connect(lambda checked=False, p=path: self._open_image_path(p))
            self.recent_menu.addAction(action)

    # ------------------------------------------------------------------
    # Folder navigation
    # ------------------------------------------------------------------
    def next_image(self) -> None:
        self._step_image(1)

    def save_and_next(self) -> None:
        """Fast validation loop: save the current annotation, then jump to
        the next image, in one keystroke (Ctrl+Enter)."""
        if self.model.is_dirty():
            self.save_annotation()
            if self.model.is_dirty():  # save failed or was cancelled
                return
        self.next_image()

    def prev_image(self) -> None:
        self._step_image(-1)

    def _step_image(self, direction: int) -> None:
        if not self.folder_images or not self.current_image_path:
            return
        try:
            idx = self.folder_images.index(self.current_image_path)
        except ValueError:
            return
        new_idx = idx + direction
        if 0 <= new_idx < len(self.folder_images):
            self._open_image_path(self.folder_images[new_idx])

    # ------------------------------------------------------------------
    # Drag & drop
    # ------------------------------------------------------------------
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if Path(path).suffix.lower() in SUPPORTED_IMAGE_EXTS:
            self._open_image_path(path)

    # ------------------------------------------------------------------
    # View controls
    # ------------------------------------------------------------------
    def fit_image(self) -> None:
        self.view.fit_to_image(self.scene.sceneRect())

    def show_about_dialog(self) -> None:
        QMessageBox.about(
            self, f"About {APP_NAME}",
            f"<b>{APP_NAME}</b> v{APP_VERSION}<br><br>"
            "A manual review/correction tool for existing OCR text and "
            "bounding-box annotations.<br><br>"
            "<b>This application performs no object detection or AI "
            "inference of any kind.</b> Detection is assumed to have "
            "already been done by an upstream system; this tool only "
            "loads, displays, edits, and saves the results a human "
            "operator verifies.<br><br>"
            "Built with PyQt6, OpenCV, and NumPy."
        )

    def show_shortcuts_dialog(self) -> None:
        rows = [
            ("Ctrl+O", "Open Image"),
            ("Ctrl+L", "Load Annotation"),
            ("Ctrl+S", "Save Annotation"),
            ("Ctrl+Shift+S", "Save As..."),
            ("Ctrl+Z", "Undo"),
            ("Ctrl+Y", "Redo"),
            ("Delete", "Delete Selected Box(es)"),
            ("Ctrl+C", "Copy Box(es) to Clipboard"),
            ("Ctrl+V", "Paste Box(es) (from Clipboard or Previous Frame)"),
            ("Ctrl+Alt+V", "Paste Box(es) with Custom X/Y Offset..."),
            ("Ctrl+Shift+V / P", "Copy from Previous Frame (frame-to-frame workflow)"),
            ("Ctrl+D", "Duplicate Selected Box(es)"),
            ("Ctrl+K", "Lock / Unlock Selected Box(es)"),
            ("U", "Focus Selected Box OCR Text Field for Editing"),
            ("N", "New Bounding Box Mode"),
            ("Click", "Select a Box"),
            ("Shift/Ctrl + Click", "Add / Remove Box from Multi-Selection"),
            ("Shift + Drag (canvas)", "Rubber-band Select Multiple Boxes"),
            ("Ctrl+A", "Select All Boxes + Open Edit All Text (reading order)"),
            ("Escape", "Deselect All"),
            ("Drag Selected Box", "Move Box -- Moves all Selected Boxes together (Single Undo)"),
            ("Middle Drag / Space + Drag", "Pan the Image (Never Moves Boxes)"),
            ("Mouse Wheel", "Smooth Zoom around Mouse Cursor"),
            ("F", "Fit Image to View"),
            ("R", "Reset Zoom (100%)"),
            ("Page Up / Left arrow", "Previous Image"),
            ("Page Down / Right arrow", "Next Image"),
            ("Ctrl+Enter", "Save && Next Image (Fast Validation Loop)"),
            ("Ctrl+Shift+D", "Mark / Unmark Current Crop as Doubt (Auto-saved to CSV)"),
            ("Ctrl+Alt+Left / Right", "Navigate between Doubt Crops"),
            ("Right-Click on Crop/Label", "Context Menu: Mark Doubt with Note / Unmark / Export CSV"),
            ("M", "Toggle between labels and labels2 Folders"),
            ("F1", "Keyboard Shortcuts Reference"),
        ]
        table_rows = "".join(
            f"<tr><td style='padding:2px 14px 2px 0;'><b>{key}</b></td>"
            f"<td style='padding:2px 0;'>{desc}</td></tr>"
            for key, desc in rows
        )
        QMessageBox.information(
            self, "Keyboard Shortcuts",
            f"<table>{table_rows}</table>"
            "<br><i>Tip: the Reference Zoom panel also responds to your "
            "mouse -- scroll over it to zoom in/out on the selected box, "
            "click it to reset. The Character Editor panel lets you fix "
            "one letter at a time, or strip stray quote characters with "
            "one click.</i>"
        )

    # ------------------------------------------------------------------
    # Box creation / deletion / clipboard / lock
    # ------------------------------------------------------------------
    def _on_new_box_toggled(self, checked: bool) -> None:
        self.view.set_create_mode(checked)

    def _on_box_created(self, rect: QRectF) -> None:
        ann = Annotation(
            id=new_annotation_id(),
            label="text",
            text="",
            bbox=BoundingBox(rect.x(), rect.y(), rect.width(), rect.height()),
        )
        self.undo_stack.push(AddAnnotationCommand(self.model, ann))
        self.act_new_box.setChecked(False)
        self.view.set_create_mode(False)

    def _select_all_boxes(self) -> None:
        for item in self.bbox_items.values():
            item.setSelected(True)
        self._refresh_batch_text_panel()
        self.batch_text_dock.show()
        self.batch_text_dock.raise_()

    def _deselect_all_boxes(self) -> None:
        self.scene.clearSelection()

    @staticmethod
    def _reading_order(annotations: List[Annotation]) -> List[Annotation]:
        """Sort annotations into approximate reading order: top-to-bottom
        rows, then left-to-right within each row.

        Boxes are grouped into the same "row" if their vertical centers
        fall within one row-bucket of each other. The bucket size scales
        to the boxes' own (median) height rather than a fixed pixel
        value, so it works whether these are whole-plate boxes (~50-100px
        tall) or individual character boxes (~15-30px tall) -- a fixed
        pixel threshold would either merge separate lines of small text
        or split a single line of big text depending on font size.
        """
        if not annotations:
            return []
        heights = sorted(a.bbox.height for a in annotations if a.bbox.height > 0)
        median_height = heights[len(heights) // 2] if heights else 20.0
        row_bucket = max(median_height * 0.6, 10.0)
        return sorted(
            annotations,
            key=lambda a: (round(a.bbox.center_y / row_bucket), a.bbox.center_x),
        )

    def _refresh_batch_text_panel(self) -> None:
        ids = self._selected_annotation_ids()
        source = [self.model.get(i) for i in ids] if ids else self.model.all()
        annotations = [a for a in source if a is not None]
        self.batch_text_panel.show_annotations(self._reading_order(annotations))

    def _apply_batch_text_edits(self, changes: Dict[str, str]) -> None:
        if not changes:
            self.status.showMessage("No text corrections to apply", 2500)
            return
        self.undo_stack.beginMacro(f"Correct {len(changes)} text field(s)")
        try:
            for ann_id, new_text in changes.items():
                ann = self.model.get(ann_id)
                if ann is None or ann.locked or ann.text == new_text:
                    continue
                self.undo_stack.push(FieldChangeCommand(self.model, ann_id, "text", ann.text, new_text))
        finally:
            self.undo_stack.endMacro()
        self.status.showMessage(f"Applied {len(changes)} text correction(s)", 3000)
        self._refresh_batch_text_panel()

    def _apply_batch_text_edits_and_save(self, changes: Dict[str, str]) -> None:
        if changes:
            self._apply_batch_text_edits(changes)
        self.save_annotation()

    def _selected_annotation_ids(self) -> List[str]:
        """All annotation ids currently selected in the graphics scene, in
        model order (falls back to the single model-selected id if the
        scene has nothing selected, e.g. selection driven from the list
        panel)."""
        ids = [item.annotation_id for item in self.scene.selectedItems()
               if isinstance(item, BBoxGraphicsItem)]
        if ids:
            return ids
        sel = self.model.selected_id()
        return [sel] if sel else []

    def _delete_selected(self) -> None:
        ids = self._selected_annotation_ids()
        if not ids:
            return
        if len(ids) == 1:
            self._delete_annotation_by_id(ids[0])
            return
        self.undo_stack.beginMacro(f"Delete {len(ids)} boxes")
        try:
            for ann_id in ids:
                ann = self.model.get(ann_id)
                if ann is None:
                    continue
                index = self.model.index_of(ann_id)
                self.undo_stack.push(DeleteAnnotationCommand(self.model, ann.clone(), index))
        finally:
            self.undo_stack.endMacro()
        self.status.showMessage(f"Deleted {len(ids)} box(es)", 3000)

    def _delete_annotation_by_id(self, ann_id: Optional[str]) -> None:
        if not ann_id:
            return
        ann = self.model.get(ann_id)
        if ann is None:
            return
        index = self.model.index_of(ann_id)
        self.undo_stack.push(DeleteAnnotationCommand(self.model, ann.clone(), index))

    def _copy_selected(self) -> None:
        ids = self._selected_annotation_ids()
        if ids:
            boxes = [self.model.get(i) for i in ids if self.model.get(i) is not None]
        else:
            boxes = self.model.all()

        if not boxes:
            self.status.showMessage("No boxes to copy", 2000)
            return

        self.clipboard_annotations = [a.clone() for a in boxes]
        self.clipboard_source_image = self.current_image_path
        if ids:
            self.status.showMessage(f"Copied {len(boxes)} selected box(es) to clipboard", 2500)
        else:
            self.status.showMessage(f"Copied all {len(boxes)} box(es) from current frame to clipboard", 2500)

    def _paste_clipboard(self, offset_x: Optional[float] = None, offset_y: Optional[float] = None) -> None:
        source_boxes = self.clipboard_annotations
        if not source_boxes:
            if self.previous_frame_annotations:
                self._copy_from_previous_frame(offset_x or 0.0, offset_y or 0.0)
                return
            self.status.showMessage("Clipboard is empty", 2000)
            return

        if offset_x is None or offset_y is None:
            if self.clipboard_source_image == self.current_image_path:
                dx = 15.0
                dy = 15.0
            else:
                dx = 0.0
                dy = 0.0
        else:
            dx = offset_x
            dy = offset_y

        new_boxes: List[Annotation] = []
        for ann in source_boxes:
            new_ann = ann.clone()
            new_ann.id = new_annotation_id()
            new_ann.bbox.x += dx
            new_ann.bbox.y += dy
            new_boxes.append(new_ann)

        if self.model.parser is None and self.previous_frame_parser is not None:
            self.model.parser = self.previous_frame_parser

        self.undo_stack.beginMacro(f"Paste {len(new_boxes)} box(es)")
        try:
            for new_ann in new_boxes:
                self.undo_stack.push(AddAnnotationCommand(self.model, new_ann))
        finally:
            self.undo_stack.endMacro()

        self.scene.clearSelection()
        for new_ann in new_boxes:
            item = self.bbox_items.get(new_ann.id)
            if item is not None:
                item.setSelected(True)

        if dx != 0.0 or dy != 0.0:
            self.status.showMessage(f"Pasted {len(new_boxes)} box(es) with offset ({dx:+.1f}, {dy:+.1f})", 3000)
        else:
            self.status.showMessage(f"Pasted {len(new_boxes)} box(es)", 3000)

    def _copy_from_previous_frame(self, offset_x: float = 0.0, offset_y: float = 0.0) -> None:
        source_boxes = self.previous_frame_annotations
        source_name = f"previous frame ({Path(self.previous_frame_path).name if self.previous_frame_path else 'previous'})"
        if not source_boxes:
            if self.clipboard_annotations:
                source_boxes = self.clipboard_annotations
                source_name = "clipboard"
            else:
                self.status.showMessage("No bounding boxes available from previous frame or clipboard", 3000)
                return

        new_boxes: List[Annotation] = []
        for ann in source_boxes:
            new_ann = ann.clone()
            new_ann.id = new_annotation_id()
            new_ann.bbox.x += offset_x
            new_ann.bbox.y += offset_y
            new_boxes.append(new_ann)

        if self.model.parser is None and self.previous_frame_parser is not None:
            self.model.parser = self.previous_frame_parser

        self.undo_stack.beginMacro(f"Copy {len(new_boxes)} box(es) from {source_name}")
        try:
            for new_ann in new_boxes:
                self.undo_stack.push(AddAnnotationCommand(self.model, new_ann))
        finally:
            self.undo_stack.endMacro()

        self.scene.clearSelection()
        for new_ann in new_boxes:
            item = self.bbox_items.get(new_ann.id)
            if item is not None:
                item.setSelected(True)

        if offset_x != 0.0 or offset_y != 0.0:
            self.status.showMessage(
                f"Copied {len(new_boxes)} box(es) from {source_name} with offset ({offset_x:+.1f}, {offset_y:+.1f})",
                3500
            )
        else:
            self.status.showMessage(f"Copied {len(new_boxes)} box(es) from {source_name}", 3500)

    def _paste_with_offset_dialog(self) -> None:
        if self.clipboard_annotations:
            source_boxes = self.clipboard_annotations
            source_name = "Clipboard"
        elif self.previous_frame_annotations:
            source_boxes = self.previous_frame_annotations
            source_name = f"Previous Frame ({Path(self.previous_frame_path).name if self.previous_frame_path else ''})"
        else:
            self.status.showMessage("Nothing copied in clipboard or previous frame", 2500)
            return

        dlg = PasteOffsetDialog(
            self,
            num_boxes=len(source_boxes),
            source_name=source_name,
            default_x=self.paste_offset_x,
            default_y=self.paste_offset_y,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            dx, dy = dlg.offset()
            self.paste_offset_x = dx
            self.paste_offset_y = dy
            self.settings.setValue("paste_offset_x", dx)
            self.settings.setValue("paste_offset_y", dy)
            self._paste_clipboard(offset_x=dx, offset_y=dy)

    def _duplicate_selected(self) -> None:
        ids = self._selected_annotation_ids()
        if not ids:
            ann = self.model.selected()
            if ann:
                ids = [ann.id]
        if not ids:
            return
        boxes = [self.model.get(i) for i in ids if self.model.get(i) is not None]
        if not boxes:
            return

        new_boxes = []
        for ann in boxes:
            new_ann = ann.clone()
            new_ann.id = new_annotation_id()
            new_ann.bbox.x += 15.0
            new_ann.bbox.y += 15.0
            new_boxes.append(new_ann)

        self.undo_stack.beginMacro(f"Duplicate {len(new_boxes)} box(es)")
        try:
            for new_ann in new_boxes:
                self.undo_stack.push(AddAnnotationCommand(self.model, new_ann))
        finally:
            self.undo_stack.endMacro()

        self.scene.clearSelection()
        for new_ann in new_boxes:
            item = self.bbox_items.get(new_ann.id)
            if item is not None:
                item.setSelected(True)

    def _toggle_lock_selected(self) -> None:
        ann = self.model.selected()
        if ann is None:
            return
        self.undo_stack.push(FieldChangeCommand(self.model, ann.id, "locked", ann.locked, not ann.locked))

    def _focus_selected_text_editor(self) -> None:
        ann = self.model.selected()
        if ann is None:
            self.status.showMessage("Select a box first", 2000)
            return
        if ann.locked:
            self.status.showMessage("This box is locked", 2000)
            return
        self.properties_panel.show_annotation(ann)
        self.properties_panel.text_edit.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self.properties_panel.text_edit.selectAll()
        self.properties_panel.text_edit.deselect()
        self.properties_panel.text_edit.setCursorPosition(len(self.properties_panel.text_edit.text()))
        self.properties_panel.text_edit.setFocus()

    # ------------------------------------------------------------------
    # AnnotationController protocol (called by BBoxGraphicsItem)
    # ------------------------------------------------------------------
    def select_annotation(self, ann_id: Optional[str]) -> None:
        self.model.select(ann_id)

    def commit_geometry_change(self, ann_id: str, old_bbox: BoundingBox, new_bbox: BoundingBox) -> None:
        self.undo_stack.push(GeometryChangeCommand(self.model, ann_id, old_bbox, new_bbox))

    def commit_group_geometry_change(self, changes: List[Tuple[str, BoundingBox, BoundingBox]]) -> None:
        if not changes:
            return
        if len(changes) == 1:
            ann_id, old_bbox, new_bbox = changes[0]
            self.commit_geometry_change(ann_id, old_bbox, new_bbox)
            return
        self.undo_stack.beginMacro(f"Move {len(changes)} boxes")
        try:
            for ann_id, old_bbox, new_bbox in changes:
                self.undo_stack.push(GeometryChangeCommand(self.model, ann_id, old_bbox, new_bbox))
        finally:
            self.undo_stack.endMacro()

    def bbox_item_for(self, ann_id: str) -> Optional[BBoxGraphicsItem]:
        return self.bbox_items.get(ann_id)

    def live_geometry_preview(self, ann_id: str, bbox: BoundingBox) -> None:
        if ann_id != self.model.selected_id():
            return
        self.properties_panel.update_geometry_only(bbox)
        self.reference_panel.show_region(bbox)

    # ------------------------------------------------------------------
    # Properties panel <-> model wiring
    # ------------------------------------------------------------------
    def _on_label_changed(self, text: str) -> None:
        ann = self.model.selected()
        if ann and ann.label != text:
            self.undo_stack.push(FieldChangeCommand(self.model, ann.id, "label", ann.label, text))

    def _on_text_changed(self, text: str) -> None:
        ann = self.model.selected()
        if ann and ann.text != text:
            self.undo_stack.push(FieldChangeCommand(self.model, ann.id, "text", ann.text, text))

    def _on_geometry_field_changed(self, x: float, y: float, w: float, h: float) -> None:
        ann = self.model.selected()
        if ann is None:
            return
        new_bbox = BoundingBox(x, y, max(w, 1.0), max(h, 1.0))
        old_bbox = ann.bbox.clone()
        if new_bbox.as_tuple() != old_bbox.as_tuple():
            self.undo_stack.push(GeometryChangeCommand(self.model, ann.id, old_bbox, new_bbox))

    def _on_lock_toggled(self, checked: bool) -> None:
        ann = self.model.selected()
        if ann and ann.locked != checked:
            self.undo_stack.push(FieldChangeCommand(self.model, ann.id, "locked", ann.locked, checked))

    # ------------------------------------------------------------------
    # Model -> View synchronization
    # ------------------------------------------------------------------
    def _on_model_reset(self) -> None:
        for item in list(self.bbox_items.values()):
            self.scene.removeItem(item)
        self.bbox_items.clear()
        for ann in self.model.all():
            self._create_graphics_item(ann)
        self.list_panel.refresh(self.model.all(), self.model.selected_id())
        self.properties_panel.show_annotation(None)
        self.batch_text_panel.show_annotations([])

    def _create_graphics_item(self, ann: Annotation) -> None:
        item = BBoxGraphicsItem(ann, self)
        self.scene.addItem(item)
        self.bbox_items[ann.id] = item

    def _on_model_annotation_added(self, ann_id: str) -> None:
        ann = self.model.get(ann_id)
        if ann is None:
            return
        if ann_id not in self.bbox_items:
            self._create_graphics_item(ann)
        self.list_panel.refresh(self.model.all(), self.model.selected_id())

    def _on_model_annotation_removed(self, ann_id: str) -> None:
        item = self.bbox_items.pop(ann_id, None)
        if item is not None:
            self.scene.removeItem(item)
        self.list_panel.refresh(self.model.all(), self.model.selected_id())

    def _on_model_annotation_updated(self, ann_id: str) -> None:
        ann = self.model.get(ann_id)
        item = self.bbox_items.get(ann_id)
        if ann is None or item is None:
            return
        item.sync_from_model(ann.bbox)
        item.set_locked(ann.locked)
        if ann_id == self.model.selected_id():
            self.properties_panel.show_annotation(ann)
            self.reference_panel.show_region(ann.bbox)
        self.list_panel.refresh(self.model.all(), self.model.selected_id())

    def _on_model_selection_changed(self, ann_id: Optional[str]) -> None:
        # When the model's "primary" selection changed as an ECHO of a
        # scene-side selection event (a box click, Shift/Ctrl-click, or a
        # rubber-band drag), the scene's own selection state is already
        # correct -- forcing every item to `setSelected(aid == ann_id)`
        # here would collapse a multi-selection down to just the one
        # "primary" box. Only force-sync items when the selection was
        # driven from elsewhere (e.g. the Annotation List panel, Select
        # All, or programmatic navigation).
        if not self._scene_originated_selection:
            self._suppress_selection_echo = True
            try:
                for aid, item in self.bbox_items.items():
                    item.setSelected(aid == ann_id)
            finally:
                self._suppress_selection_echo = False
        selected_ann = self.model.get(ann_id) if ann_id else None
        self.properties_panel.show_annotation(selected_ann)
        self.reference_panel.show_region(selected_ann.bbox if selected_ann else None)
        self.list_panel.refresh(self.model.all(), ann_id)

    def _on_scene_selection_changed(self) -> None:
        if self._suppress_selection_echo:
            return
        selected_items = [it for it in self.scene.selectedItems() if isinstance(it, BBoxGraphicsItem)]
        self._scene_originated_selection = True
        try:
            if selected_items:
                # Keep the model's single "primary" selection (drives the
                # Properties / Character Editor / Reference Zoom panels)
                # pointed at the most recently selected box, while the
                # scene itself may hold a larger multi-selection for group
                # move / group delete.
                self.model.select(selected_items[-1].annotation_id)
            else:
                self.model.select(None)
        finally:
            self._scene_originated_selection = False

    def _on_dirty_changed(self, dirty: bool) -> None:
        self._update_title()
        if dirty and self.act_auto_save.isChecked():
            self.auto_save_timer.start()
        else:
            self.auto_save_timer.stop()

    def _update_title(self) -> None:
        img_name = Path(self.current_image_path).name if self.current_image_path else "(no image)"
        ann_name = Path(self.current_annotation_path).name if self.current_annotation_path else "(no annotation)"
        dirty_mark = " *" if self.model.is_dirty() else ""
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION} - {img_name} | {ann_name}{dirty_mark}")

    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        # Fallback routing for Left/Right image navigation: the QActions
        # already carry these shortcuts, but if a key event bubbles all
        # the way here unhandled (e.g. focus quirks on some platforms),
        # navigate anyway. Text fields and lists never let arrows reach
        # this point, so typing/cursor movement is unaffected.
        if event.key() == Qt.Key.Key_Left:
            self.prev_image()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Right:
            self.next_image()
            event.accept()
            return
        super().keyPressEvent(event)

    def _restore_window_geometry(self) -> None:
        saved_geometry = self.settings.value("window_geometry")
        saved_state = self.settings.value("window_state")
        if saved_geometry:
            self.restoreGeometry(saved_geometry)
        if saved_state:
            self.restoreState(saved_state)
        self._ensure_window_visible()

    def _ensure_window_visible(self) -> None:
        if self.isMaximized():
            return
        screen = QApplication.screenAt(self.geometry().center()) or QApplication.primaryScreen()
        if screen is None:
            self.resize(1440, 900)
            return
        avail = screen.availableGeometry()
        rect = self.frameGeometry()
        if rect.width() <= 0 or rect.height() <= 0:
            self.resize(1440, 900)
            self.move(max(0, (avail.width() - 1440) // 2), max(0, (avail.height() - 900) // 2))
            return
        if rect.width() > avail.width() or rect.height() > avail.height():
            self.resize(max(900, int(avail.width() * 0.9)), max(700, int(avail.height() * 0.9)))
        if not avail.contains(rect):
            self.setGeometry(
                max(avail.left(), min(rect.left(), avail.right() - max(900, rect.width()))),
                max(avail.top(), min(rect.top(), avail.bottom() - max(700, rect.height()))),
                max(900, min(rect.width(), avail.width())),
                max(700, min(rect.height(), avail.height())),
            )

    def _save_window_state(self) -> None:
        self.settings.setValue("window_geometry", self.saveGeometry())
        self.settings.setValue("window_state", self.saveState())
        self.settings.setValue("auto_save", self.act_auto_save.isChecked())

    def closeEvent(self, event) -> None:
        self._save_window_state()
        if self._confirm_discard_changes():
            event.accept()
        else:
            event.ignore()


# ======================================================================
# 13. SELF-TEST  (offline sanity check — no display required)
# ======================================================================

def run_self_test() -> bool:
    """
    Headless smoke test a user (or CI) can run to verify the install and
    the core guarantees of the app without needing a display:
      - image load, annotation auto-match, parser round-trip (JSON + TXT)
      - geometry edit -> undo -> redo
      - per-region independence: editing one box never touches another
      - multi-select group move -> undo -> redo as ONE step
      - multi-select group delete -> undo restores both boxes
      - save writes the edit back to disk in the original format
    Run with:  python ocr_annotation_editor.py --selftest
    Exits 0 and prints "SELF-TEST PASSED" on success, 1 and a traceback
    on failure.
    """
    import tempfile

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication(sys.argv)

    with tempfile.TemporaryDirectory() as tmp:
        img_path = os.path.join(tmp, "selftest.jpg")
        json_path = os.path.join(tmp, "selftest.json")

        canvas = np.full((200, 500, 3), 255, dtype=np.uint8)
        cv2.putText(canvas, "AB", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 5)
        cv2.putText(canvas, "CD", (260, 130), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 5)
        cv2.imwrite(img_path, canvas)

        anns = [
            Annotation(id="t1", label="seg1", text="AB", bbox=BoundingBox(30, 60, 140, 90)),
            Annotation(id="t2", label="seg2", text="CD", bbox=BoundingBox(250, 60, 140, 90)),
        ]
        ParserRegistry.for_extension(".json").save(json_path, anns)

        win = MainWindow()
        win._open_image_path(img_path)
        win.model.select("t1")
        win._focus_selected_text_editor()
        assert win.properties_panel.text_edit.hasFocus(), "shortcut helper did not focus the selected text editor"

        assert win.current_annotation_path == json_path, "annotation did not auto-match"
        assert len(win.model.all()) == 2 and len(win.bbox_items) == 2, "load / box-count mismatch"

        before_seg2 = win.model.get("t2").bbox.clone()

        old_bbox = win.model.get("t1").bbox.clone()
        new_bbox = BoundingBox(old_bbox.x + 10, old_bbox.y + 5, old_bbox.width, old_bbox.height)
        win.commit_geometry_change("t1", old_bbox, new_bbox)
        assert win.model.get("t1").bbox.x == new_bbox.x, "geometry edit did not apply"

        after_seg2 = win.model.get("t2").bbox
        assert (after_seg2.x, after_seg2.y) == (before_seg2.x, before_seg2.y), \
            "editing one box affected an unrelated box (independence guarantee broken)"

        win.undo_stack.undo()
        assert win.model.get("t1").bbox.x == old_bbox.x, "undo failed"
        win.undo_stack.redo()
        assert win.model.get("t1").bbox.x == new_bbox.x, "redo failed"

        win.save_annotation()
        reloaded = json.load(open(json_path, encoding="utf-8"))
        saved_t1 = next(a for a in reloaded["annotations"] if a["id"] == "t1")
        assert saved_t1["bbox"]["x"] == new_bbox.x, "save did not persist the edit"

        # -- multi-select group move: one undo step moves both boxes ------
        win._select_all_boxes()
        assert len(win.scene.selectedItems()) == 2, "select-all did not select both boxes"
        t1_before = win.model.get("t1").bbox.clone()
        t2_before = win.model.get("t2").bbox.clone()
        win.commit_group_geometry_change([
            ("t1", t1_before, BoundingBox(t1_before.x + 20, t1_before.y, t1_before.width, t1_before.height)),
            ("t2", t2_before, BoundingBox(t2_before.x + 20, t2_before.y, t2_before.width, t2_before.height)),
        ])
        assert win.model.get("t1").bbox.x == t1_before.x + 20, "group move did not apply to t1"
        assert win.model.get("t2").bbox.x == t2_before.x + 20, "group move did not apply to t2"
        win.undo_stack.undo()
        assert win.model.get("t1").bbox.x == t1_before.x, "group move undo (single step) failed for t1"
        assert win.model.get("t2").bbox.x == t2_before.x, "group move undo (single step) failed for t2"
        win.undo_stack.redo()
        assert win.model.get("t1").bbox.x == t1_before.x + 20, "group move redo failed for t1"

        # -- multi-select group delete: single undo restores both --------
        win._select_all_boxes()
        win._delete_selected()
        assert len(win.model.all()) == 0, "group delete did not remove all selected boxes"
        win.undo_stack.undo()
        assert len(win.model.all()) == 2, "group delete undo (single step) did not restore both boxes"

        # -- Text edits still flow through the main property editor and autosave flow --
        win.model.select("t1")
        win.undo_stack.push(FieldChangeCommand(win.model, "t1", "text", win.model.get("t1").text, '"AB"'))
        win.properties_panel.show_annotation(win.model.get("t1"))
        assert win.model.get("t1").text == '"AB"', "text edit did not reach the model"

        # -- Ctrl+A -> batch "Edit All Text" panel, sequential reading order
        win._select_all_boxes()
        assert len(win.batch_text_panel._entries) == 2, "Ctrl+A did not populate both rows"
        # t1 is left of t2 (x=30 vs x=250) -> reading order must list t1 first
        assert win.batch_text_panel._entries[0][0] == "t1", "reading order (left-to-right) is wrong"
        assert win.batch_text_panel._entries[1][0] == "t2", "reading order (left-to-right) is wrong"

        # Edit both rows in the panel, apply as ONE undo step. (Both
        # boxes are currently text "AB" -- the model was never actually
        # changed by the char-editor demo above, since it round-tripped
        # back to the same string -- so capture the true pre-batch text
        # from the model, not an assumption, before applying.)
        text_before = {aid: win.model.get(aid).text for aid in ("t1", "t2")}
        win.batch_text_panel._entries[0][1].setText("XY")
        win.batch_text_panel._entries[1][1].setText("ZQ")
        changes = win.batch_text_panel._collect_changes()
        assert changes == {"t1": "XY", "t2": "ZQ"}, "batch panel did not detect both row edits"
        win._apply_batch_text_edits(changes)
        assert win.model.get("t1").text == "XY", "batch apply did not update t1's text"
        assert win.model.get("t2").text == "ZQ", "batch apply did not update t2's text"
        # The defining property of a batched edit: ONE undo() call must
        # revert BOTH fields together (not just the last one changed),
        # and ONE redo() must reapply both.
        win.undo_stack.undo()
        assert win.model.get("t1").text == text_before["t1"] and win.model.get("t2").text == text_before["t2"], \
            "single-step undo of the batch edit failed to restore both rows"
        win.undo_stack.redo()
        assert win.model.get("t1").text == "XY" and win.model.get("t2").text == "ZQ", \
            "single-step redo of the batch edit failed"

        # Apply && Save should also persist to disk
        win.batch_text_panel._entries[0][1].setText("XY")  # unchanged this time
        win._apply_batch_text_edits_and_save({})
        reloaded2 = json.load(open(json_path, encoding="utf-8"))
        saved_t1_text = next(a for a in reloaded2["annotations"] if a["id"] == "t1")["text"]
        assert saved_t1_text == "XY", "Apply && Save did not persist the batch-edited text"

        # The Character Editor dock is no longer shown in the main UI.
        assert not any(dock.windowTitle() == "Character Editor" for dock in win.findChildren(QDockWidget)), \
            "Character Editor dock should be removed from the UI"

        # Auto-save should persist a dirty edit without requiring a manual save.
        win.act_auto_save.setChecked(True)
        win._on_text_changed("AUTO")
        win._auto_save_if_needed()
        reloaded3 = json.load(open(json_path, encoding="utf-8"))
        saved_t1_autosave = next(a for a in reloaded3["annotations"] if a["id"] == "t1")["text"]
        assert saved_t1_autosave == "AUTO", "Auto-save did not persist the text change"

        # -- Frame-to-frame copy-paste and selected-box copy tests ----------
        # Test 1: Copy selected box to clipboard
        win.scene.clearSelection()
        win.bbox_items["t1"].setSelected(True)
        win._copy_selected()
        assert len(win.clipboard_annotations) == 1 and win.clipboard_annotations[0].text == "AUTO", \
            "Selected box copy failed"

        # Test 2: Paste clipboard box into same frame with offset
        win._paste_clipboard(offset_x=10.0, offset_y=10.0)
        assert len(win.model.all()) == 3, "Pasting single box failed"
        pasted_box = win.model.all()[-1]
        assert pasted_box.id != "t1", "Pasted box must have a new unique ID"
        assert pasted_box.bbox.x == win.model.get("t1").bbox.x + 10.0, "Pasted box offset X mismatch"
        win.undo_stack.undo()
        assert len(win.model.all()) == 2, "Undo paste single box failed"

        # Test 3: Frame-to-frame copy across images
        # Create a second test image in tmp
        img_path2 = os.path.join(td, "test_frame2.png")
        cv2.imwrite(img_path2, img)
        # Open frame 2 (empty of annotations)
        win._open_image_path(img_path2)
        assert len(win.model.all()) == 0, "Frame 2 should start empty"
        assert len(win.previous_frame_annotations) == 2, "Previous frame annotations not captured"

        # Copy from previous frame with offset
        win._copy_from_previous_frame(offset_x=5.0, offset_y=-3.0)
        assert len(win.model.all()) == 2, "Copy from previous frame failed to create 2 boxes"
        f2_b1 = win.model.all()[0]
        assert f2_b1.id not in ("t1", "t2"), "Copied frame-to-frame boxes must have new unique IDs"
        assert f2_b1.text == "AUTO", "Preserved text mismatch on frame-to-frame copy"
        assert f2_b1.bbox.x == 30 + 20 + 5.0, "Frame-to-frame offset X mismatch"

        # Single undo restores empty frame
        win.undo_stack.undo()
        assert len(win.model.all()) == 0, "Single-step undo of frame-to-frame copy failed"
        win.undo_stack.redo()
        assert len(win.model.all()) == 2, "Redo of frame-to-frame copy failed"

        # Test 4: Smooth zoom & Space pan state
        win.view._space_pan = True
        assert win.view._space_pan is True, "Space pan mode flag failed"
        zoom_before = win.view._zoom
        win.view.set_zoom(zoom_before * 1.25)
        assert abs(win.view._zoom - zoom_before * 1.25) < 0.001, "Smooth zoom scale failed"
        win.view._space_pan = False

        # -- Doubt marking & CSV persistence tests ------------------------
        # Test 1: Mark image as doubt
        win._on_doubt_toggled(img_path2, True, "Blurry crop")
        assert img_path2 in win.doubts, "Doubt was not added to win.doubts"
        assert win.doubts[img_path2]["reason"] == "Blurry crop", "Doubt reason not recorded"
        assert os.path.isfile(win.doubts_csv_path), "doubts.csv was not created on disk"

        # Verify CSV contents on disk
        with open(win.doubts_csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 1, "doubts.csv row count mismatch"
            assert reader[0]["reason"] == "Blurry crop", "CSV reason mismatch"
            assert reader[0]["filename"] == "test_frame2.png", "CSV filename mismatch"

        # Test 2: Toggle doubt filter in nav_panel
        assert win.nav_panel._is_path_doubt(img_path2) is True, "nav_panel did not recognize doubt path"
        win.nav_panel._set_doubt_filter(True)
        assert win.nav_panel.list_widget.count() == 1, "Doubt filter did not show 1 item"
        win.nav_panel._set_doubt_filter(False)
        assert win.nav_panel.list_widget.count() >= 1, "All filter did not restore items"

        # Test 3: Unmark doubt and verify CSV is updated
        win._on_doubt_toggled(img_path2, False, "")
        assert img_path2 not in win.doubts, "Doubt was not removed from win.doubts"
        with open(win.doubts_csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 0, "doubts.csv was not emptied on unmark"

    return True


# ======================================================================
# 14. ENTRY POINT
# ======================================================================

def main() -> int:
    if "--selftest" in sys.argv:
        try:
            run_self_test()
        except Exception:
            print("SELF-TEST FAILED\n")
            traceback.print_exc()
            return 1
        print("SELF-TEST PASSED")
        return 0

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_ORG)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    if len(sys.argv) > 1:
        candidate = sys.argv[1]
        if os.path.isfile(candidate) and Path(candidate).suffix.lower() in SUPPORTED_IMAGE_EXTS:
            window._open_image_path(candidate)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

