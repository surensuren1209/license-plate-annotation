import os
import sys
import json
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Optional, Tuple, Set, Dict
import cv2
import numpy as np

from PyQt6.QtCore import Qt, QRectF, QPoint, QPointF, pyqtSignal, QObject, QSettings, QSize
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QFileDialog, QMessageBox, QLabel,
    QToolBar, QStatusBar, QInputDialog, QGraphicsView, QGraphicsScene, 
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsItem, QStyleOptionGraphicsItem,
    QGroupBox, QLineEdit, QSpinBox, QDoubleSpinBox, QPushButton, QFormLayout,
    QGraphicsPathItem, QGraphicsPolygonItem, QGraphicsTextItem, QGridLayout, QButtonGroup, QDialog,
    QComboBox, QMenu, QFrame, QCheckBox, QRadioButton
)
from PyQt6.QtGui import QPainter, QPen, QColor, QCursor, QPixmap, QFont, QBrush, QAction, QKeySequence, QDragEnterEvent, QDropEvent, QPainterPath, QImage, QPolygonF


# ==============================================================================
# 1. UTILITIES & GEOMETRY
# ==============================================================================

def natural_sort_key(s: str) -> list:
    """
    Returns a natural human alphanumeric sorting key based on filename (e.g. img_2.jpg < img_10.jpg),
    ignoring directory path differences like 'doubt/' so crop files maintain a consistent
    1-to-1 order matching the file manager!
    """
    if not s:
        return []
    filename = os.path.basename(s)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

def absolute_to_normalized(
    x: float, y: float, w: float, h: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Converts absolute pixel coords (x, y, w, h) to normalized YOLO format (cx, cy, w, h)."""
    if img_w <= 0 or img_h <= 0:
        return 0.0, 0.0, 0.0, 0.0
    cx = x + w / 2.0
    cy = y + h / 2.0
    return cx / img_w, cy / img_h, w / img_w, h / img_h


def normalized_to_absolute(
    norm_cx: float, norm_cy: float, norm_w: float, norm_h: float, img_w: int, img_h: int
) -> Tuple[int, int, int, int]:
    """Converts normalized YOLO format (cx, cy, w, h) to absolute pixel coords (x, y, w, h)."""
    w = norm_w * img_w
    h = norm_h * img_h
    cx = norm_cx * img_w
    cy = norm_cy * img_h
    x = cx - w / 2.0
    y = cy - h / 2.0
    return int(round(x)), int(round(y)), max(1, int(round(w))), max(1, int(round(h)))


def order_quad_points(pts: np.ndarray) -> List[Tuple[float, float]]:
    """Orders 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return [(float(p[0]), float(p[1])) for p in rect]





def parse_sftp_url(url: str) -> Tuple[Optional[str], Optional[str]]:
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


def get_ssh_cmd(user_host: str, remote_cmd: str) -> List[str]:
    control_path = "/tmp/sc-%r@%h:%p"
    print(f"[LOG] Preparing SSH Command: ssh {user_host} \"{remote_cmd}\" (using ControlPersist)")
    return [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ControlMaster=auto",
        "-o", "ControlPath=" + control_path,
        "-o", "ControlPersist=15m",
        user_host,
        remote_cmd
    ]


def read_remote_file(user_host: str, remote_path: str) -> Optional[str]:
    import subprocess
    print(f"[LOG] Fetching remote file: {remote_path} from host: {user_host}")
    cmd = get_ssh_cmd(user_host, f"cat '{remote_path}'")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if res.returncode == 0:
            print(f"[LOG] Fetch remote file success ({len(res.stdout)} chars): {remote_path}")
            return res.stdout
        else:
            print(f"[LOG] Fetch remote file failed (code {res.returncode}): {res.stderr.strip()}")
    except Exception as e:
        print(f"[LOG] Error fetching remote file: {e}")
    return None


def write_remote_file(user_host: str, remote_path: str, local_path: str) -> bool:
    import subprocess
    print(f"[LOG] Uploading local file {local_path} to remote: {remote_path} on host: {user_host}")
    try:
        with open(local_path, "r", encoding="utf-8") as lf:
            content = lf.read()
        cmd = get_ssh_cmd(user_host, f"cat > '{remote_path}'")
        res = subprocess.run(cmd, input=content, capture_output=True, text=True, timeout=15)
        success = res.returncode == 0
        if success:
            print(f"[LOG] Upload remote file success: {remote_path}")
        else:
            print(f"[LOG] Upload remote file failed (code {res.returncode}): {res.stderr.strip()}")
        return success
    except Exception as e:
        print(f"[LOG] Error uploading remote file: {e}")
        return False


class ClickableLabel(QLabel):
    clicked = pyqtSignal()
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


# ==============================================================================
# 2. DATA MODELS
# ==============================================================================

class BoundingBox:
    """Represents a 2D bounding box with integer pixel coordinates and optional polygon points."""
    def __init__(self, x: float, y: float, width: float, height: float, points: List[Tuple[float, float]] = None):
        if points is not None and len(points) >= 3:
            self.points = [(float(p[0]), float(p[1])) for p in points]
        else:
            self.points = [
                (float(x), float(y)),
                (float(x + width), float(y)),
                (float(x + width), float(y + height)),
                (float(x), float(y + height))
            ]
        self._update_bounds()

    def _update_bounds(self):
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        self._x = int(round(min(xs)))
        self._y = int(round(min(ys)))
        self._width = max(1, int(round(max(xs) - self._x)))
        self._height = max(1, int(round(max(ys) - self._y)))

    @property
    def x(self) -> int: return self._x
    @x.setter
    def x(self, val: float):
        dx = int(round(val)) - self._x
        if dx != 0:
            self.points = [(p[0] + dx, p[1]) for p in self.points]
            self._update_bounds()

    @property
    def y(self) -> int: return self._y
    @y.setter
    def y(self, val: float):
        dy = int(round(val)) - self._y
        if dy != 0:
            self.points = [(p[0], p[1] + dy) for p in self.points]
            self._update_bounds()

    @property
    def width(self) -> int: return self._width
    @width.setter
    def width(self, val: float):
        val = max(1, int(round(val)))
        factor = val / self._width if self._width > 0 else 1.0
        self.points = [(self._x + (p[0] - self._x) * factor, p[1]) for p in self.points]
        self._update_bounds()

    @property
    def height(self) -> int: return self._height
    @height.setter
    def height(self, val: float):
        val = max(1, int(round(val)))
        factor = val / self._height if self._height > 0 else 1.0
        self.points = [(p[0], self._y + (p[1] - self._y) * factor) for p in self.points]
        self._update_bounds()

    @property
    def center_x(self) -> float:
        return self._x + self._width / 2.0
    @center_x.setter
    def center_x(self, val: float):
        self.x = val - self._width / 2.0

    @property
    def center_y(self) -> float:
        return self._y + self._height / 2.0
    @center_y.setter
    def center_y(self, val: float):
        self.y = val - self._height / 2.0

    def clone(self) -> 'BoundingBox':
        return BoundingBox(self._x, self._y, self._width, self._height, [p for p in self.points])

    def to_dict(self) -> dict:
        return {
            "x": self._x, "y": self._y, "width": self._width, "height": self._height,
            "center_x": self.center_x, "center_y": self.center_y,
            "points": self.points
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'BoundingBox':
        return cls(d["x"], d["y"], d["width"], d["height"], d.get("points"))


class AnnotationItem:
    """Represents a single annotated object on an image."""
    def __init__(self, label: str, box: BoundingBox, locked: bool = False):
        self.label = label
        self.box = box
        self.locked = locked

    @property
    def x(self) -> int: return self.box.x
    @x.setter
    def x(self, val: float):
        if not self.locked: self.box.x = val

    @property
    def y(self) -> int: return self.box.y
    @y.setter
    def y(self, val: float):
        if not self.locked: self.box.y = val

    @property
    def width(self) -> int: return self.box.width
    @width.setter
    def width(self, val: float):
        if not self.locked: self.box.width = val

    @property
    def height(self) -> int: return self.box.height
    @height.setter
    def height(self, val: float):
        if not self.locked: self.box.height = val

    @property
    def center_x(self) -> float: return self.box.center_x
    @center_x.setter
    def center_x(self, val: float):
        if not self.locked: self.box.center_x = val

    @property
    def center_y(self) -> float: return self.box.center_y
    @center_y.setter
    def center_y(self, val: float):
        if not self.locked: self.box.center_y = val

    def clone(self) -> 'AnnotationItem':
        return AnnotationItem(self.label, self.box.clone(), self.locked)

    def to_dict(self) -> dict:
        return {"label": self.label, "box": self.box.to_dict(), "locked": self.locked}

    @classmethod
    def from_dict(cls, d: dict) -> 'AnnotationItem':
        return cls(d["label"], BoundingBox.from_dict(d["box"]), d.get("locked", False))


# ==============================================================================
# 3. COMMAND PATTERN (UNDO / REDO)
# ==============================================================================

class Command:
    def execute(self) -> None: raise NotImplementedError
    def undo(self) -> None: raise NotImplementedError


class AddItemCommand(Command):
    def __init__(self, items_list: List[AnnotationItem], item: AnnotationItem, on_change):
        self.items_list = items_list
        self.item = item
        self.on_change = on_change

    def execute(self) -> None:
        self.items_list.append(self.item)
        self.on_change()

    def undo(self) -> None:
        if self.item in self.items_list:
            self.items_list.remove(self.item)
        self.on_change()


class DeleteItemCommand(Command):
    def __init__(self, items_list: List[AnnotationItem], items_to_delete: List[AnnotationItem], on_change):
        self.items_list = items_list
        self.items_to_delete = items_to_delete
        self.deleted_indices = []
        self.on_change = on_change

    def execute(self) -> None:
        self.deleted_indices = []
        for item in self.items_to_delete:
            if item in self.items_list:
                idx = self.items_list.index(item)
                self.deleted_indices.append((idx, item))
                self.items_list.remove(item)
        self.on_change()

    def undo(self) -> None:
        for idx, item in reversed(self.deleted_indices):
            self.items_list.insert(idx, item)
        self.on_change()


class ModifyItemCommand(Command):
    def __init__(self, item: AnnotationItem, old_label: str, new_label: str,
                 old_box: BoundingBox, new_box: BoundingBox,
                 old_locked: bool, new_locked: bool, on_change):
        self.item = item
        self.old_label = old_label
        self.new_label = new_label
        self.old_box = old_box.clone()
        self.new_box = new_box.clone()
        self.old_locked = old_locked
        self.new_locked = new_locked
        self.on_change = on_change

    def execute(self) -> None:
        self.item.label = self.new_label
        self.item.box = self.new_box.clone()
        self.item.locked = self.new_locked
        self.on_change()

    def undo(self) -> None:
        self.item.label = self.old_label
        self.item.box = self.old_box.clone()
        self.item.locked = self.old_locked
        self.on_change()


class CommandHistory:
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self._undo_stack: List[Command] = []
        self._redo_stack: List[Command] = []
        self.on_state_change = None

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._notify()

    def push(self, command: Command) -> None:
        command.execute()
        self._undo_stack.append(command)
        self._redo_stack.clear()
        if len(self._undo_stack) > self.max_history:
            self._undo_stack.pop(0)
        self._notify()

    def undo(self) -> None:
        if not self.can_undo(): return
        cmd = self._undo_stack.pop()
        cmd.undo()
        self._redo_stack.append(cmd)
        self._notify()

    def redo(self) -> None:
        if not self.can_redo(): return
        cmd = self._redo_stack.pop()
        cmd.execute()
        self._undo_stack.append(cmd)
        self._notify()

    def can_undo(self) -> bool: return len(self._undo_stack) > 0
    def can_redo(self) -> bool: return len(self._redo_stack) > 0

    def _notify(self) -> None:
        if self.on_state_change:
            self.on_state_change(self.can_undo(), self.can_redo())


# ==============================================================================
# 4. PARSERS LAYER
# ==============================================================================

class BaseParser:
    def load(self, file_path: str, img_width: int, img_height: int) -> List[AnnotationItem]: raise NotImplementedError
    def save(self, file_path: str, annotations: List[AnnotationItem], img_width: int, img_height: int) -> None: raise NotImplementedError


class YoloParser(BaseParser):
    def __init__(self):
        self.image_link = None

    def load(self, file_path: str, img_width: int, img_height: int) -> List[AnnotationItem]:
        self.image_link = None
        annotations = []
        
        if file_path.startswith("sftp://"):
            user_host, path = parse_sftp_url(file_path)
            content = read_remote_file(user_host, path)
            lines = content.splitlines() if content else []
        else:
            if not os.path.exists(file_path): return annotations
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        
        if not lines: return annotations
        
        first_line = lines[0].strip()
        start_idx = 0
        parts = first_line.split()
        if len(parts) == 1 or first_line.startswith(("http://", "https://")):
            self.image_link = first_line
            start_idx = 1

        for line in lines[start_idx:]:
            parts = line.strip().split()
            if len(parts) < 5: continue
            label = parts[0]
            if len(parts) >= 7 and len(parts) % 2 == 1:  # Rotated polygon (label + 2*N coordinates)
                try:
                    coords = list(map(float, parts[1:]))
                except ValueError:
                    continue
                points = []
                for i in range(0, len(coords), 2):
                    px = coords[i] * img_width
                    py = coords[i+1] * img_height
                    points.append((px, py))
                
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                box = BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y, points)
                annotations.append(AnnotationItem(label, box))
            else:  # Standard YOLO cx, cy, w, h format
                try:
                    cx, cy, w, h = map(float, parts[1:5])
                except ValueError:
                    continue
                x, y, abs_w, abs_h = normalized_to_absolute(cx, cy, w, h, img_width, img_height)
                annotations.append(AnnotationItem(label, BoundingBox(x, y, abs_w, abs_h)))
        return annotations

    def save(self, file_path: str, annotations: List[AnnotationItem], img_width: int, img_height: int, image_link: Optional[str] = None) -> None:
        lines = []
        if image_link:
            lines.append(f"{image_link}\n")
        elif self.image_link:
            lines.append(f"{self.image_link}\n")
            
        for item in annotations:
            label = item.label
            pts = item.box.points
            if len(pts) >= 3:
                norm_coords = []
                for pt in pts:
                    nx = max(0.0, min(1.0, pt[0] / img_width))
                    ny = max(0.0, min(1.0, pt[1] / img_height))
                    norm_coords.append(f"{nx:.6f} {ny:.6f}")
                lines.append(f"{label} {' '.join(norm_coords)}\n")
            else:
                cx, cy, w, h = absolute_to_normalized(item.x, item.y, item.width, item.height, img_width, img_height)
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))
                lines.append(f"{label} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)


class XmlParser(BaseParser):
    def load(self, file_path: str, img_width: int, img_height: int) -> List[AnnotationItem]:
        annotations = []
        try:
            if file_path.startswith("sftp://"):
                user_host, path = parse_sftp_url(file_path)
                content = read_remote_file(user_host, path)
                if not content: return annotations
                root = ET.fromstring(content)
            else:
                if not os.path.exists(file_path): return annotations
                tree = ET.parse(file_path)
                root = tree.getroot()
            for obj in root.findall("object"):
                name = obj.find("name")
                bndbox = obj.find("bndbox")
                if name is None or bndbox is None: continue
                label = name.text or "0"
                try:
                    xmin = int(round(float(bndbox.find("xmin").text)))
                    ymin = int(round(float(bndbox.find("ymin").text)))
                    xmax = int(round(float(bndbox.find("xmax").text)))
                    ymax = int(round(float(bndbox.find("ymax").text)))
                except (ValueError, AttributeError):
                    continue
                annotations.append(AnnotationItem(label, BoundingBox(xmin, ymin, max(1, xmax - xmin), max(1, ymax - ymin))))
        except Exception as e:
            print(f"XML parse error: {e}")
        return annotations

    def save(self, file_path: str, annotations: List[AnnotationItem], img_width: int, img_height: int) -> None:
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = os.path.basename(os.path.dirname(file_path))
        ET.SubElement(root, "filename").text = os.path.splitext(os.path.basename(file_path))[0] + ".jpg"
        ET.SubElement(root, "path").text = os.path.splitext(file_path)[0] + ".jpg"
        
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(img_width)
        ET.SubElement(size, "height").text = str(img_height)
        ET.SubElement(size, "depth").text = "3"
        ET.SubElement(root, "segmented").text = "0"

        for item in annotations:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = item.label
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            
            box = ET.SubElement(obj, "bndbox")
            ET.SubElement(box, "xmin").text = str(item.x)
            ET.SubElement(box, "ymin").text = str(item.y)
            ET.SubElement(box, "xmax").text = str(item.x + item.width)
            ET.SubElement(box, "ymax").text = str(item.y + item.height)

        pretty_xml = minidom.parseString(ET.tostring(root, "utf-8")).toprettyxml(indent="  ")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)


class JsonParser(BaseParser):
    def load(self, file_path: str, img_width: int, img_height: int) -> List[AnnotationItem]:
        annotations = []
        try:
            if file_path.startswith("sftp://"):
                user_host, path = parse_sftp_url(file_path)
                content = read_remote_file(user_host, path)
                if not content: return annotations
                data = json.loads(content)
            else:
                if not os.path.exists(file_path): return annotations
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            for item in data.get("annotations", []):
                label = item.get("label", "0")
                locked = item.get("locked", False)
                box_d = item.get("box", {})
                x = box_d.get("x", 0)
                y = box_d.get("y", 0)
                w = box_d.get("w", box_d.get("width", 50))
                h = box_d.get("h", box_d.get("height", 20))
                annotations.append(AnnotationItem(label, BoundingBox(x, y, w, h), locked))
        except Exception as e:
            print(f"JSON parse error: {e}")
        return annotations

    def save(self, file_path: str, annotations: List[AnnotationItem], img_width: int, img_height: int) -> None:
        data = {
            "image_path": os.path.splitext(file_path)[0] + ".jpg",
            "image_width": img_width,
            "image_height": img_height,
            "annotations": [
                {
                    "label": item.label,
                    "locked": item.locked,
                    "box": {"x": item.x, "y": item.y, "w": item.width, "h": item.height, "cx": item.center_x, "cy": item.center_y}
                } for item in annotations
            ]
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)


class CsvParser(BaseParser):
    def load(self, file_path: str, img_width: int, img_height: int) -> List[AnnotationItem]:
        annotations = []
        try:
            import csv
            if file_path.startswith("sftp://"):
                user_host, path = parse_sftp_url(file_path)
                content = read_remote_file(user_host, path)
                if not content: return annotations
                reader = csv.reader(content.splitlines())
            else:
                if not os.path.exists(file_path): return annotations
                f = open(file_path, "r", encoding="utf-8")
                reader = csv.reader(f)
            
            for row in reader:
                    if not row or len(row) < 5: continue
                    label = row[0]
                    if len(row) >= 9:
                        try:
                            coords = list(map(float, row[1:9]))
                        except ValueError:
                            continue
                        points = []
                        for i in range(0, len(coords), 2):
                            px = coords[i] * img_width
                            py = coords[i+1] * img_height
                            points.append((px, py))
                        xs = [p[0] for p in points]
                        ys = [p[1] for p in points]
                        min_x, max_x = min(xs), max(xs)
                        min_y, max_y = min(ys), max(ys)
                        box = BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y, points)
                        annotations.append(AnnotationItem(label, box))
                    elif len(row) == 5:
                        try:
                            cx, cy, w, h = map(float, row[1:5])
                        except ValueError:
                            continue
                        x, y, abs_w, abs_h = normalized_to_absolute(cx, cy, w, h, img_width, img_height)
                        annotations.append(AnnotationItem(label, BoundingBox(x, y, abs_w, abs_h)))
            if not file_path.startswith("sftp://"):
                f.close()
        except Exception as e:
            print(f"CSV load error: {e}")
        return annotations

    def save(self, file_path: str, annotations: List[AnnotationItem], img_width: int, img_height: int) -> None:
        import csv
        try:
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for item in annotations:
                    label = item.label
                    pts = item.box.points
                    if len(pts) >= 3:
                        row = [label]
                        for pt in pts:
                            nx = max(0.0, min(1.0, pt[0] / img_width))
                            ny = max(0.0, min(1.0, pt[1] / img_height))
                            row.append(f"{nx:.6f}")
                            row.append(f"{ny:.6f}")
                        writer.writerow(row)
                    else:
                        cx, cy, w, h = absolute_to_normalized(item.x, item.y, item.width, item.height, img_width, img_height)
                        cx = max(0.0, min(1.0, cx))
                        cy = max(0.0, min(1.0, cy))
                        w = max(0.0, min(1.0, w))
                        h = max(0.0, min(1.0, h))
                        writer.writerow([label, f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"])
        except Exception as e:
            print(f"CSV save error: {e}")


class ParserFactory:
    _parsers = {".txt": YoloParser, ".xml": XmlParser, ".json": JsonParser, ".csv": CsvParser}

    @classmethod
    def get_parser_for_file(cls, file_path: str) -> Optional[BaseParser]:
        ext = os.path.splitext(file_path)[1].lower()
        parser_cls = cls._parsers.get(ext)
        return parser_cls() if parser_cls else None

    @classmethod
    def get_parser_by_name(cls, format_name: str) -> Optional[BaseParser]:
        name = format_name.lower()
        if name in ("yolo", "txt"): return YoloParser()
        if name in ("xml", "voc"): return XmlParser()
        if name in ("json", "custom"): return JsonParser()
        if name in ("csv",): return CsvParser()
        return None

    @classmethod
    def supported_extensions(cls) -> List[str]:
        return list(cls._parsers.keys())


# ==============================================================================
# 5. CONTROLLER LAYER
# ==============================================================================

class EditorController(QObject):
    stateChanged = pyqtSignal()
    imageLoaded = pyqtSignal(str)
    undoRedoStatus = pyqtSignal(bool, bool)
    dirtyStateChanged = pyqtSignal(bool)
    statusChanged = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.image_path: str = ""
        self.anno_path: str = ""
        self.image_width: int = 0
        self.image_height: int = 0
        self.annotations: List[AnnotationItem] = []
        self.image_list: List[str] = []
        self.image_to_label: Dict[str, str] = {}
        self.current_idx: int = -1
        self.clipboard_item: Optional[AnnotationItem] = None
        self.clipboard_annotations: List[AnnotationItem] = []
        self.last_error: str = ""
        self.current_pixmap: Optional[QPixmap] = None
        self.finished_images: Set[str] = set()
        self.doubt_images: Set[str] = set()
        self.project_json_path: str = ""
        
        self.history = CommandHistory()
        self.history.on_state_change = self.undoRedoStatus.emit
        
        self._is_dirty: bool = False
        self._saved_annotations_state: List[dict] = []

    @property
    def is_dirty(self) -> bool: return self._is_dirty

    @property
    def verified_images(self) -> Set[str]:
        return self.doubt_images

    def load_verification_status(self) -> None:
        self.doubt_images.clear()
        self.finished_images.clear()
        if self.project_json_path and os.path.exists(self.project_json_path):
            try:
                with open(self.project_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    doubt_list = data.get("doubt_images", data.get("verified_images", []))
                    self.doubt_images = set(doubt_list)
                    finished_list = data.get("finished_images", [])
                    self.finished_images = set(finished_list)
                    print(f"[LOG] Loaded {len(self.doubt_images)} doubt images and {len(self.finished_images)} finished images from {self.project_json_path}")
                self.sync_doubt_files_to_doubt_folders()
            except Exception as e:
                print(f"[LOG] Error loading verification status from {self.project_json_path}: {e}")

    def save_verification_status(self) -> None:
        if self.project_json_path:
            try:
                from datetime import datetime
                data = {
                    "doubt_images": list(self.doubt_images),
                    "finished_images": list(self.finished_images),
                    "total_doubt": len(self.doubt_images),
                    "total_finished": len(self.finished_images),
                    "last_updated": datetime.now().isoformat()
                }
                with open(self.project_json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                print(f"[LOG] Saved verification status ({len(self.finished_images)} finished, {len(self.doubt_images)} doubt) to {self.project_json_path}")
            except Exception as e:
                print(f"[LOG] Error saving verification status to {self.project_json_path}: {e}")

    def toggle_finished(self, image_path: Optional[str] = None) -> bool:
        target = image_path or self.image_path
        if not target:
            return False
        if target in self.finished_images:
            self.finished_images.remove(target)
            is_finished = False
        else:
            self.finished_images.add(target)
            is_finished = True
        self.save_verification_status()
        self.statusChanged.emit()
        return is_finished

    def mark_finished(self, image_path: Optional[str] = None) -> bool:
        target = image_path or self.image_path
        if not target:
            return False
        self.finished_images.add(target)
        self._saved_annotations_state = [item.to_dict() for item in self.annotations]
        self.set_dirty(False)
        self.save_verification_status()
        self.statusChanged.emit()
        return True

    def mark_unfinished(self, image_path: Optional[str] = None) -> bool:
        target = image_path or self.image_path
        if not target:
            return False
        if target in self.finished_images:
            self.finished_images.remove(target)
            self.save_verification_status()
            self.statusChanged.emit()
            return True
        return False

    def load_annotations_for_image(self, target_img_path: str) -> List[AnnotationItem]:
        if not target_img_path or target_img_path.startswith("sftp://"):
            return []
        lbl_path = self.image_to_label.get(target_img_path)
        if not lbl_path:
            base_no_ext = os.path.splitext(target_img_path)[0]
            for ext in (".txt", ".xml", ".json", ".csv"):
                p = base_no_ext + ext
                if os.path.exists(p):
                    lbl_path = p
                    break
            if not lbl_path:
                lbl_path = base_no_ext + ".txt"

        if lbl_path and os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
            parser = ParserFactory.get_parser_for_file(lbl_path)
            if parser:
                w = self.image_width if self.image_width > 0 else 1280
                h = self.image_height if self.image_height > 0 else 720
                return parser.load(lbl_path, w, h)
        return []

    def copy_annotations_from_image(self, src_img_path: str, replace_existing: bool = True) -> int:
        if not self.image_path or not src_img_path:
            return 0
        src_annos = self.load_annotations_for_image(src_img_path)
        if not src_annos:
            return 0

        if replace_existing:
            self.annotations.clear()

        copied_count = 0
        for item in src_annos:
            cloned = item.clone()
            self.annotations.append(cloned)
            copied_count += 1

        if copied_count > 0:
            self.save_annotations()
            self._on_annotations_changed()
        return copied_count

    def copy_annotations_from_prev_image(self) -> Tuple[int, str]:
        if self.current_idx <= 0 or not self.image_list:
            return 0, ""
        prev_path = self.image_list[self.current_idx - 1]
        count = self.copy_annotations_from_image(prev_path)
        return count, os.path.basename(prev_path)

    def copy_annotations_from_next_image(self) -> Tuple[int, str]:
        if self.current_idx < 0 or self.current_idx >= len(self.image_list) - 1:
            return 0, ""
        next_path = self.image_list[self.current_idx + 1]
        count = self.copy_annotations_from_image(next_path)
        return count, os.path.basename(next_path)

    def _move_file_to_doubt(self, img_path: str) -> str:
        """
        Moves the image crop file to `crops/doubt/` and its label file to `labels/doubt/`.
        Returns the updated image path.
        """
        if not img_path or img_path.startswith("sftp://") or not os.path.exists(img_path):
            return img_path

        img_dir = os.path.dirname(img_path)
        img_name = os.path.basename(img_path)

        if os.path.basename(img_dir).lower() == "doubt":
            return img_path

        import shutil
        parent_dir = os.path.dirname(img_dir) if os.path.basename(img_dir).lower() in ("crops", "images") else img_dir
        doubt_crop_dir = os.path.join(parent_dir, "crops", "doubt")
        os.makedirs(doubt_crop_dir, exist_ok=True)
        new_img_path = os.path.join(doubt_crop_dir, img_name)

        try:
            shutil.move(img_path, new_img_path)
            print(f"[LOG] Moved crop image to doubt directory: {new_img_path}")
        except Exception as e:
            print(f"[LOG] Error moving crop image to doubt directory: {e}")
            return img_path

        lbl_path = self.image_to_label.get(img_path)
        new_lbl_path = None

        candidate_lbls = []
        if lbl_path:
            candidate_lbls.append(lbl_path)
            
        base_name_no_ext = os.path.splitext(img_name)[0]
        parent_dir = os.path.dirname(img_dir)
        parallel_lbl = os.path.join(parent_dir, "labels", base_name_no_ext + ".txt")
        if os.path.exists(parallel_lbl) and parallel_lbl not in candidate_lbls:
            candidate_lbls.append(parallel_lbl)

        for src_lbl in candidate_lbls:
            if os.path.exists(src_lbl):
                l_dir = os.path.dirname(src_lbl)
                l_name = os.path.basename(src_lbl)
                d_lbl_dir = os.path.join(l_dir, "doubt") if os.path.basename(l_dir).lower() != "doubt" else l_dir
                os.makedirs(d_lbl_dir, exist_ok=True)
                dest_lbl = os.path.join(d_lbl_dir, l_name)
                try:
                    if src_lbl != dest_lbl:
                        shutil.move(src_lbl, dest_lbl)
                        print(f"[LOG] Moved label file to doubt directory: {dest_lbl}")
                    if not new_lbl_path:
                        new_lbl_path = dest_lbl
                except Exception as e:
                    print(f"[LOG] Error moving label file to doubt directory: {e}")

        if not new_lbl_path:
            if lbl_path:
                l_dir = os.path.dirname(lbl_path)
                l_name = os.path.basename(lbl_path)
                d_lbl_dir = os.path.join(l_dir, "doubt") if os.path.basename(l_dir).lower() != "doubt" else l_dir
                new_lbl_path = os.path.join(d_lbl_dir, l_name)
            else:
                new_lbl_path = os.path.splitext(new_img_path)[0] + ".txt"

        if img_path in self.image_list:
            idx = self.image_list.index(img_path)
            self.image_list[idx] = new_img_path

        if img_path in self.image_to_label:
            del self.image_to_label[img_path]
            self.image_to_label[new_img_path] = new_lbl_path

        if self.image_path == img_path:
            self.image_path = new_img_path
            self.anno_path = new_lbl_path

        if img_path in self.finished_images:
            self.finished_images.remove(img_path)
            self.finished_images.add(new_img_path)

        return new_img_path

    def _move_file_from_doubt(self, img_path: str) -> str:
        """
        Moves the image crop file and its label file back from `doubt/` subfolder to parent main folder.
        Returns the updated image path.
        """
        if not img_path or img_path.startswith("sftp://") or not os.path.exists(img_path):
            return img_path

        img_dir = os.path.dirname(img_path)
        img_name = os.path.basename(img_path)

        if os.path.basename(img_dir).lower() != "doubt":
            return img_path

        import shutil
        parent_img_dir = os.path.dirname(img_dir)
        new_img_path = os.path.join(parent_img_dir, img_name)

        try:
            shutil.move(img_path, new_img_path)
            print(f"[LOG] Moved crop image back from doubt directory: {new_img_path}")
        except Exception as e:
            print(f"[LOG] Error moving crop image from doubt directory: {e}")
            return img_path

        lbl_path = self.image_to_label.get(img_path)
        new_lbl_path = None

        candidate_lbls = []
        if lbl_path:
            candidate_lbls.append(lbl_path)

        base_name_no_ext = os.path.splitext(img_name)[0]
        parent_dir = os.path.dirname(parent_img_dir)
        parallel_lbl_doubt = os.path.join(parent_dir, "labels", "doubt", base_name_no_ext + ".txt")
        if os.path.exists(parallel_lbl_doubt) and parallel_lbl_doubt not in candidate_lbls:
            candidate_lbls.append(parallel_lbl_doubt)

        for src_lbl in candidate_lbls:
            if os.path.exists(src_lbl):
                l_dir = os.path.dirname(src_lbl)
                l_name = os.path.basename(src_lbl)
                parent_l_dir = os.path.dirname(l_dir) if os.path.basename(l_dir).lower() == "doubt" else l_dir
                dest_lbl = os.path.join(parent_l_dir, l_name)
                try:
                    if src_lbl != dest_lbl:
                        shutil.move(src_lbl, dest_lbl)
                        print(f"[LOG] Moved label file back from doubt directory: {dest_lbl}")
                    if not new_lbl_path:
                        new_lbl_path = dest_lbl
                except Exception as e:
                    print(f"[LOG] Error moving label file back from doubt directory: {e}")

        if not new_lbl_path:
            if lbl_path:
                l_dir = os.path.dirname(lbl_path)
                l_name = os.path.basename(lbl_path)
                parent_l_dir = os.path.dirname(l_dir) if os.path.basename(l_dir).lower() == "doubt" else l_dir
                new_lbl_path = os.path.join(parent_l_dir, l_name)
            else:
                new_lbl_path = os.path.splitext(new_img_path)[0] + ".txt"

        if img_path in self.image_list:
            idx = self.image_list.index(img_path)
            self.image_list[idx] = new_img_path

        if img_path in self.image_to_label:
            del self.image_to_label[img_path]
            self.image_to_label[new_img_path] = new_lbl_path

        if self.image_path == img_path:
            self.image_path = new_img_path
            self.anno_path = new_lbl_path

        if img_path in self.finished_images:
            self.finished_images.remove(img_path)
            self.finished_images.add(new_img_path)

        return new_img_path

    def sync_doubt_files_to_doubt_folders(self) -> None:
        """
        Ensures all doubt images in project_verification.json and their label files are moved to `doubt/` subfolders.
        """
        updated_doubt = set()
        for img_path in list(self.doubt_images):
            if os.path.exists(img_path):
                img_dir = os.path.dirname(img_path)
                if os.path.basename(img_dir).lower() != "doubt":
                    new_path = self._move_file_to_doubt(img_path)
                    updated_doubt.add(new_path)
                else:
                    updated_doubt.add(img_path)
            else:
                updated_doubt.add(img_path)
        self.doubt_images = updated_doubt
        self.save_verification_status()

    def toggle_verification(self, image_path: str) -> Tuple[bool, str]:
        if not image_path:
            return False, image_path
        if image_path in self.doubt_images:
            self.doubt_images.remove(image_path)
            new_path = self._move_file_from_doubt(image_path)
            is_doubt = False
        else:
            new_path = self._move_file_to_doubt(image_path)
            self.doubt_images.add(new_path)
            is_doubt = True
        self.save_verification_status()
        return is_doubt, new_path

    def toggle_doubt(self, image_path: str) -> Tuple[bool, str]:
        return self.toggle_verification(image_path)

    def set_dirty(self, dirty: bool) -> None:
        if self._is_dirty != dirty:
            self._is_dirty = dirty
            self.dirtyStateChanged.emit(dirty)

    def _check_dirty(self) -> None:
        current_state = [item.to_dict() for item in self.annotations]
        is_changed = (current_state != self._saved_annotations_state)
        self.set_dirty(is_changed)
        if is_changed and self.image_path and self.image_path in self.finished_images:
            print(f"[LOG] Annotation modified on finished image {self.image_path} -> Auto-reverting to RED / Unfinished")
            self.finished_images.remove(self.image_path)
            self.save_verification_status()
            self.statusChanged.emit()

    def _on_annotations_changed(self) -> None:
        self._check_dirty()
        self.stateChanged.emit()

    def load_folder(self, folder_path: str) -> None:
        if not os.path.isdir(folder_path): return
        supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        
        files = []
        self.image_to_label = {}
        
        for root, dirs, _ in os.walk(folder_path):
            for d in dirs:
                if d.lower() == "crops":
                    crops_dir = os.path.join(root, d)
                    labels_dir = None
                    try:
                        siblings = os.listdir(root)
                    except OSError:
                        siblings = []
                    for sibling in siblings:
                        if sibling.lower() in ("labels", "lables", "label", "lable", "annotations"):
                            sibling_path = os.path.join(root, sibling)
                            if os.path.isdir(sibling_path):
                                labels_dir = sibling_path
                                break
                    if not labels_dir:
                        labels_dir = os.path.join(root, "labels")
                        
                    for crop_root, _, filenames in os.walk(crops_dir):
                        for f in filenames:
                            if os.path.splitext(f)[1].lower() in supported:
                                img_path = os.path.join(crop_root, f)
                                files.append(img_path)
                                rel_path = os.path.relpath(img_path, crops_dir)
                                rel_no_ext = os.path.splitext(rel_path)[0]
                                found_label = None
                                for ext in (".txt", ".xml", ".json", ".csv"):
                                    p = os.path.join(labels_dir, rel_no_ext + ext)
                                    if os.path.exists(p):
                                        found_label = p
                                        break
                                if not found_label:
                                    found_label = os.path.join(labels_dir, rel_no_ext + ".txt")
                                self.image_to_label[img_path] = found_label

        if not files:
            for root, _, filenames in os.walk(folder_path):
                for f in filenames:
                    if os.path.splitext(f)[1].lower() in supported:
                        img_path = os.path.join(root, f)
                        files.append(img_path)
                        found_label = None
                        for ext in (".txt", ".xml", ".json", ".csv"):
                            p = os.path.splitext(img_path)[0] + ext
                            if os.path.exists(p):
                                found_label = p
                                break
                        if not found_label:
                            found_label = os.path.splitext(img_path)[0] + ".txt"
                        self.image_to_label[img_path] = found_label

        self.image_list = sorted(files, key=natural_sort_key)
        self.current_idx = -1
        self.project_json_path = os.path.join(folder_path, "project_verification.json")
        self.load_verification_status()

    def open_image(self, file_path: str) -> bool:
        print(f"[LOG] open_image called with: {file_path}")
        self.active_image_link = None
        
        img = None
        if file_path.startswith("sftp://"):
            print(f"[LOG] Image is remote SFTP path. Loading over SSH...")
            img = self._load_image_from_link(file_path)
        else:
            print(f"[LOG] Image is local path. Checking existence...")
            if not os.path.exists(file_path):
                print(f"[LOG] Local image does not exist: {file_path}")
                return False
            img = cv2.imread(file_path)
            
        if img is None:
            print(f"[LOG] Failed to load/decode image: {file_path}")
            return False
            
        self.image_height, self.image_width = img.shape[:2]
        print(f"[LOG] Image loaded successfully. Dimensions: {self.image_width}x{self.image_height}")
        self.image_path = file_path
        
        # Convert BGR to RGB and cache QPixmap in memory
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.current_pixmap = QPixmap.fromImage(qimg.copy())
        
        if not file_path.startswith("sftp://"):
            folder = os.path.dirname(file_path)
            if file_path not in self.image_list:
                print(f"[LOG] Image not in current list. Loading parent folder: {folder}")
                self.load_folder(folder)
                
        if file_path in self.image_list:
            self.current_idx = self.image_list.index(file_path)
        else:
            self.image_list = [file_path]
            self.current_idx = 0
        print(f"[LOG] Active image index set to {self.current_idx} of {len(self.image_list)} files.")

        self.history.clear()
        self.annotations.clear()
        
        base_no_ext = os.path.splitext(file_path)[0]
        label_path = self.image_to_label.get(file_path)
        if not label_path:
            if not file_path.startswith("sftp://"):
                for ext in (".txt", ".xml", ".json", ".csv"):
                    p = base_no_ext + ext
                    if os.path.exists(p):
                        label_path = p
                        break
            if not label_path:
                label_path = base_no_ext + ".txt"
        print(f"[LOG] Associated label file path: {label_path}")
            
        self.anno_path = label_path
        
        loaded = False
        parser = ParserFactory.get_parser_for_file(label_path)
        if parser:
            print(f"[LOG] Parsing label file using: {parser.__class__.__name__}")
            self.annotations = parser.load(label_path, self.image_width, self.image_height)
            if isinstance(parser, YoloParser) and parser.image_link:
                self.active_image_link = parser.image_link
            loaded = True
                
        if not loaded:
            print(f"[LOG] No parser found or failed to parse label: {label_path}")
            self.annotations = []
        else:
            print(f"[LOG] Parsed annotations count: {len(self.annotations)}")

        self._saved_annotations_state = [item.to_dict() for item in self.annotations]
        self.set_dirty(False)
        self.imageLoaded.emit(self.image_path)
        self.stateChanged.emit()
        return True

    def open_annotation_file(self, file_path: str) -> bool:
        self.last_error = ""
        self.active_image_link = None
        if not os.path.exists(file_path):
            self.last_error = f"File does not exist: {file_path}"
            return False
        
        print(f"[DEBUG] open_annotation_file called with path: {file_path}")
        print(f"[DEBUG] File exists: {os.path.exists(file_path)}")
        ext = os.path.splitext(file_path)[1].lower()
        print(f"[DEBUG] Ext: {ext}")
        if ext == ".txt":
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
            except Exception as e:
                self.last_error = f"Failed to read TXT file: {e}"
                return False
            print(f"[DEBUG] txt first_line: {first_line}")
            if first_line.startswith(("http://", "https://")) or not first_line.split() or len(first_line.split()) == 1:
                img = self._load_image_from_link(first_line)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w, ch = img_rgb.shape
                    qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                    self.current_pixmap = QPixmap.fromImage(qimg.copy())
                    self.image_height, self.image_width = img.shape[:2]
                    self.image_path = first_line
                    self.anno_path = file_path
                    self.active_image_link = first_line
                    self.history.clear()
                    parser = YoloParser()
                    self.annotations = parser.load(file_path, self.image_width, self.image_height)
                    self._saved_annotations_state = [item.to_dict() for item in self.annotations]
                    self.set_dirty(False)
                    self.imageLoaded.emit(self.image_path)
                    self.stateChanged.emit()
                    print(f"[DEBUG] txt link loader success!")
                    return True
                else:
                    self.last_error = f"Failed to fetch linked image URL: {first_line}"
                    return False
        elif ext == ".csv":
            try:
                import csv
                is_global_csv = False
                csv_rows = []
                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    first_row = next(reader, None)
                    print(f"[DEBUG] csv first_row: {first_row}")
                    if first_row and len(first_row) >= 2:
                        if "image_path" in first_row[0].lower() or "label_path" in first_row[1].lower():
                            is_global_csv = True
                        else:
                            for ext_img in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                                if first_row[0].lower().endswith(ext_img):
                                    is_global_csv = True
                                    csv_rows.append(first_row)
                                    break
                        if is_global_csv:
                            for r in reader:
                                if len(r) >= 2:
                                    csv_rows.append(r)
                print(f"[DEBUG] csv is_global_csv: {is_global_csv}, rows parsed: {len(csv_rows)}")
                if is_global_csv:
                    has_remote_paths = False
                    for r in csv_rows:
                        if r[0].startswith("sftp://"):
                            has_remote_paths = True
                            break
                            
                    local_mount_dir = None
                    if has_remote_paths:
                        from PyQt6.QtWidgets import QMessageBox, QFileDialog
                        res = QMessageBox.question(
                            None, "Remote SFTP Paths Detected",
                            "The loaded CSV contains remote sftp:// paths.\n\n"
                            "Do you want to select a local folder where this remote directory is mounted (using SSHFS/sftp) for 100x faster loading?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                        )
                        if res == QMessageBox.StandardButton.Yes:
                            local_mount_dir = QFileDialog.getExistingDirectory(
                                None, "Select Local SSHFS Mount Directory"
                            )
                            if not local_mount_dir:
                                local_mount_dir = None
                                
                    csv_dir = os.path.dirname(file_path)
                    files = []
                    self.image_to_label = {}
                    
                    remote_prefix = None
                    if local_mount_dir and csv_rows:
                        first_img = csv_rows[0][0]
                        for marker in ("/images/", "/labels/", "/crops/"):
                            if marker in first_img:
                                remote_prefix = first_img.split(marker)[0] + marker.rstrip("/")
                                break
                        if not remote_prefix:
                            remote_prefix = "/".join(first_img.split("/")[:-2])
                            
                    for row in csv_rows:
                        img_p = row[0]
                        lbl_p = row[1]
                        
                        if local_mount_dir and remote_prefix:
                            base_url = remote_prefix
                            for suffix in ("/images", "/labels", "/crops"):
                                if base_url.endswith(suffix):
                                    base_url = base_url[:-len(suffix)]
                                    break
                                    
                            if img_p.startswith(base_url):
                                rel_img = img_p[len(base_url):].lstrip("/")
                                img_p = os.path.join(local_mount_dir, rel_img)
                            if lbl_p.startswith(base_url):
                                rel_lbl = lbl_p[len(base_url):].lstrip("/")
                                lbl_p = os.path.join(local_mount_dir, rel_lbl)
                                
                        if not os.path.isabs(img_p) and not img_p.startswith("sftp://"):
                            img_p = os.path.join(csv_dir, img_p)
                        if not os.path.isabs(lbl_p) and not lbl_p.startswith("sftp://"):
                            lbl_p = os.path.join(csv_dir, lbl_p)
                        if not img_p.startswith("sftp://"):
                            img_p = os.path.abspath(img_p)
                        if not lbl_p.startswith("sftp://"):
                            lbl_p = os.path.abspath(lbl_p)
                        files.append(img_p)
                        self.image_to_label[img_p] = lbl_p
                    
                    self.finished_images.clear()
                    for f in files:
                        lbl = self.image_to_label.get(f)
                        if lbl:
                            if lbl.startswith("sftp://"):
                                self.finished_images.add(f)
                            elif os.path.exists(lbl) and os.path.getsize(lbl) > 0:
                                self.finished_images.add(f)
                                
                    print(f"[DEBUG] csv resolved files count: {len(files)}")
                    if files:
                        self.image_list = sorted(files)
                        self.current_idx = -1
                        if file_path.endswith(".csv"):
                            self.project_json_path = os.path.splitext(file_path)[0] + "_verification.json"
                        else:
                            self.project_json_path = os.path.join(os.path.dirname(file_path), "project_verification.json")
                        self.load_verification_status()
                        print(f"[DEBUG] csv loading first image: {self.image_list[0]}")
                        success = self.open_image(self.image_list[0])
                        print(f"[DEBUG] csv open_image status: {success}")
                        if not success:
                            self.last_error = f"Failed to open first CSV image path:\n{self.image_list[0]}\n(Check if the file exists and is valid)"
                        return success
                    else:
                        self.last_error = "No image paths found in global CSV."
                        return False
            except Exception as e:
                print(f"[DEBUG] Error parsing global CSV loader: {e}")
                self.last_error = f"Error reading global CSV loader: {e}"
                return False
                    
        # Fallback to loading standard matching image
        base_no_ext = os.path.splitext(file_path)[0]
        base_filename = os.path.basename(base_no_ext)
        parent_dir = os.path.dirname(file_path)
        grandparent_dir = os.path.dirname(parent_dir)
        
        img_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
        possible_imgs = []

        # 1. Direct path check with image extensions
        for img_ext in img_extensions:
            possible_imgs.append(base_no_ext + img_ext)

        # 2. Check inside sibling folders of parent directory (e.g. grandparent/crops, grandparent/images, grandparent/full_images, grandparent)
        for folder_name in ("crops", "images", "full_images", ""):
            target_dir = os.path.join(grandparent_dir, folder_name) if folder_name else grandparent_dir
            for img_ext in img_extensions:
                possible_imgs.append(os.path.join(target_dir, base_filename + img_ext))

        # 3. Path regex replacement for /labels/, /lables/, /label/, /lable/, /annotations/ -> /crops/, /images/, /full_images/
        for pattern in [r'[/\\\\]la?be?ls?[/\\\\]', r'[/\\\\]annotations?[/\\\\]']:
            for sub_folder in ("crops", "images", "full_images", ""):
                sub_str = f'/{sub_folder}/' if sub_folder else '/'
                new_path = re.sub(pattern, sub_str, file_path, flags=re.IGNORECASE)
                for img_ext in img_extensions:
                    possible_imgs.append(os.path.splitext(new_path)[0] + img_ext)

        # Preserve order while removing duplicates
        seen = set()
        unique_possible_imgs = []
        for img_p in possible_imgs:
            if img_p not in seen:
                seen.add(img_p)
                unique_possible_imgs.append(img_p)

        print(f"[DEBUG] Fallback unique_possible_imgs count: {len(unique_possible_imgs)}")
        for possible_img in unique_possible_imgs:
            exists = os.path.exists(possible_img)
            if exists:
                print(f"[DEBUG] Found matching image: {possible_img}")
                self.image_to_label[possible_img] = file_path
                success = self.open_image(possible_img)
                print(f"[DEBUG] Fallback open_image status: {success}")
                if not success:
                    self.last_error = f"Found matching image, but failed to load it:\n{possible_img}"
                return success

        self.last_error = f"Could not find matching image for label file:\n{file_path}\nChecked paths:\n" + "\n".join(unique_possible_imgs[:15])
        print(f"[DEBUG] Fallback failed completely.")
        return False

    def _load_image_from_link(self, link: str) -> Optional[np.ndarray]:
        if link.startswith(("http://", "https://")):
            try:
                import urllib.request
                req = urllib.request.Request(link, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10) as response:
                    img_bytes = response.read()
                nparr = np.frombuffer(img_bytes, np.uint8)
                return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"Error loading image from URL {link}: {e}")
                return None
        elif link.startswith("sftp://"):
            try:
                user_host, path = parse_sftp_url(link)
                import subprocess
                cmd = get_ssh_cmd(user_host, f"cat '{path}'")
                res = subprocess.run(cmd, capture_output=True, timeout=15)
                if res.returncode == 0:
                    img_bytes = res.stdout
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                else:
                    print(f"Failed to fetch SFTP image: {res.stderr.decode('utf-8', errors='ignore')}")
            except Exception as e:
                print(f"Error fetching SFTP image: {e}")
            return None
        else:
            if os.path.exists(link):
                return cv2.imread(link)
            return None

    def reload_annotation(self) -> None:
        if not self.image_path: return
        self.history.clear()
        self.annotations.clear()
        if os.path.exists(self.anno_path):
            parser = ParserFactory.get_parser_for_file(self.anno_path)
            if parser:
                self.annotations = parser.load(self.anno_path, self.image_width, self.image_height)
        self._saved_annotations_state = [item.to_dict() for item in self.annotations]
        self.set_dirty(False)
        self.stateChanged.emit()

    def _save_to_path_internal(self, target_path: str, parser: BaseParser, image_link: Optional[str] = None) -> bool:
        is_remote = target_path.startswith("sftp://")
        local_path = target_path
        if is_remote:
            temp_dir = "/home/sxr23/snap/antigravity/5/.gemini/antigravity/scratch/license_plate_annotator"
            local_path = os.path.join(temp_dir, ".temp_save" + os.path.splitext(target_path)[1])
            
        if isinstance(parser, YoloParser) and image_link:
            parser.save(local_path, self.annotations, self.image_width, self.image_height, image_link)
        else:
            parser.save(local_path, self.annotations, self.image_width, self.image_height)
            
        if not is_remote:
            # DUAL SAVE: Automatically save to parallel labels/ folder if existing or present
            base_dir = os.path.dirname(target_path)
            base_name = os.path.basename(target_path)
            parent_dir = os.path.dirname(base_dir)
            
            alt_dirs = [
                os.path.join(parent_dir, "labels"),
                os.path.join(base_dir, "labels")
            ]
            if os.path.basename(base_dir) == "labels":
                alt_dirs.append(os.path.join(parent_dir, "crops"))
                alt_dirs.append(os.path.join(parent_dir, "images"))
                
            for alt_dir in alt_dirs:
                if os.path.exists(alt_dir) and alt_dir != base_dir:
                    alt_path = os.path.join(alt_dir, base_name)
                    try:
                        if isinstance(parser, YoloParser) and image_link:
                            parser.save(alt_path, self.annotations, self.image_width, self.image_height, image_link)
                        else:
                            parser.save(alt_path, self.annotations, self.image_width, self.image_height)
                        print(f"[LOG] Dual saved label file to parallel directory: {alt_path}")
                    except Exception as e:
                        print(f"[LOG] Dual save warning: {e}")

        if is_remote:
            user_host, r_path = parse_sftp_url(target_path)
            success = write_remote_file(user_host, r_path, local_path)
            if os.path.exists(local_path):
                try: os.remove(local_path)
                except Exception: pass
            return success
        return True

    def save_annotations(self) -> bool:
        print(f"[LOG] save_annotations called. Target: {self.anno_path}")
        if not self.image_path or not self.anno_path:
            print(f"[LOG] Save aborted: image_path or anno_path not set.")
            return False
        parser = ParserFactory.get_parser_for_file(self.anno_path) or ParserFactory.get_parser_by_name("yolo")
        try:
            image_link = getattr(self, "active_image_link", None)
            print(f"[LOG] Target parser: {parser.__class__.__name__}. Annotations to save: {len(self.annotations)}")
            success = self._save_to_path_internal(self.anno_path, parser, image_link)
            if success:
                print(f"[LOG] save_annotations success: {self.anno_path}")
                self._saved_annotations_state = [item.to_dict() for item in self.annotations]
                self.set_dirty(False)
                if self.image_path:
                    self.finished_images.add(self.image_path)
                return True
            print(f"[LOG] save_annotations failed: {self.anno_path}")
            return False
        except Exception as e:
            print(f"[LOG] Save error: {e}")
            return False

    def save_annotations_as(self, file_path: str) -> bool:
        print(f"[LOG] save_annotations_as called. Target: {file_path}")
        parser = ParserFactory.get_parser_for_file(file_path)
        if not parser:
            print(f"[LOG] Save As aborted: unsupported target extension.")
            return False
        try:
            image_link = getattr(self, "active_image_link", None)
            print(f"[LOG] Target parser: {parser.__class__.__name__}. Annotations to save: {len(self.annotations)}")
            success = self._save_to_path_internal(file_path, parser, image_link)
            if success:
                print(f"[LOG] save_annotations_as success: {file_path}")
                self.anno_path = file_path
                self._saved_annotations_state = [item.to_dict() for item in self.annotations]
                self.set_dirty(False)
                return True
            print(f"[LOG] save_annotations_as failed: {file_path}")
            return False
        except Exception as e:
            print(f"[LOG] Save As error: {e}")
            return False

    def export_as_json(self, file_path: str) -> bool:
        parser = ParserFactory.get_parser_by_name("json")
        if not parser: return False
        try:
            parser.save(file_path, self.annotations, self.image_width, self.image_height)
            return True
        except Exception as e:
            print(f"Export error: {e}")
            return False

    def next_image(self) -> bool:
        if not self.image_list: return False
        idx = (self.current_idx + 1) % len(self.image_list)
        return self.open_image(self.image_list[idx])

    def prev_image(self) -> bool:
        if not self.image_list: return False
        idx = (self.current_idx - 1 + len(self.image_list)) % len(self.image_list)
        return self.open_image(self.image_list[idx])

    def clear_state(self) -> None:
        self.image_path = ""
        self.anno_path = ""
        self.image_width = 0
        self.image_height = 0
        self.annotations.clear()
        self.history.clear()
        self._saved_annotations_state.clear()
        self.set_dirty(False)
        self.stateChanged.emit()

    def add_annotation(self, geom) -> None:
        if isinstance(geom, BoundingBox):
            box = geom
        elif isinstance(geom, QRectF):
            box = BoundingBox(geom.x(), geom.y(), geom.width(), geom.height())
        else:
            box = BoundingBox(0, 0, 1, 1, geom)
        item = AnnotationItem("0", box)
        self.history.push(AddItemCommand(self.annotations, item, self._on_annotations_changed))

    def delete_annotations(self, items: List[AnnotationItem]) -> None:
        if not items: return
        self.history.push(DeleteItemCommand(self.annotations, items, self._on_annotations_changed))

    def duplicate_annotation(self, item: AnnotationItem) -> Optional[AnnotationItem]:
        if not item: return None
        offset = 15
        new_x = min(item.x + offset, self.image_width - item.width)
        new_y = min(item.y + offset, self.image_height - item.height)
        dup_box = BoundingBox(new_x, new_y, item.width, item.height)
        dup_item = AnnotationItem(item.label, dup_box, item.locked)
        self.history.push(AddItemCommand(self.annotations, dup_item, self._on_annotations_changed))
        return dup_item

    def copy_annotation(self, item: Optional[AnnotationItem] = None) -> int:
        self.clipboard_annotations.clear()
        if item:
            self.clipboard_annotations.append(item.clone())
            self.clipboard_item = item.clone()
        elif self.annotations:
            self.clipboard_annotations = [anno.clone() for anno in self.annotations]
            self.clipboard_item = self.annotations[0].clone()
        else:
            self.clipboard_item = None
        return len(self.clipboard_annotations)

    def paste_annotation(self, exact: bool = True) -> Optional[AnnotationItem]:
        if not self.clipboard_annotations and self.clipboard_item:
            self.clipboard_annotations = [self.clipboard_item.clone()]
            
        if not self.clipboard_annotations:
            return None

        self.annotations.clear()
        for item in self.clipboard_annotations:
            cloned = item.clone()
            if not exact:
                offset = 20
                cloned.x = max(0, min(cloned.x + offset, self.image_width - cloned.width))
                cloned.y = max(0, min(cloned.y + offset, self.image_height - cloned.height))
            self.annotations.append(cloned)

        self.save_annotations()
        self._on_annotations_changed()
        return self.annotations[0] if self.annotations else None

    def commit_geometry_change(self, item, old_box: BoundingBox) -> None:
        anno_item = item.annotation_item if hasattr(item, "annotation_item") else item
        new_box = anno_item.box.clone()
        if old_box.points != new_box.points:
            cmd = ModifyItemCommand(anno_item, anno_item.label, anno_item.label, old_box, new_box, anno_item.locked, anno_item.locked, self._on_annotations_changed)
            self.history.push(cmd)

    def modify_property(self, item: AnnotationItem, prop_name: str, value) -> None:
        old_box = item.box.clone()
        new_box = item.box.clone()
        old_lbl, new_lbl = item.label, item.label
        old_lock, new_lock = item.locked, item.locked

        if prop_name == "label": new_lbl = str(value)
        elif prop_name == "locked": new_lock = bool(value)
        elif prop_name in ("x", "y", "width", "height", "center_x", "center_y"):
            temp = AnnotationItem(item.label, new_box, False)
            setattr(temp, prop_name, value)
            new_box = temp.box

        cmd = ModifyItemCommand(item, old_lbl, new_lbl, old_box, new_box, old_lock, new_lock, self._on_annotations_changed)
        self.history.push(cmd)

    def delete_image_files_from_disk(self, target_image_path: Optional[str] = None) -> bool:
        """
        Permanently deletes the crop image file and its associated label files from disk.
        Cleans up from image_list, image_to_label, doubt_images, and finished_images.
        """
        img_path = target_image_path or self.image_path
        if not img_path:
            return False

        lbl_path = self.image_to_label.get(img_path) or (os.path.splitext(img_path)[0] + ".txt")

        base_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
        img_dir = os.path.dirname(img_path)
        parent_dir = os.path.dirname(img_dir) if os.path.basename(img_dir).lower() == "doubt" else os.path.dirname(img_dir)
        if os.path.basename(os.path.dirname(parent_dir)).lower() in ("crops", "labels", "train", "val", "test"):
            root_dir = os.path.dirname(parent_dir)
        else:
            root_dir = parent_dir

        files_to_delete = set()
        if os.path.exists(img_path):
            files_to_delete.add(img_path)
        if lbl_path and os.path.exists(lbl_path):
            files_to_delete.add(lbl_path)

        candidate_paths = [
            os.path.join(root_dir, "crops", base_name_no_ext + ".jpg"),
            os.path.join(root_dir, "crops", base_name_no_ext + ".png"),
            os.path.join(root_dir, "crops", "doubt", base_name_no_ext + ".jpg"),
            os.path.join(root_dir, "crops", "doubt", base_name_no_ext + ".png"),
            os.path.join(root_dir, "labels", base_name_no_ext + ".txt"),
            os.path.join(root_dir, "labels", base_name_no_ext + ".xml"),
            os.path.join(root_dir, "labels", base_name_no_ext + ".json"),
            os.path.join(root_dir, "labels", base_name_no_ext + ".csv"),
            os.path.join(root_dir, "labels", "doubt", base_name_no_ext + ".txt"),
            os.path.join(root_dir, "labels", "doubt", base_name_no_ext + ".xml"),
            os.path.join(root_dir, "labels", "doubt", base_name_no_ext + ".json"),
            os.path.join(root_dir, "labels", "doubt", base_name_no_ext + ".csv"),
            os.path.splitext(img_path)[0] + ".txt",
            os.path.splitext(img_path)[0] + ".xml",
            os.path.splitext(img_path)[0] + ".json",
            os.path.splitext(img_path)[0] + ".csv",
        ]
        for cp in candidate_paths:
            if os.path.exists(cp):
                files_to_delete.add(cp)

        for p in files_to_delete:
            try:
                os.remove(p)
                print(f"[LOG] Deleted file from disk: {p}")
            except Exception as e:
                print(f"[LOG] Error deleting file {p}: {e}")

        was_current = (img_path == self.image_path)
        cur_idx = self.current_idx if (0 <= self.current_idx < len(self.image_list)) else -1

        if img_path in self.image_list:
            cur_idx = self.image_list.index(img_path)
            self.image_list.remove(img_path)

        if img_path in self.image_to_label:
            del self.image_to_label[img_path]

        if img_path in self.finished_images:
            self.finished_images.remove(img_path)

        if img_path in self.doubt_images:
            self.doubt_images.remove(img_path)
            self.save_verification_status()

        if was_current:
            if self.image_list:
                next_idx = min(cur_idx, len(self.image_list) - 1)
                self.open_image(self.image_list[next_idx])
            else:
                self.image_path = ""
                self.anno_path = ""
                self.annotations = []
                self.current_pixmap = None
                self.current_idx = -1

        return True

    def undo(self) -> None: self.history.undo()
    def redo(self) -> None: self.history.redo()


# ==============================================================================
# 6. GRAPHICS VIEW COMPONENTS
# ==============================================================================

class BBoxItemSignals(QObject):
    geometryChanged = pyqtSignal(object)
    geometryEditFinished = pyqtSignal(object, object)
    selected = pyqtSignal(object)


class BBoxGraphicItem(QGraphicsPolygonItem):
    HANDLE_SIZE = 14.0

    def __init__(self, annotation_item: AnnotationItem):
        super().__init__()
        self.annotation_item = annotation_item
        self.signals = BBoxItemSignals()

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self._is_resizing = False
        self._active_handle = -1
        self._drag_start_box = None
        self._drag_start_points = []
        self._drag_start_pos = QPointF()

        self.border_color = QColor(0, 162, 232)
        self.selected_color = QColor(255, 127, 39)
        self.locked_color = QColor(127, 127, 127)
        self.update_from_model()

    def update_from_model(self) -> None:
        from PyQt6.QtGui import QPolygonF
        box = self.annotation_item.box
        poly = QPolygonF([QPointF(p[0], p[1]) for p in box.points])
        self.setPolygon(poly)
        self.setPos(0, 0)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, not self.annotation_item.locked)
        self.update()

    def get_view_scale(self) -> float:
        if self.scene() and self.scene().views():
            return self.scene().views()[0].transform().m11()
        return 1.0

    def get_handle_rects(self) -> list[QRectF]:
        poly = self.polygon()
        scale = self.get_view_scale()
        h_sz = self.HANDLE_SIZE / scale
        half = h_sz / 2.0
        rects = []
        for i in range(poly.size()):
            pt = poly.at(i)
            rects.append(QRectF(pt.x() - half, pt.y() - half, h_sz, h_sz))
        return rects

    def _get_handle_at_point(self, scene_pos: QPointF) -> int:
        local_pos = self.mapFromScene(scene_pos)
        for idx, h_rect in enumerate(self.get_handle_rects()):
            if h_rect.contains(local_pos): return idx
        return -1

    def _get_cursor_for_handle(self, idx: int) -> Qt.CursorShape:
        return Qt.CursorShape.CrossCursor

    def hoverMoveEvent(self, event) -> None:
        if self.annotation_item.locked:
            self.setCursor(Qt.CursorShape.ForbiddenCursor)
            super().hoverMoveEvent(event)
            return
        if self.isSelected():
            h_idx = self._get_handle_at_point(event.scenePos())
            if h_idx != -1:
                self.setCursor(self._get_cursor_for_handle(h_idx))
                return
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event) -> None:
        self.signals.selected.emit(self)
        if self.annotation_item.locked:
            event.accept()
            return
        self._drag_start_box = self.annotation_item.box.clone()
        self._drag_start_points = [QPointF(p[0], p[1]) for p in self.annotation_item.box.points]
        self._drag_start_pos = event.scenePos()
        if self.isSelected():
            h_idx = self._get_handle_at_point(event.scenePos())
            if h_idx != -1:
                self._is_resizing = True
                self._active_handle = h_idx
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.annotation_item.locked:
            event.accept()
            return
        if self._is_resizing:
            delta = event.scenePos() - self._drag_start_pos
            idx = self._active_handle
            orig_pt = self._drag_start_points[idx]
            new_pt = orig_pt + delta
            self.annotation_item.box.points[idx] = (new_pt.x(), new_pt.y())
            self.annotation_item.box._update_bounds()
            self.update_from_model()
            self.signals.geometryChanged.emit(self)
            event.accept()
            return
        super().mouseMoveEvent(event)
        dx, dy = self.x(), self.y()
        pts = [(p.x() + dx, p.y() + dy) for p in self._drag_start_points]
        self.annotation_item.box.points = pts
        self.annotation_item.box._update_bounds()
        self.signals.geometryChanged.emit(self)

    def mouseReleaseEvent(self, event) -> None:
        if self.annotation_item.locked:
            event.accept()
            return
        if self._is_resizing:
            self._is_resizing = False
            self.signals.geometryEditFinished.emit(self, self._drag_start_box)
            event.accept()
            return
        dx, dy = self.x(), self.y()
        if dx != 0 or dy != 0:
            pts = [(p.x() + dx, p.y() + dy) for p in self._drag_start_points]
            self.annotation_item.box.points = pts
            self.annotation_item.box._update_bounds()
            self.setPos(0, 0)
            self.update_from_model()
        super().mouseReleaseEvent(event)
        self.signals.geometryEditFinished.emit(self, self._drag_start_box)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget = None) -> None:
        poly = self.polygon()
        if poly.isEmpty(): return
        scale = self.get_view_scale()
        color = self.locked_color if self.annotation_item.locked else (self.selected_color if self.isSelected() else self.border_color)

        pen = QPen(color, 3.0 / scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.MiterJoin)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 25 if self.isSelected() else 10)))
        painter.drawPolygon(poly)

        label = self.annotation_item.label
        if label:
            painter.save()
            font = QFont("Outfit", int(round(max(10.0, 12.0 / scale))))
            painter.setFont(font)
            fm = painter.fontMetrics()
            tw, th = fm.horizontalAdvance(label), fm.height()
            
            pts = [poly.at(i) for i in range(poly.size())]
            top_pt = min(pts, key=lambda p: p.y())
            banner = QRectF(top_pt.x(), top_pt.y() - th - (4.0 / scale), tw + (8.0 / scale), th + (2.0 / scale))
            
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 200)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(banner)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(banner.translated(4.0 / scale, 1.0 / scale), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)
            painter.restore()

        if not self.annotation_item.locked:
            painter.save()
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(color, 1.5 / scale))
            for h_rect in self.get_handle_rects():
                painter.drawRect(h_rect)
            painter.restore()


class ImageViewer(QGraphicsView):
    bboxCreated = pyqtSignal(object)
    selectionChanged = pyqtSignal()
    zoomChanged = pyqtSignal(float)

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.scene_obj = QGraphicsScene(self)
        self.setScene(self.scene_obj)
        self.image_item = None
        self._raw_pixmap = None

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        
        self.zoom_factor = 1.15
        self.min_zoom, self.max_zoom = 0.05, 50.0
        self._mode = "select"
        self._is_drawing_new = False
        self._draw_start_pt = QPointF()
        self._temp_rect_item = None
        self._is_panning = False
        self._pan_start_pos = QPointF()

        # 4-Point Click Polygon Drawing Fields
        self._polygon_points = []
        self._temp_poly_item = None
        self._temp_point_markers = []

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        if mode == "pan":
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)

        if mode in ("create", "polygon"):
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif mode == "pan":
            pass # ScrollHandDrag sets hand cursors automatically
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        if mode != "polygon":
            self.clear_polygon_drawing()

    def clear_polygon_drawing(self) -> None:
        self._polygon_points = []
        if self._temp_poly_item:
            if self._temp_poly_item.scene() == self.scene_obj:
                self.scene_obj.removeItem(self._temp_poly_item)
            self._temp_poly_item = None
        for marker in self._temp_point_markers:
            if marker.scene() == self.scene_obj:
                self.scene_obj.removeItem(marker)
        self._temp_point_markers = []

    def load_image(self, image_path: str) -> QPixmap:
        self.clear_polygon_drawing()
        self.scene_obj.clear()
        self.image_item = None
        self._raw_pixmap = None
        if not os.path.exists(image_path): return None
        
        pixmap = QPixmap(image_path)
        if pixmap.isNull(): return None
        
        self._raw_pixmap = pixmap
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.image_item.setZValue(-1)
        self.scene_obj.addItem(self.image_item)
        self.scene_obj.setSceneRect(QRectF(0, 0, pixmap.width(), pixmap.height()))
        self.reset_zoom()
        return pixmap

    def load_image_from_pixmap(self, pixmap: QPixmap) -> QPixmap:
        self.clear_polygon_drawing()
        self.scene_obj.clear()
        self.image_item = None
        self._raw_pixmap = None
        if pixmap.isNull(): return None
        
        self._raw_pixmap = pixmap
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.image_item.setZValue(-1)
        self.scene_obj.addItem(self.image_item)
        self.scene_obj.setSceneRect(QRectF(0, 0, pixmap.width(), pixmap.height()))
        self.reset_zoom()
        return pixmap

    def fit_image(self) -> None:
        if not self._raw_pixmap: return
        self._is_manually_zoomed = False
        self.fitInView(self.scene_obj.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.zoomChanged.emit(self.transform().m11() * 100.0)

    def fit_bbox(self, target_rect: QRectF, padding_factor: float = 0.35) -> None:
        if not self._raw_pixmap or target_rect.isEmpty():
            return
        
        w = target_rect.width()
        h = target_rect.height()
        pad_w = max(w * padding_factor, 15.0)
        pad_h = max(h * padding_factor, 15.0)
        
        padded_rect = QRectF(
            target_rect.x() - pad_w,
            target_rect.y() - pad_h,
            target_rect.width() + (2 * pad_w),
            target_rect.height() + (2 * pad_h)
        ).intersected(self.scene_obj.sceneRect())
        
        if padded_rect.width() > 0 and padded_rect.height() > 0:
            self._is_manually_zoomed = True
            self.fitInView(padded_rect, Qt.AspectRatioMode.KeepAspectRatio)
            self.zoomChanged.emit(self.transform().m11() * 100.0)

    def reset_zoom(self) -> None:
        self._is_manually_zoomed = True
        self.setTransform(self.transform().fromScale(1.0, 1.0))
        self.centerOn(self.scene_obj.sceneRect().center())
        self.zoomChanged.emit(100.0)

    def zoom_by_factor(self, factor: float) -> None:
        if not self._raw_pixmap: return
        self._is_manually_zoomed = True
        cursor_pos = self.mapFromGlobal(QCursor.pos())
        local_pos = self.mapFromGlobal(QCursor.pos())
        if self.viewport().rect().contains(self.mapFromGlobal(QCursor.pos())):
            old_scene_pos = self.mapToScene(self.mapFromGlobal(QCursor.pos()))
        else:
            old_scene_pos = self.mapToScene(self.viewport().rect().center())
            
        new_scale = self.transform().m11() * factor
        if new_scale < self.min_zoom: factor = self.min_zoom / self.transform().m11()
        elif new_scale > self.max_zoom: factor = self.max_zoom / self.transform().m11()
        
        self.scale(factor, factor)
        
        if self.viewport().rect().contains(self.mapFromGlobal(QCursor.pos())):
            new_scene_pos = self.mapToScene(self.mapFromGlobal(QCursor.pos()))
        else:
            new_scene_pos = self.mapToScene(self.viewport().rect().center())
            
        delta = old_scene_pos - new_scene_pos
        self.horizontalScrollBar().setValue(int(round(self.horizontalScrollBar().value() + delta.x() * self.transform().m11())))
        self.verticalScrollBar().setValue(int(round(self.verticalScrollBar().value() + delta.y() * self.transform().m22())))
        self.zoomChanged.emit(self.transform().m11() * 100.0)

    def wheelEvent(self, event) -> None:
        if not self._raw_pixmap:
            super().wheelEvent(event)
            return
        self._is_manually_zoomed = True
        old_scene_pos = self.mapToScene(event.position().toPoint())
        factor = self.zoom_factor if event.angleDelta().y() > 0 else (1.0 / self.zoom_factor)
        new_scale = self.transform().m11() * factor
        if new_scale < self.min_zoom: factor = self.min_zoom / self.transform().m11()
        elif new_scale > self.max_zoom: factor = self.max_zoom / self.transform().m11()
        
        self.scale(factor, factor)
        new_scene_pos = self.mapToScene(event.position().toPoint())
        delta = old_scene_pos - new_scene_pos
        self.horizontalScrollBar().setValue(int(round(self.horizontalScrollBar().value() + delta.x() * self.transform().m11())))
        self.verticalScrollBar().setValue(int(round(self.verticalScrollBar().value() + delta.y() * self.transform().m22())))
        self.zoomChanged.emit(self.transform().m11() * 100.0)

    def mousePressEvent(self, event) -> None:
        is_left = event.button() == Qt.MouseButton.LeftButton
        is_zoomed = (self.horizontalScrollBar().maximum() > 0 or self.verticalScrollBar().maximum() > 0)
        
        # Check if clicking on empty space / non-bbox
        item = self.itemAt(event.position().toPoint())
        is_bbox_element = False
        curr_item = item
        while curr_item:
            if isinstance(curr_item, BBoxGraphicItem):
                is_bbox_element = True
                break
            curr_item = curr_item.parentItem()
            
        prev_selected = self.get_selected_item()
        
        should_pan = False
        if is_left and is_zoomed and self._mode in ("select", "pan"):
            if not is_bbox_element:
                should_pan = True

        if event.button() == Qt.MouseButton.MiddleButton or (event.button() == Qt.MouseButton.LeftButton and event.modifiers() == Qt.KeyboardModifier.ShiftModifier) or should_pan:
            self._is_panning = True
            self._pan_start_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if self._mode == "polygon":
            if event.button() == Qt.MouseButton.LeftButton:
                scene_pos = self.mapToScene(event.position().toPoint())
                s_rect = self.scene_obj.sceneRect()
                if s_rect.contains(scene_pos):
                    if len(self._polygon_points) < 4:
                        self._polygon_points.append(scene_pos)
                        
                        scale = self.transform().m11()
                        r = 4.0 / scale
                        marker = self.scene_obj.addEllipse(
                            scene_pos.x() - r, scene_pos.y() - r, 2*r, 2*r,
                            QPen(QColor(0, 162, 232), 1.5 / scale),
                            QBrush(QColor(0, 162, 232, 200))
                        )
                        self._temp_point_markers.append(marker)
                        
                        if len(self._polygon_points) > 1:
                            path = QPainterPath()
                            path.moveTo(self._polygon_points[0])
                            for pt in self._polygon_points[1:]:
                                path.lineTo(pt)
                            
                            if len(self._polygon_points) == 4:
                                path.closeSubpath()

                            if not self._temp_poly_item:
                                self._temp_poly_item = QGraphicsPathItem()
                                self._temp_poly_item.setPen(QPen(QColor(0, 162, 232), 2.0 / scale, Qt.PenStyle.DashLine))
                                self._temp_poly_item.setBrush(QBrush(QColor(0, 162, 232, 25)))
                                self.scene_obj.addItem(self._temp_poly_item)
                                
                            self._temp_poly_item.setPath(path)

                        if len(self._polygon_points) == 4:
                            self._finish_polygon_drawing()
                event.accept()
                return
            elif event.button() == Qt.MouseButton.RightButton:
                self.clear_polygon_drawing()
                self.set_mode("select")
                event.accept()
                return
        if self._mode == "create" and event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            if self.scene_obj.sceneRect().contains(scene_pos):
                self._is_drawing_new = True
                self._draw_start_pt = scene_pos
                self._temp_rect_item = QGraphicsRectItem(QRectF(scene_pos.x(), scene_pos.y(), 0, 0))
                self._temp_rect_item.setPen(QPen(QColor(255, 127, 39), 2.0 / self.transform().m11(), Qt.PenStyle.DashLine))
                self.scene_obj.addItem(self._temp_rect_item)
                event.accept()
                return
        super().mousePressEvent(event)
        
        # Restore selection if clicked in empty space in select mode
        if self._mode == "select" and not is_bbox_element and prev_selected:
            self.select_item(prev_selected)

    def _finish_polygon_drawing(self) -> None:
        if len(self._polygon_points) == 4:
            pts_tuples = [(pt.x(), pt.y()) for pt in self._polygon_points]
            xs = [p[0] for p in pts_tuples]
            ys = [p[1] for p in pts_tuples]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            box = BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y, pts_tuples)
            self.clear_polygon_drawing()
            
            if box.width >= 5.0 and box.height >= 5.0:
                self.bboxCreated.emit(box)
        else:
            self.clear_polygon_drawing()
    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        main_win = self.window()
        if hasattr(main_win, "chk_always_fit") and main_win.chk_always_fit and main_win.chk_always_fit.isChecked():
            if not getattr(self, "_is_manually_zoomed", False):
                self.fit_image()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.scene_obj.clearSelection()
            self.selectionChanged.emit()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._prev_mode = self._mode
            self.set_mode("pan")
            event.accept()
            return
        if self._mode == "polygon" and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._finish_polygon_drawing()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            if hasattr(self, "_prev_mode"):
                self.set_mode(self._prev_mode)
                # Sync toolbar buttons in MainWindow
                main_win = self.window()
                if hasattr(main_win, "_set_editor_mode"):
                    main_win._set_editor_mode(self._prev_mode)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._is_panning:
            delta = event.position() - self._pan_start_pos
            self._pan_start_pos = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(round(delta.x())))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(round(delta.y())))
            event.accept()
            return
        if self._mode == "polygon" and self._polygon_points:
            curr = self.mapToScene(event.position().toPoint())
            scale = self.transform().m11()
            
            path = QPainterPath()
            path.moveTo(self._polygon_points[0])
            for pt in self._polygon_points[1:]:
                path.lineTo(pt)
            path.lineTo(curr)
            
            if len(self._polygon_points) == 3:
                path.closeSubpath()
                
            try:
                if self._temp_poly_item and self._temp_poly_item.scene() != self.scene_obj:
                    self._temp_poly_item = None
            except RuntimeError:
                self._temp_poly_item = None

            if not self._temp_poly_item:
                self._temp_poly_item = QGraphicsPathItem()
                self._temp_poly_item.setPen(QPen(QColor(0, 162, 232), 2.0 / scale, Qt.PenStyle.DashLine))
                self._temp_poly_item.setBrush(QBrush(QColor(0, 162, 232, 25)))
                self.scene_obj.addItem(self._temp_poly_item)
                
            try:
                self._temp_poly_item.setPath(path)
            except RuntimeError:
                self._temp_poly_item = QGraphicsPathItem()
                self._temp_poly_item.setPen(QPen(QColor(0, 162, 232), 2.0 / scale, Qt.PenStyle.DashLine))
                self._temp_poly_item.setBrush(QBrush(QColor(0, 162, 232, 25)))
                self.scene_obj.addItem(self._temp_poly_item)
                self._temp_poly_item.setPath(path)

            event.accept()
            return
        if self._is_drawing_new:
            try:
                if self._temp_rect_item and self._temp_rect_item.scene() == self.scene_obj:
                    curr = self.mapToScene(event.position().toPoint())
                    s_rect = self.scene_obj.sceneRect()
                    cx = max(s_rect.left(), min(curr.x(), s_rect.right()))
                    cy = max(s_rect.top(), min(curr.y(), s_rect.bottom()))
                    x, y = min(self._draw_start_pt.x(), cx), min(self._draw_start_pt.y(), cy)
                    w, h = abs(self._draw_start_pt.x() - cx), abs(self._draw_start_pt.y() - cy)
                    self._temp_rect_item.setRect(QRectF(x, y, w, h))
            except RuntimeError:
                self._is_drawing_new = False
                self._temp_rect_item = None
            event.accept()
            return

        if event.buttons() == Qt.MouseButton.NoButton:
            is_zoomed = (self.horizontalScrollBar().maximum() > 0 or self.verticalScrollBar().maximum() > 0)
            if is_zoomed and self._mode in ("select", "pan"):
                item = self.itemAt(event.position().toPoint())
                is_bbox_element = False
                curr_item = item
                while curr_item:
                    if isinstance(curr_item, BBoxGraphicItem):
                        is_bbox_element = True
                        break
                    curr_item = curr_item.parentItem()
                if not is_bbox_element:
                    self.setCursor(Qt.CursorShape.OpenHandCursor)
                else:
                    self.setCursor(Qt.CursorShape.ArrowCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._is_panning:
            self._is_panning = False
            is_zoomed = (self.horizontalScrollBar().maximum() > 0 or self.verticalScrollBar().maximum() > 0)
            if is_zoomed and self._mode in ("select", "pan"):
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        if self._mode == "polygon":
            event.accept()
            return
        if self._is_drawing_new:
            self._is_drawing_new = False
            if self._temp_rect_item:
                final = self._temp_rect_item.rect()
                self.scene_obj.removeItem(self._temp_rect_item)
                self._temp_rect_item = None
                if final.width() >= 10.0 and final.height() >= 10.0:
                    self.bboxCreated.emit(final)
            self.set_mode("select")
            event.accept()
            return
        super().mouseReleaseEvent(event)
        self.selectionChanged.emit()

    def get_selected_item(self) -> Optional[BBoxGraphicItem]:
        for item in self.scene_obj.selectedItems():
            if isinstance(item, BBoxGraphicItem): return item
        return None

    def select_item(self, item: BBoxGraphicItem) -> None:
        self.scene_obj.clearSelection()
        if item:
            item.setSelected(True)
            self.selectionChanged.emit()

    def clear_canvas(self) -> None:
        for item in list(self.scene_obj.items()):
            if isinstance(item, BBoxGraphicItem): self.scene_obj.removeItem(item)


# ==============================================================================
# 7. UI CONTROLS & PROPERTIES SIDEBAR
# ==============================================================================

class PropertiesPanel(QWidget):
    propertyChanged = pyqtSignal(str, object)
    lockToggled = pyqtSignal(bool)

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._current_item = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        title = QLabel("BBOX PROPERTIES")
        title.setFont(QFont("Outfit", 12, QFont.Weight.Bold))
        title.setStyleSheet("color: #00A2E8;")
        layout.addWidget(title)

        self.group_box = QGroupBox("Selected Box Details")
        self.group_box.setFont(QFont("Outfit", 10))
        group_layout = QFormLayout(self.group_box)
        group_layout.setSpacing(10)

        self.lbl_input = QLineEdit()
        self.lbl_input.setPlaceholderText("License Plate Text")
        self.spn_x = QSpinBox()
        self.spn_x.setRange(0, 99999)
        self.spn_y = QSpinBox()
        self.spn_y.setRange(0, 99999)
        self.spn_w = QSpinBox()
        self.spn_w.setRange(1, 99999)
        self.spn_h = QSpinBox()
        self.spn_h.setRange(1, 99999)
        self.spn_cx = QDoubleSpinBox()
        self.spn_cx.setRange(0.0, 99999.0)
        self.spn_cx.setDecimals(1)
        self.spn_cy = QDoubleSpinBox()
        self.spn_cy.setRange(0.0, 99999.0)
        self.spn_cy.setDecimals(1)

        group_layout.addRow("Label (Plate):", self.lbl_input)
        group_layout.addRow("Left X (px):", self.spn_x)
        group_layout.addRow("Top Y (px):", self.spn_y)
        group_layout.addRow("Width (px):", self.spn_w)
        group_layout.addRow("Height (px):", self.spn_h)
        group_layout.addRow("Center X (px):", self.spn_cx)
        group_layout.addRow("Center Y (px):", self.spn_cy)

        layout.addWidget(self.group_box)

        self.btn_lock = QPushButton("Lock Box")
        self.btn_lock.setCheckable(True)
        self.btn_lock.setStyleSheet("""
            QPushButton { background-color: #333333; color: white; border: 1px solid #555555; padding: 8px; border-radius: 4px; font-weight: bold; }
            QPushButton:checked { background-color: #E81123; border-color: #E81123; }
        """)
        layout.addWidget(self.btn_lock)

        btn_layout = QHBoxLayout()
        self.btn_duplicate = QPushButton("Duplicate")
        self.btn_duplicate.setStyleSheet("background-color: #2F3542; color: white; padding: 6px; border-radius: 4px;")
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setStyleSheet("background-color: #FF4757; color: white; padding: 6px; border-radius: 4px;")
        btn_layout.addWidget(self.btn_duplicate)
        btn_layout.addWidget(self.btn_delete)
        layout.addLayout(btn_layout)
        layout.addStretch()

        self.lbl_input.textChanged.connect(lambda t: self.propertyChanged.emit("label", t))
        self.spn_x.valueChanged.connect(lambda v: self.propertyChanged.emit("x", v))
        self.spn_y.valueChanged.connect(lambda v: self.propertyChanged.emit("y", v))
        self.spn_w.valueChanged.connect(lambda v: self.propertyChanged.emit("width", v))
        self.spn_h.valueChanged.connect(lambda v: self.propertyChanged.emit("height", v))
        self.spn_cx.valueChanged.connect(lambda v: self.propertyChanged.emit("center_x", v))
        self.spn_cy.valueChanged.connect(lambda v: self.propertyChanged.emit("center_y", v))
        self.btn_lock.toggled.connect(self._on_lock_toggled)

        self.set_selected_item(None)

    def set_selected_item(self, item: AnnotationItem, include_crop: bool = True) -> None:
        self._current_item = item
        if not item:
            self.group_box.setEnabled(False)
            self.btn_lock.setEnabled(False)
            self.btn_duplicate.setEnabled(False)
            self.btn_delete.setEnabled(False)
            self._clear_fields()
            return

        self.group_box.setEnabled(True)
        self.btn_lock.setEnabled(True)
        self.btn_duplicate.setEnabled(True)
        self.btn_delete.setEnabled(True)

        self._block_signals(True)
        self.lbl_input.setText(item.label)
        self.spn_x.setValue(item.x)
        self.spn_y.setValue(item.y)
        self.spn_w.setValue(item.width)
        self.spn_h.setValue(item.height)
        self.spn_cx.setValue(item.center_x)
        self.spn_cy.setValue(item.center_y)
        self.btn_lock.setChecked(item.locked)
        self.btn_lock.setText("Unlock Box" if item.locked else "Lock Box")

        locked = item.locked
        self.lbl_input.setEnabled(not locked)
        self.spn_x.setEnabled(not locked)
        self.spn_y.setEnabled(not locked)
        self.spn_w.setEnabled(not locked)
        self.spn_h.setEnabled(not locked)
        self.spn_cx.setEnabled(not locked)
        self.spn_cy.setEnabled(not locked)
        self._block_signals(False)

    def update_fields(self, include_crop: bool = True) -> None:
        if self._current_item: self.set_selected_item(self._current_item, include_crop=include_crop)

    def _clear_fields(self) -> None:
        self._block_signals(True)
        self.lbl_input.clear()
        self.spn_x.setValue(0)
        self.spn_y.setValue(0)
        self.spn_w.setValue(1)
        self.spn_h.setValue(1)
        self.spn_cx.setValue(0.0)
        self.spn_cy.setValue(0.0)
        self.btn_lock.setChecked(False)
        self.btn_lock.setText("Lock Box")
        self._block_signals(False)

    def _block_signals(self, block: bool) -> None:
        self.lbl_input.blockSignals(block)
        self.spn_x.blockSignals(block)
        self.spn_y.blockSignals(block)
        self.spn_w.blockSignals(block)
        self.spn_h.blockSignals(block)
        self.spn_cx.blockSignals(block)
        self.spn_cy.blockSignals(block)
        self.btn_lock.blockSignals(block)

    def _on_lock_toggled(self, checked: bool) -> None:
        self.btn_lock.setText("Unlock Box" if checked else "Lock Box")
        self.lockToggled.emit(checked)


# ==============================================================================
# 7.5. ANNOTATION PREVIEW DIALOG
# ==============================================================================

class AnnotationPreviewView(QGraphicsView):
    """
    A specialized QGraphicsView for the Annotation Preview popup.
    Supports smooth zooming, mouse-wheel zooming, dragging/panning, and auto-fit.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setStyleSheet("background-color: #121214; border: 1px solid #2A2A2E; border-radius: 4px;")
        self.zoom_factor = 1.15
        self.zoom_changed_callback = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_in()
        elif delta < 0:
            self.zoom_out()
        event.accept()

    def zoom_in(self):
        self.scale(self.zoom_factor, self.zoom_factor)
        if self.zoom_changed_callback:
            self.zoom_changed_callback(self.transform().m11())

    def zoom_out(self):
        self.scale(1.0 / self.zoom_factor, 1.0 / self.zoom_factor)
        if self.zoom_changed_callback:
            self.zoom_changed_callback(self.transform().m11())

    def set_zoom_level(self, level: float):
        current = self.transform().m11()
        if current > 0 and level > 0:
            factor = level / current
            self.scale(factor, factor)
            if self.zoom_changed_callback:
                self.zoom_changed_callback(self.transform().m11())

    def fit_image(self):
        if self.scene() and self.scene().sceneRect().isValid():
            self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            if self.zoom_changed_callback:
                self.zoom_changed_callback(self.transform().m11())


class AnnotationPreviewDialog(QDialog):
    """
    Separate Full Image + Annotation Preview Popup Window.
    Displays the full original image with all bounding boxes, polygons, and readable label badges.
    Allows changing preview image dimensions (e.g. 720x720, 560x720, 1280x720) with live scaling of boxes & labels.
    Live updates automatically when image or annotations change in the main window.
    """
    def __init__(self, controller: EditorController, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Annotation Preview - Full Image & Annotations")
        self.resize(1020, 780)
        self.setMinimumSize(540, 400)
        
        self.setWindowFlags(Qt.WindowType.Window)
        self.setStyleSheet("background-color: #18181C; color: #E2E8F0;")

        self.scene = QGraphicsScene(self)
        self._is_first_load = True
        self._auto_fit = True
        self._updating_dimensions = False
        self._target_w: Optional[int] = None
        self._target_h: Optional[int] = None

        self._setup_ui()
        self._connect_signals()
        self.refresh_preview(initial=True)

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # 1. Header bar with Image title and annotation counter
        header_layout = QHBoxLayout()
        self.lbl_title = QLabel("🖼️ Annotation Preview")
        self.lbl_title.setFont(QFont("Outfit", 11, QFont.Weight.Bold))
        self.lbl_title.setStyleSheet("color: #00A2E8;")
        
        self.lbl_info = QLabel("No image loaded")
        self.lbl_info.setFont(QFont("Outfit", 9))
        self.lbl_info.setStyleSheet("color: #A0A5B5;")

        header_layout.addWidget(self.lbl_title)
        header_layout.addStretch()
        header_layout.addWidget(self.lbl_info)
        main_layout.addLayout(header_layout)

        # 2. Image Resize & Dimension Controls Bar
        resize_group = QGroupBox("Resize Preview Image Dimensions")
        resize_group.setFont(QFont("Outfit", 9, QFont.Weight.Bold))
        resize_group.setStyleSheet("""
            QGroupBox {
                background-color: #1E1E24;
                border: 1px solid #2A2A2E;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 10px;
                padding-bottom: 6px;
                padding-left: 8px;
                padding-right: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 4px;
                color: #00A2E8;
            }
        """)
        resize_layout = QHBoxLayout(resize_group)
        resize_layout.setContentsMargins(8, 8, 8, 8)
        resize_layout.setSpacing(8)

        lbl_preset = QLabel("Preset:")
        lbl_preset.setFont(QFont("Outfit", 9))
        resize_layout.addWidget(lbl_preset)

        self.combo_presets = QComboBox()
        self.combo_presets.setStyleSheet("""
            QComboBox {
                background-color: #2F3542;
                color: white;
                border: 1px solid #3F4452;
                border-radius: 4px;
                padding: 4px 10px;
                font-weight: bold;
                font-size: 9pt;
                min-width: 140px;
            }
            QComboBox QAbstractItemView {
                background-color: #1E1E24;
                color: white;
                selection-background-color: #00A2E8;
            }
        """)
        self.combo_presets.addItems([
            "Original Size",
            "Square: 720 x 720",
            "Square: 640 x 640",
            "Portrait: 560 x 720",
            "HD: 1280 x 720",
            "FHD: 1920 x 1080",
            "Custom Size"
        ])
        self.combo_presets.currentIndexChanged.connect(self._on_preset_changed)
        resize_layout.addWidget(self.combo_presets)

        lbl_w = QLabel("Width:")
        lbl_w.setFont(QFont("Outfit", 9))
        resize_layout.addWidget(lbl_w)

        spin_style = """
            QSpinBox {
                background-color: #2F3542;
                color: white;
                border: 1px solid #3F4452;
                border-radius: 4px;
                padding: 3px 6px;
                font-weight: bold;
                font-size: 9pt;
                min-width: 75px;
            }
        """

        self.spn_w = QSpinBox()
        self.spn_w.setRange(16, 99999)
        self.spn_w.setSuffix(" px")
        self.spn_w.setStyleSheet(spin_style)
        self.spn_w.valueChanged.connect(self._on_width_changed)
        resize_layout.addWidget(self.spn_w)

        lbl_h = QLabel("Height:")
        lbl_h.setFont(QFont("Outfit", 9))
        resize_layout.addWidget(lbl_h)

        self.spn_h = QSpinBox()
        self.spn_h.setRange(16, 99999)
        self.spn_h.setSuffix(" px")
        self.spn_h.setStyleSheet(spin_style)
        self.spn_h.valueChanged.connect(self._on_height_changed)
        resize_layout.addWidget(self.spn_h)

        self.chk_lock_aspect = QCheckBox("Lock Aspect Ratio")
        self.chk_lock_aspect.setFont(QFont("Outfit", 9))
        self.chk_lock_aspect.setChecked(False)
        self.chk_lock_aspect.setStyleSheet("color: #E2E8F0; margin-left: 4px;")
        resize_layout.addWidget(self.chk_lock_aspect)

        btn_style_small = """
            QPushButton {
                background-color: #2F3542;
                color: #FFFFFF;
                border: 1px solid #3F4452;
                padding: 4px 10px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 9pt;
            }
            QPushButton:hover {
                background-color: #3F4452;
                border-color: #00A2E8;
            }
            QPushButton:pressed {
                background-color: #1E1E24;
            }
        """

        self.btn_reset_size = QPushButton("↺ Original Size")
        self.btn_reset_size.setToolTip("Reset to Original Image Dimensions")
        self.btn_reset_size.setStyleSheet(btn_style_small)
        self.btn_reset_size.clicked.connect(self._on_reset_size_clicked)
        resize_layout.addWidget(self.btn_reset_size)

        resize_layout.addStretch()
        main_layout.addWidget(resize_group)

        # 3. Central graphics view
        self.view = AnnotationPreviewView(self)
        self.view.setScene(self.scene)
        self.view.zoom_changed_callback = self._on_zoom_changed
        main_layout.addWidget(self.view, 1)

        # 4. Bottom Control Toolbar
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)

        lbl_zoom = QLabel("Zoom:")
        lbl_zoom.setFont(QFont("Outfit", 9, QFont.Weight.Bold))
        lbl_zoom.setStyleSheet("color: #E2E8F0;")
        controls_layout.addWidget(lbl_zoom)

        btn_style = """
            QPushButton {
                background-color: #2F3542;
                color: #FFFFFF;
                border: 1px solid #3F4452;
                padding: 5px 12px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 9pt;
            }
            QPushButton:hover {
                background-color: #3F4452;
                border-color: #00A2E8;
            }
            QPushButton:pressed {
                background-color: #1E1E24;
            }
        """

        self.btn_zoom_out = QPushButton(" - ")
        self.btn_zoom_out.setToolTip("Zoom Out")
        self.btn_zoom_out.setStyleSheet(btn_style)
        self.btn_zoom_out.clicked.connect(self._on_zoom_out_clicked)
        controls_layout.addWidget(self.btn_zoom_out)

        self.combo_zoom = QComboBox()
        self.combo_zoom.setStyleSheet("""
            QComboBox {
                background-color: #2F3542;
                color: white;
                border: 1px solid #3F4452;
                border-radius: 4px;
                padding: 4px 10px;
                font-weight: bold;
                font-size: 9pt;
                min-width: 80px;
            }
            QComboBox QAbstractItemView {
                background-color: #1E1E24;
                color: white;
                selection-background-color: #00A2E8;
            }
        """)
        self.combo_zoom.addItems(["25%", "50%", "75%", "100%", "150%", "200%"])
        self.combo_zoom.setCurrentText("100%")
        self.combo_zoom.currentIndexChanged.connect(self._on_combo_zoom_changed)
        controls_layout.addWidget(self.combo_zoom)

        self.btn_zoom_in = QPushButton(" + ")
        self.btn_zoom_in.setToolTip("Zoom In")
        self.btn_zoom_in.setStyleSheet(btn_style)
        self.btn_zoom_in.clicked.connect(self._on_zoom_in_clicked)
        controls_layout.addWidget(self.btn_zoom_in)

        self.btn_fit = QPushButton("Fit Image")
        self.btn_fit.setToolTip("Fit entire image into preview window")
        self.btn_fit.setStyleSheet(btn_style)
        self.btn_fit.clicked.connect(self._on_fit_clicked)
        controls_layout.addWidget(self.btn_fit)

        self.btn_100 = QPushButton("100%")
        self.btn_100.setToolTip("Reset Zoom to 100% Original Size")
        self.btn_100.setStyleSheet(btn_style)
        self.btn_100.clicked.connect(self._on_100_clicked)
        controls_layout.addWidget(self.btn_100)

        controls_layout.addStretch()

        self.lbl_zoom_status = QLabel("Zoom: 100%")
        self.lbl_zoom_status.setFont(QFont("Outfit", 9))
        self.lbl_zoom_status.setStyleSheet("color: #00A2E8; font-weight: bold;")
        controls_layout.addWidget(self.lbl_zoom_status)

        main_layout.addLayout(controls_layout)

    def _connect_signals(self):
        self.controller.imageLoaded.connect(self._on_controller_image_loaded)
        self.controller.stateChanged.connect(self._on_controller_state_changed)

    def _on_controller_image_loaded(self, path: str):
        if self.combo_presets.currentIndex() == 0:  # Original Size
            self._target_w = None
            self._target_h = None
        self._auto_fit = True
        self.refresh_preview()

    def _on_controller_state_changed(self):
        self.refresh_preview(keep_view=True)

    def _on_preset_changed(self, idx: int):
        if self._updating_dimensions: return
        pixmap = self._get_original_pixmap()
        orig_w = pixmap.width() if pixmap else 1280
        orig_h = pixmap.height() if pixmap else 720

        if idx == 0:  # Original Size
            self._set_dimensions(orig_w, orig_h)
            self._target_w = None
            self._target_h = None
        elif idx == 1:  # Square: 720 x 720
            self._set_dimensions(720, 720)
        elif idx == 2:  # Square: 640 x 640
            self._set_dimensions(640, 640)
        elif idx == 3:  # Portrait: 560 x 720
            self._set_dimensions(560, 720)
        elif idx == 4:  # HD: 1280 x 720
            self._set_dimensions(1280, 720)
        elif idx == 5:  # FHD: 1920 x 1080
            self._set_dimensions(1920, 1080)
        
        self.refresh_preview()

    def _set_dimensions(self, w: int, h: int):
        self._updating_dimensions = True
        self._target_w = w
        self._target_h = h
        self.spn_w.setValue(w)
        self.spn_h.setValue(h)
        self._updating_dimensions = False

    def _on_width_changed(self, new_w: int):
        if self._updating_dimensions: return
        pixmap = self._get_original_pixmap()
        if pixmap and not pixmap.isNull() and self.chk_lock_aspect.isChecked() and pixmap.width() > 0:
            ratio = pixmap.height() / pixmap.width()
            new_h = max(16, int(round(new_w * ratio)))
            self._updating_dimensions = True
            self.spn_h.setValue(new_h)
            self._updating_dimensions = False

        self._target_w = self.spn_w.value()
        self._target_h = self.spn_h.value()
        self._sync_preset_combobox()
        self.refresh_preview(keep_view=True)

    def _on_height_changed(self, new_h: int):
        if self._updating_dimensions: return
        pixmap = self._get_original_pixmap()
        if pixmap and not pixmap.isNull() and self.chk_lock_aspect.isChecked() and pixmap.height() > 0:
            ratio = pixmap.width() / pixmap.height()
            new_w = max(16, int(round(new_h * ratio)))
            self._updating_dimensions = True
            self.spn_w.setValue(new_w)
            self._updating_dimensions = False

        self._target_w = self.spn_w.value()
        self._target_h = self.spn_h.value()
        self._sync_preset_combobox()
        self.refresh_preview(keep_view=True)

    def _sync_preset_combobox(self):
        w = self.spn_w.value()
        h = self.spn_h.value()
        pixmap = self._get_original_pixmap()
        orig_w = pixmap.width() if pixmap else 0
        orig_h = pixmap.height() if pixmap else 0

        self.combo_presets.blockSignals(True)
        if (w, h) == (orig_w, orig_h) and orig_w > 0:
            self.combo_presets.setCurrentIndex(0)
        elif (w, h) == (720, 720):
            self.combo_presets.setCurrentIndex(1)
        elif (w, h) == (640, 640):
            self.combo_presets.setCurrentIndex(2)
        elif (w, h) == (560, 720):
            self.combo_presets.setCurrentIndex(3)
        elif (w, h) == (1280, 720):
            self.combo_presets.setCurrentIndex(4)
        elif (w, h) == (1920, 1080):
            self.combo_presets.setCurrentIndex(5)
        else:
            self.combo_presets.setCurrentIndex(6)
        self.combo_presets.blockSignals(False)

    def _on_reset_size_clicked(self):
        pixmap = self._get_original_pixmap()
        if pixmap and not pixmap.isNull():
            self._set_dimensions(pixmap.width(), pixmap.height())
            self._target_w = None
            self._target_h = None
            self.combo_presets.blockSignals(True)
            self.combo_presets.setCurrentIndex(0)
            self.combo_presets.blockSignals(False)
            self.refresh_preview()

    def _get_original_pixmap(self) -> Optional[QPixmap]:
        pixmap = self.controller.current_pixmap
        img_path = self.controller.image_path
        if not pixmap and img_path and os.path.exists(img_path):
            pixmap = QPixmap(img_path)
        return pixmap

    def _on_zoom_in_clicked(self):
        self._auto_fit = False
        self.view.zoom_in()

    def _on_zoom_out_clicked(self):
        self._auto_fit = False
        self.view.zoom_out()

    def _on_fit_clicked(self):
        self._auto_fit = True
        self.view.fit_image()

    def _on_100_clicked(self):
        self._auto_fit = False
        self.view.set_zoom_level(1.0)

    def _on_combo_zoom_changed(self, idx: int):
        text = self.combo_zoom.currentText().replace("%", "").strip()
        try:
            val = float(text) / 100.0
            self._auto_fit = False
            self.view.set_zoom_level(val)
        except ValueError:
            pass

    def _on_zoom_changed(self, scale: float):
        pct = int(round(scale * 100))
        self.lbl_zoom_status.setText(f"Zoom: {pct}%")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._auto_fit:
            self.view.fit_image()

    def showEvent(self, event):
        super().showEvent(event)
        if self._is_first_load:
            self._is_first_load = False
            self.view.fit_image()

    def refresh_preview(self, keep_view: bool = False, initial: bool = False):
        pixmap = self._get_original_pixmap()
        img_path = self.controller.image_path

        if not pixmap or pixmap.isNull():
            self.scene.clear()
            self.lbl_info.setText("No image loaded")
            return

        orig_w, orig_h = pixmap.width(), pixmap.height()

        if (self._target_w is None or self._target_h is None) and not self._updating_dimensions:
            self._set_dimensions(orig_w, orig_h)
            target_w, target_h = orig_w, orig_h
        else:
            target_w = self._target_w or orig_w
            target_h = self._target_h or orig_h

        # Compute aspect-ratio preserving scale factor and letterbox padding offsets
        scale = min(target_w / float(orig_w), target_h / float(orig_h)) if orig_w > 0 and orig_h > 0 else 1.0
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        offset_x = (target_w - new_w) / 2.0
        offset_y = (target_h - new_h) / 2.0

        if (new_w, new_h) == (orig_w, orig_h):
            display_pixmap = pixmap
        else:
            display_pixmap = pixmap.scaled(
                new_w, new_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

        prev_transform = self.view.transform() if keep_view else None

        self.scene.clear()

        # 1. Add background canvas with padding border (target_w x target_h)
        canvas_bg = QGraphicsRectItem(0, 0, target_w, target_h)
        canvas_bg.setPen(QPen(QColor("#2A2A2E"), 1.0))
        canvas_bg.setBrush(QBrush(QColor("#0F0F12")))
        canvas_bg.setZValue(-2)
        self.scene.addItem(canvas_bg)

        # 2. Add aspect-ratio preserved centered image at offset (offset_x, offset_y)
        pix_item = QGraphicsPixmapItem(display_pixmap)
        pix_item.setPos(offset_x, offset_y)
        pix_item.setZValue(-1)
        self.scene.addItem(pix_item)
        self.scene.setSceneRect(0, 0, target_w, target_h)

        box_pen = QPen(QColor(0, 162, 232), 2.5)
        box_brush = QBrush(QColor(0, 162, 232, 40))

        anno_count = len(self.controller.annotations)

        # 3. Render all scaled & padded bounding boxes / polygons & label badges
        for item in self.controller.annotations:
            if item.box.points and len(item.box.points) >= 3:
                scaled_pts = [QPointF(offset_x + p[0] * scale, offset_y + p[1] * scale) for p in item.box.points]
                poly = QPolygonF(scaled_pts)
                poly_item = QGraphicsPolygonItem(poly)
                poly_item.setPen(box_pen)
                poly_item.setBrush(box_brush)
                poly_item.setZValue(1)
                self.scene.addItem(poly_item)
                min_x = min(p.x() for p in scaled_pts)
                min_y = min(p.y() for p in scaled_pts)
            else:
                rx = offset_x + item.x * scale
                ry = offset_y + item.y * scale
                rw = item.width * scale
                rh = item.height * scale
                rect_item = QGraphicsRectItem(QRectF(rx, ry, rw, rh))
                rect_item.setPen(box_pen)
                rect_item.setBrush(box_brush)
                rect_item.setZValue(1)
                self.scene.addItem(rect_item)
                min_x = rx
                min_y = ry

            label_text = item.label.strip() if item.label and item.label.strip() else "Unknown"
            text_item = QGraphicsTextItem(label_text)
            text_item.setFont(QFont("Outfit", 10, QFont.Weight.Bold))
            text_item.setDefaultTextColor(QColor("#FFFFFF"))

            text_rect = text_item.boundingRect()
            pad_x, pad_y = 6, 2
            badge_rect = QRectF(0, 0, text_rect.width() + pad_x * 2, text_rect.height() + pad_y * 2)
            bg_rect = QGraphicsRectItem(badge_rect)
            bg_rect.setPen(QPen(QColor(0, 162, 232), 1.0))
            bg_rect.setBrush(QBrush(QColor(18, 18, 20, 220)))

            badge_x = min_x
            badge_y = max(0.0, min_y - badge_rect.height() - 2)
            bg_rect.setPos(badge_x, badge_y)
            text_item.setPos(badge_x + pad_x, badge_y + pad_y)

            bg_rect.setZValue(10)
            text_item.setZValue(11)

            self.scene.addItem(bg_rect)
            self.scene.addItem(text_item)

        base_name = os.path.basename(img_path) if img_path else "Current Image"
        size_str = f"({orig_w}x{orig_h})" if (target_w, target_h) == (orig_w, orig_h) else f"({orig_w}x{orig_h} ➜ {target_w}x{target_h} Padded)"
        self.lbl_info.setText(f"{base_name} {size_str} | {anno_count} Annotation{'s' if anno_count != 1 else ''}")

        if keep_view and prev_transform:
            self.view.setTransform(prev_transform)
        elif self._auto_fit or initial:
            self.view.fit_image()

    def closeEvent(self, event):
        try:
            self.controller.imageLoaded.disconnect(self._on_controller_image_loaded)
            self.controller.stateChanged.disconnect(self._on_controller_state_changed)
        except Exception:
            pass
        super().closeEvent(event)


# ==============================================================================
# 8. MAIN WINDOW INTEGRATION
# ==============================================================================

class MainWindow(QMainWindow):
    def __init__(self, controller: EditorController):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("PlateAnnotate - License Plate Annotation Editor")
        self.resize(1200, 800)
        self.settings = QSettings("Antigravity", "PlateAnnotate")

        self.viewer = None
        self.properties_panel = None
        self.file_list_widget = None
        self._preview_dialog = None

        self._setup_ui()
        self._setup_menus_and_shortcuts()
        self._connect_signals()
        self._apply_theme()
        
        self._update_undo_redo_actions(False, False)
        self.setAcceptDrops(True)
        self._update_recent_files_menu()

    def _get_tool_btn_stylesheet(self) -> str:
        return """
            QPushButton {
                background-color: #2F3542;
                color: #E2E8F0;
                border: 1px solid #3F4452;
                padding: 10px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #3F4452;
                color: white;
            }
            QPushButton:checked {
                background-color: #00A2E8;
                border-color: #00A2E8;
                color: white;
            }
        """

    def _get_action_btn_stylesheet(self, bg_color: str) -> str:
        return f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                border: none;
                padding: 10px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 10pt;
            }}
            QPushButton:hover {{
                background-color: {bg_color}DD;
            }}
            QPushButton:pressed {{
                background-color: {bg_color}AA;
            }}
            QPushButton:disabled {{
                background-color: #1E1E24;
                color: #555558;
            }}
        """

    def _setup_ui(self) -> None:
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left sidebar navigator
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(6, 6, 6, 6)
        sidebar_layout.setSpacing(6)
        
        sidebar_title = QLabel("IMAGE FOLDER")
        sidebar_title.setFont(QFont("Outfit", 12, QFont.Weight.Bold))
        sidebar_title.setStyleSheet("color: #00A2E8; padding: 2px;")
        sidebar_layout.addWidget(sidebar_title)

        # Filter Dropdown
        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(4)
        lbl_filter = QLabel("Filter:")
        lbl_filter.setFont(QFont("Outfit", 9, QFont.Weight.Bold))
        lbl_filter.setStyleSheet("color: #A0A5B5;")
        
        self.combo_filter = QComboBox()
        self.combo_filter.setFont(QFont("Outfit", 9))
        self.combo_filter.addItems(["📁 All Images", "🔴 Unfinished Only", "🟢 Finished Only", "❓ Doubt Only"])
        self.combo_filter.setStyleSheet("""
            QComboBox {
                background-color: #1E1E24;
                color: #E2E8F0;
                border: 1px solid #2A2A2E;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #1E1E24;
                color: #E2E8F0;
                selection-background-color: #00A2E8;
            }
        """)
        self.combo_filter.currentIndexChanged.connect(self._apply_file_filter)
        filter_layout.addWidget(lbl_filter)
        filter_layout.addWidget(self.combo_filter, 1)
        sidebar_layout.addLayout(filter_layout)

        self.file_list_widget = QListWidget()
        self.file_list_widget.setFont(QFont("Outfit", 9))
        self.file_list_widget.setStyleSheet("""
            QListWidget {
                background-color: #121214;
                border: 1px solid #2A2A2E;
                border-radius: 4px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 6px;
                border-radius: 4px;
                margin-bottom: 2px;
            }
            QListWidget::item:selected {
                background-color: #00A2E8;
                color: white;
            }
            QListWidget::item:hover:!selected {
                background-color: #1E1E24;
            }
        """)
        self.file_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_list_widget.customContextMenuRequested.connect(self._show_file_list_context_menu)
        sidebar_layout.addWidget(self.file_list_widget)

        # Doubt status counter label
        self.lbl_verification_counter = QLabel("Finished: 0 / Total: 0")
        self.lbl_verification_counter.setFont(QFont("Outfit", 9, QFont.Weight.Bold))
        self.lbl_verification_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_verification_counter.setStyleSheet("""
            QLabel {
                background-color: #18181C;
                color: #2ED573;
                border: 1px solid #2A2A2E;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        sidebar_layout.addWidget(self.lbl_verification_counter)

        # Navigation Buttons (Previous / Next) inside left panel
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(6)
        
        self.btn_prev = QPushButton("◀ Prev")
        self.btn_prev.setStyleSheet("""
            QPushButton { background-color: #2F3542; color: white; padding: 10px; font-weight: bold; border-radius: 4px; font-size: 10pt; }
            QPushButton:hover { background-color: #3F4452; }
            QPushButton:pressed { background-color: #1E1E24; }
        """)
        self.btn_prev.clicked.connect(self._on_action_prev)
        
        self.btn_next = QPushButton("Next ▶")
        self.btn_next.setStyleSheet("""
            QPushButton { background-color: #2F3542; color: white; padding: 10px; font-weight: bold; border-radius: 4px; font-size: 10pt; }
            QPushButton:hover { background-color: #3F4452; }
            QPushButton:pressed { background-color: #1E1E24; }
        """)
        self.btn_next.clicked.connect(self._on_action_next)
        
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        sidebar_layout.addLayout(nav_layout)

        main_splitter.addWidget(sidebar_widget)

        # Center Container Setup
        center_container = QWidget()
        center_layout = QVBoxLayout(center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(5)

        self.viewer = ImageViewer()
        center_layout.addWidget(self.viewer, 1)

        # Crop Preview Group at bottom of center area
        self.crop_group = QGroupBox("Crop Preview")
        self.crop_group.setFont(QFont("Outfit", 10))
        self.crop_group.setMaximumHeight(230)
        crop_layout = QHBoxLayout(self.crop_group)
        crop_layout.setContentsMargins(10, 5, 10, 5)
        crop_layout.setSpacing(15)

        self.lbl_crop = ClickableLabel("Select a box to preview crop")
        self.lbl_crop.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_crop.setFixedSize(300, 180)
        self.lbl_crop.setStyleSheet("background-color: #121214; border: 1px dashed #2A2A2E; border-radius: 4px; color: #747D8C;")
        crop_layout.addWidget(self.lbl_crop)

        crop_btn_layout = QVBoxLayout()
        crop_btn_layout.setSpacing(10)

        self.btn_save_crop = QPushButton("Save Crop Image")
        self.btn_save_crop.setStyleSheet("""
            QPushButton { background-color: #00A2E8; color: white; padding: 12px; font-size: 11pt; font-weight: bold; border-radius: 4px; }
            QPushButton:disabled { background-color: #2F3542; color: #747D8C; }
        """)
        self.btn_save_crop.setEnabled(False)
        self.btn_save_crop.clicked.connect(self._on_btn_save_crop_clicked)
        crop_btn_layout.addWidget(self.btn_save_crop)

        crop_info_lbl = QLabel("Automatically exports the selected crop to the 'crops/' subfolder.")
        crop_info_lbl.setStyleSheet("color: #747D8C; font-size: 9pt;")
        crop_info_lbl.setWordWrap(True)
        crop_btn_layout.addWidget(crop_info_lbl)
        crop_btn_layout.addStretch()

        crop_layout.addLayout(crop_btn_layout, 1)
        center_layout.addWidget(self.crop_group)

        main_splitter.addWidget(center_container)

        # Right Sidebar Container (with Editor Tools and Properties panel)
        right_sidebar = QWidget()
        right_layout = QVBoxLayout(right_sidebar)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(10)

        # Editor Tools Group
        tools_group = QGroupBox("Editor Tools")
        tools_group.setFont(QFont("Outfit", 10))
        tools_layout = QVBoxLayout(tools_group)
        tools_layout.setSpacing(8)

        # Mode Buttons (Horizontal Layout)
        modes_layout = QHBoxLayout()
        modes_layout.setSpacing(6)

        self.btn_tool_select = QPushButton("Select")
        self.btn_tool_select.setCheckable(True)
        self.btn_tool_select.setChecked(True)
        self.btn_tool_select.setStyleSheet(self._get_tool_btn_stylesheet())
        self.btn_tool_select.clicked.connect(lambda: self._set_editor_mode("select"))

        self.btn_tool_create = QPushButton("Draw Box")
        self.btn_tool_create.setCheckable(True)
        self.btn_tool_create.setStyleSheet(self._get_tool_btn_stylesheet())
        self.btn_tool_create.clicked.connect(lambda: self._set_editor_mode("create"))

        self.btn_tool_poly = QPushButton("Polygon Draw")
        self.btn_tool_poly.setCheckable(True)
        self.btn_tool_poly.setStyleSheet(self._get_tool_btn_stylesheet())
        self.btn_tool_poly.clicked.connect(lambda: self._set_editor_mode("polygon"))

        modes_layout.addWidget(self.btn_tool_select)
        modes_layout.addWidget(self.btn_tool_create)
        modes_layout.addWidget(self.btn_tool_poly)
        tools_layout.addLayout(modes_layout)

        # Button Group for automatic mutual exclusion
        self.tool_group = QButtonGroup(self)
        self.tool_group.addButton(self.btn_tool_select)
        self.tool_group.addButton(self.btn_tool_create)
        self.tool_group.addButton(self.btn_tool_poly)

        # Action Buttons Grid
        actions_grid = QGridLayout()
        actions_grid.setSpacing(6)

        self.btn_undo = QPushButton("Undo")
        self.btn_undo.setStyleSheet(self._get_action_btn_stylesheet("#2F3542"))
        self.btn_undo.clicked.connect(self.controller.undo)
        
        self.btn_redo = QPushButton("Redo")
        self.btn_redo.setStyleSheet(self._get_action_btn_stylesheet("#2F3542"))
        self.btn_redo.clicked.connect(self.controller.redo)
        
        self.btn_save = QPushButton("Save")
        self.btn_save.setStyleSheet(self._get_action_btn_stylesheet("#2ED573"))
        self.btn_save.clicked.connect(self._on_action_save)

        self.btn_finish = QPushButton("✅ Finish (Shift+F)")
        self.btn_finish.setToolTip("Mark image as Reviewed/Finished (GREEN status)")
        self.btn_finish.setStyleSheet(self._get_action_btn_stylesheet("#2ED573"))
        self.btn_finish.clicked.connect(self._on_action_finish)
        
        self.btn_fit = QPushButton("Fit Image")
        self.btn_fit.setStyleSheet(self._get_action_btn_stylesheet("#2F3542"))
        self.btn_fit.clicked.connect(self.viewer.fit_image)

        self.btn_export_finished = QPushButton("📤 Export Finished")
        self.btn_export_finished.setToolTip("Export ONLY labels of GREEN / Finished images")
        self.btn_export_finished.setStyleSheet(self._get_action_btn_stylesheet("#00A2E8"))
        self.btn_export_finished.clicked.connect(self._on_action_export_finished)
        
        self.btn_preview = QPushButton("🖼️ Annotation Preview")
        self.btn_preview.setToolTip("Full Image + Annotation Preview Popup (Ctrl+Shift+P)")
        self.btn_preview.setStyleSheet(self._get_action_btn_stylesheet("#00A2E8"))
        self.btn_preview.clicked.connect(self.show_annotation_preview)

        self.btn_delete_file = QPushButton("🗑️ Delete Image File")
        self.btn_delete_file.setStyleSheet(self._get_action_btn_stylesheet("#FF4757"))
        self.btn_delete_file.clicked.connect(lambda: self._on_action_delete_files_from_disk())

        self.btn_copy_prev = QPushButton("📋 Copy Prev Bboxes (E)")
        self.btn_copy_prev.setToolTip("Copy bounding boxes from previous image/frame (Shortcut: E or [)")
        self.btn_copy_prev.setStyleSheet(self._get_action_btn_stylesheet("#3A3F51"))
        self.btn_copy_prev.clicked.connect(self._on_action_copy_prev_bboxes)

        self.btn_copy_next = QPushButton("📋 Copy Next Bboxes (N)")
        self.btn_copy_next.setToolTip("Copy bounding boxes from next image/frame (Shortcut: N or ])")
        self.btn_copy_next.setStyleSheet(self._get_action_btn_stylesheet("#3A3F51"))
        self.btn_copy_next.clicked.connect(self._on_action_copy_next_bboxes)

        self.btn_zoom_plate = QPushButton("🔍 Zoom to Plate (Z)")
        self.btn_zoom_plate.setToolTip("Perfect zoom centered on license plate / bbox (Shortcut: Z)")
        self.btn_zoom_plate.setStyleSheet(self._get_action_btn_stylesheet("#00A2E8"))
        self.btn_zoom_plate.clicked.connect(self.zoom_to_license_plate)

        actions_grid.addWidget(self.btn_undo, 0, 0)
        actions_grid.addWidget(self.btn_redo, 0, 1)
        actions_grid.addWidget(self.btn_save, 1, 0)
        actions_grid.addWidget(self.btn_finish, 1, 1)
        actions_grid.addWidget(self.btn_copy_prev, 2, 0)
        actions_grid.addWidget(self.btn_copy_next, 2, 1)
        actions_grid.addWidget(self.btn_fit, 3, 0)
        actions_grid.addWidget(self.btn_zoom_plate, 3, 1)
        actions_grid.addWidget(self.btn_export_finished, 4, 0, 1, 2)
        actions_grid.addWidget(self.btn_preview, 5, 0, 1, 2)
        actions_grid.addWidget(self.btn_delete_file, 6, 0, 1, 2)
        tools_layout.addLayout(actions_grid)

        # Fit & Zoom Mode Options
        fit_options_layout = QHBoxLayout()
        fit_options_layout.setSpacing(10)

        self.chk_always_fit = QCheckBox("Always Fit Image")
        self.chk_always_fit.setFont(QFont("Outfit", 9, QFont.Weight.Bold))
        self.chk_always_fit.setStyleSheet("color: #E2E8F0; margin-top: 4px;")
        is_always_fit = self.settings.value("alwaysFitImage", True)
        if isinstance(is_always_fit, str):
            is_always_fit = is_always_fit.lower() == "true"
        elif not isinstance(is_always_fit, bool):
            is_always_fit = bool(is_always_fit)
        self.chk_always_fit.setChecked(is_always_fit)
        self.chk_always_fit.toggled.connect(self._on_toggle_always_fit)

        self.chk_always_zoom_plate = QCheckBox("Auto-Zoom to Plate")
        self.chk_always_zoom_plate.setFont(QFont("Outfit", 9, QFont.Weight.Bold))
        self.chk_always_zoom_plate.setStyleSheet("color: #00A2E8; margin-top: 4px;")
        is_always_zoom = self.settings.value("alwaysZoomPlate", False)
        if isinstance(is_always_zoom, str):
            is_always_zoom = is_always_zoom.lower() == "true"
        elif not isinstance(is_always_zoom, bool):
            is_always_zoom = bool(is_always_zoom)
        self.chk_always_zoom_plate.setChecked(is_always_zoom)
        self.chk_always_zoom_plate.toggled.connect(self._on_toggle_always_zoom_plate)

        fit_options_layout.addWidget(self.chk_always_fit)
        fit_options_layout.addWidget(self.chk_always_zoom_plate)
        tools_layout.addLayout(fit_options_layout)

        right_layout.addWidget(tools_group)

        self.properties_panel = PropertiesPanel()
        right_layout.addWidget(self.properties_panel)
        right_layout.addStretch()

        main_splitter.addWidget(right_sidebar)

        # Set stretch factors and initial sizes to keep center widget big
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 5)
        main_splitter.setStretchFactor(2, 2)
        main_splitter.setSizes([200, 800, 300])
        self.setCentralWidget(main_splitter)

        # Toolbar
        self.toolbar = QToolBar("Main Toolbar")
        self.toolbar.setIconSize(QSize(20, 20))
        self.toolbar.setMovable(False)
        self.addToolBar(self.toolbar)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_image_path = QLabel("No image loaded")
        self.status_zoom = QLabel("Zoom: 100%")
        self.status_mode = QLabel("Mode: Select")
        self.status_bar.addWidget(self.status_image_path, 3)
        self.status_bar.addPermanentWidget(self.status_mode, 1)
        self.status_bar.addPermanentWidget(self.status_zoom, 1)

    def _setup_menus_and_shortcuts(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        self.act_open_image = QAction("&Open Image...", self)
        self.act_open_image.setShortcut(QKeySequence("Ctrl+O"))
        self.act_open_image.triggered.connect(self._on_action_open_image)
        file_menu.addAction(self.act_open_image)

        self.act_load_anno = QAction("&Load Annotation...", self)
        self.act_load_anno.setShortcut(QKeySequence("Ctrl+L"))
        self.act_load_anno.triggered.connect(self._on_action_load_annotation)
        file_menu.addAction(self.act_load_anno)

        self.act_open_folder = QAction("Open &Folder...", self)
        self.act_open_folder.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self.act_open_folder.triggered.connect(self._on_action_open_folder)
        file_menu.addAction(self.act_open_folder)

        self.recent_menu = file_menu.addMenu("Open Recent")
        file_menu.addSeparator()

        self.act_save = QAction("&Save", self)
        self.act_save.setShortcut(QKeySequence("Ctrl+S"))
        self.act_save.triggered.connect(self._on_action_save)
        file_menu.addAction(self.act_save)

        self.act_save_as = QAction("Save &As...", self)
        self.act_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.act_save_as.triggered.connect(self._on_action_save_as)
        file_menu.addAction(self.act_save_as)

        self.act_export_json = QAction("&Export to JSON...", self)
        self.act_export_json.triggered.connect(self._on_action_export_json)
        file_menu.addAction(self.act_export_json)
        file_menu.addSeparator()

        self.act_reload = QAction("&Reload Annotation", self)
        self.act_reload.triggered.connect(self._on_action_reload)
        file_menu.addAction(self.act_reload)
        file_menu.addSeparator()

        self.act_autosave = QAction("&Auto-Save on Navigate", self)
        self.act_autosave.setCheckable(True)
        is_autosave = self.settings.value("autoSave", False)
        if isinstance(is_autosave, str):
            is_autosave = is_autosave.lower() == "true"
        elif not isinstance(is_autosave, bool):
            is_autosave = bool(is_autosave)
        self.act_autosave.setChecked(is_autosave)
        self.act_autosave.triggered.connect(self._on_toggle_autosave)
        file_menu.addAction(self.act_autosave)
        file_menu.addSeparator()

        self.act_exit = QAction("E&xit", self)
        self.act_exit.triggered.connect(self.close)
        file_menu.addAction(self.act_exit)

        # Edit Menu
        edit_menu = menubar.addMenu("&Edit")
        self.act_undo = QAction("&Undo", self)
        self.act_undo.setShortcut(QKeySequence("Ctrl+Z"))
        self.act_undo.triggered.connect(self.controller.undo)
        edit_menu.addAction(self.act_undo)

        self.act_redo = QAction("&Redo", self)
        self.act_redo.setShortcut(QKeySequence("Ctrl+Y"))
        self.act_redo.triggered.connect(self.controller.redo)
        edit_menu.addAction(self.act_redo)
        edit_menu.addSeparator()

        self.act_copy = QAction("&Copy Box", self)
        self.act_copy.setShortcut(QKeySequence("Ctrl+C"))
        self.act_copy.triggered.connect(self._on_action_copy)
        edit_menu.addAction(self.act_copy)

        self.act_paste = QAction("&Paste Box", self)
        self.act_paste.setShortcut(QKeySequence("Ctrl+V"))
        self.act_paste.triggered.connect(self._on_action_paste)
        edit_menu.addAction(self.act_paste)

        self.act_duplicate = QAction("D&uplicate Box", self)
        self.act_duplicate.setShortcut(QKeySequence("Ctrl+D"))
        self.act_duplicate.triggered.connect(self._on_action_duplicate)
        edit_menu.addAction(self.act_duplicate)

        self.act_copy_prev_bboxes = QAction("📋 Copy Bboxes from &Previous Image", self)
        self.act_copy_prev_bboxes.setShortcuts([QKeySequence("E"), QKeySequence("["), QKeySequence("Ctrl+Shift+C")])
        self.act_copy_prev_bboxes.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self.act_copy_prev_bboxes.triggered.connect(self._on_action_copy_prev_bboxes)
        self.addAction(self.act_copy_prev_bboxes)
        edit_menu.addAction(self.act_copy_prev_bboxes)

        self.act_copy_next_bboxes = QAction("📋 Copy Bboxes from &Next Image", self)
        self.act_copy_next_bboxes.setShortcuts([QKeySequence("N"), QKeySequence("]"), QKeySequence("Ctrl+Shift+V")])
        self.act_copy_next_bboxes.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self.act_copy_next_bboxes.triggered.connect(self._on_action_copy_next_bboxes)
        self.addAction(self.act_copy_next_bboxes)
        edit_menu.addAction(self.act_copy_next_bboxes)

        self.act_delete = QAction("&Delete Selected", self)
        self.act_delete.setShortcuts([QKeySequence("W"), QKeySequence("Delete"), QKeySequence("Backspace")])
        self.act_delete.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self.act_delete.triggered.connect(self._on_action_delete)
        self.addAction(self.act_delete)
        edit_menu.addAction(self.act_delete)
        edit_menu.addSeparator()

        self.act_toggle_verify = QAction("Mark/Unmark &Doubt", self)
        self.act_toggle_verify.setShortcut(QKeySequence("V"))
        self.act_toggle_verify.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self.act_toggle_verify.triggered.connect(self._on_action_toggle_verify)
        self.addAction(self.act_toggle_verify)
        edit_menu.addAction(self.act_toggle_verify)
        edit_menu.addSeparator()

        self.act_delete_file_disk = QAction("🗑️ &Delete Image & Label Files from Disk", self)
        self.act_delete_file_disk.setShortcuts([QKeySequence("Shift+Delete"), QKeySequence("Shift+S")])
        self.act_delete_file_disk.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self.act_delete_file_disk.triggered.connect(lambda: self._on_action_delete_files_from_disk())
        self.addAction(self.act_delete_file_disk)
        edit_menu.addAction(self.act_delete_file_disk)

        # View Menu
        view_menu = menubar.addMenu("&View")
        self.act_fit_image = QAction("&Fit Image", self)
        self.act_fit_image.setShortcut(QKeySequence("F"))
        self.act_fit_image.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self.act_fit_image.triggered.connect(self.viewer.fit_image)
        self.addAction(self.act_fit_image)
        view_menu.addAction(self.act_fit_image)

        self.act_always_fit = QAction("Always &Fit Image", self)
        self.act_always_fit.setCheckable(True)
        self.act_always_fit.setChecked(self.chk_always_fit.isChecked())
        self.act_always_fit.triggered.connect(self.chk_always_fit.setChecked)
        view_menu.addAction(self.act_always_fit)

        self.act_reset_zoom = QAction("&Reset Zoom", self)
        self.act_reset_zoom.setShortcut(QKeySequence("R"))
        self.act_reset_zoom.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self.act_reset_zoom.triggered.connect(self.viewer.reset_zoom)
        self.addAction(self.act_reset_zoom)
        view_menu.addAction(self.act_reset_zoom)

        self.act_zoom_in = QAction("Zoom In", self)
        self.act_zoom_in.setShortcuts([QKeySequence("Ctrl+="), QKeySequence("+")])
        self.act_zoom_in.triggered.connect(self._on_action_zoom_in)
        self.addAction(self.act_zoom_in)
        view_menu.addAction(self.act_zoom_in)

        self.act_zoom_out = QAction("Zoom Out", self)
        self.act_zoom_out.setShortcuts([QKeySequence("Ctrl+-"), QKeySequence("-")])
        self.act_zoom_out.triggered.connect(self._on_action_zoom_out)
        self.addAction(self.act_zoom_out)
        view_menu.addAction(self.act_zoom_out)
        view_menu.addSeparator()

        self.act_anno_preview = QAction("🖼️ &Annotation Preview", self)
        self.act_anno_preview.setShortcut(QKeySequence("Ctrl+Shift+P"))
        self.act_anno_preview.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self.act_anno_preview.triggered.connect(self.show_annotation_preview)
        self.addAction(self.act_anno_preview)
        view_menu.addAction(self.act_anno_preview)
        view_menu.addSeparator()

        self.act_prev = QAction("&Previous Image", self)
        self.act_prev.setShortcut(QKeySequence("A"))
        self.act_prev.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self.act_prev.triggered.connect(self._on_action_prev)
        self.addAction(self.act_prev)
        view_menu.addAction(self.act_prev)

        self.act_next = QAction("&Next Image", self)
        self.act_next.setShortcut(QKeySequence("D"))
        self.act_next.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self.act_next.triggered.connect(self._on_action_next)
        self.addAction(self.act_next)
        view_menu.addAction(self.act_next)

        # Left / Right Arrow navigation overrides
        act_arrow_prev = QAction(self)
        act_arrow_prev.setShortcut(QKeySequence(Qt.Key.Key_Left))
        act_arrow_prev.triggered.connect(self._on_action_prev)
        self.addAction(act_arrow_prev)

        act_arrow_next = QAction(self)
        act_arrow_next.setShortcut(QKeySequence(Qt.Key.Key_Right))
        act_arrow_next.triggered.connect(self._on_action_next)
        self.addAction(act_arrow_next)

        # Toolbar items
        self.toolbar.addAction(self.act_open_image)
        self.toolbar.addAction(self.act_open_folder)
        
        self.act_tb_select = QAction("Select", self)
        self.act_tb_select.setCheckable(True)
        self.act_tb_select.setChecked(True)
        self.act_tb_select.triggered.connect(lambda: self._set_editor_mode("select"))

        self.act_tb_create = QAction("Draw Box", self)
        self.act_tb_create.setCheckable(True)
        self.act_tb_create.triggered.connect(lambda: self._set_editor_mode("create"))

        self.act_tb_poly = QAction("Polygon Draw", self)
        self.act_tb_poly.setCheckable(True)
        self.act_tb_poly.setShortcut(QKeySequence("Q"))
        self.act_tb_poly.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self.act_tb_poly.triggered.connect(lambda: self._set_editor_mode("polygon"))
        self.addAction(self.act_tb_poly)

        self.act_tb_pan = QAction("Pan / Move", self)
        self.act_tb_pan.setCheckable(True)
        self.act_tb_pan.triggered.connect(lambda: self._set_editor_mode("pan"))

        # Help
        help_menu = menubar.addMenu("&Help")
        act_about = QAction("&About PlateAnnotate", self)
        act_about.triggered.connect(self._on_action_about)
        help_menu.addAction(act_about)

    def _connect_signals(self) -> None:
        self.controller.imageLoaded.connect(self._on_controller_image_loaded)
        self.controller.stateChanged.connect(self._on_controller_state_changed)
        self.controller.statusChanged.connect(self._on_controller_status_changed)
        self.controller.undoRedoStatus.connect(self._update_undo_redo_actions)
        self.controller.dirtyStateChanged.connect(self._on_dirty_state_changed)

        self.viewer.bboxCreated.connect(self.controller.add_annotation)
        self.viewer.selectionChanged.connect(self._on_viewer_selection_changed)
        self.viewer.zoomChanged.connect(self._on_viewer_zoom_changed)

        self.properties_panel.propertyChanged.connect(self._on_property_edited)
        self.properties_panel.lockToggled.connect(lambda locked: self._on_property_edited("locked", locked))
        self.properties_panel.btn_duplicate.clicked.connect(self._on_action_duplicate)
        self.properties_panel.btn_delete.clicked.connect(self._on_action_delete)

        self.file_list_widget.itemSelectionChanged.connect(self._on_sidebar_selection_changed)
        if hasattr(self, "lbl_crop") and self.lbl_crop:
            self.lbl_crop.clicked.connect(self._on_crop_preview_clicked)

    def _on_controller_status_changed(self) -> None:
        current_path = self.controller.image_path
        if current_path and current_path in self.controller.image_list:
            idx = self.controller.image_list.index(current_path)
            item = self.file_list_widget.item(idx)
            if item:
                self._update_item_style(item, idx, current_path)
        self._apply_file_filter()

    def _apply_theme(self) -> None:
        self.setFont(QFont("Outfit", 9))
        self.setStyleSheet("""
            QMainWindow { background-color: #1A1A1E; }
            QToolBar { background-color: #121214; border-bottom: 1px solid #2A2A2E; spacing: 10px; padding: 4px; }
            QToolButton { background-color: #1E1E24; color: #E2E8F0; border: 1px solid #2A2A2E; border-radius: 4px; padding: 4px 8px; }
            QToolButton:hover { background-color: #2F3542; border-color: #00A2E8; }
            QToolButton:checked { background-color: #00A2E8; border-color: #00A2E8; color: white; }
            QMenuBar { background-color: #121214; color: #A0A5B5; border-bottom: 1px solid #1E1E24; }
            QMenuBar::item:selected { background-color: #1E1E24; color: #E2E8F0; }
            QMenu { background-color: #121214; color: #A0A5B5; border: 1px solid #2A2A2E; }
            QMenu::item:selected { background-color: #00A2E8; color: white; }
            QStatusBar { background-color: #121214; color: #747D8C; border-top: 1px solid #2A2A2E; }
            QSplitter::handle { background-color: #2A2A2E; }
            QLabel { color: #E2E8F0; }
            QLineEdit, QSpinBox, QDoubleSpinBox { background-color: #1E1E24; border: 1px solid #2A2A2E; border-radius: 4px; color: #E2E8F0; padding: 4px; }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #00A2E8; }
        """)

    # Drag & Drop
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls(): event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if not urls: return
        path = urls[0].toLocalFile()
        if os.path.isdir(path):
            if self._prompt_unsaved_changes():
                self.controller.load_folder(path)
                if self.controller.image_list: self.controller.open_image(self.controller.image_list[0])
        elif os.path.isfile(path):
            if os.path.splitext(path)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
                if self._prompt_unsaved_changes(): self.controller.open_image(path)

    def _prompt_unsaved_changes(self) -> bool:
        if not self.controller.is_dirty: return True
        is_autosave = self.settings.value("autoSave", False)
        if isinstance(is_autosave, str):
            is_autosave = is_autosave.lower() == "true"
        elif not isinstance(is_autosave, bool):
            is_autosave = bool(is_autosave)
        if is_autosave:
            return self.controller.save_annotations()
        res = QMessageBox.warning(
            self, "Unsaved Changes", "You have unsaved annotation changes. Do you want to save them?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save
        )
        if res == QMessageBox.StandardButton.Save: return self.controller.save_annotations()
        if res == QMessageBox.StandardButton.Discard: return True
        return False

    def closeEvent(self, event) -> None:
        if self._prompt_unsaved_changes(): event.accept()
        else: event.ignore()

    def _add_to_recent_files(self, file_path: str) -> None:
        recent = self.settings.value("recentFiles", [])
        if not isinstance(recent, list): recent = []
        if file_path in recent: recent.remove(file_path)
        recent.insert(0, file_path)
        self.settings.setValue("recentFiles", recent[:10])
        self._update_recent_files_menu()

    def _update_recent_files_menu(self) -> None:
        self.recent_menu.clear()
        recent = self.settings.value("recentFiles", [])
        if not recent:
            no_recent = QAction("No Recent Files", self)
            no_recent.setEnabled(False)
            self.recent_menu.addAction(no_recent)
            return
        for path in recent:
            action = QAction(os.path.basename(path), self)
            action.setData(path)
            action.setToolTip(path)
            action.triggered.connect(self._on_recent_file_clicked)
            self.recent_menu.addAction(action)

    def _on_recent_file_clicked(self) -> None:
        action = self.sender()
        if isinstance(action, QAction):
            path = action.data()
            if os.path.exists(path):
                if self._prompt_unsaved_changes(): self.controller.open_image(path)
            else:
                QMessageBox.warning(self, "File Not Found", f"File could not be found:\n{path}")
                recent = self.settings.value("recentFiles", [])
                if path in recent:
                    recent.remove(path)
                    self.settings.setValue("recentFiles", recent)
                self._update_recent_files_menu()

    def _update_item_style(self, item: QListWidgetItem, idx: int, f: str) -> None:
        is_doubt = f in self.controller.doubt_images
        is_finished = f in self.controller.finished_images
        base_name = os.path.basename(f)
        
        font = item.font()
        if is_doubt:
            item.setText(f"❓  {idx + 1}. {base_name}")
            item.setForeground(QColor("#FFB300"))  # Amber/Orange warning color for Doubt
            item.setBackground(QBrush(QColor(255, 179, 0, 25)))  # Subtle amber background highlight
            font.setBold(True)
            item.setFont(font)
            item.setToolTip(f"❓ Marked as Doubt: {base_name}")
        elif is_finished:
            item.setText(f"🟢  {idx + 1}. {base_name}")
            item.setForeground(QColor("#2ED573"))  # Vibrant Green color for Finished/Verified
            item.setBackground(QBrush(QColor(46, 213, 115, 25)))  # Subtle green background tint
            font.setBold(True)
            item.setFont(font)
            item.setToolTip(f"🟢 Finished / Verified: {base_name}")
        else:
            item.setText(f"🔴  {idx + 1}. {base_name}")
            item.setForeground(QColor("#FF4D4D"))  # Vivid Red color for Unfinished / Pending Review
            item.setBackground(QBrush(QColor(255, 77, 77, 20)))  # Subtle red background tint
            font.setBold(False)
            item.setFont(font)
            item.setToolTip(f"🔴 Unfinished / Pending Review: {base_name}")

    def _populate_file_list(self) -> None:
        self.file_list_widget.blockSignals(True)
        self.file_list_widget.clear()
        for idx, f in enumerate(self.controller.image_list):
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, f)
            self._update_item_style(item, idx, f)
            self.file_list_widget.addItem(item)
        self.file_list_widget.blockSignals(False)
        self._apply_file_filter()
        self._update_file_list_selection()

    def _apply_file_filter(self) -> None:
        filter_mode = self.combo_filter.currentIndex() if hasattr(self, "combo_filter") else 0
        total_count = len(self.controller.image_list)
        doubt_count = len(self.controller.doubt_images)
        finished_count = len(self.controller.finished_images)
        unfinished_count = total_count - finished_count
        visible_count = 0

        self.file_list_widget.blockSignals(True)
        for i in range(self.file_list_widget.count()):
            item = self.file_list_widget.item(i)
            f = item.data(Qt.ItemDataRole.UserRole)
            is_doubt = f in self.controller.doubt_images
            is_finished = f in self.controller.finished_images
            
            show = True
            if filter_mode == 1:     # 🔴 Unfinished Only
                show = not is_finished
            elif filter_mode == 2:   # 🟢 Finished Only
                show = is_finished
            elif filter_mode == 3:   # ❓ Doubt Only
                show = is_doubt

            item.setHidden(not show)
            if show:
                visible_count += 1
        self.file_list_widget.blockSignals(False)

        pct = (finished_count / total_count * 100.0) if total_count > 0 else 0.0
        txt = f"Finished: {finished_count} / {total_count} ({pct:.1f}%) | ❓ Doubt: {doubt_count}"

        if hasattr(self, "lbl_verification_counter"):
            self.lbl_verification_counter.setText(txt)

    def _on_action_finish(self) -> None:
        if not self.controller.image_path:
            return
        if self.controller.is_dirty:
            self._on_action_save()

        is_finished = self.controller.toggle_finished()
        current_path = self.controller.image_path

        if current_path in self.controller.image_list:
            idx = self.controller.image_list.index(current_path)
            item = self.file_list_widget.item(idx)
            if item:
                self._update_item_style(item, idx, current_path)

        self._apply_file_filter()

    def _on_action_export_finished(self) -> None:
        if not self.controller.finished_images:
            QMessageBox.information(
                self, "Export Finished Labels",
                "No images are currently marked as Finished (GREEN).\n\n"
                "Please review and mark images as Finished first by clicking the Finish button or pressing Shift+F."
            )
            return

        export_dir = QFileDialog.getExistingDirectory(
            self, "Select Destination Folder for Exporting Finished Labels"
        )
        if not export_dir:
            return

        exported_count = 0
        missing_count = 0

        for img_path in list(self.controller.finished_images):
            label_path = self.controller.image_to_label.get(img_path)
            base_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]

            if not label_path or not os.path.exists(label_path):
                alt_txt = os.path.splitext(img_path)[0] + ".txt"
                if os.path.exists(alt_txt):
                    label_path = alt_txt

            if label_path and os.path.exists(label_path):
                dest_file = os.path.join(export_dir, f"{base_name_no_ext}.txt")
                try:
                    import shutil
                    shutil.copy2(label_path, dest_file)
                    exported_count += 1
                except Exception as e:
                    print(f"[ERROR] Exporting label {label_path}: {e}")
            else:
                missing_count += 1

        msg = f"Successfully exported {exported_count} finished label file(s) to:\n{export_dir}"
        if missing_count > 0:
            msg += f"\n\n({missing_count} finished image(s) did not have label files on disk)."

        QMessageBox.information(self, "Export Complete", msg)

    def _on_action_copy_prev_bboxes(self) -> None:
        if self.controller.clipboard_annotations or self.controller.clipboard_item:
            pasted = self.controller.paste_annotation(exact=True)
            if pasted:
                count = len(self.controller.annotations)
                self.statusBar().showMessage(f"📋 Pasted {count} copied bbox & polygon coordinate(s) on current frame & saved", 4000)
                if hasattr(self, "chk_always_zoom_plate") and self.chk_always_zoom_plate and self.chk_always_zoom_plate.isChecked():
                    self.zoom_to_license_plate()
                if self.controller.image_path in self.controller.image_list:
                    idx = self.controller.image_list.index(self.controller.image_path)
                    item = self.file_list_widget.item(idx)
                    if item:
                        self._update_item_style(item, idx, self.controller.image_path)
                return

        count, prev_name = self.controller.copy_annotations_from_prev_image()
        if count > 0:
            self.statusBar().showMessage(f"📋 Copied {count} exact bbox(es) from previous frame ({prev_name}) & saved", 4000)
            if hasattr(self, "chk_always_zoom_plate") and self.chk_always_zoom_plate and self.chk_always_zoom_plate.isChecked():
                self.zoom_to_license_plate()
            if self.controller.image_path in self.controller.image_list:
                idx = self.controller.image_list.index(self.controller.image_path)
                item = self.file_list_widget.item(idx)
                if item:
                    self._update_item_style(item, idx, self.controller.image_path)
        else:
            if self.controller.current_idx <= 0:
                self.statusBar().showMessage("⚠️ No previous frame or copied clipboard bbox available", 3000)
            else:
                self.statusBar().showMessage(f"⚠️ No bboxes found in previous frame ({prev_name})", 3000)

    def _on_action_copy_next_bboxes(self) -> None:
        count, next_name = self.controller.copy_annotations_from_next_image()
        if count > 0:
            self.statusBar().showMessage(f"📋 Copied {count} exact bbox(es) from next frame ({next_name}) & saved", 4000)
            if hasattr(self, "chk_always_zoom_plate") and self.chk_always_zoom_plate and self.chk_always_zoom_plate.isChecked():
                self.zoom_to_license_plate()
            if self.controller.image_path in self.controller.image_list:
                idx = self.controller.image_list.index(self.controller.image_path)
                item = self.file_list_widget.item(idx)
                if item:
                    self._update_item_style(item, idx, self.controller.image_path)
        else:
            if self.controller.current_idx >= len(self.controller.image_list) - 1:
                self.statusBar().showMessage("⚠️ No next frame available", 3000)
            else:
                self.statusBar().showMessage(f"⚠️ No bboxes found in next frame ({next_name})", 3000)

    def _copy_bboxes_from_specific_image(self, src_img_path: str) -> None:
        count = self.controller.copy_annotations_from_image(src_img_path)
        base_name = os.path.basename(src_img_path)
        if count > 0:
            self.statusBar().showMessage(f"📋 Copied {count} exact bbox(es) from {base_name} & saved", 4000)
            if hasattr(self, "chk_always_zoom_plate") and self.chk_always_zoom_plate and self.chk_always_zoom_plate.isChecked():
                self.zoom_to_license_plate()
            if self.controller.image_path in self.controller.image_list:
                idx = self.controller.image_list.index(self.controller.image_path)
                item = self.file_list_widget.item(idx)
                if item:
                    self._update_item_style(item, idx, self.controller.image_path)
        else:
            self.statusBar().showMessage(f"⚠️ No bboxes found in {base_name}", 3000)

    def keyPressEvent(self, event) -> None:
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QSpinBox, QDoubleSpinBox)):
            super().keyPressEvent(event)
            return

        key = event.key()
        if key == Qt.Key.Key_D:
            self._on_action_next()
            event.accept()
            return
        elif key == Qt.Key.Key_A:
            self._on_action_prev()
            event.accept()
            return
        elif key in (Qt.Key.Key_E, Qt.Key.Key_BracketLeft):
            self._on_action_copy_prev_bboxes()
            event.accept()
            return
        elif key in (Qt.Key.Key_N, Qt.Key.Key_BracketRight):
            self._on_action_copy_next_bboxes()
            event.accept()
            return
        elif key == Qt.Key.Key_F:
            if event.modifiers() & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier):
                self._on_action_finish()
            else:
                self.viewer.fit_image()
            event.accept()
            return
        elif key == Qt.Key.Key_R:
            self.viewer.reset_zoom()
            event.accept()
            return
        elif key == Qt.Key.Key_V:
            self._on_action_toggle_verify()
            event.accept()
            return
        elif key == Qt.Key.Key_Z:
            self.zoom_to_license_plate()
            event.accept()
            return

        super().keyPressEvent(event)

    def _show_file_list_context_menu(self, pos: QPoint) -> None:
        item = self.file_list_widget.itemAt(pos)
        if not item:
            return
        image_path = item.data(Qt.ItemDataRole.UserRole)
        if not image_path:
            return
        
        is_doubt = image_path in self.controller.doubt_images
        is_finished = image_path in self.controller.finished_images
        
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background-color: #1E1E24; color: #E2E8F0; border: 1px solid #2A2A2E; padding: 4px; }
            QMenu::item { padding: 6px 20px; border-radius: 2px; }
            QMenu::item:selected { background-color: #00A2E8; color: white; }
        """)
        
        act_finish = QAction("🟢 Unmark Finished" if is_finished else "✅ Mark Finished", self)
        act_finish.triggered.connect(lambda: self._toggle_item_finished(image_path))
        menu.addAction(act_finish)

        if image_path != self.controller.image_path:
            act_copy_from = QAction("📋 Copy Bboxes from this Image to Current", self)
            act_copy_from.triggered.connect(lambda: self._copy_bboxes_from_specific_image(image_path))
            menu.addAction(act_copy_from)

        if is_doubt:
            act_toggle = QAction("❌ Remove Doubt Mark", self)
        else:
            act_toggle = QAction("❓ Mark as Doubt", self)
        act_toggle.triggered.connect(lambda: self._toggle_item_doubt(image_path))
        menu.addAction(act_toggle)

        menu.addSeparator()
        act_del = QAction("🗑️ Delete File from Disk", self)
        act_del.triggered.connect(lambda: self._on_action_delete_files_from_disk(image_path))
        menu.addAction(act_del)

        menu.exec(self.file_list_widget.mapToGlobal(pos))

    def _toggle_item_finished(self, image_path: str) -> None:
        self.controller.toggle_finished(image_path)
        if image_path in self.controller.image_list:
            idx = self.controller.image_list.index(image_path)
            item = self.file_list_widget.item(idx)
            if item:
                self._update_item_style(item, idx, image_path)
        self._apply_file_filter()

    def _toggle_item_doubt(self, image_path: str) -> None:
        is_doubt, new_path = self.controller.toggle_doubt(image_path)
        if new_path in self.controller.image_list:
            idx = self.controller.image_list.index(new_path)
            item = self.file_list_widget.item(idx)
            if item:
                self._update_item_style(item, idx, new_path)
        self._apply_file_filter()
            
        act_toggle.triggered.connect(lambda: self._toggle_verification_for_path(image_path))
        menu.addAction(act_toggle)
        menu.addSeparator()

        act_del_disk = QAction("🗑️ Delete Files from Disk", self)
        act_del_disk.triggered.connect(lambda: self._on_action_delete_files_from_disk(image_path))
        menu.addAction(act_del_disk)

        menu.exec(self.file_list_widget.mapToGlobal(pos))

    def _on_action_delete_files_from_disk(self, target_path: Optional[str] = None) -> None:
        img_path = target_path or self.controller.image_path
        if not img_path:
            return

        base_name = os.path.basename(img_path)
        reply = QMessageBox.question(
            self,
            "Confirm Permanent File Deletion",
            f"Are you sure you want to permanently delete image file '{base_name}' and its associated label files from disk?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            success = self.controller.delete_image_files_from_disk(img_path)
            if success:
                self._populate_file_list()
                self.status_bar.showMessage(f"🗑️ Permanently deleted '{base_name}' and associated label files from disk", 4000)
            else:
                QMessageBox.warning(self, "Delete Error", f"Failed to delete '{base_name}' from disk.")

    def _toggle_verification_for_path(self, image_path: str) -> None:
        is_doubt, new_path = self.controller.toggle_verification(image_path)
        base_name = os.path.basename(image_path)
        
        self._populate_file_list()
        
        status_msg = f"❓ Marked '{base_name}' as Doubt & moved to doubt folder" if is_doubt else f"○ Removed Doubt mark for '{base_name}' & moved back"
        self.status_bar.showMessage(status_msg, 3000)

    def _on_action_toggle_verify(self) -> None:
        target_f = None
        if 0 <= self.controller.current_idx < len(self.controller.image_list):
            target_f = self.controller.image_list[self.controller.current_idx]
        else:
            sel = self.file_list_widget.selectedItems()
            if sel:
                target_f = sel[0].data(Qt.ItemDataRole.UserRole)
                
        if target_f:
            self._toggle_verification_for_path(target_f)

    def _update_file_list_selection(self) -> None:
        self.file_list_widget.blockSignals(True)
        idx = self.controller.current_idx
        if 0 <= idx < self.file_list_widget.count():
            self.file_list_widget.setCurrentRow(idx)
        self.file_list_widget.blockSignals(False)

    def zoom_to_license_plate(self) -> None:
        if not hasattr(self, "viewer") or not self.viewer:
            return
        
        target_rect = None
        selected_items = self.viewer.selected_items() if hasattr(self.viewer, "selected_items") else []
        
        if selected_items and hasattr(selected_items[0], "box"):
            box = selected_items[0].box
            target_rect = QRectF(box.x, box.y, box.width, box.height)
        elif self.controller.annotations:
            box = self.controller.annotations[0].box
            target_rect = QRectF(box.x, box.y, box.width, box.height)
            
        if target_rect and not target_rect.isEmpty():
            self.viewer.fit_bbox(target_rect)
            self.statusBar().showMessage("🔍 Zoomed to License Plate / Bbox", 3000)
        else:
            self.viewer.fit_image()
            self.statusBar().showMessage("Fit Full Image (No License Plate Bbox)", 3000)

    def _on_toggle_always_zoom_plate(self, checked: bool) -> None:
        self.settings.setValue("alwaysZoomPlate", checked)
        if hasattr(self, "act_always_zoom_plate") and self.act_always_zoom_plate:
            self.act_always_zoom_plate.blockSignals(True)
            self.act_always_zoom_plate.setChecked(checked)
            self.act_always_zoom_plate.blockSignals(False)
        if checked:
            if hasattr(self, "chk_always_fit") and self.chk_always_fit:
                self.chk_always_fit.blockSignals(True)
                self.chk_always_fit.setChecked(False)
                self.chk_always_fit.blockSignals(False)
                self.settings.setValue("alwaysFitImage", False)
            if hasattr(self, "act_always_fit") and self.act_always_fit:
                self.act_always_fit.blockSignals(True)
                self.act_always_fit.setChecked(False)
                self.act_always_fit.blockSignals(False)
            self.zoom_to_license_plate()

    def _on_toggle_always_fit(self, checked: bool) -> None:
        self.settings.setValue("alwaysFitImage", checked)
        if hasattr(self, "act_always_fit") and self.act_always_fit:
            self.act_always_fit.blockSignals(True)
            self.act_always_fit.setChecked(checked)
            self.act_always_fit.blockSignals(False)
        if checked:
            if hasattr(self, "chk_always_zoom_plate") and self.chk_always_zoom_plate:
                self.chk_always_zoom_plate.blockSignals(True)
                self.chk_always_zoom_plate.setChecked(False)
                self.chk_always_zoom_plate.blockSignals(False)
                self.settings.setValue("alwaysZoomPlate", False)
            if hasattr(self, "act_always_zoom_plate") and self.act_always_zoom_plate:
                self.act_always_zoom_plate.blockSignals(True)
                self.act_always_zoom_plate.setChecked(False)
                self.act_always_zoom_plate.blockSignals(False)
            if hasattr(self, "viewer") and self.viewer:
                self.viewer.fit_image()

    def _on_controller_image_loaded(self, path: str) -> None:
        self._add_to_recent_files(path)
        self._set_editor_mode("select")
        if self.controller.current_pixmap:
            pixmap = self.viewer.load_image_from_pixmap(self.controller.current_pixmap)
        else:
            pixmap = self.viewer.load_image(path)
        if pixmap:
            self._update_file_list_selection()
            if hasattr(self, "chk_always_zoom_plate") and self.chk_always_zoom_plate and self.chk_always_zoom_plate.isChecked():
                self.zoom_to_license_plate()
            elif hasattr(self, "chk_always_fit") and self.chk_always_fit and self.chk_always_fit.isChecked():
                self.viewer.fit_image()
            
            current_f = self.controller.image_list[self.controller.current_idx] if (0 <= self.controller.current_idx < len(self.controller.image_list)) else None
            if current_f:
                item = self.file_list_widget.item(self.controller.current_idx)
                if item:
                    self._update_item_style(item, self.controller.current_idx, current_f)
            
            label_exists = True if self.controller.anno_path.startswith("sftp://") else os.path.exists(self.controller.anno_path)
            label_status = "" if label_exists else " [Missing Label]"
            display_name = os.path.basename(current_f) if current_f else os.path.basename(path)
            self.status_image_path.setText(f"File: {display_name} ({pixmap.width()}x{pixmap.height()}){label_status}")
            if not label_exists:
                self.status_bar.showMessage("Warning: Missing Label file", 5000)
        else:
            self.status_image_path.setText("Failed to load image.")

    def _on_controller_state_changed(self) -> None:
        selected_item = self.viewer.get_selected_item()
        selected_anno = selected_item.annotation_item if selected_item else None
        
        self.viewer.clear_canvas()
        to_select = None
        for item in self.controller.annotations:
            g_item = BBoxGraphicItem(item)
            self.viewer.scene_obj.addItem(g_item)
            g_item.signals.selected.connect(self.viewer.select_item)
            g_item.signals.geometryChanged.connect(self._on_viewer_item_moving)
            g_item.signals.geometryEditFinished.connect(self.controller.commit_geometry_change)
            if selected_anno and item is selected_anno:
                to_select = g_item
                
        if to_select:
            self.viewer.select_item(to_select)
        else:
            self._on_viewer_selection_changed()

    def _on_dirty_state_changed(self, dirty: bool) -> None:
        title = "PlateAnnotate - License Plate Annotation Editor"
        if dirty: title += " *"
        self.setWindowTitle(title)

    def _update_undo_redo_actions(self, can_undo: bool, can_redo: bool) -> None:
        self.act_undo.setEnabled(can_undo)
        self.act_redo.setEnabled(can_redo)
        if hasattr(self, "btn_undo") and self.btn_undo:
            self.btn_undo.setEnabled(can_undo)
        if hasattr(self, "btn_redo") and self.btn_redo:
            self.btn_redo.setEnabled(can_redo)

    def _on_viewer_selection_changed(self) -> None:
        selected = self.viewer.get_selected_item()
        self.properties_panel.set_selected_item(selected.annotation_item if selected else None)
        self.update_crop_preview()

    def _on_viewer_item_moving(self, g_item: BBoxGraphicItem) -> None:
        self.properties_panel.update_fields(include_crop=True)
        self.update_crop_preview()

    def update_crop_preview(self) -> None:
        selected = self.viewer.get_selected_item()
        if not selected or not selected.annotation_item:
            self.lbl_crop.setText("Select a box to preview crop")
            self.lbl_crop.setPixmap(QPixmap())
            self.btn_save_crop.setEnabled(False)
            return

        self.btn_save_crop.setEnabled(True)
        warped_pixmap = self._get_warped_crop(selected.annotation_item)
        if warped_pixmap and not warped_pixmap.isNull():
            scaled = warped_pixmap.scaled(
                300, 180,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            target = QPixmap(300, 180)
            target.fill(QColor("#121214"))
            
            painter = QPainter(target)
            dx = (300 - scaled.width()) // 2
            dy = (180 - scaled.height()) // 2
            painter.drawPixmap(dx, dy, scaled)
            painter.end()
            
            self.lbl_crop.setPixmap(target)
            return

        if self.viewer._raw_pixmap and not self.viewer._raw_pixmap.isNull():
            item = selected.annotation_item
            raw_pixmap = self.viewer._raw_pixmap
            px_w = raw_pixmap.width()
            px_h = raw_pixmap.height()
            x = max(0, min(item.x, px_w - 1))
            y = max(0, min(item.y, px_h - 1))
            w = max(1, min(item.width, px_w - x))
            h = max(1, min(item.height, px_h - y))
            
            cropped = raw_pixmap.copy(x, y, w, h)
            scaled = cropped.scaled(
                300, 180,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            target = QPixmap(300, 180)
            target.fill(QColor("#121214"))
            
            painter = QPainter(target)
            dx = (300 - scaled.width()) // 2
            dy = (180 - scaled.height()) // 2
            painter.drawPixmap(dx, dy, scaled)
            painter.end()
            
            self.lbl_crop.setPixmap(target)
            return
        
        self.lbl_crop.setText("No image loaded")
        self.lbl_crop.setPixmap(QPixmap())

    def _on_btn_save_crop_clicked(self) -> None:
        selected = self.viewer.get_selected_item()
        if not selected or not selected.annotation_item:
            QMessageBox.warning(self, "No Selection", "Please select a bounding box to save its crop.")
            return

        if self.controller.image_path:
            item = selected.annotation_item
            warped_pixmap = self._get_warped_crop(item)
            
            img_path = self.controller.image_path
            img_dir = os.path.dirname(img_path)
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            
            crops_dir = os.path.join(img_dir, "crops")
            os.makedirs(crops_dir, exist_ok=True)
            
            safe_label = "".join([c for c in item.label if c.isalnum() or c in ("-", "_")]).strip()
            if not safe_label: safe_label = "crop"
            
            out_filename = f"{img_name}_{safe_label}.png"
            out_path = os.path.join(crops_dir, out_filename)
            
            if warped_pixmap and not warped_pixmap.isNull():
                success = warped_pixmap.save(out_path, "PNG")
            else:
                raw_pixmap = self.viewer._raw_pixmap
                if raw_pixmap and not raw_pixmap.isNull():
                    px_w = raw_pixmap.width()
                    px_h = raw_pixmap.height()
                    x = max(0, min(item.x, px_w - 1))
                    y = max(0, min(item.y, px_h - 1))
                    w = max(1, min(item.width, px_w - x))
                    h = max(1, min(item.height, px_h - y))
                    cropped = raw_pixmap.copy(x, y, w, h)
                    success = cropped.save(out_path, "PNG")
                else:
                    success = False
            
            if success:
                QMessageBox.information(self, "Crop Saved", f"Successfully saved crop image to:\n{out_path}")
            else:
                QMessageBox.critical(self, "Save Failed", f"Failed to save crop image to:\n{out_path}")

    def _get_warped_crop(self, item: AnnotationItem) -> Optional[QPixmap]:
        if not self.controller.image_path or not os.path.exists(self.controller.image_path):
            return None
            
        import cv2
        import numpy as np
        
        image = cv2.imread(self.controller.image_path)
        if image is None:
            return None
            
        h_img, w_img = image.shape[:2]
        pts = item.box.points
        if not pts or len(pts) != 4:
            x, y, w, h = item.x, item.y, item.width, item.height
            pts = [
                (float(x), float(y)),
                (float(x + w), float(y)),
                (float(x + w), float(y + h)),
                (float(x), float(y + h))
            ]
            
        clipped_pts = []
        for p in pts:
            cx = max(0.0, min(float(p[0]), float(w_img - 1)))
            cy = max(0.0, min(float(p[1]), float(h_img - 1)))
            clipped_pts.append((cx, cy))
            
        pts_np = np.array(clipped_pts, dtype="float32")
        
        rect = np.zeros((4, 2), dtype="float32")
        s = pts_np.sum(axis=1)
        rect[0] = pts_np[np.argmin(s)]
        rect[2] = pts_np[np.argmax(s)]
        
        diff = np.diff(pts_np, axis=1).flatten()
        rect[1] = pts_np[np.argmin(diff)]
        rect[3] = pts_np[np.argmax(diff)]
        
        (tl, tr, br, bl) = rect
        
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        if maxWidth < 5 or maxHeight < 5:
            return None
            
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
            
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        h_warp, w_warp, ch = warped_rgb.shape
        bytes_per_line = ch * w_warp
        qimg = QImage(warped_rgb.data, w_warp, h_warp, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    def _on_crop_preview_clicked(self) -> None:
        selected = self.viewer.get_selected_item()
        if not selected or not selected.annotation_item:
            return
            
        warped_pixmap = self._get_warped_crop(selected.annotation_item)
        if not warped_pixmap or warped_pixmap.isNull():
            return
            
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Crop Preview - {selected.annotation_item.label}")
        dialog.setStyleSheet("background-color: #1A1A1E; color: #E2E8F0;")
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        lbl_large = QLabel()
        lbl_large.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        scaled = warped_pixmap.scaled(
            600, 400,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        lbl_large.setPixmap(scaled)
        layout.addWidget(lbl_large, 1)
        
        info_lbl = QLabel(f"Size: {warped_pixmap.width()} x {warped_pixmap.height()} | Label: {selected.annotation_item.label}")
        info_lbl.setStyleSheet("color: #747D8C; font-size: 10pt;")
        info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_lbl)
        
        btn_close = QPushButton("Close")
        btn_close.setStyleSheet("""
            QPushButton { background-color: #2F3542; color: white; padding: 8px 16px; font-weight: bold; border-radius: 4px; border: 1px solid #2A2A2E; }
            QPushButton:hover { background-color: #3F4452; border-color: #00A2E8; }
        """)
        btn_close.clicked.connect(dialog.accept)
        layout.addWidget(btn_close, 0, Qt.AlignmentFlag.AlignCenter)
        
        dialog.setLayout(layout)
        dialog.exec()
    def _on_viewer_zoom_changed(self, percentage: float) -> None:
        self.status_zoom.setText(f"Zoom: {percentage:.0f}%")

    def _on_property_edited(self, prop_name: str, new_val) -> None:
        selected = self.viewer.get_selected_item()
        if selected:
            self.controller.modify_property(selected.annotation_item, prop_name, new_val)
            selected.update_from_model()

    def _on_sidebar_selection_changed(self) -> None:
        sel = self.file_list_widget.selectedItems()
        if not sel: return
        target_path = sel[0].data(Qt.ItemDataRole.UserRole)
        if target_path != self.controller.image_path:
            if self._prompt_unsaved_changes():
                self.controller.open_image(target_path)
            else:
                self.file_list_widget.blockSignals(True)
                for i in range(self.file_list_widget.count()):
                    item = self.file_list_widget.item(i)
                    if item.data(Qt.ItemDataRole.UserRole) == self.controller.image_path:
                        self.file_list_widget.setCurrentItem(item)
                        break
                self.file_list_widget.blockSignals(False)

    def _on_action_open_image(self) -> None:
        if not self._prompt_unsaved_changes(): return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image or Annotation File", "",
            "All Files (*.jpg *.jpeg *.png *.bmp *.tiff *.webp *.txt *.xml *.json *.csv);;Images (*.jpg *.jpeg *.png *.bmp *.tiff *.webp);;Annotations (*.txt *.xml *.json *.csv)"
        )
        if path:
            ext = os.path.splitext(path)[1].lower()
            if ext in (".txt", ".xml", ".json", ".csv"):
                success = self.controller.open_annotation_file(path)
                if success:
                    self._populate_file_list()
                else:
                    msg = f"Failed to load annotation file or fetch its linked image:\n{path}"
                    if self.controller.last_error:
                        msg += f"\n\nDetails:\n{self.controller.last_error}"
                    QMessageBox.critical(self, "Load Error", msg)
            else:
                self.controller.open_image(path)

    def _on_action_load_annotation(self) -> None:
        default_dir = os.path.dirname(self.controller.image_path) if self.controller.image_path else ""
        path, _ = QFileDialog.getOpenFileName(self, "Load Annotation File", default_dir, "Annotation Files (*.txt *.xml *.json *.csv)")
        if path:
            if not self.controller.image_path:
                success = self.controller.open_annotation_file(path)
                if success:
                    self._populate_file_list()
                else:
                    msg = f"Failed to load annotation file or fetch its linked image:\n{path}"
                    if self.controller.last_error:
                        msg += f"\n\nDetails:\n{self.controller.last_error}"
                    QMessageBox.critical(self, "Load Error", msg)
            else:
                parser = ParserFactory.get_parser_for_file(path)
                if not parser:
                    QMessageBox.critical(self, "Unsupported Format", "Selected annotation extension not supported.")
                    return
                self.controller.anno_path = path
                self.controller.reload_annotation()

    def _on_action_open_folder(self) -> None:
        if not self._prompt_unsaved_changes(): return
        path = QFileDialog.getExistingDirectory(self, "Open Folder")
        if path:
            self.controller.load_folder(path)
            self._populate_file_list()
            if self.controller.image_list: self.controller.open_image(self.controller.image_list[0])
            else: QMessageBox.information(self, "Empty Folder", "No supported images found in folder.")

    def _on_action_save(self) -> None:
        if self.controller.save_annotations():
            self.status_bar.showMessage("Annotations saved.", 3000)
            idx = self.controller.current_idx
            if 0 <= idx < self.file_list_widget.count():
                current_f = self.controller.image_list[idx]
                if len(self.controller.annotations) > 0:
                    self.controller.finished_images.add(current_f)
                else:
                    self.controller.finished_images.discard(current_f)
                item = self.file_list_widget.item(idx)
                if item:
                    is_finished = current_f in self.controller.finished_images
                    item.setForeground(QColor("#2ECC71" if is_finished else "#A0A5B5"))
        else:
            self.status_bar.showMessage("Failed to save.", 3000)

    def _on_action_save_as(self) -> None:
        if not self.controller.image_path: return
        path, f_filter = QFileDialog.getSaveFileName(self, "Save Annotations As", os.path.splitext(self.controller.image_path)[0], "YOLO Text (*.txt);;VOC XML (*.xml);;Custom JSON (*.json);;CSV Labels (*.csv)")
        if path:
            ext = ".txt" if "YOLO" in f_filter else (".xml" if "VOC" in f_filter else (".json" if "JSON" in f_filter else ".csv"))
            if not path.lower().endswith(ext): path += ext
            if self.controller.save_annotations_as(path): self.status_bar.showMessage(f"Saved: {os.path.basename(path)}", 3000)
            else: self.status_bar.showMessage("Failed to save.", 3000)

    def _on_action_export_json(self) -> None:
        if not self.controller.image_path: return
        path, _ = QFileDialog.getSaveFileName(self, "Export JSON File", os.path.splitext(self.controller.image_path)[0] + ".json", "JSON Files (*.json)")
        if path:
            if self.controller.export_as_json(path): self.status_bar.showMessage("Exported JSON.", 3000)
            else: QMessageBox.critical(self, "Export Failed", "Failed to export.")

    def _on_action_reload(self) -> None:
        res = QMessageBox.question(self, "Confirm Reload", "Reload annotation? Discards unsaved edits.", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if res == QMessageBox.StandardButton.Yes: self.controller.reload_annotation()

    def _on_toggle_autosave(self, checked: bool) -> None:
        self.settings.setValue("autoSave", checked)
        self.status_bar.showMessage(f"Auto-Save {'Enabled' if checked else 'Disabled'}", 2000)

    def _on_action_copy(self) -> None:
        sel = self.viewer.get_selected_item()
        target_item = sel.annotation_item if sel else None
        count = self.controller.copy_annotation(target_item)
        if count > 0:
            self.status_bar.showMessage(f"📋 Copied {count} bbox & polygon coordinate(s) (Press E on next frame to paste)", 3000)
        else:
            self.status_bar.showMessage("⚠️ No bboxes found to copy on current frame", 3000)

    def _on_action_paste(self) -> None:
        pasted = self.controller.paste_annotation(exact=True)
        if pasted:
            if hasattr(self, "chk_always_zoom_plate") and self.chk_always_zoom_plate and self.chk_always_zoom_plate.isChecked():
                self.zoom_to_license_plate()
            self.status_bar.showMessage(f"📋 Pasted copied bbox coordinates ({pasted.label}) on current frame & saved", 3000)

    def _on_action_duplicate(self) -> None:
        sel = self.viewer.get_selected_item()
        if sel:
            dup = self.controller.duplicate_annotation(sel.annotation_item)
            if dup:
                for item in self.viewer.scene_obj.items():
                    if isinstance(item, BBoxGraphicItem) and item.annotation_item is dup:
                        self.viewer.select_item(item)
                        break

    def _on_action_delete(self) -> None:
        sel = self.viewer.get_selected_item()
        if sel: self.controller.delete_annotations([sel.annotation_item])

    def _on_action_prev(self) -> None:
        if self._prompt_unsaved_changes(): self.controller.prev_image()

    def _on_action_next(self) -> None:
        if self._prompt_unsaved_changes(): self.controller.next_image()

    def _set_editor_mode(self, mode: str) -> None:
        self.viewer.set_mode(mode)
        self.status_mode.setText(f"Mode: {mode.capitalize()}")
        
        # Sync Actions
        self.act_tb_select.blockSignals(True)
        self.act_tb_create.blockSignals(True)
        self.act_tb_poly.blockSignals(True)
        self.act_tb_pan.blockSignals(True)
        self.act_tb_select.setChecked(mode == "select")
        self.act_tb_create.setChecked(mode == "create")
        self.act_tb_poly.setChecked(mode == "polygon")
        self.act_tb_pan.setChecked(mode == "pan")
        self.act_tb_select.blockSignals(False)
        self.act_tb_create.blockSignals(False)
        self.act_tb_poly.blockSignals(False)
        self.act_tb_pan.blockSignals(False)

        # Sync Sidebar Buttons
        if hasattr(self, "btn_tool_select") and self.btn_tool_select:
            self.btn_tool_select.blockSignals(True)
            self.btn_tool_create.blockSignals(True)
            self.btn_tool_poly.blockSignals(True)
            self.btn_tool_select.setChecked(mode == "select")
            self.btn_tool_create.setChecked(mode == "create")
            self.btn_tool_poly.setChecked(mode == "polygon")
            self.btn_tool_select.blockSignals(False)
            self.btn_tool_create.blockSignals(False)
            self.btn_tool_poly.blockSignals(False)

    def _on_action_zoom_in(self) -> None:
        self.viewer.zoom_by_factor(self.viewer.zoom_factor)

    def _on_action_zoom_out(self) -> None:
        self.viewer.zoom_by_factor(1.0 / self.viewer.zoom_factor)

    def _on_action_about(self) -> None:
        QMessageBox.about(self, "About PlateAnnotate", "<h3>PlateAnnotate v1.0.0</h3><p>Single-file License Plate Annotation Editor.</p>")

    def show_annotation_preview(self) -> None:
        """
        Opens the separate Annotation Preview popup window.
        Brings existing preview window to front if already opened.
        """
        if self._preview_dialog is None or not self._preview_dialog.isVisible():
            self._preview_dialog = AnnotationPreviewDialog(self.controller, self)
            self._preview_dialog.show()
        else:
            self._preview_dialog.raise_()
            self._preview_dialog.activateWindow()
            self._preview_dialog.refresh_preview()


# ==============================================================================
# 9. ENTRY POINT
# ==============================================================================

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Outfit", 9))
    
    controller = EditorController()
    window = MainWindow(controller)
    window.show()
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            if os.path.isdir(path):
                controller.load_folder(path)
                window._populate_file_list()
                if controller.image_list: controller.open_image(controller.image_list[0])
            elif os.path.isfile(path):
                ext = os.path.splitext(path)[1].lower()
                if ext in (".txt", ".xml", ".json", ".csv"):
                    success = controller.open_annotation_file(path)
                    if success:
                        window._populate_file_list()
                else:
                    controller.open_image(path)
                    window._populate_file_list()
                
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
