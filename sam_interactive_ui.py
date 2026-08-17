# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Interactive SAM Segmentation UI with PyQt5
Click to add foreground points (left click) or background points (right click)
Press 'r' to reset, arrow keys to navigate images
"""

import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QListWidget, QPushButton, 
                             QFileDialog, QSplitter, QStatusBar, QScrollArea)
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor

# Add parent directory to path for SAM imports
sys.path.append(str(Path(__file__).parent / "segment-anything-main"))
from segment_anything import sam_model_registry, SamPredictor


class ImageLabel(QLabel):
    """Custom QLabel for displaying images with mouse interaction."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = None
        self.setMouseTracking(True)
        self.setScaledContents(False)
    
    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if self.parent_app and self.pixmap():
            # Get click position relative to widget
            pos = event.pos()
            x, y = pos.x(), pos.y()
            
            # Get pixmap position within label (accounting for alignment)
            pixmap = self.pixmap()
            label_width = self.width()
            label_height = self.height()
            pixmap_width = pixmap.width()
            pixmap_height = pixmap.height()
            
            # Calculate offset for centered image
            offset_x = (label_width - pixmap_width) / 2
            offset_y = (label_height - pixmap_height) / 2
            
            # Adjust coordinates relative to pixmap
            x = x - offset_x
            y = y - offset_y
            
            # Check if click is within pixmap bounds
            if x < 0 or y < 0 or x >= pixmap_width or y >= pixmap_height:
                return
            
            # Convert to image coordinates accounting for zoom
            img_x, img_y = self.parent_app.screen_to_image_coords(x, y)
            
            if event.button() == Qt.LeftButton:
                # Left click - foreground point
                self.parent_app.add_point(img_x, img_y, label=1)
            elif event.button() == Qt.RightButton:
                # Right click - background point
                self.parent_app.add_point(img_x, img_y, label=0)
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming and scrolling."""
        if self.parent_app:
            modifiers = QApplication.keyboardModifiers()
            delta = event.angleDelta().y()
            
            if modifiers == Qt.ControlModifier:
                # Ctrl+Scroll: Zoom in/out
                if delta > 0:
                    self.parent_app.zoom_in()
                else:
                    self.parent_app.zoom_out()
                event.accept()
            else:
                # No modifier or Shift: Let the scroll area handle it
                # This allows natural scrolling behavior
                event.ignore()


class SAMInteractiveApp(QMainWindow):
    """Interactive SAM segmentation application with PyQt5."""
    
    def __init__(self, sam_checkpoint="sam_vit_h_4b8939.pth", model_type="vit_h", device="cuda"):
        """Initialize SAM model and application.
        
        Args:
            sam_checkpoint (str): Path to SAM checkpoint file
            model_type (str): SAM model type (vit_h, vit_l, vit_b)
            device (str): Device to run model on (cuda/cpu)
        """
        super().__init__()
        
        # Initialize SAM model
        print(f"Loading SAM model: {model_type}")
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        self.device = device
        
        # Application state
        self.image = None
        self.image_display = None
        self.points = []
        self.labels = []
        self.mask = None
        
        # Image list management
        self.image_paths = []
        self.current_image_index = 0
        
        # Display settings
        self.point_radius = 5
        self.fg_color = (0, 255, 0)  # Green for foreground
        self.bg_color = (0, 0, 255)  # Red for background
        self.mask_color = (30, 144, 255)  # Blue mask overlay
        self.mask_alpha = 0.4
        
        # Zoom settings
        self.zoom_level = 1.0
        self.zoom_min = 0.1
        self.zoom_max = 10.0
        self.zoom_step = 0.1
        
        # Setup UI
        self.init_ui()
        
        print("SAM Interactive App initialized")
        print("Controls:")
        print("  Left Click      - Add foreground point")
        print("  Right Click     - Add background point")
        print("  Ctrl+Wheel      - Zoom in/out")
        print("  Mouse Wheel     - Scroll vertically")
        print("  Shift+Wheel     - Scroll horizontally")
        print("  '+' / '-'       - Zoom in/out")
        print("  Left/Right Arrow- Previous/Next image")
        print("  'r' - Reset points")
        print("  'z' - Reset zoom")
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("SAM Interactive Segmentation")
        
        # Get screen size and set window size to fit
        screen = QApplication.primaryScreen().geometry()
        width = min(1400, screen.width() - 100)
        height = min(800, screen.height() - 100)
        self.setGeometry(100, 100, width, height)
        
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Image display
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Scroll area for image
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setAlignment(Qt.AlignCenter)
        scroll_area.setStyleSheet("background-color: #2b2b2b;")
        
        # Image label
        self.image_label = ImageLabel()
        self.image_label.parent_app = self
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555;")
        
        scroll_area.setWidget(self.image_label)
        left_layout.addWidget(scroll_area)
        
        # Right panel - Controls and image list
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_panel.setMaximumWidth(350)
        right_panel.setMinimumWidth(250)
        
        # Buttons
        btn_select_folder = QPushButton("Select Folder")
        btn_select_folder.clicked.connect(self.select_folder)
        right_layout.addWidget(btn_select_folder)
        
        btn_reset_points = QPushButton("Reset Points (R)")
        btn_reset_points.clicked.connect(self.reset_points)
        right_layout.addWidget(btn_reset_points)
        
        btn_reset_zoom = QPushButton("Reset Zoom (Z)")
        btn_reset_zoom.clicked.connect(self.reset_zoom)
        right_layout.addWidget(btn_reset_zoom)
        
        # Image list
        list_label = QLabel("Images:")
        right_layout.addWidget(list_label)
        
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.on_list_item_clicked)
        right_layout.addWidget(self.image_list)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status()
    
    def keyPressEvent(self, event):
        """Handle keyboard events."""
        key = event.key()
        
        if key == Qt.Key_R:
            self.reset_points()
        elif key == Qt.Key_Z:
            self.reset_zoom()
        elif key == Qt.Key_Plus or key == Qt.Key_Equal:
            self.zoom_in()
        elif key == Qt.Key_Minus or key == Qt.Key_Underscore:
            self.zoom_out()
        elif key == Qt.Key_Left:
            self.previous_image()
        elif key == Qt.Key_Right:
            self.next_image()
    
    def select_folder(self):
        """Open folder dialog to select a directory."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder with Images"
        )
        
        if folder_path:
            self.load_images_from_folder(folder_path)
    
    def load_images_from_folder(self, folder_path):
        """Recursively load all image files from folder and subdirectories."""
        folder = Path(folder_path)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        self.image_paths = []
        for ext in image_extensions:
            self.image_paths.extend(folder.rglob(f'*{ext}'))
            self.image_paths.extend(folder.rglob(f'*{ext.upper()}'))
        
        # Remove duplicates and sort
        self.image_paths = sorted(list(set(self.image_paths)))
        
        print(f"Found {len(self.image_paths)} images in {folder_path}")
        
        # Populate list widget
        self.image_list.clear()
        for i, img_path in enumerate(self.image_paths):
            self.image_list.addItem(f"{i+1}. {img_path.name}")
        
        # Load first image
        if self.image_paths:
            self.current_image_index = 0
            self.image_list.setCurrentRow(0)
            self.load_image(str(self.image_paths[0]))
    
    def on_list_item_clicked(self, item):
        """Handle list item click."""
        index = self.image_list.row(item)
        if index != self.current_image_index:
            self.current_image_index = index
            self.load_image(str(self.image_paths[index]))
    
    def load_image(self, image_path):
        """Load and prepare image for SAM."""
        self.image = cv2.imread(image_path)
        if self.image is None:
            print(f"Could not load image: {image_path}")
            return
        
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image_display = self.image.copy()
        
        # Set image in predictor
        print(f"Processing image: {image_path}")
        self.predictor.set_image(self.image)
        
        # Reset state
        self.points = []
        self.labels = []
        self.mask = None
        self.zoom_level = 1.0
        
        print(f"Image loaded: {self.image.shape}")
        self.update_display()
    
    def screen_to_image_coords(self, x, y):
        """Convert screen coordinates to original image coordinates."""
        # Get display scale factor
        display_scale = getattr(self, 'display_scale', 1.0)
        
        # Account for display scale and zoom
        final_scale = display_scale * self.zoom_level
        img_x = int(x / final_scale)
        img_y = int(y / final_scale)
        
        # Clamp to image bounds
        if self.image is not None:
            img_x = max(0, min(img_x, self.image.shape[1] - 1))
            img_y = max(0, min(img_y, self.image.shape[0] - 1))
        
        return img_x, img_y
    
    def order_points(self, pts):
        """Order points in consistent order: top-left, top-right, bottom-right, bottom-left."""
        # Convert to float for calculations
        pts = pts.astype(np.float32)
        
        # Sort by y-coordinate
        sorted_by_y = pts[np.argsort(pts[:, 1])]
        
        # Top two points (smaller y)
        top_points = sorted_by_y[:2]
        # Bottom two points (larger y)
        bottom_points = sorted_by_y[2:]
        
        # Sort top points by x: left then right
        top_sorted = top_points[np.argsort(top_points[:, 0])]
        tl, tr = top_sorted[0], top_sorted[1]
        
        # Sort bottom points by x: left then right
        bottom_sorted = bottom_points[np.argsort(bottom_points[:, 0])]
        bl, br = bottom_sorted[0], bottom_sorted[1]
        
        # Return in order: top-left, top-right, bottom-right, bottom-left
        return np.array([tl, tr, br, bl], dtype=np.int32)
    
    def add_point(self, x, y, label):
        """Add a point and run prediction."""
        self.points.append([x, y])
        self.labels.append(label)
        point_type = "foreground" if label == 1 else "background"
        print(f"Added {point_type} point: ({x}, {y})")
        self.predict_and_display()
    
    def predict_and_display(self):
        """Run SAM prediction and update display."""
        if len(self.points) == 0:
            self.image_display = self.image.copy()
            self.update_display()
            return
        
        # Convert points to numpy array
        input_points = np.array(self.points)
        input_labels = np.array(self.labels)
        
        # Run prediction
        try:
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True if len(self.points) == 1 else False,
            )
            
            # Use best mask (highest score)
            if len(masks.shape) == 3:
                best_idx = np.argmax(scores)
                self.mask = masks[best_idx]
            else:
                self.mask = masks[0]
            
            print(f"Prediction complete. Score: {scores.max():.3f}")
            
        except Exception as e:
            print(f"Prediction error: {e}")
            self.mask = None
        
        self.update_display()
    
    def update_display(self):
        """Update display with mask overlay, points, quadbox, and zoom."""
        if self.image is None:
            return
        
        # Start with original image
        self.image_display = self.image.copy()
        
        # Calculate scale factor to fit screen
        screen = QApplication.primaryScreen().geometry()
        max_display_width = screen.width() - 450  # Reserve space for sidebar
        max_display_height = screen.height() - 200  # Reserve space for title bar and status
        
        h_orig, w_orig = self.image.shape[:2]
        display_scale = 1.0
        if w_orig > max_display_width or h_orig > max_display_height:
            display_scale = min(max_display_width / w_orig, max_display_height / h_orig)
        
        # Store display scale for coordinate transformation
        self.display_scale = display_scale
        
        # Overlay mask if available (BEFORE scaling)
        if self.mask is not None:
            # Create colored mask overlay
            mask_overlay = np.zeros_like(self.image_display, dtype=np.uint8)
            mask_overlay[self.mask] = self.mask_color
            
            # Blend with original image
            self.image_display = cv2.addWeighted(
                self.image_display,
                1.0,
                mask_overlay,
                self.mask_alpha,
                0
            )
            
            # Draw quadrilateral boxes around mask
            mask_uint8 = (self.mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) < 10:
                    continue
                
                # Get convex hull - this ensures all mask points are inside
                hull = cv2.convexHull(contour)
                
                # Simple and reliable: Approximate the hull to a quadrilateral
                # Douglas-Peucker algorithm will find the best 4-sided polygon
                epsilon = 0.015 * cv2.arcLength(hull, True)
                approx = None
                
                # Try increasing epsilon until we get exactly 4 points
                for attempt in range(20):
                    approx = cv2.approxPolyDP(hull, epsilon, True)
                    
                    if len(approx) == 4:
                        # Perfect! We have a quadrilateral
                        break
                    elif len(approx) < 4:
                        # Too few points, reduce epsilon
                        epsilon *= 0.8
                    else:
                        # Too many points, increase epsilon
                        epsilon *= 1.2
                
                # If approximation didn't give us 4 points, fall back to minAreaRect
                # which always gives exactly 4 corners
                if approx is None or len(approx) != 4:
                    rect = cv2.minAreaRect(hull)
                    box_points = cv2.boxPoints(rect)
                    box = box_points.astype(np.int32)
                else:
                    box = approx.reshape(-1, 2).astype(np.int32)
                
                # Order points consistently (top-left, top-right, bottom-right, bottom-left)
                # This ensures proper drawing
                box = self.order_points(box)
                
                # Draw quadrilateral box
                cv2.polylines(self.image_display, [box], True, (0, 255, 0), 2)
                
                # Draw corner points with numbers
                for i, corner in enumerate(box):
                    cv2.circle(self.image_display, tuple(corner), 5, (255, 0, 0), -1)
                    cv2.circle(self.image_display, tuple(corner), 6, (255, 255, 255), 1)
                    # Add corner number
                    cv2.putText(
                        self.image_display,
                        str(i+1),
                        (corner[0] + 8, corner[1] + 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1
                    )
                
                # Add label with area
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    area = int(cv2.contourArea(contour))
                    label_text = f"Area: {area}"
                    cv2.putText(
                        self.image_display,
                        label_text,
                        (cx - 30, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
        
        # Draw points on original image
        for point, label in zip(self.points, self.labels):
            color = self.fg_color if label == 1 else self.bg_color
            cv2.circle(self.image_display, tuple(point), self.point_radius, color, -1)
            cv2.circle(self.image_display, tuple(point), self.point_radius + 2, (255, 255, 255), 2)
        
        # Apply display scale first, then zoom
        if display_scale != 1.0 or self.zoom_level != 1.0:
            h, w = self.image_display.shape[:2]
            final_scale = display_scale * self.zoom_level
            new_w = int(w * final_scale)
            new_h = int(h * final_scale)
            self.image_display = cv2.resize(self.image_display, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Convert to QPixmap and display
        h, w, ch = self.image_display.shape
        bytes_per_line = ch * w
        qt_image = QImage(self.image_display.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)
        
        self.update_status()
    
    def update_status(self):
        """Update status bar."""
        image_info = f"Image {self.current_image_index + 1}/{len(self.image_paths)}" if self.image_paths else "No images"
        points_info = f"Points: {len(self.points)}"
        zoom_info = f"Zoom: {self.zoom_level:.2f}x"
        self.status_bar.showMessage(f"{image_info} | {points_info} | {zoom_info}")
    
    def reset_points(self):
        """Reset all points and mask."""
        self.points = []
        self.labels = []
        self.mask = None
        print("Points reset")
        self.update_display()
    
    def reset_zoom(self):
        """Reset zoom to 1.0x."""
        self.zoom_level = 1.0
        print("Zoom reset to 1.0x")
        self.update_display()
    
    def zoom_in(self):
        """Zoom in."""
        old_zoom = self.zoom_level
        self.zoom_level = min(self.zoom_max, self.zoom_level + self.zoom_step)
        if old_zoom != self.zoom_level:
            print(f"Zoom level: {self.zoom_level:.2f}x")
            self.update_display()
    
    def zoom_out(self):
        """Zoom out."""
        old_zoom = self.zoom_level
        self.zoom_level = max(self.zoom_min, self.zoom_level - self.zoom_step)
        if old_zoom != self.zoom_level:
            print(f"Zoom level: {self.zoom_level:.2f}x")
            self.update_display()
    
    def next_image(self):
        """Load next image in the list."""
        if not self.image_paths:
            return
        
        self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
        self.image_list.setCurrentRow(self.current_image_index)
        self.load_image(str(self.image_paths[self.current_image_index]))
        print(f"Image {self.current_image_index + 1}/{len(self.image_paths)}")
    
    def previous_image(self):
        """Load previous image in the list."""
        if not self.image_paths:
            return
        
        self.current_image_index = (self.current_image_index - 1) % len(self.image_paths)
        self.image_list.setCurrentRow(self.current_image_index)
        self.load_image(str(self.image_paths[self.current_image_index]))
        print(f"Image {self.current_image_index + 1}/{len(self.image_paths)}")


def main():
    """Main entry point."""
    # Configuration
    sam_checkpoint = "segment-anything-main/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Check if checkpoint exists
    if not Path(sam_checkpoint).exists():
        print(f"Error: SAM checkpoint not found at {sam_checkpoint}")
        print("Please download the checkpoint from:")
        print("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        return
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create and show main window
    window = SAMInteractiveApp(sam_checkpoint, model_type, device)
    window.show()
    
    # Run application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
    """Interactive SAM segmentation application with mouse-based point selection."""
    
    def __init__(self, sam_checkpoint="sam_vit_h_4b8939.pth", model_type="vit_h", device="cuda"):
        """Initialize SAM model and application state.
        
        Args:
            sam_checkpoint (str): Path to SAM checkpoint file
            model_type (str): SAM model type (vit_h, vit_l, vit_b)
            device (str): Device to run model on (cuda/cpu)
        """
        # Initialize SAM model
        print(f"Loading SAM model: {model_type}")
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        self.device = device
        
        # Application state
        self.image = None
        self.image_display = None
        self.points = []
        self.labels = []
        self.mask = None
        self.window_name = "SAM Interactive Segmentation"
        
        # Image list management
        self.image_paths = []
        self.current_image_index = 0
        self.sidebar_window = None
        self.image_listbox = None
        
        # Display settings
        self.point_radius = 5
        self.fg_color = (0, 255, 0)  # Green for foreground
        self.bg_color = (0, 0, 255)  # Red for background
        self.mask_color = (30, 144, 255)  # Blue mask overlay
        self.mask_alpha = 0.4
        
        # Zoom and pan settings
        self.zoom_level = 1.0
        self.zoom_min = 0.1
        self.zoom_max = 10.0
        self.zoom_step = 0.1
        
        print("SAM Interactive App initialized")
        print("Controls:")
        print("  Left Click      - Add foreground point")
        print("  Right Click     - Add background point")
        print("  Mouse Wheel     - Zoom in/out")
        print("  '+' / '-'       - Zoom in/out")
        print("  Left/Right Arrow- Previous/Next image")
        print("  'r' - Reset points")
        print("  'z' - Reset zoom")
        print("  'f' - Select new folder")
        print("  'q' - Quit")
    
    def select_folder(self):
        """Open folder dialog to select a directory."""
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        folder_path = filedialog.askdirectory(
            title="Select Folder with Images"
        )
        root.destroy()
        
        if folder_path:
            return folder_path
        return None
    
    def load_images_from_folder(self, folder_path):
        """Recursively load all image files from folder and subdirectories."""
        folder = Path(folder_path)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        self.image_paths = []
        for ext in image_extensions:
            # Case-insensitive search
            self.image_paths.extend(folder.rglob(f'*{ext}'))
            self.image_paths.extend(folder.rglob(f'*{ext.upper()}'))
        
        # Remove duplicates and sort
        self.image_paths = sorted(list(set(self.image_paths)))
        
        print(f"Found {len(self.image_paths)} images in {folder_path}")
        return len(self.image_paths) > 0
    
    def create_sidebar(self):
        """Create sidebar window with image list."""
        self.sidebar_window = Toplevel()
        self.sidebar_window.title("Image List")
        self.sidebar_window.geometry("300x600")
        
        # Create frame
        frame = Frame(self.sidebar_window)
        frame.pack(fill='both', expand=True)
        
        # Create scrollbar
        scrollbar = Scrollbar(frame, orient=VERTICAL)
        scrollbar.pack(side='right', fill='y')
        
        # Create listbox
        self.image_listbox = Listbox(
            frame,
            yscrollcommand=scrollbar.set,
            selectmode=SINGLE,
            font=('Arial', 10)
        )
        self.image_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.image_listbox.yview)
        
        # Populate listbox
        for i, img_path in enumerate(self.image_paths):
            display_name = f"{i+1}. {img_path.name}"
            self.image_listbox.insert(END, display_name)
        
        # Bind selection event
        self.image_listbox.bind('<<ListboxSelect>>', self.on_listbox_select)
        
        # Select first item
        if self.image_paths:
            self.image_listbox.select_set(0)
            self.image_listbox.activate(0)
    
    def on_listbox_select(self, event):
        """Handle listbox selection."""
        selection = self.image_listbox.curselection()
        if selection:
            index = selection[0]
            if index != self.current_image_index:
                self.current_image_index = index
                self.load_image(str(self.image_paths[index]))
                self.update_display()
    
    def update_sidebar_selection(self):
        """Update sidebar selection to match current image."""
        if self.image_listbox:
            self.image_listbox.selection_clear(0, END)
            self.image_listbox.select_set(self.current_image_index)
            self.image_listbox.activate(self.current_image_index)
            self.image_listbox.see(self.current_image_index)
    
    def next_image(self):
        """Load next image in the list."""
        if not self.image_paths:
            return
        
        self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
        self.load_image(str(self.image_paths[self.current_image_index]))
        self.update_sidebar_selection()
        self.update_display()
        print(f"Image {self.current_image_index + 1}/{len(self.image_paths)}")
    
    def previous_image(self):
        """Load previous image in the list."""
        if not self.image_paths:
            return
        
        self.current_image_index = (self.current_image_index - 1) % len(self.image_paths)
        self.load_image(str(self.image_paths[self.current_image_index]))
        self.update_sidebar_selection()
        self.update_display()
        print(f"Image {self.current_image_index + 1}/{len(self.image_paths)}")
    
    def load_image(self, image_path):
        """Load and prepare image for SAM."""
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image_display = self.image.copy()
        
        # Set image in predictor
        print(f"Processing image: {image_path}")
        self.predictor.set_image(self.image)
        
        # Reset state
        self.points = []
        self.labels = []
        self.mask = None
        self.zoom_level = 1.0
        
        print(f"Image loaded: {self.image.shape}")
    
    def screen_to_image_coords(self, x, y):
        """Convert screen coordinates to original image coordinates."""
        # Simply divide by zoom level since we're resizing the entire image
        img_x = int(x / self.zoom_level)
        img_y = int(y / self.zoom_level)
        
        # Clamp to image bounds
        img_x = max(0, min(img_x, self.image.shape[1] - 1))
        img_y = max(0, min(img_y, self.image.shape[0] - 1))
        
        return img_x, img_y
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for point selection and zoom."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click - foreground point
            img_x, img_y = self.screen_to_image_coords(x, y)
            self.points.append([img_x, img_y])
            self.labels.append(1)
            print(f"Added foreground point: ({img_x}, {img_y})")
            self.predict_and_display()
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click - background point
            img_x, img_y = self.screen_to_image_coords(x, y)
            self.points.append([img_x, img_y])
            self.labels.append(0)
            print(f"Added background point: ({img_x}, {img_y})")
            self.predict_and_display()
            
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Mouse wheel - zoom in/out
            old_zoom = self.zoom_level
            
            if flags > 0:  # Scroll up - zoom in
                self.zoom_level = min(self.zoom_max, self.zoom_level + self.zoom_step)
            else:  # Scroll down - zoom out
                self.zoom_level = max(self.zoom_min, self.zoom_level - self.zoom_step)
            
            if old_zoom != self.zoom_level:
                print(f"Zoom level: {self.zoom_level:.2f}x")
                self.update_display()
    
    def predict_and_display(self):
        """Run SAM prediction and update display."""
        if len(self.points) == 0:
            self.image_display = self.image.copy()
            self.update_display()
            return
        
        # Convert points to numpy array
        input_points = np.array(self.points)
        input_labels = np.array(self.labels)
        
        # Run prediction
        try:
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True if len(self.points) == 1 else False,
            )
            
            # Use best mask (highest score)
            if len(masks.shape) == 3:
                best_idx = np.argmax(scores)
                self.mask = masks[best_idx]
            else:
                self.mask = masks[0]
            
            print(f"Prediction complete. Score: {scores.max():.3f}")
            
        except Exception as e:
            print(f"Prediction error: {e}")
            self.mask = None
        
        self.update_display()
    
    def update_display(self):
        """Update display with mask overlay, points, and zoom."""
        # Start with original image
        self.image_display = self.image.copy()
        
        # Overlay mask if available
        if self.mask is not None:
            # Create colored mask overlay
            mask_overlay = np.zeros_like(self.image_display, dtype=np.uint8)
            mask_overlay[self.mask] = self.mask_color
            
            # Blend with original image
            self.image_display = cv2.addWeighted(
                self.image_display,
                1.0,
                mask_overlay,
                self.mask_alpha,
                0
            )
            
            # Draw quadrilateral boxes around mask
            # Convert mask to uint8 for contour detection
            mask_uint8 = (self.mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Skip very small contours
                if cv2.contourArea(contour) < 10:
                    continue
                
                # Get minimum area rotated rectangle (returns 4 corners)
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = box.astype(int)
                
                # Draw quadrilateral box
                cv2.drawContours(self.image_display, [box], 0, (0, 255, 0), 2)
                
                # Draw corner points
                for corner in box:
                    cv2.circle(self.image_display, tuple(corner), 4, (255, 0, 0), -1)
                
                # Add label with area and angle
                center, (width, height), angle = rect
                label_text = f"{int(width)}x{int(height)} {int(angle)}°"
                cv2.putText(
                    self.image_display,
                    label_text,
                    (int(center[0]), int(center[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
        
        # Draw points on original image
        for point, label in zip(self.points, self.labels):
            color = self.fg_color if label == 1 else self.bg_color
            cv2.circle(self.image_display, tuple(point), self.point_radius, color, -1)
            cv2.circle(self.image_display, tuple(point), self.point_radius + 2, (255, 255, 255), 2)
        
        # Apply zoom
        if self.zoom_level != 1.0:
            h, w = self.image_display.shape[:2]
            new_w = int(w * self.zoom_level)
            new_h = int(h * self.zoom_level)
            self.image_display = cv2.resize(self.image_display, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Convert back to BGR for display
        display_bgr = cv2.cvtColor(self.image_display, cv2.COLOR_RGB2BGR)
        
        # Add instruction text
        image_info = f"Image {self.current_image_index + 1}/{len(self.image_paths)}" if self.image_paths else "No folder"
        cv2.putText(
            display_bgr,
            f"{image_info} | Points: {len(self.points)} | Zoom: {self.zoom_level:.2f}x | Arrows: navigate, 'r': reset, 'z': zoom, 'f': folder, 'q': quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
        
        cv2.imshow(self.window_name, display_bgr)
    
    def reset_points(self):
        """Reset all points and mask."""
        self.points = []
        self.labels = []
        self.mask = None
        print("Points reset")
        self.update_display()
    
    def reset_zoom(self):
        """Reset zoom to 1.0x."""
        self.zoom_level = 1.0
        print("Zoom reset to 1.0x")
        self.update_display()
    
    def run(self):
        """Run the interactive application."""
        # Select folder with images
        folder_path = self.select_folder()
        if not folder_path:
            print("No folder selected. Exiting.")
            return
        
        # Load all images from folder
        if not self.load_images_from_folder(folder_path):
            print("No images found in folder. Exiting.")
            return
        
        # Create sidebar
        self.create_sidebar()
        
        # Load first image
        self.current_image_index = 0
        self.load_image(str(self.image_paths[0]))
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Initial display
        self.update_display()
        
        # Main loop
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('r'):
                self.reset_points()
            elif key == ord('z'):
                self.reset_zoom()
            elif key == ord('+') or key == ord('='):
                # Zoom in
                old_zoom = self.zoom_level
                self.zoom_level = min(self.zoom_max, self.zoom_level + self.zoom_step)
                if old_zoom != self.zoom_level:
                    print(f"Zoom level: {self.zoom_level:.2f}x")
                    self.update_display()
            elif key == ord('-') or key == ord('_'):
                # Zoom out
                old_zoom = self.zoom_level
                self.zoom_level = max(self.zoom_min, self.zoom_level - self.zoom_step)
                if old_zoom != self.zoom_level:
                    print(f"Zoom level: {self.zoom_level:.2f}x")
                    self.update_display()
            elif key == ord('f'):
                # Select new folder
                folder_path = self.select_folder()
                if folder_path and self.load_images_from_folder(folder_path):
                    self.current_image_index = 0
                    self.load_image(str(self.image_paths[0]))
                    # Recreate sidebar
                    if self.sidebar_window:
                        self.sidebar_window.destroy()
                    self.create_sidebar()
                    self.update_display()
            elif key == 83 or key == 3:  # Right arrow (key code may vary)
                self.next_image()
            elif key == 81 or key == 2:  # Left arrow (key code may vary)
                self.previous_image()
            
            # Update sidebar window
            if self.sidebar_window:
                try:
                    self.sidebar_window.update()
                except:
                    # Sidebar closed
                    self.sidebar_window = None
                    self.image_listbox = None
        
        # Cleanup
        if self.sidebar_window:
            self.sidebar_window.destroy()
        cv2.destroyAllWindows()


def main():
    """Main entry point."""
    # Configuration
    sam_checkpoint = "segment-anything-main/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Check if checkpoint exists
    if not Path(sam_checkpoint).exists():
        print(f"Error: SAM checkpoint not found at {sam_checkpoint}")
        print("Please download the checkpoint from:")
        print("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        return
    
    # Run application
    app = SAMInteractiveApp(sam_checkpoint, model_type, device)
    app.run()


if __name__ == "__main__":
    main()
