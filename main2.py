import sys
import os
import pandas as pd
import numpy as np
import cv2
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QLabel, QPushButton, QFileDialog,
                           QListWidget, QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox, QScrollArea)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor
from PIL import Image
import logging
import datetime
import math
from PIL import Image

# Create logs directory and set up logging
current_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(current_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

log_filename = f"debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file = os.path.join(logs_dir, log_filename)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)

print("\n" + "="*70, flush=True)
print("APPLICATION STARTING", flush=True)
print(f"Current directory: {current_dir}", flush=True)
print(f"Logs directory: {logs_dir}", flush=True)
print(f"Log file: {log_file}", flush=True)
print("Debug output will appear in this terminal and in the log file above", flush=True)
print("="*70 + "\n", flush=True)

# --- Helper functions for color histogram extraction and comparison ---

def extract_color_histogram(image_path, box):
    image = cv2.imread(image_path)
    x1, y1, x2, y2 = [int(v) for v in box]
    # Clamp coordinates to image size to avoid errors
    h, w = image.shape[:2]
    x1 = np.clip(x1, 0, w-1)
    x2 = np.clip(x2, 0, w-1)
    y1 = np.clip(y1, 0, h-1)
    y2 = np.clip(y2, 0, h-1)
    cropped = image[y1:y2, x1:x2]

    if cropped.size == 0:
        return np.zeros(8*8*4)
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 4], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def bhattacharyya_distance(hist1, hist2):
    return cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)

def compute_angle_and_distances(x1, x2, actual_depth, frame_center_x, frame_width_px=1280):
    box_center_x = (x1 + x2) / 2
    latitudinal_dist_px = box_center_x - frame_center_x  # in pixels

    # Calculate meters per pixel at current depth (actual_depth)
    meters_visible_width = 2 * actual_depth  # Based on 90° FOV assumption
    meters_per_pixel = meters_visible_width / frame_width_px

    latitudinal_dist_m = latitudinal_dist_px * meters_per_pixel

    if actual_depth == 0:
        angle_deg = 0.0
        longitudinal_dist = 0.0
    else:
        ratio = latitudinal_dist_m / actual_depth
        ratio = max(min(ratio, 1), -1)  # clamp ratio to [-1,1]
        angle_rad = math.asin(ratio)
        angle_deg = math.degrees(angle_rad)
        longitudinal_dist = math.cos(angle_rad) * actual_depth

    return angle_deg, latitudinal_dist_m, longitudinal_dist
# --New NotificationWidget class for displaying messages --

class NotificationWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                color: white;
                background-color: #2ecc71;
                border-radius: 10px;
                padding: 10px 20px;
                margin: 10px;
                font-size: 14px;
            }
        """)
        self.hide()
        
    def showMessage(self, message, duration=2000):
        self.setText(message)
        self.adjustSize()
        # Center at the top of the parent
        parent_rect = self.parent().rect()
        self.move(
            (parent_rect.width() - self.width()) // 2,
            10
        )
        self.show()
        # Auto-hide after duration
        QTimer.singleShot(duration, self.hide)

# --- VehicleTracker with enhanced matching logic ---

class VehicleTracker:
    def __init__(self, iou_threshold=0.5, max_frames_skip=5, coord_threshold=10):
        self.next_id = 1
        self.vehicle_tracks = {}  # Stores active tracks with associated info
        self.iou_threshold = iou_threshold
        self.max_frames_skip = max_frames_skip
        self.coord_threshold = coord_threshold

    def assign_ids(self, df: pd.DataFrame, frame_image_dir: str, threshold_match=50.0, threshold_distance=100) -> pd.DataFrame:
        df = df.sort_values('Frame')
        df['id'] = -1
        self.vehicle_tracks = {}
        self.next_id = 1
        prev_frame_data = None

        for Frame in df['Frame'].unique():
            frame_data = df[df['Frame'] == Frame]

            # Try to find image for histograms
            image_path = os.path.join(frame_image_dir, f"frame_{Frame}.png")
            if not os.path.exists(image_path):
                image_path = None
                for ext in ['.jpg', '.jpeg', '.bmp', '.gif', '.tiff']:
                    test_path = os.path.join(frame_image_dir, f"frame_{Frame}{ext}")
                    if os.path.exists(test_path):
                        image_path = test_path
                        break

            # Prepare current detections with histogram and centroid
            current_objects = []
            for idx, row in frame_data.iterrows():
                bbox = [row['x1'], row['y1'], row['x2'], row['y2']]
                centroid = ((row['x1'] + row['x2']) / 2, (row['y1'] + row['y2']) / 2)
                if image_path:
                    hist = extract_color_histogram(image_path, bbox)
                else:
                    hist = np.zeros(8*8*4)
                current_objects.append({
                    'index': idx,
                    'bbox': bbox,
                    'centroid': centroid,
                    'CLASS': str(row['CLASS']).strip().lower(),
                    'histogram': hist
                })

            assigned_detections = set()

            if prev_frame_data is not None:
                # Match current detections to existing tracks
                for obj in current_objects:
                    best_id = None
                    best_cost = float('inf')
                    for track_id, track_info in self.vehicle_tracks.items():
                        if obj['CLASS'] != track_info['CLASS']:
                            continue
                        dist = np.linalg.norm(np.array(obj['centroid']) - np.array(track_info['centroid']))
                        color_dist = bhattacharyya_distance(obj['histogram'], track_info['histogram'])
                        total_cost = dist + 20 * color_dist  # Weight color_dist more for importance
                        if total_cost < threshold_match and dist < threshold_distance:  # Threshold distance to limit unrealistic jumps
                            best_cost = total_cost
                            best_id = track_id
                    if best_id is not None and best_cost < 50:  # Threshold for valid match
                        df.at[obj['index'], 'id'] = best_id
                        self.vehicle_tracks[best_id] = {
                            'bbox': obj['bbox'],
                            'centroid': obj['centroid'],
                            'histogram': obj['histogram'],
                            'CLASS': obj['CLASS'],
                            'last_frame': Frame,
                            'frames_missing': 0
                        }
                        assigned_detections.add(obj['index'])

            # Assign new IDs for unmatched
            for obj in current_objects:
                if obj['index'] not in assigned_detections:
                    df.at[obj['index'], 'id'] = self.next_id
                    self.vehicle_tracks[self.next_id] = {
                        'bbox': obj['bbox'],
                        'centroid': obj['centroid'],
                        'histogram': obj['histogram'],
                        'CLASS': obj['CLASS'],
                        'last_frame': Frame,
                        'frames_missing': 0
                    }
                    self.next_id += 1

            # Update prev_frame_data
            prev_frame_data = df[df['Frame'] == Frame]

        return df

# --- GUI Classes (ErrorDialog, FrameViewer, MainWindow) ---

class ErrorDialog(QMessageBox):
    def __init__(self, title, message):
        super().__init__()
        self.setIcon(QMessageBox.Icon.Critical)
        self.setWindowTitle(title)
        self.setText(message)
        self.setStandardButtons(QMessageBox.StandardButton.Ok)
        logging.error(f"Error Dialog: {title} - {message}")

class FrameViewer(QLabel):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)
        self.boxes = []
        self.selected_id = None

    def sizeHint(self):
        if self.pixmap():
            return self.pixmap().size()
        return super().sizeHint()

    def set_image(self, image_path):
        if not os.path.exists(image_path):
            self.setText("Image not found!")
            return False

        try:
            with Image.open(image_path) as img:
                pixmap = QPixmap(image_path)
                self.setPixmap(pixmap)
                self.setFixedSize(pixmap.size())
                self.updateGeometry()
                return True
        except Exception as e:
            self.setText(f"Error loading image: {str(e)}")
            return False

    def set_boxes(self, boxes, selected_id=None):
        self.boxes = boxes
        self.selected_id = selected_id
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.pixmap() or not self.boxes:
            return

        painter = QPainter(self)
        painter.begin(self)

        # painter.fillRect(self.rect(), QColor(0, 0, 255, 127))

        # Draw all boxes except selected last
        for box in self.boxes:
            if box['id'] == self.selected_id:
                continue
            self.draw_box_and_info(painter, box, selected=False)

        for box in self.boxes:
            if box['id'] == self.selected_id:
                self.draw_box_and_info(painter, box, selected=True)
                break

        painter.end()

    def draw_box_and_info(self, painter, box, selected=False):
        x1, y1, x2, y2 = box['coords']
        object_id = box['id']
        actual_depth = box['actual_depth']
        predicted_depth = box['predicted_depth']
        CLASS = box['CLASS']
        UTC_time = box['UTC_time']

        # New fields (use default 0.0 if not present)
        angle = box.get('angle', 0.0)
        latitudinal_distance = box.get('latitudinal_distance', 0.0)
        longitudinal_distance = box.get('longitudinal_distance', 0.0)

        draw_x = int(x1)
        draw_y = int(y1)
        draw_width = int(x2 - x1)
        draw_height = int(y2 - y1)

        pen_color = QColor(255, 0, 0) if selected else QColor(0, 255, 0)
        pen = QPen(pen_color, 2)
        painter.setPen(pen)
        painter.drawRect(draw_x, draw_y, draw_width, draw_height)

        # Updated info text to include angle and distances
        text = (f"ID: {object_id}\n"
            f"Actual: {actual_depth:.2f} m  Pred: {predicted_depth:.2f} m\n"
            f"Angle: {angle:.2f}°\n"
            f"Latitude: {latitudinal_distance:.2f} m\n"      # <- corrected unit here
            f"Longitudinal: {longitudinal_distance:.2f} m\n" # <- added units here
            f"Type: {CLASS}\n"
            f"Time: {UTC_time}")


        text_rect = painter.boundingRect(
            draw_x + draw_width + 5, draw_y, 250, 120,  # Adjust height for extra lines
            Qt.TextFlag.TextWordWrap, text
        )

        painter.fillRect(text_rect, QColor(255, 255, 255, 180))
        painter.drawText(text_rect, Qt.TextFlag.TextWordWrap, text)

class MainWindow(QMainWindow):
    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vehicle Detection Viewer")
        self.setMinimumSize(1200, 800)

        self.data = None
        self.current_frame = None
        self.frames_dir = None
        self.frame_numbers = []
        self.vehicle_tracker = VehicleTracker()
        self.notification = NotificationWidget(self)
        self.setup_ui()

    def reassign_ids(self):
        if self.data is None:
            ErrorDialog("Error", "No data loaded to reassign IDs.").exec()
            return
        if self.frames_dir is None:
            ErrorDialog("Error", "Please select the Frames Directory first!").exec()
            return
        
        # Read parameters from GUI controls
        iou_threshold = self.iou_spinbox.value()
        max_frames_skip = self.max_skip_spinbox.value()
        coord_threshold = self.coord_thresh_spinbox.value()
        threshold_match = self.threshold_match_spinbox.value()
        threshold_distance = self.threshold_dist_spinbox.value()
        
        try:
            # Recreate tracker with current parameters
            tracker = VehicleTracker(
                iou_threshold=iou_threshold,
                max_frames_skip=max_frames_skip,
                coord_threshold=coord_threshold
            )
            # Re-assign IDs on current data using frame directory and thresholds
            self.data = tracker.assign_ids(
                self.data,
                self.frames_dir,
                threshold_match=threshold_match,
                threshold_distance=threshold_distance
            )
            self.vehicle_tracker = tracker
            self.update_frame_list()
            self.show_frame(self.current_frame)
            self.showNotification("IDs reassigned successfully!")
        except Exception as e:
            logging.error(f"Error reassigning IDs: {str(e)}")
            ErrorDialog("Error", f"Error reassigning IDs: {str(e)}").exec()

    def showNotification(self, message):
        self.notification.showMessage(message)

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar.setMaximumWidth(300)

        param_widget = QWidget()
        param_layout = QVBoxLayout(param_widget)

        # IoU Threshold input (float)
        self.iou_spinbox = QDoubleSpinBox()
        self.iou_spinbox.setRange(0.0, 1.0)
        self.iou_spinbox.setSingleStep(0.05)
        self.iou_spinbox.setValue(0.5)  # Default value
        param_layout.addWidget(QLabel("IoU Threshold"))
        param_layout.addWidget(self.iou_spinbox)

        # Max Frames Skip (int)
        self.max_skip_spinbox = QSpinBox()
        self.max_skip_spinbox.setRange(0, 30)
        self.max_skip_spinbox.setValue(5)  # Default
        param_layout.addWidget(QLabel("Max Frames Skip"))
        param_layout.addWidget(self.max_skip_spinbox)

        # Coordinate Threshold (int)
        self.coord_thresh_spinbox = QSpinBox()
        self.coord_thresh_spinbox.setRange(0, 100)
        self.coord_thresh_spinbox.setValue(10)  # Default
        param_layout.addWidget(QLabel("Coordinate Threshold"))
        param_layout.addWidget(self.coord_thresh_spinbox)

        # Threshold Match (float for total cost)
        self.threshold_match_spinbox = QDoubleSpinBox()
        self.threshold_match_spinbox.setRange(0.0, 200.0)
        self.threshold_match_spinbox.setSingleStep(1.0)
        self.threshold_match_spinbox.setValue(50.0)  # Default
        param_layout.addWidget(QLabel("Threshold Match"))
        param_layout.addWidget(self.threshold_match_spinbox)

        # Threshold Distance (int for spatial closeness)
        self.threshold_dist_spinbox = QSpinBox()
        self.threshold_dist_spinbox.setRange(0, 200)
        self.threshold_dist_spinbox.setValue(100)  # Default
        param_layout.addWidget(QLabel("Threshold Distance"))
        param_layout.addWidget(self.threshold_dist_spinbox)

        sidebar_layout.addWidget(param_widget)

        load_data_widget = QWidget()
        load_data_layout = QVBoxLayout(load_data_widget)

        self.file_format_combo = QComboBox()
        self.file_format_combo.addItems(["Excel (.xlsx, .xls)", "CSV (.csv)"])
        load_data_layout.addWidget(QLabel("Data File Format:"))
        load_data_layout.addWidget(self.file_format_combo)

        load_data_btn = QPushButton("Load Data File")
        load_data_btn.clicked.connect(self.load_data_file)
        load_data_layout.addWidget(load_data_btn)

        load_frames_btn = QPushButton("Select Frames Directory")
        load_frames_btn.clicked.connect(self.select_frames_directory)
        load_data_layout.addWidget(load_frames_btn)

        frame_nav = QWidget()
        frame_nav_layout = QVBoxLayout(frame_nav)

        frame_jump = QWidget()
        frame_jump_layout = QHBoxLayout(frame_jump)
        self.frame_jump_input = QSpinBox()
        self.frame_jump_input.setMinimum(0)
        self.frame_jump_input.setMaximum(999999)
        jump_btn = QPushButton("Jump to Frame")
        jump_btn.clicked.connect(self.jump_to_frame)

        frame_jump_layout.addWidget(QLabel("Frame:"))
        frame_jump_layout.addWidget(self.frame_jump_input)
        frame_jump_layout.addWidget(jump_btn)

        basic_nav = QWidget()
        basic_nav_layout = QHBoxLayout(basic_nav)
        self.prev_frame_btn = QPushButton("Previous")
        self.next_frame_btn = QPushButton("Next")
        self.frame_label = QLabel("Frame: None")

        basic_nav_layout.addWidget(self.prev_frame_btn)
        basic_nav_layout.addWidget(self.frame_label)
        basic_nav_layout.addWidget(self.next_frame_btn)

        frame_nav_layout.addWidget(frame_jump)
        frame_nav_layout.addWidget(basic_nav)

        self.prev_frame_btn.clicked.connect(self.show_previous_frame)
        self.next_frame_btn.clicked.connect(self.show_next_frame)

        self.id_list = QListWidget()
        self.id_list.itemClicked.connect(self.on_id_selected)

        sidebar_layout.addWidget(load_data_widget)
        sidebar_layout.addWidget(frame_nav)
        sidebar_layout.addWidget(QLabel("Object IDs:"))
        sidebar_layout.addWidget(self.id_list)

        # Add export button
        export_btn = QPushButton("Export Data with IDs")
        export_btn.clicked.connect(self.export_data)
        sidebar_layout.addWidget(export_btn)

        self.reassign_btn = QPushButton("Reassign IDs")
        sidebar_layout.addWidget(self.reassign_btn)   # Add to layout, not sidebar widget itself
        self.reassign_btn.clicked.connect(self.reassign_ids)


        self.frame_viewer = FrameViewer()

        # sidebar_scroll = QScrollArea()
        # sidebar_scroll.setWidgetResizable(True)
        # sidebar_scroll.setWidget(sidebar)

        self.frame_viewer = FrameViewer()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.frame_viewer)

        # Add scroll area (which wraps frame_viewer)
        layout.addWidget(scroll_area)

        # Add sidebar
        layout.addWidget(sidebar)

    def load_data_file(self):
        logging.info("\n" + "="*50)
        logging.info("ENTERING load_data_file")
        logging.info("="*50)

        if self.frames_dir is None:
            ErrorDialog("Error", "Please select the Frames Directory first!").exec()
            return

        file_format = self.file_format_combo.currentText()
        logging.info(f"Selected file format: {file_format}")

        if "Excel" in file_format:
            file_filter = "Excel Files (*.xlsx *.xls)"
        else:
            file_filter = "CSV Files (*.csv)"

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            "",
            file_filter
        )

        logging.info(f"Selected file path: {file_path}")

        if file_path:
            try:
                if file_path.endswith((".xlsx", ".xls")):
                    self.data = pd.read_excel(file_path)
                else:
                    self.data = pd.read_csv(file_path)

                # Frame number extraction and conversions...
                # for angles
                frame_width_px = 1280
                frame_center_x = frame_width_px / 2

                def apply_angle_lat_long(row):
                    return compute_angle_and_distances(
                        row['x1'], row['x2'], row['actual_depth'], frame_center_x, frame_width_px
                    )

                angles_lat_long = self.data.apply(
                    lambda row: pd.Series(apply_angle_lat_long(row)),
                    axis=1
                )

                angles_lat_long.columns = ['angle', 'latitudinal_distance', 'longitudinal_distance']
                self.data = pd.concat([self.data, angles_lat_long], axis=1)


                # ----- READ TUNABLE PARAMETERS FROM GUI CONTROLS -----
                iou_threshold = self.iou_spinbox.value()
                max_frames_skip = self.max_skip_spinbox.value()
                coord_threshold = self.coord_thresh_spinbox.value()
                threshold_match = self.threshold_match_spinbox.value()
                threshold_distance = self.threshold_dist_spinbox.value()

                # ----- CREATE TRACKER WITH TUNED PARAMETERS -----
                tracker = VehicleTracker(
                    iou_threshold=iou_threshold, 
                    max_frames_skip=max_frames_skip, 
                    coord_threshold=coord_threshold
                )

                # ----- ASSIGN IDS PASSING ADDITIONAL THRESHOLDS -----
                self.data = tracker.assign_ids(
                    self.data,
                    self.frames_dir,
                    threshold_match=threshold_match,
                    threshold_distance=threshold_distance
                )

                self.vehicle_tracker = tracker

                self.update_frame_list()
                logging.info("Finished loading data successfully!")
                logging.info("="*50)
                self.showNotification("Data file loaded successfully!")
            except Exception as e:
                logging.error(f"Error loading data: {str(e)}")
                ErrorDialog("Error Loading Data", str(e)).exec()

    def select_frames_directory(self):
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Frames Directory"
        )
        if dir_path:
            self.frames_dir = dir_path
            self.showNotification("Frames directory loaded successfully!")
            if self.current_frame:
                self.show_frame(self.current_frame)

    def find_frame_file(self, Frame):
        if not self.frames_dir:
            return None

        # Default naming
        file_path = os.path.join(self.frames_dir, f"frame_{Frame}.png")
        if os.path.exists(file_path):
            return file_path

        for ext in self.SUPPORTED_IMAGE_FORMATS:
            file_path = os.path.join(self.frames_dir, f"frame_{Frame}{ext}")
            if os.path.exists(file_path):
                return file_path

        dir_contents = os.listdir(self.frames_dir)
        frame_str = f"frame_{Frame}"
        for filename in dir_contents:
            name, ext = os.path.splitext(filename.lower())
            if name == frame_str.lower() and ext in [x.lower() for x in self.SUPPORTED_IMAGE_FORMATS]:
                return os.path.join(self.frames_dir, filename)

        return None

    def update_frame_list(self):
        if self.data is not None:
            try:
                self.frame_numbers = [int(float(x)) for x in self.data['Frame'].unique()]
                self.frame_numbers.sort()

                if self.frame_numbers:
                    self.current_frame = self.frame_numbers[0]
                    max_frame = int(self.frame_numbers[-1])
                    self.frame_jump_input.setMaximum(max_frame)
                    self.show_frame(self.current_frame)
            except Exception as e:
                logging.error(f"Error in update_frame_list: {str(e)}")
                ErrorDialog("Error", f"Error in update_frame_list: {str(e)}").exec()

    def show_frame(self, Frame):
        if self.frames_dir is None or self.data is None:
            return

        try:
            Frame = int(Frame)
            self.current_frame = Frame
            self.frame_label.setText(f"Frame: {Frame}")
            self.frame_jump_input.setValue(Frame)

            image_path = self.find_frame_file(Frame)
            if not image_path:
                raise FileNotFoundError(f"No supported image file found for frame {Frame}")

            if not self.frame_viewer.set_image(image_path):
                return

            frame_data = self.data[self.data['Frame'] == Frame]

            self.id_list.clear()
            unique_ids = frame_data['id'].unique()
            self.id_list.addItems([str(id_) for id_ in sorted(unique_ids)])

            boxes = []
            for _, row in frame_data.iterrows():
                box_info = {
                    'id': int(row['id']),
                    'coords': [float(row['x1']), float(row['y1']),
                            float(row['x2']), float(row['y2'])],
                    'actual_depth': float(row['actual_depth']),
                    'predicted_depth': float(row['predicted_depth']),
                    'CLASS': str(row['CLASS']),
                    'UTC_time': str(row['UTC_time']),
                    'angle': float(row.get('angle', 0.0)),   # <-- add these three
                    'latitudinal_distance': float(row.get('latitudinal_distance', 0.0)),
                    'longitudinal_distance': float(row.get('longitudinal_distance', 0.0)),
                }
                boxes.append(box_info)
            self.frame_viewer.set_boxes(boxes)

        except Exception as e:
            logging.error(f"Error in show_frame: {str(e)}")
            ErrorDialog("Error", f"Error in show_frame: {str(e)}").exec()

    def jump_to_frame(self):
        target_frame = self.frame_jump_input.value()
        if target_frame in self.frame_numbers:
            self.show_frame(target_frame)
        else:
            ErrorDialog("Error", f"Frame {target_frame} not found in data").exec()

    def show_previous_frame(self):
        if self.current_frame is None:
            return
        current_idx = self.frame_numbers.index(self.current_frame)
        if current_idx > 0:
            self.show_frame(self.frame_numbers[current_idx - 1])

    def show_next_frame(self):
        if self.current_frame is None:
            return
        current_idx = self.frame_numbers.index(self.current_frame)
        if current_idx < len(self.frame_numbers) - 1:
            self.show_frame(self.frame_numbers[current_idx + 1])

    def on_id_selected(self, item):
        selected_id = int(item.text())
        self.frame_viewer.set_boxes(self.frame_viewer.boxes, selected_id)

    def export_data(self):
        if self.data is None:
            ErrorDialog("Error", "No data loaded to export").exec()
            return

        import numpy as np
        import pandas as pd

        try:
            # Ask user where to save file
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Data with IDs",
                "",
                "Excel Files (*.xlsx);;CSV Files (*.csv)"
            )

            if not file_path:
                return

            # ----- 1. Sort by id and Frame -----
            df = self.data.sort_values(['id', 'Frame']).copy()

            # ----- 2. Segment / minima detection for each vehicle -----
            def find_minima_segments(distances):
                """Return segment labels & minima indices for a series of distances."""
                if len(distances) < 3:
                    return np.zeros(len(distances), dtype=int), []
                minima_indices = (np.diff(np.sign(np.diff(distances))) > 0).nonzero()[0] + 1
                segment_ids = np.zeros(len(distances), dtype=int)
                start = 0
                for seg_idx, min_idx in enumerate(minima_indices):
                    segment_ids[start:min_idx+1] = seg_idx
                    start = min_idx + 1
                segment_ids[start:] = len(minima_indices)
                return segment_ids + 1, minima_indices  # make segment IDs start from 1

            segment_labels = []
            minima_flags = []

            for vid, group in df.groupby('id', sort=False):
                distances = group['actual_depth'].values
                seg_ids, minima_indices = find_minima_segments(distances)

                segment_labels.extend(seg_ids)

                is_minima = np.zeros(len(distances), dtype=int)
                is_minima[minima_indices] = 1
                minima_flags.extend(is_minima)

            # Add new columns
            df['segment'] = segment_labels
            df['is_minima'] = minima_flags

            # ----- 3. Export according to file type -----
            if file_path.endswith('.xlsx'):
                df.to_excel(file_path, index=False)
            else:
                df.to_csv(file_path, index=False)

            # Show success
            QMessageBox.information(
                self,
                "Success",
                f"Data successfully exported to:\n{file_path}"
            )

        except Exception as e:
            logging.error(f"Error exporting data: {str(e)}")
            ErrorDialog("Export Error", f"Failed to export data: {str(e)}").exec()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
