import os
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QRadioButton, QLabel, QLineEdit, QFrame, QPushButton,
    QGridLayout, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QButtonGroup, QMessageBox,
    QScrollArea
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QIntValidator
from PyQt5.QtCore import Qt, QObject, QEvent, QPoint, QRect
import numpy as np
import yaml
from functools import partial
import traceback

from file_functions import (loadFolder, saveImages)
from image_functions import (
    translate_image,
    apply_text_rotation, rotate_image_incremental,
    morphImages,
    registerImages, 
)

# Validator for non-negative integers
class NonNegativeIntValidator(QIntValidator):
    def __init__(self, parent=None):
        super().__init__(0, 2147483647, parent) # Min 0, Max is default max int

    def validate(self, input_str, pos):
        # Allow empty string during editing
        if not input_str:
            return (QIntValidator.Intermediate, input_str, pos)
        return super().validate(input_str, pos)

class MainWindow(QMainWindow):
    CONFIG_FILE = 'config.yaml'

    def __init__(self):
        super().__init__()
        self.anchor_rect_img_coords = None
        self.zoom_factor = 1.0
        self.image_data = []

        # --- Mouse Drag State ---
        self.selection_start_point = None
        self.selection_end_point = None
        self.is_selecting = False

        # Cached pixmaps
        self._ref_base_pixmap = None
        self._current_base_pixmap = None
        self._ref_anchor_pixmap = None # Ref pixmap with anchor drawn
        self._current_diff_pixmap = None # Diff view pixmap

        # Config related attributes
        self.last_load_folder = None
        self.last_save_folder = None
        self.last_morph_folder = None
        self.load_config() # Load config early

        self.initUI()
        # Initialize indices after UI potentially ready
        self.ref_image_idx = -1 # Initialize invalid until images loaded
        self.current_image_idx = -1

    # --- Config Handling ---
    def load_config(self):
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    config = yaml.safe_load(f)
                    if config: # Check if config is not None or empty
                        self.last_load_folder = config.get('last_load_folder', None)
                        self.last_save_folder = config.get('last_save_folder', None)
                        self.last_morph_folder = config.get('last_morph_folder', None)
                        print("Loaded config:", config)
            except Exception as e:
                print(f"Error reading config file '{self.CONFIG_FILE}': {e}")

    def save_config(self):
        config_data = {
            'last_load_folder': self.last_load_folder,
            'last_save_folder': self.last_save_folder,
            'last_morph_folder': self.last_morph_folder,
            # Add other settings here if needed later
        }
        try:
            with open(self.CONFIG_FILE, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            # print("Config saved.") # Optional confirmation
        except Exception as e:
            print(f"Error writing config file '{self.CONFIG_FILE}': {e}")

    # --- UI Init Methods ---
    def keyPressEvent(self, event):
        handled = False
        if self.image_list.hasFocus():
            current_row = self.image_list.currentRow()
            key = event.key()
            if key == Qt.Key_Up and current_row > 0:
                self.image_list.setCurrentRow(current_row - 1)
                handled = True
            elif key == Qt.Key_Down and current_row < self.image_list.count() - 1:
                self.image_list.setCurrentRow(current_row + 1)
                handled = True
        # Check if handled before passing up
        if not handled:
            super().keyPressEvent(event)

    def initUI(self):
        central_widget = QFrame(self)
        central_widget.setMinimumWidth(1100)
        central_widget.setMinimumHeight(800)
        self.layout = QGridLayout(central_widget)
        self.layout.setContentsMargins(5, 5, 5, 5)

        self.init_image_frames()
        self.init_control_frame()

        self.setCentralWidget(central_widget)
        self.setWindowTitle("Image Registration Tool")

    def init_image_frames(self):
        image_area_frame = QFrame(self)
        image_area_layout = QHBoxLayout(image_area_frame)
        image_area_layout.setContentsMargins(0, 0, 0, 0)

        image_display_frame = QFrame(self)
        image_display_layout = QGridLayout(image_display_frame)
        image_display_layout.setContentsMargins(0, 0, 0, 0)

        text_ref = QLabel('Reference Image', self)
        text_ref.setAlignment(Qt.AlignCenter)
        text_ref.setMaximumHeight(20)
        text_ref.setStyleSheet("font-weight: bold;")
        image_display_layout.addWidget(text_ref, 0, 0)

        text_current = QLabel('Current Image', self)
        text_current.setAlignment(Qt.AlignCenter)
        text_current.setMaximumHeight(20)
        text_current.setStyleSheet("font-weight: bold;")
        image_display_layout.addWidget(text_current, 0, 1)

        self.ref_image = QLabel(self)
        self.ref_image.setAlignment(Qt.AlignCenter)
        self.ref_image.setStyleSheet("background-color: #202020;")
        self.ref_image.setMouseTracking(True) # Enable mouse move events
        self.ref_image.installEventFilter(self)

        self.current_image = QLabel(self)
        self.current_image.setAlignment(Qt.AlignCenter)
        self.current_image.setStyleSheet("background-color: #202020;")

        self.ref_scroll_area = QScrollArea(self)
        self.ref_scroll_area.setWidgetResizable(False)
        self.ref_scroll_area.setWidget(self.ref_image)
        self.ref_scroll_area.setMinimumSize(350, 350)
        image_display_layout.addWidget(self.ref_scroll_area, 1, 0)

        self.current_scroll_area = QScrollArea(self)
        self.current_scroll_area.setWidgetResizable(False)
        self.current_scroll_area.setWidget(self.current_image)
        self.current_scroll_area.setMinimumSize(350, 350)
        image_display_layout.addWidget(self.current_scroll_area, 1, 1)

        image_controls_frame = QFrame(self)
        image_controls_layout = QHBoxLayout(image_controls_frame)
        image_controls_layout.setContentsMargins(0, 5, 0, 0)

        btn_zoom_out = QPushButton("-", self)
        btn_zoom_out.setToolTip("Zoom Out")
        btn_zoom_out.setMaximumWidth(30)
        btn_zoom_out.clicked.connect(self.zoom_out)

        btn_zoom_in = QPushButton("+", self)
        btn_zoom_in.setToolTip("Zoom In")
        btn_zoom_in.setMaximumWidth(30)
        btn_zoom_in.clicked.connect(self.zoom_in)

        btn_zoom_fit = QPushButton("Fit", self)
        btn_zoom_fit.setToolTip("Fit image to view")
        btn_zoom_fit.clicked.connect(self.fit_view)

        btn_zoom_100 = QPushButton("100%", self)
        btn_zoom_100.setToolTip("Zoom to 100%")
        btn_zoom_100.clicked.connect(self.zoom_100)

        image_controls_layout.addStretch(1)
        image_controls_layout.addWidget(btn_zoom_out)
        image_controls_layout.addWidget(btn_zoom_in)
        image_controls_layout.addWidget(btn_zoom_fit)
        image_controls_layout.addWidget(btn_zoom_100)

        self.radio_buttons = {'rad_normal': QRadioButton("Original"), 'rad_diff': QRadioButton("Difference")}
        display_radio_group = QButtonGroup(self)
        for button in self.radio_buttons.values():
            display_radio_group.addButton(button)
            image_controls_layout.addWidget(button)
            button.toggled.connect(self.updatePixmap)
        self.radio_buttons['rad_normal'].setChecked(True)
        image_controls_layout.addStretch(1)

        image_display_layout.addWidget(image_controls_frame, 2, 0, 1, 2)

        list_frame = QFrame(self)
        list_layout = QGridLayout(list_frame)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_label = QLabel("Images (Check = Reference)")
        list_label.setMaximumHeight(20)
        list_label.setStyleSheet("font-weight: bold;")
        self.image_list = QListWidget(self)
        self.image_list.setFixedWidth(220)
        self.image_list.itemClicked.connect(self.item_changed)
        self.image_list.currentItemChanged.connect(self.current_item_changed_slot)
        self.image_list.itemChanged.connect(self.check_state_changed)
        list_layout.addWidget(list_label, 0, 0, 1, 2)
        list_layout.addWidget(self.image_list, 1, 0, 1, 2)

        btn_open = QPushButton('Open folder', self)
        btn_open.setToolTip("Select image folder.")
        btn_open.clicked.connect(partial(loadFolder, self))
        list_layout.addWidget(btn_open, 2, 0, 1, 2)

        btn_save_all = QPushButton('Save all', self)
        btn_save_all.setToolTip("Save all images.")
        btn_save_all.clicked.connect(partial(saveImages, self))
        list_layout.addWidget(btn_save_all, 3, 0)

        btn_save_current = QPushButton('Save current', self)
        btn_save_current.setToolTip("Save current image.")
        btn_save_current.clicked.connect(partial(saveImages, self))
        list_layout.addWidget(btn_save_current, 3, 1)

        image_area_layout.addWidget(image_display_frame, 1)
        image_area_layout.addWidget(list_frame)

        self.layout.addWidget(image_area_frame, 1, 0)

    def init_control_frame(self):
        control_frame = QFrame(self)
        control_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        control_layout = QGridLayout(control_frame)
        control_layout.setContentsMargins(5, 5, 5, 5)

        #radio_layout.addStretch(1)
        #control_layout.addWidget(radio_frame)

        # Registration Settings
        reg_settings_frame = QFrame(self)
        reg_settings_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        reg_settings_layout = QGridLayout(reg_settings_frame)
        reg_settings_layout.setContentsMargins(5, 5, 5, 5)
        reg_title = QLabel("Image Registration:")
        reg_title.setStyleSheet("font-weight: bold;")
        reg_settings_layout.addWidget(reg_title, 0, 0, 1, 2)
        self.reg_method_fft_radio = QRadioButton("FFT (Fast, Subpixel)")
        self.reg_method_fft_radio.setToolTip("Uses image_registration.chi2_shift.\nAnchor area is ignored.")
        self.reg_method_scan_radio = QRadioButton("Scan SSD (Anchor Required)")
        self.reg_method_scan_radio.setToolTip("Scans around anchor for minimum difference.\nRequires anchor. Integer shifts only.")
        self.reg_method_group = QButtonGroup(self)
        self.reg_method_group.addButton(self.reg_method_fft_radio)
        self.reg_method_group.addButton(self.reg_method_scan_radio)
        reg_settings_layout.addWidget(self.reg_method_fft_radio, 1, 0, 1, 2)
        reg_settings_layout.addWidget(self.reg_method_scan_radio, 2, 0, 1, 2)
        self.reg_method_fft_radio.setChecked(True) # Default to FFT
        
        btn_register_all = QPushButton('Register all', self)
        btn_register_all.setToolTip("Register all to reference.")
        btn_register_all.clicked.connect(partial(registerImages, self))
        
        reg_settings_layout.addWidget(btn_register_all, 3, 1)
        
        btn_register_current = QPushButton('Register current', self)
        btn_register_current.setToolTip("Register current to reference.")
        btn_register_current.clicked.connect(partial(registerImages, self))
        
        reg_settings_layout.addWidget(btn_register_current, 3, 0)
        
        control_layout.addWidget(reg_settings_frame, 0, 0)

        # Anchor Area Input
        anchor_frame = QFrame(self)
        anchor_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        anchor_layout = QGridLayout(anchor_frame)
        anchor_layout.setContentsMargins(5, 5, 5, 5)
        anchor_title = QLabel("Anchor Area (Pixels)")
        anchor_title.setStyleSheet("font-weight: bold;")
        anchor_layout.addWidget(anchor_title, 0, 0, 1, 4)
        self.anchor_x = QLineEdit("0", self)
        self.anchor_y = QLineEdit("0", self)
        self.anchor_w = QLineEdit("100", self)
        self.anchor_h = QLineEdit("100", self)
        validator = NonNegativeIntValidator(self)
        self.anchor_x.setValidator(validator)
        self.anchor_y.setValidator(validator)
        self.anchor_w.setValidator(validator)
        self.anchor_h.setValidator(validator)
        input_width = 50
        self.anchor_x.setMaximumWidth(input_width)
        self.anchor_y.setMaximumWidth(input_width)
        self.anchor_w.setMaximumWidth(input_width)
        self.anchor_h.setMaximumWidth(input_width)
        anchor_layout.addWidget(QLabel("X:"), 1, 0)
        anchor_layout.addWidget(self.anchor_x, 1, 1)
        anchor_layout.addWidget(QLabel("Y:"), 2, 0)
        anchor_layout.addWidget(self.anchor_y, 2, 1)
        anchor_layout.addWidget(QLabel("W:"), 1, 2)
        anchor_layout.addWidget(self.anchor_w, 1, 3)
        anchor_layout.addWidget(QLabel("H:"), 2, 2)
        anchor_layout.addWidget(self.anchor_h, 2, 3)
        self.btn_apply_anchor = QPushButton("Apply Anchor", self)
        self.btn_apply_anchor.setToolTip("Validate and apply anchor values.")
        self.btn_apply_anchor.clicked.connect(self.apply_anchor_from_inputs)
        anchor_layout.addWidget(self.btn_apply_anchor, 3, 0, 1, 2)
        self.btn_clear_anchor = QPushButton("Clear Anchor", self)
        self.btn_clear_anchor.setToolTip("Remove anchor definition.")
        self.btn_clear_anchor.clicked.connect(self.clear_anchor_area)
        self.btn_clear_anchor.setEnabled(False)
        anchor_layout.addWidget(self.btn_clear_anchor, 3, 2, 1, 2)
        control_layout.addWidget(anchor_frame, 0, 1)

        # Manual Adjustment
        manual_adj_frame = QFrame(self)
        manual_adj_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        manual_adj_layout = QGridLayout(manual_adj_frame)
        manual_adj_layout.setContentsMargins(5,0,5,0)
        trans_label = QLabel("Manual Translate:")
        trans_label.setStyleSheet("font-weight: bold;")
        manual_adj_layout.addWidget(trans_label, 0, 0, 1, 4)
        btn_left = QPushButton('← Left', self)
        btn_right = QPushButton('Right →', self)
        btn_up = QPushButton('↑ Up', self)
        btn_down = QPushButton('↓ Down', self)
        btn_left.clicked.connect(partial(translate_image, self, 'Left'))
        btn_right.clicked.connect(partial(translate_image, self, 'Right'))
        btn_up.clicked.connect(partial(translate_image, self, 'Up'))
        btn_down.clicked.connect(partial(translate_image, self, 'Down'))
        manual_adj_layout.addWidget(btn_up, 1, 1)
        manual_adj_layout.addWidget(btn_left, 2, 0)
        manual_adj_layout.addWidget(btn_down, 2, 1)
        manual_adj_layout.addWidget(btn_right, 2, 2)
        shift_txt = QLabel('Shift [px]:')
        self.shift_val = QLineEdit('1', self)
        self.shift_val.setMaximumWidth(40)
        self.shift_val.setValidator(NonNegativeIntValidator(self))
        self.shift_val.editingFinished.connect(partial(self.check_input_is_int, self.shift_val, 1, min_val=1))
        manual_adj_layout.addWidget(shift_txt, 3, 0, 1, 2)
        manual_adj_layout.addWidget(self.shift_val, 3, 2, 1, 1)
        rot_label = QLabel("Manual Rotate:")
        rot_label.setStyleSheet("font-weight: bold;")
        manual_adj_layout.addWidget(rot_label, 0, 4, 1, 4)
        btn_rot_l = QPushButton('↺ CCW', self)
        btn_rot_r = QPushButton('CW ↻', self)
        btn_rot_l.clicked.connect(partial(rotate_image_incremental, self, 'Left'))
        btn_rot_r.clicked.connect(partial(rotate_image_incremental, self, 'Right'))
        manual_adj_layout.addWidget(btn_rot_l, 1, 4)
        manual_adj_layout.addWidget(btn_rot_r, 1, 5)
        rot_txt = QLabel('Total Rot [°]:')
        self.rot_val = QLineEdit('0.0', self)
        self.rot_val.setMaximumWidth(50)
        self.rot_val.editingFinished.connect(partial(self.check_input_is_float, self.rot_val, 0.0))
        manual_adj_layout.addWidget(rot_txt, 2, 4)
        manual_adj_layout.addWidget(self.rot_val, 2, 5)
        self.btn_apply_rotation = QPushButton("Apply Rot", self)
        self.btn_apply_rotation.setToolTip("Apply rotation in text box")
        self.btn_apply_rotation.clicked.connect(partial(apply_text_rotation, self))
        manual_adj_layout.addWidget(self.btn_apply_rotation, 2, 6)
        manual_adj_layout.setColumnStretch(7, 1)
        control_layout.addWidget(manual_adj_frame, 0, 2)

        # Morph images
        morph_frame = QFrame(self)
        morph_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        morph_layout = QGridLayout(morph_frame)
        morph_layout.setContentsMargins(5,0,5,0)
        morph_label = QLabel("Morphing:")
        morph_label.setStyleSheet("font-weight: bold;")
        morph_layout.addWidget(morph_label, 0, 0, 1, 2)
        btn_morph = QPushButton('Morph images', self)
        btn_morph.setToolTip("Create morph sequence.")
        btn_morph.clicked.connect(partial(morphImages, self))
        morph_layout.addWidget(btn_morph, 1, 0, 1, 1)

        text_fps = QLabel('Morph FPS:', self)
        text_fps.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        morph_layout.addWidget(text_fps, 2, 0, 1, 1)

        self.fps = QLineEdit('5', self)
        self.fps.setToolTip("Morph FPS (min 2).")
        self.fps.setValidator(NonNegativeIntValidator(self))
        self.fps.editingFinished.connect(partial(self.check_input_is_int, self.fps, 5, min_val=2))
        self.fps.setMaximumWidth(50)
        morph_layout.addWidget(self.fps, 3, 0, 1, 1)
        control_layout.addWidget(morph_frame, 0, 3)

        self.layout.addWidget(control_frame, 2, 0)

    def _update_rotation_textbox(self):
        """Updates the rotation text box based on the current image's stored rotation."""
        if 0 <= self.current_image_idx < len(self.image_data):
             current_rotation = self.image_data[self.current_image_idx]['total_rotation']
             self.rot_val.setText(f"{current_rotation:.1f}")
        else:
             self.rot_val.setText("0.0")

    # --- Zoom Slots ---
    def zoom_in(self):
        self.set_zoom(self.zoom_factor * 1.25)

    def zoom_out(self):
        self.set_zoom(self.zoom_factor / 1.25)

    def zoom_100(self):
        self.set_zoom(1.0)

    def fit_view(self):
        if not self._ref_base_pixmap or self._ref_base_pixmap.isNull():
            self.set_zoom(1.0)
            return

        vp_size = self.ref_scroll_area.viewport().size()
        # Add small margin to viewport size for fitting to avoid potential scrollbars appearing
        margin = 2 # pixels
        vp_width = max(1, vp_size.width() - margin)
        vp_height = max(1, vp_size.height() - margin)

        if vp_width <= 0 or vp_height <= 0:
            self.set_zoom(1.0)
            return

        pix_size = self._ref_base_pixmap.size()
        if pix_size.width() <= 0 or pix_size.height() <= 0:
             self.set_zoom(1.0)
             return

        w_scale = vp_width / pix_size.width()
        h_scale = vp_height / pix_size.height()
        fit_factor = min(w_scale, h_scale)

        self.set_zoom(fit_factor)

    def set_zoom(self, factor):
        min_zoom = 0.01
        max_zoom = 16.0
        new_factor = max(min_zoom, min(max_zoom, factor))
        if abs(new_factor - self.zoom_factor) > 1e-6:
            self.zoom_factor = new_factor
            print(f"Setting zoom factor to: {self.zoom_factor:.3f}")
            self.updatePixmap(update_base=False)

    # --- Coordinate Mapping for Mouse Drag ---
    def map_label_point_to_image_point(self, label_point):
        """Maps a QPoint from QLabel coordinates to original image coordinates."""
        if not self._ref_base_pixmap or self._ref_base_pixmap.isNull() or self.zoom_factor <= 1e-9 or label_point is None:
            print("Cannot map point: Missing data or zero zoom")
            return None # Cannot map if pixmap/zoom invalid

        original_size = self._ref_base_pixmap.size()
        img_w = original_size.width()
        img_h = original_size.height()

        if img_w <= 0 or img_h <= 0:
            print("Cannot map point: Invalid original image size")
            return None

        # Coordinates from event.pos() are relative to the QLabel top-left
        label_x = label_point.x()
        label_y = label_point.y()

        # Map back using zoom factor
        img_x = label_x / self.zoom_factor
        img_y = label_y / self.zoom_factor

        # Clamp to image bounds [0, img_dim]
        img_x = max(0.0, min(img_w, img_x))
        img_y = max(0.0, min(img_h, img_y))

        return QPoint(int(round(img_x)), int(round(img_y)))

    # --- Anchor Handling ---
    def _update_anchor_textboxes(self, img_rect):
        """Updates anchor text boxes based on a QRect in image coordinates."""
        if img_rect and isinstance(img_rect, QRect):
            self.anchor_x.setText(str(img_rect.x()))
            self.anchor_y.setText(str(img_rect.y()))
            self.anchor_w.setText(str(img_rect.width()))
            self.anchor_h.setText(str(img_rect.height()))
        else: # Clear if invalid rect passed
            self.anchor_x.setText("0")
            self.anchor_y.setText("0")
            self.anchor_w.setText("100")
            self.anchor_h.setText("100")

    def apply_anchor_from_inputs(self):
        if not hasattr(self, 'images') or not self.images: QMessageBox.warning(self, "No Image", "Load images first."); return
        if not (0 <= self.ref_image_idx < len(self.images)): QMessageBox.warning(self, "Invalid Reference", "Cannot get ref dims."); return
        try:
            img_h, img_w = self.images[self.ref_image_idx].shape[:2]
            x0 = int(self.anchor_x.text()); y0 = int(self.anchor_y.text()); w = int(self.anchor_w.text()); h = int(self.anchor_h.text())
            if w <= 0 or h <= 0: raise ValueError("W/H > 0");
            if x0 < 0 or y0 < 0: raise ValueError("X/Y >= 0");
            if x0 + w > img_w or y0 + h > img_h: raise ValueError("Anchor outside bounds")
            self.anchor_rect_img_coords = QRect(x0, y0, w, h)
            print(f"Anchor area applied from text: {self.anchor_rect_img_coords}")
            self.btn_clear_anchor.setEnabled(True)
            self.is_selecting = False # Ensure dragging state is off
            self.selection_start_point = None
            self.selection_end_point = None
            self._prepare_anchor_pixmap() # Update cached anchor drawing
            self.updatePixmap(update_base=False) # Redraw
        except ValueError as e: QMessageBox.warning(self, "Invalid Input", f"Invalid anchor: {e}")
        except Exception as e: QMessageBox.critical(self, "Error", f"Error applying anchor: {e}"); traceback.print_exc()

    def clear_anchor_area(self):
        self.anchor_rect_img_coords = None
        self._ref_anchor_pixmap = None
        # Reset mouse drag state as well
        self.is_selecting = False
        self.selection_start_point = None
        self.selection_end_point = None
        # Reset text boxes
        self._update_anchor_textboxes(None) # Clears them to defaults
        self.btn_clear_anchor.setEnabled(False)
        print("Anchor area cleared.")
        self.updatePixmap(update_base=False)

    # --- Event Filter for Mouse Drag ---
    def eventFilter(self, watched, event):
        # Process events for the reference image label
        if watched == self.ref_image:
            # Check if base pixmap exists, otherwise mapping is impossible
            if not self._ref_base_pixmap or self._ref_base_pixmap.isNull():
                return super().eventFilter(watched, event) # Pass event up

            # Mouse Press: Start selection
            if event.type() == QEvent.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    self.selection_start_point = event.pos() # Store label coords
                    self.selection_end_point = event.pos()
                    self.is_selecting = True
                    self.updatePixmap(update_base=False) # Redraw to show selection start/clear old rect
                    return True # Event handled

            # Mouse Move: Update selection rectangle
            elif event.type() == QEvent.MouseMove:
                if self.is_selecting:
                    self.selection_end_point = event.pos()
                    self.updatePixmap(update_base=False) # Redraw with temp rect
                    return True # Event handled

            # Mouse Release: Finalize selection
            elif event.type() == QEvent.MouseButtonRelease:
                if event.button() == Qt.LeftButton and self.is_selecting:
                    self.selection_end_point = event.pos()
                    self.is_selecting = False

                    # Map selection points (label coords) to image coords
                    p1_img = self.map_label_point_to_image_point(self.selection_start_point)
                    p2_img = self.map_label_point_to_image_point(self.selection_end_point)

                    if p1_img and p2_img:
                        # Create normalized QRect in image coords
                        img_rect = QRect(p1_img, p2_img).normalized()

                        # Validate size (e.g., minimum 2x2 pixels)
                        if img_rect.width() > 1 and img_rect.height() > 1:
                            self.anchor_rect_img_coords = img_rect # Store the result
                            print(f"Anchor area set by drag: {self.anchor_rect_img_coords}")
                            self._update_anchor_textboxes(self.anchor_rect_img_coords) # Update text boxes
                            self.btn_clear_anchor.setEnabled(True)
                            self._prepare_anchor_pixmap() # Update cache with final rect drawn
                        else:
                            print("Selection too small, anchor not set.")
                            # Optionally clear existing anchor or leave it? Leave it for now.
                            # self.clear_anchor_area()
                    else:
                        print("Could not map selection points to image coordinates.")

                    # Redraw to show final state (solid green or no rect)
                    self.updatePixmap(update_base=False)
                    return True # Event handled

        # Pass unhandled events to the base class
        return super().eventFilter(watched, event)

    # --- List Widget Handling ---
    def current_item_changed_slot(self, current_item, previous_item):
        if current_item is None:
            return
        self.item_changed(current_item)

    def check_state_changed(self, item):
        if item.checkState() == Qt.Checked:
            self.set_reference_item(item)
        else:
            checked_items = [self.image_list.item(i) for i in range(self.image_list.count())
                             if self.image_list.item(i).checkState() == Qt.Checked]
            if not checked_items and self.image_list.count() > 0:
                 self.image_list.blockSignals(True)
                 item.setCheckState(Qt.Checked)
                 self.image_list.blockSignals(False)

    def item_changed(self, item):
        if item is None:
            return
        new_current_idx = self.image_list.row(item)
        if new_current_idx != self.current_image_idx:
            self.current_image_idx = new_current_idx
            self._update_rotation_textbox()
            self.updatePixmap(update_base=True)

    def set_reference_item(self, ref_item):
         new_ref_idx = self.image_list.row(ref_item)
         if new_ref_idx == self.ref_image_idx and ref_item.checkState() == Qt.Checked:
             return
         self.image_list.blockSignals(True)
         ref_item.setCheckState(Qt.Checked)
         for i in range(self.image_list.count()):
              item_i = self.image_list.item(i)
              if item_i != ref_item and item_i.checkState() == Qt.Checked:
                  item_i.setCheckState(Qt.Unchecked)
         self.image_list.blockSignals(False)
         self.ref_image_idx = new_ref_idx
         self._update_rotation_textbox() # Update rot display
         self.updatePixmap(update_base=True)

    # --- Image Data & Display ---
    def get_image_data(self, get_raw=False):
        if not hasattr(self, 'image_data') or not self.image_data:
            return None, None
        num_images = len(self.image_data)
        ref_idx_valid = 0 <= self.ref_image_idx < num_images
        current_idx_valid = 0 <= self.current_image_idx < num_images
        if not (ref_idx_valid and current_idx_valid):
            return None, None

        ref_im = self.image_data[self.ref_image_idx]['image']
        current_im_data = self.image_data[self.current_image_idx]['image']
        return ref_im, current_im_data # Return raw data

    def _create_base_pixmap(self, image):
        if image is None:
            return None
        try:
            img_copy = image
            if not img_copy.flags['C_CONTIGUOUS']:
                img_copy = np.ascontiguousarray(img_copy)
            height, width = img_copy.shape[:2]
            if img_copy.ndim == 3 and img_copy.shape[2] == 3:
                channel = 3
                q_format = QImage.Format_RGB888
            else:
                print(f"Error: _create_base_pixmap received non-RGB data shape {img_copy.shape}")
                return None
            bytes_per_line = channel * width
            q_image = QImage(img_copy.data, width, height, bytes_per_line, q_format)
            if q_image.isNull():
                print("Error: Failed QImage creation")
                return None
            return QPixmap.fromImage(q_image)
        except Exception as e:
            print(f"Error creating base pixmap: {e}")
            # traceback.print_exc() # Verbose
            return None

    def _prepare_base_pixmaps(self):
        ref_data, current_data = self.get_image_data(get_raw=True)
        self._ref_base_pixmap = self._create_base_pixmap(ref_data)
        self._current_base_pixmap = self._create_base_pixmap(current_data)
        self._prepare_anchor_pixmap()

    def _prepare_anchor_pixmap(self):
        self._ref_anchor_pixmap = None
        if self.anchor_rect_img_coords and self._ref_base_pixmap:
             try:
                pixmap_to_draw_on = self._ref_base_pixmap.copy()
                painter = QPainter(pixmap_to_draw_on)
                pen = QPen(QColor(0, 255, 0, 200))
                pen.setWidth(2)
                pen.setStyle(Qt.SolidLine)
                painter.setPen(pen)
                painter.drawRect(self.anchor_rect_img_coords)
                painter.end()
                self._ref_anchor_pixmap = pixmap_to_draw_on
             except Exception as e:
                 print(f"Error drawing anchor: {e}")

    def updatePixmap(self, update_base=True):
        if update_base:
            if not hasattr(self, 'image_data') or not self.image_data:
                self.ref_image.clear()
                self.current_image.clear()
                self.ref_image.setText("No Image")
                self.current_image.setText("No Image")
                self._ref_base_pixmap = None
                self._current_base_pixmap = None
                self._ref_anchor_pixmap = None
                self._current_diff_pixmap = None
                return
            self._prepare_base_pixmaps()
            self._update_diff_pixmap()

        if not self._ref_base_pixmap:
            self.ref_image.clear()
            self.ref_image.setText("Error Base Ref")
        else:
            source_ref_pixmap = self._ref_anchor_pixmap if self._ref_anchor_pixmap else self._ref_base_pixmap
            self._apply_zoom_and_set_pixmap(source_ref_pixmap, self.ref_image)

        if not self._current_base_pixmap:
            self.current_image.clear()
            self.current_image.setText("Error Base Cur")
        else:
            in_diff_mode = self.radio_buttons['rad_diff'].isChecked()
            source_current_pixmap = self._current_diff_pixmap if in_diff_mode and self._current_diff_pixmap else self._current_base_pixmap
            self._apply_zoom_and_set_pixmap(source_current_pixmap, self.current_image)

    def _update_diff_pixmap(self):
        self._current_diff_pixmap = None
        if self.radio_buttons['rad_diff'].isChecked():
            ref_data, current_data = self.get_image_data(get_raw=True)
            if ref_data is not None and current_data is not None and ref_data.shape == current_data.shape:
                diff_im = np.abs(ref_data.astype(np.float32) - current_data.astype(np.float32))
                diff_im_uint8 = np.clip(diff_im, 0, 255).astype(np.uint8)
                self._current_diff_pixmap = self._create_base_pixmap(diff_im_uint8)

    def _apply_zoom_and_set_pixmap(self, source_pixmap, label):
        if source_pixmap is None or source_pixmap.isNull():
            label.clear()
            label.setText("Error Source")
            return
        try:
            original_size = source_pixmap.size()
            target_w = int(round(original_size.width() * self.zoom_factor))
            target_h = int(round(original_size.height() * self.zoom_factor))
            target_w = max(1, target_w)
            target_h = max(1, target_h)
            scaled_pixmap = source_pixmap.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled_pixmap)
            label.adjustSize()
        except Exception as e:
            print(f"Error applying zoom: {e}")
            label.setText("Error Display")

    def resizeEvent(self, event):
        super().resizeEvent(event)

    def update_list_widget(self, items):
        self.image_list.clear()
        if not items:
             self.ref_image_idx = -1
             self.current_image_idx = -1
             self.updatePixmap()
             return
        self.image_list.blockSignals(True)
        for idx, item_name in enumerate(items):
            list_item = QListWidgetItem(item_name)
            list_item.setFlags(list_item.flags() | Qt.ItemIsUserCheckable)
            list_item.setCheckState(Qt.Checked if idx == 0 else Qt.Unchecked)
            self.image_list.addItem(list_item)
        self.image_list.blockSignals(False)
        self.ref_image_idx = 0
        self.current_image_idx = 0
        self.image_list.setCurrentRow(0)
        self._update_rotation_textbox()
        self.updatePixmap(update_base=True)

    # --- Input Validation Helpers ---
    def check_input_is_int(self, line_edit_widget, default_value, min_val=None, max_val=None):
        # (No change needed, already clean)
        val_str = line_edit_widget.text()
        valid = False
        try:
            val = int(val_str)
            valid = True
            corrected_val = val
            if min_val is not None and val < min_val:
                corrected_val = min_val
                valid = False
                print(f"Val >= {min_val}")
            if max_val is not None and val > max_val:
                corrected_val = max_val
                valid = False
                print(f"Val <= {max_val}")
            val = corrected_val
        except ValueError:
            val = default_value
            print(f"Input '{val_str}' invalid integer.")
            valid = False
        if not valid:
            line_edit_widget.setText(str(val))

    def check_input_is_float(self, line_edit_widget, default_value):
        # (No change needed, already clean)
        val_str = line_edit_widget.text().replace(',', '.')
        valid = False
        try:
            val = float(val_str)
            valid = True
        except ValueError:
            val = default_value
            print(f"Input '{val_str}' invalid float.")
            valid = False
        if not valid:
            line_edit_widget.setText(f"{val:.1f}")


# Main execution block
if __name__ == '__main__':
    try: # High DPI
        if hasattr(Qt, 'AA_EnableHighDpiScaling'):
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception as e:
        print(f"High DPI settings error: {e}")

    app = QApplication(sys.argv)

    window = MainWindow()
    # Disable FFT if needed

    window.show()
    sys.exit(app.exec_())