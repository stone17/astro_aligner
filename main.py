import os
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QRadioButton, QLabel, QLineEdit, QFrame, QPushButton,
    QGridLayout, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QButtonGroup, QMessageBox,
    QScrollArea
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QIntValidator
from PyQt5.QtCore import Qt, QObject, QEvent, QPoint, QRect, QLine, pyqtSignal, pyqtSlot
import numpy as np
import yaml
from functools import partial
import traceback

from file_functions import (loadFolder, saveImages)
from image_functions import (
    translate_image,
    apply_text_rotation, rotate_image_incremental, apply_M_transformation,
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

# Add near top of main_gui.py after imports

class ClickableLabel(QLabel):
    """A QLabel subclass that emits a signal with click coordinates."""
    # Signal signature: Emits the QPoint where the click occurred within the label
    clicked_at = pyqtSignal(QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)

    def mousePressEvent(self, event):
        # Emit the signal only for left mouse button clicks
        if event.button() == Qt.LeftButton:
            self.clicked_at.emit(event.pos()) # event.pos() gives coords relative to this label
        # Optional: Call base class implementation if needed for other mouse handling
        # super().mousePressEvent(event)
        # Or just accept the event here
        event.accept()

class MainWindow(QMainWindow):
    CONFIG_FILE = 'config.yaml'

    def __init__(self):
        super().__init__()
        self.anchor_rect_img_coords = None
        self.zoom_factor = 1.0
        self.image_data = []

        # --- Point Selection State ---
        self.setting_ref_points = False  # NEW flag
        self.setting_current_points = False # NEW flag
        self.ref_points = []
        self.current_points = []

        # --- Mouse Drag State (For Anchor Box) ---
        self.selection_start_point = None
        self.selection_end_point = None
        self.is_selecting = False

        # Cached pixmaps
        self._ref_base_pixmap = None
        self._current_base_pixmap = None
        self._ref_anchor_pixmap = None # Ref pixmap with anchor drawn
        self._current_diff_pixmap = None # Diff view pixmap
        self.show_crosshair = False

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

        self.ref_image = ClickableLabel(self)
        self.ref_image.setAlignment(Qt.AlignCenter)
        self.ref_image.setStyleSheet("background-color: #202020;")
        self.ref_image.setMouseTracking(True) # Enable mouse move events
        self.ref_image.installEventFilter(self)
        self.ref_image.clicked_at.connect(self.handle_ref_label_click)

        self.current_image = ClickableLabel(self)
        self.current_image.setAlignment(Qt.AlignCenter)
        self.current_image.setStyleSheet("background-color: #202020;")
        self.current_image.setMouseTracking(True)
        #self.current_image.installEventFilter(self)
        self.current_image.clicked_at.connect(self.handle_current_label_click)

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

        # --- Add Crosshair Toggle Button ---
        self.btn_toggle_crosshair = QPushButton("Crosshair", self)
        self.btn_toggle_crosshair.setToolTip("Toggle crosshair/bullseye overlay on Reference Image")
        self.btn_toggle_crosshair.setCheckable(True) # Make it a toggle button
        self.btn_toggle_crosshair.setChecked(self.show_crosshair) # Set initial state
        self.btn_toggle_crosshair.toggled.connect(self.toggle_crosshair) # Connect signal
        image_controls_layout.addWidget(self.btn_toggle_crosshair) # Add to layout

        self.radio_buttons = {
            'rad_normal': QRadioButton("Original"),
            'rad_diff': QRadioButton("Difference"),
            }
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
        self.reg_method_scan_radio = QRadioButton("Scan Shift")
        self.reg_method_scan_radio.setToolTip("Scans around anchor for minimum difference.\nRequires anchor. Integer shifts only.")
        self.reg_method_scan_rot_radio = QRadioButton("Scan Rot")
        self.reg_method_scan_rot_radio.setToolTip("Rotate around anchor for minimum difference.\nRequires anchor.")
        self.reg_method_group = QButtonGroup(self)
        self.reg_method_group.addButton(self.reg_method_fft_radio)
        self.reg_method_group.addButton(self.reg_method_scan_radio)
        self.reg_method_group.addButton(self.reg_method_scan_rot_radio)
        reg_settings_layout.addWidget(self.reg_method_fft_radio, 1, 0, 1, 2)
        reg_settings_layout.addWidget(self.reg_method_scan_radio, 2, 0, 1, 1)
        reg_settings_layout.addWidget(self.reg_method_scan_rot_radio, 2, 1, 1, 1)
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
        manual_adj_layout.setContentsMargins(5, 5, 5, 5)
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
        manual_adj_layout.addWidget(btn_right, 1, 0)
        shift_txt = QLabel('Shift [px]:')
        self.shift_val = QLineEdit('1', self)
        self.shift_val.setMaximumWidth(40)
        self.shift_val.setValidator(NonNegativeIntValidator(self))
        self.shift_val.editingFinished.connect(partial(self.check_input_is_int, self.shift_val, 1, min_val=1))
        manual_adj_layout.addWidget(shift_txt, 3, 0, 1, 2)
        manual_adj_layout.addWidget(self.shift_val, 3, 1, 1, 1)
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
        self.btn_apply_rotation = QPushButton("Apply Rotation", self)
        self.btn_apply_rotation.setToolTip("Apply rotation in text box")
        self.btn_apply_rotation.clicked.connect(partial(apply_text_rotation, self))
        manual_adj_layout.addWidget(self.btn_apply_rotation, 3, 4, 1, 2)
        manual_adj_layout.setColumnStretch(7, 1)
        control_layout.addWidget(manual_adj_frame, 0, 2)

        # --- Point Select & Morph (Modified Point Select Section) ---
        point_morph_widget = QFrame(self)
        point_morph_widget.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        point_morph_layout = QVBoxLayout(point_morph_widget)
        point_morph_layout.setContentsMargins(5, 5, 5, 5)

        # -- Point Selection Sub-Group --
        point_box = QFrame(self)
        point_box_layout = QGridLayout(point_box)
        point_title = QLabel("2-Point Rotation Aid:") # Updated title
        point_title.setStyleSheet("font-weight: bold;")
        point_box_layout.addWidget(point_title, 0, 0, 1, 3) # Span 3 cols

        # Checkable buttons for setting points - text updated dynamically
        self.btn_set_ref_pts = QPushButton("Ref Points (0/2)", self)
        self.btn_set_ref_pts.setCheckable(True)
        self.btn_set_ref_pts.toggled.connect(self.toggle_set_ref_mode)
        point_box_layout.addWidget(self.btn_set_ref_pts, 1, 0, 1, 1) # Span 3 cols

        self.btn_set_cur_pts = QPushButton("Target Points (0/2)", self)
        self.btn_set_cur_pts.setCheckable(True)
        self.btn_set_cur_pts.toggled.connect(self.toggle_set_cur_mode)
        point_box_layout.addWidget(self.btn_set_cur_pts, 2, 0, 1, 1) # Span 3 cols

        # Action/Clear buttons
        self.btn_clear_ref_points = QPushButton("Clear Ref Pts")
        self.btn_clear_ref_points.clicked.connect(self.clear_ref_points)
        point_box_layout.addWidget(self.btn_clear_ref_points, 1, 1)

        self.btn_clear_current_points = QPushButton("Clear Cur Pts")
        self.btn_clear_current_points.clicked.connect(self.clear_current_points)
        point_box_layout.addWidget(self.btn_clear_current_points, 2, 1)

        self.btn_get_rotation = QPushButton("Get Rotation")
        self.btn_get_rotation.setToolTip("Calculate rotation from 2+2 points\nand update 'Total Rot' field")
        self.btn_get_rotation.clicked.connect(self.calculate_rotation_from_points)
        point_box_layout.addWidget(self.btn_get_rotation, 4, 0, 1, 2)
        point_morph_layout.addWidget(point_box) # Add point box to the column
        control_layout.addWidget(point_morph_widget, 0, 3) # Add whole column widget

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
        control_layout.addWidget(morph_frame, 0, 4)

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

# --- Coordinate Mapping for Mouse Drag / Point Click ---
    # MODIFIED: Use self.zoom_factor, ensure base_pixmap arg exists if called from point select handlers
    def map_label_point_to_image_point(self, label_point, base_pixmap=None):
        """Maps a QPoint from QLabel coordinates to original image coordinates."""

        # If called for anchor dragging, base_pixmap won't be passed, use self._ref_base_pixmap
        # If called for point selection, base_pixmap IS passed.
        if base_pixmap is None:
            base_pixmap = self._ref_base_pixmap # Default to ref for anchor drag

        if base_pixmap is None or base_pixmap.isNull() or self.zoom_factor <= 1e-9 or label_point is None:
            # Check self.zoom_factor here
            print("Cannot map point: Missing base pixmap, zero zoom, or no label point")
            return None

        original_size = base_pixmap.size()
        img_w = original_size.width()
        img_h = original_size.height()

        if img_w <= 0 or img_h <= 0:
            print("Cannot map point: Invalid original image size from base pixmap")
            return None

        label_x = label_point.x()
        label_y = label_point.y()

        # Map back using MainWindow's zoom factor
        img_x = label_x / self.zoom_factor # Use self.zoom_factor
        img_y = label_y / self.zoom_factor # Use self.zoom_factor

        # Clamp to image bounds
        img_x = max(0.0, min(img_w, img_x))
        img_y = max(0.0, min(img_h, img_y))

        return QPoint(int(round(img_x)), int(round(img_y)))

    # --- Crosshair Toggle Slot ---
    def toggle_crosshair(self, checked):
        """Slot connected to the crosshair button's toggled signal."""
        self.show_crosshair = checked
        print(f"Crosshair overlay {'enabled' if checked else 'disabled'}")
        # Trigger redraw only (don't need to reload base images)
        self.updatePixmap(update_base=False)

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
        if not hasattr(self, 'image_data') or not self.image_data:
            QMessageBox.warning(self, "No Image", "Load images first.")
            return
        if not (0 <= self.ref_image_idx < len(self.image_data)):
            QMessageBox.warning(self, "Invalid Reference", "Cannot get ref dims.")
            return
        try:
            img_h, img_w = self.images[self.ref_image_idx].shape[:2]
            x0 = int(self.anchor_x.text())
            y0 = int(self.anchor_y.text())
            w = int(self.anchor_w.text())
            h = int(self.anchor_h.text())
            if w <= 0 or h <= 0:
                raise ValueError("W/H > 0");
            if x0 < 0 or y0 < 0:
                raise ValueError("X/Y >= 0");
            if x0 + w > img_w or y0 + h > img_h:
                raise ValueError("Anchor outside bounds")
            self.anchor_rect_img_coords = QRect(x0, y0, w, h)
            print(f"Anchor area applied from text: {self.anchor_rect_img_coords}")
            self.btn_clear_anchor.setEnabled(True)
            self.is_selecting = False
            self.selection_start_point = None
            self.selection_end_point = None
            # self._prepare_anchor_pixmap() # No longer needed
            self.updatePixmap(update_base=False) # Redraw with new anchor
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Invalid anchor: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying anchor: {e}")
            traceback.print_exc()

    def clear_anchor_area(self):
        self.anchor_rect_img_coords = None
        # self._ref_anchor_pixmap = None # No longer needed
        self.is_selecting = False
        self.selection_start_point = None
        self.selection_end_point = None
        self._update_anchor_textboxes(None)
        self.btn_clear_anchor.setEnabled(False)
        print("Anchor area cleared.")
        self.updatePixmap(update_base=False) # Redraw without anchor

    # --- Add NEW Slots for Label Clicks ---
    @pyqtSlot(QPoint) # Decorator specifies the argument type from the signal
    def handle_ref_label_click(self, label_point):
        """Handles left-clicks on the reference image label."""
        # Check if in the correct mode and haven't selected max points
        if self.setting_ref_points and len(self.ref_points) < 2:
            base_pixmap = self._ref_base_pixmap
            if not base_pixmap or base_pixmap.isNull():
                print("Warning: Ref base pixmap missing for point mapping.")
                return

            img_point = self.map_label_point_to_image_point(label_point, base_pixmap) # Pass base pixmap now
            if img_point:
                self.ref_points.append(img_point)
                print(f"Added Ref Point {len(self.ref_points)}/2: Img Coords ({img_point.x()}, {img_point.y()})")
                self._update_point_button_text()
                self.updatePixmap(update_base=False) # Redraw with marker

                # Auto-uncheck button if max points reached
                if len(self.ref_points) == 2:
                    print("Max Ref points reached, exiting mode.")
                    self.btn_set_ref_pts.setChecked(False)
            else:
                 print("Failed to map Ref click coordinates.")

    @pyqtSlot(QPoint)
    def handle_current_label_click(self, label_point):
        """Handles left-clicks on the current image label."""
        if self.setting_current_points and len(self.current_points) < 2:
            base_pixmap = self._current_base_pixmap
            if not base_pixmap or base_pixmap.isNull():
                print("Warning: Current base pixmap missing for point mapping.")
                return

            img_point = self.map_label_point_to_image_point(label_point, base_pixmap) # Pass base pixmap now
            if img_point:
                self.current_points.append(img_point)
                print(f"Added Cur Point {len(self.current_points)}/2: Img Coords ({img_point.x()}, {img_point.y()})")
                self._update_point_button_text()
                self.updatePixmap(update_base=False)

                if len(self.current_points) == 2:
                    print("Max Current points reached, exiting mode.")
                    self.btn_set_cur_pts.setChecked(False)
            else:
                print("Failed to map Current click coordinates.")

# --- Add Event Filter back for Anchor Box Dragging ONLY ---
    def eventFilter(self, watched, event):
        # Handle events only for the reference image label for anchor dragging
        if watched == self.ref_image:
            # Anchor drag logic - check if NOT in point selection mode
            if not self.setting_ref_points and not self.setting_current_points:

                if event.type() == QEvent.MouseButtonPress:
                    # Start anchor selection drag
                    if event.button() == Qt.LeftButton:
                        # Check base pixmap exists before starting drag
                        if self._ref_base_pixmap and not self._ref_base_pixmap.isNull():
                            self.selection_start_point = event.pos()
                            self.selection_end_point = event.pos()
                            self.is_selecting = True # Start selecting anchor box
                            self.updatePixmap(update_base=False)
                            return True # Consume event

                elif event.type() == QEvent.MouseMove:
                    # Update anchor selection rectangle if dragging
                    if self.is_selecting:
                        self.selection_end_point = event.pos()
                        self.updatePixmap(update_base=False)
                        return True # Consume event

                elif event.type() == QEvent.MouseButtonRelease:
                    # Finalize anchor selection
                    if event.button() == Qt.LeftButton and self.is_selecting:
                        self.selection_end_point = event.pos()
                        self.is_selecting = False # Finish selecting anchor box

                        # Map selection points (label coords) to image coords
                        # Call mapping function WITHOUT base_pixmap arg, it will default to ref
                        p1_img = self.map_label_point_to_image_point(self.selection_start_point)
                        p2_img = self.map_label_point_to_image_point(self.selection_end_point)

                        if p1_img and p2_img:
                            img_rect = QRect(p1_img, p2_img).normalized()
                            if img_rect.width() > 1 and img_rect.height() > 1:
                                self.anchor_rect_img_coords = img_rect
                                print(f"Anchor area set by drag: {self.anchor_rect_img_coords}")
                                self._update_anchor_textboxes(self.anchor_rect_img_coords)
                                self.btn_clear_anchor.setEnabled(True)
                                # self._prepare_anchor_pixmap() # No longer needed
                            else:
                                print("Selection too small, anchor not set.")
                        else:
                            print("Could not map selection points to image coordinates.")

                        # Final redraw after releasing mouse
                        self.updatePixmap(update_base=False)
                        return True # Consume event

        # Pass unhandled events to the base class implementation
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

    # --- Modify List Widget Handlers ---
    def item_changed(self, item): # When selection highlight changes
        if item is None: return
        new_current_idx = self.image_list.row(item)
        if new_current_idx != self.current_image_idx:
            self.current_image_idx = new_current_idx
            # --- Clear only CURRENT points when changing current image ---
            self.current_points = []
            self._update_point_button_text() # Update counters on buttons
            self._update_rotation_textbox() # Update rot display
            self.updatePixmap(update_base=True) # Force regen of current base pixmap

    def set_reference_item(self, ref_item): # When CHECKBOX changes
         new_ref_idx = self.image_list.row(ref_item)
         if new_ref_idx == self.ref_image_idx and ref_item.checkState() == Qt.Checked: return

         # --- Clear BOTH point sets when reference changes ---
         self.ref_points = []
         self.current_points = []
         self._update_point_button_text() # Update counters on buttons

         self.image_list.blockSignals(True); ref_item.setCheckState(Qt.Checked)
         for i in range(self.image_list.count()):
              item_i = self.image_list.item(i)
              if item_i != ref_item and item_i.checkState() == Qt.Checked: item_i.setCheckState(Qt.Unchecked)
         self.image_list.blockSignals(False)
         self.ref_image_idx = new_ref_idx
         self._update_rotation_textbox() # Update rot display
         self.updatePixmap(update_base=True) # Force regen of ref base pixmap

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
        #self._prepare_anchor_pixmap()

    def updatePixmap(self, update_base=True):
        # Updates display, handles drawing selection/anchor/crosshair/points
        if update_base:
            if not hasattr(self, 'image_data') or not self.image_data:
                self.ref_image.clear()
                self.current_image.clear()
                self.ref_image.setText("No Image")
                self.current_image.setText("No Image")
                self._ref_base_pixmap = None
                self._current_base_pixmap = None
                # self._ref_anchor_pixmap = None # No longer cached separately
                self._current_diff_pixmap = None
                return
            self._prepare_base_pixmaps() # Prepares base ref/current pixmaps
            self._update_diff_pixmap() # Prepares diff view pixmap cache

        # --- Process Reference Image ---
        if not self._ref_base_pixmap:
            self.ref_image.clear()
            self.ref_image.setText("Error Base Ref")
            # Set final ref pixmap to None so current image processing doesn't error
            pixmap_for_ref_label = None
        else:
            # Start with a fresh copy of the base pixmap to draw overlays on
            pixmap_to_draw_on = self._ref_base_pixmap.copy()
            painter = QPainter(pixmap_to_draw_on)
            img_w = self._ref_base_pixmap.width()
            img_h = self._ref_base_pixmap.height()

            # 1. Draw FINAL anchor rectangle (Solid Green) if defined and not selecting
            if self.anchor_rect_img_coords and not self.is_selecting:
                pen_anchor = QPen(QColor(0, 255, 0, 200)) # Green
                pen_anchor.setWidth(2)
                pen_anchor.setStyle(Qt.SolidLine)
                painter.setPen(pen_anchor)
                painter.drawRect(self.anchor_rect_img_coords) # Image coords

            # 2. Draw TEMP mouse drag selection rectangle (Dashed Red) if selecting
            if self.is_selecting and self.selection_start_point and self.selection_end_point:
                # --- CORRECTED CALLS (Removed self._ref_base_pixmap argument) ---
                p1_img = self.map_label_point_to_image_point(self.selection_start_point)
                p2_img = self.map_label_point_to_image_point(self.selection_end_point)
                # --- End Correction ---
                if p1_img and p2_img:
                    temp_rect_img = QRect(p1_img, p2_img).normalized()
                    pen_temp = QPen(QColor(255, 0, 0, 180)) # Red
                    pen_temp.setWidth(1)
                    pen_temp.setStyle(Qt.DashLine)
                    painter.setPen(pen_temp)
                    painter.drawRect(temp_rect_img) # Image coords

            # 3. Draw Reference Point Markers (Cyan)
            if self.ref_points: # Check if list is not empty
                pen_ref_pts = QPen(QColor("cyan"))
                pen_ref_pts.setWidth(1)
                painter.setPen(pen_ref_pts)
                marker_radius = 4 # Radius in image pixels
                for img_point in self.ref_points:
                    painter.drawEllipse(img_point, marker_radius, marker_radius)
                    painter.drawLine(img_point.x() - marker_radius, img_point.y(), img_point.x() + marker_radius, img_point.y())
                    painter.drawLine(img_point.x(), img_point.y() - marker_radius, img_point.x(), img_point.y() + marker_radius)

            # 4. Draw Crosshair/Bullseye (Red) if enabled
            if hasattr(self, 'show_crosshair') and self.show_crosshair and img_w > 0 and img_h > 0:
                 center_x = img_w // 2
                 center_y = img_h // 2
                 pen_cross = QPen(QColor(255, 0, 0, 200)) # Red
                 pen_cross.setWidth(2)
                 painter.setPen(pen_cross)
                 # Crosshair lines
                 painter.drawLine(0, center_y, img_w, center_y) # Horizontal
                 painter.drawLine(center_x, 0, center_x, img_h) # Vertical
                 # Concentric Rings
                 min_dim = min(img_w, img_h)
                 radii = [int(min_dim * r) for r in [0.05, 0.10, 0.15]] # Example radii
                 center_point = QPoint(center_x, center_y)
                 for r in radii:
                     if r > 0:
                         painter.drawEllipse(center_point, r, r)

            painter.end()
            # Use the pixmap with drawings for applying zoom
            pixmap_for_ref_label = pixmap_to_draw_on

        # Apply zoom and set pixmap for the reference label
        self._apply_zoom_and_set_pixmap(pixmap_for_ref_label, self.ref_image)


        # --- Process Current Image ---
        if not self._current_base_pixmap:
            self.current_image.clear()
            self.current_image.setText("Error Base Cur")
            pixmap_for_cur_label = None
        else:
            in_diff_mode = self.radio_buttons['rad_diff'].isChecked()
            # Select source pixmap (diff view or normal)
            source_pixmap = self._current_diff_pixmap if in_diff_mode and self._current_diff_pixmap else self._current_base_pixmap

            # Draw points on current image if they exist
            if self.current_points:
                pixmap_to_draw_on_cur = source_pixmap.copy()
                painter_cur = QPainter(pixmap_to_draw_on_cur)
                pen_cur_pts = QPen(QColor("lime")) # Lime green for current points
                pen_cur_pts.setWidth(2)
                painter_cur.setPen(pen_cur_pts)
                marker_radius = 4
                for img_point in self.current_points:
                    painter_cur.drawEllipse(img_point, marker_radius, marker_radius)
                    painter_cur.drawLine(img_point.x() - marker_radius, img_point.y(), img_point.x() + marker_radius, img_point.y())
                    painter_cur.drawLine(img_point.x(), img_point.y() - marker_radius, img_point.x(), img_point.y() + marker_radius)
                painter_cur.end()
                # Use the pixmap with points drawn
                pixmap_for_cur_label = pixmap_to_draw_on_cur
            else:
                # No points, use the source pixmap directly
                pixmap_for_cur_label = source_pixmap

        # Apply zoom and set pixmap for the current label
        self._apply_zoom_and_set_pixmap(pixmap_for_cur_label, self.current_image)

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

  # --- Add NEW Toggle Slots ---
    def toggle_set_ref_mode(self, checked):
        """Handles state when 'Set Ref Points' button is toggled."""
        print(f"--- toggle_set_ref_mode called with checked={checked} ---") # DIAGNOSTIC
        self.setting_ref_points = checked
        if checked:
            print("  Entering Ref Point Mode...") # DIAGNOSTIC
            if self.btn_set_cur_pts.isChecked():
                print("  Unchecking Set Target Points button programmatically.") # DIAGNOSTIC
                self.btn_set_cur_pts.blockSignals(True)
                self.btn_set_cur_pts.setChecked(False)
                self.btn_set_cur_pts.setStyleSheet("")
                self.btn_set_cur_pts.blockSignals(False)
                self.setting_current_points = False

            self.btn_set_ref_pts.setStyleSheet("background-color: lightblue;")
            QApplication.setOverrideCursor(Qt.CrossCursor)
        else:
            print("  Exiting Ref Point Mode...") # DIAGNOSTIC
            # Only restore cursor if the other button *isn't* now active
            if not self.setting_current_points:
                print("  Restoring cursor (current points mode not active).") # DIAGNOSTIC
                QApplication.restoreOverrideCursor()
            else:
                 print("  Not restoring cursor (current points mode IS active).") # DIAGNOSTIC
            self.btn_set_ref_pts.setStyleSheet("")
        # Update button text regardless of state change source
        self._update_point_button_text()
        print(f"--- toggle_set_ref_mode finished ---") # DIAGNOSTIC

    def toggle_set_cur_mode(self, checked):
        """Handles state when 'Set Target Points' button is toggled."""
        self.setting_current_points = checked
        if checked:
            # If the other button is checked, uncheck it programmatically
            if self.btn_set_ref_pts.isChecked():
                self.btn_set_ref_pts.blockSignals(True)
                self.btn_set_ref_pts.setChecked(False)
                self.btn_set_ref_pts.setStyleSheet("")
                self.btn_set_ref_pts.blockSignals(False)
                self.setting_ref_points = False

            self.btn_set_cur_pts.setStyleSheet("background-color: lightgreen;") # Indicate active
            QApplication.setOverrideCursor(Qt.CrossCursor)
            print("Mode: Setting Current/Target Points")
        else:
            # Only restore cursor if the other button *isn't* now active
            if not self.setting_ref_points:
                QApplication.restoreOverrideCursor()
            self.btn_set_cur_pts.setStyleSheet("") # Reset style
            print("Exited setting Current Points mode.")
        self._update_point_button_text()

    # --- Add NEW Helper for Button Text ---
    def _update_point_button_text(self):
        """Updates the text of point selection buttons to show counts."""
        ref_count = len(self.ref_points)
        cur_count = len(self.current_points)
        # Check if buttons exist before setting text
        if hasattr(self, 'btn_set_ref_pts'):
            self.btn_set_ref_pts.setText(f"Set Ref Points ({ref_count}/2)")
        if hasattr(self, 'btn_set_cur_pts'):
            self.btn_set_cur_pts.setText(f"Set Target Points ({cur_count}/2)")

    # --- Add NEW Slots for Clear Buttons ---
    def clear_ref_points(self):
        """Clears only the reference points."""
        self.ref_points = []
        self._update_point_button_text() # Update count display
        self.updatePixmap(update_base=False) # Redraw ref image without points
        print("Reference points cleared.")

    def clear_current_points(self):
        """Clears only the current points."""
        self.current_points = []
        self._update_point_button_text() # Update count display
        self.updatePixmap(update_base=False) # Redraw current image without points
        print("Current points cleared.")

    # --- Add NEW Slot for Get Rotation Button ---
    def calculate_rotation_from_points(self):
        """Calculates rotation from 2+2 points and updates the rotation text box."""
        if len(self.ref_points) != 2 or len(self.current_points) != 2:
            QMessageBox.warning(self, "Not Enough Points", "Please select exactly 2 points on both images.")
            return

        try:
            ref1, ref2 = self.ref_points[0], self.ref_points[1]
            cur1, cur2 = self.current_points[0], self.current_points[1]

            dx_ref = ref2.x() - ref1.x()
            dy_ref = ref2.y() - ref1.y()
            dx_curr = cur2.x() - cur1.x()
            dy_curr = cur2.y() - cur1.y()

            if abs(dx_ref) < 1e-6 and abs(dy_ref) < 1e-6:
                QMessageBox.warning(self, "Invalid Points", "Ref points identical.")
                return
            if abs(dx_curr) < 1e-6 and abs(dy_curr) < 1e-6:
                QMessageBox.warning(self, "Invalid Points", "Current points identical.")
                return

            angle_ref_rad = np.arctan2(dy_ref, dx_ref)
            angle_curr_rad = np.arctan2(dy_curr, dx_curr)
            rotation_rad = angle_ref_rad - angle_curr_rad
            rotation_deg = np.degrees(rotation_rad)

            # Calculate the *potential* new total rotation if this delta were applied
            new_target_rotation = 0.0 # Default
            if 0 <= self.current_image_idx < len(self.image_data):
                 current_total_rotation = self.image_data[self.current_image_idx].get('total_rotation', 0.0)
                 new_target_rotation = current_total_rotation - rotation_deg
            else:
                 # If no valid current image, just show the delta?
                 new_target_rotation = -rotation_deg # Or maybe disable button?

            print(f"Calculated Rotation Delta from points: {rotation_deg:.3f} degrees")
            # Update the text box with the calculated potential NEW total rotation
            self.rot_val.setText(f"{new_target_rotation:.2f}")
            apply_text_rotation(self)

        except Exception as e:
             QMessageBox.critical(self, "Error", f"Error calculating rotation from points: {e}")
             traceback.print_exc()

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