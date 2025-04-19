import os
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QRadioButton, QLabel, QLineEdit, QFrame, QPushButton,
    QGridLayout, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QButtonGroup, QMessageBox,
    QScrollArea, QProgressDialog, QFileDialog
    )
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QIntValidator
from PyQt5.QtCore import Qt, QObject, QEvent, QPoint, QRect, QLine, pyqtSignal, pyqtSlot, QThread # Added QThread
import numpy as np
import yaml
from functools import partial
import traceback
import copy # Needed for deep copying image data for worker
import cv2 # Import OpenCV for _create_base_pixmap

from file_functions import (loadFolder, saveImages)
# Import only helpers now, not the main loop functions
from image_functions import (
    translate_image,
    apply_text_rotation, rotate_image_incremental, apply_M_transformation,
    _perform_cv_rotation, # Keep if needed by main thread funcs like apply_M
    _register_fft, _register_scan_ssd, _register_scan_ssd_rot # Keep for worker import
)
# Import Worker classes
from workers import RegistrationWorker, MorphWorker


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
        self.setting_ref_points = False
        self.setting_current_points = False
        self.ref_points = []
        self.current_points = []

        # --- Mouse Drag State (For Anchor Box) ---
        self.selection_start_point = None
        self.selection_end_point = None
        self.is_selecting = False

        # Cached pixmaps
        self._ref_base_pixmap = None
        self._current_base_pixmap = None
        # self._ref_anchor_pixmap = None # No longer needed
        self._current_diff_pixmap = None
        self.show_crosshair = False

        # Config related attributes
        self.last_load_folder = None
        self.last_save_folder = None
        self.last_morph_folder = None
        self.load_config() # Load config early

        # --- Worker Thread Attributes ---
        self.worker_thread = None
        self.worker = None
        self.progress_dialog = None

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

        # --- Assign buttons to self ---
        self.btn_open = QPushButton('Open folder', self)
        self.btn_open.setToolTip("Select image folder.")
        self.btn_open.clicked.connect(partial(loadFolder, self))
        list_layout.addWidget(self.btn_open, 2, 0, 1, 2)

        self.btn_save_all = QPushButton('Save all', self)
        self.btn_save_all.setToolTip("Save all images.")
        self.btn_save_all.clicked.connect(partial(saveImages, self))
        list_layout.addWidget(self.btn_save_all, 3, 0)

        self.btn_save_current = QPushButton('Save current', self)
        self.btn_save_current.setToolTip("Save current image.")
        self.btn_save_current.clicked.connect(partial(saveImages, self))
        list_layout.addWidget(self.btn_save_current, 3, 1)
        # --- End assignment ---

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

        # --- MODIFIED: Connect buttons to start_registration ---
        self.btn_register_all = QPushButton('Register all', self)
        self.btn_register_all.setToolTip("Register all to reference.")
        self.btn_register_all.clicked.connect(partial(self.start_registration, 'all')) # Use partial for mode
        reg_settings_layout.addWidget(self.btn_register_all, 3, 1)

        self.btn_register_current = QPushButton('Register current', self)
        self.btn_register_current.setToolTip("Register current to reference.")
        self.btn_register_current.clicked.connect(partial(self.start_registration, 'current')) # Use partial for mode
        reg_settings_layout.addWidget(self.btn_register_current, 3, 0)
        # --- END MODIFICATION ---

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
        self.btn_left = QPushButton('← Left', self)
        self.btn_right = QPushButton('Right →', self)
        self.btn_up = QPushButton('↑ Up', self)
        self.btn_down = QPushButton('↓ Down', self)
        self.btn_left.clicked.connect(partial(translate_image, self, 'Left'))
        self.btn_right.clicked.connect(partial(translate_image, self, 'Right'))
        self.btn_up.clicked.connect(partial(translate_image, self, 'Up'))
        self.btn_down.clicked.connect(partial(translate_image, self, 'Down'))
        manual_adj_layout.addWidget(self.btn_up, 1, 1)
        manual_adj_layout.addWidget(self.btn_left, 2, 0)
        manual_adj_layout.addWidget(self.btn_down, 2, 1)
        manual_adj_layout.addWidget(self.btn_right, 1, 0)
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
        self.btn_rot_l = QPushButton('↺ CCW', self)
        self.btn_rot_r = QPushButton('CW ↻', self)
        self.btn_rot_l.clicked.connect(partial(rotate_image_incremental, self, 'Left'))
        self.btn_rot_r.clicked.connect(partial(rotate_image_incremental, self, 'Right'))
        manual_adj_layout.addWidget(self.btn_rot_l, 1, 4)
        manual_adj_layout.addWidget(self.btn_rot_r, 1, 5)
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

        # --- MODIFIED: Connect button to start_morphing ---
        self.btn_morph = QPushButton('Morph images', self)
        self.btn_morph.setToolTip("Create morph sequence.")
        self.btn_morph.clicked.connect(self.start_morphing) # Connect to start method
        morph_layout.addWidget(self.btn_morph, 1, 0, 1, 1)
        # --- END MODIFICATION ---

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
             # Use .get() for safety in case 'total_rotation' is missing
             current_rotation = self.image_data[self.current_image_idx].get('total_rotation', 0.0)
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
            # Use the actual image data dimensions
            ref_img_data = self.image_data[self.ref_image_idx]['image']
            img_h, img_w = ref_img_data.shape[:2]

            x0 = int(self.anchor_x.text())
            y0 = int(self.anchor_y.text())
            w = int(self.anchor_w.text())
            h = int(self.anchor_h.text())

            # Validate inputs
            if w <= 0 or h <= 0:
                raise ValueError("Width and Height must be positive.")
            if x0 < 0 or y0 < 0:
                raise ValueError("X and Y coordinates cannot be negative.")
            if x0 + w > img_w or y0 + h > img_h:
                raise ValueError(f"Anchor area exceeds image bounds ({img_w}x{img_h}).")

            self.anchor_rect_img_coords = QRect(x0, y0, w, h)
            print(f"Anchor area applied from text: {self.anchor_rect_img_coords}")
            self.btn_clear_anchor.setEnabled(True)
            self.is_selecting = False
            self.selection_start_point = None
            self.selection_end_point = None
            self.updatePixmap(update_base=False) # Redraw with new anchor
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Invalid anchor values: {e}")
            # Optionally reset text boxes to previous valid state or defaults
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying anchor: {e}")
            traceback.print_exc()


    def clear_anchor_area(self):
        self.anchor_rect_img_coords = None
        # self._ref_anchor_pixmap = None # No longer needed
        self.is_selecting = False
        self.selection_start_point = None
        self.selection_end_point = None
        self._update_anchor_textboxes(None) # Reset text boxes
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
                    self.btn_set_ref_pts.setChecked(False) # This will trigger toggle_set_ref_mode(False)
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
                    self.btn_set_cur_pts.setChecked(False) # This will trigger toggle_set_cur_mode(False)
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
                            # Validate selection size before applying
                            if img_rect.width() > 4 and img_rect.height() > 4: # Require minimum size
                                self.anchor_rect_img_coords = img_rect
                                print(f"Anchor area set by drag: {self.anchor_rect_img_coords}")
                                self._update_anchor_textboxes(self.anchor_rect_img_coords) # Update text boxes
                                self.btn_clear_anchor.setEnabled(True)
                            else:
                                print("Selection too small, anchor not set.")
                                # Clear any temporary drawing but don't set anchor_rect_img_coords
                                self.selection_start_point = None
                                self.selection_end_point = None
                        else:
                            print("Could not map selection points to image coordinates.")
                            self.selection_start_point = None
                            self.selection_end_point = None


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
            # Ensure at least one item remains checked if possible
            checked_items = [self.image_list.item(i) for i in range(self.image_list.count())
                             if self.image_list.item(i).checkState() == Qt.Checked]
            if not checked_items and self.image_list.count() > 0:
                 # If the user unchecked the last checked item, re-check it
                 print("Prevented unchecking the last reference item.")
                 self.image_list.blockSignals(True)
                 item.setCheckState(Qt.Checked)
                 self.image_list.blockSignals(False)
                 # No need to call set_reference_item again, state didn't actually change

    # --- Modify List Widget Handlers ---
    def item_changed(self, item): # When selection highlight changes
        if item is None: return
        new_current_idx = self.image_list.row(item)
        if new_current_idx != self.current_image_idx:
            print(f"Current image changed to index: {new_current_idx}")
            self.current_image_idx = new_current_idx
            # --- Clear only CURRENT points when changing current image ---
            if self.current_points:
                print("Clearing current points due to image change.")
                self.current_points = []
                self._update_point_button_text() # Update counters on buttons
            self._update_rotation_textbox() # Update rot display
            self.updatePixmap(update_base=True) # Force regen of current base pixmap

    def set_reference_item(self, ref_item): # When CHECKBOX changes
         new_ref_idx = self.image_list.row(ref_item)
         # Only proceed if the reference index is actually changing
         if new_ref_idx == self.ref_image_idx:
             # If the check state is already checked, do nothing
             if ref_item.checkState() == Qt.Checked:
                 return
             # If it was somehow unchecked (should be prevented by check_state_changed), re-check it
             else:
                 self.image_list.blockSignals(True)
                 ref_item.setCheckState(Qt.Checked)
                 self.image_list.blockSignals(False)
                 return

         print(f"Reference image changed to index: {new_ref_idx}")
         # --- Clear BOTH point sets when reference changes ---
         if self.ref_points or self.current_points:
             print("Clearing ref and current points due to reference change.")
             self.ref_points = []
             self.current_points = []
             self._update_point_button_text() # Update counters on buttons

         # Update check states, ensuring only the new ref_item is checked
         self.image_list.blockSignals(True)
         ref_item.setCheckState(Qt.Checked)
         for i in range(self.image_list.count()):
              item_i = self.image_list.item(i)
              if item_i != ref_item and item_i.checkState() == Qt.Checked:
                  item_i.setCheckState(Qt.Unchecked)
         self.image_list.blockSignals(False)

         self.ref_image_idx = new_ref_idx
         self._update_rotation_textbox() # Update rot display
         self.updatePixmap(update_base=True) # Force regen of ref base pixmap

    # --- Image Data & Display ---
    def get_image_data(self, get_raw=False):
        # This function remains mostly the same, just returns data
        if not hasattr(self, 'image_data') or not self.image_data:
            return None, None
        num_images = len(self.image_data)
        ref_idx_valid = 0 <= self.ref_image_idx < num_images
        current_idx_valid = 0 <= self.current_image_idx < num_images
        if not (ref_idx_valid and current_idx_valid):
            # Try to return at least one valid image if possible
            ref_im = self.image_data[self.ref_image_idx]['image'] if ref_idx_valid else None
            current_im_data = self.image_data[self.current_image_idx]['image'] if current_idx_valid else None
            return ref_im, current_im_data

        ref_im = self.image_data[self.ref_image_idx]['image']
        current_im_data = self.image_data[self.current_image_idx]['image']
        return ref_im, current_im_data # Return raw data

    def _create_base_pixmap(self, image):
        if image is None:
            return None
        try:
            # Ensure data is uint8 and contiguous
            if image.dtype != np.uint8:
                 # Attempt conversion if safe (e.g., from bool or low-range int)
                 if np.issubdtype(image.dtype, np.integer) or image.dtype == bool:
                     image = image.astype(np.uint8) # Potential data loss if > 255
                 else:
                     print(f"Error: _create_base_pixmap received non-uint8 data type {image.dtype}")
                     return None # Cannot safely convert unknown types

            if not image.flags['C_CONTIGUOUS']:
                img_copy = np.ascontiguousarray(image)
            else:
                img_copy = image # Already contiguous

            height, width = img_copy.shape[:2]
            if img_copy.ndim == 3 and img_copy.shape[2] == 3:
                channel = 3
                q_format = QImage.Format_RGB888
            elif img_copy.ndim == 2: # Grayscale - convert to RGB for display consistency
                 img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)
                 height, width, channel = img_copy.shape
                 q_format = QImage.Format_RGB888
                 if not img_copy.flags['C_CONTIGUOUS']: # Check contiguity again after conversion
                     img_copy = np.ascontiguousarray(img_copy)
            else:
                print(f"Error: _create_base_pixmap received unsupported data shape {img_copy.shape}")
                return None

            bytes_per_line = channel * width
            q_image = QImage(img_copy.data, width, height, bytes_per_line, q_format)
            if q_image.isNull():
                print("Error: Failed QImage creation")
                return None
            # Crucial: Create a QPixmap from the QImage. The QImage must be kept alive
            # or the data buffer might become invalid. Returning the QPixmap ensures this.
            # A deep copy of the QImage might be safer if the underlying numpy array changes later.
            # pixmap = QPixmap.fromImage(q_image.copy()) # Use copy for safety
            pixmap = QPixmap.fromImage(q_image) # Usually sufficient if numpy array isn't immediately reused/changed
            return pixmap
        except Exception as e:
            print(f"Error creating base pixmap: {e}")
            traceback.print_exc() # Verbose
            return None

    def _prepare_base_pixmaps(self):
        ref_data, current_data = self.get_image_data(get_raw=True)
        self._ref_base_pixmap = self._create_base_pixmap(ref_data)
        self._current_base_pixmap = self._create_base_pixmap(current_data)
        #self._prepare_anchor_pixmap() # Removed

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
                pen_anchor.setWidth(2) # Use int width
                pen_anchor.setStyle(Qt.SolidLine)
                painter.setPen(pen_anchor)
                painter.drawRect(self.anchor_rect_img_coords) # Image coords

            # 2. Draw TEMP mouse drag selection rectangle (Dashed Red) if selecting
            if self.is_selecting and self.selection_start_point and self.selection_end_point:
                # Map label coords to image coords for drawing the temp rect
                p1_img = self.map_label_point_to_image_point(self.selection_start_point, self._ref_base_pixmap)
                p2_img = self.map_label_point_to_image_point(self.selection_end_point, self._ref_base_pixmap)
                if p1_img and p2_img:
                    temp_rect_img = QRect(p1_img, p2_img).normalized()
                    pen_temp = QPen(QColor(255, 0, 0, 180)) # Red
                    pen_temp.setWidth(2) # Use int width
                    pen_temp.setStyle(Qt.DashLine)
                    painter.setPen(pen_temp)
                    painter.drawRect(temp_rect_img) # Image coords

            # 3. Draw Reference Point Markers (Cyan)
            if self.ref_points: # Check if list is not empty
                pen_ref_pts = QPen(QColor("cyan"))
                pen_ref_pts.setWidth(2) # Use int width
                painter.setPen(pen_ref_pts)
                marker_radius = 4 # Radius in image pixels
                for img_point in self.ref_points:
                    # Ensure img_point is QPoint
                    if isinstance(img_point, QPoint):
                        painter.drawEllipse(img_point, marker_radius, marker_radius)
                        painter.drawLine(img_point.x() - marker_radius, img_point.y(), img_point.x() + marker_radius, img_point.y())
                        painter.drawLine(img_point.x(), img_point.y() - marker_radius, img_point.x(), img_point.y() + marker_radius)

            # 4. Draw Crosshair/Bullseye (Red) if enabled
            if hasattr(self, 'show_crosshair') and self.show_crosshair and img_w > 0 and img_h > 0:
                 center_x = img_w // 2
                 center_y = img_h // 2
                 pen_cross = QPen(QColor(255, 0, 0, 200)) # Red
                 pen_cross.setWidth(2) # Use int width
                 painter.setPen(pen_cross)
                 # Crosshair lines
                 painter.drawLine(0, center_y, img_w, center_y) # Horizontal
                 painter.drawLine(center_x, 0, center_x, img_h) # Vertical
                 # Concentric Rings
                 min_dim = min(img_w, img_h)
                 radii = [int(min_dim * r) for r in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]] # Example radii
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
                pen_cur_pts.setWidth(2) # Use int width
                painter_cur.setPen(pen_cur_pts)
                marker_radius = 4
                for img_point in self.current_points:
                     # Ensure img_point is QPoint
                    if isinstance(img_point, QPoint):
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
                # Ensure both are float32 for subtraction
                diff_im = np.abs(ref_data.astype(np.float32) - current_data.astype(np.float32))
                # Normalize difference to 0-255 for display
                min_val, max_val = np.min(diff_im), np.max(diff_im)
                if max_val > min_val:
                    diff_im = 255 * (diff_im - min_val) / (max_val - min_val)
                else:
                    diff_im = np.zeros_like(diff_im) # Avoid division by zero if flat

                diff_im_uint8 = np.clip(diff_im, 0, 255).astype(np.uint8)
                self._current_diff_pixmap = self._create_base_pixmap(diff_im_uint8)
            elif ref_data is not None and current_data is not None:
                print("Cannot create diff image: Shapes mismatch")
            # else: print("Cannot create diff image: Missing data")


    def _apply_zoom_and_set_pixmap(self, source_pixmap, label):
        if source_pixmap is None or source_pixmap.isNull():
            label.clear()
            label.setText("No Image Data") # More informative message
            label.adjustSize() # Adjust size even for text
            return
        try:
            original_size = source_pixmap.size()
            target_w = int(round(original_size.width() * self.zoom_factor))
            target_h = int(round(original_size.height() * self.zoom_factor))

            # Ensure minimum size of 1x1
            target_w = max(1, target_w)
            target_h = max(1, target_h)

            # Use SmoothTransformation only if upscaling significantly or downscaling
            # Qt.FastTransformation might be faster for near 1:1 or small changes
            transform_mode = Qt.SmoothTransformation # Default to smooth
            # if 0.8 < self.zoom_factor < 1.2: # Example threshold
            #     transform_mode = Qt.FastTransformation

            scaled_pixmap = source_pixmap.scaled(target_w, target_h, Qt.KeepAspectRatio, transform_mode)

            label.setPixmap(scaled_pixmap)
            label.adjustSize() # Adjust label size to scaled pixmap
        except Exception as e:
            print(f"Error applying zoom/setting pixmap: {e}")
            traceback.print_exc()
            label.setText("Error Display")
            label.adjustSize()


    def resizeEvent(self, event):
        # This might be useful later for auto-fitting on resize, but keep simple for now
        super().resizeEvent(event)
        # print(f"Window resized to: {event.size()}") # Debugging

    def update_list_widget(self, items):
        self.image_list.clear()
        if not items:
             self.ref_image_idx = -1
             self.current_image_idx = -1
             self.updatePixmap() # Clear display
             return

        self.image_list.blockSignals(True)
        for idx, item_name in enumerate(items):
            list_item = QListWidgetItem(item_name)
            list_item.setFlags(list_item.flags() | Qt.ItemIsUserCheckable)
            # Set first item as checked (reference) by default
            list_item.setCheckState(Qt.Checked if idx == 0 else Qt.Unchecked)
            self.image_list.addItem(list_item)
        self.image_list.blockSignals(False)

        # Set initial indices and select the first item
        self.ref_image_idx = 0
        self.current_image_idx = 0
        self.image_list.setCurrentRow(0) # Highlight the first item

        # Clear points when loading new folder
        self.ref_points = []
        self.current_points = []
        self._update_point_button_text()

        self._update_rotation_textbox() # Update rotation display for the new current image
        self.updatePixmap(update_base=True) # Update display with the first image pair

    # --- Add NEW Toggle Slots ---
    def toggle_set_ref_mode(self, checked):
        """Handles state when 'Set Ref Points' button is toggled."""
        # print(f"--- toggle_set_ref_mode called with checked={checked} ---") # DIAGNOSTIC
        self.setting_ref_points = checked
        if checked:
            # print("  Entering Ref Point Mode...") # DIAGNOSTIC
            # If the other button is checked, uncheck it programmatically
            if self.btn_set_cur_pts.isChecked():
                # print("  Unchecking Set Target Points button programmatically.") # DIAGNOSTIC
                self.btn_set_cur_pts.blockSignals(True) # Prevent its toggle slot from firing
                self.btn_set_cur_pts.setChecked(False)
                self.btn_set_cur_pts.setStyleSheet("") # Reset style
                self.btn_set_cur_pts.blockSignals(False)
                self.setting_current_points = False # Manually update state flag

            self.btn_set_ref_pts.setStyleSheet("background-color: lightblue;") # Indicate active
            QApplication.setOverrideCursor(Qt.CrossCursor)
            print("Mode: Setting Reference Points")
        else:
            # print("  Exiting Ref Point Mode...") # DIAGNOSTIC
            # Only restore cursor if the other button *isn't* now active
            if not self.setting_current_points:
                # print("  Restoring cursor (current points mode not active).") # DIAGNOSTIC
                QApplication.restoreOverrideCursor()
            # else:
                 # print("  Not restoring cursor (current points mode IS active).") # DIAGNOSTIC
            self.btn_set_ref_pts.setStyleSheet("") # Reset style
            print("Exited setting Reference Points mode.")
        # Update button text regardless of state change source
        self._update_point_button_text()
        # print(f"--- toggle_set_ref_mode finished ---") # DIAGNOSTIC

    def toggle_set_cur_mode(self, checked):
        """Handles state when 'Set Target Points' button is toggled."""
        # print(f"--- toggle_set_cur_mode called with checked={checked} ---") # DIAGNOSTIC
        self.setting_current_points = checked
        if checked:
            # print("  Entering Current Point Mode...") # DIAGNOSTIC
            # If the other button is checked, uncheck it programmatically
            if self.btn_set_ref_pts.isChecked():
                # print("  Unchecking Set Ref Points button programmatically.") # DIAGNOSTIC
                self.btn_set_ref_pts.blockSignals(True) # Prevent its toggle slot from firing
                self.btn_set_ref_pts.setChecked(False)
                self.btn_set_ref_pts.setStyleSheet("") # Reset style
                self.btn_set_ref_pts.blockSignals(False)
                self.setting_ref_points = False # Manually update state flag

            self.btn_set_cur_pts.setStyleSheet("background-color: lightgreen;") # Indicate active
            QApplication.setOverrideCursor(Qt.CrossCursor)
            print("Mode: Setting Current/Target Points")
        else:
            # print("  Exiting Current Point Mode...") # DIAGNOSTIC
            # Only restore cursor if the other button *isn't* now active
            if not self.setting_ref_points:
                # print("  Restoring cursor (ref points mode not active).") # DIAGNOSTIC
                QApplication.restoreOverrideCursor()
            # else:
                # print("  Not restoring cursor (ref points mode IS active).") # DIAGNOSTIC
            self.btn_set_cur_pts.setStyleSheet("") # Reset style
            print("Exited setting Current Points mode.")
        self._update_point_button_text()
        # print(f"--- toggle_set_cur_mode finished ---") # DIAGNOSTIC


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
        if self.ref_points:
            self.ref_points = []
            self._update_point_button_text() # Update count display
            self.updatePixmap(update_base=False) # Redraw ref image without points
            print("Reference points cleared.")

    def clear_current_points(self):
        """Clears only the current points."""
        if self.current_points:
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
            # Ensure points are QPoint objects
            if not all(isinstance(p, QPoint) for p in self.ref_points + self.current_points):
                 QMessageBox.critical(self, "Internal Error", "Point data is not in the expected format.")
                 return

            ref1, ref2 = self.ref_points[0], self.ref_points[1]
            cur1, cur2 = self.current_points[0], self.current_points[1]

            # Calculate vectors
            dx_ref = float(ref2.x() - ref1.x())
            dy_ref = float(ref2.y() - ref1.y())
            dx_curr = float(cur2.x() - cur1.x())
            dy_curr = float(cur2.y() - cur1.y())

            # Check for zero-length vectors
            if abs(dx_ref) < 1e-6 and abs(dy_ref) < 1e-6:
                QMessageBox.warning(self, "Invalid Points", "Reference points are too close or identical.")
                return
            if abs(dx_curr) < 1e-6 and abs(dy_curr) < 1e-6:
                QMessageBox.warning(self, "Invalid Points", "Current points are too close or identical.")
                return

            # Calculate angles using atan2
            angle_ref_rad = np.arctan2(dy_ref, dx_ref)
            angle_curr_rad = np.arctan2(dy_curr, dx_curr)

            # Calculate rotation difference (angle to rotate current TO match reference)
            # If ref angle is 30 deg and curr angle is 10 deg, we need to rotate curr by +20 deg
            rotation_rad = angle_ref_rad - angle_curr_rad
            rotation_deg = np.degrees(rotation_rad)

            # --- IMPORTANT: Apply rotation relative to current state ---
            # The calculated rotation_deg is the *additional* rotation needed.
            current_total_rotation = 0.0
            if 0 <= self.current_image_idx < len(self.image_data):
                 current_total_rotation = self.image_data[self.current_image_idx].get('total_rotation', 0.0)

            # The new target rotation is the current rotation PLUS the calculated difference
            # However, the convention might feel reversed depending on how points are picked.
            # Let's assume the goal is to rotate the *current* image so its points align with the *reference* points.
            # If current vector needs rotating CCW (positive angle) to match ref, rotation_deg will be positive.
            # We ADD this delta to the current image's total rotation.
            # Correction: OpenCV rotation is CW for positive angle. If we want CCW for positive delta, use -rotation_deg.
            # Let's stick to the calculated delta and let _perform_cv_rotation handle the direction.
            # The delta angle_ref - angle_curr is the angle needed to ADD to current to match ref.
            new_target_rotation = current_total_rotation + rotation_deg

            print(f"Current Total Rot: {current_total_rotation:.3f} deg")
            print(f"Calculated Rotation Delta from points: {rotation_deg:.3f} degrees")
            print(f"New Target Total Rotation: {new_target_rotation:.3f} deg")

            # Update the text box with the calculated potential NEW total rotation
            self.rot_val.setText(f"{new_target_rotation:.1f}") # Use .1f for consistency

            # Automatically apply this rotation
            print("Applying calculated rotation...")
            apply_text_rotation(self) # This function reads the text box and applies

        except Exception as e:
             QMessageBox.critical(self, "Error", f"Error calculating rotation from points: {e}")
             traceback.print_exc()


    # --- Input Validation Helpers ---
    def check_input_is_int(self, line_edit_widget, default_value, min_val=None, max_val=None):
        # (No change needed, already clean)
        val_str = line_edit_widget.text()
        valid = False
        corrected_val = default_value # Start with default
        try:
            val = int(val_str)
            valid = True
            corrected_val = val # Use parsed value initially
            if min_val is not None and val < min_val:
                corrected_val = min_val
                valid = False # Mark as invalid if correction needed
                print(f"Input corrected: Value must be >= {min_val}")
            if max_val is not None and val > max_val:
                corrected_val = max_val
                valid = False # Mark as invalid if correction needed
                print(f"Input corrected: Value must be <= {max_val}")
            val = corrected_val
        except ValueError:
            val = default_value
            print(f"Input '{val_str}' invalid integer, using default {default_value}.")
            valid = False

        # Update text only if the input was invalid or corrected
        if not valid or str(val) != val_str:
            line_edit_widget.setText(str(val))

    def check_input_is_float(self, line_edit_widget, default_value):
        # (No change needed, already clean)
        val_str = line_edit_widget.text().replace(',', '.')
        valid = False
        corrected_val = default_value # Start with default
        try:
            val = float(val_str)
            valid = True
            corrected_val = val # Use parsed value
        except ValueError:
            val = default_value
            print(f"Input '{val_str}' invalid float, using default {default_value:.1f}.")
            valid = False

        # Update text only if the input was invalid or format needs correction
        formatted_val_str = f"{val:.1f}"
        if not valid or formatted_val_str != val_str:
             line_edit_widget.setText(formatted_val_str)


    # --- Worker Setup and Control ---

    def _setup_progress_dialog(self, title, label_text, max_value):
        """Creates and configures the progress dialog."""
        self.progress_dialog = QProgressDialog(label_text, "Cancel", 0, max_value, self)
        self.progress_dialog.setWindowTitle(title)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setAutoReset(True) # Reset if max value reached
        self.progress_dialog.setAutoClose(True) # Close if max value reached or cancelled
        self.progress_dialog.setValue(0) # Start at 0

    def _set_ui_enabled(self, enabled):
        """Enable/disable relevant UI elements during worker execution."""
        # List widget and file buttons
        self.image_list.setEnabled(enabled)
        # --- Access buttons directly via self ---
        self.btn_open.setEnabled(enabled)
        self.btn_save_all.setEnabled(enabled)
        self.btn_save_current.setEnabled(enabled)
        # --- End direct access ---

        # Registration controls
        self.reg_method_fft_radio.setEnabled(enabled)
        self.reg_method_scan_radio.setEnabled(enabled)
        self.reg_method_scan_rot_radio.setEnabled(enabled)
        self.btn_register_all.setEnabled(enabled)
        self.btn_register_current.setEnabled(enabled)

        # Anchor controls
        self.anchor_x.setEnabled(enabled)
        self.anchor_y.setEnabled(enabled)
        self.anchor_w.setEnabled(enabled)
        self.anchor_h.setEnabled(enabled)
        self.btn_apply_anchor.setEnabled(enabled)
        # Only enable clear anchor if an anchor exists AND UI is enabled
        self.btn_clear_anchor.setEnabled(enabled and self.anchor_rect_img_coords is not None)

        # Manual Adjust controls
        self.btn_left.setEnabled(enabled)
        self.btn_right.setEnabled(enabled)
        self.btn_up.setEnabled(enabled)
        self.btn_down.setEnabled(enabled)
        self.shift_val.setEnabled(enabled)
        self.btn_rot_l.setEnabled(enabled)
        self.btn_rot_r.setEnabled(enabled)
        self.rot_val.setEnabled(enabled)
        self.btn_apply_rotation.setEnabled(enabled)

        # Point selection controls
        self.btn_set_ref_pts.setEnabled(enabled)
        self.btn_set_cur_pts.setEnabled(enabled)
        self.btn_clear_ref_points.setEnabled(enabled)
        self.btn_clear_current_points.setEnabled(enabled)
        self.btn_get_rotation.setEnabled(enabled)

        # Morph controls
        self.btn_morph.setEnabled(enabled)
        self.fps.setEnabled(enabled)

        # Zoom controls (usually okay to leave enabled)
        # self.findChild(QPushButton, 'btn_zoom_out').setEnabled(enabled)
        # ...

    def start_registration(self, mode='all'):
        """Starts the registration process in a worker thread."""
        if self.worker_thread is not None and self.worker_thread.isRunning():
            QMessageBox.warning(self, "Busy", "Another process is already running.")
            return

        # --- Initial Checks (similar to original function) ---
        if not hasattr(self, 'image_data') or len(self.image_data) < 2:
            QMessageBox.warning(self, "Not Ready", "Need at least two images loaded.")
            return
        num_images = len(self.image_data)
        if not (0 <= self.ref_image_idx < num_images):
             QMessageBox.warning(self, "Invalid Reference", f"Reference image index ({self.ref_image_idx}) is invalid.")
             return

        # Determine registration method
        use_fft_method = self.reg_method_fft_radio.isChecked()
        use_scan_method = self.reg_method_scan_radio.isChecked()
        use_scan_rot_method = self.reg_method_scan_rot_radio.isChecked()
        reg_method_str = 'fft' if use_fft_method else ('scan' if use_scan_method else 'scan_rot')

        print(f"Starting registration thread using {reg_method_str} method.")

        # --- Determine Indices ---
        indices_to_register = []
        if mode == 'all':
            indices_to_register = [i for i in range(num_images) if i != self.ref_image_idx]
            if not indices_to_register:
                QMessageBox.information(self, "Register All", "No other images to register to the reference.")
                return
            print(f"Processing all ({len(indices_to_register)}) images...")
        elif mode == 'current':
            if self.current_image_idx == self.ref_image_idx:
                QMessageBox.information(self, "Register Current", "Current image is the reference image. Cannot register.")
                return
            if not (0 <= self.current_image_idx < num_images):
                QMessageBox.warning(self, "Invalid Selection", "Invalid current image index selected.")
                return
            indices_to_register = [self.current_image_idx]
            print(f"Processing current image (index {self.current_image_idx})...")
        else:
            QMessageBox.warning(self, "Error", f"Unknown registration mode '{mode}'.")
            return

        if not indices_to_register:
            QMessageBox.information(self, "Register", f"No images selected for registration ('{mode}' mode).")
            return

        # --- Prepare Reference Image Data ---
        try:
            ref_image_full_color = self.image_data[self.ref_image_idx]['image']
            # Convert to grayscale float32 once
            ref_grey = np.dot(ref_image_full_color[..., :3].astype(np.float32), [0.2989, 0.5870, 0.1140])
        except Exception as e:
            QMessageBox.critical(self, "Registration Error", f"Error preparing reference image:\n{e}")
            traceback.print_exc()
            return

        # --- Prepare Reference Anchor (if needed) ---
        ref_anchor = None
        anchor_details = {} # Store x,y,w,h if scan method used
        if self.anchor_rect_img_coords:
            try:
                rect = self.anchor_rect_img_coords
                anc_x = rect.left()
                anc_y = rect.top()
                anc_w = rect.width()
                anc_h = rect.height()
                anchor_details = {'x': anc_x, 'y': anc_y, 'w': anc_w, 'h': anc_h} # Pass details
                # Check anchor validity against reference image
                if not (0 <= anc_y < anc_y + anc_h <= ref_grey.shape[0] and
                        0 <= anc_x < anc_x + anc_w <= ref_grey.shape[1]):
                    QMessageBox.critical(self, "Registration Error", f"Anchor rectangle is invalid for reference image dimensions {ref_grey.shape}.")
                    return
                # Extract the anchor patch (as float32 for consistency)
                ref_anchor = ref_grey[anc_y : anc_y + anc_h, anc_x : anc_x + anc_w].astype(np.float32)
            except Exception as e:
                 QMessageBox.critical(self, "Registration Error", f"Error preparing reference anchor:\n{e}")
                 traceback.print_exc()
                 return
        else:
            # Check if anchor is required but missing
            if use_scan_method or use_scan_rot_method:
                QMessageBox.warning(self, "Anchor Required", f"{reg_method_str.upper()} method requires an anchor area to be defined (draw or input values and click Apply).")
                return # Stop registration

        # --- Get Shift Value (for Scan SSD step) ---
        try:
            shift_val_step = int(self.shift_val.text())
            if shift_val_step <= 0: shift_val_step = 1
        except ValueError:
            shift_val_step = 1

        # --- Create Worker and Thread ---
        # Pass copies of data where necessary, although worker primarily reads
        # Pass the list of dicts - worker will access 'image', 'image_orig', 'total_rotation'
        # Make a deep copy of the list structure and potentially numpy arrays if modification is feared
        # For now, assume worker reads and emits updates, main thread applies updates.
        worker_image_data = copy.deepcopy(self.image_data) # Deep copy to isolate worker data

        self.worker = RegistrationWorker(
            image_data_list=worker_image_data, # Pass the deep copy
            ref_image_idx=self.ref_image_idx,
            indices_to_register=indices_to_register,
            reg_method=reg_method_str,
            anchor_details=anchor_details, # Pass dict
            ref_grey=ref_grey.copy(), # Pass copy
            ref_anchor=ref_anchor.copy() if ref_anchor is not None else None, # Pass copy
            shift_val=shift_val_step
        )
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)

        # --- Connect Signals ---
        self.worker.progress.connect(self.handle_registration_progress)
        self.worker.finished.connect(self.handle_registration_finished)
        self.worker.error.connect(self.handle_worker_error)
        self.worker.log_message.connect(self.handle_worker_log)
        self.worker.image_updated.connect(self.handle_image_update)
        self.worker.rotation_updated.connect(self.handle_rotation_update)

        self.worker_thread.started.connect(self.worker.run)
        # Cleanup connections
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._on_worker_thread_finished) # General cleanup slot

        # --- Setup Progress Dialog ---
        num_to_process = len(indices_to_register)
        self._setup_progress_dialog("Registering Images", "Registering...", num_to_process)
        self.progress_dialog.canceled.connect(self.cancel_worker) # Connect cancel button
        self.progress_dialog.show()

        # --- Disable UI and Start ---
        self._set_ui_enabled(False)
        self.worker_thread.start()

    def start_morphing(self):
        """Starts the morphing process in a worker thread."""
        if self.worker_thread is not None and self.worker_thread.isRunning():
            QMessageBox.warning(self, "Busy", "Another process is already running.")
            return

        # --- Initial Checks ---
        if not hasattr(self, 'image_data') or len(self.image_data) < 2:
            QMessageBox.warning(self, "Not Enough Images", "Need at least 2 images loaded to morph.")
            return

        # Get FPS
        try:
            frame_rate = int(self.fps.text())
            if frame_rate < 2:
                raise ValueError("FPS must be >= 2")
        except ValueError:
            frame_rate = 5
            self.fps.setText('5')
            QMessageBox.warning(self, "Invalid FPS", "FPS must be an integer >= 2. Using default 5.")
            # Proceed with default FPS

        # Get Save details
        save_filter = "PNG Sequence (*.png);;JPEG Sequence (*.jpg *.jpeg);;BMP Sequence (*.bmp)"
        start_dir = self.last_morph_folder if self.last_morph_folder and os.path.isdir(self.last_morph_folder) else os.getcwd()
        suggested_filename = os.path.join(start_dir, "frame_.png")
        fileName, selectedFilter = QFileDialog.getSaveFileName(self,"Select Base Filename and Format for Morph",suggested_filename, save_filter)
        if not fileName:
            return # User cancelled save dialog

        # Process filename and extension
        save_folder = os.path.dirname(fileName)
        base_name_with_ext = os.path.basename(fileName)
        base_name, save_ext = os.path.splitext(base_name_with_ext)
        # Ensure base name ends with a separator for frame numbers
        if not base_name.endswith(("_", "-", ".")):
            base_name += "_"
        # Determine save extension based on filter or default to png
        if "JPEG" in selectedFilter:
            save_ext = ".jpg"
        elif "BMP" in selectedFilter:
            save_ext = ".bmp"
        else: # Default or PNG selected
            save_ext = ".png"

        # Save config and create dir
        self.last_morph_folder = save_folder
        self.save_config()
        try:
            os.makedirs(save_folder, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Folder Error", f"Could not create save folder:\n{save_folder}\n{e}")
            return

        print(f"Starting morph thread (FPS={frame_rate}) into: {save_folder} as '{base_name}*{save_ext}'")

        # --- Create Worker and Thread ---
        # Pass a deep copy of image data to prevent issues if main thread modifies it
        worker_image_data = copy.deepcopy(self.image_data)

        self.worker = MorphWorker(
            image_data_list=worker_image_data, # Pass the deep copy
            frame_rate=frame_rate,
            save_folder=save_folder,
            base_name=base_name,
            save_ext=save_ext
        )
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)

        # --- Connect Signals ---
        self.worker.progress.connect(self.handle_morph_progress)
        self.worker.finished.connect(self.handle_morph_finished)
        self.worker.error.connect(self.handle_worker_error)
        self.worker.log_message.connect(self.handle_worker_log)

        self.worker_thread.started.connect(self.worker.run)
        # Cleanup connections
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._on_worker_thread_finished) # General cleanup slot

        # --- Setup Progress Dialog ---
        num_images = len(self.image_data)
        total_expected_frames = (num_images - 1) * (frame_rate - 1) + num_images if num_images > 0 else 0
        if total_expected_frames <= 0: total_expected_frames = 1 # Avoid 0 max
        self._setup_progress_dialog("Morphing Images", "Generating frames...", total_expected_frames)
        self.progress_dialog.canceled.connect(self.cancel_worker) # Connect cancel button
        self.progress_dialog.show()

        # --- Disable UI and Start ---
        self._set_ui_enabled(False)
        self.worker_thread.start()

    @pyqtSlot()
    def cancel_worker(self):
        """Requests cancellation of the currently running worker, if it exists."""
        print("Cancel requested by user.")
        # --- ADD CHECK HERE ---
        if self.worker is not None and hasattr(self.worker, 'cancel'):
            # Only call cancel if the worker object still exists
            try:
                self.worker.cancel()
                if self.progress_dialog:
                    self.progress_dialog.setLabelText("Cancelling...")
            except RuntimeError as e:
                # Catch the specific error if it happens due to a race condition
                # where the worker is deleted between the check and the call.
                print(f"Error calling worker cancel (likely already deleted): {e}")
        else:
            print("Worker already finished or does not exist.")
            # Optionally, ensure the dialog label reflects this if it's still visible
            # if self.progress_dialog:
            #     self.progress_dialog.setLabelText("Process finished.")

    # --- Worker Signal Handlers ---

    @pyqtSlot(int, int, str)
    def handle_registration_progress(self, current_image_num, total_images, image_name):
        """Updates the progress dialog during registration."""
        if self.progress_dialog is not None:
            try: # Add try block
                self.progress_dialog.setValue(current_image_num)
                self.progress_dialog.setLabelText(f"Registering {current_image_num}/{total_images}: {image_name}")
            except AttributeError: # Catch error if dialog becomes None unexpectedly
                print("Warning: Progress dialog became None during registration progress update.")
            except RuntimeError as e: # Catch potential C++ object deleted error
                 print(f"Warning: Runtime error accessing progress dialog during registration update: {e}")

    @pyqtSlot(int, int)
    def handle_registration_finished(self, registered_count, error_count):
        """Handles completion of the registration worker."""
        print("Registration thread finished.")
        if self.progress_dialog:
            self.progress_dialog.close() # Close dialog cleanly

        # Display results
        if error_count > 0:
            QMessageBox.warning(self, "Registration Issues", f"Registration finished.\nProcessed: {registered_count}\nErrors: {error_count}\nCheck console log for details.")
        elif registered_count > 0:
            QMessageBox.information(self, "Registration Complete", f"Successfully registered {registered_count} image(s).")
        else:
            # This case might happen if cancelled early or no images processed
             QMessageBox.information(self, "Registration Info", f"Registration process finished.\nProcessed: {registered_count}\nErrors: {error_count}")

        # Update display with potentially modified images
        # UpdatePixmap might be slow if many images changed; consider updating only current view if needed
        self.updatePixmap(update_base=True)
        # Also update rotation text box if the current image was rotated
        self._update_rotation_textbox()


    @pyqtSlot(int, np.ndarray)
    def handle_image_update(self, index, modified_image_array):
        """Updates the image data in the main list when worker sends update.
           Switches the 'Current Image' view to show this newly processed image.
        """
        if 0 <= index < len(self.image_data):
            # Always update the underlying data store
            self.image_data[index]['image'] = modified_image_array
            print(f"Received image update for index {index}. Switching view.")

            # --- Switch the Current Image View ---
            # 1. Update the internal current index
            self.current_image_idx = index

            # 2. Update the selection highlight in the QListWidget
            # Block signals temporarily to prevent item_changed from firing and potentially
            # clearing points or causing other side effects of manual selection change.
            self.image_list.blockSignals(True)
            self.image_list.setCurrentRow(index)
            self.image_list.blockSignals(False)

            # 3. Regenerate the base pixmap for the *new* current image
            self._current_base_pixmap = self._create_base_pixmap(modified_image_array)

            # 4. Update the difference pixmap if needed (uses the new current image)
            # Note: This requires the reference pixmap (_ref_base_pixmap) to be up-to-date.
            # If the reference image itself could be processed (unlikely but possible),
            # we might need to ensure _ref_base_pixmap is also updated. Assuming ref is fixed during run.
            if self.radio_buttons['rad_diff'].isChecked():
                self._update_diff_pixmap()

            # 5. Redraw the pixmaps with overlays and zoom.
            # update_base=False prevents regenerating base pixmaps again.
            self.updatePixmap(update_base=False)

            # 6. Update the rotation text box to match the new current image
            self._update_rotation_textbox()
            # --- End View Switch ---

        else:
            print(f"Warning: Received image update for invalid index {index}")

    @pyqtSlot(int, float)
    def handle_rotation_update(self, index, new_total_rotation):
        """Updates the rotation data. If it's the currently viewed image, update the text box."""
        if 0 <= index < len(self.image_data):
            print(f"Received rotation update for index {index}: {new_total_rotation:.2f}")
            current_data = self.image_data[index]
            # Rely on apply_text_rotation/apply_M_transformation for 'image_orig' creation
            current_data['total_rotation'] = new_total_rotation

            # Update text box ONLY if this rotation update corresponds to the
            # image *currently* being viewed (which might have just been switched by handle_image_update)
            if index == self.current_image_idx:
                self.rot_val.setText(f"{new_total_rotation:.1f}")
        else:
            print(f"Warning: Received rotation update for invalid index {index}")

    @pyqtSlot(int, int)
    def handle_morph_progress(self, current_frame, total_frames):
        """Updates the progress dialog during morphing."""
        if self.progress_dialog is not None:
            try: # Add try block
                # It's possible total_frames estimate changes slightly, update max if needed
                if self.progress_dialog.maximum() != total_frames:
                     self.progress_dialog.setMaximum(total_frames)
                self.progress_dialog.setValue(current_frame)
                self.progress_dialog.setLabelText(f"Generating frame {current_frame}/{total_frames}...")
            except AttributeError: # Catch error if dialog becomes None unexpectedly
                print("Warning: Progress dialog became None during morph progress update.")
            except RuntimeError as e: # Catch potential C++ object deleted error
                 print(f"Warning: Runtime error accessing progress dialog during morph update: {e}")

    @pyqtSlot(int)
    def handle_morph_finished(self, generated_count):
        """Handles completion of the morphing worker."""
        print("Morphing thread finished.")
        if self.progress_dialog:
            self.progress_dialog.close()

        # Use the saved folder path from before starting the worker
        save_folder = self.last_morph_folder # Assumes it was set correctly
        QMessageBox.information(self, "Morph Complete", f"Generated {generated_count} frames in\n{save_folder}")


    @pyqtSlot(str)
    def handle_worker_error(self, error_message):
        """Handles critical errors reported by a worker."""
        print(f"Worker Error: {error_message}")
        if self.progress_dialog:
            self.progress_dialog.close()
        QMessageBox.critical(self, "Worker Error", f"An error occurred during processing:\n{error_message}\nCheck console log.")
        # UI should be re-enabled by _on_worker_thread_finished

    @pyqtSlot(str)
    def handle_worker_log(self, message):
        """Prints log messages from the worker to the console."""
        print(f"Worker: {message}") # Simple console logging

    @pyqtSlot()
    def _on_worker_thread_finished(self):
        """General cleanup actions after any worker thread finishes."""
        print("Worker thread finished signal received. Cleaning up.")
        self._set_ui_enabled(True) # Re-enable UI
        # Reset worker/thread attributes
        self.worker_thread = None
        self.worker = None
        # Ensure progress dialog is closed if it wasn't already
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        print("UI re-enabled.")

    def closeEvent(self, event):
        """Ensure worker is stopped if window is closed."""
        if self.worker_thread is not None and self.worker_thread.isRunning():
            print("Window closing, attempting to cancel worker...")
            self.cancel_worker()
            # Give the thread a moment to finish cancelling
            if not self.worker_thread.wait(1000): # Wait 1 second
                 print("Worker thread did not stop gracefully, terminating.")
                 self.worker_thread.terminate() # Force stop if needed
                 self.worker_thread.wait() # Wait for termination
        event.accept()


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
