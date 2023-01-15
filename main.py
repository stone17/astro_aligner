import os
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QRadioButton, QLabel, QLineEdit, QFrame, QPushButton,
    QGridLayout, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QButtonGroup
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QObject, QEvent
import numpy as np
import yaml
import imageio
from functools import partial

from image_registration import chi2_shift
from image_registration.fft_tools import shift
import image_editing as image_edit

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the user interface
        self.initUI()
        self.ref_image_idx = 0
        self.current_image_idx = 0
        self.installEventFilter(self)
        self.keyPressEvent = on_key_press

    def on_key_press(event):
        if event.key() in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right):
            print("Arrow key pressed")

    def initUI(self):
        # Load the last used folder from the config file, if present
        self.folder = None
        if os.path.exists('config.yaml'):
            try:
                with open('config.yaml', 'r') as f:
                    config = yaml.safe_load(f)
                    if 'folder' in config:
                        self.folder = config['folder']
            except Exception as e:
                print(e)

        # Create a central widget and set its layout
        central_widget = QFrame(self)
        central_widget.setMinimumWidth(800)
        central_widget.setMinimumHeight(800)
        self.layout = QGridLayout(central_widget)

        self.init_buttons()
        self.init_image_frames()
        self.init_bottom_frame()

        # Set the central widget of the main window
        self.setCentralWidget(central_widget)

    def init_buttons(self):
        button_frame = QFrame(self)
        button_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        button_frame.setMaximumHeight(70)

        # Create a button to open a file dialog
        btn_open = QPushButton('Open folder', self)
        btn_open.clicked.connect(self.loadFolder)

        # Create a button to register the images
        btn_register = QPushButton('Register images', self)
        btn_register.clicked.connect(self.registerImages)

        # Create a button to save the images
        btn_save = QPushButton('Save images', self)
        btn_save.clicked.connect(self.saveImages)

        # Create a button to save the images
        btn_morph = QPushButton('Morph images', self)
        btn_morph.clicked.connect(self.morphImages)

        # Create a button to save the images
        self.fps = QLineEdit('Frame rate', self)
        self.fps.setText('5')
        self.fps.editingFinished.connect(self.check_frame_rate)

        # Create a button to save the images
        text_fps = QLabel('Frame rate', self)

        layout_buttons = QGridLayout(button_frame)
        layout_buttons.setContentsMargins(1, 1, 1, 1)
        layout_buttons.addWidget(btn_open, 0, 0)
        layout_buttons.addWidget(btn_register, 0, 1)
        layout_buttons.addWidget(btn_save, 0, 2)
        layout_buttons.addWidget(btn_morph, 0, 3, 1, 2)
        layout_buttons.addWidget(text_fps, 1, 3)
        layout_buttons.addWidget(self.fps, 1, 4)

        self.layout.addWidget(button_frame, 0, 0)

    def init_image_frames(self):
        # Create a list widget to display the image names
        self.image_list = QListWidget(self)
        self.image_list.setFixedWidth(200)
        self.image_list.itemClicked.connect(self.item_changed)
        self.image_list.currentItemChanged.connect(self.item_changed)
        self.image_list.itemChanged.connect(self.item_changed)

        # Create a labels to display the images
        self.ref_image = QLabel(self)
        self.ref_image.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        text_ref = QLabel('Reference Image', self)
        text_ref.setAlignment(Qt.AlignCenter)
        text_ref.setMaximumHeight(20)

        self.current_image = QLabel(self)
        self.current_image.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        text_current = QLabel('Current Image', self)
        text_current.setAlignment(Qt.AlignCenter)
        text_current.setMaximumHeight(20)
        layout_images = QGridLayout()
        layout_images.addWidget(text_ref, 0, 0)
        layout_images.addWidget(self.ref_image, 1, 0)
        layout_images.addWidget(text_current, 2, 0)
        layout_images.addWidget(self.current_image, 3, 0)
        layout_images.addWidget(self.image_list, 0, 1, 4, 1)

        self.layout.addLayout(layout_images, 1, 0)

    def init_bottom_frame(self):
        bottom_frame = QFrame(self)
        bottom_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        bottom_frame.setMaximumHeight(40)
        bottom_layout = QGridLayout(bottom_frame)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        # Add radio buttons to group and layout
        radio_frame = QFrame(self)
        radio_layout = QHBoxLayout(radio_frame)
        # Create radio buttons
        self.radio_buttons = {
            'rad_normal': QRadioButton("Original"),
            'rad_diff': QRadioButton("Difference"),
        }
        radio_group = QButtonGroup()
        for button in self.radio_buttons:
            b = self.radio_buttons[button]
            radio_group.addButton(b)
            radio_layout.addWidget(b)
            b.toggled.connect(self.updatePixmap)
        self.radio_buttons['rad_normal'].setChecked(True)
        bottom_layout.addWidget(radio_frame)

        # Add buttons to shift and rotate image
        shift_buttons = {
            'btn_left': QPushButton('Left', self),
            'btn_right': QPushButton('Right', self),
            'btn_up': QPushButton('Up', self),
            'btn_down': QPushButton('Down', self),
            'btn_rot_l': QPushButton('Rotate Left', self),
            'btn_rot_r': QPushButton('Rotate Right', self),
        }
        
        idx = 1
        for button in shift_buttons:
            b = shift_buttons[button]
            b.clicked.connect(self.shift_image)
            bottom_layout.addWidget(b, 0, idx)
            idx += 1

        self.layout.addWidget(bottom_frame, 2, 0)

    def item_changed(self, item):
        if item is None:
            return

        self.image_list.blockSignals(True)
        if item.checkState() == Qt.Checked:
            for i in range(self.image_list.count()):
                if self.image_list.item(i) != item:
                    self.image_list.item(i).setCheckState(Qt.Unchecked)
                else:
                    self.ref_image_idx = i
        else:
            checked_item_found = False
            for i in range(self.image_list.count()):
                if self.image_list.item(i).checkState() == Qt.Checked:
                    self.ref_image_idx = i
                    checked_item_found = True
                    break
            if not checked_item_found:
                self.image_list.item(0).setCheckState(Qt.Checked)
                self.ref_image_idx = 0

        selected_item = self.image_list.currentItem()
        if selected_item is None:
            self.current_image_idx = 0
        else:
            self.current_image_idx = self.image_list.row(selected_item)
        self.updatePixmap()
        self.image_list.blockSignals(False)

    def get_image_data(self):
        ref_im = self.images[self.ref_image_idx]
        current_im = self.images[self.current_image_idx].copy()

        if self.radio_buttons['rad_diff'].isChecked():
            current_im = np.abs(ref_im - current_im)

        return [ref_im, current_im]

    def updatePixmap(self):
        # Check if a folder has been selected and if there are any pixmaps
        if not hasattr(self, 'folder') or not hasattr(self, 'images'):
            return

        im_data = self.get_image_data()
        im_labels = [self.ref_image, self.current_image]
        pixmaps = [self.ref_pixmap, self.current_pixmap]

        for image_data, label, pixmap in zip(im_data, im_labels, pixmaps):
            # Convert the array to a QImage
            image = QImage(
                image_data.data,
                image_data.shape[1],
                image_data.shape[0],
                image_data.strides[0],
                QImage.Format_RGB888
            )
            # Convert the QImage to a QPixmap
            pixmap = QPixmap.fromImage(image)
            label.setPixmap(pixmap.scaled(self.ref_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def loadFolder(self):
        # Set the default path for the file dialog to the last used folder, if available
        default_path = self.folder if self.folder else os.getcwd()
        # Show a file dialog to select a folder
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", default_path)
        if folder:
            # Save the selected folder to the config file
            with open('config.yaml', 'w') as f:
                yaml.dump({'folder': folder}, f)
            # Load the image files from the folder into QPixmaps
            self.images = []
            self.image_names = []
            self.image_paths = []
            self.ref_pixmap = None
            self.current_pixmap = None
            for file in os.listdir(folder):
                if file.endswith(".jpg") or file.endswith(".JPG"):
                    image_path = os.path.join(folder, file)
                    self.image_names.append(file)
                    self.image_paths.append(image_path)
                    self.images.append(imageio.imread(image_path))
            # Update the listwidget
            self.update_list_widget(self.image_names)
            # Check if any images were found
            #if self.images:
            #    self.updatePixmap(None)
            print("Loaded {} images".format(len(self.images)))

    def update_list_widget(self, items):
        self.image_list.clear()
        for idx, item in enumerate(items):
            list_item = QListWidgetItem(item)
            list_item.setFlags(list_item.flags() | Qt.ItemIsUserCheckable)
            list_item.setCheckState(Qt.Unchecked)
            self.image_list.addItem(list_item)
            if idx == 0:
                list_item.setCheckState(Qt.Checked)
            else:
                list_item.setCheckState(Qt.Unchecked)

    def saveImages(self):
        # Check if there are any images
        if not hasattr(self, 'images'):
            print("No images were found in the selected folder")
            return

        default_path = self.folder if self.folder else os.getcwd()
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", default_path)
        if folder:
            for idx, image in enumerate(self.images):
                # Save the Image to a file
                orig_name = self.image_names[idx][:-4]
                file_name = f"{orig_name}_reg.jpg"
                imageio.imwrite(os.path.join(folder, file_name), image)

    def shift_image(self):
        mode = self.sender().text()
        current_im = self.images[self.current_image_idx]
        if mode == 'Left':
            current_im = np.roll(current_im, -1, axis=1)
        elif mode == 'Right':
            current_im = np.roll(current_im, 1, axis=1)
        elif mode == 'Up':
            current_im = np.roll(current_im, -1, axis=0)
        elif mode == 'Down':
            current_im = np.roll(current_im, 1, axis=0)
        elif mode == 'Rotate Left':
            current_im = image_edit.rotate_image(current_im, -1)
        elif mode == 'Rotate Right':
            current_im = image_edit.rotate_image(current_im, 1)

        self.images[self.current_image_idx] = current_im

        self.updatePixmap()

    def morphImages(self):
        # Check if there are any images
        if not hasattr(self, 'images'):
            print("No images were found in the selected folder")
            return

        default_path = self.folder if self.folder else os.getcwd()
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", default_path)

        if folder:
            frame_rate = int(self.fps.text())
            counter = 1
            for idx, img1 in enumerate(self.images):
                if idx == len(self.images) - 1:
                    break

                img2 = self.images[idx+1]

                # Create an empty array to store the interpolated images
                interpolated_images = []

                # Interpolate between the two images
                for i in range(frame_rate - 1):
                    # Calculate the interpolation factor
                    alpha = i / (frame_rate - 1)
                    # Interpolate the pixel values
                    img = img1 * (1 - alpha) + img2 * alpha
                    # Convert the interpolated image to 8-bit unsigned integers
                    img = img.astype(np.uint8)
                    # save the images to disk
                    padded_index = str(counter).zfill(3)
                    print(padded_index)
                    filename = f'{padded_index}.jpg'
                    imageio.imwrite(os.path.join(folder, filename), img)
                    counter += 1

    def registerImages(self):
        # Check if a folder has been selected
        if not hasattr(self, 'folder'):
            print("No folder has been selected")
            return

        # Check if there are any images
        if not hasattr(self, 'images'):
            print("No images were found in the selected folder")
            return

        for i in range(self.image_list.count()):
            if self.image_list.item(i).checkState() == Qt.Checked:
                ref_index = i
                break

        # Convert the images to grayscale
        grey_images = [np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]) for image in self.images]

        # Register the images against the first image
        registered_images = []
        for idx, image in enumerate(grey_images):
            if idx == ref_index:
                continue
            xoff, yoff, exoff, eyoff = chi2_shift(
                grey_images[ref_index],
                grey_images[idx],
                1,
                return_error=True,
                upsample_factor='auto'
            )
            print('Image {} of {}: Xoff: {}, Yoff: {}'.format(idx+1, len(grey_images), xoff, yoff))
            corrected_image = np.roll(self.images[idx], (-int(yoff), -int(xoff)), axis=(0, 1)).copy()
            self.images[idx] = corrected_image

    def check_frame_rate(self):
        fps = self.fps.text()
        try:
            fps = int(fps)
        except Exception:
            print("Frame rate needs to be an integer")
            self.fps.setText('5')

def on_key_press(event):
    if event.key() in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right):
        print("Arrow key pressed")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
