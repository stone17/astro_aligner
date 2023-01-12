import os
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QLineEdit, QFrame, QPushButton, QGridLayout, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QObject, QEvent
import numpy as np
import yaml
import imageio

from image_registration import chi2_shift
from image_registration.fft_tools import shift

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the user interface
        self.initUI()

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

        # Create a list widget to display the image names
        self.image_list = QListWidget(self)
        self.image_list.setFixedWidth(200)
        self.image_list.itemClicked.connect(self.updatePixmap)
        self.image_list.currentItemChanged.connect(self.updatePixmap)

        # Create a label to display the image
        self.label = QLabel(self)
        self.label.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)

        # Create a central widget and set its layout
        central_widget = QFrame(self)
        central_widget.setMinimumWidth(600)
        central_widget.setMinimumHeight(400)
        layout = QGridLayout(central_widget)
        layout_buttons = QGridLayout()
        layout_buttons.addWidget(btn_open, 0, 0)
        layout_buttons.addWidget(btn_register, 0, 1)
        layout_buttons.addWidget(btn_save, 0, 2)
        layout_buttons.addWidget(btn_morph, 0, 3)
        layout_buttons.addWidget(self.fps, 1, 3)
        layout_images = QHBoxLayout()
        layout_images.addWidget(self.label)
        layout_images.addWidget(self.image_list)
        layout.addLayout(layout_buttons, 0, 0)
        layout.addLayout(layout_images, 1, 0)

        # Set the central widget of the main window
        self.setCentralWidget(central_widget)

        # Create an event filter to handle the resizeEvent event
        event_filter = EventFilter(self.label)
        # Install the event filter on the main window
        self.installEventFilter(event_filter)

    def updatePixmap(self, event):
        # Check if a folder has been selected and if there are any pixmaps
        if not hasattr(self, 'folder') or not hasattr(self, 'images'):
            return

        selected_item = self.image_list.currentItem()
        if selected_item is None:
            index = 0
        else:
            index = self.image_list.row(selected_item)
        image_data = self.images[index]

        # Get the size of the main window and the pixmap
        if self.pixmap is not None:
            pixmap_size = self.pixmap.size()
            window_size = self.size()

            # Check if the size of the main window has changed
            if window_size != pixmap_size:
                pass

        # Convert the array to a QImage
        image = QImage(
            image_data.data,
            image_data.shape[1],
            image_data.shape[0],
            image_data.strides[0],
            QImage.Format_RGB888
        )
        # Convert the QImage to a QPixmap
        self.pixmap = QPixmap.fromImage(image)
        self.label.setPixmap(self.pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

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
            self.pixmap = None
            self.image_list.clear()
            for file in os.listdir(folder):
                if file.endswith(".jpg") or file.endswith(".JPG"):
                    image_path = os.path.join(folder, file)
                    self.image_names.append(file)
                    self.image_paths.append(image_path)
                    self.images.append(imageio.imread(image_path))
            # Check if any images were found
            if self.images:
                # Scale the first pixmap to fit the display
                self.image_list.addItems(self.image_names)
                self.updatePixmap(None)
            print("Loaded {} images".format(len(self.images)))

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

        selected_item = self.image_list.currentItem()
        if selected_item is None:
            ref_index = 0
        else:
            ref_index = self.image_list.row(selected_item)

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


class EventFilter(QObject):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Resize:
            # Check if a folder has been selected and if there are any pixmaps
            if not hasattr(self.main_window, 'folder') or not hasattr(self.main_window, 'pixmaps'):
                return False

            # Get the size of the main window and the pixmap
            window_size = self.main_window.size()
            pixmap_size = self.main_window.pixmaps[0].size()

            # Check if the size of the main window has changed
            if window_size != pixmap_size:
                # Scale the first pixmap to fit the display
                pixmap = self.main_window.pixmaps[0]
                self.main_window.label.setPixmap(pixmap.scaled(self.main_window.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        return


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
