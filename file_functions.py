import os
import sys
import numpy as np
import imageio
import traceback # For better error reporting
import time
from datetime import datetime, timezone # Added timezone for robust timestamp conversion
from PIL import Image
from PIL.ExifTags import TAGS
from PyQt5.QtWidgets import (QFileDialog, QMessageBox)


# --- File Loading ---
def loadFolder(self): # 'self' is the MainWindow instance
    print(f"Attempting load. Last folder: {self.last_load_folder}")
    start_dir = self.last_load_folder if self.last_load_folder and os.path.isdir(self.last_load_folder) else os.getcwd()
    folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing Images", start_dir)

    if folder:
        self.last_load_folder = folder
        self.save_config() # Assumes save_config is a method on MainWindow instance (self)

        # Reset MainWindow state attributes via self
        self.image_data = [] # Clear previous image data
        self.ref_image_idx = -1
        self.current_image_idx = -1
        self.image_list.clear()
        self.clear_anchor_area()
        self._ref_base_pixmap = None
        self._current_base_pixmap = None
        self._ref_anchor_pixmap = None
        self._current_diff_pixmap = None

        print(f"Loading images from: {folder}")
        supported_load_ext = (".jpg", ".jpeg", ".png", ".bmp", ".fits", ".fit", ".fts")

        try:
            all_files_in_dir = os.listdir(folder)
            # Filter AND sort the files
            files = sorted([f for f in all_files_in_dir if f.lower().endswith(supported_load_ext)])

            if not files:
                QMessageBox.warning(self, "No Images", f"No supported images found in folder.")
                # Call MainWindow's updatePixmap if needed (though state is already cleared)
                # self.updatePixmap() # Probably not needed here as list update triggers it
                return

            load_count = 0
            fail_count = 0
            num_files = len(files) # Count only supported files

            print(f"Found {num_files} supported files. Starting load...")
            for idx, file in enumerate(files):
                image_path = os.path.join(folder, file)
                file_lower = file.lower()
                img_data = None # Reset for each file attempt
                timestamp = None # Reset timestamp for each file
                header = None # For FITS header

                # Print progress before try block
                print(f'File {idx+1:>{len(str(num_files))}}/{num_files}: {file}', end='\r')

                try:
                    # --- Load Image Data ---
                    if file_lower.endswith((".fits", ".fit", ".fts")):
                        if _astropy_check_ok:
                            with astro_fits.open(image_path, memmap=False) as hdul:
                                hdu_index = 0
                                if len(hdul) > 1 and hdul[0].data is None:
                                    hdu_index = 1
                                if hdu_index < len(hdul) and hdul[hdu_index].data is not None:
                                    fits_data = hdul[hdu_index].data
                                    header = hdul[hdu_index].header # Store header
                                    if fits_data.ndim > 2:
                                        # Simple slice, assuming first axis is non-spatial
                                        fits_data = fits_data[0, :, :]
                                    img_data_uint8 = _scale_fits_to_uint8(fits_data) # Use helper
                                    if img_data_uint8 is not None:
                                        if img_data_uint8.ndim == 2:
                                            img_data = np.stack((img_data_uint8,) * 3, axis=-1)
                                        elif img_data_uint8.ndim == 3 and img_data_uint8.shape[2] == 3:
                                            img_data = img_data_uint8
                        else:
                            # Skip if astropy not available
                            pass # Fail count incremented later if img_data is None

                    elif file_lower.endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        img_data_raw = imageio.imread(image_path)
                        # Convert standard formats to RGB uint8
                        if img_data_raw.ndim == 2:
                            img_data = np.stack((img_data_raw,) * 3, axis=-1).astype(np.uint8)
                        elif img_data_raw.ndim == 3 and img_data_raw.shape[2] == 4: # RGBA
                            img_data = img_data_raw[..., :3].astype(np.uint8)
                        elif img_data_raw.ndim == 3 and img_data_raw.shape[2] == 3: # RGB
                            img_data = img_data_raw.astype(np.uint8)
                        # else shape unsupported, img_data remains None

                    # --- Extract Timestamp (after successful data load) ---
                    if img_data is not None:
                        try:
                            if file_lower.endswith((".fits", ".fit", ".fts")) and header is not None:
                                date_obs = header.get('DATE-OBS', header.get('DATE', None))
                                if date_obs:
                                    try:
                                        if isinstance(date_obs, str):
                                            dt_object = datetime.fromisoformat(date_obs)
                                        elif isinstance(date_obs, datetime):
                                            dt_object = date_obs
                                        else:
                                            dt_object = None
                                        if dt_object:
                                            if dt_object.tzinfo is None:
                                                dt_object = dt_object.replace(tzinfo=timezone.utc)
                                            timestamp = dt_object.timestamp()
                                    except Exception as ts_parse_err:
                                        print(f"\nWarning: Could not parse FITS date '{date_obs}': {ts_parse_err}")
                            elif file_lower.endswith((".jpg", ".jpeg")):
                                try:
                                    with Image.open(image_path) as pil_img:
                                        exif = pil_img._getexif()
                                        if exif:
                                            exif_dict = {TAGS.get(k, k): v for k, v in exif.items()}
                                            date_str = exif_dict.get('DateTimeOriginal', exif_dict.get('DateTime'))
                                            if date_str:
                                                dt_object = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
                                                timestamp = dt_object.timestamp()
                                except AttributeError:
                                    pass # Ignore if _getexif missing
                                except Exception as exif_err:
                                    print(f"\nWarning: Error reading EXIF for {file}: {exif_err}")

                            # Fallback Timestamp
                            if timestamp is None:
                                try:
                                    mtime = os.path.getmtime(image_path)
                                    timestamp = mtime
                                    print(f"\nWarning: Using file modification time for {file}")
                                except Exception as stat_err:
                                    print(f"\nError getting file mtime for {file}: {stat_err}")

                        except Exception as ts_err:
                            print(f"\nError extracting timestamp for {file}: {ts_err}")

                    # --- Append Data to MainWindow's list ---
                    if img_data is not None:
                        self.image_data.append({
                            'name': file,
                            'path': image_path,
                            'image': img_data,
                            'total_rotation': 0.0,
                            'timestamp': timestamp # Add the timestamp (can be None)
                        })
                        load_count += 1
                    else:
                        # Failed loading image data itself
                        if file_lower.endswith(supported_load_ext):
                             print(f"\nFailed to load/process image data for {file}.")
                             fail_count += 1

                except Exception as read_err:
                    # Catch errors during the loading/processing of a single file
                    print(f"\nError reading/processing file {file}: {read_err}")
                    fail_count += 1
                    # traceback.print_exc() # Optional for detailed debug

            print(f"\nLoading finished. Loaded: {load_count}, Failed/Skipped: {fail_count}")
            if not self.image_data: # Check if list is empty after loop
                QMessageBox.warning(self, "Load Failed", "Could not load any valid images.")
                self.updatePixmap() # Clear display
            else:
                # --- Sort images by timestamp if possible ---
                valid_timestamp_count = sum(1 for item in self.image_data if item.get('timestamp') is not None)
                if valid_timestamp_count == len(self.image_data):
                    print("Sorting images by timestamp...")
                    self.image_data.sort(key=lambda item: item['timestamp'])
                    # Get names AFTER sorting
                    names_only = [item['name'] for item in self.image_data]
                else:
                    print("Warning: Not all images have valid timestamps, using original file order.")
                    names_only = [item['name'] for item in self.image_data] # Use original order

                # Call MainWindow methods to update UI
                self.update_list_widget(names_only)
                self._update_rotation_textbox() # Update rotation display
                self.fit_view() # Set initial zoom

        except Exception as e:
             # Error reading the directory itself
             print(f"Error reading folder contents: {e}")
             QMessageBox.critical(self, "Loading Error", f"Error reading image folder:\n{e}")
             # traceback.print_exc() # Optional for detailed debug
             # Ensure UI is cleared / reset
             self.update_list_widget([])
             self.updatePixmap()

# --- FITS Scaling Helper ---
def _scale_fits_to_uint8(self, data, p_low=1.0, p_high=99.0):
    try:
        img = data.astype(np.float32)
        if np.all(np.diff(img.ravel()) == 0): # Check if flat more reliably
            return np.zeros_like(img, dtype=np.uint8)

        finite_img = img[np.isfinite(img)]
        if finite_img.size == 0: return np.zeros_like(img, dtype=np.uint8)

        vmin, vmax = np.percentile(finite_img, [p_low, p_high])

        if vmax <= vmin:
             vmin = np.min(finite_img)
             vmax = np.max(finite_img)
             if vmax <= vmin: return np.zeros_like(img, dtype=np.uint8)

        img[~np.isfinite(img)] = vmin # Replace non-finite with min
        img = np.clip(img, vmin, vmax)
        denominator = vmax - vmin
        if denominator < 1e-8: denominator = 1e-8
        img = (img - vmin) / denominator
        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img_uint8
    except Exception as e:
        print(f"Error during FITS scaling: {e}")
        return None

# --- Saving ---
def saveImages(self):
    if not hasattr(self, 'image_data') or not self.image_data:
        QMessageBox.warning(self, "No Images", "No images loaded.")
        return
    sender_button = self.sender()
    mode = sender_button.text() if sender_button else 'all'
    print(sender_button.text())

    image_list_to_save, image_names_to_save, timestamps = [], [], []
    if 'all' in mode:
        if not self.image_data:
            QMessageBox.warning(self, "No Images", "No images to save.")
            return

        for image in self.image_data:
            image_list_to_save.append(image['image'])
            image_names_to_save.append(image['name'])
            timestamps.append(image['timestamp'])
    else: # Save current
         if not (0 <= self.current_image_idx < len(self.image_data)):
            QMessageBox.warning(self, "Invalid Selection", "Invalid index.")
            return
         image_list_to_save.append(self.image_data[self.current_image_idx]['image'])
         image_names_to_save.append(self.image_data[self.current_image_idx]['name'])
         timestamps.append(self.image_data[self.current_image_idx]['timestamp'])

    save_filter = "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg);;BMP Images (*.bmp)"
    start_dir = self.last_save_folder if self.last_save_folder and os.path.isdir(self.last_save_folder) else os.getcwd()
    save_folder = start_dir
    save_ext = ".png"

    if 'all' in mode:
        save_dialog_title = "Select Folder and Format for Saving All"
        suggested_filename = os.path.join(start_dir, f"{os.path.splitext(image_names_to_save[0])[0]}_reg.png")
        fileName, selectedFilter = QFileDialog.getSaveFileName(self, save_dialog_title, suggested_filename, save_filter)
        if not fileName: return
        save_folder = os.path.dirname(fileName)
        if "JPEG" in selectedFilter:
            save_ext = ".jpg"
        elif "BMP" in selectedFilter:
            save_ext = ".bmp"
        else: save_ext = ".png"
        print(f"Saving all images to folder: {save_folder} as {save_ext}")
    else: # Save current
        save_dialog_title = "Save Current Image As"
        suggested_filename = os.path.join(start_dir, f"{image_names_to_save[0]}_reg.png")
        fileName, selectedFilter = QFileDialog.getSaveFileName(self, save_dialog_title, suggested_filename, save_filter)
        if not fileName: return
        save_folder = os.path.dirname(fileName)
        _, actual_ext = os.path.splitext(fileName)
        if actual_ext.lower() not in ['.png', '.jpg', '.jpeg', '.bmp']:
            QMessageBox.warning(self, "Invalid Extension", f"Unsupported extension '{actual_ext}'. Saving as .png")
            save_ext = ".png"
            fileName = os.path.splitext(fileName)[0] + save_ext
        else: save_ext = actual_ext
        print(f"Saving current image as: {fileName}")

    self.last_save_folder = save_folder
    self.save_config()
    saved_count, errors = 0, 0
    for idx, image_data in enumerate(image_list_to_save):
        try:
            if 'all' in mode:
                orig_name, _ = os.path.splitext(image_names_to_save[idx])
                file_name_only = f"{orig_name}_reg{save_ext}"
                save_path = os.path.join(save_folder, file_name_only)
            else: save_path = fileName # Full path for single file
            print(f"  Saving: {os.path.basename(save_path)}")
            imageio.imwrite(save_path, image_data)
            saved_count += 1

            # --- Set Modification Time ---
            stored_timestamp = timestamps[idx]
            if stored_timestamp is not None:
                try:
                    # os.utime takes (path, (atime, mtime)) - use same for both
                    os.utime(save_path, (stored_timestamp, stored_timestamp))
                    # Optional: Convert timestamp back to readable string for printing
                    dt_obj = datetime.fromtimestamp(stored_timestamp)
                    print(f"    Set modification time to: {dt_obj.strftime('%Y-%m-%d %H:%M:%S')}")
                except TypeError:
                     print(f"    Warning: Invalid timestamp type ({type(stored_timestamp)}) for {img_name}. Cannot set mtime.")
                except OSError as e: # Catch potential permission errors etc.
                    print(f"    Warning: Could not set modification time for {os.path.basename(save_path)}: {e}")
                except Exception as e: # Catch other errors like invalid timestamp value
                     print(f"    Warning: Error setting modification time for {os.path.basename(save_path)}: {e}")
                     # traceback.print_exc() # More detail if needed
            else:
                print(f"    Warning: No original timestamp available for {img_name}. Modification time not set.")
            # --- End Set Modification Time ---

        except Exception as e:
            print(f"Error saving image {image_names_to_save[idx]}: {e}")
            errors += 1
    print(f"Save complete. Saved: {saved_count}, Errors: {errors}")
    QMessageBox.information(self, "Save Complete", f"Saved {saved_count} image(s).\nErrors: {errors}")