import os
import sys
import numpy as np
import imageio
import traceback # For better error reporting

from PyQt5.QtWidgets import (QFileDialog, QMessageBox)


# --- File Loading ---
def loadFolder(self):
    print(self.last_load_folder)
    start_dir = self.last_load_folder if self.last_load_folder and os.path.isdir(self.last_load_folder) else os.getcwd()
    folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing Images", start_dir)

    if folder:
        self.last_load_folder = folder
        self.save_config()
        # Reset state
        self.ref_image_idx = -1
        self.image_data = []
        self.current_image_idx = -1
        self.clear_anchor_area() # Also clears pixmaps
        self._ref_base_pixmap = None
        self._current_base_pixmap = None

        print(f"Loading images from: {folder}")
        supported_load_ext = (".jpg", ".jpeg", ".png", ".bmp", ".fits", ".fit", ".fts")
        try:
             files = sorted([f for f in os.listdir(folder) if f.lower().endswith(supported_load_ext)])
             if not files:
                  QMessageBox.warning(self, "No Images", f"No supported images found.")
                  self.updatePixmap()
                  return

             load_count = 0
             fail_count = 0

             num_files = 0
             for file in files:
                if file.lower().endswith((".fits", ".fit", ".fts", ".jpg", ".jpeg", ".png", ".bmp")):
                    num_files +=1
             for idx, file in enumerate(files):
                image_path = os.path.join(folder, file)
                file_lower = file.lower()
                img_data = None # Reset before load attempt

                try:
                    if file_lower.endswith((".fits", ".fit", ".fts")):
                        print(f'File {idx+1:5d}/{num_files:d}', end='\r')
                        if _astropy_check_ok:
                            print(f"Loading FITS file: {file}")
                            with astro_fits.open(image_path, memmap=False) as hdul:
                                hdu_index = 0
                                if len(hdul) > 1 and hdul[0].data is None:
                                    hdu_index = 1 # Try first extension if primary is empty
                                if hdu_index < len(hdul) and hdul[hdu_index].data is not None:
                                    fits_data = hdul[hdu_index].data
                                    if fits_data.ndim > 2:
                                        fits_data = fits_data[0, :, :] # Simple slice
                                    img_data_uint8 = _scale_fits_to_uint8(fits_data)
                                    if img_data_uint8 is not None:
                                        if img_data_uint8.ndim == 2:
                                            img_data = np.stack((img_data_uint8,) * 3, axis=-1) # Gray to RGB
                                        elif img_data_uint8.ndim==3 and img_data_uint8.shape[2]==3:
                                             img_data = img_data_uint8 # Already RGB?
                        # else: Astropy not available

                    elif file_lower.endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        print(f'File {idx+1:5d}/{num_files:d}', end='\r')
                        img_data_raw = imageio.imread(image_path)
                        # Convert common formats to RGB uint8
                        if img_data_raw.ndim == 2:
                            img_data = np.stack((img_data_raw,) * 3, axis=-1).astype(np.uint8)
                        elif img_data_raw.ndim == 3 and img_data_raw.shape[2] == 4: # RGBA
                            img_data = img_data_raw[..., :3].astype(np.uint8)
                        elif img_data_raw.ndim == 3 and img_data_raw.shape[2] == 3: # RGB
                            img_data = img_data_raw.astype(np.uint8)
                        # else: Unsupported shape

                    if img_data is not None and img_data.ndim==3 and img_data.shape[2]==3 and img_data.dtype==np.uint8:
                         self.image_data.append({
                             'name': file,
                             'path': image_path,
                             'image': img_data,
                             'total_rotation': 0.0
                         })
                         load_count += 1
                    else:
                         # Only count as fail if img_data stayed None or format was wrong
                         if img_data is None and file_lower.endswith(supported_load_ext):
                              print(f"Failed to load/process image data for {file}.")
                              fail_count += 1
                         # else: wasn't a supported extension anyway or skipped due to shape

                except Exception as read_err:
                    print(f"Error reading/processing file {file}: {read_err}")
                    fail_count += 1

             print(f"Loading finished. Loaded: {load_count}, Failed/Skipped: {fail_count}")
             if not self.image_data: # Check the correct list
                 QMessageBox.warning(self, "Load Failed", "Could not load any valid images.")
                 self.updatePixmap()
             else:
                 names_only = [item['name'] for item in self.image_data]
                 self.update_list_widget(names_only) # Pass only names
                 self._update_rotation_textbox() # Set initial rotation display
                 self.fit_view()
        except Exception as e:
             print(f"Error reading folder contents: {e}")
             QMessageBox.critical(self, "Loading Error", f"Error reading image folder:\n{e}")
             self.updatePixmap()
             traceback.print_exc() 

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

    image_list_to_save, image_names_to_save = [], []
    if 'all' in mode:
        if not self.image_data:
            QMessageBox.warning(self, "No Images", "No images to save.")
            return

        for image in self.image_data:
            image_list_to_save.append(image['image'])
            image_names_to_save.append(image['name'])
    else: # Save current
         if not (0 <= self.current_image_idx < len(self.image_data)):
            QMessageBox.warning(self, "Invalid Selection", "Invalid index.")
            return
         image_list_to_save = [self.image_data[self.current_image_idx]['image']]
         image_names_to_save = [self.image_data[self.current_image_idx]['name']]

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
        except Exception as e:
            print(f"Error saving image {image_names_to_save[idx]}: {e}")
            errors += 1
    print(f"Save complete. Saved: {saved_count}, Errors: {errors}")
    QMessageBox.information(self, "Save Complete", f"Saved {saved_count} image(s).\nErrors: {errors}")