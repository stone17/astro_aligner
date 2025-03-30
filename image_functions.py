import numpy as np
import cv2
import imageio
import traceback
from image_registration import chi2_shift
from astropy.io import fits as astro_fits

from PyQt5.QtWidgets import (QFileDialog, QMessageBox)

# --- Other Actions ---
def translate_image(self, direction):
    # Check if images are loaded
    if not hasattr(self, 'images') or not self.images:
        print("No images loaded.")
        return
    # Check if current index is valid
    num_images = len(self.images)
    if not (0 <= self.current_image_idx < num_images):
        print(f"Invalid index {self.current_image_idx}")
        return
    # Get and validate shift value
    try:
        shift_val = int(self.shift_val.text())
        if shift_val <= 0:
            raise ValueError("Shift must be positive")
    except ValueError:
        shift_val = 1
        self.shift_val.setText('1')

    # Get image and dimensions
    current_im = self.images[self.current_image_idx]
    rows, cols = current_im.shape[:2]
    shifted_im = current_im.copy()
    print(f"Translating {direction} by {shift_val}px")

    # Apply roll and fill edges based on direction
    # Check shift value against dimensions to avoid rolling > image size
    if direction == 'Left':
        if shift_val < cols:
            shifted_im = np.roll(shifted_im, -shift_val, axis=1)
            shifted_im[:, -shift_val:] = 0 # Fill revealed area with black
        else:
            print("Shift value too large (left)")
            return # Or fill whole image? Return for now.
    elif direction == 'Right':
        if shift_val < cols:
            shifted_im = np.roll(shifted_im, shift_val, axis=1)
            shifted_im[:, :shift_val] = 0 # Fill revealed area
        else:
            print("Shift value too large (right)")
            return
    elif direction == 'Up':
        if shift_val < rows:
            shifted_im = np.roll(shifted_im, -shift_val, axis=0)
            shifted_im[-shift_val:, :] = 0 # Fill revealed area
        else:
            print("Shift value too large (up)")
            return
    elif direction == 'Down':
         if shift_val < rows:
            shifted_im = np.roll(shifted_im, shift_val, axis=0)
            shifted_im[:shift_val, :] = 0 # Fill revealed area
         else:
            print("Shift value too large (down)")
            return
    else:
        print(f"Unknown direction or shift too large: {direction}, {shift_val}")
        return

    # Update image data and display
    self.images[self.current_image_idx] = shifted_im
    # Force update of base pixmap for current image as data changed
    self.updatePixmap(update_base=True)

def rotate_image(self, mode):
    # Check if images are loaded
    if not hasattr(self, 'images') or not self.images:
        return
    # Check if current index is valid
    num_images = len(self.images)
    if not (0 <= self.current_image_idx < num_images):
        return
    # Get current rotation value from UI
    try:
        current_rot_val_in_box = float(self.rot_val.text().replace(',', '.'))
    except ValueError:
        current_rot_val_in_box = 0.0
        self.rot_val.setText('0.0')

    angle_step = 0.1 # Rotation step for buttons
    angle_to_apply = 0.0

    # Determine angle based on button click
    if mode == 'Left': # CCW button click
        angle_to_apply = -angle_step
        new_rot_val = current_rot_val_in_box + angle_to_apply
        self.rot_val.setText(f"{new_rot_val:.1f}") # Update UI text
    elif mode == 'Right': # CW button click
        angle_to_apply = angle_step
        new_rot_val = current_rot_val_in_box + angle_to_apply
        self.rot_val.setText(f"{new_rot_val:.1f}") # Update UI text
    else:
        # Currently, rotation triggered by direct value entry is disabled
        print("Rotation via value entry disabled. Use buttons.")
        return # Exit if mode is not Left or Right

    # Get the image to rotate
    image_to_rotate = self.images[self.current_image_idx]

    # Perform rotation using external library/module
    try:
        print(f"Applying incremental rotation of {angle_to_apply:.1f} degrees")
        rotated_im = image_edit.rotate_image(image_to_rotate, angle_to_apply, reshape=False) # Keep size
        # Update image data in memory, ensuring correct dtype
        self.images[self.current_image_idx] = rotated_im.astype(np.uint8)
        # Update the display (force update of base pixmap as data changed)
        self.updatePixmap(update_base=True)
    except Exception as e:
        # Handle errors during the rotation process
        print(f"Error during rotation: {e}")
        traceback.print_exc()

def morphImages(self):
        # Initial checks
        if not hasattr(self, 'images') or len(self.images) < 2: # OK
            QMessageBox.warning(self, "Not Enough Images", "Need >= 2 images.") # OK
            return # OK
        # Get FPS
        try: # OK
            frame_rate = int(self.fps.text()) # OK
            if frame_rate < 2: # OK
                raise ValueError("FPS must be >= 2") # OK
        except ValueError: # OK
            frame_rate = 5 # OK
            self.fps.setText('5') # OK
            QMessageBox.warning(self, "Invalid FPS", "Using default 5.") # OK
            # Proceed with default FPS, don't return

        # Get Save details
        save_filter = "PNG Sequence (*.png);;JPEG Sequence (*.jpg *.jpeg);;BMP Sequence (*.bmp)" # String assignment OK
        start_dir = self.last_morph_folder if self.last_morph_folder and os.path.isdir(self.last_morph_folder) else os.getcwd() # OK
        suggested_filename = os.path.join(start_dir, "frame_.png") # OK
        fileName, selectedFilter = QFileDialog.getSaveFileName(self,"Select Base Filename and Format for Morph",suggested_filename, save_filter) # OK
        if not fileName: # OK
            return # OK

        # Process filename and extension
        save_folder = os.path.dirname(fileName) # OK
        base_name_with_ext = os.path.basename(fileName) # OK
        base_name, save_ext = os.path.splitext(base_name_with_ext) # OK
        if not base_name.endswith(("_", "-", ".")): # OK
            base_name += "_" # OK
        if save_ext.lower() not in ['.png', '.jpg', '.jpeg', '.bmp']: # OK
            save_ext = ".png" # OK

        # Save config and create dir
        self.last_morph_folder = save_folder # OK
        self.save_config() # OK
        os.makedirs(save_folder, exist_ok=True) # OK
        print(f"Starting morph (FPS={frame_rate}) into: {save_folder} as '{base_name}*{save_ext}'") # OK
        counter = 0 # OK
        num_images = len(self.images) # OK
        generated_count = 0 # OK
        try: # OK
             # Save first frame
             img_first = self.images[0] # OK
             padded_index = str(counter).zfill(5) # OK
             filename = f'{base_name}{padded_index}{save_ext}' # OK
             save_path = os.path.join(save_folder, filename) # OK
             imageio.imwrite(save_path, img_first) # OK
             generated_count += 1 # OK
             counter += 1 # OK

             # Loop through pairs
             for idx in range(num_images - 1): # OK
                  img1 = self.images[idx].astype(np.float32) # OK
                  img2 = self.images[idx + 1].astype(np.float32) # OK
                  # Check shape mismatch
                  if img1.shape != img2.shape: # OK
                       print(f"Shape mismatch {idx}-{idx+1}. Skipping interp.") # OK
                       img_next = self.images[idx+1] # OK
                       padded_index = str(counter).zfill(5) # OK
                       filename = f'{base_name}{padded_index}{save_ext}' # OK
                       save_path = os.path.join(save_folder, filename) # OK
                       imageio.imwrite(save_path, img_next) # OK
                       generated_count += 1 # OK
                       counter += 1 # OK
                       continue # OK

                  # Interpolate frames
                  num_steps = frame_rate - 1 # OK
                  for i in range(num_steps): # OK
                       alpha = (i + 1.0) / frame_rate # OK
                       interp_img = np.clip(img1 * (1.0 - alpha) + img2 * alpha, 0, 255).astype(np.uint8) # OK
                       padded_index = str(counter).zfill(5) # OK
                       filename = f'{base_name}{padded_index}{save_ext}' # OK
                       save_path = os.path.join(save_folder, filename) # OK
                       imageio.imwrite(save_path, interp_img) # OK
                       generated_count += 1 # OK
                       counter += 1 # OK

                  # Save second image of pair (end frame)
                  img_second = self.images[idx+1] # OK
                  padded_index = str(counter).zfill(5) # OK
                  filename = f'{base_name}{padded_index}{save_ext}' # OK
                  save_path = os.path.join(save_folder, filename) # OK
                  imageio.imwrite(save_path, img_second) # OK
                  generated_count += 1 # OK
                  counter += 1 # OK
                  # Progress update
                  if counter % 20 == 0 or idx == num_images - 2: # OK
                      print(f"Generated frames up to index {counter-1}...") # OK

             # Final messages
             print(f"Finished morphing. Generated {generated_count} frames.") # OK
             QMessageBox.information(self, "Morph Complete", f"Generated {generated_count} frames in {save_folder}.") # OK
        except Exception as e: # OK
             print(f"An error during morphing: {e}") # OK
             traceback.print_exc() # OK
             QMessageBox.critical(self, "Morph Error", f"An error occurred:\n{e}") # OK

# --- Registration ---

def registerImages(self):
    # Main entry point for registration. Handles checks, loops, method selection, and application.

    # --- Initial Checks ---
    if not hasattr(self, 'images') or len(self.images) < 2:
        QMessageBox.warning(self, "Not Ready", "Need at least two images loaded.")
        return
    num_images = len(self.images)
    if not (0 <= self.ref_image_idx < num_images):
         QMessageBox.warning(self, "Invalid Reference", f"Ref index ({self.ref_image_idx}) invalid.")
         return

    sender_button = self.sender()
    mode = sender_button.text() if sender_button else 'all'
    use_fft_method = self.reg_method_fft_radio.isChecked()
    use_scan_method = self.reg_method_scan_radio.isChecked()

    print(f"Starting registration using {'FFT' if use_fft_method else 'Scan SSD'} method.")

    # --- Determine Indices ---
    indices_to_register = []
    if 'all' in mode:
        indices_to_register = [i for i in range(num_images) if i != self.ref_image_idx]
        if not indices_to_register:
            QMessageBox.information(self, "Register All", "No other images to register.")
            return
        print(f"Processing all ({len(indices_to_register)}) images...")
    else: # Register current
        if self.current_image_idx == self.ref_image_idx:
            QMessageBox.information(self, "Register Current", "Current is reference.")
            return
        if not (0 <= self.current_image_idx < num_images):
            QMessageBox.warning(self, "Invalid Selection", "Invalid current index.")
            return
        indices_to_register = [self.current_image_idx]
        print(f"Processing current image (index {self.current_image_idx})...")

    if not indices_to_register:
        QMessageBox.information(self, "Register", f"No images selected ('{mode}' mode).")
        return

    # --- Prepare Reference Image ---
    try:
        ref_image_full_color = self.images[self.ref_image_idx]
        # Convert to grayscale float32 once
        ref_grey = np.dot(ref_image_full_color[..., :3].astype(np.float32), [0.2989, 0.5870, 0.1140])
    except Exception as e:
        QMessageBox.critical(self, "Registration Error", f"Error preparing reference image:\n{e}")
        return

    # --- Prepare Reference Anchor for Scan SSD (if needed) ---
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
                QMessageBox.critical(self, "Registration Error", f"Anchor invalid for reference image shape {ref_grey.shape}.")
                return
            # Extract the anchor patch
            ref_anchor = ref_grey[anc_y : anc_y + anc_h, anc_x : anc_x + anc_w]
        except Exception as e:
             QMessageBox.critical(self, "Registration Error", f"Error preparing reference anchor:\n{e}")
             return
    else:
        if use_scan_method:
            QMessageBox.warning(self, "Anchor Required", "Scan SSD method requires an anchor area to be defined and applied.")
            return # Stop registration if Scan selected but no anchor

    # --- Registration Loop ---
    registered_count = 0
    errors = 0
    for idx in indices_to_register:
        print(f"--- Processing Image {idx} ('{self.image_names[idx]}') ---")
        try:
            current_image_full_color = self.images[idx]
            current_grey = np.dot(current_image_full_color[..., :3].astype(np.float32), [0.2989, 0.5870, 0.1140])

            shift_x_int, shift_y_int = 0, 0 # Initialize shift for this image

            # --- Call appropriate registration method ---
            if use_fft_method:
                if ref_anchor is not None:
                    # Extract anchor details
                    y_start = anchor_details['y']
                    y_end = y_start + anchor_details['h']
                    x_start = anchor_details['x']
                    x_end = x_start + anchor_details['w']
                    current_grey = current_grey[y_start:y_end, x_start:x_end]
                    ref_grey = ref_anchor

                xoff, yoff = _register_fft(self, ref_grey, current_grey)
                if xoff is not None and yoff is not None: # Check for success
                    shift_y_int = -int(round(yoff))
                    shift_x_int = -int(round(xoff))
                else:
                    errors += 1 # Error occurred in _register_fft
                    continue # Skip to next image

            elif use_scan_method:
                # Anchor validity checked before loop, ref_anchor should exist
                if ref_anchor is None: # Should not happen if check above worked
                     print(f"  Internal Error: ref_anchor not prepared for Scan SSD. Skipping image {idx}.")
                     errors += 1
                     continue

                best_dx, best_dy = _register_scan_ssd(self, ref_anchor, current_grey, anchor_details)
                # _register_scan_ssd prints its own result, just apply the shift
                shift_y_int = -best_dy
                shift_x_int = -best_dx
                # Add error handling if _register_scan_ssd could fail (e.g., return None)

            # --- Apply Calculated Shift ---
            if abs(shift_y_int) > 0 or abs(shift_x_int) > 0:
                print(f"  Applying final shift (dX:{shift_x_int}, dY:{shift_y_int})")
                corrected_image = np.roll(current_image_full_color, (shift_y_int, shift_x_int), axis=(0, 1))
                # Fill edges
                if shift_y_int > 0:
                    corrected_image[:shift_y_int, :] = 0 # Top
                elif shift_y_int < 0:
                    corrected_image[shift_y_int:, :] = 0 # Bottom
                if shift_x_int > 0:
                    corrected_image[:, :shift_x_int] = 0 # Left
                elif shift_x_int < 0:
                    corrected_image[:, shift_x_int:] = 0 # Right
                # Update the image in memory
                self.images[idx] = corrected_image.copy()
            else:
                print(f"  Image {idx}: Calculated shift is zero, no change applied.")

            registered_count += 1

        except Exception as e:
            # Catch errors during processing of a single image
            print(f"Error registering image {idx} ('{self.image_names[idx]}'): {e}")
            traceback.print_exc()
            errors += 1
    # --- End Registration Loop ---

    print(f"--- Registration Finished ---")
    print(f"Processed: {registered_count}, Errors: {errors}")
    if errors > 0:
        QMessageBox.warning(self, "Registration Issues", f"Finished with {errors} error(s). Check console.")
    elif registered_count > 0:
        QMessageBox.information(self, "Registration Complete", f"Successfully processed {registered_count} image(s).")

    # Update display, force update of base pixmaps as image data may have changed
    self.updatePixmap(update_base=True)


def _register_fft(self, ref_grey, current_grey):
    """Registers one image using FFT (chi2_shift). Returns (xoff, yoff) or (None, None)."""
    print(f"  Using FFT (chi2_shift)...")

    try:
        # upsample_factor improves subpixel precision but slows it down
        xoff, yoff, exoff, eyoff = chi2_shift(
            ref_grey,
            current_grey,
            upsample_factor='auto', # Or try 10, 100 etc.
            return_error=True
        )
        print(f'  FFT Offset Found (X:{xoff:.3f}, Y:{yoff:.3f})')
        return xoff, yoff
    except Exception as fft_err:
        print(f"  Error during FFT registration: {fft_err}")
        # traceback.print_exc() # Optional detailed traceback
        return None, None


def _register_scan_ssd(self, ref_anchor, current_grey, anchor_details):
    """Registers one image using Scan SSD. Returns (best_dx, best_dy)."""
    print(f"  Using Scan SSD...")
    try:
        step_size = int(self.shift_val.text())
        if step_size <= 0:
            step_size = 1
    except ValueError:
        step_size = 1
    scan_range_pixels = 15 # Define scan range

    # Extract anchor details
    anc_x = anchor_details['x']
    anc_y = anchor_details['y']
    anc_w = anchor_details['w']
    anc_h = anchor_details['h']

    img_h, img_w = current_grey.shape # Dimensions of current image

    min_ssd = np.inf
    best_dx = 0
    best_dy = 0

    # Scan loop
    for dy in range(-scan_range_pixels, scan_range_pixels + 1, step_size):
        for dx in range(-scan_range_pixels, scan_range_pixels + 1, step_size):
            # Current anchor coordinates in the potentially shifted image
            curr_y_start = anc_y + dy
            curr_y_end = curr_y_start + anc_h
            curr_x_start = anc_x + dx
            curr_x_end = curr_x_start + anc_w

            # Boundary check
            if (0 <= curr_y_start and curr_y_end <= img_h and
                0 <= curr_x_start and curr_x_end <= img_w):

                current_shifted_anchor = current_grey[curr_y_start:curr_y_end, curr_x_start:curr_x_end]

                # Calculate SSD (ensure shapes match - paranoia check)
                if current_shifted_anchor.shape == ref_anchor.shape:
                    diff = current_shifted_anchor - ref_anchor
                    ssd = np.sum(diff**2)
                    # Update minimum
                    if ssd < min_ssd:
                        min_ssd = ssd
                        best_dx = dx
                        best_dy = dy

    print(f"  Scan Best Shift Found (dX:{best_dx}, dY:{best_dy}), Min SSD: {min_ssd:.4g}")
    return best_dx, best_dy # Return the shift dx, dy found