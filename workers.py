# c:\Toolz\AA_Astro\astro_aligner\workers.py
import numpy as np
import traceback
import imageio
import os
import time # For potential delays if needed

from PyQt5.QtCore import QObject, pyqtSignal, QThread, pyqtSlot
from image_registration import chi2_shift
import cv2 # Make sure cv2 is imported if used directly here

# Import necessary functions from image_functions (adjust path if needed)
# We might need to move some helper functions here or pass them if they don't rely on 'self'
from image_functions import _perform_cv_rotation, _register_fft, _register_scan_ssd_rot

# --- Registration Worker ---

class RegistrationWorker(QObject):
    """
    Worker object to perform image registration in a separate thread.
    """
    # Signals
    # progress(current_image_index, total_images, image_name)
    progress = pyqtSignal(int, int, str)
    # finished(registered_count, error_count)
    finished = pyqtSignal(int, int)
    # error(error_message)
    error = pyqtSignal(str)
    # log_message(message)
    log_message = pyqtSignal(str)
    # image_updated(index, modified_image_array) - Sends back processed image data
    image_updated = pyqtSignal(int, np.ndarray)
    # rotation_updated(index, new_total_rotation) - Sends back updated rotation
    rotation_updated = pyqtSignal(int, float)


    def __init__(self, image_data_list, ref_image_idx, indices_to_register,
                 reg_method, anchor_details, ref_grey, ref_anchor, shift_val):
        super().__init__()
        # Store necessary data (consider making deep copies if mutable objects are modified)
        # For simplicity now, we assume the main thread won't modify these during processing
        self.image_data_list = image_data_list # List of dicts
        self.ref_image_idx = ref_image_idx
        self.indices_to_register = indices_to_register
        self.reg_method = reg_method # e.g., 'fft', 'scan', 'scan_rot'
        self.anchor_details = anchor_details # dict or None
        self.ref_grey = ref_grey # np.ndarray
        self.ref_anchor = ref_anchor # np.ndarray or None
        self.shift_val = shift_val # int (for scan step)

        self._is_cancelled = False

    @pyqtSlot()
    def cancel(self):
        """Slot to signal cancellation."""
        self.log_message.emit("Cancellation requested...")
        self._is_cancelled = True

    @pyqtSlot()
    def run(self):
        """The main registration loop executed in the thread."""
        self.log_message.emit(f"Starting registration ({self.reg_method})...")
        registered_count = 0
        errors = 0
        num_to_process = len(self.indices_to_register)
        total_images_in_list = len(self.image_data_list) # For context if needed

        try:
            for i, idx in enumerate(self.indices_to_register):
                if self._is_cancelled:
                    self.log_message.emit("Registration cancelled.")
                    break # Exit the loop if cancelled

                image_info = self.image_data_list[idx]
                image_name = image_info['name']
                self.progress.emit(i + 1, num_to_process, image_name) # Progress: 1 to num_to_process
                self.log_message.emit(f"--- Processing Image {idx + 1}/{total_images_in_list} ('{image_name}') ---")

                try:
                    current_image_full_color = image_info['image'] # Use data passed to worker
                    # Ensure it's contiguous for potential C-API calls (like in OpenCV)
                    if not current_image_full_color.flags['C_CONTIGUOUS']:
                        current_image_full_color = np.ascontiguousarray(current_image_full_color)

                    current_grey = np.dot(current_image_full_color[..., :3].astype(np.float32), [0.2989, 0.5870, 0.1140])

                    shift_x_int, shift_y_int = 0, 0
                    rot_angle = 0.0 # Rotation delta for this image
                    corrected_image = None # Store result here

                    # --- Call appropriate registration method ---
                    # Note: These _register_* functions are imported from image_functions
                    if self.reg_method == 'fft':
                        current_grey_reg = current_grey
                        ref_grey_reg = self.ref_grey
                        if self.ref_anchor is not None and self.anchor_details:
                             y_start = self.anchor_details['y']
                             y_end = y_start + self.anchor_details['h']
                             x_start = self.anchor_details['x']
                             x_end = x_start + self.anchor_details['w']
                             # Check bounds before slicing
                             if 0 <= y_start < y_end <= current_grey.shape[0] and \
                                0 <= x_start < x_end <= current_grey.shape[1]:
                                 current_grey_reg = current_grey[y_start:y_end, x_start:x_end]
                                 ref_grey_reg = self.ref_anchor # Use pre-extracted anchor
                             else:
                                 self.log_message.emit(f"  Warning: Anchor out of bounds for image {idx}. Using full FFT.")
                                 # Fallback to full image FFT if anchor is invalid for current image

                        xoff, yoff = _register_fft(ref_grey_reg, current_grey_reg) # Pass data directly
                        if xoff is not None and yoff is not None:
                            shift_y_int = -int(round(yoff))
                            shift_x_int = -int(round(xoff))
                        else:
                            self.log_message.emit(f"  FFT failed for image {idx}.")
                            errors += 1
                            continue

                    elif self.reg_method == 'scan':
                        if self.ref_anchor is None:
                            self.log_message.emit(f"  Scan SSD requires an anchor. Skipping image {idx}.")
                            errors += 1
                            continue
                        # Pass shift_val (step_size) to scan function
                        # *** We need to adjust _register_scan_ssd to accept step_size ***
                        # (This adjustment will be done in image_functions.py)
                        xoff, yoff = _register_scan_ssd(self.ref_anchor, current_grey, self.anchor_details, self.shift_val)
                        if xoff is not None and yoff is not None:
                            shift_y_int = -yoff
                            shift_x_int = -xoff
                        else:
                            self.log_message.emit(f"  Scan SSD failed for image {idx}.")
                            errors += 1
                            continue

                    elif self.reg_method == 'scan_rot':
                        if self.ref_anchor is None:
                            self.log_message.emit(f"  Scan Rot requires an anchor. Skipping image {idx}.")
                            errors += 1
                            continue
                        rot_delta = _register_scan_ssd_rot(self.ref_anchor, current_grey, self.anchor_details)
                        if rot_delta is not None:
                            rot_angle = rot_delta # Store the calculated rotation delta
                        else:
                            self.log_message.emit(f"  Scan Rot failed for image {idx}.")
                            errors += 1
                            continue
                    else:
                         self.log_message.emit(f"  Unknown registration method: {self.reg_method}")
                         errors += 1
                         continue


                    # --- Apply Calculated Transform ---
                    if abs(shift_y_int) > 0 or abs(shift_x_int) > 0:
                        self.log_message.emit(f"  Applying final shift (dX:{shift_x_int}, dY:{shift_y_int})")
                        # Use the original full color image passed to the worker
                        corrected_image = np.roll(current_image_full_color, (shift_y_int, shift_x_int), axis=(0, 1))
                        # Fill edges
                        if shift_y_int > 0: corrected_image[:shift_y_int, :] = 0
                        elif shift_y_int < 0: corrected_image[shift_y_int:, :] = 0
                        if shift_x_int > 0: corrected_image[:, :shift_x_int] = 0
                        elif shift_x_int < 0: corrected_image[:, shift_x_int:] = 0
                        # Emit the updated image data
                        self.image_updated.emit(idx, corrected_image.copy()) # Send copy back

                    elif abs(rot_angle) > 1e-4: # Apply rotation if significant
                         # Get the *original* rotation state passed in image_data_list
                         current_total_rotation = image_info.get('total_rotation', 0.0)
                         new_total_rotation = current_total_rotation + rot_angle # Apply delta

                         # Determine the base image for rotation (prefer original if available)
                         image_to_rotate = image_info.get('image_orig', current_image_full_color)
                         if not image_to_rotate.flags['C_CONTIGUOUS']:
                             image_to_rotate = np.ascontiguousarray(image_to_rotate)

                         self.log_message.emit(f"  Applying rotation delta: {rot_angle:.2f} deg (New Total: {new_total_rotation:.2f})")
                         rotated_im = _perform_cv_rotation(image_to_rotate, new_total_rotation)

                         if rotated_im is not None:
                             corrected_image = rotated_im.astype(np.uint8)
                             # Emit the updated image data AND the new total rotation
                             self.image_updated.emit(idx, corrected_image.copy())
                             self.rotation_updated.emit(idx, new_total_rotation)
                             # Also update 'image_orig' if it didn't exist before rotation
                             if 'image_orig' not in image_info:
                                 # This is tricky - the worker shouldn't modify the original list directly.
                                 # The main thread should handle setting 'image_orig' upon receiving the first rotation update.
                                 pass # Main thread will handle 'image_orig' creation
                         else:
                             self.log_message.emit(f"  Rotation failed for image {idx}.")
                             errors += 1
                             continue # Skip to next image if rotation failed
                    else:
                        self.log_message.emit(f"  Image {idx}: Calculated transform is zero, no change applied.")
                        # No need to emit image_updated if no change

                    registered_count += 1

                except Exception as e:
                    # Catch errors during processing of a single image
                    error_msg = f"Error registering image {idx} ('{image_name}'): {e}"
                    self.log_message.emit(error_msg)
                    # self.error.emit(error_msg) # Maybe too noisy? Log is better.
                    traceback.print_exc() # Log detailed traceback
                    errors += 1
            # --- End Loop ---

        except Exception as e:
            # Catch errors in the overall worker setup/loop logic
            error_msg = f"Critical error during registration worker execution: {e}"
            self.log_message.emit(error_msg)
            traceback.print_exc()
            self.error.emit(error_msg) # Emit critical error signal
            # Ensure finished is emitted even on critical error, but report counts
            self.finished.emit(registered_count, errors + (num_to_process - registered_count - errors))
            return # Stop execution

        # --- Finished Signal ---
        if not self._is_cancelled:
            self.log_message.emit("--- Registration Finished ---")
            self.log_message.emit(f"Processed: {registered_count}, Errors: {errors}")
        self.finished.emit(registered_count, errors)


# --- Morphing Worker ---

class MorphWorker(QObject):
    """
    Worker object to perform image morphing in a separate thread.
    """
    # Signals
    # progress(current_frame_number, total_expected_frames)
    progress = pyqtSignal(int, int)
    # finished(generated_frame_count)
    finished = pyqtSignal(int)
    # error(error_message)
    error = pyqtSignal(str)
    # log_message(message)
    log_message = pyqtSignal(str)

    def __init__(self, image_data_list, frame_rate, save_folder, base_name, save_ext):
        super().__init__()
        self.image_data_list = image_data_list # List of dicts (read-only access needed)
        self.frame_rate = frame_rate
        self.save_folder = save_folder
        self.base_name = base_name
        self.save_ext = save_ext
        self._is_cancelled = False

    @pyqtSlot()
    def cancel(self):
        """Slot to signal cancellation."""
        self.log_message.emit("Cancellation requested...")
        self._is_cancelled = True

    @pyqtSlot()
    def run(self):
        """The main morphing loop executed in the thread."""
        self.log_message.emit(f"Starting morph (FPS={self.frame_rate}) into: {self.save_folder} as '{self.base_name}*{self.save_ext}'")
        counter = 0
        num_images = len(self.image_data_list)
        generated_count = 0
        # Estimate total frames for progress bar:
        # (num_images - 1) pairs * (frame_rate - 1) interpolated frames + num_images original frames
        total_expected_frames = (num_images - 1) * (self.frame_rate - 1) + num_images if num_images > 0 else 0
        if total_expected_frames <= 0: total_expected_frames = 1 # Avoid division by zero

        try:
            if num_images == 0:
                 self.log_message.emit("No images to morph.")
                 self.finished.emit(0)
                 return

            # Save first frame
            img_first = self.image_data_list[0]['image']
            padded_index = str(counter).zfill(5)
            filename = f'{self.base_name}{padded_index}{self.save_ext}'
            save_path = os.path.join(self.save_folder, filename)
            imageio.imwrite(save_path, img_first)
            generated_count += 1
            counter += 1
            self.progress.emit(counter, total_expected_frames)

            # Loop through pairs
            for idx in range(num_images - 1):
                if self._is_cancelled:
                    self.log_message.emit("Morphing cancelled.")
                    break

                self.log_message.emit(f'Processing pair {idx + 1}/{num_images-1}')
                img1_data = self.image_data_list[idx]['image']
                img2_data = self.image_data_list[idx + 1]['image']

                # Check shape mismatch
                if img1_data.shape != img2_data.shape:
                    self.log_message.emit(f"Shape mismatch between image {idx} and {idx + 1}. Skipping interpolation, saving frame {idx + 1}.")
                    # Save the second image directly as the next frame
                    img_next = img2_data
                    padded_index = str(counter).zfill(5)
                    filename = f'{self.base_name}{padded_index}{self.save_ext}'
                    save_path = os.path.join(self.save_folder, filename)
                    imageio.imwrite(save_path, img_next)
                    generated_count += 1
                    counter += 1
                    self.progress.emit(counter, total_expected_frames)
                    continue # Move to the next pair

                # Convert to float32 for interpolation
                img1 = img1_data.astype(np.float32)
                img2 = img2_data.astype(np.float32)

                # Interpolate frames
                num_steps = self.frame_rate - 1
                for i in range(num_steps):
                    if self._is_cancelled: break # Check inside inner loop too

                    alpha = (i + 1.0) / self.frame_rate
                    interp_img = np.clip(img1 * (1.0 - alpha) + img2 * alpha, 0, 255).astype(np.uint8)
                    padded_index = str(counter).zfill(5)
                    filename = f'{self.base_name}{padded_index}{self.save_ext}'
                    save_path = os.path.join(self.save_folder, filename)
                    imageio.imwrite(save_path, interp_img)
                    generated_count += 1
                    counter += 1
                    self.progress.emit(counter, total_expected_frames)

                if self._is_cancelled: break # Check after inner loop

                # Save second image of pair (end frame)
                img_second = img2_data # Use the original uint8 data
                padded_index = str(counter).zfill(5)
                filename = f'{self.base_name}{padded_index}{self.save_ext}'
                save_path = os.path.join(self.save_folder, filename)
                imageio.imwrite(save_path, img_second)
                generated_count += 1
                counter += 1
                self.progress.emit(counter, total_expected_frames)

            # --- End Loop ---

        except Exception as e:
            error_msg = f"Critical error during morphing worker execution: {e}"
            self.log_message.emit(error_msg)
            traceback.print_exc()
            self.error.emit(error_msg)
            self.finished.emit(generated_count) # Still emit finished
            return

        # --- Finished Signal ---
        if not self._is_cancelled:
            self.log_message.emit(f"Finished morphing. Generated {generated_count} frames.")
        self.finished.emit(generated_count)

# --- Helper function adjustments ---
# Make sure these helpers don't rely on 'self' from MainWindow
# If _register_scan_ssd used self.shift_val.text(), it needs shift_val passed now.

def _register_scan_ssd(ref_anchor, current_grey, anchor_details, step_size=1):
    """Registers one image using Scan SSD. Returns (best_dx, best_dy).
       Now takes step_size explicitly.
    """
    # print(f"  Using Scan SSD (Step: {step_size})...") # Log from worker is better
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
                    diff = current_shifted_anchor.astype(np.float32) - ref_anchor.astype(np.float32) # Use float32 for SSD
                    ssd = np.sum(diff**2)
                    # Update minimum
                    if ssd < min_ssd:
                        min_ssd = ssd
                        best_dx = dx
                        best_dy = dy
                # else: # Debugging shape mismatches
                #    print(f"Shape mismatch in SSD Scan: Current={current_shifted_anchor.shape}, Ref={ref_anchor.shape} at dx={dx}, dy={dy}")


    # print(f"  Scan Best Shift Found (dX:{best_dx}, dY:{best_dy}), Min SSD: {min_ssd:.4g}") # Log from worker
    if min_ssd == np.inf: # Check if no valid position was found
        print("  Scan SSD Warning: No valid anchor positions found within scan range.")
        return None, None
    return best_dx, best_dy # Return the shift dx, dy found

# _register_fft and _register_scan_ssd_rot seem okay as they don't use 'self' directly.
# _perform_cv_rotation is also fine.

