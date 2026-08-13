# c:\Toolz\AA_Astro\astro_aligner\workers.py
import numpy as np
import traceback
import imageio
import os
import time # For potential delays if needed

from concurrent.futures import ThreadPoolExecutor

from PyQt5.QtCore import QObject, pyqtSignal, QThread, pyqtSlot
from image_registration import chi2_shift
import cv2 # Make sure cv2 is imported if used directly here

# Import necessary functions from image_functions
from image_functions import (
    _perform_cv_rotation,
    _register_fft,
    _register_scan_ssd,
    _register_scan_ssd_rot,
    match_brightness
)

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
    # image_updated(index, ref_index, modified_image_array) - Sends back processed image data
    image_updated = pyqtSignal(int, int, np.ndarray)
    # rotation_updated(index, new_total_rotation) - Sends back updated rotation
    rotation_updated = pyqtSignal(int, float)


    def __init__(self, image_data_list, ref_image_idx, indices_to_register,
                 reg_method, anchor_details, ref_grey=None, ref_anchor=None, shift_val=1,
                 registration_pairs=None, sync_brightness=False):
        super().__init__()
        # Store necessary data
        self.image_data_list = image_data_list # List of dicts
        self.ref_image_idx = ref_image_idx
        self.indices_to_register = indices_to_register
        self.reg_method = reg_method # e.g., 'fft', 'scan', 'scan_rot'
        self.anchor_details = anchor_details # dict or None
        self.ref_grey = ref_grey # np.ndarray or None
        self.ref_anchor = ref_anchor # np.ndarray or None
        self.shift_val = shift_val # int (for scan step)
        self.sync_brightness = sync_brightness

        # Ensure initial master reference gray and anchor are computed if missing
        if (self.ref_grey is None or self.ref_anchor is None) and 0 <= ref_image_idx < len(image_data_list):
            initial_ref_img = image_data_list[ref_image_idx]['image']
            if not initial_ref_img.flags['C_CONTIGUOUS']:
                initial_ref_img = np.ascontiguousarray(initial_ref_img)
            self.ref_grey = np.dot(initial_ref_img[..., :3].astype(np.float32), [0.2989, 0.5870, 0.1140])
            if self.anchor_details:
                anc_x = self.anchor_details['x']
                anc_y = self.anchor_details['y']
                anc_w = self.anchor_details['w']
                anc_h = self.anchor_details['h']
                if (0 <= anc_y < anc_y + anc_h <= self.ref_grey.shape[0] and
                    0 <= anc_x < anc_x + anc_w <= self.ref_grey.shape[1]):
                    self.ref_anchor = self.ref_grey[anc_y : anc_y + anc_h, anc_x : anc_x + anc_w].astype(np.float32)

        if registration_pairs is not None:
            self.registration_pairs = registration_pairs
        else:
            self.registration_pairs = [(idx, ref_image_idx) for idx in indices_to_register]

        self._is_cancelled = False

    @pyqtSlot()
    def cancel(self):
        """Slot to signal cancellation."""
        self.log_message.emit("Cancellation requested...")
        self._is_cancelled = True

    def _process_single_pair(self, idx, current_ref_idx):
        """Processes registration for a single target/reference pair."""
        logs = []
        target_info = self.image_data_list[idx]
        target_name = target_info['name']
        ref_info = self.image_data_list[current_ref_idx]
        ref_name = ref_info['name']

        total_images_in_list = len(self.image_data_list)
        logs.append(
            f"--- Processing Image {idx + 1}/{total_images_in_list} ('{target_name}') "
            f"against Ref Image {current_ref_idx + 1} ('{ref_name}') ---"
        )

        try:
            ref_image_full_color = ref_info['image']
            if not ref_image_full_color.flags['C_CONTIGUOUS']:
                ref_image_full_color = np.ascontiguousarray(ref_image_full_color)

            ref_grey = np.dot(ref_image_full_color[..., :3].astype(np.float32), [0.2989, 0.5870, 0.1140])

            rolling_ref_anchor = None
            if self.anchor_details:
                anc_x = self.anchor_details['x']
                anc_y = self.anchor_details['y']
                anc_w = self.anchor_details['w']
                anc_h = self.anchor_details['h']
                if (0 <= anc_y < anc_y + anc_h <= ref_grey.shape[0] and
                    0 <= anc_x < anc_x + anc_w <= ref_grey.shape[1]):
                    rolling_ref_anchor = ref_grey[anc_y : anc_y + anc_h, anc_x : anc_x + anc_w].astype(np.float32)

            current_image_full_color = target_info['image']
            if not current_image_full_color.flags['C_CONTIGUOUS']:
                current_image_full_color = np.ascontiguousarray(current_image_full_color)

            current_grey = np.dot(current_image_full_color[..., :3].astype(np.float32), [0.2989, 0.5870, 0.1140])

            shift_x_int, shift_y_int = 0, 0
            rot_angle = 0.0
            corrected_image = None
            new_total_rotation = None

            if self.reg_method == 'fft':
                xoff, yoff = None, None
                if self.anchor_details:
                    anc_x = self.anchor_details['x']
                    anc_y = self.anchor_details['y']
                    anc_w = self.anchor_details['w']
                    anc_h = self.anchor_details['h']
                    pad = 50

                    y0 = max(0, anc_y - pad)
                    y1 = min(current_grey.shape[0], anc_y + anc_h + pad)
                    x0 = max(0, anc_x - pad)
                    x1 = min(current_grey.shape[1], anc_x + anc_w + pad)

                    current_crop = current_grey[y0:y1, x0:x1]
                    ref_crop = ref_grey[y0:y1, x0:x1]

                    if ref_crop.shape == current_crop.shape and ref_crop.size > 0:
                        xoff_crop, yoff_crop = _register_fft(ref_crop, current_crop)
                        if xoff_crop is not None and yoff_crop is not None:
                            xoff, yoff = xoff_crop, yoff_crop
                            logs.append("  FFT matched Anchor region.")

                if xoff is None:
                    xoff_full, yoff_full = _register_fft(ref_grey, current_grey)
                    if xoff_full is not None and yoff_full is not None:
                        xoff, yoff = xoff_full, yoff_full
                        logs.append("  FFT matched Full Image.")

                if xoff is not None and yoff is not None:
                    shift_y_int = -int(round(yoff))
                    shift_x_int = -int(round(xoff))
                else:
                    logs.append(f"  FFT failed for image {idx}.")
                    return idx, current_ref_idx, None, None, False, logs

            elif self.reg_method == 'scan':
                if rolling_ref_anchor is None and self.ref_anchor is None:
                    logs.append(f"  Scan SSD requires an anchor. Skipping image {idx}.")
                    return idx, current_ref_idx, None, None, False, logs

                xoff, yoff = None, None
                scan_range = max(int(self.shift_val) if self.shift_val else 100, 100)

                # 1. Try Rolling Reference Anchor first (for sequential stability)
                if rolling_ref_anchor is not None:
                    xoff_roll, yoff_roll, score_roll = _register_scan_ssd(
                        rolling_ref_anchor, current_grey, self.anchor_details, scan_range=scan_range
                    )
                    if xoff_roll is not None and score_roll is not None and score_roll >= 0.20:
                        xoff, yoff = xoff_roll, yoff_roll
                        logs.append(f"  Scan SSD matched Rolling Anchor (correlation: {score_roll:.4f})")

                # 2. Try Initial Master Anchor as fallback if rolling anchor failed (e.g. clouds)
                if xoff is None and self.ref_anchor is not None:
                    xoff_init, yoff_init, score_init = _register_scan_ssd(
                        self.ref_anchor, current_grey, self.anchor_details, scan_range=scan_range
                    )
                    if xoff_init is not None and score_init is not None and score_init >= 0.40:
                        xoff, yoff = xoff_init, yoff_init
                        logs.append(f"  Scan SSD matched Initial Master Anchor (correlation: {score_init:.4f})")

                if xoff is not None and yoff is not None:
                    shift_x_int = -int(round(xoff))
                    shift_y_int = -int(round(yoff))
                else:
                    logs.append(f"  Scan SSD failed for image {idx}.")
                    return idx, current_ref_idx, None, None, False, logs

            elif self.reg_method == 'scan_rot':
                if rolling_ref_anchor is None and self.ref_anchor is None:
                    logs.append(f"  Scan Rot requires an anchor. Skipping image {idx}.")
                    return idx, current_ref_idx, None, None, False, logs

                rot_delta = None
                if self.ref_anchor is not None:
                    rot_delta = _register_scan_ssd_rot(self.ref_anchor, current_grey, self.anchor_details)
                    if rot_delta is not None:
                        logs.append("  Scan Rot matched Initial Master Anchor.")

                if rot_delta is None and rolling_ref_anchor is not None:
                    rot_delta = _register_scan_ssd_rot(rolling_ref_anchor, current_grey, self.anchor_details)
                    if rot_delta is not None:
                        logs.append("  Scan Rot matched Rolling Anchor.")

                if rot_delta is not None:
                    rot_angle = rot_delta
                else:
                    logs.append(f"  Scan Rot failed for image {idx}.")
                    return idx, current_ref_idx, None, None, False, logs
            else:
                logs.append(f"  Unknown registration method: {self.reg_method}")
                return idx, current_ref_idx, None, None, False, logs

            image_changed = False
            if abs(shift_y_int) > 0 or abs(shift_x_int) > 0:
                logs.append(f"  Applying final shift (dX:{shift_x_int}, dY:{shift_y_int})")
                corrected_image = np.roll(current_image_full_color, (shift_y_int, shift_x_int), axis=(0, 1))
                if shift_y_int > 0: corrected_image[:shift_y_int, :] = 0
                elif shift_y_int < 0: corrected_image[shift_y_int:, :] = 0
                if shift_x_int > 0: corrected_image[:, :shift_x_int] = 0
                elif shift_x_int < 0: corrected_image[:, shift_x_int:] = 0
                image_changed = True

            elif abs(rot_angle) > 1e-4:
                current_total_rotation = target_info.get('total_rotation', 0.0)
                new_total_rotation = current_total_rotation + rot_angle

                image_to_rotate = target_info.get('image_orig', current_image_full_color)
                if not image_to_rotate.flags['C_CONTIGUOUS']:
                    image_to_rotate = np.ascontiguousarray(image_to_rotate)

                logs.append(f"  Applying rotation delta: {rot_angle:.2f} deg (New Total: {new_total_rotation:.2f})")
                rotated_im = _perform_cv_rotation(image_to_rotate, new_total_rotation)

                if rotated_im is not None:
                    corrected_image = rotated_im.astype(np.uint8)
                    image_changed = True
                else:
                    logs.append(f"  Rotation failed for image {idx}.")
                    return idx, current_ref_idx, None, None, False, logs
            else:
                logs.append(f"  Image {idx}: Calculated transform is zero.")

            if corrected_image is None:
                corrected_image = current_image_full_color.copy()

            if self.sync_brightness:
                logs.append(f"  Syncing brightness for image {idx} against ref {current_ref_idx}...")
                corrected_image = match_brightness(corrected_image, ref_info['image'], self.anchor_details)
                image_changed = True

            if not image_changed:
                corrected_image = None

            return idx, current_ref_idx, corrected_image, new_total_rotation, True, logs

        except Exception as e:
            logs.append(f"Error registering image {idx} ('{target_name}'): {e}")
            return idx, current_ref_idx, None, None, False, logs

    @pyqtSlot()
    def run(self):
        """The main registration loop executed in the thread."""
        self.log_message.emit(f"Starting registration ({self.reg_method})...")
        registered_count = 0
        errors = 0
        num_to_process = len(self.registration_pairs)

        is_rolling = len(self.registration_pairs) > 1 and any(
            self.registration_pairs[i][1] == self.registration_pairs[i - 1][0]
            for i in range(1, len(self.registration_pairs))
        )

        try:
            if is_rolling:
                self.log_message.emit("Running sequential registration (rolling master mode)...")
                for i, (idx, current_ref_idx) in enumerate(self.registration_pairs):
                    if self._is_cancelled:
                        self.log_message.emit("Registration cancelled.")
                        break

                    target_name = self.image_data_list[idx]['name']
                    self.progress.emit(i + 1, num_to_process, target_name)

                    idx, ref_idx, corrected_img, new_rot, success, logs = self._process_single_pair(idx, current_ref_idx)
                    for msg in logs:
                        self.log_message.emit(msg)

                    if success:
                        registered_count += 1
                        if corrected_img is not None:
                            self.image_data_list[idx]['image'] = corrected_img
                            self.image_updated.emit(idx, ref_idx, corrected_img.copy())
                        if new_rot is not None:
                            self.rotation_updated.emit(idx, new_rot)
                    else:
                        errors += 1
            else:
                max_workers = min(32, (os.cpu_count() or 4) + 4)
                self.log_message.emit(f"Running parallel multi-core registration ({max_workers} worker threads)...")

                def _pair_worker(pair):
                    if self._is_cancelled:
                        return None
                    return self._process_single_pair(pair[0], pair[1])

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(_pair_worker, pair) for pair in self.registration_pairs]
                    for i, future in enumerate(futures):
                        if self._is_cancelled:
                            break
                        res = future.result()
                        if res is None:
                            continue
                        idx, ref_idx, corrected_img, new_rot, success, logs = res
                        target_name = self.image_data_list[idx]['name']
                        self.progress.emit(i + 1, num_to_process, target_name)
                        for msg in logs:
                            self.log_message.emit(msg)

                        if success:
                            registered_count += 1
                            if corrected_img is not None:
                                self.image_data_list[idx]['image'] = corrected_img
                                self.image_updated.emit(idx, ref_idx, corrected_img.copy())
                            if new_rot is not None:
                                self.rotation_updated.emit(idx, new_rot)
                        else:
                            errors += 1

        except Exception as e:
            error_msg = f"Critical error during registration worker execution: {e}"
            self.log_message.emit(error_msg)
            traceback.print_exc()
            self.error.emit(error_msg)
            self.finished.emit(registered_count, errors + (num_to_process - registered_count - errors))
            return

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

    def _process_morph_pair(self, idx, total_expected_frames):
        """Generates interpolated frames for pair idx in parallel."""
        if self._is_cancelled:
            return 0, [f"Cancelled pair {idx + 1}"]

        logs = []
        logs.append(f'Processing morph pair {idx + 1}/{len(self.image_data_list) - 1}')
        img1_data = self.image_data_list[idx]['image']
        img2_data = self.image_data_list[idx + 1]['image']

        start_counter = 1 + idx * self.frame_rate
        generated = 0

        if img1_data.shape != img2_data.shape:
            logs.append(f"Shape mismatch between image {idx} and {idx + 1}. Skipping interpolation.")
            padded_index = str(start_counter).zfill(5)
            filename = f'{self.base_name}{padded_index}{self.save_ext}'
            save_path = os.path.join(self.save_folder, filename)
            imageio.imwrite(save_path, img2_data)
            return 1, logs

        img1 = img1_data.astype(np.float32)
        img2 = img2_data.astype(np.float32)
        num_steps = self.frame_rate - 1

        for i in range(num_steps):
            if self._is_cancelled:
                break
            alpha = (i + 1.0) / self.frame_rate
            interp_img = np.clip(img1 * (1.0 - alpha) + img2 * alpha, 0, 255).astype(np.uint8)
            padded_index = str(start_counter + i).zfill(5)
            filename = f'{self.base_name}{padded_index}{self.save_ext}'
            save_path = os.path.join(self.save_folder, filename)
            imageio.imwrite(save_path, interp_img)
            generated += 1

        if not self._is_cancelled:
            padded_index = str(start_counter + num_steps).zfill(5)
            filename = f'{self.base_name}{padded_index}{self.save_ext}'
            save_path = os.path.join(self.save_folder, filename)
            imageio.imwrite(save_path, img2_data)
            generated += 1

        return generated, logs

    @pyqtSlot()
    def run(self):
        """The main morphing loop executed in the thread."""
        self.log_message.emit(f"Starting morph (FPS={self.frame_rate}) into: {self.save_folder} as '{self.base_name}*{self.save_ext}'")
        num_images = len(self.image_data_list)
        generated_count = 0
        total_expected_frames = (num_images - 1) * (self.frame_rate - 1) + num_images if num_images > 0 else 0
        if total_expected_frames <= 0:
            total_expected_frames = 1

        try:
            if num_images == 0:
                self.log_message.emit("No images to morph.")
                self.finished.emit(0)
                return

            # Save first frame
            img_first = self.image_data_list[0]['image']
            padded_index = "0".zfill(5)
            filename = f'{self.base_name}{padded_index}{self.save_ext}'
            save_path = os.path.join(self.save_folder, filename)
            imageio.imwrite(save_path, img_first)
            generated_count += 1
            self.progress.emit(generated_count, total_expected_frames)

            max_workers = min(32, (os.cpu_count() or 4) + 4)
            self.log_message.emit(f"Running multi-core parallel morphing ({max_workers} worker threads)...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self._process_morph_pair, idx, total_expected_frames) for idx in range(num_images - 1)]
                for future in futures:
                    if self._is_cancelled:
                        break
                    pair_generated, logs = future.result()
                    for log_msg in logs:
                        self.log_message.emit(log_msg)
                    generated_count += pair_generated
                    self.progress.emit(generated_count, total_expected_frames)

        except Exception as e:
            error_msg = f"Critical error during morphing worker execution: {e}"
            self.log_message.emit(error_msg)
            traceback.print_exc()
            self.error.emit(error_msg)
            self.finished.emit(generated_count)
            return

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

