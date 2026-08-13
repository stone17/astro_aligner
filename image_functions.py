import numpy as np
import cv2
import imageio
import traceback
from concurrent.futures import ThreadPoolExecutor
from image_registration import chi2_shift
from astropy.io import fits as astro_fits
import os
from PyQt5.QtWidgets import (QFileDialog, QMessageBox)

# --- Other Actions ---
def translate_image(self, direction):
    # Check if images are loaded
    if not hasattr(self, 'image_data') or not self.image_data:
        print("No images loaded.")
        return
    # Check if current index is valid
    num_images = len(self.image_data)
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
    current_im = self.image_data[self.current_image_idx]['image']
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
    self.image_data[self.current_image_idx]['image'] = shifted_im
    # Force update of base pixmap for current image as data changed
    self.updatePixmap(update_base=True)

def _extract_transform_params(M):
    """
    Extracts translation, rotation angle, and scale from a 2x3 affine matrix.

    Args:
        M (np.ndarray): A 2x3 affine transformation matrix.

    Returns:
        tuple: (tx, ty, angle_degrees, scale) or None if matrix is invalid.
    """
    if M is None or M.shape != (2, 3):
        print("Error: Invalid matrix provided.")
        return None, None, None, None

    # Translation is directly available
    tx = M[0, 2]
    ty = M[1, 2]

    # Calculate scale using the length of the first column vector [a, c]
    # (or second column [b, d], they should be equal for similarity transforms)
    scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)

    # Calculate rotation angle using atan2(y, x) = atan2(c, a)
    # Ensure scale is not zero to avoid division errors
    if scale < 1e-6:
        # Cannot determine angle if scale is essentially zero
        angle_rad = 0.0
        print("Warning: Scale is near zero, cannot reliably determine angle.")
    else:
        angle_rad = np.arctan2(M[1, 0], M[0, 0])

    # Convert angle to degrees
    angle_deg = np.degrees(angle_rad)

    return tx, ty, angle_deg, scale


def apply_M_transformation(self, M):
    """Applies an affine transformation matrix M (rotation + translation part)."""
    if not self.image_data or not (0 <= self.current_image_idx < len(self.image_data)):
        print("No current image selected for applying M transformation.")
        return

    tx, ty, angle_deg, scale = _extract_transform_params(M)
    if tx is None:
        return # Invalid matrix

    current_data = self.image_data[self.current_image_idx]
    current_total_rotation = current_data.get('total_rotation', 0.0)
    target_rotation = current_total_rotation - angle_deg # Apply the delta

    print(f"Applying M Transform: Angle Delta {-angle_deg:.2f}, Target Rot {target_rotation:.2f}, Shift ({tx:.2f}, {ty:.2f})")

    # --- Apply Rotation First ---
    if abs(angle_deg) > 1e-4:
        # Ensure 'image_orig' exists
        if 'image_orig' not in current_data:
            current_data['image_orig'] = current_data['image'].copy()

        image_to_rotate = current_data['image_orig']
        rotated_im = _perform_cv_rotation(image_to_rotate, target_rotation)

        if rotated_im is not None:
            current_data['image'] = rotated_im.astype(np.uint8)
            current_data['total_rotation'] = target_rotation
            self.rot_val.setText(f"{target_rotation:.1f}") # Update text box
            # Don't update pixmap yet, translation comes next
        else:
            QMessageBox.warning(self, "Rotation Error", "Failed M transformation rotation.")
            # Restore text box?
            self.rot_val.setText(f"{current_total_rotation:.1f}")
            return # Stop if rotation failed

    # --- Apply Translation Second (on the potentially rotated image) ---
    shift_x_int = int(round(tx))
    shift_y_int = int(round(ty))
    if abs(shift_y_int) > 0 or abs(shift_x_int) > 0:
        print(f"  Applying M shift (dX:{shift_x_int}, dY:{shift_y_int})")
        # Apply roll to the *current* image data (which might have just been rotated)
        image_to_shift = current_data['image']
        corrected_image = np.roll(image_to_shift, (shift_y_int, shift_x_int), axis=(0, 1))
        # Fill edges
        if shift_y_int > 0: corrected_image[:shift_y_int, :] = 0 # Top
        elif shift_y_int < 0: corrected_image[shift_y_int:, :] = 0 # Bottom
        if shift_x_int > 0: corrected_image[:, :shift_x_int] = 0 # Left
        elif shift_x_int < 0: corrected_image[:, shift_x_int:] = 0 # Right
        # Update the image in memory
        current_data['image'] = corrected_image.copy()

    # --- Final Update ---
    # Update pixmap only once after both rotation and translation
    self.updatePixmap(update_base=True)


def rotate_image_incremental(self, direction):
    """Applies small incremental rotation via CCW/CW buttons."""
    if not self.image_data or not (0 <= self.current_image_idx < len(self.image_data)):
        print("No current image selected for incremental rotation.")
        return

    current_data = self.image_data[self.current_image_idx]
    current_total_rotation = current_data.get('total_rotation', 0.0)
    angle_step = 0.1
    incremental_angle = -angle_step if direction == 'Left' else angle_step
    new_total_rotation = current_total_rotation + incremental_angle

    print(f"Applying incremental rotation: {incremental_angle:.1f} degrees")
    self.rot_val.setText(f"{new_total_rotation:.1f}")
    # Call apply_text_rotation to actually perform the rotation based on the new text value
    apply_text_rotation(self)

def apply_text_rotation(self):
    """Applies rotation to match the absolute value in the text box."""
    if not self.image_data or not (0 <= self.current_image_idx < len(self.image_data)):
        print("No current image selected for applying text rotation.")
        return

    current_data = self.image_data[self.current_image_idx]
    current_total_rotation = current_data.get('total_rotation', 0.0)

    # Ensure 'image_orig' exists before applying rotation from text
    # If it doesn't exist, the current image *is* the original state for this rotation op
    if 'image_orig' not in current_data:
        current_data['image_orig'] = current_data['image'].copy()
        print("Created 'image_orig' backup before text rotation.")

    image_to_rotate = current_data['image_orig'] # Always rotate from original

    try:
        target_rotation = float(self.rot_val.text().replace(',', '.'))
    except ValueError:
        QMessageBox.warning(self, "Invalid Input", "Rotation value must be a valid number.")
        self.rot_val.setText(f"{current_total_rotation:.1f}") # Reset text to current actual
        return

    # Check if rotation is actually needed
    if abs(target_rotation - current_total_rotation) < 1e-4:
        print("Target rotation matches current. No change.")
        # Ensure text box matches precisely if slightly different due to float formatting
        self.rot_val.setText(f"{current_total_rotation:.1f}")
        return

    print(f"Applying text rotation to target: {target_rotation:.1f} degrees")
    rotated_im = _perform_cv_rotation(image_to_rotate, target_rotation)

    if rotated_im is not None:
        current_data['image'] = rotated_im.astype(np.uint8)
        current_data['total_rotation'] = target_rotation # Update the stored total rotation
        # Update text box again to ensure consistent formatting (e.g., ".1f")
        self.rot_val.setText(f"{target_rotation:.1f}")
        self.updatePixmap(update_base=True) # Force update of base pixmap as data changed
    else:
        QMessageBox.warning(self, "Rotation Error", "Failed text rotation.")
        # Reset text box to the value *before* this failed attempt
        self.rot_val.setText(f"{current_total_rotation:.1f}")


def _perform_cv_rotation(image, angle):
    """Internal helper using OpenCV for rotation."""
    if image is None:
        return None
    try:
        # Ensure image is contiguous
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)

        h, w = image.shape[:2]
        cX, cY = (w // 2, h // 2)
        # Use negative angle for OpenCV's definition if we want CCW for positive angle input
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        return rotated
    except Exception as e:
        print(f"Error during OpenCV rotation: {e}")
        traceback.print_exc()
        return None


# --- Registration Helpers (Called by Worker) ---

def _register_fft(ref_grey, current_grey):
    """Registers one image using FFT (chi2_shift). Returns (xoff, yoff) or (None, None)."""
    # print(f"  Using FFT (chi2_shift)...") # Worker logs this
    try:
        # upsample_factor improves subpixel precision but slows it down
        xoff, yoff, exoff, eyoff = chi2_shift(
            ref_grey,
            current_grey,
            upsample_factor='auto', # Or try 10, 100 etc.
            return_error=True
        )
        # print(f"  register fft Result: Shift=(x:{xoff:.2f}, y:{yoff:.2f})") # Worker logs this
        return xoff, yoff
    except Exception as fft_err:
        print(f"  Error during FFT registration: {fft_err}")
        # traceback.print_exc() # Optional detailed traceback in worker log
        return None, None


def _register_scan_ssd(ref_anchor, current_grey, anchor_details, scan_range=100):
    """Registers one image using Scan SSD / Normalized Cross-Correlation via multi-threaded C++ template matching.
    Returns (best_dx, best_dy, max_val) or (None, None, None).
    `best_dx` and `best_dy` are integer displacements of current_grey relative to ref_anchor.
    `max_val` is the normalized cross-correlation coefficient in range [-1.0, 1.0].
    """
    if ref_anchor is None or current_grey is None or not anchor_details:
        return None, None, None

    anc_x = anchor_details['x']
    anc_y = anchor_details['y']
    anc_w = anchor_details['w']
    anc_h = anchor_details['h']

    img_h, img_w = current_grey.shape

    y_min = max(0, anc_y - scan_range)
    y_max = min(img_h, anc_y + anc_h + scan_range)
    x_min = max(0, anc_x - scan_range)
    x_max = min(img_w, anc_x + anc_w + scan_range)

    search_region = current_grey[y_min:y_max, x_min:x_max]
    if search_region.shape[0] < ref_anchor.shape[0] or search_region.shape[1] < ref_anchor.shape[1]:
        return None, None, None

    try:
        res = cv2.matchTemplate(search_region.astype(np.float32), ref_anchor.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        _, max_val, _, (mx, my) = cv2.minMaxLoc(res)

        best_dx = (x_min + mx) - anc_x
        best_dy = (y_min + my) - anc_y
        return best_dx, best_dy, float(max_val)
    except Exception as e:
        print(f"Scan SSD error: {e}")
        return None, None, None


def _register_scan_ssd_rot(ref_anchor, current_grey, anchor_details):
    """Registers one image rotation using parallel SSD comparison across angles. Returns best_angle."""
    if ref_anchor is None or current_grey is None or not anchor_details:
        return None

    step_size = 0.2
    scan_range_degrees = 5.0
    anc_x = anchor_details['x']
    anc_y = anchor_details['y']
    anc_w = anchor_details['w']
    anc_h = anchor_details['h']

    y_start = anc_y
    y_end = y_start + anc_h
    x_start = anc_x
    x_end = x_start + anc_w

    ref_anchor_float = ref_anchor.astype(np.float32)
    angles = np.arange(-scan_range_degrees, scan_range_degrees + step_size, step_size)

    def _eval_angle(angle):
        rotated_grey = _perform_cv_rotation(current_grey, angle)
        if rotated_grey is None:
            return np.inf, angle
        if (0 <= y_start < y_end <= rotated_grey.shape[0] and
            0 <= x_start < x_end <= rotated_grey.shape[1]):
            current_rotated_anchor = rotated_grey[y_start:y_end, x_start:x_end]
            if current_rotated_anchor.shape == ref_anchor_float.shape:
                diff = current_rotated_anchor.astype(np.float32) - ref_anchor_float
                ssd = np.sum(diff**2)
                return float(ssd), angle
        return np.inf, angle

    max_workers = min(32, (os.cpu_count() or 4) + 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_eval_angle, angles))

    min_ssd, best_angle = min(results, key=lambda item: item[0])
    if min_ssd == np.inf:
        print("  Scan SSD Rot Warning: No valid anchor positions found within scan range.")
        return None
    return best_angle


def match_brightness(target_img, ref_img, anchor_details=None):
    """
    Adjusts the brightness of target_img to match ref_img.
    If anchor_details is provided, calculates scaling ratio within the anchor region.
    Supports both 3D color images (H, W, 3) and 2D grayscale images (H, W).
    """
    if target_img is None or ref_img is None:
        return target_img

    target_patch = target_img
    ref_patch = ref_img

    if anchor_details:
        anc_x = anchor_details.get('x', 0)
        anc_y = anchor_details.get('y', 0)
        anc_w = anchor_details.get('w', 0)
        anc_h = anchor_details.get('h', 0)
        if (0 <= anc_y < anc_y + anc_h <= ref_img.shape[0] and
            0 <= anc_x < anc_x + anc_w <= ref_img.shape[1] and
            0 <= anc_y < anc_y + anc_h <= target_img.shape[0] and
            0 <= anc_x < anc_x + anc_w <= target_img.shape[1]):
            ref_patch = ref_img[anc_y:anc_y+anc_h, anc_x:anc_x+anc_w]
            target_patch = target_img[anc_y:anc_y+anc_h, anc_x:anc_x+anc_w]

    synced_img = target_img.astype(np.float32)

    if target_img.ndim == 3 and target_img.shape[2] == 3:
        for c in range(3):
            ref_c = ref_patch[..., c]
            target_c = target_patch[..., c]

            ref_mask = ref_c > 5
            target_mask = target_c > 5

            ref_mean = np.mean(ref_c[ref_mask]) if np.any(ref_mask) else np.mean(ref_c)
            target_mean = np.mean(target_c[target_mask]) if np.any(target_mask) else np.mean(target_c)

            if target_mean > 1e-4:
                gain = ref_mean / target_mean
                gain = np.clip(gain, 0.2, 5.0)
            else:
                gain = 1.0

            synced_img[..., c] *= gain
    else:
        ref_mask = ref_patch > 5
        target_mask = target_patch > 5

        ref_mean = np.mean(ref_patch[ref_mask]) if np.any(ref_mask) else np.mean(ref_patch)
        target_mean = np.mean(target_patch[target_mask]) if np.any(target_mask) else np.mean(target_patch)

        if target_mean > 1e-4:
            gain = ref_mean / target_mean
            gain = np.clip(gain, 0.2, 5.0)
        else:
            gain = 1.0

        synced_img *= gain

    return np.clip(synced_img, 0, 255).astype(np.uint8)