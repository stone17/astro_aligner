# Image Registration & Morphing Tool

![Snapshot](https://github.com/stone17/astro_aligner/blob/main/snapshot.png?raw=true)

## Introduction

This tool was developed to address specific challenges encountered when registering image sequences of dynamic astronomical events, particularly **solar and lunar eclipses**.

Standard full-frame image registration algorithms often struggle with eclipse sequences due to:

1.  **High Dynamic Range:** Features like the solar corona or the bright lunar limb coexist with much fainter details (inner corona, Earthshine, background stars).
2.  **Transient or Changing Features:** The appearance of the corona, Bailey's beads, prominences, or the relative position of the Moon against background stars changes rapidly.

Aligning based on the *entire* image can cause these dominant or changing features to improperly influence the result, leading to poor alignment of the primary subject (the solar or lunar disk/limb).

This application solves this by allowing the user to define a specific **anchor region** (a subsection) within the image. The registration process (whether using the FFT or the direct Scan SSD method) can then be focused primarily on the features within this stable anchor area – such as the solar/lunar limb, a prominent sunspot, or a specific crater – minimizing the influence of problematic areas elsewhere in the frame. This accurate alignment is crucial for the ultimate goal of **creating smooth, high-quality animations (videos or GIFs)** showcasing the progression of the eclipse event from the image sequence, often facilitated by the included morphing tool.

## Features

* **Load Images:** Load multiple images (JPG, PNG, BMP, FITS) from a selected folder. Timestamps are read from EXIF/FITS headers where possible, otherwise file modification time is used (images are sorted by time if available for all).
* **Image Display:**
    * View reference and currently selected images side-by-side with zoom/pan.
    * Display the absolute difference between the reference and current image.
    * Toggle a centered crosshair/bullseye overlay (red) on the reference image view.
* **Reference Selection:** Choose any image from the list as the reference frame. Reference points selected for 2-point rotation persist when changing the *current* image.
* **Registration Methods:**
    * **FFT (Fast, Subpixel):** Uses the `image-registration` library (`chi2_shift`) for fast, subpixel alignment. Uses the anchor area if defined for calculation.
    * **Scan Shift SSD (Anchor Required):** Performs a brute-force scan for X/Y shift around an anchor area, minimizing the Sum of Squared Differences (SSD) to find the best integer pixel shift. Requires an anchor area.
    * **Scan Rotation SSD (Anchor Required):** Performs a brute-force scan for rotation around an anchor area, minimizing SSD. Slower but can find rotation in certain cases. Requires an anchor area. Rotation is around the image center!
* **Anchor Area:** Define a specific rectangular region via:
    * Text input (X, Y, Width, Height) and "Apply Anchor".
    * Clicking and dragging on the Reference Image view.
    * The defined anchor is visualized as a green rectangle on the reference image.
* **Manual Adjustments:**
    * Translate the current image pixel-by-pixel (Up/Down/Left/Right).
    * Rotate the current image:
        * Incrementally using CCW/CW buttons.
        * Applying an absolute target rotation angle using the text box and "Apply Rot" button.
        * Total accumulated rotation for the current image is tracked and displayed.
* **2-Point Rotation Aid:**
    * Enter dedicated modes to select 2 points on the Reference image and 2 corresponding points on the Current image.
    * Button ("Get Rotation") calculates the rotation delta needed to align the point pairs and updates the "Total Rot [°]" text box. This rotation can then be applied using the "Apply Rot" button.
    * Reference points persist when changing the Current image. Point selections are cleared when changing the Reference image.
* **Morphing:** Generate a sequence of interpolated frames between the (ideally registered) images in the list. Can use image timestamps (if available for all images) to create variable frame steps for a smoother time-lapse effect, or fallback to fixed steps. Save frames as PNG, JPG, or BMP.
* **Saving:**
    * Save the currently selected image or all processed images.
    * Choose output format (PNG, JPG, BMP) in the save dialog.
    * Attempts to preserve the original image timestamp (from EXIF/FITS/mtime) as the file modification time of the saved file.
* **Configuration:** Automatically saves the last used folders (load, save, morph) to `config.yaml`.

## Requirements

* Python 3.x
* PyQt5 (`pip install PyQt5`)
* NumPy (`pip install numpy`)
* imageio (`pip install imageio[pyav]`) - *`[pyav]` optional but recommended for better video format support if saving animations as video later*
* PyYAML (`pip install PyYAML`)
* image-registration (`pip install image-registration`) - *Required for FFT registration*
* Astropy (`pip install astropy`) - *Required for FITS support*
* Pillow (`pip install Pillow`) - *Required for EXIF timestamp reading*
* OpenCV (`pip install opencv-python`) - *Required for rotation and 2/3-point alignment*

## Installation

1.  **Clone or Download:** Get the project files (`main.py`, `file_functions.py`, `image_functions.py`, etc.).
2.  **Install Dependencies:** It's strongly recommended to use a Python virtual environment.
    ```bash
    # Create a virtual environment (e.g., named 'venv')
    python -m venv venv

    # Activate the virtual environment
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate

    # Install required packages
    pip install PyQt5 numpy imageio[pyav] PyYAML image-registration astropy Pillow opencv-python
    ```
    Alternatively, run:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Activate your virtual environment (if using one).
2.  Run the main application script from your terminal:
    ```bash
    python main.py
    ```

3.  **Workflow Example:**
    * Click "Open folder" to load images. Images will attempt to sort by time if possible.
    * Select the reference image (checkbox) and current image (click name) from the list.
    * Use zoom/pan and optionally toggle the "Crosshair" to inspect alignment.
    * Define an anchor box using text input + "Apply Anchor" or by dragging on the reference image.
    * Select the desired "Registration Method".
    * Click "Register current" or "Register all".
    * *Alternatively*, use the "2-Point Rotation Aid":
        * Click "Set Ref Points", click two points on the reference image.
        * Click "Set Target Points", click the *corresponding* two points on the current image.
        * Click "Get Rotation". The "Total Rot [°]" box updates.
        * Click "Apply Rot" to apply the calculated rotation.
    * Use manual translate/rotate buttons for fine-tuning if needed.
    * Click "Save current" or "Save all", choosing the desired output format (PNG/JPG/BMP).
    * Optionally, click "Morph images", choosing a base filename and format.

## Configuration

The application automatically creates/updates a `config.yaml` file in the same directory. This stores the paths to the last folders used for loading images, saving registered images, and saving morph sequences.