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

* **Load Images:** Load multiple images (JPG, PNG, BMP, FITS) from a selected folder.
* **Image Display:**
    * View reference and currently selected images side-by-side with zoom/pan.
    * Display the absolute difference between the reference and current image.
* **Reference Selection:** Choose any image from the list as the reference frame.
* **Registration Methods:**
    * **FFT (Fast, Subpixel):** Uses the `image-registration` library (`chi2_shift`) for fast, subpixel alignment. Uses the anchor area if defined for calculation.
    * **Scan SSD (Anchor Required):** Performs a brute-force scan around an anchor area, minimizing the Sum of Squared Differences (SSD) to find the best integer pixel shift. **Requires** an anchor area to be defined.
* **Anchor Area:** Define a specific rectangular region (X, Y, Width, Height) in pixel coordinates to focus the registration. The defined anchor is visualized on the reference image.
* **Manual Adjustments:**
    * Translate the current image pixel-by-pixel (Up/Down/Left/Right).
    * Rotate the current image incrementally (requires `image_editing.py`).
* **Morphing:** Generate a sequence of interpolated frames between the (ideally registered) images in the list. Save frames as PNG, JPG, or BMP.
* **Saving:**
    * Save the currently selected image or all processed images.
    * Choose output format (PNG, JPG, BMP). Registered images are typically saved with a `_reg` suffix.
* **Configuration:** Automatically saves the last used folders (load, save, morph) to `config.yaml`.
* **Threading:** Long operations (Register All, Morph Images) run in a background thread with a progress dialog and cancellation support.

## Requirements

* Python 3.x
* PyQt5 (`pip install PyQt5`)
* NumPy (`pip install numpy`)
* imageio (`pip install imageio`)
* PyYAML (`pip install PyYAML`)
* image-registration (`pip install image-registration`)
* Astropy (`pip install astropy`) - *Required for FITS support*
* `image_editing.py` (Requires separate implementation, assumed available) - *Required for manual rotation*

## Installation

1.  **Clone or Download:** Get the project files (`main_gui.py`, `image_editing.py` if applicable, etc.).
2.  **Install Dependencies:** It's recommended to use a virtual environment.
    ```bash
    # Create a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

    # Install packages
    pip install PyQt5 numpy imageio PyYAML image-registration astropy
    ```
    Alternatively, create a `requirements.txt` file with the package names (one per line) and run:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Place `image_editing.py`:** Ensure the `image_editing.py` file is in the same directory as the main script, or accessible via your Python path if you use manual rotation.

## Usage

1.  Run the main application script from your terminal:
    ```bash
    python main_gui.py
    ```
    *(Replace `main_gui.py` with the actual name of your main Python script if different).*

2.  **Workflow:**
    * Click "Open folder" to load images.
    * Select the reference image using the checkbox.
    * Click an image name to view it as the "Current Image".
    * Use zoom/pan controls.
    * Select the "Registration Method".
    * If using Scan SSD or optionally FFT, define the "Anchor Area" and click "Apply Anchor".
    * Click "Register current" or "Register all".
    * Use manual adjustments if needed.
    * Click "Save current" or "Save all", choosing the format.
    * Click "Morph images", choosing a base filename and format.

## Configuration

The application automatically creates/updates a `config.yaml` file in the same directory. This stores the paths to the last folders used for loading images, saving images, and saving morph sequences.

*(Optional: Add License section)*