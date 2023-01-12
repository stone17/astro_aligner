# astro_aligner

Tool to register lunar (eclipse) images using the image_registration package.
At the moment only jpg is supported.
Rotation is not supported.

### Setup

#### Dependencies

All Python package dependencies are listed in `requirements.txt`. Install them using pip:
```
python -m pip install -U --user -r requirements.txt
```
Depending on your environment, you may have to replace `python` with `python3` or `py`.

### Usage
```
python astro_aligner.py
```

Step 1:
Select folder containing images, e.g. "Test" folder in this repo.
Select your master image from the list. All other images will be registered to it!

Step 2:
Click "Register Images"

This will take a few minutes depending on how many images you have loaded.
When the registration is completed, the images in the GUI can be inspected to see how well the registration worked.

Step 3:
Click "Save files"

Select the folder where to save the registered images. The file names are kept and a postfix "_reg" will be added.

Step 4:
Click "Morph images"

Select the folder where to save the morphed images. Images will be morphed linearly with the frame rate set in the GUI and saved in %d.jpg.

Step 5:
Generate gif. Navigate to the folder where you saved the morphed images and execute ffmpeg:
```
ffmpeg -f image2 -i %3d.jpg -vf "fps=15,scale=1280:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 output.gif
```