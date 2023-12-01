# Pytorch CNN train example for semantic segmentation

Some fixes to auto-generate train set, install pytorch
and run on linux with NVIDIA 3090 for training or
CPU-only for usage.

# Annotation using GIMP

If there is large input image like 40000x100000 pixels, it will be too
big to work with gimp. It should be split to smaller tiles, minimum
900x900 to 4000x4000.

Increase imagemagic policy range, to accept images of large resolution
and disk usage. edit "/etc/ImageMagick-6/policy.xml"

    <policy domain="resource" name="width" value="64KP"/>
    <policy domain="resource" name="height" value="64KP"/>
    <policy domain="resource" name="disk" value="8GiB"/>

Use script "tools/split-image.sh bigimage.jpg". Split tiles will be in
"/tmp/airvoid" directory.

Choose tile of interested and open it with gimp. Create 3 additional layers:
"black", "red", "green". Create "red" and "green" filled with transparent
background, "black" filled with non-transparent black (background).

On the left window in the middle, find two overlapping squares and
click lower right square, then select full black (0,0,0) as background
color.

Right-click on lower right vertical window (layers), a pop-up
menu will appear.

Click on "New Layer", a "Create New Layer" window will appear.

Create New Layer, select:

    Layer name:  1_stone
    Color tag:   red
    Fill with:   Transparency

Create New Layer, select:

    Layer name:  2_void
    Color tag:   green
    Fill with:   Transparency

Create New Layer, select:

    Layer name:  0_black
    Color tag:   black
    Fill with:   Background color

Black is required. If black layer is missing, otherwise a "bug" will appear.
Pixels erased with "rubber" tools will look transparent (invisible) to user
but they will be exported as gray instead of black.

Layers should be ordered like this (drag with mouse up-down):

    enable  color  name     background
    ------  -----  ----     ----------
    [x]     green  1_stone  transparent
    [x]     red    2_void   transparent
    [x]            image to be annotated
    [x]     black  0_black  black

If already annotated image is available as .png file
preferrable with transparency, it can be imported
as new layer from menu "File" -> "Open as Layers" and
select file name and click "Open".

If there's no transparency in .png file then after
importing as layer, transparency can be created with:

    "Colors" -> "Color to Alpha"

"Colors" menu has many options to convert colors.

To manually annotate, use tools: sharp pencil, rubber, fill.
Create 100% green (0,255,0) and 100% red color (255,0,0)
and add those colors to palette for quick selection while working.
Select layer 1 or 2, select pencil color from palette
and draw annotations.

When finished, select "green" and "black"

    enable  color  name     background
    ------  -----  ----     ----------
    [x]     green  1_stone  transparent
    [ ]     red    2_void   transparent
    [ ]            image to be annotated
    [x]     black  0_black  black

and export it as 8-bit GRAY .png file to
directory "generated/Semantic/1_stone".

Similar select "red" and "black"

    enable  color  name     background
    ------  -----  ----     ----------
    [ ]     green  1_stone  transparent
    [x]     red    2_void   transparent
    [ ]            image to be annotated
    [x]     black  0_black  black

and export it as 8-bit GRAY .png file to
directory "generated/Semantic/2_void".

Optinally, after export, to save disk space,
images can be converted from 8-bit to 1-bit gray:

    mogrify -threshold 20% *.png

Original image should be copied or similarly selected and exported

    enable  color  name     background
    ------  -----  ----     ----------
    [ ]     green  1_stone  transparent
    [ ]     red    2_void   transparent
    [x]            image to be annotated
    [x]     black  0_black  black

to directory "generated/Image".

# Train

Train generates "knowledge" .pth file, it containins 161MB of coefficients
derived from annotated training set of images. It doesn't include image
content.

To diversify training set, transformation with crop to 900x900 pixels at random
offset and random flip vertical and/or horizontal is applied to input image and
corresponding annotation set. Input size of 800x800-1000x1000, 1:1 aspect
ratio is recommended for resnet50 cnn model. Too large images may loose detail
resolution, too small images may not have sufficient information for CNN to
recognize what is on the image.

PC with recent gaming NVIDIA CUDA GPU (RTX 3090 for example) is recommended.
Results will be visible quicky,
full train done in hour or two. Code will work without GPU but it will be
very slow on i3-i7 CPU needs overnight or few days run, and too slow
to be practical on Celeron CPU, would run for a week or month.

Run, check if no errors, and let it work. Script will save .pth file
every 50 iterations. Script can be stopped and started, it will resume
from saved file.

    ./train.sh

# Infer

To apply trained file for semantic segmentation of some image,
at least one color image of approx 900x900 should be prepared.
Tool internally resizes the input image so some smaller or larger
could be used.

Given one "image.png" or "image.jpg" as paramter "infer.sh" script will
apply trained coefficients, write segmented grayscale image "image_seg.png"
with 3 possible pixel values:

      0 : black : paste
    127 : gray  : stone
    254 : white : void

and display on screen source image
and segmented one.

     ./infer.sh input.jpg

If 2 or more image files as arguments are passed, script will batch-process all
of them and won't display any.

     ./infer.sh tile*.png

To properly segment large image, it should be split to 900x900 tiles,
each tile segmented and results joined back into one large image.
There is automated script for this:

     tools/infer.sh input.jpg
     ...
     scale written to /tmp/airvoid/full_seg.png

Script "split-tiles.sh" center-splits large image to tiles of
900x900 pixels each, leaving out the edges of the image that
wouldn't make full 900x900 tiles.

There are scripts to do this:

     tools/split-tiles.sh image.jpg
     ./infer.sh /tmp/airvoid/tile*.png
     tools/join-tiles.sh
     tools/set-scale.sh /tmp/full_seg.png

Result will be in "/tmp/full_seg.png", and final script sets real scale
of this image, it should normally have same scale as input "image.png",
scale and all other metadata can be printed with:

     identify -verbose image.jpg
     ... long multiline report ...

or

     identify -format "%xx%y %U" image.jpg
     1889.7599999999999909x1889.7599999999999909 PixelsPerCentimeter

And then appled to final image. Imagemagick will recompress image.
.png keeps exact pixel data, .jpg loses a bit of pixel details:

     mogrify -density 4800 -units PixelsPerInch /tmp/full_seg.png


# Original source

# Train neural network for semantic segmentation (deep lab V3) with pytorch in 50 lines of code

Train net semantic segmentation net using LabPics dataset: [https://zenodo.org/record/3697452/files/LabPicsV1.zip?download=1](https://zenodo.org/record/3697452/files/LabPicsV1.zip?download=1) in less then 50 lines of code (note including spaces)

Full toturial that goes with the code can be find here:
https://medium.com/@sagieppel/train-neural-net-for-semantic-segmentation-with-pytorch-in-50-lines-of-code-830c71a6544f
