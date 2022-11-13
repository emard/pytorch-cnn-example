    [x] from train set, take center image and few random crops around avoid scaling image
    [x] support 1-bit mono .png annotations - probably it already works but needs testing
        convert in.png -threshold 20% out.png # 1-bit mono
        convert in.png -colorspace gray +matte -colors 2 -depth 1 out.png # 1-bit mono
    [ ] convert script to export layers from .xcf to .png files
    [ ] progress report on large image
    [x] every 1-2 minutes save unfinished output to track progress
    [ ] margin few pixels tile overlap to avoid boundary conditions
    [x] improve code to erase previous file
    [ ] separate random filp transform
