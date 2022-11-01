    [x] from train set, take center image and few random crops around avoid scaling image
    [x] support 1-bit mono .png annotations - probably it already works but needs testing
        convert in.png -threshold 20% out.png # 1-bit mono
        convert in.png -colorspace gray +matte -colors 2 -depth 1 out.png # 1-bit mono
    [ ] convert script to export layers from .xcf to .png files
