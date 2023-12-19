#!/bin/sh

# scanner LIDE400 ima slab kontrast boje za kredu
# "IRWIN fluorescent orange marking chalk"
# umjesto intenzivne narancaste boje dobije se
# bijela koja sasvim malo vuce na ruzicasto

# -auto-level pojaca kontrast razvlacenjem histograma
# -sharpen 0x5.0 poveca ostrinu konvolucijom
# skomprimira .png u .jpg

# kreda na uzorku se bolje vidi
# file se smanji 600->45 MB

# -unsharp RADIUSxSIGMA+GAIN+THRESHOLD is faster than -sharpen RADIUSxSIGMA
# -unsharp can use GPU, then it will be be even more faster

convert \
$1 \
-unsharp 0x7+1+0 \
$2
