#!/bin/bash

#! Have package installed: imagemagick

magick -delay 20 -loop 0 ../output_images/256x256/*.png ../256x256.gif
