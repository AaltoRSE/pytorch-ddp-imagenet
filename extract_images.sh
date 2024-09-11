#!/bin/bash
mkdir -p /tmp/ILSVRC2012_extracted

tar -xvf /scratch/shareddata/dldata/imagenet/ILSVRC2012_img_train.tar -C /tmp/ILSVRC2012_extracted 

# Parallel extraction of tar files
find /tmp/ILSVRC2012_extracted -name "*.tar" | parallel -j+0 '
    foldername=$(basename {} .tar)
    mkdir -p /tmp/ILSVRC2012_extracted/"$foldername"
    tar -xf {} -C /tmp/ILSVRC2012_extracted/"$foldername"
    rm {}
'

