#!/bin/bash
mkdir -p /tmp/ILSVRC2012_extracted

tar -xvf /scratch/shareddata/dldata/imagenet/ILSVRC2012_img_train.tar -C /tmp/ILSVRC2012_extracted 
for tarfile in /tmp/ILSVRC2012_extracted/*.tar; do
    foldername=$(basename "$tarfile" .tar)
    mkdir -p /tmp/ILSVRC2012_extracted/"$foldername"
    tar -xf "$tarfile" -C /tmp/ILSVRC2012_extracted/"$foldername"
done


