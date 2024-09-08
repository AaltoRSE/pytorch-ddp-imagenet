#!/bin/bash

# Step 1: List all tar files in the archive and save to a temporary file
tar -tf /scratch/shareddata/dldata/imagenet/ILSVRC2012_img_train.tar > all_classes.txt

# Step 2: Randomly pick 5 tar files or just the first 5 tar files
# shuf -n 5 all_classes.txt > sample_5classes.txt
head -n 5 all_classes.txt > sample_5classes.txt


# Step 3: Loop through each selected tar file, create a folder, and extract it
while read -r file; do
  # Extract the base name of the tar file (without the .tar extension)
  folder_name=$(basename "$file" .tar)
  
  # Create a folder with the same name in the current working directory
  mkdir -p "$folder_name"
  
  # Extract the tar file, preserving the directory structure
  tar --extract --file=/scratch/shareddata/dldata/imagenet/ILSVRC2012_img_train.tar "$file" --directory "$folder_name"
  
  # # Now extract the inner tar file into the same folder
  tar --extract --file="$file" --directory "$folder_name"

  rm "$file"
done < sample_5classes.txt

# Clean up temporary files
rm all_classes.txt sample_5classes.txt

echo "Extraction complete! All images are in their respective folders."
