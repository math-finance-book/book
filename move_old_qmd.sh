#!/bin/bash

# Create destination directory if it doesn't exist
mkdir -p /mnt/c/Users/kerry/repos/oldqmd

# Move all *_old.qmd files to the oldqmd directory
mv *_old.qmd /mnt/c/Users/kerry/repos/oldqmd/

echo "Moved all *_old.qmd files to /mnt/c/Users/kerry/repos/oldqmd/"