#!/usr/bin/env bash

VENVNAME=as2-cmk #Environment name

cd ..

# Create and activate environment
echo "Creating environment"
python3 -m venv $VENVNAME

echo "Activating environment"
source $VENVNAME/bin/activate

#Upgrade pip and install requirements
echo "Upgrading pip"
pip install --upgrade pip

# Move to datafolder and download glove data
cd src

#Run script
echo "running script"
python3 assignment-2-christoffer.py $@

# Deavtivate environment
echo "deactivating and removing environment"
deactivate

#move back to main dir
cd ..

# Remove virtual environment
rm -rf $VENVNAME