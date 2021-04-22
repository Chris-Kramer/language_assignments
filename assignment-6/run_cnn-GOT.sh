#!/usr/bin/env bash

VENVNAME=as6-cmk #Environment name

# Create and activate environment
echo "Creating environment"
python3 -m venv $VENVNAME

echo "Activating environment"
source $VENVNAME/bin/activate

#Upgrade pip and install requirements
echo "Upgrading pip"
pip install --upgrade pip

echo "installing requirements"
# test for problems when installing from requirements.txt and install
test -f requirements.txt && pip install -r requirements.txt

# Move to datafolder and download glove data
cd data/glove

#Download glove data
echo "downloading glove pre-trained data"
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip -q glove.6B.zip

# Move to source folder
cd ../../src

#Run script
echo "running script"
python3 cnn-GOT.py $@

# Deavtivate environment
echo "deactivating and removing environment"
deactivate

#Remove glove data (So I can push repo to git)
cd ../data/glove
rm -rf glove.6B.zip
rm -rf *.txt
cd ../..

# Remove virtual environment
rm -rf $VENVNAME

echo "Done! The results can be found in the folder 'output'"