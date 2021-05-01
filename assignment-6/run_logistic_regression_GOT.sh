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
# test requirements.txt and install
test -f requirements.txt && pip install -r requirements.txt

# Move to source folder
cd src

#Run script
echo "running script"
python3 logistic_regression_GOT.py $@

# Deavtivate environment
cd ..
echo "deactivating and removing environment"
deactivate

# Remove virtual environment
rm -rf $VENVNAME

echo "Done! The results can be found in the folder 'output'"