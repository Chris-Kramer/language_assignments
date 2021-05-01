#!/usr/bin/env bash

VENVNAME=as4-venv #Environment name

# Create and activate environment
echo "Creating environment"
python3 -m venv $VENVNAME

source $VENVNAME/bin/activate

# Upgrade pip
echo "Upgrading pip"
pip install --upgrade pip

#Test and install requirements
test -f requirements.txt && pip install -r requirements.txt

# Download en_core_web nlp model
python -m spacy download en_core_web_sm

#Move to src folder
cd src

echo "running script"
python3 create_edgelist.py $@

echo "deactivating and removing environment"
deactivate

# move back to parent dir
cd ..

# Remove virtual environment
rm -rf $VENVNAME

echo "Done! The edgelist have been created"