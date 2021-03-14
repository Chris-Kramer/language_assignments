#!/usr/bin/env bash

VENVNAME=as4-venv

cd ..

source $VENVNAME/bin/activate

pip install --upgrade pip
python -m spacy download en_core_web_sm

test -f requirements.txt && pip install -r requirements.txt

cd src

python3 assignment4-cmk.py $@

deactivate

echo "Done! The centrality measures have been calculated and saved in a csv-file, and the network have been graphed"