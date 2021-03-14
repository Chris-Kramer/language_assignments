#!/usr/bin/env bash

VENVNAME=as4-venv

cd ..

source $VENVNAME/bin/activate

pip install --upgrade pip
python -m spacy download en_core_web_sm

test -f requirements.txt && pip install -r requirements.txt

cd src

python3 create_edgelist.py $@

deactivate

echo "Done! The edgelist have been created"