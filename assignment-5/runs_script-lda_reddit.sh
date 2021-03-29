#!/usr/bin/env bash

VENVNAME=as5-venv-cmk

source $VENVNAME/bin/activate

pip install --upgrade pip

test -f requirements.txt && pip install -r requirements.txt
python -m spacy download en_core_web_sm

cd data

unzip r_wallstreetbets_posts.csv.zip

cd ../src

python3 lda-reddit.py $@

cd ../data

rm -f r_wallstreetbets_posts.csv 

deactivate

echo "Done! The results can be found in the folder 'output'"