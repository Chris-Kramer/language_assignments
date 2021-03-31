#!/usr/bin/env bash

#Environment name
VENVNAME=as5-venv-cmk

#Activate environment
source $VENVNAME/bin/activate

#Upgrade pip
pip install --upgrade pip

#Test requirements and install requirements
test -f requirements.txt && pip install -r requirements.txt

#Download en_core_web nlp
python -m spacy download en_core_web_sm

#Move to data folder
cd data

#Unzip csv file (The file is to big to upload)
unzip r_wallstreetbets_posts.csv.zip

#Move to source folder
cd ../src

#Run python script
python lda-reddit.py $@

#Move to data folder
cd ../data

#Remove unzipped csv file (this is done, so I can push the repo without hitting the limit for data storage)
rm -f r_wallstreetbets_posts.csv 

#Deavtivate environment
deactivate

#Print this to the screen 
echo "Done! The results can be found in the folder 'output'"