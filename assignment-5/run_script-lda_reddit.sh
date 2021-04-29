#!/usr/bin/env bash

#Environment name
VENVNAME=as5-venv-cmk

#Create environment
echo "Creating environment"
python -m venv $VENVNAME

#Activate environment
source $VENVNAME/bin/activate

echo "Upgrading pip and installing dependencies"
#Upgrade pip
pip install --upgrade pip

#Test requirements and install requirements
test -f requirements.txt && pip install -r requirements.txt

#Download en_core_web nlp
python -m spacy download en_core_web_sm

#Move to data folder
cd data

echo "Unzipping data"
#Unzip csv file (The file is to big to upload)
unzip r_wallstreetbets_posts.csv.zip

#Move to source folder
cd ../src

echo "Running script"
#Run python script
python lda-reddit.py $@

#Move to data folder
cd ../data

echo "Removing unzipped file"
#Remove unzipped csv file (this is done, so I can push the repo without hitting the limit for data storage)
rm -f r_wallstreetbets_posts.csv 

echo "Deactivating and removing environment"
#Deavtivate environment
deactivate

cd ..
rm -rf $VENVNAME

#Print this to the screen 
echo "Done! The results can be found in the folder 'output'"