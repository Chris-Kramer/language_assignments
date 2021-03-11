#!/usr/bin/env bash

VENVNAME=assign4-cmk

cd ..

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate

pip install --upgrade pip

pip install ipython
pip install jupyter

python -m ipykernel install --user --name=$VENVNAME

test -f requirements.txt && pip install -r requirements.txt

deactivate
echo "build $VENVNAME"
