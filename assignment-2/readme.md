# Assignment 2- String processing with Python
**Christoffer Kramer**  
**15-02-2021**  

Using a text corpus found on the cds-language GitHub repo or a corpus of your own found on a site such as Kaggle, write a Python script which calculates collocates for a specific keyword.

- The script should take a directory of text files, a keyword, and a window size (number of words) as input parameters, and an output file called out/{filename}.csv
- These parameters can be defined in the script itself
- Find out how often each word collocates with the target across the corpus
- Use this to calculate mutual information between the target word and all collocates across the corpus
- Save result as a single file consisting of four columns: collocate, raw_frequency, MI

## How to run

**Step 1: Clone repo**
- open terminal
- Navigate to destination for repo
- type the following command:  
```console
git clone https://github.com/Chris-Kramer/language_assignments.git
```  
**Step 2 navigate to src folder**
- Navigate to the _src_ folder in _assignment-2_:  
```console
cd assignment-2/src
```   
**Step 3 run bash script**
- Type the following command:  
```console
bash run_assignment-2-christoffer.sh
```
The bash script will create a virtual environment, install dependencies, upgrade pip and run the python script.  
The script will after a couple of minutes start printning out words, their frequency and the MI value to the terminal. 
**_since the script might take a while to finish, you can stop it manually by pressing ctrl c, the output-file will still be available._** 

### Output
The output is a csv-file with collocates and their MI value.  

### Parameters
The script takes the following parameters:  
- `--keyword` The target word you want to find collocates with.  
    - Default: "bald"  
- `--window` The size of the window for the KWIC lines, the size indicates how many characters on each side of the target word, the window should contain.  
    - Default: 55  

- `--corpus` A path to a corpus. The corpus must be a folder with txt files. Subfolders and non-txt files will not be included in the MI calculations.  
    - Default: "../100_english_novels/corpus"  
- `--output` Path and filename for the output csv-file.  
    - Default: "../output/output.csv"  

Example:  
```console
bash run_assignment-2-christoffer.sh --keyword single --window 70
```

## Windows users
If you're running on a local windows machine, and do run with a bash shell, you have to set up a virtual environment, activate it and then run the script manually. All packages are part of python v3 standard library. So no need to install any reqirements.txt file. 