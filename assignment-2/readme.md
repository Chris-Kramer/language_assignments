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
- type the following command
    - _git clone https://github.com/Chris-Kramer/language_assignments.git_

**step 2: set op virtual enviroment:**
- Navigate to the folder "assignment 2".
    - _cd assignment-2_  
-set up enviroment by running one of these commands:
    - _source lang_HW2_chris-env/bin/activate_ (Mac and linux)
    - _lang_HW2_chris-env\Scripts\activate.bat_ (Windows)
        
**Step 3: Download requirements**
- _pip install -r requirements.txt_
        
**Step 4: Execute script**
- navigate to the folder with the script (src)
    - _cd src_
- run script
    - _python3 assignment-2-christoffer.py_

**_since the script might take a while to finish, you can stop it manually by pressing ctrl c, the output-file will still be available._** 

### Output
The output is a csv-file which can be found in the folder "output" under the name "output.csv".

### Parameters
The  parameters are the following:
- keyword = "bald"
- window_size = 55
- corpus_dir = os.path.join("..", "100_english_novels", "corpus") #path to directory
- output = os.path.join("..", "output", "output.csv") #destination and name for output-file

They are located in line 14-17. If you whish to change them. 