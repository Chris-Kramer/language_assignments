# Assignment 4 - Network analysis
**Christoffer Kramer**  
**14-03-2021**  

**Creating reusable network analysis pipeline**  
This exercise is building directly on the work we did in class. I want you to take the code we developed together and in you groups and turn it into a reusable command-line tool. You can see the code from class here:  
https://github.com/CDS-AU-DK/cds-language/blob/main/notebooks/session6.ipynb  
This command-line tool will take a given dataset and perform simple network analysis. In particular, it will build networks based on entities appearing together in the same documents, like we did in class.  
- Your script should be able to be run from the command line  
- It should take any weighted edgelist as an input, providing that edgelist is saved as a CSV with the column headers "nodeA", "nodeB"  
- For any given weighted edgelist given as an input, your script should be used to create a network visualization, which will be saved in a folder called viz.  
- It should also create a data frame showing the degree, betweenness, and eigenvector centrality for each node. It should save this as a CSV in a folder called output.  

## How to run
NOTE: If using your own custom edgelist it might take a while to run!  
**Step 1: Clone repo**
- open terminal
- Navigate to destination for repo
- type the following command
 ```console
 git clone https://github.com/Chris-Kramer/language_assignments.git
 ```
**step 2: Run bash script(s):**
- Navigate to "assignment-4".
```console
cd assignment-4
```  
- Use the bash script _run-script_assignment4-cmk.sh_ to set up environment and run the script for vizualisation and centrality calculations:  
```console
bash run-script_assignment4-cmk.sh
```
- (OPTIONAL) Use the bash script _run-script_create_edgelist.sh_ to create an edgelist (need an input - see section "Creating edgelists" for more information):  
```console
bash run-script_create_edgelist.sh --input_file some-input-file.txt
```

## Output
The output is a csv-file with the centrality measures which by can be found in the folder "output", and a graph of the network, which can be found under the folder "viz".

## Parameters
This script takes the following parameters, it have already been supplied with default values. But you are welcome to try and change the default parameters. I have added some edgelists under "data/edgelists", you can use. 

- `--edgelist` The filename of the edgelist. The edgelist must be located in the folder "data/edgelists"  
    - Default = real_news_edgelist.csv  
- `--filter` Specify how large a weight an edge should have in order to be included in the centrality measures. calculate centrality measures.  
    - Default = 500  
- `--csv_output` The filename for the csv-file with centrality measures. The file will be located in the folder "output".  
    - Default = edgelist_centrality.csv  
- `--viz_output` The filename for the network visualisation. The file will be located in the folder "viz".  
    - Default = edgelist_graph.png    

Example:  
```console
bash run-script_assignment4-cmk.sh --edgelist fake_news_edgelist.csv --filter 800 --csv_output fake_news_centrality.csv --viz_output fake_news_viz.png
```

## Creating edgelists
NOTE: If creating your own custom edgelist it might take a while to run!  
I have created some utility functions which can create an edgelist from either a directory of txt-files, a txt-file or a csv-file with a column called "text". I have included the corpus of 100 english novels, if you wish to create your own edgelist from one of the files. Note that if you use the whole corpus as an edgelist it will take a very long time to run. 
The script creates a csv-file from a txt-file (or directory of txt-files) with the columns "title" and "text". The csv-file is then used for creating an edgelist. The name of the csv-file will be the same as the input file and it will be saved in the folder "data/raw_data.  

Use the bash scrip _run-script_create_edgelist.sh_ for creating an edgelist. It takes the following parameters:  
- `--input_file` This is the path to the txt-file, the csv-file or the directory of txt-files. It must be located in the folder "data/raw_data". So if you wish to use a file from the corpus you must move it to the correct directory (in this case the parent directory) first.  
    - Default = NO DEFAULT    
- `--label` The Entity label you wish to use as nodes. These labels comes from SpaCy's library and can be found here https://spacy.io/models/en.  
    - Default = PERSON  
- `--output_edgelist` The name of the edgelist. The edgelist will be saved in the folder "data/edgelists".  
    - Default = edgelist.csv  
    
Example:  
```console
bash run-script_create_edgelist.sh --input_file Barclay_Postern_1911.txt --label ORG --output_edgelist Barclay_Postern_edgelist.csv
```

# Running on Windows
This script have not been tested on a Windows machine and the bash script is made for Linux/mac users. If you're running on a local windows machine, and don't have an Unix shell with bash, you have to set up a virtual environment, activate it, install dependencies (requirements.txt and SpaCy's en_core_web_sm nlp model) and then run the script manually. Paths should be handled automatically by the python script without any problems.