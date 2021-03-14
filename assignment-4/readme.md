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

**Step 1: Clone repo**
- open terminal
- Navigate to destination for repo
- type the following command
 ```console
 git clone https://github.com/Chris-Kramer/language_assignments.git
 ```
**step 2: Run bash script:**
- Navigate to the folder "assignment-4".
```console
cd assignment-4
```  
- Navigate to the folder "src".  
```console
cd src
```  
- Use the bash script _run-script_assignment4-cmk.sh_ to set up environment and run the script:  
```console
bash run-script_assignment4-cmk.sh
```  

## Output
The output is a csv-file with the centrality measures which by default can be found in the folder "output", and a graph of the weighted network, which by default can be found under the folder "viz".

## Parameters
This script takes the following parameters, it have already been supplied with default values. But you are welcome to try and change the default parameters. I have added some edgelists under "data/edgelists", you can use. 

`--edgelist` This is the path and filename of the edgelist  
Default = ../data/edgelists/real_news_edgelist.csv  
`--filter` If you only wish to graph and calculate centrality for edges above a certain weight you can use this parameter.  
Default = 500  
`--output_edgelist` The destination and filename for csv-file with centrality measures  
Default = ../output/edgelist_centrality.csv  
`--viz_output` The destination and filename for the graph.  
Default = ../viz/edgelist_graph.png    

Example:  
```console
bash run-script_assignment4-cmk.sh --edgelist ../data/edgelists/fake_news_edgelist.csv --filter 800 --output_centrality ../output/fake_news_centrality.csv --viz_output ../viz/fake_news_viz.png
```

## Creating edgelists
I have created some utility functions which can create an edgelist from either a directory of txt-files, a txt-file or a csv-file with a column called "text". I have included the corpus of 100 english novels, if you whish to create your own edgelist from one of the files. IMPORTANT: DO NOT USE THE WHOLE CORPUS FOR AN EDGELIST. I TRIED AND IT TAKES HOURS TO COMPLETE. If you want to use a directory of text-files use a directory with an appropriate size.  
Use the bash scrip _run-script_create_edgelist.sh_ for creating an edgelist. It takes the following parameters:  
`--input_file` This is the path to the txt-file, the csv-file or the directory of txt-files.  
Default = NO DEFAULT  
`--output_csv` The script creates a csv-file from (a) txt-file(s) with a column called "text". This parameteter sets the destination for the csv-file. It is not required if you are using a csv-file as input.  
Default = NO DEFAULT  
`--label` The Entity label you wish to use as nodes.  
Default = PERSON  
`--output_edgelist` The destination path for the edgelist  
Default = ../data/edgelists/edgelist.csv  

Example:  
```console
bash run-script_create_edgelist.sh --input_file ../data/raw_data/100_english_novels/corpus/Barclay_Postern_1911.txt --output_csv ../data/raw_data/Barclay_Postern.csv --output_edgelist ../data/edgelists/Barclay_Postern_edgelist.csv
```

## Troubleshooting
I'm using pygraphviz on the worker02 server. If you are running the scripts on a local windows machine, you might experience problems. Try running it on the worker02 server, or a mac/linux machine if you have problems.  
Moreover, networkx is terrible slow so it might take a while to run.