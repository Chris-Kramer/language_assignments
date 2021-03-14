"""
----------- Import libs ------------
"""
import os
import sys
sys.path.append(os.path.join(".."))
import argparse
from utils.utils import dir_to_csv
from utils.utils import txt_to_csv
from utils.utils import create_edgelist
"""
---------- Main function ----------
"""
def main():
    """
    ---------- Parameters -----------
    """
    ap = argparse.ArgumentParser(description = "[INFO]Make a txt file, or a directory of txt files of a csv file with a column called 'text' into an edgelist,") 
    #input 
    ap.add_argument("-i_f", "--input_file",
                    required = True,
                    type = str,
                    help = "Must be a path to a txt file, a directory with txt files or a csv_file with a text column called 'text'")
    #Output csv
    ap.add_argument("-o_csv", "--output_csv",
                     required = False,
                     type = str,
                     help = "Must be a path to the output of the csv-file. EXAMPLE: ../data/raw_data/100_english_novels.csv")
    #Entity label
    ap.add_argument("-l", "--label",
                     required = False,
                     default = "PERSON",
                     type = str,
                     help = "The entity label for entity recognition. EXAMPLE: PERSON")
    #Output edgelist
    ap.add_argument("-o_e", "--output_edgelist",
                     required = False,
                     default = "../data/edgelists/edgelist.csv",
                     type = str,
                     help = "Must be a path to the output of the edgelist. EXAMPLE: ../data/edgelists/edgelist.csv")
    
    args = vars(ap.parse_args())
    
    #Save arguments in variables for readability
    input_file = args["input_file"]
    output_csv = args["output_csv"]
    output_edgelist = args["output_edgelist"]
    label = args["label"]
    """
    ---------- Test input type and create csv -----------
    """
    #If input is a txt file
    if input_file.endswith(".txt"):
        txt_to_csv(input_file, output_csv)
    #Else if input is a directory    
    elif os.path.isdir(input_file):
        dir_to_csv(input_file, output_csv)
    #Else if input file is a csv file
    elif input_file.endswith(".csv"):
        create_edgelist(input_file, output_edgelist, label)
        sys.exit()
    #If input is neither a txt, csv or dir
    else:
        print("The input must be a txt file, a csv file or a directory of txt files")
        sys.exit()
    """
    ---------- Create edgelist ----------
    """
    #Create edgelist
    create_edgelist(output_csv, output_edgelist, label)
    
#Define behaviour when called from commandline
if __name__=="__main__":
    main()