#!/usr/bin/env python
"""
---------- Import libs ----------
"""
# System tools
import os
# Data analysis
import pandas as pd
from collections import Counter
from itertools import combinations 
from tqdm import tqdm
# NLP
import spacy
nlp = spacy.load("en_core_web_sm")
# drawing
import networkx as nx
import matplotlib.pyplot as plt
import argparse
plt.rcParams["figure.figsize"] = (20,20)


"""
----------- Main function ------------
"""
def main():
    """
    ---------- Parameters -----------
    """
    #Create an argument parser from argparse
    ap = argparse.ArgumentParser(description = "[INFO] Plot edgelist and calculate centrality measures") 
    
    #edgelidt 
    ap.add_argument("-el", "--edgelist",
                    required = False,
                    default = "../data/edgelists/real_news_edgelist.csv",
                    type = str,
                    help = "Edgelist path must be a csv file, Default: ../data/edgelists/real_news_edgelist.csv")
    
    #Cut of weight point
    ap.add_argument("-f", "--filter",
                    required = False,
                    default = 500,
                    type = int,
                    help = "If you only want edges with a weight above a certain number, Default: 500")
    
    #dest for csv output
    ap.add_argument("-oc", "--output_centrality",
                    required = False,
                    default = "../output/edgelist_centrality.csv",
                    type = str,
                    help = "Path to csv output file, Default: ../output/edgelist_centrality.csv")
    
    #dest for plot output
    ap.add_argument("-vo", "--viz_output",
                    default = "../viz/edgelist_graph.png",
                    required = False,
                    type = str,
                    help = "Path to plot output file, Default: ../viz/edgelist_graph.csv")
    args = vars(ap.parse_args())
    
    #Save parameters in variables (this is done for readability)
    data = pd.read_csv(args["edgelist"])
    weighted_point = args["filter"]
    output_file = args["output_centrality"]
    viz_output = args["viz_output"]
    
    #filter data based on weight point
    data = data[data["weight"]>weighted_point]
    
    """
    ------ Create network and plot it -------
    """
    #Create graph netvwork
    G=nx.from_pandas_edgelist(data, 'nodeA', 'nodeB', ["weight"])
    #Plot it
    pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
    #Draw plot
    nx.draw(G, pos, with_labels=True, node_size=20, font_size=10)
    #Save plot
    plt.savefig(viz_output, dpi=300, bbox_inches="tight")
    
    """
    -------- Calc centrality --------
    """
    #Calc Eigenvector centrality
    ev = nx.eigenvector_centrality(G)
    #Calc Betweenness centrality
    bc = nx.betweenness_centrality(G)
    #Calc Degree centrality
    dc = nx.degree_centrality(G)
    
    """
    --------- Create dataframes ---------
    """
    #Create dataframes
    #eigenvector
    df_ev = pd.DataFrame(ev.items()).sort_values(1, ascending=False)
    #change eigenvector columnname
    df_ev = df_ev.rename(columns={1: "eigenvector"})
    
    #Betweenes centrality
    df_bc = pd.DataFrame(bc.items()).sort_values(1, ascending=False)
    #change Betweenes centrality columnname
    df_bc = df_bc.rename(columns={1: "Betweenes"})
    
    #Degree centrality
    df_dc =pd.DataFrame(dc.items()).sort_values(1, ascending=False)
    #change Degree centrality columnname
    df_dc = df_dc.rename(columns={1: "Degree"})
    
    """
    --------- Merge dataframes and save csv---------
    """
    #Merge dataframes
    result = pd.merge(df_ev, df_bc,  how="left", on=[0, 0])
    result = pd.merge(result, df_dc,  how="left", on=[0, 0])
    
    #Rename column "0" to node
    result = result.rename(columns={0: "node"})
    #Save df as csv
    result.to_csv(output_file, index = False)
    
#Define behaviour when called from command line
if __name__ == "__main__":
    main()