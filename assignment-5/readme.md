# Assignment 5 - (Un)supervised machine learning
**Christoffer Kramer**  
**29-03-2021**  
Train an LDA model on your data to extract structured information that can provide insight into your data. For example, maybe you are interested in seeing how different authors cluster together or how concepts change over time in this dataset.  

You should formulate a short research statement explaining why you have chosen this dataset and what you hope to investigate. This only needs to be a paragraph or two long and should be included as a README file along with the code. E.g.: I chose this dataset because I am interested in... I wanted to see if it was possible to predict X for this corpus
In this case, your peer reviewer will not just be looking to the quality of your code. Instead, they'll also consider the whole project including choice of data, methods, and output. Think about how you want your output to look. Should there be visualizations? CSVs?
You should also include a couple of paragraphs in the README on the results, so that a reader can make sense of it all. E.g.: I wanted to study if it was possible to predict X. The most successful model I trained had a weighted accuracy of 0.6, implying that it is not possible to predict X from the text content alone. And so on.  

## Research
I'm going to investigate which topics redditors where discussing during the gamestop saga, where redditors on the subreddit r/Wallstreetbets invested in gamestop stocks. Because of hardware limitations I'm only gonna be looking at the period where the stock prices was most volatile (the 10th of january 2021 - the 17th of february 2021).

_Research question:_ What topics of discussion dominated WallStreetBets during the gamestop saga? 

### Data
I'm using a data set containing all reddit posts and comments from the subreddit r/WallStreetBets. The csv-file is BIG (209 MB), so I'm choosing a limited time window. 
I found the data set on kaggle here: https://www.kaggle.com/unanimad/reddit-rwallstreetbets

### Conclusions
My analysis is only preliminary. However, from a purely computational and statistical point of view, my model preformed better (lower perplexity and higher coherence) when I had 4 topics, however from a human readability perspective the topics seemed more coherent and understandable when using 3 topics. When I had more than 3 topics, my model tended to cluster a few very dominant topics together and then have a lot of topics which only made up a fraction of the overall topics. 
This clustering made me worry that my model made mistakes, however after looking at the pyLDAvis board I realised, that the dominant topics where exlusively related to gamestop stocks and the gamestop saga, while the less prominent topics, where related to the gamestop saga AND a few other stocks. This indicates that the gamestop saga was so dominant, that it essentially created a hegemonic discourse.
When looking at the three topics it is clear, that topic 0 and 1 are related to gamestop and buying stocks, while topic 2 (which is more or less non-existing until the 8th of february) also relates to other stocks. It is also interesting to note that topic 2 contains a lot of words related to mariuanna (weed, pot, cannabis etc.). This indicates that the redditors gained interest in stocks related to mariuanna, once the gamestop stocks started to lose their value. 

## Running the script
If you whish to run the script, you should be aware, that it will take a long time to run (up to on hour). The subreddit was extremely active during this periode. So have something to do, while the script runs. 

### How to run  
**Step 1: Clone repo**  
- open terminal  
- Navigate to destination for repo  
- type the following command  
```console
 git clone https://github.com/Chris-Kramer/language_assignments.git
```  
**step 2: Run bash script:**  
- Navigate to the folder "assignment-5".  
```console
cd assignment-5
```  
- Use the bash script _run_script-lda_reddit.sh_ to set up environment, unzip the csv-file, and run the script:  
```console
bash run_script-lda_reddit.sh
```  
### Output
The bash scripts will save an html-file with a dashboard over topics, a plot which shows how well the models performs based on number of topics, a csv-file with keywords and topics, and a chart which shows how the topics have changed over time (5 day rolling average). These are located in the folder "output".

### Parameters
The script takes the following parameters, it has already ben supplied with default values, but feel free to change them.
- `start_date` Get posts from after this date. The input is a list with the following format YYYY M D  
    - Default: 2021 1 10  
- `end_date` Exclude posts from after this date. The input is a list with the following format YYYY M D  
    - Default: 2021 2 17  
- `rolling_avg` The rolling average that is used for plotting. The value represents days.
    - Default: 5  
- `topics` the number of topics you wish to have in the model  
    - Default: 3  
- `test_limit` The max amount of topics the model should test coherence scores for.
    - Default: 10  
    
    
Example:  
```console
bash run_script-lda_reddit.sh --start_date 2020 12 1 --end_date 2021 2 17 --rolling_avg 10 --topics 6
```
# Running on Windows
This script have not been tested on a Windows machine and the bash script is made for Linux/mac users. If you're running on a local windows machine, and don't have an Unix shell with bash, you have to set up a virtual environment, activate it, install dependencies (requirements.txt and SpaCy's en_core_web_sm nlp model) and unzip the zip-file in the data folder. After that you should be able to run the script from the terminal, since paths will be handled automatically by the python script without any problems.
