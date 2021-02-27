# Assignment 3 - Sentiment Analysis
**Christoffer Kramer**  
**27-02-2021**  

Dictionary-based sentiment analysis with Python
Download the following CSV file from Kaggle:
https://www.kaggle.com/therohk/million-headlines
This is a dataset of over a million headlines taken from the Australian news source ABC (Start Date: 2003-02-19 ; End Date: 2020-12-31).
- Calculate the sentiment score for every headline in the data. You can do this using the spaCyTextBlob approach that we covered in class or any other dictionary-based approach in Python.
- Create and save a plot of sentiment over time with a 1-week rolling average
- Create and save a plot of sentiment over time with a 1-month rolling average
- Make sure that you have clear values on the x-axis and that you include the following: a plot title; labels for the x and y axes; and a legend for the plot
- Write a short summary (no more than a paragraph) describing what the two plots show. You should mention the following points: 1) What (if any) are the general trends? 2) What (if any) inferences might you draw from them?

## How to run

**Step 1: Clone repo**
- open terminal
- Navigate to destination for repo
- type the following command
    - _git clone https://github.com/Chris-Kramer/language_assignments.git_

**step 2: set op virtual enviroment:**
- Navigate to the folder "assignment 3".
    - _cd assignment-3_  
- Set up enviroment by running one of this command:
    - _source assignment3_cmk/bin/activate_  
        
**Step 3: Download requirements**
- _pip install -r requirements.txt_
        
**Step 4: Execute script**
- navigate to the folder with the script (src)
    - _cd src_
- run script
    - _python3 assignment3-cmk.py_    
    
The script will probably run for about an hour or an hour and a half, so make sure, you have something to do while waiting. This is probably just me who is bad at optimizing the code :D

### Troubleshooting
- If you have problem installing the requirements file try upgrading pip with "pip install --upgrade pip".

### Output
The output is a jpg-file which can be found in the folder "output" under the name "sentiment_score.jpg".

### Data
source: https://www.kaggle.com/therohk/million-headlines
This file is huge, so it will take a while to run through it.