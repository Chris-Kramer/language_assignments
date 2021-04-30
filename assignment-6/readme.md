# Assignment 6 - Text classification using Deep Learning
**Christoffer Kramer**  
**20-04-2021**  
In class this week, we've seen how deep learning models like CNNs can be used for text classification purposes. For your assignment this week, I want you to see how successfully you can use these kind of models to classify a specific kind of cultural data - scripts from the TV series Game of Thrones.  
You can find the data here: https://www.kaggle.com/albenft/game-of-thrones-script-all-seasons  
In particular, I want you to see how accurately you can model the relationship between each season and the lines spoken. That is to say - can you predict which season a line comes from? Or to phrase that another way, is dialogue a good predictor of season?  
Start by making a baseline using a 'classical' ML solution such as CountVectorization + LogisticRegression and use this as a means of evaluating how well your model performs. Then you should try to come up with a solution which uses a DL model, such as the CNNs we went over in class.

## Approach


### Data


### Conclusions


## Running the script


### How to run  
**Step 1: Clone repo**  
- open terminal  
- Navigate to destination for repo  
- type the following command  
```console
 git clone https://github.com/Chris-Kramer/language_assignments.git
```  
**step 2: Run bash script:**  
- Navigate to the folder "assignment-6".  
```console
cd assignment-6
```  
- Use the bash script _runs_script-lda_reddit.sh_ to set up environment, unzip the csv-file, and run the script:  
```console
bash runs_script-lda_reddit.sh
```  
### Output


### Parameters
```console
bash runs_script-lda_reddit.sh --start_date 2021 1 15 --end_date 2021 2 15 --topics 2
```

## Running on windows
