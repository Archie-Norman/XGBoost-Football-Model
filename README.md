# XGBoost-Football-Model
Predicting Football Games Using Historic Results and XGBoost


**1.py**
makes requests to the API to get the current results data, including both the previous and upcoming results for this season.
(This does make redundant calls to the API as if you want new results, you have to get all results.)

**2.py**
cleans and calculates fixtures such as current points, ELO ratings and other metrics/scores

**3.py**
Trains a model on the historic data and then makes predictions on the upcoming games. Also uses bootstrapping
(yet again redundant to retrain the model every time you want to make a prediction)

**4.py**
Add the odds to the output from 3.py, and then 4.py calculates the expected outcome and Kelly criterion for each prediction 

**model eval.py**
a basic way to evaluate the model, mainly focused on log loss and Brier scores, as the accuracy of probability outputs is more important than the accuracy of classifications  in this context.
