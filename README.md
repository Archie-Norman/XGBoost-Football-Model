# XGBoost-Football-Model
Predicting Football Games Using Historic Results and XGBoost

**report.docx**
contains a report of the porject 

**step-1(results.api).py**
the code used to get the data from the api

**step-2(cleaning).py**
where the cleaning and feature engineering took place

**step-3(xgboost predict).py**
code for the model and predictions

**step-4(EV bets).py**
code that took the predictions and gives the bets and the bet amounts

*the odds have to be manly input into the csv between steps 3 and 4

**back testing cleaning stuff.py**
the same as step-2(cleaning).py but altered to take the historical data csvs and clean them all at once. must have all the histical data in the same folder and nothing else.

**back test model with flat bet.py**
takes the histocial data, spltis it then makes predictions and then calcualtes winnigs.

**eval.py**
sort of a test bench for the model with evaluation metrics
