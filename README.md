# ixhw

There are several steps to my project:

* EDA - done

* Random Forest Model - done

* Neural Network Model - Tabular - pending


# EDA
Objectives:
* Identify missing data, if any
  * Impute accordingly
* Examine all "object" data types within dataframe for anomalous values
* Examine categorical input data
  * Cardinality
  * Evalute encoding options for each categorical field: Ordinal, OHE, Target Encoding, and Embedding (for NN)
* Examine target data for imbalance
* Examine input data for imbalance

![EDA1](eda_charts_1.png)
![EDA2](eda_chart_2.png)

Outcomes of EDA:
* No missing data
* "Objects" in dataframe had no anomalous values 
* Used ordinal encoding for 'month' and 'education' fields
  * Made 'sense' for a tree model:  data looked like it could split, nicely, in ordinal manner.  
* Used OHE for: 'marital','default', 'housing', 'loan', 'contact'
* Used cross-validated target encoding for 'job'
* Target data highly imbalanced:  roughly 89%/11%
* Some input data extremely imbalanced: default data 98% negative and 2% positive.  Some jobs in 'job' and also some months in 'month' were also very rare.

# Imblearn RF Classifier
* Use RF to try and learn features without lots of tuning/parameter optimization - good for comparison, later.
* Maybe get lucky and have a good model (not in our case!)
* Use imblearn BalancedRandomForest classifier in conjuction with sklearn RepeatedStratifiedKfold cv.
* Had to hand-code train-test loop for cv mean target encoding or else would have to edit and test new objects within API - hence, no pipelines, etc.
* Expectation was that between the model and the repeated, stratified, and random cross-validation, my fitted model would capture at least some of the lower-frequency data.
* This was not the case: using 5 stratified folds and 10 iterations of the folding, the mean recall was over 80%, while mean f1 score was 54% and mean precision was 41%.
* The BalancedRandomForest model identified 'duration' as the most important feature.
* I want to find the most important feature using a NN and contrast/compare them.
![RF1](RF_features.png)
