# ixhw

There are several main steps to my project:

* EDA

* Random Forest Model

* Neural Network Model - Tabular


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

Outcomes of EDA:
* No missing data
* "Objects" in dataframe had no anomalous values 
* Used ordinal encoding for 'month' and 'education' fields
* Used OHE for: 'marital','default', 'housing', 'loan', 'contact'
* Used cross-validated target encoding for 'job'
* Target data highly imbalanced:  roughly 89%/11%
* Some input data extremely imbalanced: default data 98% negative and 2% positive.  Some jobs in 'job' and also some months in 'month' were also very rare.

#Imblearn RF Classifier

