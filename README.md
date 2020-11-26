# week11_machine_learning_project

In this project, the risk of loan default is predicted using various Machine Learning sampling techniques and Random classifiers.

## Libraries/Technologies used
- numpy
- pandas
- pathlib
- warnings
- collections
- sklearn.metrics
- sklearn.linear_model
- sklearn.model_selection
- sklearn.preprocessing
- imblearn.over_sampling
- imblearn.under_sampling
- imblearn.combine
- imblearn.ensemble

## Tools used
- Github
- Gitbash
- Gitlab
- Slack
- Jupyter lab
- Microsoft CSV
- WinZip

## Data Given
- Loan data of customers from Jan 2019- Mar 2019. Source (LendingClub)

## Notebooks created
- credit_risk_resampling (Predictions by resampling the data)
- credit_risk_ensemble (Predictions by Random classifiers)

## Preset values for the matrics
Wherever applicable, the following values are preset throughout the model:
- The random_state is 42
- The number of estimators is 100

## Data preparation/ cleaning
For both the Jupyter notebooks, the data preparation and cleaning methodology is the same.
* The final two rows of the CSV contains redundant data, thus, its omitted.
* All the null value columns are dropped.
* The loans that have 'Issued' status are removed.
* The '%' symbol from interest rate is removed. The value is converted to a float datatype.
* The target value is transformed to a binary state. The loans having 'Current' status is changed to 'low_risk', while all the other loan statuses are assumed as 'high_risk'.
* The target variable (y) is the loan status, while the feature variables(X) are everything else.
* The X variables that contain strings are binary encoded via dummy values, so that they can be scaled.

## Feature and target data
- Target: Loan Status
- Feature: All columns except loan status
- Positive is the number of predicted low-risk loans
- Negative is the number of predicted high-risk loans

## Train-test split/ Data Scaling
- The feature and target data is randomly split into testing and training. A random_state of 42 is assigned for all instances.
- The target values in the training data is heavily disproportionate, the low-risk loans far outweighs the high-risk loans.
- The scaler function is fit to the training data.
- The testing and training data is sclaled to eliminate bias in the prediction.

## Prediction methods
### Resampling methods
- Naive oversampling
- Synthetic Minority Oversampling Technique (SMOTE)
- Cluster Centroid (CC) undersampling
- Synthetic Minority Oversampling Technique Edited Nearest Neighbours (SMOTEENN)

### Ensemble methods
- Balanced Random Forest Classifier
- Easy Ensemble Classifier

## Metrics calcuated
For each of the prediction methods listed above, the following metrics are computed.
* Balanced Accuracy score
* Confusion Matrix
* Imbalanced classification report

## Performance comparision
### Resampling methods
#### Balanced accuracy score: 
- Naive: 0.82
- SMOTE: 0.80
- CC: 0.78
- SMOTEENN: 0.81
- Among the resampling methods, the Naive method has the top accuracy score, narrowly ahead of SMOTEENN and SMOTE methods.

#### Confusion matrix
- False positive will have the greatest damage
- Naive: 21 False Positives
- SMOTE: 27 False Positives
- CC: 19 False Positives
- SMOTEENN: 2261 False Positives (largest).

#### Recall score
- Naive: 0.85
- SMOTE: 0.88
- CC: 0.76
- SMOTEENN: 0.87
- Among the resampling methods, the SMOTE method has the top recall score, narrowly ahead of SMOTEENN and Naive.

#### Geometric mean score
- Naive: 0.82
- SMOTE: 0.81
- CC: 0.79
- SMOTEENN: 0.18
- Naive method has the best Geometric mean score

### Ensemble learning
#### Balanced accuracy score: 
- BRFC: 0.77
- EEC: 0.93

#### Confusion matrix
- BRFC: 35 False Positives
- EEC: 8 False Positives

#### Recall score
- BRFC: 0.88
- EEC: 0.94

#### Geometric mean score
- BRFC: 0.76
- EEC: 0.93

## Interpretation
- SMOTEENN method has the highest amount of False Positives and the lowest geometric mean among all the models.
- Naive method is apparently the best among resampling methods.
- SMOTE method fares well in many of the calculated metrics.
- CC method has the most false-negatives (4066), though it is not undesirable. Yet it is unreliable.
- Top 3 features in BRFC are:
    - 'total_rec_prncp'
    - 'total_pymnt_inv'
    - 'total_rec_int'

## Conclusion
- EEC method comes out as the most reliable among all, with a substantial accuracy, recall, geometric mean and least amount of false-positives.

## Contributors
- Satheesh Narasimman

## People who helped
- Khaled Karman, Bootcamp tutor

## References
- https://imbalanced-learn.org/stable/generated/imblearn.metrics.classification_report_imbalanced.html

- https://imbalanced-learn.readthedocs.io/en/stable/api.html#module-imblearn.combine

- https://imbalanced-learn.readthedocs.io/en/stable/api.html#module-imblearn.under_sampling

- https://imbalanced-learn.readthedocs.io/en/stable/api.html#module-imblearn.over_sampling

- https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.BalancedRandomForestClassifier.html

- https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.EasyEnsembleClassifier.html

- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

- https://scikit-learn.org/stable/modules/model_evaluation.html