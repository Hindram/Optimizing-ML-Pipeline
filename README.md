# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset shows information about the marketing campaign of a bank,  The aim of the analysis is to predict potential customers who are willing to contribute to the bank's future loan or deposit plans. Hence, improving the overall future performance. 
The dataset contains 11162 records with 17 columns.

The best performing model was a VotingEnsemble pipeline using AutoML with 0.917 accuracy rate. 

## Scikit-learn Pipeline
After retrieving the dataset from the provided URL using TabularDataFactory class. The followed steps have been applied to for Scikit-learn Pipeline:
- Dataset has been cleared and preprocessed using clean_data method in train.py script. The preprocessing involved dropping nulls values, encoding categorical features using a one-hot numeric approach, and more. 
- Then splitting the data into training and testing set using train_test_split class.
- LogisticRegression class has been used to fit the model. 
- Defining the estimator then passed it to HyperDriveConfig script.
- Parameter sampler 'RandomParameterSampling' holds the tuning hyperparameters (--C: Inverse of regularization, --max_iter: Maximum number of iterations) to be passed to the HyperDriveConfig script. 
- Early termination policy has been added to the script then experiment submission. 

The best model of the pipeline was recorded with the following result: accuracy = 0.913, C=0.001, and max_iter=50.

**What are the benefits of the parameter sampler you chose?**
Discrete values with 'choice' have been used for both tuned parameters. RandomParameterSampling has been selected due to its fast performance, simple approach, and would provide random unbiased search in the overall population.  

**What are the benefits of the early stopping policy you chose?**
BanditPolicy has been used as an early stopping policy to improve the performance of the computational resources by automatically terminating poorly and delayed performing runs. 

## AutoML
Used hyperparameters for AutoML:

** automl_config = AutoMLConfig(
    experiment_timeout_minutes=20,
    task='classification',
    primary_metric='accuracy',
    training_data=ds,
    compute_target=compute_target,
    label_column_name='y',
    n_cross_validations=5) **

- experiment_timeout_minutes has been reduced to 20 minutes to avoid the run time out failure. 
- Experiment type set to 'classification'.
- Accuracy has been selected as a primary metric.
- 5 folds have been selected for cross-validation parameter. 

images/AutoML-best-model.png

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
