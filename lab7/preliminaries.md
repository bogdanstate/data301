# Preliminaries for lab 7

The following exercises should help with better understanding
the requirements of lab 7. You can complete them using either
R or Python, although the solutions will be provided in R. 

1. Read "train.csv" into a data frame. Keep only the following
columns in your dataset: 
```
"Age", "SibSp", "Parch", "Fare",
"Pclass", "Sex", "Embarked", "Survived"
```
Which columns have missing data? 

2. We will want to train a binary classifier to predict the 
survival of passengers in our dataset:
- Is our problem balanced or imbalanced in the dependent variable?
- If the classification problem is imbalanced, how would we 
balance our dataset prior to training?

3. True or false: 

- the test set should be a random sample from the
population for which we're trying to make predictions?
- the training set must be a random sample from the
population for which we're trying to make predictions?
- if we augment a training dataset and then train a
classifier based on it, we can use its predictions without
any adjustments on the test set?

4. Implement min-max scaling for the fare "Fare" Variable. 

5. What is "one-hot encoding"? Implement it for the PClass
variable.

6. What is mean imputation? Mode / "most frequent" imputation?
Implement mean imputation for the `Age` variable, and mode 
imputation for the `Embarked` variable.

7. Train a binary logistic classifier predicting survival, 
using just the `Fare`, `Embarked`, and `Pclass` variables as features.

8. Evaluate the classifier's performance on the training set.
How should we interpret this measurement?

9. Can we evaluate the classifier's performance on the test
set? If so, how would we do it? If not, how could we guess it?

10. What is k-fold cross-validation? What is it good for? Can 
you give an example of how it works?

11. Implement 10-fold cross-validation for our logistic classifier.

12. What is the ROC curve? What does the Area Under the Curve (AUC)
tell us about classifier performance? When is it appropriate to use 
an ROC curve?

13. Generate and plot AUC curves for each of the 10 folds in our
sample.
