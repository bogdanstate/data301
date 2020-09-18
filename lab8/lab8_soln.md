0. Run this code first:

```python
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
```

1. Using random sampling, break down your dataset into 3 parts:
   - "primary" training (~70%),
   - "secondary" training (~20%), and
   - validation (~10%).

*Hints*:
- you may find `np.random.choice` useful here.
- you can select rows at a particular location in a Pandas data frame using
  `.iloc[list\_of\_indices]`

```python
idx = np.random.choice(3, len(train_data), p=[0.7, 0.2, 0.1])
which_idxs_are_0 = np.arange(len(train_data))[idx == 0]
which_idxs_are_1 = np.arange(len(train_data))[idx == 1]
which_idxs_are_2 = np.arange(len(train_data))[idx == 2]
primary = train_data.iloc[which_idxs_are_0, :]
secondary = train_data.iloc[which_idxs_are_1, :]
valid = train_data.iloc[which_idxs_are_2, :]
```

2. Featurize your primary dataset, by using the following strategy:
  - after replacing NaN values using the empty string, use one-hot encoding
  on the following categorical variables:
    - `Pclass`: the passenger class.
    - `Sex`: passenger sex.
    - `SibSp`: # of siblings or spouses.
    - `Parch`: # of parents / children aboard.
    - `Embarked`: The place where each passenger embarked.

*Hints*:
- you can use `sklearn.preprocessing.OneHotEncoder`
- to replace NaNs w/ an empty strign in pandas you can use ".fillna('', inplace=True)"
- a very lazy way to get the unique values of a Pandas column in a compliant
  format for `OneHotEncoder` is:
  ```
  col_mapping = df['Embarked'].dropna().unique().reshape(-1, 1)
  ```

```python
from sklearn.preprocessing import OneHotEncoder

pclass_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
pclass_col_mapping = train_data['Pclass'].dropna().unique().reshape(-1, 1)
pclass_encoder.fit(pclass_col_mapping)
pclass_X = pclass_encoder.transform(primary[['Pclass']].fillna(''))

sex_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
sex_col_mapping = train_data['Sex'].dropna().unique().reshape(-1, 1)
sex_encoder.fit(sex_col_mapping)
sex_X = sex_encoder.transform(primary[['Sex']].fillna(''))

sibsp_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
sibsp_col_mapping = train_data['SibSp'].dropna().unique().reshape(-1, 1)
sibsp_encoder.fit(sibsp_col_mapping)
sibsp_X = sibsp_encoder.transform(primary[['SibSp']].fillna(''))

parch_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
parch_col_mapping = train_data['Parch'].dropna().unique().reshape(-1, 1)
parch_encoder.fit(parch_col_mapping)
parch_X = parch_encoder.transform(primary[['Parch']].fillna(''))

embarked_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
embarked_col_mapping = train_data['Embarked'].dropna().unique().reshape(-1, 1)
embarked_encoder.fit(embarked_col_mapping)
embarked_X = embarked_encoder.transform(primary[['Embarked']].fillna(''))
```

3. Using the same training dataset (the "primary" dataset from before), train
the following classifiers to predict survival:
    - Logistic classifier
    - KNeighbors classifier
    - Decision tree classifier
    - Support Vector Machine Classifier

Use the following features:
    - `Pclass`: the passenger class.
    - `Sex`: passenger sex.
    - `SibSp`: # of siblings or spouses.
    - `Parch`: # of parents / children aboard.
    - `Embarked`: The place where each passenger embarked.

*Hints*:
- survival is indicated by the `Survived` column
- you can use `np.concatenate([list of arrays], axis=1)` to concatenate np
  arrays and array-likes along their columns

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

X = np.concatenate(
    (pclass_X, sex_X, sibsp_X, parch_X, embarked_X),
    axis=1
)
y = primary['Survived']

logistic = LogisticRegression().fit(X, y)
knn = KNeighborsClassifier().fit(X, y)
decision_tree = DecisionTreeClassifier().fit(X, y)
svm = SVC().fit(X, y)
```

4. Using the four classifiers you trained previously to predict outcomes on the
   validation data. Which of the four classifiers has
   the highest accuracy?

   Important: Remember you need to also featurize the validation data,
   but you need to use the same encoders when featurizing.

   Hint: you can compute accuracy using `sklearn.metrics.accuracy_score`

```python
pclass_X_valid = pclass_encoder.transform(valid[['Pclass']].fillna(''))
sex_X_valid = sex_encoder.transform(valid[['Sex']].fillna(''))
sibsp_X_valid = sibsp_encoder.transform(valid[['SibSp']].fillna(''))
parch_X_valid = parch_encoder.transform(valid[['Parch']].fillna(''))
embarked_X_valid = embarked_encoder.transform(valid[['Embarked']].fillna(''))
X_valid = np.concatenate((pclass_X_valid, sex_X_valid, sibsp_X_valid,
                          parch_X_valid, embarked_X_valid), axis=1)
y_valid = valid['Survived']

pred_logistic_valid = logistic.predict(X_valid)
pred_knn_valid = knn.predict(X_valid)
pred_decision_tree_valid = decision_tree.predict(X_valid)
pred_svm_valid = svm.predict(X_valid)

from sklearn.metrics import accuracy_score
accuracy_logistic_valid = accuracy_score(y_valid, pred_logistic_valid)
accuracy_knn_valid = accuracy_score(y_valid, pred_knn_valid)
accuracy_decision_tree_valid = accuracy_score(y_valid, pred_decision_tree_valid)
accuracy_svm_valid = accuracy_score(y_valid, pred_svm_valid)
print("Logistic: %.4f, KNN: %.4f, Decision tree: %.4f, SVM: %.4f" % (
    accuracy_logistic_valid, accuracy_knn_valid,
    accuracy_decision_tree_valid, accuracy_svm_valid
))
```

5. Get the majority vote from the 4 classifiers, breaking ties at random. What
   is the accuracy of this strategy?

```python
import random
casewise_predictions = zip(pred_logistic_valid, pred_knn_valid,
                           pred_decision_tree_valid, pred_svm_valid)
number_of_votes_for_survival = [sum(x) for x in casewise_predictions]
reconciled_predictions = [
    1 if x > 2 else 0 if x < 2 else random.randint(0, 1)
    for x in number_of_votes_for_survival
]
accuracy_reconciled_valid = accuracy_score(y_valid, reconciled_predictions)
print(accuracy_reconciled_valid)
```

6. Train a logistic regression using as its features the predictions of the
   four models on "secondary" data. Then use this model to weight the
   predictions on the validation data. How well does this strategy do on the validation
   data?

Hint: you can use `np.stack` to "stack" together multiple arrays / array-likes,
to create a higher-dimensional array.

```python
pclass_X_secondary = pclass_encoder.transform(secondary[['Pclass']].fillna(''))
sex_X_secondary = sex_encoder.transform(secondary[['Sex']].fillna(''))
sibsp_X_secondary = sibsp_encoder.transform(secondary[['SibSp']].fillna(''))
parch_X_secondary = parch_encoder.transform(secondary[['Parch']].fillna(''))
embarked_X_secondary = embarked_encoder.transform(secondary[['Embarked']].fillna(''))
X_secondary = np.concatenate((pclass_X_secondary, sex_X_secondary, sibsp_X_secondary,
                          parch_X_secondary, embarked_X_secondary), axis=1)
y_secondary = secondary['Survived']

pred_logistic_secondary = logistic.predict(X_secondary)
pred_knn_secondary = knn.predict(X_secondary)
pred_decision_tree_secondary = decision_tree.predict(X_secondary)
pred_svm_secondary = svm.predict(X_secondary)

pred_X_secondary = np.stack((
    pred_logistic_secondary, pred_knn_secondary,
    pred_decision_tree_secondary, pred_svm_secondary
), axis=1)

blender = LogisticRegression().fit(pred_X_secondary, y_secondary)

pred_X_valid = np.stack((
    pred_logistic_valid, pred_knn_valid,
    pred_decision_tree_valid, pred_svm_valid
), axis=1)
blender_pred_valid = blender.predict(pred_X_valid)
accuracy_blender_valid = accuracy_score(y_valid, blender_pred_valid)
print(accuracy_blender_valid)
```

7. Does the stacking classifier increase accuracy compared to any of the
   individual classifiers? Why / why not? Explain the rough strategy for how we
   would implement cross-validation in this situation.

> The stacking classifier does not necessarily improve accuracy. This is likely
> related to our dealing with a small sample for the training and validation
> sets. We could use random sub-sampling cross validation to assess the
> accuracy of this model, simply be re-running the code from question (1) and
> collecting the results.

8. Train blender models with any number of the 4 classifiers as their features.
   Which model has the best accuracy? Is this a robust way to find the best
   classifier?

```python
from itertools import combinations
training_features = {
    'logistic': pred_logistic_secondary,
    'knn': pred_knn_secondary,
    'decision_tree': pred_decision_tree_secondary,
    'svm': pred_svm_secondary,
}
valid_features = {
    'logistic': pred_logistic_valid,
    'knn': pred_knn_valid,
    'decision_tree': pred_decision_tree_valid,
    'svm': pred_svm_valid,
}

accuracies = {}

for features_comb in {
    y
    for r in range(1, len(training_features))
    for y in combinations(training_features.keys(), r)
}:
    pred_X_secondary = np.stack(
        [training_features[x] for x in features_comb],
        axis=1
    )
    blender = LogisticRegression().fit(pred_X_secondary, y_secondary)
    pred_X_valid = np.stack(
        [valid_features[x] for x in features_comb],
        axis=1
    )
    print(features_comb)
    blender_pred_valid = blender.predict(pred_X_valid)
    accuracy_blender_valid = accuracy_score(y_valid, blender_pred_valid)
    accuracies[features_comb] = accuracy_blender_valid

print(accuracies)
print(
    "The following model has the best accuracy: %s -- %.4f." % (
        ", ".join(max(accuracies.keys(), key=lambda x: accuracies[x])),
        max(accuracies.values())
    )
)
```

> I got logistic as the model with the best accuracy. This is not a robust way
> to find a classifier, since it requires us to go through O(2^k) combinations
> of features. Also, we would need yet another held-out set, beyond the
> validation set to avoid overfitting. This is where strategies like AutoML
> come in.
