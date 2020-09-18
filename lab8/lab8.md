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

2. Featurize your primary dataset, by using the following strategy featurizers:
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

4. Using the four classifiers you trained previously to predict outcomes on the
   validation data. Which of the four classifiers has
   the highest accuracy?

   Important: Remember you need to also featurize the validation data,
   but you need to use the same encoders when featurizing.

   Hint: you can compute accuracy using `sklearn.metrics.accuracy_score`

5. Get the majority vote from the 4 classifiers, breaking ties at random. What
   is the accuracy of this strategy?


6. Train a logistic regression using as its features the predictions of the
   four models on "secondary" data. Then use this model to weight the
   predictions on the validation data. How well does this strategy do on the validation
   data?

Hint: you can use `np.stack` to "stack" together multiple arrays / array-likes,
to create a higher-dimensional array.


7. Does the stacking classifier increase accuracy compared to any of the
   individual classifiers? Why / why not? Explain the rough strategy for how we
   would implement cross-validation in this situation.

8. Train blender models with any number of the 4 classifiers as their features.
   Which model has the best accuracy? Is this a robust way to find the best
   classifier?
