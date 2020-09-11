# Preliminaries for lab 7

The following exercises should help with better understanding
the requirements of lab 7. You can complete them using either
R or Python, although the solutions will be provided in R. 

1. Read "train.csv" into a data frame. Keep only the following
columns in your dataset: 
```R
"Age", "SibSp", "Parch", "Fare",
"Pclass", "Sex", "Embarked", "Survived"
```
Which columns have missing data? 

```R
data <- read.csv('train.csv')
data <- data[,c(
  "Age", "SibSp", "Parch", "Fare",
  "Pclass", "Sex", "Embarked", "Survived"
)]
summary(data)
```

> `Age` has missing data (see NAs). `Embarked` also has missing
data (see empty string as value).

2. We will want to train a binary classifier to predict the 
survival of passengers in our dataset:
- Is our problem balanced or imbalanced in the dependent variable?
> `table(data$survived)` reveals the problem is imbalanced. It is
not a *severe* imbalance, however.
- If the classification problem is imbalanced, how would we 
balance our dataset prior to training?
> The easiest solution would be random undersampling of the majority
class. Techniques such as SMOTE are a more sophisticated option.
3. True or false: 

- the test set should be a random sample from the
population for which we're trying to make predictions?
> True: we're trying to say something about our classifier's
generalizability to unseen data drawn from the same data generating
process that generated our dataset. 
- the training set must be a random sample from the
population for which we're trying to make predictions?
> False: Not necessarily. We can use data augmentation techniques
on the training set.
- if we augment a training dataset and then train a
classifier based on it, we can use its predictions without
any adjustments on the test set?
> False: We need to adjust our predicted probabilities, to account
for the balancing strategies.

4. Implement min-max scaling for the fare "Fare" Variable. 

```R
data$Fare_minmax <- (data$Fare - min(data$Fare)) / max(data$Fare)
```

5. What is "one-hot encoding"? Implement it for the PClass
variable.

> One hot encoding involves turning a categorical variable with
k values into a feature vector of dimension k, which is 0 everywhere,
except for the value corresponding to the column assignment in
each row. 

```R
data$Pclass_1 <- ifelse(data$Pclass == 1, 1, 0)
data$Pclass_2 <- ifelse(data$Pclass == 2, 1, 0)
data$Pclass_3 <- ifelse(data$Pclass == 3, 1, 0)
```
*Note that for the single-valued categorical variable case,
 the k-th entry in the vector can always be written as a sum of 
the other ones (this is called multi-collinearity).*

6. What is mean imputation? Mode / "most frequent" imputation?
Implement mean imputation for the `Age` variable, and mode 
imputation for the `Embarked` variable.

Mean imputation involves replacing missing values of continuous
numerical variables with their mean. Mode imputation involves
replacing missing values of discrete variables (whether numerical
or categorical) with their mode.

```R
data$Age <- ifelse(is.na(data$Age), mean(data$Age, na.rm=T), data$Age)
most_frequent_Embarked <- names(sort(table(data$Embarked), decreasing=T))[1]
data$Embarked <- ifelse(data$Embarked == '', most_frequent_Embarked, data$Embarked) 
```

7. Train a binary logistic classifier predicting survival, 
using just the `Fare`, `Embarked`, and `Pclass` variables as features.

```R
model <- glm(Survived~Fare+Embarked+Pclass, data=data, family=binomial(link = "logit"))
```
*Note:* The `glm` function automatically expands categorical variables via
one-hot encoding.

8. Evaluate the classifier's performance on the training set.
How should we interpret this measurement?

> We can use Accuracy ("the fraction of predictions our model got right")
as a measurement. We can compute this via:

```R
predicted.probs <- plogis(predict.glm(model))
threshold <- mean(data$Survived)
ground.truth <- data$Survived
predicted.labels <- as.numeric(predicted.probs >= threshold)
accuracy <- sum(predicted.labels == ground.truth) / length(ground.truth)
```

But note: WE ARE CHEATING. We're predicting on our training set. Perfectly
possible to get 100% accuracy just by overfitting!

9. Can we evaluate the classifier's performance on the test
set? If so, how would we do it? If not, how could we guess it?

> We cannot since the test set provides no labels (this is often the case
in challenges). We could guess this performance by setting up a validation
set, e.g. by sampling at random from the (unaugmented) training set.

10. What is k-fold cross-validation? What is it good for? Can 
you give an example of how it works?

> Divide training set into k folds. Take data in the first fold and treat
it as the validation set. Use the rest of the data as your training set.
Measure performance. Repeat for folds 2..k.

11. Implement 10-fold cross-validation for our logistic classifier.

```R
data$fold <- sample(1:10, nrow(data), replace=T)
accuracies <- sapply(1:10, function(i) {
  train.data <- data[data$fold != i,]
  valid.data <- data[data$fold == i,]
  model <- glm(Survived~Fare+Embarked+Pclass, data=train.data,
               family=binomial(link = "logit"))
  predicted.probs <- plogis(predict.glm(model, newdata=valid.data))
  threshold <- mean(train.data$Survived)
  ground.truth <- valid.data$Survived
  predicted.labels <- as.numeric(predicted.probs >= threshold)
  accuracy <- sum(predicted.labels == ground.truth) / length(ground.truth)
  return(accuracy)
})
```

*Note:* Model trained on `train.data` and predicting on `valid.data`.
The calculation of `threshold` is based on `train.data`.

12. What is the ROC curve? What does the Area Under the Curve (AUC)
tell us about classifier performance? When is it appropriate to use 
an ROC curve?

According to [Wikipedia](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
the ROC curve plots *sensitivity* (fraction of positives in the ground truth
correctly identified by the classifier) against the *false positive rate* 
(fraction of negatives in the ground truth identified as false positives by the classifier).

13. Generate and plot AUC curves for each of the 10 folds in our
sample.

```R
roc.curves <- lapply(1:10, function(i) {
  train.data <- data[data$fold != i,]
  valid.data <- data[data$fold == i,]
  model <- glm(Survived~Fare+Embarked+Pclass, data=train.data,
               family=binomial(link = "logit"))
  predicted.probs <- plogis(predict.glm(model, newdata=valid.data))
  thresholds <- unique(predicted.probs)
  thresholds <- thresholds[order(thresholds)]
  roc.measures <- lapply(thresholds, function(t) {
    ground.truth <- valid.data$Survived
    predicted.labels <- as.numeric(predicted.probs >= t)
    tp <- sum((predicted.labels == 1) * (ground.truth == 1))
    fn <- sum((predicted.labels == 0) * (ground.truth == 1))
    tn <- sum((predicted.labels == 0) * (ground.truth == 0))
    fp <- sum((predicted.labels == 1) * (ground.truth == 0))
    return(data.frame(fold=i, threshold=t, tp=tp, fp=fp, tn=tn, fn=fn))
  })
  roc.measures <- do.call("rbind", roc.measures)
  return(roc.measures)  
})
roc.curves <- do.call("rbind", roc.curves)
library('ggplot2')
p <- ggplot(data=roc.curves)
p <- p + geom_line(aes(y=tp / (tp + fn), x = fp / (fp + tn), group=fold, color=fold))
print(p)
```
