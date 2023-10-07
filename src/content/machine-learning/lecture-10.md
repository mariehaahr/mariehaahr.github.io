---
title: 'Lecture 10'
description: 'Design and Evaluation of Machine Learning Experiments'
pubDate: 'Oct 05 2023'
heroImage: '/lec10_ml.png'
---

**Readings**: p 88-102 Hands-on, 148-152 in ISLwR

#### Table of contents
- [Before You Start Building Models](#before-you-start-building-models)
	- [Preparation Phase](#preparation-phase)
	- [Model Building](#model-building)
		- [Train, Test and Validation Splits](#train-test-and-validation-splits)
		- [Evaluating Hyper Parameters With Validation Split](#evaluating-hyper-parameters-with-validation-split)
		- [Overfitting](#overfitting)
- [Training a Binary Classifier](#training-a-binary-classifier)
		- [Stochastic Gradient Descent (SGD) Classifier](#stochastic-gradient-descent-sgd-classifier)
		- [Measuring Accuracy Using Cross Validation](#measuring-accuracy-using-cross-validation)
- [Performance Metrics, Loss Function/Matrix](#performance-metrics-loss-functionmatrix)
	- [Confusion Matrix](#confusion-matrix)
		- [The $F\_1$ score](#the-f_1-score)
		- [Micro:](#micro)
		- [Macro:](#macro)
	- [Inherent Tradeoff](#inherent-tradeoff)
		- [How do you decide which threshold to use?](#how-do-you-decide-which-threshold-to-use)
	- [Cross-validation](#cross-validation)
		- [**Leave-one-out cross-validation (LOOCV)**](#leave-one-out-cross-validation-loocv)
- [Evaluation](#evaluation)
	- [Regression Setting](#regression-setting)
	- [Classification Setting](#classification-setting)
		- [Accuracy](#accuracy)
		- [Confusion Matrix](#confusion-matrix-1)
		- [Precision](#precision)
		- [Recall](#recall)
		- [F1-Score](#f1-score)
	- [The ROC curve](#the-roc-curve)
		- [AUC](#auc)
	- [The ROC curve](#the-roc-curve-1)

# Before You Start Building Models

The whole design process of building and deploying a machine learning model can be decomposed into some steps:

1. **Preparation phase**: Data Collection, EDA, Data Cleaning, Dimensionality Reduction
2. **Model Building phase**: Feature Engineering (Extracting Features), Feature scaling, Model Architecture, Hyper-parameter Tuning, Data splits, Grid Search, Training
3. **Evaluation**: Metrics, Loss, Accuracy, Confusion Matrix, Recall, Precision,

## Preparation Phase

The preparation phase is pretty much the same no matter the problem you are given. This includes Data collection, Data cleaning (missing values, wrong values etc), Exploratory Data Analysis. This includes plotting the data, plotting the class distributions. Just in general, see if you can figure out if there are some clear patterns already there. Dimensionality reduction if that is necessary. 

## Model Building

Now the model building phase depends on what problem you have. If this supervised or unsupervised, regression or classification etc etc. How much data do you have available, how complex is your problem. 

****************************************We often ask the question: Are we interested in making accurate predictions (Decision) or are we more interested in finding some patterns and relationships between features (Inference)?**************************************** 

****************************************There is a trade-off between model interpretability and model performance.**************************************** 

Now when this decision has been made, and you have found a model that fits you problem the question is, how do i find the perfect set of hyper parameters $H(.)$, that yield the optimal performance / generalisation. Because we don’t just want the best performance for the training data, we want the model to be good in general.

So now the question is how do we choose these hyper parameters, so our model don’t overfit the training data?

### Train, Test and Validation Splits

Before training and evaluating:

When evaluating a machine learning model a good place to start is with a ‘simple’ `train` and `test` split. This means that you split your dataset in two usually 80% training and 20% test. 

************************************IMPORTANT (Classification):************************************ If you for classification tasks have uneven class distribution, it is especially important to ‘stratify’ your splits. Which essentially means to make sure that your two splits have equal class distribution. this makes sure that your splits are representative of full data set.

******************CRUCIAL:****************** With this split data, you use the training data only for training/fitting your model to, you DO NOT touch the test data until it is time to evaluate your models performance.

For finding the best ********************************hyper parameters********************************, to get an optimal performance on the unseen data, we need another split. The ********************************Validation Split********************************. 

This basically amounts to further splitting the training data into a train and validation split, again with the same ‘requirements’ as above (stratification). 

### Evaluating Hyper Parameters With Validation Split

You would then use this validation split for evaluating different hyper parameters either by **trial and error** or more systematic optimisation strategies e.g., by using grid search - where you specify a range/list of hyper parameters to try and test all possible combinations of these.

**This depends on whether you have a couple of hyper parameters or a lot (like NN and DT).**

Then you would save the hyper parameters of the best performing model and use those to evaluate on the test split and report those results.

### Overfitting

When you overfit the model because you did not apply any regularisation methods, the model is less general.


<div style="text-align: center;">
    <img src="/Screenshot_2023-01-10_at_14.57.14.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

This could happen to a model like a neural network, where you did not implement early stopping.

# Training a Binary Classifier

We will learn how to train binary classifiers, choose the appropriate metric for the task, evaluate your classifiers using cross-validation, select the precision/recall trade-off that fits your needs, and use the ROC curbes and ROC AUC scores to compare various models.

### Stochastic Gradient Descent (SGD) Classifier

This classifier has the advantage of being capable of handling very large datasets efficiently. This is in part because SGD deals with training instances independently, one at a time (which also makes SGD well suited for online learning)

### Measuring Accuracy Using Cross Validation

<div style="text-align: center;">
    <img src="/Screenshot_2022-10-02_at_12.00.54.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

 Here, at each iteration the code creates a clone of the classifier, trains that clone on the training folds, and makes predictions on the test fold. Then it counts the number of the correct predictions and outputs the ratio of correct predictions.

**Accuracy is generally not the preferred performance measure for classifiers,** especially when you are dealing with skewed datasets. (when some classes are more frequent than others).

# Performance Metrics, Loss Function/Matrix

The important things to think about are what each metric actually measures and whether some types of errors are more problematic than others (the latter is naturally extremely application dependent!). When we specify a loss function/matrix, we exactly consider carefully how to penalise different types of errors. Classifiers can then be compared by the expected loss – this is what we have already often computed as the test error. For Bayes classifiers that uses 0-1 loss, the test error is simply the misclassification error (i.e. 1-accuracy). So long as you are aware of whether the metric captures what you need, there is absolutely nothing wrong with using accuracy (or its complement, the error rate) as way of summarising or evaluating the performance of a classifier. You may also wish to use different metrics for building/training classifiers, selecting between them, and reporting results. In particular, the reporting will typically use a few easily comprehensible summaries of the classifier.

## Confusion Matrix

```python
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
```

`cross_val_predict()` performs K-fold cross-validation, but instead of returning the evaluation scores, it returns the predictions made on each test fold. 

This mean you get a **clean prediction** (Meaning that the prediction is made by a model that never saw the data during training).

Now make the confusion matrix:

```python
>>> from sklearn.metrics import confusion_matrix
>>> confusion_matrix(y_train_5, y_train_pred)

array([[53057, 1522],
			[ 1325, 4096]])
```

Each row in a confusion matrix represents an actual class, while each column represent the predicted class. 

A perfect classifier would only have true positives and true negatives. 

<div style="text-align: center;">
    <img src="/Untitled3_10.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>


<div style="text-align: center;">
    <img src="/Untitled1_10.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

### The $F_1$ score

The $F_1$ score is the harmonic mean of precision and recall.

The harmonic mean gives more weight to low values, and this results in the classifier will only get a high $F_1$ score if both the precision and recall is high.

$$
F_1 = \frac{2}{\frac{1}{precision}+\frac{1}{recall}} = \frac{precision \times recall}{precision+recall} = \frac{TP}{TP + \frac{FN + FP}{2}}
$$

```python
>>> from sklearn.metrics import f1_score
>>> f1_score(y_train_5, y_train_pred)

# 0.7420962043663375
```

### Micro:

Precision and Recall are per class, but sometimes we’d like to have one single number to characterise our performance.
Accuracy is a single number, but is problematic with unbalanced data.
Micro-averaged Precision, Recall and F-score: Add the counts of all classes, then compute Precision, Recall and F-score.

If each example has only one label, this is the same as accuracy

### Macro:

Macro-averaged Precision, Recall and F-score: Compute P, R and F for each class, then take the arithmetic mean.

Macro-averaging is sensitive to outlier classes! (can enforce balance, but also cause problems)

**Macro-averaged Recall is least sensitive to imbalance.**

## Inherent Tradeoff


<div style="text-align: center;">
    <img src="/Screenshot_2022-09-30_at_11.39.52.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

The better you are at detecting the blue class (lowering the threshold so you will almost always classify to blue) you will have a lower probability of detecting the red class. You can plot precision against recall for all classifiers (with different thresholds) and see which one is best.

### How do you decide which threshold to use?

First, we use the `cross_val_predict()` function to get the scores of all instances in the training set, and in code we specify that we want the ‘decision_function’, so we get the decision scores.

```python
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,method="decision_function")
```

With these scores we use the `precision_recall_curve()` function to compute precision and recall for all possible thresholds:

```python
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
```

And use Matplotlib to plot it:

```python
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
[...] # highlight the threshold and add the legend, axis label, and grid
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
```

You will probably want to select a precision/recall trade-off just before the drop in the curve:

<div style="text-align: center;">
    <img src="/Screenshot_2022-10-02_at_12.22.38.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

You can search for the lowest threshold that gives you at leat 90% precision:

```python
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)] # ~7816
```

## Cross-validation

It is not always the case, that we have enough data to split it into 3, and have an optimal amount of data. So a better approach is cross-validation.

The most common/basic approach to cross validation is $k$-fold cross-validation.

With cross-validation we ‘drop’ the validation split and instead only have a train and test split. However we (with $k$-fold cross-validation) split the training data into $k$ folds/splits. The model is then trained on $k-1$ folds with the last fold being used for validation. In the end we then average whatever performance measure we chose to use for evaluation (e.g., accuracy).

Cross-validation could be used in combination with **grid search** where you for each possible combination of hyper parameters ‘do’ cross-validation and then save the combination of hyper parameters that yielded the best average performance measure. 

Then finally use those hyper parameters to report your performance on the test split made in the beginning.

### **Leave-one-out cross-validation (LOOCV)**

LOOCV is the an exhaustive holdout splitting approach that k-fold enhances. It has one additional step of building k models tested with each example. This approach is quite expensive and requires each holdout example to be tested using a model.

# Evaluation

Now the last part after designing your model, comes the evaluation. These two last parts are steps you can redo, since fine-tuning a model implies of both training and evaluation. 

## Regression Setting

If you are in the regression setting, we usually use the mean squared error as an evaluation of “how bad is the model”. 

$$
\text{MSE}=\frac{\sum_i^n\big(y_i-\hat{y_i}\big)^2}{n}
$$

## Classification Setting

### Accuracy

$$
\text{Accuracy }=\frac{\text{True predictions}}{\text{All predictions}}
$$

Now in the classification setting there are numerous ways of evaluating your model. The accuracy of a model is probably the most used metric we see out there, but it can be a misleading metric. If the class distribution is very uneven, and we have f.x 1 out of 10000 patients who has cancer, then a model that predicts “no cancer” all the time, would have an accuracy of almost 100%.

### Confusion Matrix

Confusion matrix on the other hand, are a great metric to use. 

<div style="text-align: center;">
    <img src="/Untitled2_10.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

### Precision

*The accuracy of the positive predictions.*

**So, with a precision of 75%, would mean that we for example got 3 TP out of 4. So all of Positive predictions we actually made, 3 of them were correct, out of 4.**

The number of correctly predicted positive samples within a specific class over total number of predictions to that class.

So a model that produces no false positives has a precision of 1.0. Though if we had a classifier that strictly classified to that class all of the time, a precision of 100% is easily obtained.

$$
\text{Precision}=\frac{TP}{TP+FP}
$$

### Recall

The number of correctly classified samples within a specific class over the total number of samples in that class.

******************************************************************************************************************************************While with recall, we look at all of the positives, the actual positives. Lets say a Recall of 60% is because we detected 3 out of 5 of the thing we actually wanted to predict. So 5 labels are the true ones, but we only predicted 3 of them, while the 2 remaining was predicted to something else.******************************************************************************************************************************************

*This is the ratio of positive instances that are correctly detected by the classifier*

So a model that produces no false negatives has a recall of 1.0. 

$$
\text{Recall}=\frac{TP}{TP+FN}
$$

<div style="text-align: center;">
    <img src="/Screenshot_2023-01-22_at_17.17.25.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

### F1-Score

The F1 score is simply the harmonic mean of Precision and Recall. We can also consider this in terms of true positives and negatives. Now we take the harmonic mean instead of a normal mean, because the harmonic mean ensures that both Precision and Recall has to be high, in order for the F1 score to be high.

$$
\text{F1}=\frac{Pr \cdot Re}{Pr + Re}
$$

## The ROC curve

The ROC curve plots the true positive rate (recall) against the false positive rate. It works for binary classification, but can be extended to multi-class classification.

An **ROC curve** (**receiver operating characteristic curve**) is a graph showing the performance of a classification model at all classification thresholds. This curve plots two parameters:

- True Positive Rate
- False Positive Rate

**True Positive Rate** (**TPR**) is a synonym for recall and is therefore defined as follows:

$$
TPR=\frac{TP}{TP+FN}
$$

**False Positive Rate** (**FPR**) is defined as follows:

$$
FPR=\frac{FP}{FP+TN}
$$

**As we have seen above, varying the classifier threshold changes its true positive and false positive rate. These are also called the sensitivity and one minus the specificity of our classifier.** 

<aside>
📌 **Hence, the ROC curve plots sensitivity (recall) versus 1 – specificity.**

</aside>

An ROC curve plots TPR vs. FPR at different classification thresholds. Lowering the classification threshold classifies more items as positive, thus increasing both False Positives and True Positives. The following figure shows a typical ROC curve.

In a multi-class model, we can plot the N number of AUC ROC Curves for N number classes using the One vs ALL methodology. So for example, If you have **three** classes named **X, Y,** and **Z** , you will have one ROC for X classified against Y and Z, another ROC for Y classified against X and Z, and the third one of Z classified against Y and X

### AUC

Higher the AUC, the better the model is at predicting 0 classes as 0 and 1 classes as 1

To compare classifiers you can measure the AUC (area under the curve). A perfect classifier will have a ROC AUC equal to 1, whereas purely random classifier (flipping a coin) will have a ROC AUC on 0.5.

```python
>>> from sklearn.metrics import roc_auc_score
>>> roc_auc_score(y_train_5, y_scores)
# 0.9611778893101814
```

**Multi-class Classification:** Logistic Regression classifiers, Random Forest classifiers and naive Bayes Classifiers are capable of handling multiple classes.

## The ROC curve

The ROC curve plots the true positive rate (recall) against the false positive rate.

```python
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

# plot FPR against TPR

def plot_roc_curve(fpr, tpr, label=None):
	plt.plot(fpr, tpr, linewidth=2, label=label)
	plt.plot([0, 1], [0, 1], 'k--') 
	# Dashed diagonal[...] # Add axis labels and grid
	plot_roc_curve(fpr, tpr)
	plt.show()
```
