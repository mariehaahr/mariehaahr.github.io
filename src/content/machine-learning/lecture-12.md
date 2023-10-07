---
title: 'Lecture 12'
description: 'Ensemble Methods'
pubDate: 'Oct 07 2023'
heroImage: '/lec12_ml.png'
---


**Readings:** ISLwR, pages 209-212  (Chapter 5.2), Hands on: Chapter 7 + 13: ISLwR p 345-348

#### Table of contents
- [7 Ensemble Learning and Random Forests](#7-ensemble-learning-and-random-forests)
	- [Voting Classifiers](#voting-classifiers)
		- [Diversity in Ensemble](#diversity-in-ensemble)
	- [Bagging and Pasting](#bagging-and-pasting)
		- [**Bagging and pasting differences**](#bagging-and-pasting-differences)
		- [Bagging and Pasting in Scikit-Learn](#bagging-and-pasting-in-scikit-learn)
		- [Out-of-Bag Evaluation](#out-of-bag-evaluation)
	- [Random Patches and Random Subspaces](#random-patches-and-random-subspaces)
	- [Random Forests](#random-forests)
		- [](#)
		- [Feature Importance](#feature-importance)
	- [Boosting](#boosting)
		- [AdaBoost](#adaboost)
		- [Gradient Boosting](#gradient-boosting)
	- [Stacking](#stacking)
	- [Notes from Lecture 12](#notes-from-lecture-12)
	- [Impurity Function](#impurity-function)
		- [Gini Index](#gini-index)
	- [Missing features](#missing-features)
		- [When to stop splitting a leaf node?](#when-to-stop-splitting-a-leaf-node)
	- [Regularisation/Shrinkage](#regularisationshrinkage)


# 7 Ensemble Learning and Random Forests

If you ask a hard question to thousands of random people and aggregate the answers (collect), you will find out that the aggregated answer is far better than 1 experts answer. 

Similarly, if you aggregate the predictions of a group of predictors (such as classifiers or regressors) you will often get better predictions than with the best individual predictor.

**A group of predictors is called an ensemble.**

An example of an ensemble methods is training a group of Decision Trees. This is called a *Random Forest.*

<aside>
📌 Normally you would use Ensemble methods near the end of a project, once you have already built a few good predictors, to combine them into an even better predictor

</aside>

## Voting Classifiers

A simple way of creating a better classifier is to aggregate the predictions (collect the predictions) and then predict to the class that gets the most votes. This majority-vote classifier is called a *hard voting* classifier.

<aside>
📌 Ensemble methods work best when the predictors are as independent from one another as possible. (when they are different, they hopefully make different types of errors) One way to get diverse classifiers is to train them using very different algorithms. This increases the chance that they will make very different types of errors, improving the ensemble’s accuracy.

</aside>

### Diversity in Ensemble

Since we want the predictors to be as independent from each other as possible, there are two different ways of introducing diversity:

- Diversity in the predictor: Here the classifiers are different and they train on the same data.
- Diversity in the training data. Here the classifiers are the same, but they train on different subsets of the training data.

The following code creates and trains a voting classifier with three diverse classifiers:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

voting_clf = VotingClassifier(
	estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)]
	,voting='hard')voting_clf.fit(X_train, y_train)

# here are each of the classifiers' accuracy:
LogisticRegression 0.864
RandomForestClassifier 0.896
SVC 0.888
VotingClassifier 0.904
```

Here it is clear to see that the voting classifier is more accurate.

If all classifiers are able to estimate class probabilities then you can predict the class with the highest class probability, averaged over all the individual classifiers. this is called soft voting. **It often achieves a higher performance than hard voting,** because it gives more weight to highly confident votes.

```python
# to do soft voting you replace the voting="hard" to:
voting="soft"

# to ensure that all classifiers can estimate class probabilities
```

## Bagging and Pasting

Another approach is to use the same training algorithm for every predictor and train them on different random subsets of the training set. When sampling is performed with replacement, this method is called *bagging* (short for bootstrap aggregating)*.* When sampling is performed without replacement it is called *pasting.*

### **Bagging and pasting differences**

In other words, both bagging and pasting allow training instances to be sampled several times across multiple predictors, but only bagging allows training instances to be sampled several times for the same predictor.

<aside>
📌 Each individual predictor has a higher bias than if it were trained on the original training set, but aggregation reduces both bias and variance.

</aside>

### Bagging and Pasting in Scikit-Learn

This following code trains an ensemble of 500 decision tree classifiers, each trained on 100 training instances randomly sampled from the training set with replacement (bagging, bootstrapping, if you want pasting, set `bootstrap=False`).

`n_jobs` is a parameter that tells Scikit Learn the number of CPU cores ti use for training and predictions. 

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
	DecisionTreeClassifier(), n_estimators=500,
	max_samples=100, bootstrap=True, n_jobs=-1)

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
```

<aside>
📌 The `BaggingClassifier` automatically performs soft voting, if the base classifier can estimate class probabilities.

</aside>

As you can see, the ensemble’s predictions will likely generalise much better than the single Decision Tree’s predictions: the ensemble has a comparable bias but a smaller variance (it makes roughly the same number of error on the training set, but the decision boundary is less irregular)

So, bagging ends up with a slightly higher bias than pasting, BUT the extra diversity also means that the predictors end up being less correlated, so the ensemble’s variances is reduced. **Overall, bagging often results in better models.**

By creating variation in the training data, we get variation among the trained models. 
So if you have a classifier with high variance, you can use bagging. 

Since the training subsets are drawn from the same given data set, they are not truly independent, but the conclusions may still weakly hold.

### Out-of-Bag Evaluation

With bagging, some instances may be sampled several times for any given predictor, while others may not be sampled at all.

This means that only about 63% of the training instances are sampled on average for each predictor. The remaining 37% of the training instances that are not sampled are called out-of-bag (oob) instances.

```python
# in Scikit Learn you can set the ;
oob_score =True
# when creating a ;
BaggingClassifier
# to request an automatic oob evaluation after training
```

## Random Patches and Random Subspaces

In the `BaggingClassifier` the sampling is controlled by 2 hyper-parameters: `max_features` and `bootstrap_features`.

Thus, each predictor will be trained on a random subset of the input features.

This is useful when dealing with high-dimensional inputs (such as images).

<aside>
📌 Sampling both training instances and features is called the *Random Patches method.*

</aside>

## Random Forests

As we have discussed, a Random Forest9 is an ensemble of Decision Trees, generally trained via the bagging method (or sometimes pasting), typically with max_samples set to the size of the training set.

The Random Forest algorithm introduces extra randomness when growing trees:

Instead of searching for the very best feature when splitting a node, it searches for the best feature among a random subset of features.

**The algorithm results in greater tree diversity,** which again trades a higher bias for a lower variance, generally yielding an overall better model.

### 

### Feature Importance

Scikit-Learn measures a feature’s importance by looking at how much the tree nodes that use that feature reduce impurity on average.

You can access the results using the `feature_importances_` variable. 

```python
# example with the iris dataset

for name, score in zip(iris["feature_names"], rnd_clf-feature_importances_):
		print(name, score)

#sepal length (cm) 0.112492250999
#sepal width (cm) 0.0231192882825
#petal length (cm) 0.441030464364
#petal width (cm) 0.423357996355
```

## Boosting

Boosting (hypothesis boosting) refers to any Ensemble method that can combine several weak learners into a stronger learner. 

The general idea of most boosting methods is to train predictors sequentially, each trying to correct its predecessor (forekommer).

The two most popular ones are: **Adaptive Boosting (AdaBoost)** and **Gradient Boosting.**

### AdaBoost

When we need to correct its predecessor, we pay a bit more attention to the training instances that the predecessor under-fitted. This results in new predictors focusing more and more on the hard cases. This is the technique used by AdaBoost. 

First, the algorithm trains a base classifier, and uses it to make predictions. Then it increases the relative weight of misclassified training instances. Then it trains a second classifier, using the updated weights, makes predictions on the training set, updates the weights, and so on.

This sequential learning technique is quite similar to Gradient Descent (which tweaks a single predictor’s parameters to minimise a cost function), AdaBoost adds predictors to the ensemble, gradually making it better.

<aside>
📌 AdaBoost, this sequential learning technique, does not scale as well as bagging or pasting.

</aside>

See more detailed description on code and math on page 201 and forward.

If your AdaBoost ensemble is overfitting the training set, you can try reducing the number of estimators or more strongly regularising the base estimator.


<div style="text-align: center;">
    <img src="/Screenshot_2022-10-11_at_11.28.19.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

<div style="text-align: center;">
    <img src="/Screenshot_2022-10-11_at_11.39.55.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

After making each tree, the remaining errors (the mistakes of the current tree):

- influence the weights of the training points for the next tree (how important each point is)
- determine the weight of the tree in the final vote (amount of say of this tree)

<div style="text-align: center;">   
 	<img src="/Screenshot_2022-10-11_at_11.39.40.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

<div style="text-align: center;">
    <img src="/Screenshot_2022-10-11_at_11.40.06.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

### Gradient Boosting

Just like AdaBoost, Gradient Boosting works by sequentially adding predictors to an ensemble , each one correcting its predecessor. However, instead of tweaking the instance weights at every iteration like ADaBoost does, this method tries to fit the new predictor to the *residual errors* made by the previous predictor. 

A simpler way to train GBRT ensembles (Gradient Boosted Regression Trees) is to use Scikit-Learn’s `GradientBoostingRegressor` class.


<div style="text-align: center;">
    <img src="/Screenshot_2022-10-11_at_11.54.14.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

Sequential algorithm like AdaBoost
• Start with a single leaf (initial guess, e.g., the average of all labels)
• use the current ensemble to predict the training data and compute the residuals
• Fit a new tree to the residuals
• To avoid overfitting, we use a learning rate (often 0.1) for predictions
• Often tress (with max-depth=3) are used as weak learner here

```python
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
gbrt.fit(X, y)
```

The `learning_rate` hyper-parameter scales the contribution of each tree. If the learning rate is at a low value, you will need more trees in the ensemble to fit the training set, but the predictions will usually generalise better. This is called **shrinkage.**

In order to find the optimal number of trees, you can do the following code:

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_val, y_train, y_val = train_test_split(X, y)

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
gbrt.fit(X_train, y_train)errors = [mean_squared_error(y_val, y_pred)
		for y_pred in gbrt.staged_predict(X_val)]
		bst_n_estimators = np.argmin(errors) + 1

gbrt_best = GradientBoostingRegressor(max_depth=2,n_estimators=bst_n_estimators)
gbrt_best.fit(X_train, y_train)
```

It is possible to use Gradient Boosting with other cost functions. This is controlled by the `loss` hyper-parameter.

An optimised implementation of Gradient Boosting: `XGBoost`

## Stacking

Stacking (short for stacked generalisation) is baed on a simple idea:

*Instead of using trivial functions, such as hard voting, to aggregate the predictions of all predictors in an ensemble, why don’t we train a model to perform this aggregation?*

<div style="text-align: center;">
    <img src="/Screenshot_2022-10-06_at_18.48.16.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

In this figure, we have an ensemble that performs a regression task on a new instance. Each of the bottom three predictors predicts a different value, and then the final predictor (also called a blender or a meta learner) takes these predictions as inputs and makes the final predictions.

To train this “blender” we use a hold out set. Using this hold-out set, ensures that the predictions are “clean” (which means that the predictors never saw these instances during training).

**The procedure:**

The trick is to split the training set into three subsets: the first one is used to train the first layer, the second one is used to create the training set used to train the second layer (using predictions made by the predictors of the first layer), and the third one is used to create the training set to train the third layer (using predictions made by the predictors of the second layer).

Instead of simply using voting for aggregating the predictions, train a meta-learner (a learner of your choice, e.g., a decision tree) that gets the predictions of individual models as the input, and the expected labels for the data samples as the outputs

- Ensemble of M models makes vector of M inputs for the meta-learner
- We can do layers of stacking

## Notes from Lecture 12

## Impurity Function


<div style="text-align: center;">
    <img src="/Screenshot_2022-10-07_at_10.25.53.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

CORRECTION FOR ABOVE $E=1-4/6$

Gini impurity is less computational expensive, since it is harder to take the log

$\Phi$ 

Every possible split

Goodness/information gain


<div style="text-align: center;">
    <img src="/Screenshot_2022-10-07_at_10.39.21.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>


Classification error rate: choose the $1-max_{k} p_k$

### Gini Index


<div style="text-align: center;">
    <img src="/Screenshot_2022-10-07_at_10.28.07.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>


## Missing features

Imagine 50 features. There is a chance that some of these features are missing.  Suppose each feature has 5% chance of being missing independently. For a training data point with 50 feature, the probability of missing some features is as high as 92.3%! A test point to be classified may also have missing features.

How do we calculate that?

$$
1-(0.95)^{50} = 92.3 \%
$$

There are techniques to deal with this:

Surrogate splits??

### When to stop splitting a leaf node?

- When a node is fully pure
- No split reduces the impurity
- If the two point lie on top of each other, there are exactly equal
- A maximum number of nodes
- A maximum depth for a tree

## Regularisation/Shrinkage

Early stopping - stop growing tree too large

Pruning 

<div style="text-align: center;">
    <img src="/Screenshot_2022-10-07_at_11.19.52.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

Do this until removing a subtree actually affect the performance metrics. To ensure no bias in the test-set, you use cross-validation for this procedure.