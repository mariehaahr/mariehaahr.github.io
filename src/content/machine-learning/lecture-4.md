---
title: 'Lecture 4'
description: 'Cross-Validation, Shrinkage & Bias-Variance Trade-off'
pubDate: 'Sep 14 2022'
heroImage: '/blog-placeholder-3.jpg'
---


Readings:  ISLwR: 2.2.1-2, 5.1,  6.1-2  
Subject: Cross-Val, Ridge, Lasso, AIC, BIC, adj. $R^2$,

#### Table of contents
1. [Introduction](#introduction)
2. [Cross-Validation](#51-cross-validation)
   1. [Leave-One-Out Cross-Validation](#512-leave-one-out-cross-validation)
   2. [$k$-Fold Cross-Validation](#513--fold-cross-validation)
   3. [Bias-Variance Trade-Off for $k$-Fold Cross Validation](#514-bias-variance-trade-off-for--fold-cross-validation')
3. [Linear Model Selection and Regularisation](#6-linear-model-selection-and-regularisation)
4. [Subset Selection, AIC, BIC and Adjusted $R^2$](#61-subset-selection)
5. [Notes from the Lecture](#notes-from-lecture)



# Introduction <a name="introduction"></a>

When we wish to estimate the performance (or accuracy) of machine learning models, we can use different types of cross validation methods. We use it to protect against overfitting in particular in the cases where we have a limited amount of data. In cross-validation, you make a fixed number of folds (or partitions) of the data, run the analysis on each fold, and then average the overall error estimate.

## 5.1 Cross-Validation

**What is the difference between the *test error rate* and the *training error rate?***

*The test error is the average error that results from using a statistical learning method to predict the response on a new observation—that is, a measurement that was not used in training the method.* 

Given a data set, the use of a particular statistical learning method is warranted if it results in a **low test error**. **The test error** can be easily calculated if a designated test set is available. BUT THIS IS NOT THE CASE OFTEN.

In contrast the **training error** can be easily calculated by applying the statistical learning method to the observations used in its training. But the **training error rate** is often quite different from the **test error rate.**

Here we are interested in performing regression with a quantitative response. 

If we wish to use the MSE to guide the model building – selecting features, tuning hyperparameters – then we should preferably create a validation set for estimating the MSE and leave the test set for assessing performance. A normal split is 60-20-20 (but there is no formal number).

After estimating the MSE, then we train the model on both the validation and training data, and then testing that on the test data.

## 5.1.1 The Validation Set Approach

We want to estimate the test error associated with fitting a particular statistical learning method on a set of observations.

The Validation Set Approach involves randomly dividing the available set of observations into 2 parts.

- **Training set**
- **Validation set/hold-out set**

The model is fitted on the training set, and the fitted model is used to predict the responses for the observations in the validation set. 

Then we usually use MSE for getting an estimate of the test error rate (in the quantitative response).

The Validation Set Approach is easy and simple, but there are 2 potential drawbacks:

- The validation estimate of the test error rate can variate a lot, depending on precisely which observation are included in the training set and which observations are included validation set.
- The model is only fitted on a subset of the observations (the training set), and statistical methods tend to perform worse when trained on fewer observations. so the TVSA tend to overestimate the test error rate for the model fit on the entire data set.

## 5.1.2 Leave-One-Out Cross-Validation

LOOCV is close to the validation set approach. But it attempts to address the drawbacks.

Here we split into:

**Training set (all observations except 1)**

**Validation set (1 observation)**

Though the MSE is unbiased for the test error, the estimate variates a lot, because it is based on a single observation.

SO, we repeat the procedure by selecting another single observation for the validation set, and compute the MSE again.

Then repeating this $n$ times.

$$
CV_{(n)} = \frac{1}{n} \sum^n_{i=1}MSE_i
$$

**Advantages:**

It has less bias than TVSA. 

The LOOCV approach tends to not over-estimate the test error rate as much as the TVSA does.

**Disadvantages:**

It is expensive to implement, since the model has to be fitted n times. 

**OBS: This section below ONLY WORKS WITH LEAST SQUARES LINEAR POLYNOMIAL REGRESSION:**

Here we only have to do LOOCV once:

$$
CV_{(n)}= \frac{1}{n} \sum^n_{i=1}(\frac{y_i-\hat y_i}{1-h_i})^2
$$

 

## 5.1.3 $k$-Fold Cross-Validation

An alternative for LOOCV.

This approach involves randomly dividing the set of observations into $k$ groups (folds), of approx. equal size.

- First group/fold is validation set, and the model is fitted on the rest of the groups/folds, the remaining $k-1$ folds.
- Then the MSE is then computed on the validation set, the held-out fold.
- Repeat this, where you have a different group/fold of observations treated as a validation set.
- This results in $k$ estimates of the test error $MSE_1,MSE_2,...,MSE_k$. Then the $k$-Fold Cross-Validation estimate is computed by averaging these values:

$$
CV_{(k)} = \frac{1}{k}\sum^k_{i=1} MSE_i
$$

Usually $k=5$ or $k=10$. 

**IMPORTANT:**

Make sure to randomly allocate observations to folds, for instance shuffle your data before creating the folds. Note also that there is randomness in how observations are allocated to folds, so the cross-validation MSE curve varies between runs. 

**Advantages:**

Computational this is less expensive than LOOCV.

It often gives more accurate estimates of the test error rate than LOOCV. This has to do with the bias-variance trade-off.

![Screenshot 2022-09-09 at 11.51.38.png](/Screenshot_2022-09-09_at_11.51.38.png)

As you can see on the graphs, the striped line and the orange line (leave-1-out and k-folds) they are very similar, and therefore we don’t need to use Leave-1-Out (the very expensive one) and we can use a 10 fold, where we only need to fit the model 10 times instead of n times. 

## 5.1.4 Bias-Variance Trade-Off for $k$-Fold Cross Validation

The validation set approach can lead to overestimates of the test error rate, since the fitting is only with half of the observations. You would think that LOOCV is better, from the perspective of bias reduction, but bias is not the only source for concern, we need to consider variance too.

It turns out LOOCV has higher variance than $k$-fold CV.

This is because when we perform LOOCV, we are in effect averaging the outputs of n fitted models, each of which is trained on an almost identical set of observations; therefore, these outputs are highly (positively) correlated with each other. In contrast, when we perform k-fold CV with k < n, we are averaging the outputs of k fitted models that are somewhat less correlated with each other, since the overlap between the training sets in each model is smaller. Since the mean of many highly correlated quantities has higher variance than does the mean of many quantities that are not as highly correlated, the test error estimate resulting from LOOCV tends to have higher variance than does the test error estimate resulting from k-fold CV.

## 6 Linear Model Selection and Regularisation

In the regression setting, the standard linear model:

$$
Y= \beta_0 + \beta_1 X_1+\beta_2X_2 + ...+ \beta_p + X_p + \epsilon
$$

Typically we would fit this model using Least Squares.
but there are some ways that we can improve the simple linear model, for example replacing the least squares method with alternative fitting procedures. Some alternative fitting procedures can actually yield better **prediction accuracy** and **model interpretability.** 

### Prediction Accuracy

Provided that the true relationship between the response and the predictors is approximately linear, the least squares estimates will have low bias. If n >  p—that is, if n, the number of observations, is much larger than p, the number of variables—t*hen the least squares estimates tend to also have low variance,* and hence will perform well on test observations. However, if n is not much larger than p, then there can be a lot of variability in the least squares fit, resulting in overfitting and consequently poor predictions on future observations not used in model training. And if p > n, then there is no longer a unique least squares coefficient estimate: the variance is infinite so the method cannot be used at all. By constraining or shrinking the estimated coefficients, we can often substantially reduce the variance at the cost of a negligible increase in bias. This can lead to substantial improvements in the accuracy with which we can predict the response for observations not used in model training.

### Model Interpretability - too many variables

It is often the case that some or many of the variables used in a multiple regression model are in fact not associated with the response. Including such irrelevant variables leads to unnecessary complexity in the resulting model. By removing these variables—that is, by setting the corresponding coefficient estimates to zero—we can obtain a model that is more easily interpreted. Now least squares is extremely unlikely to yield any coefficient estimates that are exactly zero. In this chapter, we see some approaches for automatically performing feature selection or variable selection —that is, for excluding irrelevant variables from a multiple regression model.

With high number of features $p \approx n$ there is not enough information in the data to estimate the regression curve well.

With more parameters than features $p > n$ the MLE (maximum likelihood estimation) does not even exist. What can we do?

## 6.1 Subset Selection

Subset selection is when we have a method where we select only a subset of features (if we have too many). 

- Include only a subset of the features in the model.
- You can do model selection on the training set (hypothesis testing or with AIC) or can select based on validation set error.

All the methods below result in the creation of a set of models, each which contains a subset of the $p$ predictors. 

To apply these models we need a way to determine which of these models is best. 

<aside>
📌 The model containing all of the predictors (all $p$) will always have the smallest RSS and the largest $R^2$, since these are related to the **training error**.

</aside>

Instead, we wish to choose a model with a low test error. So RSS and $R^2$ are not suitable for selecting the best model in a set of models.

### Best Subset Selection

To perform ***********************best subset selection,*********************** we fit a separate least squares regression for each possible combination of the $p$ predictors. that is, we fit all $p$ models that contain exactly one predictor, all $\binom{p}{2} = \frac{p(p-1)}{2}$ models that contain exactly two predictors, and so forth. 

<aside>
📌 For computational reasons, best subset selections cannot be applied with very large $p$.

</aside>

### Best Subset Selection for other models than linear regression

The same ideas apply to other types of models, such as **************************************logistic regression**************************************. In case of ******************logistic regression,****************** instead of ordering models by Residual Sum of Squares (RSS), we instead use the ********deviance********, a measure that plays a role of RSS for a broader class of models. 

## 6.2 Stepwise Selection

When you have a large $p$, and best subset selection cant be used, we can use stepwise selection. 

<aside>
📌 The larger the search space, the higher the chance of finding models that look good on the training data, even though they might not have any predictive power on future data.

</aside>

When you have an enormous search space (when you have a lot of predictors) it can lead to ************************overfitting************************ and **************************high variance**************************. Stepwise methods explore a far more restricted set of models. 

### Forward Stepwise Selection

It is computationally efficient alternative to best subset selection. Here forward stepwise selection just begins with a model containing no predictors, and then adds predictors to the model, one-at-a-time, until all of the predictors are in the model. At each step, the variable that gives the greatest **********************additional improvement********************** to the fit, is added to the model. 

Though forward stepwise tends to do well in practice, it is not guaranteed to find the best possible model out of all $2^p$ models containing subsets of the $p$ predictors.

### Backwards Stepwise Selection

Again, like forward, backward stepwise selection is an efficient alternative to best subset selection. It begins with the full least squares model containing all $p$ predictors, and then iteratively removes the least useful predictor, one-at-a-time.

## $C_p$,AIC, BIC and Adjusted $R^2$

As mentioned before, the training error (MSE) will decrease as more variables are included in the model, but the test error may not. So, training set RSS and training set $R^2$ cannot be used for selection.

There are some different approaches for adjusting the training error.

### $C_p$

For a fitted least squares model containing $d$ predictors, the $C_p$ estimate of test MSE computed using the equation:

$$
C_p = \frac{1}{n} (RSS+2d \hat\sigma^2)
$$

where $\hat \sigma^2$ is an estimate of the variance of the error $\epsilon$ associated with each response measurement.

<aside>
📌 Essentially the $C_p$ statistic adds a penalty of $2d\hat \sigma^2$ to the training RSS in order to adjust for the fact that the training error tends to underestimate the test error. 

Clearly, the penalty increases as the number of predictors in the model increases; this is intended to adjust for the corresponding decrease in training RSS.

</aside>

### AIC

The AIC criterion (Akaike information criterion) is defined for a large class of models fit by maximum likelihood. In the case of the model (6.1) with Gaussian errors, maximum likelihood and least squares are the same thing. In this case AIC is given by:

$$
AIC = \frac{1}{n}(RSS+2d\hat \sigma^2)
$$

For least squares models $C_p$ and AIC are proportional to each other.

### BIC

BIC (Bayesian information criterion) is derived from a Bayesian point of view, but ends up looking similar to $C_p$ and AIC as well.

For the least squares model with $d$ predictors, the BIC is given by:

$$
BIC = \frac{1}{n} (RSS + \log(n)d\hat\sigma^2)
$$

Like Cp, the BIC will tend to take on a small value for a model with a low test error, and so generally we select the model that has the lowest BIC value.

Since $\log(n) > 2$ for any $n > 7,$ the BIC statistic generally places a heavier penalty on models with many variables, and hence results in the selection of smaller models than $C_p$.

### Adjusted $R^2$

An approach for selecting among a set of models that contain different numbers of variables. $R^2$ is normally defined as: $1-RSS/TSS$, where $TSS = \sum(y_i-\hat y_i)^2$ (total sum of squares for the response). 

Since RSS always decreases as more variables are added to the model, the $R^2$ always increases as more variables are added. For a least squares model with $d$ variables, the adjusted $R^2$ statistic is calculated as:

$$
\text{Adjusted } R^2=1-\frac{RSS/(n-d-1)}{TSS/(n-1)}
$$

Unlike $C_p$, AIC, and BIC, for which a small value indicates a model with a low test error, a large value of adjusted $R^2$ indicates a model with a small test error.

(see more on p. 234 and forward)

## Validation and Cross-Validation as Opposed to AIC, BIC, $C_p$ and $R^2$

As an alternative approach we can directly estimate the test error using the validation set and cross-validation methods. This procedure has an advantage relative to AIC, BIC, $C_p$ and Adjusted $R^2$, in that it provides a direct estimate of the test error, and makes fewer assumptions about the true underlying model. 

Nowadays with fast computers, the computations required to perform cross-validation are hardly ever an issue.

## 6.2 Shrinkage Methods

Ridge regression coefficients, and Lasso coefficients. Two types of distances to look at. Very scale dependent.

- Estimate parameters by minimising the log-likelihood added a penalty term (This corresponds to constraining the parameters).

## Ridge Regression

### The main idea

If a we have a linear regression that is trained on a few points and it does not fit well on the test data (meaning that the line is overfit). So the Least Squares fit results in a line that is over fit and has a high variance (meaning that the test data vary way more than expected, and they have large residuals).

Then the main idea behind ********************************Ridge Regression******************************** is to find a new line that doesn’t fit the training data as well. This means we introduce a small amount of ********Bias******** into how the new line is fit to the data, in the exchange of a lower ********************Variance.******************** 

So, by starting with a slightly worse fit, Ridge Regression can provide better ling term predictions.

****************************************Ridge Regression is just least squares, plus the ridge regression penalty.**************************************** 

When $\lambda = 0$, then the lasso regression line will be the same as the least squares line. As Lasso increases the $\lambda$ then the slope gets smaller and smaller. 

## The Lasso

### The main idea

Ridge and Lasso are quite similar, but there is a main difference. Instead of taking $\text{slope}^2$ we taken the absolute value of the slope: $|\text{slope}|$. 

Lambda $\lambda$ is a value between 0 and $\infty$, and is determined using ************************************cross validation.************************************ 

Like ******ridge regression, Lasso regression****** results in a line with a little bit of bias, but less variance than *************least squares*************!

<aside>
📌 The big difference between Ridge and Lasso is that Ridge can only shrink the slope asymptotically close to 0 while Lasso can shrink the slope all the way to 0.

</aside>

If we have a very big equation for a linear regression, for example:

$$
\text{Mouse weight = intercept + slope} \times \text{weight + diet difference } \times \text{high fat diet + astrological sign}  \times \text{sign}
$$

We have a lot of “silly” predictors. Here lasso would get rid of those silly predictors, and ridge would only make them asymptotically close to 0. 

<aside>
📌 Since Lasso regression can exclude useless variables from equations, it is a little better than Ridge, at reducing the Variance in the models that contain a lot of useless variables.

</aside>

# Notes from Lecture

## The Expected Squared Error

$$
E(Y_0-\hat f(x_0))^2 = Var(\hat f(x_0))+[Bias(\hat f(x_0))]^2 + Var(\epsilon)
$$

Keeping $x_0$  fixed.

Where $Var(\hat f(x_0))+[Bias(\hat f(x_0))]^2$     is the reducible error, and can be lowered by using an estimator $\hat f$ that has both low variance and low bias.

And  $Var(\epsilon)$    is the irreducible error, it is a bound on the accuracy of our prediction for $Y$ (The bound would typically be unknown).

### The Bias-Variance Trade-Off

Bias and variance for two learners:

- Fit a constant line through the training data
It has a high bias, were gonna guess with very little variance (since if we average over all the points, it may lay close to our constant line, even though it doesn’t fit the true $f$)
- Fit a curve that interpolates all points in the training data.
Still not a good fit, the variance is gonna be quite extreme, since the curve would move a lot if a point was changed (compared to the constant line). The bias though, is gonna be very low, since on average we would be around the true line.

These are 2 extremes, we want to find the perfect balance.

The higher flexibility, the higher order polynomial.

![Screenshot 2022-09-08 at 14.29.05.png](/Screenshot_2022-09-08_at_14.29.05.png)

The error of the model is constant, that is the striped line. The vertical dotted line is where the bias-variance trade off is perfect, this is the flexibility you should use. 

## The expected test MSE

We want MSE to be small for all values of $x_0$’s and not just for a specific value of the feature. 

(Se noter skrevet med rød)

We want to understand what happens across the range of $X$, but at the same time take into account the “frequency” with which the features $X$ appear in new data.

![Screenshot 2022-09-09 at 10.57.45.png](/Screenshot_2022-09-09_at_10.57.45.png)

![Screenshot 2022-09-09 at 11.00.46.png](/Screenshot_2022-09-09_at_11.00.46.png)

The MSE is a theoretical quantity that we can estimate from the data. It is better estimated from data points that were not used during training.

![Screenshot 2022-09-09 at 11.21.47.png](/Screenshot_2022-09-09_at_11.21.47.png)

![Screenshot 2022-09-09 at 11.22.33.png](/Screenshot_2022-09-09_at_11.22.33.png)

The grey line on the right, is the MSE of the training data, which is very low when the model is overfitted. The red line is then new data, and the MSE is very high when the model is overfitted (on the training data).

### Test MSE vs Training MSE

We should keep in mind a distinction between the following two
problems

- **Model selection:** estimating the prediction error of different
models with the purpose of choosing the best
one.
- **Model assessment:** having chosen a final model, estimating its
prediction error.