---
title: 'Lecture 5'
description: 'Classification, simple and multiple logistic regression, softmax, odds, gradient descent, predicting'
pubDate: 'Sep 17 2023'
heroImage: '/extra_log.png'
---

**Readings**: Chapter 4, 4.1, 4.2 & 4.3

#### Table of contents
- [Classification](#classification)
    - [Why Not Linear Regression?](#why-not-linear-regression)
- [Logistic Regression](#logistic-regression)
  - [Simple Logistic Regression Equation](#simple-logistic-regression-equation)
  - [Multiple Logistic Regression](#multiple-logistic-regression)
  - [Multinomial Logistic Regression Softmax](#multinomial-logistic-regression-softmax)
    - [Softmax](#softmax)
  - [Odds](#odds)
    - [Log-Odds (Logit)](#log-odds-logit)
  - [Estimating the Regression Coefficients](#estimating-the-regression-coefficients)
    - [Loss Function](#loss-function)
    - [Gradient Descent of the Loss Function](#gradient-descent-of-the-loss-function)
    - [Standard Errors, z-statistic and Hypothesis-Testing](#standard-errors-z-statistic-and-hypothesis-testing)
  - [Predicting](#predicting)


# Classification

**Linear Regression**assumes the response variable $Y$ to be quantitative, but in many situations the response variable is instead qualitative. This is known as **classification.**


Widely used classifiers to predict qualitative responses: `logistic regression`, `linear disrminant analysis`, `quadratic discriminant analysis`, `naive bayes`, `k-nearest neighbours`.

Some more computer-intensive methods: `trees`, `random forests`, `boosting`, `support vector machines`.

### Why Not Linear Regression?

If the response variable’s values did take on a natural ordering, such as mild, moderate, and severe, and we felt the gap between mild and moderate was similar to the gap between moderate and severe, then a 1, 2, 3 coding would be reasonable.

Unfortunately, in general there is no natural way to convert a qualitative response variable with more than two levels into a quantitative response that is ready for linear regression.

However, for binary (two level) qualitative response, the situation is better. But if we use linear regression for this, some of our estimates might be outside the $[0,1]$ interval, and they will be hard to interpret as probabilities. There are two reasons not to perform classification using regression methods:

- A regression method cannot accommodate a qualitative response with more than two classes.
- A regression method will not provide meaningful estimates of $Pr(Y|X)$, even with just two classes.

# Logistic Regression

Logistic Regression is well-suited for the case of a **binary qualitative response.**

Logistic Regression is a natural extension of linear regression to binary classification problems. As part of the family of linear models, logistic regression assumes a linear relationship between its input features. 

These classes (labels/categories) are modelled as the response $Y$, which is a discrete random variable with a finite range $R_Y=\{0,1\}$. 

<aside>
📌 ************Matrix form:************

For a data set of $n$  observations in $p$ features is collected in a feature matrix $\bold{X} \in \R^{n\times p}$.

</aside>

The output of the decision function can thus be interpreted as the probability of class 1, $P(Y=1|X)$. Then the probability of the other class (class 0) follows the ************************************************law of total probability************************************************ thus $P(Y=0|X) = 1-P(Y=1|X)$.

Given these two probabilities we can choose a threshold that determined which class we should assign the output to:

$$
\begin{cases} 1, \ \ \text{if } \hat f (\bold{X}) \ge 0.5 \\ 0, \ \ \text{otherwise}\end{cases}
$$

The threshold default is 0.5, so that we classify the label with the larger (conditional) posterior probability.

## Simple Logistic Regression Equation

Logistic regression takes the form:

$$
S(x) = \frac{1}{1+e^{-x}} = \frac{e^x}{e^x+1}
$$

However, since this has no parameters, we cant train the function. To add parameters we can compose the original linear model with the sigmoid function:

$$
\text{Simple Logistic Regression = }\frac{1}{1+e^{-\beta_0-\beta_1x}} \frac{}{}
$$

Where:

$$
\text{Logistic Model= }p(X) = \frac{e^{\beta_0+\beta_1X}}{1+e^{\beta_0+\beta_1X}}
$$

The Logistic Function will always produce an S-shaped curve.

## Multiple Logistic Regression

This is the problem of predicting a ********************binary response using multiple predictors.******************** Note that it still predict a binary response, but we have multiple features.

$$
p(X) = \frac{e^{\beta_0+\beta_1X_1+...+\beta_pX_p}}{1+e^{\beta_0+\beta_1X_1+...+\beta_pX_p}}
$$

Again, we use maximum likelihood to estimate the parameters.

********************An example******************** that illustrates the importance in some situations where it can differ a lot if you only use 1 predictor when there is several predictors playing a role for the response variable $Y$. **********************Confounding********************** see p. 139 in ISLwR.

## Multinomial Logistic Regression Softmax

We sometimes wish to classify more than two classes. However, the logistic regression approach we have seen up until now, only works for a binary response. 

It is possible to extend it, however, for $K>2$ classes. This is called ************************multinomial logistic regression.************************ To do this, we select a single class to serve as the ****************baseline****************.

But, the decision to treat the $K$th class as the baseline is unimportant, the model output will remain the same (only the coefficients differ). Though, the interpretation of the coefficients in a multinomial logistic regression is tied to the choice of the baseline.

### Softmax

In the softmax coding, rather than selecting a baseline class, we treat all $K$ classes symmetrically, and assume that for $K=1,...,K$:

$$
P(Y=k|X=x)=\frac{e^{\beta_{k0}+\beta_{k1}x_1+...+\beta_{kp}x_p}}{\sum_{l=1}^K e^{\beta_{l0}+\beta_{l1}x_1+...+\beta_{lp}x_p}}
$$

Rather than estimating coefficients for $K-1$ classes, we do for all $K$ classes.

The *multinomial (softmax) logistic regression* approach proposes a change in the decision function to not output a single probability, but a vector of probabilities.

## Odds

After some manipulation of the logistic model:

$$
\frac{p(X)}{1-p(X)} = e^{\beta_0+\beta_1X}
$$

This is called the ********odds.******** It can take on any value in the range $R=[0,\infty[$ . Values close to 0 indicate a low probability of class 1, and close to $\infty$ is a high probability of class 1 (the positive class).

****************Example:****************

1 in 5 people with an odds og $\frac{1}{4}$ will default (the default example in the book) since $p(X)=0.2$ implies an odds of $\frac{0.2}{1-0.2}=\frac{1}{4}$. Likewise, on average 9 out of every 10 people with an odds of 9 will defualt, since $p(X)=0.9$ implies an odds of $\frac{0.9}{1-0.9} = 9$. Odds are often used in betting strategies. 

### Log-Odds (Logit)

The log-odds takes the form:

$$
\log \left( \frac{p(X)}{1-p(X)}\right) = \beta_0+\beta_1X
$$

The logistic regression has a Logit that is linear in $X$.

By contrast, in a logistic regression model, increasing $X$ by 1 unit changes the log odds by $\beta_1$.

## Estimating the Regression Coefficients

The coefficients $\beta_0$ and $\beta_1$ are unknown, and must be estimated based on the available training data. In **********************************linear regression********************************** we use least squares, and we could use a non-linear least squares to fit a logistic regression. However, the method of **************maximum likelihood************** is preferred, since it has better statistical properties. 

******************Reasons why the MSE approach wont work well for classification problems:******************

1. The resulting cost-function $*J(θ)*$ is a **non-convex** function (since we squared a fraction with exponential functions), of which it is not easy to find a global minimum (ie. through GD)
2. We don't want to calculate distances but probabilities (thus scaling and penalises errors might be wrong)

### Loss Function

In the regular linear regression setting, we used the model of RSS (residual sum of squared errors) or the MSE (mean squared error$, \frac{RSS}{n})$ as a cost-function, in order to find the best-fitting model parameters. In the classification setting, this approach doesn't work as well for the following reasons:

1. The resulting cost-function *J*(*θ*) is a **non-convex** function (since we squared a fraction with exponential functions), of which it is not easy to find a global minimum (ie. through GD)
2. We don't want to calculate distances but probabilities (thus scaling and penalises errors might be wrong)

We would rather have a way of measuring the difference between the outputted probability and the actual class membership. The loss function that does this is called **cross-entropy** (also: *negative log-likelihood function*) and it is one of the most widely used loss functions for classification problems.

**************************************************Step-by-Step Differentiation of the loss function:**************************************************

$$
L_{\theta} = \prod_i^{n} \begin{cases} \hat y_i \ \ \ \ \ \ \ \ \ \text{, if } y_i = 1 \\ 1-\hat y \ \ \ \text{, if } y_i = 0 \end{cases}
$$

We can use this trick: $a^0=1, a^1=a$, to write this into a single line:

$$
L_{\theta} (y, \hat y) = \prod_i^n \hat y_i^{y_i} \cdot (1- \hat y_i)^{1-y_i} 
$$

Taking the log of this turns the product into a sum and the power terms into products. This speeds up our computation. This is the whole idea of Log-Likelihood.

$$
LL(y,\hat y) = \sum_{i=1}^n [y_i \log(\hat y) + (1-y_i) \log(1-\hat y_i)]
$$

The larger the value of this function above, the closer we come to the correct label. Since it is easier to minimise a function instead of maximising, we rewrite:

$$
NLL(y,\hat y) = -\sum_{i=1}^n [y_i \log(\hat y) + (1-y_i) \log(1-\hat y_i)]
$$

<aside>
📌 There **does not exist a closed-form solution** for the **cross-entropy** cost function. However, since it is a convex function, an optimisation algorithm like **g*radient descent*** is a good choice for optimising our parameters.

</aside>

### Gradient Descent of the Loss Function

The derived loss function + the gradient step:

<div style="text-align: center;">
    <img src="/Untitled_logistic.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

<div style="text-align: center;">
    <img src="/Screenshot_2022-11-06_at_20.04.17.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

<div style="text-align: center;">
    <img src="/Screenshot_2022-11-06_at_20.05.00.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>


<div style="text-align: center;">
    <img src="/Screenshot_2022-11-06_at_20.06.33.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>


<div style="text-align: center;">
    <img src="/Screenshot_2022-11-06_at_20.04.37.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

### Standard Errors, z-statistic and Hypothesis-Testing

There are some similarities from logistic regression to linear regression. We can measure the ****************accuracy**************** of the coefficient estimates by computing their standard erros.

For instance, the $z$-statistic associated with $\beta_0$ is equal to $\frac{\hat \beta_1}{SE(\hat \beta_1)}$. So a large absolute value of the $z$-statistic indicates evidence against the null hypothesis $H_0:\beta_1=0$. This null hypothesis indicates that $\beta_1$ does not depend on the response variable. Then we check the $p$-**********value********** to see if it statistically significant. If the $p$-value is small (below 0.1 usually) the response variable $Y$does indeed depend $\beta_1$.

<div style="text-align: center;">
    <img src="/Screenshot_2022-11-05_at_13.26.36.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

## Predicting

Example from the book with the default case:

If your have a balance of $\$1,000$, your default probability is:

$$
\hat p(X) = \frac{e^{\beta_0+\beta_1X}}{1+e^{\beta_0+\beta_1X}} = \frac{e^{-10.65+0.005 \times 1}}{1+ e^{-10.65+0.005 \times 1}} = 0.00576
$$

Which is below 1%. In contrast to $\$ 2,000$ we have a higher probability 0.586.
