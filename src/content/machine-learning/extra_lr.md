---
title: 'Extra - Multiple Linear Regression'
description: 'Residual plots, Hypothesis testing, QQ-plot, The hat matrix'
pubDate: 'Sep 09 2023'
heroImage: '/extra_ml.png'
---

#### Table of contents
- [Building and inspecting models](#building-and-inspecting-models)
- [Multiple Linear Regression](#multiple-linear-regression)
  - [Estimating the Regression Coefficient in Multiple Regression Setting](#estimating-the-regression-coefficient-in-multiple-regression-setting)
  - [Estimating parameters $\\beta$ and $\\sigma^2$ in matrix form](#estimating-parameters-beta-and-sigma2-in-matrix-form)
    - [Finding a close-form solution, isolating $\\hat \\beta$](#finding-a-close-form-solution-isolating-hat-beta)
  - [Making predictions from a linear model](#making-predictions-from-a-linear-model)
  - [Residuals](#residuals)
- [Residual plots](#residual-plots)
  - [Hypothesis Testing](#hypothesis-testing)
    - [F-test](#f-test)
    - [What does the F-test test for??](#what-does-the-f-test-test-for)
  - [QQ-plot](#qq-plot)
    - [Cooks Distance](#cooks-distance)
    - [Leverage](#leverage)
    - [Studentised Residuals](#studentised-residuals)
- [Multiple Linear Regression in Matrix Form](#multiple-linear-regression-in-matrix-form)
  - [The Hat Matrix](#the-hat-matrix)
- [The Dot Product - Recap](#the-dot-product---recap)


# Building and inspecting models

There is some general things you look at when building and inspecting a model.

**Which features should be in the model?**

**How should features enter the model?** (in terms of transformations and interactions between features)

**Does the model fit well?**(in terms of checking whether the model assumptions are met, and finding out in what way the assumptions are unsuitable)

# Multiple Linear Regression

In this setting we have several predictors variables to predict a response.

Instead of fitting a separate simple linear regression model for each predictor, a better approach is to extend the simple linear regression model so that it can directly accommodate (imødekomme) multiple predictors. 

$$
Y=\beta_0 + \beta_1X_1+\beta_2X_2 + ...+ \beta_p X_p + \epsilon \ \ \ \ \ \ \ (3.19)
$$

More generally with $p$ features an an intercept term the model is expressed by:

$Y=X \beta+\epsilon$

That is,

$$
\underbrace{\begin{bmatrix}
y_1 \\
\vdots \\
y_n 
\end{bmatrix}}_{\text{Observations}} = \underbrace{\begin{bmatrix}
1 & x_{i1} & \dots &x_{ip}\\
\vdots & \vdots & \vdots & \vdots \\
1 & x_{n1}  & \dots & x_{np}
\end{bmatrix}}_\text{design matrix} \underbrace{\begin{bmatrix}
\beta_0 \\
\vdots \\
\beta_p 
\end{bmatrix}}_{\text{model parameter vector}} + \begin{bmatrix}
\epsilon_1 \\
\vdots \\
\epsilon_n 
\end{bmatrix} 
$$

## Estimating the Regression Coefficient in Multiple Regression Setting

Given the estimates $\hat\beta_0, \hat\beta_1,...,\hat\beta_p$ we can make predictions using this formula:

$$
\hat y = \hat \beta_0 + \hat \beta_1x_1 + \hat \beta_2x_2 +...+\hat \beta_px_p
$$

The parameters are estimated using the same least squares approach that we saw in simple linear regression: We choose the beta estimates to minimise the sum of squared residuals:

$$
RSS = \sum_{i=1}^n (y_i-\hat y_i)^2 \\ = \sum_{i=1}^n (y_i-\hat\beta_0-\hat\beta_1x_{i1}- \hat \beta_2x_{i2}-...-\hat \beta_p x_{ip})^2
$$

All the term are converted into minus because of: $-\hat y_i$. 

**In matrix form:**

$$
RSS = \sum^n_{i=1} (y_i-x_i^T\beta)^2
$$

The values of all the beta estimates that minimise the equation above are the **multiple least squares regression coefficient estimates**. Unlike simple linear regression estimates, these have somewhat complicated forms that are easier to explain using matrix algebra.

## Estimating parameters $\beta$ and $\sigma^2$ in matrix form

### Finding a close-form solution, isolating $\hat \beta$

Linear regression: $Y = X \beta + \epsilon$

Design matrix: 

The RSS is given by:


<div style="text-align: center;">
    <img src="/Screenshot_2022-10-29_at_11.50.01.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

$$
RSS = \sum^n_{i=1} (y_i-x_i^T\beta)^2 = (Y-X \beta)^T (Y-X \beta)
$$

The least squares estimator $\hat \beta$ (a vector that contains all of our estimates for the parameters in our model), is the solution for $\beta$ in the equations, where we take the partial derivative of our Sum of Squares with respect to our beta terms:

 

$$
\frac{\partial L}{\partial \beta} = 0
$$

And we set those equal to 0. So essentially we have got a bunch of equations that we are setting equal to 0, and though a bit of calculus we can come up with the “normal equations”: 

$$
X^TX \hat\beta = X^TY
$$

If you solve this equation for $\hat \beta$, you get:

$$
\hat \beta = (X^TX)^{-1}X^TY
$$

So this above, is how we can calculate our regression coefficients in our model. This matrix equation proved the values of the model parameters that minimise the least squares. 

**Also from slides**


<div style="text-align: center;">
    <img src="/Screenshot_2022-10-29_at_12.09.20.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

In the gaussian model, minimising RSS is exactly the same as fitting by maximum likelihood, which uses loglikelihood:

<div style="text-align: center;">
    <img src="/Screenshot_2022-10-20_at_12.26.11.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>


Minimising RSS to get $\beta$, proof from slide Using this closed matrix formula, we can find the set of parameters $\beta$ that optimises the sum of squares-loss-function and therefore finds the linear model best fitting some training data. 

<div style="text-align: center;">
    <img src="/Screenshot_2022-10-20_at_12.38.48.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

## Making predictions from a linear model

A natural prediction is the mean of $Y_i$, which is simply the value
of the regression-line at $x_i$ (more generally, the value of the
linear predictor at $x_i$):

## Residuals

Inspecting the residuals is useful for checking whether there are hints that the model is wrong and, if so, in which ways.

The raw residuals:

$e_i = y_i-\hat y_i = y_i-x^T\beta$

Standardised residuals (that have been scalared by their standard error, they are all approx standard normal)

When something is standard normal, then the mean is 1 and the variance is 1.

$$
\frac{e_i}{\hat\sigma^2(1-h_{ii})}
$$

# Residual plots

## Hypothesis Testing

When you want to test if a feature in a model provides information about the relationship. Since the coefficient $\beta_j$ is multiplied onto the feature, then if we test if the feature is NOT important, we test for $\beta_j =0$.

The F-test statistic measures how much the RSS changes
when we use the simpler model instead of the more complex
one.

<div style="text-align: center;">
    <img src="/Screenshot_2022-10-20_at_12.53.39.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

### F-test

Two models – one “smaller” than the other because $q$ coefficients have been set to zero – can be compared with an F-test. The F-statistic is computed from the RSS of the two models (RSS0 for the small model, RSS1 for the larger). Example with 2 models $M0$ and $M1$

$$
F= \frac{(RSS_{M0}-RSS_{M1})/q}{RSS_{M1}/(n-p-1)}
$$

### What does the F-test test for??

A special case of the F-test indicates whether the set of explanatory variables included is useful at all in explaining the response.

## QQ-plot

Check if whether residuals are indeed approx standard normal - if so the plot resembles a straight line.

We want them to be standard normal because, if the dots does not follow a straight line, for example the dots follows more a curve, then it is a sign that you should find another model to fit, because it doesn’t follow a linear relationship. OR it can tell you that your should scale your data in a way, that gives you more of a linear relationship. 


<div style="text-align: center;">
    <img src="/Screenshot_2022-10-20_at_13.14.35.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

### Cooks Distance

### Leverage

### Studentised Residuals

# Multiple Linear Regression in Matrix Form

<div style="text-align: center;">
    <img src="/Untitled.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

When we do our design matrix we have to include another column of the data matrix, because we need to capture the intercept, so we add a column of 1s. 

<div style="text-align: center;">
    <img src="/Untitled%201.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

When you calculate $X^TX$, you get a symmetric matrix, and some quite interesting entries in the matrix:


<div style="text-align: center;">
    <img src="/Untitled%202.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

Also, when you calculate $X^TY$ you get an interesting matrix

<div style="text-align: center;">
    <img src="/Screenshot_2022-10-20_at_11.31.20.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

## The Hat Matrix

The hat matrix is a projection matrix, an orthogonal projection matrix. The hat matrix H, maps a vector into the column space spanned by X. See the drawing below. The column space spanned by X can by represented by a plane. Above it we have an observation Y (see red dot). If we apply the hat matrix H onto Y, $HY$, what happens then is that we project Y onto the plane (the space spanned by X). We project to the point on the plane where it is closest to the point Y. Lets all this point on the plane for $\hat Y$. On the space spanned by X, when the point $Y$ is closest, is when it is orthogonal to the point from the plane. So $H$ **********is an orthogonal projection matrix that projects a vector onto the column space spanned by X.********** 

<div style="text-align: center;">
    <img src="/Screenshot_2022-10-20_at_11.55.14.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

But why is it called Hat Matrix? 

It is called a hat matrix because, when we apply it to $Y$, $(HY)$  we are going to end up with: $HY=\hat Y$. Down below is a proof that $HY=\hat Y$

<div style="text-align: center;">
    <img src="/Screenshot_2022-10-20_at_11.55.34.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>


# The Dot Product - Recap

<div style="text-align: center;">
    <img src="/Screenshot_2022-10-29_at_12.07.52.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>
