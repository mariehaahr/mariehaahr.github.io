---
title: 'Lecture 2'
description: 'Linear regression'
pubDate: 'Sep 08 2022'
heroImage: '/blog-placeholder-3.jpg'
---

Subject: Intro to statistical Learning, Linear Regression

# Statistical Learning

## Reducible Error and Irreducible Error

The accuracy of $\hat{Y}$ as a prediction for $Y$ depends on two quantities, which we will call the reducible error and the irreducible error. In general, reducible $\hat f$ will not be a perfect estimate for $f$, and this inaccuracy will introduce error some error. This error is reducible because we can potentially improve the accuracy of $\hat f$ by using the most appropriate statistical learning technique to error estimate $f.$

Y is also a function of $ϵ$, which, by definition, cannot be predicted using $X$. Therefore, variability associated with $ϵ$ also affects the accuracy of our predictions. This is known as the irreducible error, because no matter how well we estimate $f$, we cannot reduce the error introduced by $ϵ$.

Consider a given estimate $\hat f$ and a set of predictors $X$, which yields the $\hat Y$ prediction $\hat Y = \hat f(X)$. Assume for a moment that both $\hat f$ and $X$ are fixed
so that the only variability comes from $\epsilon$. Then, it is easy to show that:

$$
E(Y-\hat Y)^2 = E[f(X) + \epsilon - \hat f(X)]^2 \\ = \underbrace{[f(X) - \hat f(X)]^2}_\text{Reducible} + \underbrace{Var(\epsilon)}_\text{Irreducible}
$$

Where $E(Y-\hat Y)^2$ represents the average, or **************expected value************** of the squared difference between the predicted and actual value of $Y$, and $Var(\epsilon)$ represents the variance associated with the error term $\epsilon$.

# Linear Regression

Supervised Machine Learning. A useful tool for predicting a quantitative response. 

Linear regression is a parametric model, that assumes our estimate $\hat f$ that models the relationship between our set of predictors $X$ and the response $Y$ to be linear, ie. each feature influences the response relative to some weight. 

### The General Idea Behind Linear Regression

We have an example of seeing if there is a linear relationship between mouse size (Y-axis) and mouse weight (X-axis).

### A little more complicated example also mentioned in the steps

Imagine we wanted to predict the mouse length from both the weight of the mouse and the length of the tail. When we plot the 3 predictors against each other, with mouse weight on the y-axis, the mouse weight on the x-axis and the tail length on the z-axis, and fit the Least Squares line on the data, now IF we imagine that the equation looks like this: $y = 0.1 +0.7x 0+0z$, we  

******1)****** The first thing you do is you fit a line to the data with Least Squares. 
- You draw a horizontal line through the data and calculate the $RSS$. You rotate the line and calculate the $RSS$ again. You keep doing that until you can make a plot of the rotations against the $RSS$s from those rotations.

<div style="text-align: center;">
    <img src="/Screenshot_2022-10-19_at_10.06.48.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

******2)****** Then you calculate $R^2$

- How good is our guess to predicting Y given X? Well if our Least Squares estimation line with form $y=ax+b$, if $a$ is not 0, then we can actually make a guess. Well how good is this guess? We can calculate that with $R^2$.
- Keep in mind that the variance of something is the average sum of squares: $Var(something) = \frac{\text{Sum of Squres}}{\text{The number of those things}}$.
If we look at the variance of Y-axis, the mouse size; we take the sum of squares around the mean of the data. If we compare that $SS(mean)$ to the $SS(fit)$ the sum of squares around the fitted line, we see that the variance in the fitted line is smaller than the variance around the mean on the Y-axis. This tells us that some of the variation of mouse size is explained by taking mouse weight into account. AKA heavier mice = bigger mice.
- $R^2$ then tells us how much of the variation in mouse size can be explained by taking mouse weight into account. $R^2 = \frac{Var(mean)-Var(fit)}{Var(mean)}$.
Lets say that the $R^2$ value is 60%. This means that there is a 60% reduction in variance when we take the mouse weight into account. So we can actually say that mouse weight explains 60% of the variation in mouse size. We can also just calculate this with the sum of squares instead of the variance: $R^2=\frac{SS(mean)-SS(fit)}{SS(mean)}$.

******3)****** Then you calculate the p-value of $R^2$

- We need a way to determine if the $R^2$ value is statistically significant:

$$
F = \frac{SS(mean)-SS(fit) / (p_{fit} - p_{mean})}{SS(fit)/(n-p_{fit})} \ \ \ \ \ \ \ \ (1)
$$

Where $(p_{fit} - p_{mean})$ and $(n-p_{fit})$ are the degrees of freedom, and they turn the sum of squares into variances. 

$p_{fit}$ is the number of parameters in the fit line. So in a normal 2-dimensional line, we have the parameters slope and y-intercept, aka $p_{fit}=2$. 

$p_{mean}$ is the number of parameters in the mean line. In general the mean line is just a horizontal line $y=intercept$. This equation has 1 parameter, so $p_{mean} = 1$.

This means that the **numerator** in equation $(1)$ is: The variance explained by the extra parameter. So lets imagine we had a plane instead of a normal line. The. the parameters in the $p_{fit}=3$ and the numerator would be divided by 2 instead of 1, this means that the variance is explained by 2 other parameters (in our example, both tail length and mouse size). 

The **denominator** in equation $(1)$ is: The variation in mouse size (the Y-axis) NOT explained by the fit. So, the sum of squares of the residuals that remain after we fit our new line to the data. But why do we divide it by $(n-p_{fit})$  and not just $n$?
- Well, intuitively, the more parameters you have in your equation, the more data you need to estimate them. For example, you only need two points to estimate a line, but you need 3 points to estimate a plane.

- So if the fit is good, then:

$$
F=\frac{\text{The variation explained by the extra parameters in the fit}}{\text{The variation NOT explained by the extra parameter in the fit} }
$$

Then this equation will $\frac{\text{Large number}}{\text{Small number}}$ and $F$ will be large. How do we turn $F$ into a p-value? A little complicated

The p-value will be smaller when there are more samples relative to the number of parameters in the fit equation. 

### A little recap

So Linear Regression works like this:

1. It quantifies the relationship in the data with $R^2$, and this value needs to large in order to be good.
2. It also determines how reliable that relationship is, and this is done with F and the p-value, where the p-value needs to be small in order to be statistically significant. 

## 3.1 Simple Linear Regression

An approach for predicting a quantitative response $Y$ on the basis of a single predictor $X$. It assumes that there is approximately a linear relationship between $X$ and $Y.$  This linear relationship is written as:

$$
Y \approx \beta_0 + \beta_1 X
$$

$\beta_0$ represents the *intercept* (The expected value of $Y$ when $X=0$)and $\beta_1$ represents the *slope* (The average increase in $Y$ associated with a one-unit increase in $X$). Together they are the model coefficients or parameters.

So we assume that $f(X)$ is linear, $Y=\beta_0+\beta_1x_1 + \epsilon$

And the noise terms $\epsilon$, are assumed to be:

- independent
- Gaussian with mean 0 and
- constant variance $σ^2$ (unknown).

## 3.1.1 Estimating the Coefficients FOR SIMPLE LINEAR REGRESSION

Before we can use the formula to make predictions, we must estimate the coefficients. We have n observations pairs, measurements of $X$ and $Y$:

$(x_1,y_1),...,(x_n,y_n)$

Our goal is to obtain estimates $\hat \beta_0$ and $\hat \beta_1$ such that the linear model fits the available data well. So that:

$y_i = \hat \beta_0 + \hat \beta_1x_i$ for $i=1,2,...,n$.

We want to find an intercept $\hat \beta_0$ and a slope $\hat \beta_1$ such that the resulting line is as close as possible to the $n$ data points. 

Down below we have the prediction for $Y$ based on the $i$th value of $X$. 

$\hat y_i = \hat \beta_0 + \hat \beta_1x_i$, and for this, $e_i = y_i - \hat y_i$ is the $i$th residual (the distance from the point $i$ to the regression line).

$$
\text{The residual sum of squares (RSS)} = e_1^2+e_2^2+...+e_n^2
$$

Which is the same as:

$$
RSS = (y_1-\hat \beta_0 - \hat \beta_1 x_1)^2 + (y_2-\hat \beta_0 - \hat \beta_1 x_2)^2 +...+ (y_n-\hat \beta_0 - \hat \beta_1 x_n)^2
$$

********************************************The least squares approach******************************************** chooses $\hat \beta_0$ and $\hat \beta_1$ to minimise the $RSS$.

### Coefficient Estimates for The Least Squares FOR SIMPLE LINEAR REGRESSION

With some calculus, we can show the the minimisers of the two estimates are this:

$$
\hat \beta_1 = \frac{\sum^n_{i=1} (x_i-\bar x)(y_i-\bar y)}{\sum^n_{i=1}(x_i-\bar x)^2} \ \ \ \ \ (3.4.a)
$$

$$
\hat \beta_0 = \bar y - \hat \beta_1 \bar x \ \ \ \ \ \ \ \ \ \ (3.4.b)
$$

Where   $\bar y = \frac{1}{n} \sum^n_{i=1} y_i$   and   $\bar x = \frac{1}{n} \sum^n_{i=1}x_i$    are the sample means.

## 3.1.2 Assessing the Accuracy of the Coefficient Estimates

The true relationship between $X$ and $Y$ takes form in  $Y = f(X) + \epsilon$ for some unknown function $f$, where $\epsilon$ is a mean-zero random error term. 

$\epsilon$ is a catch-all for what we miss with this simple model. The true relationship for some real-life data is probably not linear. The error term $\epsilon$ is typically independent of $X$.

### The concept of bias in linear regression (from some website)

What exactly does this mean? It means that on the basis of one particular set of observations  $y_1, …, y_n$ ,  $\hat \mu$  might overestimate $μ$, and on the basis of another set of observations, $\hat \mu$ might underestimate $μ$. But if we could average a huge number of estimates of $μ$ obtained from a huge number of sets of observations, then this average would exactly equal $μ$. Hence, an unbiased estimator does not systematically over- or under-estimate the true parameter. The property of unbiasedness holds for the least squares coefficient estimates given by (3.4) as well: if we estimate $β_0$ and $β_1$ on the basis of a particular data set, then our estimates won’t be exactly equal to $β_0$ and $β_1$. But if we could average the estimates obtained over a huge number of data sets, then the average of these estimates would be spot on!

**********************Another Explanation of Bias in Linear Regression:**********************

For any given phenomenon, the bias term we include in our equations is meant to represent the tendency of the data to have a distribution centred about a given value that is offset from an origin; in a way, the data is biased towards that offset. For example, when we are given a linear regression problem, if we observe from the distribution of the data that most values are centred around a number ‘b’, our resulting model would need to factor in this ‘b’. In the case of linear regression, this idea would be represented with the traditional line equation ‘y = mx + b’, where ‘b’ is called the bias term or offset and represents the tendency of the regression result to land consistently offset from the origin near b units. It is a very common intentional bias in machine learning models.

In linear models there is an easy way of controlling these errors - through regularisation. OLS is known to give unbiased results with low variance as compared to non linear models. **************Ridge************** and **********Lasso********** gives biased results with much lower variance than OLS. 



<div style="text-align: center;">
    <img src="/Screenshot_2022-10-19_at_09.52.32.png" alt="dd" width="600" height="350">
</div>

The bullseye is the true value we want to predict and the blue dots are what the model actually predicts. In

# Bias & Variance in Linear Regression

The term ****************************inductive bias**************************** refers to the assumptions that that let us *extrapolate* (translate some values from a known dataset to some unknown relations) from examples we have seen to similar examples we have not seen. ********Bias******** is what makes us prefer one model over another, usually framed as types of function that we think might be likely, or assumptions about functions. 

The ****************variance**************** of an estimator is another important effect. Imagine we gave several draws of a dataset, coming from the same distribution (or the same population, or true function). 


<div style="text-align: center;">
    <img src="/Untitled.png" alt="dd" width="600" height="350">
</div>

Here we see the 3 different drawn datasets, in blue, green and red. Here we have the mean fitted as a function to predict new points in the first picture, a linear regression in the second picture, and lastly a three degree polynomial fitted. If our predictions vary too wildly (like in the third plot), that suggests that our performance on new data will also be poor. High variance is exactly the ****overfitting effect.**** To balance these effects, we have to choose the right model complexity. Here we can use all these methods like hold-out data, where we use the test data to estimate the model’s future performance (on the test set). We can then compare several different models and complexities and choose the one that performs the best. 

see lecture 4 for all the different methods. 

## Ridge

## Lasso

### The Standard Error

How accurate is the sample mean $\hat \mu$ as an estimate of $\mu$ ?

SO, a standard deviation tells you something about how the data is distributed around the mean, while the ****************************standard error**************************** tells you something about how the mean is distributed. So we take the mean of the means and the standard deviation of the means. 

1. Take a bunch of samples from the same population, each with the same number of samples $n$.
2. Then we calculate the mean of every sample
3. Then we calculate the standard deviation of the means, and that is the standard error

You can calculate the standard error for any statistic. Just calculate a lot of that statistic for all the samples of the data, and then take the standard deviation of those statistics. 

A single estimate $\hat \mu$ may be substantial underestimate or overestimate of $\mu$ (compared to many estimates). How far off is this single estimate? We answer that question by computing the *standard error* of $\hat \mu$.

$$
Var(\hat \mu) = SE(\hat \mu)^2 = \frac{\sigma^2}{n}
$$

Here $\sigma$ is the standard deviation of each of the realisations $y_i$ of  $Y$.

The standard error tells us the average amount that this estimate $\hat \mu$ differs from the actual value of $\mu$. This deviation shrinks with $n$. The more observations, the smaller the $SE(\hat \mu)$.

**The standard error associated with $\hat \beta_0$ and $\hat \beta_1$**

$$
SE(\hat \beta_0)^2 = \sigma^2 \left[ \frac{1}{n} + \frac{\bar x^2}{\sum^n_{i=1}(x_i-\bar x)^2}\right]

$$

$$
SE(\hat \beta_1)^2 = \frac{\sigma^2}{\sum^n_{i=1}(x_i-\bar x)^2}
$$

************************************************************In this case, the beta estimates has a low standard error, if the measure of the amount of sampling variation when estimating $\hat \beta$ is low.** 

Where $\sigma^2 = Var(\epsilon)$. We need to assume that the errors $\epsilon_i$ for each observation are uncorrelated with common variance $\sigma^2$ (even though it is not true).

 The formula for $SE(\hat \beta_1)$ is smaller when the $x_i$ are more spread out. Intuitively we have more *leverage* to estimate a slope when this is the case.

We also see that the formula for $SE(\hat \beta_0)$ would be the same as $SE(\hat \mu)$ if $\bar x$ were zero.

Generally $\sigma^2$ is unknown, but it can be estimated from the data. This estimate of $\sigma$ is known as the ***********************residual standard error***********************, and is given by the formula:

$$
RSE = \sqrt{RSS/(n-2)}
$$

So when $\sigma^2$ is estimated from the data we should write $\hat{SE}(\beta_1)$ to indicate that an estimate has been made, but for simplicity of notation we will drop