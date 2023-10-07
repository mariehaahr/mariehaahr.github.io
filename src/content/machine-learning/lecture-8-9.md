---
title: 'Lecture 8 and 9'
description: 'Generative Models & Bayes Theorem'
pubDate: 'Oct 04 2023'
heroImage: '/lec89_ml.png'
---

**Readings**: Hands On: 1.5, ISLwR 4.4 & 4.5

#### Table of contents
- [Bayes Theorem](#bayes-theorem)
  - [Why is Bayes Theorem Helpful in Machine Learning?](#why-is-bayes-theorem-helpful-in-machine-learning)
    - [But why do we need another method, when we have Logistic Regression?](#but-why-do-we-need-another-method-when-we-have-logistic-regression)
    - [Estimating $\\pi\_k$](#estimating-pi_k)
    - [Non-Parametric Density Estimates](#non-parametric-density-estimates)
- [Linear Discriminant Analysis](#linear-discriminant-analysis)
  - [For 1 Predictor](#for-1-predictor)
    - [Linear Discriminants](#linear-discriminants)
  - [Linear Discriminant Analysis For \> 1 Predictor.](#linear-discriminant-analysis-for--1-predictor)
    - [Density Formula for a Multivariate Gaussian Distribution (Matrix Form)](#density-formula-for-a-multivariate-gaussian-distribution-matrix-form)
    - [Estimating Parameters](#estimating-parameters)
    - [Mahalanobis Distance](#mahalanobis-distance)
- [Quadratic Discriminant Analysis](#quadratic-discriminant-analysis)
    - [Quadratic Discriminants](#quadratic-discriminants)
    - [Trade-off between a simple and a complex model](#trade-off-between-a-simple-and-a-complex-model)
    - [Bias Variance Trade-off with LDA and QDA](#bias-variance-trade-off-with-lda-and-qda)
    - [Which is better? LDA or QDA?](#which-is-better-lda-or-qda)
- [Naive Bayes](#naive-bayes)
    - [Why is this a powerful assumption?](#why-is-this-a-powerful-assumption)
    - [Estimating Class Conditionals](#estimating-class-conditionals)
- [Connection Between LDA and Logistic Regression](#connection-between-lda-and-logistic-regression)
- [A Comparison of LDA, QDA and Naive Bayes](#a-comparison-of-lda-qda-and-naive-bayes)
- [Kernel Density Estimate](#kernel-density-estimate)
    - [Seaborn KDE plot](#seaborn-kde-plot)
    - [Kernel Density Estimation in Python](#kernel-density-estimation-in-python)

# Bayes Theorem

Bayes Theorem determines conditional probability. 

The conditional probability given that we know one thing about an event can be derived from knowing the other thing about the event.

So in situations where we don’t have all data available, Bayes theorem can help us approximate probabilities. 

Recall that the probability of A given B is equal the probability that A and B occurred (the intersection of A and B) divided by the probability of B:

$$
\text{Recall: } P(A|B)=\frac{P(A \cap B)}{P(B)} \\ \text{The same thing: } P(B|A) = \frac{P(A \cap B)}{P(A)} \\ \text{Since we know that the joint probability is: } P(A\cap B)=P(A|B)\cdot P(B) = P(B|A)\cdot P(A)\\ \text{Since the numerator in both equations are equal, we can rewrite into Bayes Theorem:} \\ P(A|B)=\frac{P(B|A)\cdot P(A)}{P(B)} \\ \text{and likewise} \\ P(B|A)=\frac{P(A|B)\cdot P(B)}{P(A)}
$$

## Why is Bayes Theorem Helpful in Machine Learning?

When we are in the side of Machine Learning where we work with Generative Models, Bayes Theorem comes in helpful. 

A Generative approach is an alternative and less direct approach (than logistic regression) to estimating the probabilities $(P(Y=k|X=x)$.

When working with **************generative models**************, we model the distribution of the predictors $X$ separately in each of the response classes. 

<aside>
📌 **Discriminative models** make predictions on the unseen data based on conditional probability and can be used either for classification or regression.
On the contrary, a **generative model** focuses on the distribution of a dataset to return a probability for a given example.

</aside>

We use **************************Bayes Theorem************************** to flip these distributions around into estimates for $P(Y=k|X=x)$. 

When the distribution of X within each class is assumed to be normal, it turns out that the model is very similar in form to logistic regression.

### But why do we need another method, when we have Logistic Regression?

- When there is substantial separation between the two classes.
- If the distribution of the predictors $X$ is approximately normal in each of the classes and the sample size is small, generative methods are more accurate that statistical ones.
- The methods in this section can naturally extended to the case of more than two response classes (which is not possible to model with logistic regression).

Let $\pi_k$ represent the overall or **prior probability** that a randomly chosen observation comes from the $k$th class. 

Let $f_k(X)=Pr(X|Y=k)$ denote the **density function** of $X$ for an observation that comes from the $k$th class.

Then Bayes Theorem states:

$$
Pr(Y=k|X=x)=\frac{\pi_k f_k(x)}{\sum_{l=1}^K \pi_l f_l(x)} \ \ \ \ \ (4.15)
$$

Here $p_k(x)=Pr(Y=k|X=x)$ is the **posterior probability** that an observation $X=x$ belongs to the $k$th class.

This equation (4.15) suggest that of directly computing the posterior probability  $p_k(x)$ as in Section 4.3.1, we can simply plug in estimates of $π_k$ and $f_k(x)$ into (4.15). 

### Estimating $\pi_k$

In general, estimating $π_k$ is easy if we have a random sample from the population: we simply compute the fraction of the training observations that belong to the kth class.

But to estimate $f_k(x)$  is more complicated, and we will typically have to make some simplifying assumptions.

**Then we have some different classifiers that use different estimates of $f_k(x)$ to approximate Bayes Classifier (which is the ‘perfect’ situation).**

- Linear Discriminant Analysis LDM
- Quadratic Discriminant Analysis QDM
- Naive Bayes

### Non-Parametric Density Estimates

You can also use general non-parametric density estimates, for instance kernel estimates and histograms.

When using a non-parametric methods, we don’t make any explicit assumptions about the form of $f$ in advance, as opposed to these apporaches mentioned in this section.

# Linear Discriminant Analysis

<aside>
📌 The main thing about LDA is that we assume that $f_k(x)$ is gaussian, there is class specific means and ********************************************************************************we assume that the variances are equal: $\sigma^2_1 = ... = \sigma^2_K$.**

</aside>

## For 1 Predictor

For now, assume that  $p = 1$ - that is, we have only one predictor. We would like to obtain an estimate for $f_k(x)$ that we can plug into (4.15) in order to estimate $p_k(x)$.

To estimate $f_k(x)$, we will first make some assumptions about its form:

- We assume $f_k(x)$ is gaussian, and takes the form:

$$
f_k(x)=\frac{1}{\sqrt{2\pi\sigma_k}}e^{-\frac{1}{2 \sigma^2_k} (x-\mu_k)^2} \ \ \ \ (4.16)
$$

Here $\mu_k$ and $\sigma_k^2$ are the mean and variance parameters for the $k$th class. 

- We assume that the variances are equal $\sigma_1^2 = ...=\sigma_K^2$
    
    **By making this assumption, the classifier becomes linear.**
    

************************************************************************This is where Bayes Theorem plays a role.************************************************************************ If we plug the gaussian form (4.16) into bayes theorem, we have this:

$$
p_k(x) = \frac{\pi_k \frac{1}{\sqrt{2\pi\sigma}}e^{-\frac{1}{2 \sigma^2} (x-\mu_k)^2}}{\sum_{l=1}^K \pi_l \frac{1}{\sqrt{2\pi\sigma}}e^{-\frac{1}{2 \sigma^2} (x-\mu_l)^2}} \ \ \ \ \ (4.17)
$$

Here $\pi_k$ is the prior probability of class $k$. The Bayes Classifier involves assigning an observation $X=x$ to the class for which $(4.17)$  is largest. 

The Bayes Rule for 0-1 loss:

$$
\hat Y (x) = \argmax_k P(Y=k|X=x)\\ = \argmax_k f_k(x)\pi_k
$$

The linear discriminant analysis method (LDS) approximates the Bayes classifier by plugging estimates for $\pi_k, \mu_k$ and $\sigma^2$ into this equation:

$$
\delta_k(x)=x \cdot \frac{\mu_k}{\sigma^2}- \frac{\mu_k^2}{2\sigma^2} + \log(\pi_k) \ \ \ \ (4.18)
$$

And here we use the estimates for $\hat \mu_k$ and $\hat \sigma^2$:


<div style="text-align: center;">
    <img src="/Screenshot_2022-09-23_at_15.09.38.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

Where $n$ is the total number of training observations, and $n_k$ is the number of training observations in the $k$th class. The estimate for $μ_k$ is simply the average of all the training observations from the $k$th class, while $\hat σ^2$ can be seen as a weighted average of the sample variances for each of the $K$ classes. Sometimes we have knowledge of the class membership probabilities $π_1, . . . , π_K ,$ which can be used directly. In the absence of any additional information, LDA estimates $π_k$ using the proportion of the training observations that belong to the $k$th class. In other words:

$$

\hat \pi_k = n_k /n \\ \frac{\#\text{Number of samples in class } k}{\text{Total \# of samples}}
$$

$$
\hat\delta_k(x)=x \cdot \frac{\hat\mu_k}{\hat\sigma^2}- \frac{\hat\mu_k^2}{2\hat\sigma^2} + \log(\hat\pi_k) \ \ \ \ \\ \text{Matrix form: } \delta_k(x) = x^T \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \log\pi_k
$$

### Linear Discriminants

We can show, that the discriminants $\delta_k(x)$ (i.e. the posterior probability distributions) are indeed linear. For simplicity, we show this in the univariate case. We can write up the posterior probabilities using Bayes Theorem and then simplify:


<div style="text-align: center;">
    <img src="/Screenshot_2022-11-12_at_12.39.03.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

*Taking the log and rearranging terms, we arrive at the final form, which is a linear equation. The word linear in the classifier's name stems from the fact that the discriminant functions $\hat{\delta}_k(x)$ are linear functions of x.*

The word Linear in the classifiers name, states from the fact that the discriminant function $\hat \delta_k(x)$ are linear functions of $x$.

In other words, the LDA classifier’s error rate is pretty close to Bayes error rate, and performs pretty well.

<aside>
📌 To reiterate, the LDA classifier results from **assumin**g that the observations within each class come from a **normal distribution** with a **class-specific mean** and a **common variance** $σ^2$, and plugging estimates for these parameters into the **Bayes classifier.**

</aside>

## Linear Discriminant Analysis For > 1 Predictor.

We now assume that $X=(X_1, X_2,...,X_p)$ is drawn from a multivariate Gaussian (multivariate normal) distribution, with a **class specific mean (mean vector) and a common covariance matrix (common variance, just like before**).

The multivariate Gaussian distribution assumes that each individual predictor follows a one-dimensional normal distribution, as in (4.16), with some correlation between each pair of predictors.

To indicate that a $p$-dimensional random variable $X$ has a multivariate Gaussian distribution, we write $X \sim N(\mu,\Sigma)$.

We assume that the observations in the $k$th class are drawn from a multivariate gaussian distribution where $\mu_k$ is a class specific mean vector, and $\Sigma$ is a covariance matrix that is also common to all $K$ classes.

<div style="text-align: center;">
    <img src="/Screenshot_2022-09-23_at_15.17.32.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

*The bell shape will be distorted if the predictors are correlated or have unequal variances, as is illustrated in the right-hand panel of Figure. (but we assume equal covariance).*

Since the covariance matrix determines the shape of the Gaussian density, in LDA, the Gaussian densities for different classes have the **same shape but are shifted versions of each other** (different mean vectors).

### Density Formula for a Multivariate Gaussian Distribution (Matrix Form)

$$
f_k(x)=\frac{1}{(2\pi)^{p/2} |\Sigma|^{1/2}} exp(-\frac{1}{2}(x-\mu_k)^T \Sigma^{-1}(x-\mu_k))
$$

********OBS:******** The $\Sigma$ in the equation is the covariance matrix, not sum!


<div style="text-align: center;">
    <img src="/Screenshot_2022-09-27_at_11.03.27.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

Here $E(X)=\mu$, is the mean of $X$ (a vector with $p$ components) and $Cov(X) = \Sigma$ is the $p$ x $p$ covariance matrix of $X$.

$$
\hat Y (x)= \delta_k(x) = x^T \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \log\pi_k
$$

**This is the final classifier. Given any $*x*$, you simply plug into this formula and see which $*k$* maximises this.**

LDA gives you a linear boundary because the quadratic term is dropped.

### Estimating Parameters

We need to estimate the unknown parameters $\mu_1,...,\mu_K$ and $\pi_1,...,\pi_K$ and $\Sigma$. To assign a new observation $X=x$ , LDA plugs these estimates into this equation to obtain quantities $\hat \delta_k(x)$ and classifies to the class for which $\hat \delta_k(x)$ is largest.

OBS: The higher the ratio of parameters $p$ to number of samples $n$, the more we expect this overfitting to play a role.

$$

\hat \pi_k = n_k /n \\ \frac{\#\text{Number of samples in class } k}{\text{Total \# of samples}}
$$

Then, the mean vector for every class is also simple. You take all of the data points in a given class and compute the average, the sample mean:

$$
\hat\mu_k = \frac{\sum_{g_i=k} x_i}{n_k}
$$

Next, the covariance matrix formula. First, you divide the data points into two given classes according to the given labels. If we were looking at class $*k*$, for every point we subtract the corresponding mean which we computed earlier. Then multiply its transpose. Remember $*x*$ is a column vector, therefore if we have a column vector multiplied by a row vector, we get a square matrix, which is what we need.

It is the weighted average of the sample variances for each of the $K$ classes.

$$
\hat \Sigma = \frac{\sum_{k=1}^K \sum_{g_i=k}(x_i-\hat\mu_k)(x_i - \hat \mu_k)^T}{(n-K)}
$$

Recall that $(a-b)^2 = (a-b)\cdot(a-b)^T$ in matrix form.

**In summary,** if you want to use LDA to obtain a classification rule, the first step would involve estimating the parameters using the formulas above. Once you have these, then go back and find the linear discriminant function and choose a class according to the discriminant functions.

See example of application on a diabetes dataset [here](https://online.stat.psu.edu/stat508/book/export/html/645).

### Mahalanobis Distance

The exponent in the equation $f_k(x)$ actually contains the Mahalanobis distance based on $\Sigma$:

$$
(x-\mu_k)^T \Sigma^{-1}(x-\mu_k)\\=||x-\mu||_{\Sigma}^2
$$

******************The Mahalanobis distance****************** is a measurement of how far x is away from the mean.

When $\Sigma=1$, all variables are independent and standard normal, and the mahalanobis distance is the squared geometric distance from $x$ to $\mu$, aka the **************euclidean distance.**************


<div style="text-align: center;">
    <img src="/Screenshot_2022-09-27_at_11.16.25.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

# Quadratic Discriminant Analysis

<aside>
📌 The main thing about QDA is that we assume that the observations from each class are drawn from a gaussian distribution, however unlike LDA, QDA ********************************************************************************************assumes that each class has its own covariance matrix.
$X \sim N(\mu_k, \Sigma_k)$**

</aside>

Because the QDA allows for more flexibility for the covariance matrix, it tends to fit the data better than LDA, **but then it has more parameters to estimate**. The number of parameters increases significantly with QDA. Because, with QDA, you will have a separate covariance matrix for every class. If you have many classes and not so many sample points, this can be a problem.

Here we follow ******************bayes theorem****************** and assign an observation $X=x$ to the class where this equation is largest:

$$
\delta_k(x)= -\frac{1}{2}\log|\Sigma_k| - \frac{1}{2}(x-\mu_k)^T\Sigma_k^{-1} (x-\mu_k)+\log \pi_k
$$

This quadratic discriminant function is very much like the linear discriminant function except that because $Σ_k$ (the covariance matrix) is not identical, you cannot throw away the quadratic terms. This discriminant function is a quadratic function and will contain second order terms.

**The decision boundaries are quadratic equations in $*x$.***

See example with computations on diabetes data set [here](https://online.stat.psu.edu/stat508/book/export/html/645).

### Quadratic Discriminants

With the variance being class-specific, the discriminants (derived in a similar fashion as before) now are quadratic, since the quadratic term depends on the class $k$.


<div style="text-align: center;">
    <img src="/Screenshot_2022-11-12_at_12.53.45.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

This means, that each of the discriminants $\hat p_k$ is quadratic, and therefore the decision boundaries (which is the set of all points $x$ where $\hat p_1=...=\hat p_k$) are also quadratic.

### Trade-off between a simple and a complex model

There are trade-offs between fitting the training data well and having a simple model to work with. A simple model sometimes fits the data just as well as a complicated model. Even if the simple model doesn't fit the training data as well as a complex model, it still might be better on the test data because it is more robust.

### Bias Variance Trade-off with LDA and QDA

Why does it matter whether or not we assume that the K classes share a common covariance matrix? In other words, why would one prefer LDA to QDA, or vice-versa? The answer lies in the bias-variance trade-off.

### Which is better? LDA or QDA?

Roughly speaking, **LDA tends to be a better** bet than QDA if there are relatively **few training observations** and so reducing variance is crucial. In contrast, **QDA is recommended** if the training set is **very large**, so that the variance of the classifier is not a major concern, or if the assumption of a common covariance matrix for the K classes is clearly untenable.

# Naive Bayes

When there is too little data to estimate the joint distribution, LDA and QDA are not very good models to use.

We have used Bayes Theorem to develop the LDA and QDA classifiers. Now we use the theorem for Naive Bayes Classifier. 

********************************************Naive Bayes Classifier******************************************** is a third generative probabilistic model, that tries to estimate the posterior class probability $\delta_k(x)$ or $P(Y=k|X=x)$ through modelling joint distributions.

It is not necessarily reliant on the fact of gaussian features (like LDA and QDA), but it can model multiple class conditionals (also different ones).

<aside>
📌 To do this, Naive Bayes makes a very strong assumption that each observed features is independent of each other feature. Instead of assuming that these functions belong to a particular family of distributions, we only make this independence assumption.

*Within the $k$-th class, the $p$ predictors are independent.*

</aside>

When we have this independence assumption of the $p$ features, the multivariate class conditional function $f_k(x)$  factors out into the product of univariate class conditionals:

$$
\hat f_k(x) = \hat f_{k1}(x_1) \times \hat f_{k2}(x_2) \times ... \times \hat f_{kp}(x_p)
$$

Class priors is still the fraction of the data points in the given class.

$$
Pr(Y=k|X=x) = \frac{\pi_k \times f_{k1}(x_1) \times f_{k2}(x_2) \times ... \times f_{kp}(x_p) }{\sum_{l=1}^K \pi_l \times f_{l1}(x_1) \times f_{l2}(x_2) \times ... \times f_{lp}(x_p)} \ \ \ (4.30)
$$

This means, that the covariance matrix in this case is the variance along the diagonal and 0 for the rest, since there is no covariance. This means that the features has no covariance, and they don’t depend on each other at all. 

If the variables are uncorrelated then the variance-covariance matrix will be a diagonal matrix with variances of the individual variables appearing on the main diagonal of the matrix and zeros everywhere else:

<div style="text-align: center;">
    <img src="/Screenshot_2022-09-24_at_11.18.38.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

### Why is this a powerful assumption?

Essentially, estimating a p-dimensional density function is challenging because we must consider not only the marginal distribution of each predictor — that is, the distribution of each predictor on its own — but also the joint distribution of the predictors — that is, the association between the different predictors.

By assuming that the $p$ covariates are independent within each class, we completely eliminate the need to worry about the association between the $p$ predictors, because we have simply assumed that there is no association between the predictors!

<aside>
📌 This assumption lead to some pretty decent results, specially in situations where $n$ is not large enough relative to $p$ for us to effectively estimate the joint distribution of the predictors within each class.

</aside>

### Estimating Class Conditionals

Now, the only question remains, how we estimate the individual class conditionals for the $p$ features. 

Generally, there exist many **different approaches** on how to approximate these distribution of features given the knowledge about some class. Usually, however, we distinguish between *qualitative* and *quantitative* features.

**Quantitative features** are commonly estimated through a normal distribution and for **qualitative features,** we simply model the discrete probability distribution through dividing the counts in each category by the total number of observed data points. One strength of the Naive Bayes model over other the other two generative models is the freedom of probability distribution to use for modelling each of the features

- **********************************************************************************************************More on estimating the one-dimensional density function using training data**********************************************************************************************************
    
    Here we have a few options, when estimating this:
    
    - If $X_j$ is quantitative, then we can assume that $X_j|Y =k\sim N(μ_{jk},σ_{jk}^2)$. In other words, we assume that within each class, the $j$th predictor is drawn from a (univariate) normal distribution. While this may sound a bit like QDA, there is one key difference, in that here we are assuming that the predictors are independent; this amounts to QDA with an additional assumption that the class-specific covariance matrix is diagonal.
    - If $X_j$ is quantitative, then another option is to use a non-parametric estimate for $f_{jk}$. A very simple way to do this is by making a histogram for the observations of the jth predictor within each class. Then we can estimate $f_{kj}(x_j)$  as the fraction of the training observations in the $k$th class that belong to the same histogram bin as $x_j$.
    Alternatively, we can use a kernel density estimator, which is essentially a smoothed version of a histogram.
    - If $X_j$ is qualitative, then we can simply count the proportion of training observations for the $j$th predictor corresponding to each class. For instance, suppose that $X_j ∈ \{1, 2, 3\},$ and we have 100 observations
    in the $k$th class. Suppose that the $j$th predictor takes on values of 1, 2, and 3 in 32, 55, and 13 of those observations, respectively. Then
    we can estimate $f_{kj}$ as:
    
    $$
    \hat f_{kj}(x_j) = \begin{cases} 0.32 & if & x_j = 1 \\ 0.55 & if & x_j=2 \\ 0.13 & if & x_j = 3 \end{cases}
    $$
    

# Connection Between LDA and Logistic Regression

**Recall Log Odds for Logistic Regression:**

The log-odds/Logit takes the form:

$$
\log \left( \frac{p(X)}{1-p(X)}\right) = \beta_0+\beta_1X
$$

The logistic regression has a Logit that is linear in $X$.

By contrast, in a logistic regression model, increasing $X$ by 1 unit changes the log odds by $\beta_1$.

********************************************************Log Odds under the model of LDA********************************************************

$$
\begin{align*} &\log\frac{P(Y=k|X=x)}{P(Y=K|X=x)} \\ &= \log \frac{\pi_k}{\pi_K}-\frac{1}{2}(\mu_k+\mu_K)^T \Sigma^{-1} (\mu_k-\mu_K) \\ &= a_{k0}+a_k^Tx \\ &= a_k + b_k^T x\end{align*}
$$

![Screenshot 2022-11-12 at 13.37.27.png](Generative%20Models%20&%20Bayes%20Theorem%207dcfe6c8fa174522ad737d2f62b67e47/Screenshot_2022-11-12_at_13.37.27.png)

The difference between linear logistic regression and LDA is that the linear logistic model only specifies the conditional distribution $Pr(Y=k|X=x)$. No assumption is made about $P(X)$ while the LDA model specifies the joint distribution of $*X*$ and $*Y*$.

 

Moreover, linear logistic regression is solved by maximising the conditional likelihood of $*Y*$ given *X*: $Pr(Y=k|X=x)$; while LDA maximises the joint likelihood of $*Y*$ and $*X*$: $Pr(X=x,Y=k)$.

If the additional assumption made by LDA is appropriate, LDA tends to estimate the parameters more efficiently by using more information about the data.

Another advantage of LDA is that samples without class labels can be used under the model of LDA. On the other hand, LDA is not robust to large outliers.  Because logistic regression relies on fewer assumptions, it seems to be more robust to the non-Gaussian type of data.

**In practice, logistic regression and LDA often give similar results.**

# A Comparison of LDA, QDA and Naive Bayes

LDA, just like Logistic regression, assumes that the log odds of the posterior probabilities is linear in $x$.

Here, the QDA assumes that the log odds of the posterior probabilities is quadratic in $x$.

There are some peculiar observations about LDA; QDA and Naive Bayes that are worth to mention:

- Any classifier with a linear decision boundary is a special case of Naive Bayes.
- LDA is a special case of QDA.
- In a special case, Naive Bayes is a special case of LDA, with $\Sigma$ restricted to be a diagonal matrix with $j$th diagonal element equal to $\sigma_j^2$.
- Neither QDA or Naive Bayes is a special case of the other.
- Because that KNN is completely non-parametric, we can expect this approach to dominate LDA and Logistic Regression, WHEN the decision boundary is highly non-linear, provided that $n$ is very large and $p$ is small. BUT in setting where the decision boundary is non-linear BUT $n$ is only modest, or $p$ is very small, then QDA may be preferred over KNN.

# Kernel Density Estimate

When you have a histogram of data, you can create bins, and count up all the values in each bin. When you create a smoothed out continuous version of this histogram, you have a **********************************************kernel density estimation.********************************************** It allows us to estimate the probability density function, from out finite dataset. We can do this is a non-parametric way (not assuming any underlying distribution). 

### Seaborn KDE plot

Github with KDE univariate and bivariate: [https://github.com/kimfetti/Videos/blob/master/Seaborn/02_KDEplot.ipynb](https://github.com/kimfetti/Videos/blob/master/Seaborn/02_KDEplot.ipynb) 

```python
# univariate - 1 variable
# decreasing the bandwidth is like increasing the variance of the kde plot, and the other way around

# if you want a cumulative density function instead of a probability, you just set cumulative = True

# bivariate kde - 2 variables
```

### Kernel Density Estimation in Python