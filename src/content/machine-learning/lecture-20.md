---
title: 'Lecture 20'
description: 'Dimensionality Reduction, PCA'
pubDate: 'Nov 13 2023'
heroImage: '/lec20_ml.png'
---



**Readings:** ISLwR 12.1 & 6.3.1

#### Table of contents
- [**Dimensionality Reduction**](#dimensionality-reduction)
    - [The Curse of Dimensionality](#the-curse-of-dimensionality)
- [**Principle Component Analysis**](#principle-component-analysis)
    - [Choosing the Loading Vector](#choosing-the-loading-vector)
    - [The next PC](#the-next-pc)
    - [Summary](#summary)
  - [Explained Variance](#explained-variance)
  - [Scaling Variables](#scaling-variables)
  - [Deciding How Many PC’s to Use](#deciding-how-many-pcs-to-use)

# **Dimensionality Reduction**

This is in the fields of unsupervised machine learning, since we don’t have any ground truth to “supervise” the learning with. We have no labels or anything. This is not classification, this is a kind of navigation of the data set.

Two of the main problems in machine learning is **dimensionality** reduction and **clustering**. In fact, **LDA** can also be used as a way of dimensionality reduction.

The main goal with dimensionality reduction is to transform the data from a high-dimensional space into a low-dimensional space, but still preserving the meaningful properties of the original data, so we lose as little information as possible. We do dimensionality reduction for these reasons:

- **Computational performance**
  - We can train and deploy our models way faster when we have a lower dimensionality, so if the number of features is compressed, the computations are faster.
- **Model performance**
  - Some models suffer from *the curse of dimensionality*, especially when there is a lot of features and not so many data points. These models perform better when trained on compressed features.
- **Visualisation**
  - When we have more than 2 features, we cannot visualise the data. Dimensionality reduction provides us with a way of visualising high-dimensional data into a 2 dimensional (or 3) space. This also unlocks some visual exploratory analysis tools (histograms, empirical PMF, KDE, scatterplot etc.).

### The Curse of Dimensionality

The curse of dimensionality refers to when you data has too many features. The phrase was first used to express the difficulty of using brute force (aka. grid-search) to optimise a function with too many input features. 

- **Problems you might run into when your have a huge number of dimensions:**  
  - If you have more features than you have observations, you run the risk of massively overfitting your model.
  - Too many dimensions causes every observation in your data set to appear *equidistant* from all the others. (Equidistant means equally distant, i.e points on a circle are equidistant from its centre). Since we use Euclidean distance in clustering, to quantify the similarity between observations, this is a problem. Because, of all the distances are equally distant, then all the observations appear equally alike, and no meaningful clusters can be formed.

# **Principle Component Analysis**

Principle Component Analysis is an unsupervised learning technique for dimensionality reduction. Given some feature matrix $\bold{X} \in \R^{n \times p}$ with a potentially large $p$, we wish to find a compression of these features into a smaller subset of features.

If we for example wish to reduce the dimension into 1 feature (so we have 2 dimensions when plotting), we are considering an initial random vector $\bold{X} \in \R^{p}$ consisting of $p$ random variables: $X = (X_1,...,X_p)$. We want to construct a new feature $Z_1$ by doing some computation on this random vector: $Z_1 = \text{PCA}(X)$. 

PCA just determines that we are only to consider linear transformations of the original features, so the construction of the feature is always gping to be of the type:

$$
Z_1 =a_{11}X_1 + ...+ a_{1p}X_p = \sum_i a_{1iX_i}
$$

Since PCA assumes linearity, all there is left to do is to find the set of $p$ scalars $a_{11},...,a_{1p}$ that scale each feature respectively, which we them sum over when we construct $Z_1$.

These scalars, also called the **loadings** of the first principle component, can we collect into a vector $a_1 = [a_{11}\dots a_{1p}]$ and multiply that with the feature random vector $\bold{X}$, then the vectorised linear computation looks like:

$$
Z_1 = a_1^T\bold{X}
$$

To get a visualisation of this, we imagine a two-dimensional space with data points. We want to find a vector $a_1$ that we can project the data onto, in order to preserve as much variance as possible.

$$
proj_{a_1} = a_1^T X_ia
$$

Where $a_1^TX_i$ is the magnitude and $a$ is the unit vector. 

<div style="text-align: center;">
    <img src="/Screenshot_2022-11-20_at_15.06.17.png" alt="dd" width="500" height="350" style="text-align: center;" >
</div>


<span style="color: grey; font-style: italic;">
Here we have a lot of variance preserved in the projection onto the blue vector.
</span>

<div style="text-align: center;">
    <img src="/Screenshot_2022-11-20_at_15.10.06.png" alt="dd" width="650" height="350" style="text-align: center;" >
</div>

<span style="color: grey; font-style: italic;">
Here, we project the data onto another vector, where the variance is not very well preserved.
</span>


Then the mean of all the projection will be:

$$
\bar {proj} = a_1^T \bar X a
$$

The variance of the projection then:

$$
\frac{1}{n} \sum_{i=1}^n (a_1^T X_i - a_1^T \bar X)^2
$$

Since the formula for variance is $\frac{1}{n} \sum (x_i-\bar x)^2$. If we rewrite this equation a bit, we get:

$$
\text{factor out } a_1^T: \\ \frac{1}{n} \sum_{i=1}^n [a_1^T(X_i- \bar X)]^2 \\ \text{write out the square term} \\ = \frac{1}{n} \sum_{i=1}^n a_1^T(X_i-\bar X)\cdot(X_i-\bar X)^T a_1 \\ \text{Since this is just number, it does not matter where I take the transpose, that is why we can write it like that} \\ = a_1^T \underbrace{\left[ \frac{1}{n} \sum^n_{i=1} (X_i-\bar X)(X_i-\bar X)^T \right]}_{\text{The closed form formula for covariance}} a_1
$$

This shows, in order to find the vector to project the data points onto, while preserving as much possible variance, we need to maximise the covariance:

$$
a_1^T \Sigma a_1
$$

### Choosing the Loading Vector

We want to choose this loading vector $a_1$, for all loadings $a_{1i}=1$, and PCA says, that we would like to construct $Z_1$ in a way that we persist maximal variance. This makes sense, since the higher the variance in our newly constructed feature, the higher the chance for it to do well on our classes.

Now we are left with an optimisation problem, to determine the loading vector $a_1$, which is:

$$
\text{maximise } Var(Z_1) = a^T_1 \Sigma a_1 \\ \text{s.t.  } a_1^Ta_1 = 1
$$

Since $Z_1$ is defined to be the linear combination $a_1^TX$, we are maximising $Var(a_1^TX)$. 

We know from applied stats, that the variance of a random vector is the covariance matrix $\Sigma$, and that for a scaled random variable $aX$ the variance is $Var(aX) = a^2Var(X)$. Therefore the expression is $a_1^T \Sigma a_1$. 

We constrain the loadings to have a sum of one, since we wish to maximise the variance, and if we did not have this constraint, we could simply make the loading infinitely large, to maximise the variance, and this is not what we want.

This constraint of the loading to be equal to one, means that the loading vector $a_1$ is a unit vector. 

Since this is an optimisation problem, we can use **lagrange multipliers**from linear algebra, to a lagrangian optimisation problem:

$$
L(a_1) = a_1^T \Sigma a_1 - \lambda_1 (1-a_1^Ta_1)
$$

This means we have to derive this with respect to the loading vector $a_1$. Deriving with respect to a vector is complicated, but fortunately this equation is simple, since we just can treat $a_1^Ta_1$ as $a_1^2$, so deriving that becomes $2a_1$.

$$
\frac{\partial L}{\partial a_1} = 2 \Sigma a_1 - 2 \lambda_1 a_1
$$

Again, deriving $\lambda_1(1-a_1^Ta_1)$ we treat $a_1^Ta_1$ as $a_1^2$, and we simply put a 2 in front of the equation.

When working with a optimisation problem, we set the equation equal to zero.

$$
2 \Sigma a_1 - 2 \lambda_1a_1 = 0 \\ \text{The 2's go out, and we have:} \\ \Sigma a_1 = \lambda a_1
$$

Now we are at an interesting point of the equation. Since the covariance matrix multiplies with $a_1$ is the same as a scalar multiplied with $a_1$ we can quickly realise that $a_1$ is an eigenvector to the eigenvalue $\lambda_1$ of the covariance matrix $\Sigma$. 

This means, that whatever direction we choose to project on, it is going to be an eigenvector of the covariance matrix $\Sigma$, since this is the definition of an eigenvector.

Continuing with the calculation, we want to figure out which eigenvector to use?

$$
\text{Multiply with } a_1^T \text{ on both sides:} \\ a_1^T \Sigma a_1 = \lambda a_1a_1^T
$$

When we now look at this equation on the RHS, we see the first constraint that we made, that $a_1a_1^T$ should sum to one. Since we know this, the equation becomes:

$$
a_1^T \Sigma a_1 = \lambda_1 \cdot 1 \\ a_1^T \Sigma a_1 = \lambda_1
$$

And since $a_1^T \Sigma a_1$ was the thing we wanted to maximise in the first place, we know now, we that simply just have to maximise $\lambda_1$, which was the eigenvalue.

This means, we want the biggest eigenvalue if the covariance matrix, because that is going to give us the biggest value for the variance of the projected data.

### The next PC

PCA tells us that the next PC’s should be constructed using the same idea to maximise the variance, but additionally they should be perpendicular to all previous principle components. To account for this, we need a slight change in the math:

$$
\text{maximise }Var(Z_2) = Var(a_2^TX) = a_2^T \Sigma a_2 \\ s.t. \ \ a_2^Ta_2 = 1 \text{ should still sum to one } \\ s.ts \ \ a_2^Ta_1 = 0 \text{ and the product of the previous and the new PC} \\ \text{should sum to 0, because then they are perpendicular}
$$

Doing the lagrangian optimisation problem again with this, shows us that $a_2$ is an eigenvector to the eigenvalue $\lambda_2$ to $\Sigma$, and we simply choose the second largest eigenvalue to construct the second principle component. This pattern continues, with the third, forth and so on.

### Summary

- **So to find the first principle component of the data set, we are going to:**
  - Figure out all the different eigenvalues for the covariance matrix $\Sigma$.
  - Pick the biggest one.
  - Find out what is the eigenvector that matches up with that $\lambda$, that is gonna be $a_1$, and we want to normalise $a$ so it is a unit vector.

## Explained Variance

**We want to know:**  
*“How much of the information in a given data set is lost by projecting the observations onto the first few principal components? That is, how much of the variance in the data is not contained in the first few principal components?”*

So, we are interested in the proportion of variance explained (PVE).  
The total sample variance is:
$$
Var(X_1)+...+Var(X_p) = Var(a_1^TX) +...+Var(a_p^TX) = \lambda_1+...+\lambda_p
$$
We can talk about the proportion of the variance explained by the $k$th principle component as:  
$$
\frac{\lambda_k}{\lambda_1+...+\lambda_p}
$$

## Scaling Variables

Since PCA finds directions of high variance, it depends a lot on the scale of the variables. This means that high variance gets a high weight. So in order to account for this, it is crucial to scale the variables, before performing PCA.

Otherwise than the variables should be centred to have mean zero,  the **results** obtained when we perform PCA will also **depend** on whether the variables have been **individually scaled** (each multiplied by a different constant).

**If we perform PCA in unscaled variables,** consequently, the first principal component loading vector will have a very large loading, since a variable could have a very high variance compared to other variables.

## Deciding How Many PC’s to Use

n general, a $n \times p$ data matrix $\bold{X}$ has $min(n-1,p)$ distinct principal components. However, we usually are not interested in all of them; rather, we would like to use just the first few principal components in order to visualise or interpret the data.


📌 We would like the smallest number of principal components requires to get a **good** understanding of the data.
Unfortunately there is no simple answer to how many that would be.


**Scree Plot**  
You can decide on the number of PC by eyeballing a scree plot.


<div style="text-align: center;">
    <img src="/Screenshot_2022-11-14_at_16.35.11.png" alt="dd" width="500" height="350" style="text-align: center;" >
</div>

<span style="color: grey; font-style: italic;">
A scree plot depicting the proportion of variance explained by each of the four principal components in the "USArrests" data.
</span>

Here you eyeball it, and look for a point at which the proportion of variance explained by each subsequent principal component drops off. This drop is referred to as the **elbow** of the scree plot.

For instance, by inspection of **the figure**, one might conclude that a fair amount of variance is explained by the first two principal components, and that there is an elbow after the second component. After all, the third principal component explains less than ten percent of the variance in the data, and the fourth principal component explains less than half that and so is essentially worthless.