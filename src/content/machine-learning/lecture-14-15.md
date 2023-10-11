---
title: 'Lecture 14 and 15'
description: ''
pubDate: 'Oct 11 2023'
heroImage: '/lec7_na.png'
---

**Readings:** ISL_wR Chapter 9, Hands-on Chapter 5, and StatQuest videos!

#### Table of contents
- [The Main Idea Behind Support Vector Machines](#the-main-idea-behind-support-vector-machines)
  - [Hyperplanes and Support Vectors](#hyperplanes-and-support-vectors)
- [**Hard Margin** (lecture 14)](#hard-margin-lecture-14)
- [9. Support Vector Machines](#9-support-vector-machines)
  - [Maximal Margin Classifier](#maximal-margin-classifier)
    - [What Is a Hyperplane?](#what-is-a-hyperplane)
  - [Classification Using a Separating Hyperplane](#classification-using-a-separating-hyperplane)
  - [The Maximal Margin Classifier](#the-maximal-margin-classifier)
    - [Construction of the Maximal Marginal Classifier](#construction-of-the-maximal-marginal-classifier)
    - [The Non-separable Case](#the-non-separable-case)
- [**Soft Margin** (lecture 15)](#soft-margin-lecture-15)
- [Non-Linear Support Vector Machines](#non-linear-support-vector-machines)
  - [Soft-Margin Support Vector Machines](#soft-margin-support-vector-machines)
    - [C](#c)
    - [$\\xi$](#xi)
    - [What of our function is better separated by a non-linear function? (for example a polynomial function)](#what-of-our-function-is-better-separated-by-a-non-linear-function-for-example-a-polynomial-function)
  - [Non-Linear Support Vector Machines](#non-linear-support-vector-machines-1)
  - [Kernel Trick- What if the Data Points are not Linearly Separable](#kernel-trick--what-if-the-data-points-are-not-linearly-separable)
  - [Kernels](#kernels)
    - [The Kernel Trick](#the-kernel-trick)
    - [Popular Kernels](#popular-kernels)
    - [The Dot Product](#the-dot-product)
- [Support Vector Machines YT, StatQuest](#support-vector-machines-yt-statquest)
    - [Soft Margin](#soft-margin)
    - [What if we have tons of overlap in our data?](#what-if-we-have-tons-of-overlap-in-our-data)
  - [Support Vector Machines, YT](#support-vector-machines-yt)
    - [The main idea behind Support Vector Machines](#the-main-idea-behind-support-vector-machines-1)
    - [How do we decide how to transform our data?](#how-do-we-decide-how-to-transform-our-data)
- [**Extra**: The Decision Rule - Proof](#extra-the-decision-rule---proof)
  - [Lagrange Multipliers](#lagrange-multipliers)

# The Main Idea Behind Support Vector Machines

It is a discriminative, supervised machine learning model.  
The goal of a Support Vector Machine is to find a hyperplane in p-dimensional space, that distinctly classifies the data points.   
But there are many different hyperplanes that can separate the two classes, so the goal is to find the a plane that has the maximum margin, meaning that there is the widest space between the two classes, the widest street. This reinforces that unseen data can be classified with more confidence. 

## Hyperplanes and Support Vectors

The hyperplane is the decision boundary that help classify the data points. The dimension og the hyperplane depends on the number of features. 2 features is a line, 3 is a plane and above that is a hyperplane, which is hard to visualise.  
The reason why it is called a support vector machine, is because the vectors (or data points) that lie on the margin of the street (or support the margin of the street) are the ones that we use to build the classifier. If those data points move, the decision boundary changes. 

# **Hard Margin** (lecture 14)

# 9. Support Vector Machines

SVM is a ***supervised learning algorithm*** which can be used for classification and regression problems as support vector classification (SVC) and support vector regression (SVR). The **Support Vector Machine** is a generalisation of a simple and intuitive classifier called *maximal marginal classifier.* Though MMC is a simple and elegant classifier, i cannot be used for many datasets, since it requires that the data can be separated by a straight line. This is why we have the *support vector classifier* which is an extension of this, and can be applied in broader ranges. Further, we will introduce the *support vector machine* which an even further extension in order to accommodate (imødekomme) non-linear class boundaries. 

## Maximal Margin Classifier

In order to understand *support vector machines* we need to understand the maximal margin classifier

### What Is a Hyperplane?

For instance, in 2 dimensions, a hyperplane is a flat one-dimensional subspace aka. just a line. In 3 dimensions it is a normal plane.  
But in $p > 3$ dimensions, it can be hard to visualise what a hyperplane is. But to simplify, it is still a $(p-1)$-dimensional flat subspace.

The mathematical definition of a hyperplane in $p$-dimensional setting:

$$
\beta_0 + \beta_1X_1 + \beta_2X_2+...+\beta_pX_p = 0 \ \ \ \ \ (9.2)
$$

for parameters $\beta_0,...,\beta_p$. Here, we mean that any $X=(X_1,...,X_p)^T$ in a $p$-dimensional space (a vector of length $p$) for which $(9.2)$ holds is a point $X$ on the hyperplane.

When equation $(9.2) >0$, then $X$ lies on one side of the hyperplane, and when $(9.2) <0$ then $X$ lies on the other side of the hyperplane.  
So, we can think of the hyperplane as dividing $p$-dimensional space into two halves. You can easily determine which side of the hyperplane the point lies, by calculating the sign of $(9.2)$ (the LHS). 

## Classification Using a Separating Hyperplane

Suppose we have a $n \times p$ data matrix $X$ that consists of $n$ training observations in $p$-dimensional space:

$$
x_1 =     \begin{bmatrix}
           x_{11} \\
           \vdots \\
           x_{1p}
         \end{bmatrix} ,..., x_n = \begin{bmatrix}
           x_{n1} \\
           \vdots \\
           x_{np}
         \end{bmatrix} 

$$

and these observations falls into either class 1 or class -1 ($y_1,...,y_n \in \{-1,1\}$). We also have a test observation $x^* = (x_1^* ...x_p^*)^T$. 

We will now see a new approach that is based upon the concept of a separating hyperplane. Suppose that it is possible ti construct a hyperplane that separates the training observations perfectly according to their classes. So class assigning is very simple. A test observation is assigned a class depending on which side of the hyperplane it is located, so if :

$$
f(x^*) = \beta_0 + \beta_1x_1^* + \beta_2x_2^*+...+\beta_px_p^* \ \ \ \ \ 
$$

is a positive number, then we assign to class 1, is the number is negative, we assign to class -1.

We can also use the magnitude (the size) of $f(x^*)$ to determine how confident we are about our class assignment. If $f(x^*)$ is far from 0, then $x^*$ lies far from the hyperplane, and we are very sure of our class assignment. On the other hand, if $f(x^*)$ is very close to 0, it means that the point $x^*$ lies close to the hyperplane, and we are not so sure.

## The Maximal Margin Classifier

In order to construct a classifier based on a separating hyperplane, we need to decide which line it should be. There are infinite possible options when drawing a straight line, here we see just 3 ways to separate the classes:

<div style="text-align: center;">
    <img src="/Screenshot_2022-10-12_at_09.35.31.png" alt="dd" width="500" height="350">
</div>

A natural choice for this is the *maximal margin hyperplane* (also known as the *optimal separating hyperplane*) which is the separating hyperplane that is **the farthest from the training observations**. 

So, if we compute the perpendicular (vinkelret) distance from each training observation to the hyperplane the smallest such distance is the minimal distance from the observations to the hyperplane, and is known as the margin. The maximal margin hyperplane is the separating hyperplane for which the margin is largest—that is, it is **the hyperplane that has the farthest minimum distance to the training observations.**


<div style="text-align: center;">
    <img src="/Untitled_14.png" alt="dd" width="500" height="350">
</div>

In other words, the maximal margin hyperplane represent the mid-line of the widest “bar” that we can insert between the two classes before it touches the points.  
Even though this classifier is often successful, it leads to overfitting when $p$ is large. 

📌 **The basic intuition to develop over here is that more the farther SV points, from the hyperplane, more is the probability of correctly classifying the points in their respective region or classes.**



<div style="text-align: center;">
    <img src="/Screenshot_2022-10-12_at_09.42.19.png" alt="dd" width="500" height="350">
</div>

On this figure, we see the maximal margin hyperplane, and the “bar” that surrounds it. There are 3 points (with the arrows) that touch this “bar”. These 3 distances from the point ti the hyperplane are exactly equal (the observations are equidistant). 

These 3 observations are known as **support vectors**, since they are vectors in $p$-dimensional space, and “support” the hyperplane. By supporting we mean that if we move these supporting points a tiny bit, the *maximal marginal hyperplane* would change. 


📌 **Interestingly, the *maximal marginal hyperplane* depends directly on the support vectors, but not on the other observations. So it only depends on a small subset of the observations, and this is important to keep in mind.**


### Construction of the Maximal Marginal Classifier

Now we wish to construct the *maximal marginal hyperplane* based on a set of $n$ training observaitons $x_1,...,x_n \in \R^p$ and associated class labels $y_1,...,y_n \in \{-1,1\}$. That is, the solution to this optimisation problem:

$$
\text{maximise}(\beta_0, \beta_1,...,\beta_p,M) \ \ M \ \ \ \ \ \ (9.9) \\ \text{subject to } \sum_{j=1}^p \beta_j^2=1 \ \ \ \ \ \ \ \ \ \ (9.10) \\ \text{where } y_i(\beta_0+\beta_1x_{i1}+ \beta_2x_{i2}+...+\beta_px_{ip}) \ge M \ \forall \ i=1,...,n \ \ \ \ \ (9.11)
$$

The last constraint $(9.11)$ guarantees that each observation will be on the correct side of the hyperplane, provided that $M$ is positive. 

One can show that with this constraint the perpendicular distance from the $i$th observation to the hyperplane is given by:

$$
y_i(\beta_0+\beta_1x_{i1}+ \beta_2x_{i2}+...+\beta_px_{ip})
$$

Therefore, the constraints $(9.10)$ and $(9.11)$ ensure that each observation is on the correct side of the hyperplane and at least a distance $M$ from the hyperplane. 

### The Non-separable Case

This is a very good method if **a separating hyperplane exists**. In many cases, no separating hyperplane actually exists, meaning that optimisation problems $(9.9)-(9.11)$ has no solution with $M>0$.

Here **Support Vector Classifiers** can extend the concept of a separating hyperplane in order to develop a hyperplane that *almost* separates the classes, using a so-called *soft-margin*.

We need an update so that our function may skip few outliers and be able to classify almost linearly separable points. For this reason, we introduce a new ***Slack variable (*** 
$\xi$) which is called *Xi.* So now we rewrite the previous equation $(9.11)$:

$$
y_i(\beta_0+\beta_1x_{i1}+ \beta_2x_{i2}+...+\beta_px_{ip}) \ge 1 - \xi_i
$$

# **Soft Margin** (lecture 15)

# Non-Linear Support Vector Machines

- Support vector machines for non linearly separable data (we cant draw a straight line that separates the data). We can separate non-linearly separable data without overfitting, using a ********soft-margin SVM.******** For this we introduce the slack term $\xi$ to the objective function.
- To produce non-linear support vector machines, we make use of the kernel function which maps out data to a features space where it becomes more likely to be linearly separable.

## Soft-Margin Support Vector Machines

In soft margin support vector machine, we allow data points to lie within the margin, and some points to lie in the wrong side (misclassification). 

**************************************************************The equation of the hyperplane:**************************************************************

$$
w^Tx+b=0
$$

******************************The equations of the margins:******************************

$$
w^Tx+b=1 \\ w^Tx+b=-1
$$

********************************************************If the data points belong to the classes in the figure above:********************************************************

$$
\text{Belongs in class A} = w^Tx_i+b \ge1 - \xi_i  \\\text{Belongs in class B}= w^Tx_i+b \le-1 + \xi_i
$$

### C

Here, **`C`** is a hyperparameter that decides the trade-off between maximizing the margin and minimizing the mistakes. When **`C`**
 is small, classification mistakes are given less importance and focus is more on maximizing the margin, whereas when **`C`**
 is large, the focus is more on avoiding misclassification at the expense of keeping the margin small.

### $\xi$

At this point, (with the variable C) we should note, however, that not all mistakes are equal. Data points that are far away on the wrong side of the decision boundary should incur more penalty as compared to the ones that are closer. This is where we introduce the slack variable $\xi$.

The idea is: for every data point **`x_i`**, we introduce a slack variable **`ξ_i`**. The value of **`ξ_i`** is the distance of **`x_i`** from the *corresponding class’s margin* if **`x_i`** is on the wrong side of the margin, otherwise zero. Thus the points that are far away from the margin on the wrong side would get more penalty.

With this idea, each data point **`x_i`**  needs to satisfy the following constraint:  $\text{st.  }y_i(w \cdot x_i + b) \ge 1-\xi_i$. There the LHS is like the “confidence” of the classification. So if the confidence score is $\ge$ 1, it means the classifier has classified the point correctly. However, if the cinfidence score is < 1, it means the the classifier did not classify the point correctly and incurring a **************************linear penalty************************** of $\xi_i$. 

### What of our function is better separated by a non-linear function? (for example a polynomial function)

This is where we use the ************************************kernel functions.************************************ 

## Non-Linear Support Vector Machines

## Kernel Trick- What if the Data Points are not Linearly Separable

**To produce non-linear support vector machines, we make use of the kernel function which maps out data to a features space where it becomes more likely to be linearly separable**. This is, we transform everything into a higher dimension and find a hyperplane in that dimension that can separate the data.

Here the **dual representation** looks like this instead:

$$
\max_\alpha \sum_i \alpha_i - \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j (\Phi(\vec x_i)\cdot \Phi(\vec x_j)) \\ s.t \ \ \ 0 \le \alpha \le C \ \ \ and \ \ \ \sum_i \alpha_iy_i =0
$$

An the **********************************decision boundary********************************** looks like:

$$
h(x)=\sum_i \alpha_i y_i (\Phi(\vec x_i)\cdot \Phi(\vec x_j))+b
$$

## Kernels

Kernel trick is used in SVM to bridge linearity and non-linearity.  


<div style="text-align: center;">
    <img src="/Screenshot_2022-10-24_at_15.44.39.png" alt="dd" width="500" height="350">
</div>

As you can see in the figure, we map a 2-dimensional space (left) to a 3-dimensional space, where it is easy to divide the points into 2 regions, with the decision surface. But when we have more and more dimensions, it becomes more and more expensive to compute, and that’s when we use the ****************************kernel trick.**************************** 

********************In essence******************** a kernel function simplifies our computations when we have many dimensions. Lets say we have 2 data points in 3 dimensions:

$x=(x_1,x_2,x_3)^T$ and $y=(y_1,y_2,y_3)^T$, and we want to map that into 9 dimensions. The following calculations would take a lot of time, so we follow this kernel function $k(x,y)$ instead, we reach the same result within the 3-dimensional space by calculating the dot product of x -transpose and y.

$$
k(x,y) = (x^Ty)^2 = (x_1y_1 + x_2y_2 + x_3y_3)^2 = \sum_{i,j=1}^3 x_ix_jy_iy_j
$$

So for training we now have the dual representation:

$$
\max_\alpha \sum_i \alpha_i - \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j K(\vec x_i,\vec x_j) \\ s.t \ \ \ 0 \le \alpha \le C \ \ \ and \ \ \ \sum_i \alpha_iy_i =0
$$

An the **********************************decision boundary********************************** looks like:

$$
h(x)=\sum_i \alpha_i y_i K(\vec x_i,\vec x_j)+b
$$

### The Kernel Trick

For some tranformations $\Phi$ there is a kernel function $K$ that can take any pair of input vectors and compute the dot product of their transformed version directly in the original feature space. 

This means we can transform any feature vector into any infinite feature space without any extra computational power. 

### Popular Kernels

- Linear Kernel $K(x_i, x_j) = x_i \cdot x_j$
- Polynomial Kernel $K(x_i, x_j) = (x_i \cdot x_j +1)^b$, where $b$ is a hyperparameter
- RBF (Radial basis function) kernel, Gaussian kernel, which transfers the input space to a space with infinite dimensions
    
    $$
    K(x_i, x_j) = e^{-\frac{||x_i-x_j||^2}{2 \sigma^2}}
    $$
    
    Where $\sigma$ is a hyperparameter
    

### The Dot Product

Quick example, recap from linear algebra 

$$
x_1 =  \begin{bmatrix} 1 \\ 2  \\ 7\end{bmatrix} \ \ \ x_2 =\begin{bmatrix} 3 \\ 2  \\ 4\end{bmatrix} \\x_1 \cdot x_2 = 1 \times 3 + 2 \times 2 + 7 \times 4 = 35
$$

# Support Vector Machines YT, StatQuest

Instead of maximal margin classifier, we will use something different, when we have outliers. So we have a low bias because we take all the training data into account, and place a decision boundary that perfectly classifies all data points, but we have a high variance when we need to classify new data points, because the decision boundary depended on the outlier

### Soft Margin

First step to solve this is to allow misclassifications. If we allow that, we would introduce some bias in exchange for some reduction in variance. 

📌 When we allow misclassifications we call the threshold **Soft Margin.**


But how do we know which soft margin is best, which points to misclassify?

To answer this, we use cross-validation to determine how many misclassifications and observations to allow inside the **********************soft margin********************** to get the best classification. 

This is called using a **********************************************soft vector classifier********************************************** since in comes from the fact that the observations on the edge of the “bar” is called ********************************support vectors******************************** and lies within the **************************soft margin.************************** 

### What if we have tons of overlap in our data?

No matter where we put our decision boundary / hyperplane boundary, we would have a lot of misclassifications. What do we do in this situation? Do we have something that is better than a maximal margin classifier and a support vector classifier? WE USE **********************************************SUPPORT VECTOR MACHINES**********************************************

## Support Vector Machines, YT

An example: Here we add a y-axis on the points and square the observations. (we transform the data with ****************kernel functions****************)


<div style="text-align: center;">
    <img src="/Untitled1_14.png" alt="dd" width="500" height="350">
</div>

We can now draw a ******************************Support Vector Classifier****************************** that separates the data points better than before. 

And the **************************************************support vector classifier************************************************** can be used to classify new observations. 

### The main idea behind Support Vector Machines

1. We start with data in a relatively low dimension. 
2. Move the data into a higher dimension
3. Find a **support vector classifier** that separates the higher dimensional data into 2 groups. 

### How do we decide how to transform our data?

Support vector machines use **********************************kernel functions********************************** to systematically find ********support vector classifiers******** in higher dimensions. 

# **Extra**: The Decision Rule - Proof

So the question is, how are we going to make a decision rule, that uses this decision boundary? We have to imagine we have got a vector of any length, constrained to be perpendicular to the street. Then we also have some unknown sample we would want to classify.

The decision rule is now a follows:

$$
\text{If }\vec w \cdot \vec x +b \ge0 \text{ then x is + sample, else }-
$$

We need to set some constraints before we begin. We enforce the condition that the point get classified correctly. 

- If $y_i = 1$ then the points need to be above the line, so: $\vec w \cdot \vec x_+ +b \ge 1$
- If $y_i = -1$ then the points need to be above the line, so: $\vec w \cdot \vec x_- -b \le -1$

These two equations can be written together in a compact form, since the product of $y_i$ and the two quantities are both bigger than or equal to one.

$$
y_i(\vec w \cdot \vec x + b) \ge 1
$$

We can rewrite this into:

$$
y_i(\vec w \cdot \vec x + b) -1 =0
$$

**********************************************************with all this into place, the question is, how do we maximise the margin?**********************************************************

For the points lying on the margin, the support vectors. Now if we draw two support vectors on each side of the street, then what is the width of the street?

If we take the difference of those two vectors: $x_+ - x_-$

Know if a I had a unit normal vector, i could simply just just the vector with he difference between $x_+$ and $x_-$, and I would have the width of the street. That was why we introduced the $\vec w$ in the first place. So to get the width of the street:

$$
width = (\vec x_+ - \vec x_-) \cdot \frac{\vec w}{||\vec w||}
$$

We just divide $\vec w$ by the magnitude of itself, and it becomes a unit-vector.

If we rewrite this equation, looking at the constraint we had before $y_i(\vec w \cdot \vec x + b) -1 =0$. If we have a positive example, then $y_i$ is +1. Then is we isolate $y_i \cdot \vec w$, we get:

$$
1\cdot(\vec w \cdot \vec x + b) -1 =0 \\ \rightarrow (\vec w \cdot \vec x +b) -1 =0 \\ \rightarrow \vec w \cdot \vec x = b-1
$$

And the same if $y_i$  is -1.

$$
-1\cdot(\vec w \cdot \vec x + b) -1 =0 \\ \rightarrow (-\vec w \cdot \vec x -b) -1 =0 \\ \rightarrow \vec w \cdot \vec x = b+1
$$

Know we can rewrite the width equation from before into:

$$
\vec x_+ = \frac{1-b}{\vec w} \ \ \ and \ \ \ \vec x_- = \frac{1+b}{\vec w}\\width = (\vec x_+ - \vec x_-) \cdot \frac{\vec w}{||\vec w||} \\ \\\text{So we replace the new expressions in the width formula} \\ width = \left( \frac{1-b}{\vec w} - \frac{1+b}{\vec w} \right) \cdot \frac{\vec w}{||\vec w||} \\width = \frac{2}{||w||}
$$

So this is what we want to maximise. 

$$
max \left (\frac{2}{||\vec w||}\right) \\ \text{Just dropping the constant:} \\ max \left (\frac{1}{||\vec w||}\right) \\ \text{which is the same as minimising this:} \\ min \left (||\vec w||\right) \\ \text{which is the same as:} \\ min \left (\frac{1}{2}||\vec w||^2\right)
$$

And we do that for mathematical convenience. 

## Lagrange Multipliers

**Know we have a minimising problem with a constraint.**

**We wish to minimise**:  $min \left (\frac{1}{2}||\vec w||^2\right)$ with the constraint that: $y_i(\vec w \cdot \vec x + b) -1 =0$

**and also the constraints that**:

$a>0$

$y_i(wx_i+b) \ge 0$

$\alpha_i[y_i(wx_i+b)-1]=0$

Because of the Karush-Kuhn-Tucker conditions (KKT). 

Either $y_i(wx_i+b)-1=0$ when this condition is actually true, and that means that $\alpha_i > 0$, and the samples are sitting on the margin = ******************support vectors.******************

Or $\alpha_i =0$ and the samples have no contribution to the decision boundary.

So we have the equation:

$$
L(w,b,\alpha) = \frac{1}{2}||\vec w||^2 - \sum \alpha_i [y_i(\vec w \cdot \vec x+b)-1]
$$

We start off by finding the partial derivatives with respect to $w$ and $b$ of the Lagrangian function:

$$
\frac{\partial L}{\partial b} = - \sum_i \alpha_i y_i  \rightarrow \sum_i \alpha_i y_i =0 \text{ this is something we can use later on} \\ \frac{\partial L}{\partial w} = \vec w - \sum \alpha_i y_i \vec x_i \rightarrow \text{set equal to 0, and solve for w} \rightarrow  \vec w = \sum \alpha_i y_i \vec x_i
$$

Now that we have another expression for $\vec w$, we replace that in the Lagrangian formula:

$$
L(w,b,\alpha)= \frac{1}{2}(\sum_i \alpha_i y_i \vec x_i) \cdot (\sum_j \alpha_j y_j \vec x_j) - (\sum_i \alpha_i y_i \vec x_i)\cdot (\sum_j \alpha_j y_j \vec x_j) - \sum_i \alpha_i y_i b + \sum_i \alpha_i \\ \text{since b is just a constant: } b \cdot \sum_i \alpha_i y_i \text{ and we know that is 0:} \\ L(w,b,\alpha)= \frac{1}{2}(\sum_i \alpha_i y_i \vec x_i) \cdot (\sum_j \alpha_j y_j \vec x_j) - (\sum_i \alpha_i y_i \vec x_i)\cdot (\sum_j \alpha_j y_j \vec x_j)  + \sum_i \alpha_i \\ \text{Now since we know that the half of x minus itself is the same as minus x:} \\ L(w,b,\alpha) = \sum_i \alpha_i - \frac{1}{2}(\sum_i \alpha_i y_i \vec x_i)(\sum_j \alpha_j y_j \vec x_j) \\ \text{write that out and it becomes:} \\ L(w,b,\alpha) = \sum_i \alpha_i - \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j \vec x_i \cdot \vec x_j
$$

An interesting thing about this Langrangian, is that, what this optimisation depends ONLY on the dot product of pairs of samples. This representation is also know as the dual representation. 


<div style="text-align: center;">
    <img src="/Untitled2_14.png" alt="dd" width="500" height="350">
</div>


Know we plug the expression for $\vec w$ into the decision rule also: 

**The decision function then looks like:**

$$
\sum_i \alpha_i y_i \vec x_i \cdot \vec u + b \ge 0 \ \ \text{then } +
$$

Again, the total dependence is on the dot product of the sample vectors (and in this case the unknown vector). So the decision boundary (the prediction of new samples) actually only depend on the support vectors, which is why SVMs are so convenient.