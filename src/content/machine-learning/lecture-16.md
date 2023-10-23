---
title: 'Lecture 16'
description: 'Gradient descent, Stochastic GD, Batch GD, Mini Batch GD'
pubDate: 'Oct 23 2023'
heroImage: '/lec16_ml.png'
---

********************Readings:******************** Hands on chapter 4, page 118-128

# Gradient Descent

## Gradients in Multivariate Calculus


📌 *The “gradient” in a multivariate calculus case, is the directions of the steepest increase. So the gradient of a function gives you the direction of steepest ascent, so basically which direction you should step to increase the function most quickly (thats why you always take the opposite step. 
The length of that vector gives an indication of how steep that steepest slope is.*



The general idea of Gradient Descent is to tweak parameters from a cost function in order to minimise the loss. 

- It measures the local gradient of the error function with regard to the parameter vector $\theta$, and it goes in the direction of descending gradient. (you go downhill).
- Once the gradient is zero, you have reached a minimum.

Training a model with GD means you are searching for a combination of model parameters that minimises a cost function. It is a search in the models parameter space. So the more parameters a model has, the more dimensions this space has, and the harder the search is. 

### Random Initialisation

You start by filling $\theta$ with random values. Then you improve it gradually, taking one baby step at a time, each step attempting to decrease the cost function, until the algorithm converges to a minimum. 

### Learning Rate

An important parameter of GD is the size of the steps, also called the learning rate. $\alpha$ 

**Too small:** if the LR is too small, then the algorithm will have to go through many iterations to converge, which will take a long time.

**Too large:** if the LR is too large, you might risk jumping over the minimum, and skipping it, and then you end on the other side of the minimum.

Not all cost functions look nice with only 1 local minimum. So if there is more than 1 minimum (maybe several local, and the best one is the global) then when we have **random initialisation** we might end up at a local minimum!

## Linear Regression, MSE cost function

Fortunately, the MSE cost function is a convex function. This means there is no local minima and only 1 global one. 

<aside>
📌 Important: When using GD you should ensure that all features have a similar scale, or else it will take much longer to converge.

</aside>

- Pick a learning rate $\alpha$
- Randomly initialise parameters fx. $\beta_0 = 0, \beta_1 = 0$
- Repeat until L is very small (or not changing)

<div style="text-align: center;">
    <img src="/Screenshot_2023-10-23_at_14.26.08.png" alt="dd" width="700" height="400">
</div>

**Be aware:** you need to update $\beta_0$ and $\beta_1$ simultaneously. So, for i in # no. of iterations (repeat until convergence)

$$
\beta_{new0} := \beta_0 - \alpha \frac{\partial}{\partial \beta_0} L(\beta_0,\beta_1) \\ \beta_{new1} := \beta_1 - \alpha \frac{\partial}{\partial \beta_1} L(\beta_0,\beta_1)  \\ \beta_0 = \beta_{new0} \\ \beta_1 = \beta_{new1}
$$

## Logistic Regression

<div style="text-align: center;">
    <img src="/Screenshot_2023-10-23_at_14.30.31.png" alt="dd" width="900" height="350">
</div>

### Explanation of proof


<div style="text-align: center;">
    <img src="/Screenshot_2023-10-23_at_14.31.45.png" alt="dd" width="900" height="350">
</div>

******************************Cost function****************************** in which we wish to minimise:

$$
L(\beta) = - \frac{1}{n} \sum_{i=1}^n y_i \log p(x_i) + (1-y_i)\log(1-p(x_i))
$$

# Batch Gradient Descent (BGD)

## Implementation, Gradients and Partial Derivatives

To implement Gradient Descent, you need to compute the gradient of the cost function with regard to each model parameter $\theta_j$. In other words, you need to calculate how much the cost function will change if you change $\theta_j$ just a little bit. This is called a partial derivative.

**📌** *Remember, partial derivative is just like “What is the slope of the mountain under my feet if i face a specific direction?”.*

**The partial derivative of the cost function with regard to parameter $\theta_j$ noted $\frac{\partial \ MSE(\theta)}{\partial \theta_j}$**

$$
\frac{\partial}{\partial \theta_j} MSE (\theta) = \frac{2}{m} \sum_{i=1}^m (\theta \bold{x}^{(i)} - y^{(i)}) x_j^{(i)}
$$

The gradient vector contains all the partial derivatives of the cost function (on for each model parameter). 

**Gradient Vector of the MSE Cost Function**

$$
\nabla_{\theta} MSE(\theta) = \begin{bmatrix}
           \frac{\partial}{\partial \theta_0}MSE(\theta) \\  \\
\frac{\partial}{\partial \theta_1}MSE(\theta) \\
           \vdots \\
           \frac{\partial}{\partial \theta_n}MSE(\theta)
         \end{bmatrix} = \frac{2}{m} X (X \theta -y)
$$

Notice that this formula involves calculations over the full training set X, at each Gradient Descent step! This is why the algorithm is called **Batch Gradient Descent**: it uses the whole batch of training data at every step. Another name, that would be better is **Full Gradient Descent.**

Once you have the gradient vector, which points uphill, just go in the opposite direction to go downhill.

### Learning Rate

This is where the **learning rate** comes in, $\eta$: multiply the gradient vector by $\eta$ to determine the size of the downhill step. 

**Gradient Descent Step:**

$$
\theta^{\text{(next step)}} = \theta - \eta  \nabla_{\theta} MSE(\theta)
$$

**When the Learning Rate is Too Low**

When the learning rate is too low, you take very small steps down the mountain. The algorithm will eventually reach the minimum, but it will take a long time. 

**When the Learning Rate is Too Large**

The algorithm diverges, jumping all over the place and eventually getting further and further away from the solution. You could risk jumping over the minima and never find a solution. 

📌  *To find a good learning rate, you can use grid search.* 

*Grid Search uses a different combination of all the specified hyper-parameters and their values and calculates the performance for each combination and selects the best value for the hyper-parameters.*

### Number of Iterations

You may wonder how to set the number of iterations. If it is **too low**, you will still be far away from the optimal solution when the algorithm stops; but if it is **too high,** you will waste time while the model parameters do not change anymore. 

**A simple solution** is to set a very large number of iterations but to interrupt the algorithm when the gradient vector becomes tiny—that is, when its norm becomes smaller than a tiny number $\epsilon$ (called the tolerance).

# Stochastic Gradient Descent (SGD)

Batch Gradient Descent uses the whole training set to compute the gradients at every step, and that is computationally a problem when the training set i large.

Though, SGD is the other extreme: It picks a random training instance at every step, and computes the gradient based only on that single instance. This makes the algorithm a lot faster, and it makes it possible to train on very large data sets.  

Though, it is much less regular than **batch gradient descent.** Instead of gently decreasing until it reaches the minimum (like **BGD**) the cost function will bounce up and down, and only decrease by average. This jumping up and down actually has an advantage. This can help the algorithm jump out of a local minima.

📌 *SGD has a better chance of finding the global minima, than BGD.*

We don’t use the whole data set, we only take 1 data point at a time

- Take a sample
- Feed it to the current model and calculate the gradient descent there
- Use the gradient descent to update the model $\beta_j:= \beta_j-\alpha(p_{\beta} (x_i)-y_i) \cdot x_{ij}$
- Repeat steps 1-3

We repeat by rounds of n iterations, these steps are called **epoch.**

Way faster than batch BD, and good for escaping local minima, but it may fluctuate too much in the parameter space and never catch a minimum

### Disadvantages with SGD

Though the randomness is good to escape the local minima, it comes with a downside. Due to the randomness, the algorithm can never settle at the minimum. But a **solution to this** is to gradually reduce the learning rate. Starting large to escape local minima, and getting smaller and smaller to allowing it to settle. 

**If the learning rate is is reduced too quickly** it may get stuck on the local minima. **If the learning rate is reduced to slowly** you may jump around in the global minima for a long time.

📌  *The training instances must be independent and identically distributed (IID) to ensure that the parameters get pulled toward the global optimum, on average.*

## Mini-Batch Gradient Descent

Mini-batch GD computes the gradients on small random sets of instances called mini-batches. 

**The main advantage** of Mini-batch GD over Stochastic GD is that you can get a performance boost from hardware optimisation of matrix operations.

The parameter space is less erratic tan SGD, and as a result the Mini-BGD will end up walking around a bit closer to the minimum (than SGD), but i may be harder to escape the local minima. 

If you take a look at all the Gradient algorithms (Batch, Stochastic and Mini) They all end up near the minimum, but Batch GD’s path actually stops at the minimum, while both Stochastic GD and Mini-batch GD continue to walk around. However, **be aware** that even though it sounds like Batch Gradient Descent is better here, it takes a very long time at each step, and the two other algorithms can also reach a minimum if you set the hyper-parameters correctly. 

- May escape local minima
- Less fluctuation than stochastic GD, especially with fairly large mini-batches.