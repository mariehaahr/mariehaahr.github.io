---
title: 'Lecture 17, 18 and 19'
description: 'Artificial Neural Networks & Concolutional Neural Networks'
pubDate: 'Nov 03 2023'
heroImage: '/lec171819_ml.png'
---

#### Table of contents
- [**What is a Neural Network**](#what-is-a-neural-network)
  - [Perceptron/Neuron aka. a Linear Model](#perceptronneuron-aka-a-linear-model)
    - [Multi-Layer Perceptrons](#multi-layer-perceptrons)
    - [Stacking Neurons Horizontally: Neural Network](#stacking-neurons-horizontally-neural-network)
  - [Activation Functions - Adding Non-Linearity](#activation-functions---adding-non-linearity)
    - [Sigmoid](#sigmoid)
    - [Tahn](#tahn)
    - [ReLU (Rectified Linear Unit)](#relu-rectified-linear-unit)
    - [Leaky ReLU](#leaky-relu)
  - [FFNN Architecture](#ffnn-architecture)
  - [Making Predictions - Forward Pass](#making-predictions---forward-pass)
  - [Learning](#learning)
    - [Back-Propagation](#back-propagation)
  - [Vanishing and Exploding Gradients](#vanishing-and-exploding-gradients)
    - [Gradient Clipping](#gradient-clipping)
  - [Bias-Variance Trade-Off in Terms of Neural Networks](#bias-variance-trade-off-in-terms-of-neural-networks)
    - [Early stopping](#early-stopping)
  - [Regularisation](#regularisation)
- [**Convolutional Neural Networks**](#convolutional-neural-networks)
  - [Convolutional Layer - The Kernel](#convolutional-layer---the-kernel)
  - [Pooling Layers](#pooling-layers)
    - [Benefits of pooling](#benefits-of-pooling)
    - [Benefits of convolution + pooling](#benefits-of-convolution--pooling)

# **What is a Neural Network**

Feed-forward neural networks are simply an extension of linear regression and logistic regression.

Neural Networks are supervised, discriminative Machine Learning models. They are the cornerstone of deep learning, and they are capable of solving both regression and classification. 

They are inspired by the giant network of biological neurons found in the brain. The same way as the neurons in the NN are connected by some weights, the neurons in the brain are connected via some synaptic terminals.

**Actually** (under some minor conditions) a neural network with one hidden layer containing a sufficient but finite number of neurons with certain non-linear activation functions can approximate any continuous function to a reasonable accuracy. 

## Perceptron/Neuron aka. a Linear Model

The perceptron is simple just a **mathematical function**, which **computes the linear combination** of a sequence of $p$ inputs $x_1, x_2,...,x_p$ and $p$ weights connected to the corresponding inputs $w_1,w_2,...,w_p$. Normally in linear regression we can add a bias, so we do that as well:

$$
Perceptron(x_1,x_2,...,x_p) = w_0 + \sum_i^p w_ix_i = w_0+w_1x_1+w_2x_2+...+w_px_p = Xw+\beta_0
$$

Just like in a linear model.

But, there is another step of the perceptron. We also wrap a so-called *activation function* around the linear combination. A non-linear function, in order to add some non-linearity to the model, otherwise it would not be able to approximate any continuous function. 

### Multi-Layer Perceptrons

If we let multiple neurons compute an output on the same set of inputs, we get a neural network. In this way, we are getting *k* outputs of the *k* linear models (each taking the linear combination of the same *p* features, but with their own set of weights). This is the exact same thing that multinomial logistic regression does, so again, this model is not as complicated as you would think.

So we still have: $perceptrons(\bold{X})= \sigma(\bold{Xw})$  in one step of a hidden layer in the network.

### Stacking Neurons Horizontally: Neural Network

This is what makes the neural networks so powerful. The idea is to use the output from the stacked perceptrons (a) and use those as inputs for the next layers perceptrons. 

So a single layer $L_1$ transforms an input feature matrix $X \in \R^{n \times p+1}$ into a matrix of activations $A_1 \in \R^{n \times k_1}$. Now the second layer $L_2$  takes the activation matrix $A_1$ from the previous layer as input and produces another output $A_2$ and so on. 

This is simple just composing a lot of functions, mathematically.

$$
NN(X) = \sigma_3(\sigma_2(\sigma_1(Xw_1)w_2)w_3)
$$

## Activation Functions - Adding Non-Linearity

These are crucial, since they add complexity to the model. If there were no activation functions, composing a linear function with a linear function is, well, still linear. So no matter how many layers, it would still be linear. 

### Sigmoid

A familiar non-linear function, but it is quite computationally expensive because of the exponential terms, and it also suffers from vanishing gradients. So the gradients are so small that they vanish. This is why scaling is important.

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$


<div style="text-align: center;">
    <img src="/Screenshot_2022-10-29_at_15.12.34.png" alt="dd" width="300" height="350" style="text-align: center;" >
</div>


### Tahn

The Tahn activation function is quite similar to the sigmoid, the range is simply just between $[-1,1]$, which also makes the derivatives more steep. Suffers from the same as the sigmois activation function.

$$
\sigma(x)=\frac{2}{1+e^{-2x}}-1 = 2 \sigma(2x)-1
$$


<div style="text-align: center;">
    <img src="/first.png" alt="dd" width="300" height="350" style="text-align: center;" >
</div>


### ReLU (Rectified Linear Unit)

Computationally light function, because it is simply either 0 or a regular linear function. For negative inputs, the gradients are zero, which means the networks stops learning (dying ReLu problem). In reality, often enough activation are positive. It is often used as default activation for hidden layers.

$$
\sigma(q) = max(0,q)
$$


<div style="text-align: center;">
    <img src="/Screenshot_2022-10-29_at_15.14.52.png" alt="dd" width="300" height="350" style="text-align: center;" >
</div>


### Leaky ReLU

Problems with unstable gradients were in part due to a poor choice of activation function. 

The ReLU activation function is not perfect. It suffers from a problem know as the ******************************dying ReLu’s:****************************** during training, some neurons effectively “die”, meaning they stop outputting anything other than 0. 

********************Solution:******************** Leaky ReLU: This function is defined as:

$$
\text{LeakyReLU}_\alpha(z)=max(\alpha z,z)
$$

<div style="text-align: center;">
    <img src="/Screenshot_2022-11-03_at_19.30.39.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>


*Leaky ReLU: like ReLU, but with a small slope for negative values*

## FFNN Architecture

Now let’s put everything together. The number of layers, the number of neurons per layer and the different activation functions are all hyper parameters that you choose before training the model. 

*How do you know how to choose these hyperparameters?*  
The easiest way is making a models that is “too complex” and simple use early stopping to prevent overfitting. That is we break the training when a parameter update no longer improves the validation data. This is a common regularisation technique for deep learning models, to avoid poor performance.

## Making Predictions - Forward Pass

As explained earlier, it is simply passing the inputs through all of the given activation functions. 

## Learning

*How do we know what all these parameters are supposed to be, and how do we train the model?*  
Since the architecture of a neural network can be so vast, there is no specific closed form formula to calculate the millions of weights you can have in a model. Therefore we use gradient decent to train the model. But it is a little more complicated than that.

After we have initialised all of the weights randomly, (both weights and biases) we need to update them. We predict with the model (forward pass) and compute some loss (cost) of the model. These loss functions vary, they depend on whether you have a regression or classification task, and whether the answer is binary or multi-class. 

**************Recall:************** the chain rule is a formula that expresses the derivative of the composition of two differentiable functions in terms of their derivatives.  
When you take the derivative of a function inside a function: $y=f(g(x))$

$\frac{\partial y}{\partial w} = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{ \partial w}$

### Back-Propagation

In order to optimise the weights, so we reach a minimum on the cost-function (and hence yield optimal predictions) we use the back-propagation algorithm. 

Shortly explained, back-propagation gradually updates the weights and biases by using gradient descent. It calculates the gradient (the derivative) of the cost function associated with a given state with respect to the weights. The weights are then updated according to the calculated gradient to reduce the cost in the next iteration. Here we can use the help of the chain-rule from calculus, which says:

$$
f(g(x))' = f'(g(x)) \cdot g'(x) \\ y = f(u) \ \ \ \ \ u=g(x) \\ \frac{\partial y}{\partial x} = \frac{\partial y}{\partial u} \cdot \frac{\partial u}{\partial x}
$$


<div style="text-align: center;">
    <img src="/Screenshot_2022-11-01_at_11.29.52.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

This rule is used a lot through a back-propagation, since the output is an input inside a function, inside a function and so on. 

In other words, it can find out how each connection weight and each bias term should be tweaked in order to reduce the error, because we can calculate the derivative of the loss function with respect to all weights in the model. Once it has these gradients, it just performs a regular Gradient Descent step, and the whole process is repeated until the network converges to the solution.

You can choose this step to be done with the whole data set (batch gradient descent) or in batches of data (mini batch) or with a single random point in the data set (stochastic gradient descent). When the whole data set is run through and the weights are updated, one epoch has successfully been run. Here it is your own choice: batch size, number of epochs, and the learning rate you use in the gradient step.  
**Gradient step:**

$$
w_{new} := w_{old}-\alpha \nabla f
$$

## Vanishing and Exploding Gradients

Unfortunately, gradients often get smaller and smaller as the algorithm progresses down to the lower layers. As a result, the Gradient Descent update leaves the lower layers’ connection weights virtually unchanged, and training never converges to a good solution. We call this the **vanishing gradients problem.** 

The opposite can happen: the gradients can grow bigger and bigger until layers get insanely large weight updates and the algorithm diverges. **This is the exploding gradients problem**, which surfaces in recurrent neural networks.

### Gradient Clipping

Another popular technique to mitigate the **exploding gradients** problem is to clip the gradients during back-propagation so that they never exceed some threshold. 

## Bias-Variance Trade-Off in Terms of Neural Networks

There is a big chance of high variance on the model when we have many parameters (as we do in ANN). When we have an overfitted model, one way to overcome this is to use more training data.  But if we don’t have that: ********************************Under-training:******************************** Don’t go all the way, trying to get 0 loss function-error, stop a little before. This is called ******************************early stopping.******************************

### Early stopping

When the performance on the validation set is getting worse, we stop the training.

## Regularisation

Similar to other ML methods, we can apply a variety of regularisation methods in the learning of ANNs. They play an important role specially when the training set is relatively small.

- **Weight decay:**
    - (l2 regularisation, lasso) (This means we introduce a small amount of ********Bias******** into how the new line is fit to the data, in the exchange of a lower ********************Variance.********************) It tries to push the magnitude of the weights towards 0. Avoid getting to close to the minimum loss when training, to prevent overfitting.
- **Dropout**
    - It injects some noise into the training process, adds some diversity. Kind of an ensemble method.
- **Early stopping**
- **Data Augmentation**
    - Make some changes in the training data. make some noise (for example if the data was images, you change the images a little bit).

# **Convolutional Neural Networks**

When working with normal feed-forward neural networks you lose the structure of the image, since you need to flatten the image array. This is where convolutional neural networks really shine, since the structure of the image is preserved. To some degree they mimic how we, as humans classify images which is by recognising specific patterns and features anywhere in the image that differentiate each distinct object class.

<div style="text-align: center;">
    <img src="/second.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>


We have a ************************************feature extraction************************************ part and a ****************************classification**************************** part (which is just a feed forward NN).

## Convolutional Layer - The Kernel

A convolution layer is made up of a large number of convolution filters, each of which is a template that determines whether a particular local feature is present in an image. A convolution operation basically amounts to repeatedly multiplying matrix elements and then adding the results. 


<div style="text-align: center;">
    <img src="/third.png" alt="dd" width="400" height="350" style="text-align: center;" >
</div>


The convolved feature can be different sizes, depending on how you want it. You can have the same size and the input image if you add some padding to the images (0’s).

The role of the Convolutional layer is to reduce the images into a form which is easier to process, without losing features which are critical for getting a good prediction. ****************Filters**************** are nothing but a **********************************feature detectors.********************************** They are **location invariant**. It can detect patterns in any location of the image.

## Pooling Layers

A pooling layer provides a way to condense a large image into a smaller summary image.

**********************Max pooling********************** summarises each non-overlapping $2 \times 2$ block of pixels in an image using the maximum value in the block. 

This operation provides some *********location invariance,********* meaning that as long as there is a large value in one of the four pixels in the block, the whole block registers as a large value in the reduced image.


<div style="text-align: center;">
    <img src="/Screenshot_2022-11-06_at_19.48.02.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>


### Benefits of pooling

- Reduces dimension & computation
- It reduces overfitting, as there are less parameters.
- Model becomes tolerant towards variations and distortions. Because you are capturing the main feature of the image when doing max pooling.

### Benefits of convolution + pooling

- Connections sparsity reduces overfitting. Meaning that not every node is connected with every other node, like in a fully connected NN. The filter we apply to the image makes sure that we are not effecting the whole image, but only the part that is relevant.
- Convolution + pooling gives invariant feature detection.
- Parameter sharing.