Gaussian Process (GP) is Easy to Understand
==================================

INSERT A PRETTY GP CHART.

In the beginning 
----

This is a follow-up blog of "Variantial AntoEncoder", in which I tried to explain VAE in a Neural network way and Bayesian machine learning. As we know, VAE is a unsupervised learning method, now, I try to explain the most important supervised learning method in Bayesian machine learning, called Gaussian Process. 

I have tried many many times to understand GP in different ways, reading paper, books, video, and even wrote my own [GP software in Theano](https://github.com/shenxudeu/gp_theano). I found most materials starting with a GP definition such as "A Gaussian process defines a distribution over functions has multivariate Gaussian distribution", which I cannot understand in a very long time. What is the distribution of functions? What does it looks like? They also talks about the called "covariance function". How does those functions present the "covariance"? In our normal person's mind, "covariance" is not defined as some sort of wired function, but just an [average of squared residuals as shown in Wikipedia](https://en.wikipedia.org/wiki/Variance)

In this blog, I try to make sense all those definitions. We do need some __linear algebra__ knowledge; if you are not familar with it, I find [Goodfellow's book chapter](http://www.deeplearningbook.org/contents/linear_algebra.html) very helpful.

This blog explains Gaussian Process in Bayesian language. If you are not familar with it, please read my previous blog [Bayesian Language](http://shenxudeu.github.io/2016/12/05/bayes-language/). 


Linear Regression
-----
I know everyone is familar with linear regression, which is the reason I start it with linear regression. Actually, Gaussian Process is rooted from it; it is just an expansion of it in a Bayesian view. As you may know, I, as a financial trader, deal with very noise data without many useful sharing (eg. publications). I personally think it is very important to start with some extremely simply algorithms, and gradually increase the complexity. For example, I normally start from a linear regression/classification or given simplier. If you want to utilize some more inputs in the data, you may want to generalize it into multi-variate regression. If you find linear relationship is not enough to catch your phenomenon, you can even generalize it to other functions such as Gaussian Process. If you want to catch the time-depdendencies on the data, you can generalize the linear system into a state-space systems. Once you find your system is too flexible in some sense, you can try to put some constrains (regularizations). In another words, we build an algorithm with "+" components and "-" components (regularization), and find a balance point for you specific problem. Anyway, my point is we can understand GP better by expanding it from a simple linear regression.

#### Linear Regression with Point Estimation (Maximum Likelihood)

Given the training data \\(X\in R^{n,m}\\) (\\(n\\) samples with \\(m\\) features) and \\(Y\in R^{n,1}\\), the linear regression is looking for an "optimal" weight \\(W, W \in R^{m,1}\\) which fits the data well by the following linear model.

$$
Y = X^TW + \epsilon 
$$

where \\(\epsilon\\) is a random noise follows a Gaussian distribution,

$$
\epsilon \sim \mathcal{N}(0,\sigma^2)
$$

In math, the term "likelihood" measures how well the model fits the data given a set of weights, which is the probability

$$
P(Y\mid X,w) = \prod_{i=1}^{n}p(y_i\mid x_i,w  )=\prod_{i=1}^{n}\frac{1}{\sqrt{2\pi}\sigma_n}exp(-\frac{(y_i-x_i^Tw)^2}{2\sigma_n^2})
$$

Through some simple linear algebra, we can get

$$
P(Y\mid X,w) \sim \mathcal{N}(X^Tw, \sigma^2I)
$$

Simply taking the \\(\log\\) likelihood, we can get the following equation

$$
\log P(Y\mid X,w) = -m\log{\sigma_n} - \frac{n}{2}\log(2\pi) - \sum_{i=1}^{m}\frac{\parallel y_{*i}-y_i)\parallel ^2}{2\sigma_n^2}
$$ 

Maxmum likelihood is simply saying find me the \\(w\\) can get largest \\(\log\\) likelihood.

$$
argmax_{w}-\sum_{i=1}^{m}\frac{\parallel y_{*i}-y_i)\parallel ^2}{2\sigma_n^2}
$$

The is the same __minimize mean square error!__ The solution is linear least square:

$$
w = (X^TX)^{-1}X^Ty
$$ 

#### Linear Regression with Bayesian View

In a "point estimation" view, once we found the best weight to fit the data, story ends here. The final prediction given a new data \\(X_*\\) is 

$$
Y_*=X_*^TW
$$

But in a Bayesian view, we think all possible weights in our prior believe has a chance. The final prediction is the __weighted average__ of all possible outputs.

$$
Y_* = \frac{1}{N}\sum_{j=1}^{N}\theta_j X_*^TW^j
$$

This is also the __sum rule__ in probability:

$$
P(y_*\mid x_*, X, y) = \int P(y_*\mid x_*, w)P(w \mid X, y) dw 
$$

As we have shown above, the first term is easy, which is a Gaussian distribution,

$$
P(y_*\mid x_*,w) \sim \mathcal{N}(x_*^Tw, \sigma^2I)
$$

The second term is called __posterior distribution__. This gives us the weights of each possible \\(w\\). Think about this way, before we observe any data, we guess all possible \\(w\\) has equal chance. After we observed some data points, we changed our belief of the prior, and put some __probability mass__ on some possible weights \\(w\\) which fits the data better. The __changed distribution__ is called __posterior distribution__.

But how to compute this posterior distribution after observing training data? __Bayesian Rule__ gives us the answer.

$$
P(w|y,X) = \frac{P(y\mid X, w)P(w)}{P(y\mid X)}
$$

Through some linear algebra, we can get

$$
P(w|X, y) \sim \mathcal{N}(\frac{1}{\sigma_n^2}A^{-1}Xy, A^{-1})
$$

where 

$$
A=\sigma_n^{-2}XX^T + \Sigma_p^{-1}
$$

where our prior of $w$ is

$$
w \sim \mathcal{N}(0, \Sigma_p^2)
$$

This is called __Bayesian Linear Regression__. It gives us the predicted mean value as well as variance. The m, ean of posterior distribution looks similar to the point estimation, but with some extra terms. Actually, the extra term is proven the same as a __\\(L_2\\) regularization__ in __ridge regression__. In another words, the Bayesian method gives us the __regularization for free__. That's why we can see many people saying Bayesian learning algorithms does NOT overfit data. 

As you can see, the concept is simple, just a linear model expressed in Bayesian language. In order to get the equations, we just need to follow some rules in basic linear algebra. Now, I gonna show you it is also very easy to code and use it.
 
Let's generate some random 1-D random data. 

```python
# Generate data
X = np.random.uniform(-3., 3., (10,1))
X = np.hstack((X,np.ones((10,1))))
Y = np.dot(X,np.expand_dims(np.array([1.5,0.5]),1))+ np.random.randn(10,1) * 0.5
plt.scatter(X[:,0:1], Y)
```

The data looks like this.

![]({{ site.baseurl  }}/img/1d_linear_data.png)

After we set our prior distribution \\(\mu_0, \Sigma_p\\) and noise variance \\(\sigma_n\\) (shown in the following block), we can compute the posterior distributions from data. 

```python
# Set-up prior
sigma_n = 0.5
mu0 = np.zeros((2,1))
mu0[0,0] = 0.0
Sigma_p = np.identity(1)
```

The prior we give to \\(w\\) is a normal distribution centered at \\(0\\) with \\(\sigma=1\\), it looks like this.
![]({{ site.baseurl  }}/img/linear_prior.png)

Computing the posterior is simplying saying we put some __probability mass__ on some \\(w\\) values to fit the data. We can do that in 3 lines of code.

```python
# Compute Posterior Distribution
A = sigma_n**(-2) * np.dot(X.T, X) + np.identity(1)
posterior_mu = sigma_n ** (-2) * np.dot(np.dot(np.linalg.inv(A), X.T),Y) +  np.dot(np.dot(np.linalg.inv(A), A), mu0)
posterior_var = np.linalg.inv(A)
```

As shown in the following chart, our learned posterior distribution put a lot of probability mass around \\(1.5\\), which is our true \\(w\\) used to generate our random data.
![]({{ site.baseurl  }}/img/linear_posterior.png)

If we use the learned posterior probabilities to make prediction, we can get the predicted \\(y\\) values along with our predictive variances. The code is also very straight-forward.

```python
# Make predictions on test data
X_test = np.expand_dims(np.arange(-4,4,0.01),1)
X_test = np.hstack((X_test,np.ones((len(X_test),1))))
y_test_mu = sigma_n**(-2) * np.dot(np.dot(np.linalg.inv(A), X.T),Y) + np.dot(np.dot(np.linalg.inv(A), A), mu0)
y_test_mu = np.dot(X_test, y_test_mu)
y_test_var = np.dot(np.dot(X_test, np.linalg.inv(A)),X_test.T)
```

We can also plot the predicted values with \\(+/- 3 \sigma\\) lines.
![]({{ site.baseurl  }}/img/linear_predictions.png)





















 







