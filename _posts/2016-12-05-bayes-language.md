Bayesian Language
-----
I have tried to conquer Bayesian modeling several times since 2010 (in the middle of a PhD term which I eventually dropped out of); read a lot paper, couple of books, and took some online classes. Yes, you can remember math terms, you may follow what they say in the paper when you are reading it, you may even be able to derive the equations just as they do. But what's hard is to really understand what's going on behind those equations, without which you are bound to forget what you think you know after a certain period. Then you might need to repeat the learning process, however only to find you stuck in a loop.

That's because there is a Bayesian language and a way to think about relationships in data, which is different from deterministic modeling such as Neural Networks. Natually, there should be a way to link every concept in Bayesian to Neural Networks. After all they are trying to solve the same problem (regression or classification) by slight different ways. Also, as we have already seen in the previous post, VAE can be expressed as a Neural Network. People have already done a lot of research to link the two, another example being the linkage of Gaussian Naive Bayes classifier and logistic regression. (from [Andrew Ng.](https://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf)).

OK, let's start one step at a time. As usual, notations first. Very often, I would ignore the notation part when reading any paper (because it is boring for sure). But please read it this time, not only because it supplies us with the building characters of the new language, it also helps us recollect some basic concepts from high school probability. [Here's a refresher](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library) I found very useful myself.

> **Notations**

> - Uppercase $$X$$ denotes a **random variable**. Different with deterministic variable, random variable does not have a fixed value, but several possible values with probabilities. 
> - Uppercase $$P(X)$$ denotes the probability distribution over that variable. We can say $$P(X) ~ N(0,1)$$, which means this random variable generates value under a standard normal distribution.
> - Lowercase $$x ~ P(X)$$ denotes a value $$x$$ sampled from the probability distribution $$P(X)$$ via some generative process.
> - Lowercase $$p(X)$$ is the density function of the distribution of $$X$$. It is a scalar function over the measure space $$X$$.
> - $$p(X=x)$$ (shorthand $$p(x)$$) denotes the density function evaluated at a particular value $$x$$.


Now, let's take a look at the first step. Normally, we are trying to model a dataset from a probability view. For example, we have an image of cat. The pixels in the image is our data (**observation** variable $$X$$ in probability view). We believe this observable variable is generated from a hidden (latent) variable $$Z$$, which can be a binary variable (cat or non-cat). We can draw this relationship via the following graph:
![image]({{ site.baseurl  }}/img/hidden_observation.png )

The edge drawn from $$Z$$ to $$X$$ relates the two variables together via the conditional distribution \\(P(X \| Z)\\). Now, it's important to jump out of the graph and conditional probability, think about the problem we try to solve, which is given the image, is this an image of cat or not? In the probability language, what's the conditional probability \\(P(Z\|X)\\)? Even if we modeled the graph, what we got is the \\(P(X\|Z)\\), how can we get to the problem we are interested? **Bayesian** comes to play here.

$$
p(Z|X)=\frac{p(X|Z)p(Z)}{p(X)}
$$

Let's assume we can model the graph \\(p(X\|Z)\\) somehow. We can get the final answer if we got $$p(Z)$$ and $$p(X)$$. In **Bayesian Language**, we have some names for all those math terms. They are just names, but would help you to read paper and discuss with "experts".

> **Bayesian Language**
> - This \\(p(Z|X)\\) is the **posterior probability**. This is the most important term in Bayesian modeling, because this is the question we are interested. 
> - This \\(p(X|Z)\\) is the **likelihood**. It means given the hidden variable $$Z$$, how likely it generates observed images as we have seen in training data. Building this is building the graph. The famous term "maximum likelihood estimation" is one way to solve this. It tries to find the best hidden variable $$Z$$ to lead to good likelihood.
> - $$p(Z)$$ is the **prior probability**. This captures any prior information we know about $$Z$$ - for example, if we think that $$\frac{1}{3}$$ of all images in existence are of cats, then \\(p(Z=1)=\frac{1}{3}\\) and \\(p(Z=0)=\frac{2}{3}\\)
> - $$p(X)$$ is called **model evidence** or **marginal likelihood**. The way to compute this is marginalizing the likelihood over hidden variable $$Z$$. 
>  - \\(p(X) = \int{p(X|Z=z)p(Z=z)}dz\\)
> - Marginalization is the bread and butter of Bayesian modeling, because this gives us the model uncertainty.


