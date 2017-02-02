Bayesian Language
-----
I have studied Bayesian modeling several times since 2010, read a lot paper, couple of books, and some online classes. Yes, you can remember math terms, follow what they said in paper, sometimes even the equation derivations. But I never really understand what's going on behind those equations. That's the reason I have to repeat the classes, but stuck in a loop. Every time I learn it, I thought I got the idea, and over and over again. 

That's because there is a Bayesian language and way to think about data modeling, which is different with deterministic modeling such as Neural Networks. We should try to link every concept in Bayesian to Neural Networks. Because they are trying to solve the same problem (regression or classification) by slight different ways. Also, as we have seen in last section, the VAE can be expressed in a Neural network. People have done a lot of research to link them already, another example is the link between Gaussian Naive Bayes classifier and logistic regression. (from [Andrew Ng.](https://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf)).

OK, let's start step by step. As usual, notations first. Very often, I would ignore the notation part when reading any paper (because it is boring for sure). But please read it this time, not only because it provides us the  characters of the new language, it also helps us to clear some basic concept from high school probability. [Here's a refresher](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library) I found very useful for myself.

> **Notations**

> - Uppercase $$$X$$$ denotes a **random variable**. Different with deterministic variable, random variable does not have a fixed value, but several possible values with probabilities. 
> - Uppercase $$$P(X)$$$ denotes the probability distribution over that variable. We can say $$$P(X) \~ N(0,1)$$$, which means this random variable generates value under a standard normal distribution.
> - Lowercase $$$x \~ P(X)$$$ denotes a value $$$x$$$ sampled from the probability distribution $$$P(X)$$$ via some generative process.
> - Lowercase $$$p(X)$$$ is the density function of the distribution of $$$X$$$. It is a scalar function over the measure space $$$X$$$.
> - $$$p(X=x)$$$ (shorthand $$$p(x)$$$) denotes the density function evaluated at a particular value $$$x$$$.

Now, let's take a look at the first step. Normally, we are trying to model a dataset from a probability view. For example, we have an image of cat. The pixels in the image is our data (**observation** variable $$$X$$$ in probability view). We believe this observable variable is generated from a hidden (latent) variable $$$Z$$$, which can be a binary variable (cat or non-cat). We can draw this relationship via the following graph:
![image]({{ site.baseurl  }}/img/hidden_observation.png )

The edge drawn from $$$Z$$$ to $$$X$$$ relates the two variables together via the conditional distribution $$$P(X|Z)$$$. Now, it's important to jump out of the graph and conditional probability, think about the problem we try to solve, which is given the image, is this an image of cat or not? In the probability language, what's the conditional probability $$$P(Z|X)$$$? Even if we modeled the graph, what we got is the $$$P(X|Z)$$$, how can we get to the problem we are interested? **Bayesian** comes to play here.

$$p(Z|X)=\frac{p(X|Z)p(Z)}{p(X)}$$

Let's assume we can model the graph $$$p(X|Z)$$$ somehow. We can get the final answer if we got $$$p(Z)$$$ and $$$p(X)$$$. In **Bayesian Language**, we have some names for all those math terms. They are just names, but would help you to read paper and discuss with "experts".

> **Bayesian Language**

> - $$$p(Z|X)$$$ is the **posterior probability**. This is the most important term in Bayesian modeling, because this is the question we are interested. 
> - $$$p(X|Z)$$$ it the **likelihood**. It means given the hidden variable $$$Z$$$, how likely it generates observed images as we have seen in training data. Building this is building the graph. The famous term "maximum likelihood estimation" is one way to solve this. It tries to find the best hidden variable $$$Z$$$ to lead to good likelihood.
> - $$$p(Z)$$$ is the **prior probability**. This captures any prior information we know about $$$Z$$$ - for example, if we think that $$$\frac{1}{3}$$$ of all images in existence are of cats, then $$$p(Z=1)=\frac{1}{3}$$$ and $$$p(Z=0)=\frac{2}{3}$$$
> - $$$p(X)$$$ is called **model evidence** or **marginal likelihood**. The way to compute this is marginalizing the likelihood over hidden variable $$$Z$$$. 
> $$p(X) = \int{p(X|Z=z)p(Z=z)}dz$$
> - Marginalization is the bread and butter of Bayesian modeling, because this gives us the model uncertainty.

This is the Bayesian language. It's easy to follow, but too **abstract** to understand, right? Because everything here is probability, but not straight-forward equations we can code up. I'm agree, and feel the same. Now, let's visualize it through a simple example under **Naive Bayesian Classifier**.  

![image]({{ site.baseurl  }}/img/naive_bayes.png )

This is the structure graph of Naive Bayesian classifier. Very similar to the previous graph, but with one assumption, all the observations are conditional independent given the hidden variable. Let's say we have 3 observed binary variables $$$X_1, X_2 and X_3$$$ and one binary hidden variable $$$X, Z=\\{0,1\\}$$$. Given a dataset containing the observed variables and hidden variables value $$$<X, Z>$$$, how can we learn the graph and how to do the inference (solve the posterior probability) $$$p(Z=1|X_1=1,X_2=0,X_3=1)$$$?  

As we have shown above, in order to solve the posterior probability, we need to learn the likelihood $$$p(X|Z)$$$, the prior $$$p(Z)$$$ and the model evidence $$$p(X)$$$. 

It's easy to get the prior $$$p(Z=1)$$$, just estimate it from the training data 
$$p(Z=1) = \frac{\\#Z==1}{\\#Total}$$

Hard part is the likelihood, $$$p(X_1=x_1,X_2=x_2,X_3=x_3|Z=1)$$$, which is a conditional joint probability. Thanks to the independency assumption of Naive Bayes, we can write this likelihood like this:
$$p(X_1=x_1,X_2=x_2,X_3=x_3|Z=1) = p(X_1=x_1|Z=1)p(X_2=x_2|Z=1)p(X_3=x_3|Z=1)$$

Compute the conditional probability with one variable is easy:

$$p(X_i|Z)=\frac{P(X_i\cap Z)}{P(Z)}=\frac{\\#(X_i \\& Z)}{\\#(Z)}$$

The model evidence is just a integral of those posteriors.


Now, hope we have a clear picture how does Bayes model works. Keep in mind, the posterior is easy under the Naive Bayes assumptions, but hard ('nontrackable') in most cases. You can imagine it would be even harder to compute the model evidence. 

Until now, I guess you may have the same question as I have. The "hidden variable" is our target, which is observable in the training data. Many situations, the real "hidden variable" is the variables we do not even know in the training data, such as some edge features in the image. How can we define these kind of problem? How can we present them in the graph?

![image]({{ site.baseurl  }}/img/real_graph.png )

This is a graph defines more complicated and real life problem. Given the training inputs $$$X=\\{x_1,...,x_N\\}$$$ and their corresponding outputs $$$Y=\\{y_1,...,y_N\\}$$$, in **Bayesian (parametric) modeling**, we would like to find the parameters $$$\theta$$$ of a function $$$y=f^{\theta}(x)$$$ that are likely to have generated our outputs. In another word, what parameters are likely to have generated out data?

The **model forward (testing/inference)** is not the posterior probability anymore. Given a new input point $$$x'$$$ and the training data, we would like to infer what's the probability of corresponding value of $$$y$$$

$$p(y'|x', X, Y) = \int{p(y|x', \theta)p(\theta|X,Y)d\theta}$$

It can also be written as 

$$p(y'|x', X, Y) = \int{f_{\theta}(x')p(\theta|X,Y)d\theta}$$

We can see that is marginalizing likelihood over posterior. Also remember, in the Bayesian modeling, $$$\theta$$$ is not one best value, but a set of possible values with corresponding probabilities. Comparing with the "Bayesian Language" shown above, we need to slightly modify the language definition. 

> **Bayesian Language Update**

> - **Posterior Probability** $$$p(\theta|X,Y) = \frac{p(Y|X,\theta)p(\theta)}{p(Y|X)}$$$
> - **Likelihood** $$$p(Y|X,\theta)$$$
> - **Prior Probability** $$$p(\theta)$$$
> - **Model Evidence** $$$p(Y|X) = \int{p(Y|X,\theta)p(\theta)d\theta}$$$

The same as previous examples, the most important part is the posterior $$$p(\theta|X,Y)$$$. It cannot usually be evaluated analytically. Instead we seek some estimations such as MC based sampling method or by an approximating **variational distribution**

One more point I need to make here is, the output of Bayesian inference is not just a value, but an expectation value and uncertainty. If we deal with the regression problem, the inference can be express as Gaussian likelihood.

$$\mathbf{E}(y') = \int{f_{\theta}(x')p(\theta|X,Y)d\theta}$$ 
$$var(y') = \tau^{-1} I$$

If we deal with classification problem, the expectation is softmax likelihood. [Yarin Gal](http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html) described a way to extract prediction exception in a Bayesian view from Neural Network with dropout, which provides a good link between NN and Bayesian modeling. 


