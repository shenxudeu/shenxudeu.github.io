## Little Math on Stocastic Policy Gradients

We have a Neural Network generating a probability distribution of the output (eg two class discrete distribution), $p(x|w,I)$. The parameters govern the NN output is $w$, input of the probability distribution function is $I$, eg an image. (We used shorthand $p(x)$ to reduce clutter). $x$ could be $0$ or $1$. The distribution can be written explicitly as $Prob(x=1|w,I)$ and $Prob(x=0|w,I)$

Also, we have an environment reward function, $r(x)$ takes the action $x$ as input, and return a reward.

Taken from Policy Network definition, we are looking for a parameter set $w$, can generate good expected reward. As such, the learning goal(target) is maximize __reward expectation__, $E_{x\~p(x|w,I)}[r(x)]$
