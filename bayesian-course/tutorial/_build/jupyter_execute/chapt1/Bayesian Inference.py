# Bayesian Inference  <a class="anchor" id="guideBayesian"></a> 

Taken Bayesian theorem as follow:

$$ 	 \underbrace{P(\theta|X)}_{posterior} \; \alpha  \; \underbrace{P(X|\theta)}_{likelihood} \cdot \underbrace{P(\theta)}_{prior} $$ 


Bayesian Inference has three steps.

Step 1. [Prior] Choose a PDF to model your parameter $\theta$, aka the prior distribution P($\theta$). This is your best guess about parameters before seeing the data X.

Step 2. [Likelihood] Choose a PDF for P(X|$\theta$). Basically you are modeling how the data X will look like given the parameter $\theta$.

Step 3. [Posterior] Calculate the posterior distribution P($\theta$|X) and pick the $\theta$ that has the highest P($\theta$|X).
And the posterior becomes the new prior. Repeat step 3 as you get more data.

## Credible intervals

As the Bayesian inference returns a distribution of possible effect values (the posterior), 
the credible interval is just the range containing a particular percentage of probable values. Credible intervals are not unique on a posterior distribution. Methods for defining a suitable credible interval include:

- Choosing the narrowest interval, which for a unimodal distribution will involve choosing those values of highest probability density including the mode (the maximum a posteriori). This is sometimes called the highest posterior density interval (HPDI).

- Choosing the interval where the probability of being below the interval is as likely as being above it. This interval will include the median. This is sometimes called the equal-tailed interval (ETI).

- Assuming that the mean exists, choosing the interval for which the mean is the central point.


```{figure} credible_interval.png
---
height: 300px
name: log-figure
---
```