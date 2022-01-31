# Bayes's theorem  <a class="anchor" id="bayes"></a>

Bayes' theorem is stated mathematically as the following equation: 
 
$$ \textrm{P(A|B)} = \frac{\textrm{P(B|A) P(A)}}{\textrm{P(B)}} =\frac{\textrm{P(B|A) P(A)} }{\textrm{P(B|A) P(A) +P(B|A}^c\textrm{) P(A}^c) }  \tag{1}$$ 

where $P(B)>0$ and also.

The simplification of eq.1 no taking account factor parameters

$$ 	 \underbrace{\textrm{P(A|B)}}_{posterior} \; \alpha  \; \underbrace{\textrm{P(B|A)}}_{likelihood} \cdot \underbrace{\textrm{P(A)}}_{prior} $$ 

> Note:  $P(B) = \textrm{P(B|A) P(A) +P(B|A}^c\textrm{) P(A}^c)$


## Likelihood function

The likelihood function (often simply called the likelihood) measures the goodness of fit
of a statistical model to a sample of data for given values of the unknown parameters. 

It is formed from the joint probability distribution of the sample, 
but viewed and used as a function of the parameters only, thus treating the random 
variables as fixed at the observed values

```{admonition} Example

In an hospital there are 400 patients affected by heart attacks of which 72 died after 1 month and 328 were released. What is the mortality rate?

We can say: each patient come from a Bernoulli distribution: $Y_i\sim  B(\theta) \Rightarrow P(Y_i =1) = \theta$. The Bernoulli distribution is controlled by the parameter $\theta$.

The probability density function for all set is 
$$
P(Y =y|\theta)=P(Y_1 =y_1, ... ,Y_n =y_n |\theta) = \textrm{are independent } = 
P(Y_1 =y_1 |\theta) .. P(Y_n =y_n |\theta) =  \prod_{i=1}^{n} P( y_i |\theta) =\prod_{i=1}^{n} \theta^{y_i} (1-\theta)^{1-y_i}
$$

If we think in terms of $\theta$ given y, the likelihood function is:

$$
\mathbb{L} (\theta | y) = \prod_{i=1}^{n} \theta^{y_i} (1-\theta)^{1-y_i}
$$

```

> Warning: Likelihood is not a probability function!


## Maximum Likelihood estimation
Maximum likelihood estimation (MLE) is a method of estimating the parameters of a probability distribution
by maximizing a likelihood function, so that under the assumed statistical model the observed data is most probable. 


```{admonition} Example 1
 Maximum Likelihood Estimator $\Rightarrow$ $\hat{\theta} = \textrm{argmax} \mathbb{L}(\theta|\hat{y_i})$

Let's move to log:

$$l(\theta)= \textrm{log} \, \mathbb{L}(\theta|\hat{y_i}) \Rightarrow$$


$$l(\theta)= \textrm{log} \prod_{i=1}^{n} \theta^{y_i} (1-\theta)^{1-y_i} =
\sum \textrm{log} \theta^{y_i} (1-\theta)^{1-^{y_i}} = \sum \left[ y_i \textrm{log} \theta + (1-y_i) \, \textrm{log}(1-\theta) \right] =

\left( \sum  y_i \right) \textrm{log} \theta +   \left(\sum 1-y_i \right) \, \textrm{log}(1-\theta)  
$$

Maximizing: 

$$ \frac{d l}{d \theta} =0 \Rightarrow \frac{1}{\theta} \sum y_i - \frac{1}{1- \theta} \sum (1-y_i)=0 $$

$$ \hat{\theta} = \frac{\sum y_i}{n} $$

This is the value of $\theta$ that maximizes the likelihood for the bernoulli distribution.


> In our precedent example $ \hat{p} = \frac{72}{400}=0.18$

```


```{admonition} MLE Exponential distribution
$X \sim Exp(\lambda) \Rightarrow$ the density function for independent events 

$$
f(X| \lambda) = \prod_{i=1}^{n} \lambda e^{-\lambda x_i} = \lambda^n  e^{-\lambda \sum x_i}
$$

$$
\mathbb{L} (\lambda | x) = \lambda^n  e^{-\lambda \sum x_i}
$$

Let's do the logaritm:

$$
l(\lambda)=\textrm{n log} (\lambda) - \lambda \sum x_i
$$
Doing the derivate:

$$ l^{'}(\lambda)=\frac{n}{\lambda} -  \sum x_i=0 \Rightarrow$$
The MLE is

$$ \hat{\lambda}=\frac{n}{\sum x_i}= \frac{1}{x}$$

which corresponds to the inverse of the mean of $x$. 

```

```{admonition} Uniform distribution
$X_i \sim U[0,\theta] \Rightarrow$ 

$$
f(X| \theta) = \prod_{i=1}^{n} \frac{1}{\theta} I_{\{ 0 \leq x_i \leq \theta \} } 
$$

$$
\mathbb{L} (\theta | x) = \theta^{-n}  I_{\{ 0 \leq min x_i \leq max x_i \leq \theta \} } 
$$

$$
l^{'} (\theta ) = -n \theta^{-(n+1)}  I_{\{ 0 \leq min x_i \leq max x_i \leq \theta \} } 
$$

$$\hat{\theta}=max x_i $$
```

## Example Coin: Frequentist versus Bayesian approaches
We launch a coin, that is supposed to give 70% of time heads. 
Now you launch it 5 times and you obtain 2 heads. Is it fair or unfair coin?

### Frequentist approach
$\theta = \{ \textrm{fair, loaded} \} $

$X \sim Bin(5,?), \\ f(X|\theta) \Rightarrow$ 

$$\binom{5}{x} (0.5)^5  \tag{if $\theta$ is fair} $$ 

$$\binom{5}{x} (0.7)^x \, (0.3)^{5-x} \tag{if $\theta$ is loaded} $$


$$\Rightarrow  \binom{5}{x} (0.5)^5  I_{  \{ \theta = \textrm{fair}  \} }  +\binom{5}{x} (0.7)^x \, (0.3)^{5-x} I_{ \{ \theta = \textrm{loaded}  \} }$$

If we put X =2, we have 2 observed tails $f(\theta|X=2) \Rightarrow$


$$ 0.3125  \tag{if $\theta$ is fair} $$ 

$$ 0.1323 \tag{if $\theta$ is loaded} $$

Note that the previous values are the value for the parameter $\hat{\theta}$ in the two scenarios $\theta$ is fair or $\theta$ is loaded: those are scalar point estimates! No info here about the probability that $\theta$ is fair or loaded, just his value. From those values, we obtain which one maximize the likelihood.

MLE is obtained with $\hat{\theta}$ is fair. 
The answer to the previous question is that given the current data, is most likely that the coin is fair. But how sure are we? This is not easy to answer with the frequentistic approach to the question $P(\theta =fair|X=2)=?$ or in other words, what is the probability that $\theta$ is fair given what we observed?

> We made the assumption on the probability of the distribution following the binomial distribution. 


### Bayesian approach
Let's assum a prior $P(loaded)=0.6 \Rightarrow$ 

$$
f(\theta|X)=\frac{f(X|\theta) f(\theta)}{\sum f(X|\theta) f(\theta)}= 
\frac{ \binom{5}{x}\left[(0.5)^5 (0.4) I_{\theta=\textrm{fair}} + (0.7)^x (0.3)^{5-x} (0.6) I_{\theta=\textrm{loaded}} \right] }
{ \binom{5}{x}\left[(0.5)^5 (0.4) + (0.7)^x (0.3)^{5-x} (0.6) \right]} \Rightarrow
$$

$$
f(\theta|X=2)=0.612 I_{\theta=\textrm{fair}} + 0.388 I_{\theta=\textrm{loaded}}
$$
Now we can answer to the question:
$P(\theta =loaded|X=2)=0.388$

What happen with different prior?
$P(loaded)=0.5 \Rightarrow P(\theta =fair|X=2)=0.297 $
$P(loaded)=0.9 \Rightarrow P(\theta =fair|X=2)=0.792 $

At this stage of knowledge, we can see that the posteriori is dependent on the priori and there is a subjective knowledge of the priori. But the assumption on the model are clear.  Moreover we can build confidence interval named credible interval.