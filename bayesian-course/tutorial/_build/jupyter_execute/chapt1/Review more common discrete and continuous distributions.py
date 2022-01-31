# Discrete and continuous Distributions  <a class="anchor" id="distributions"></a>  

## Discrete Distributions <a class="anchor" id="section_1_1"></a>
We will review the main distributions that we will use in the following chapters.

### Bernoulli (discrete)
$X \sim B(p)$ $\Rightarrow$ 

$$P(X=1) = p  \tag{Probability of success} $$ 

$$P(X=1) = 1-p  \tag{Probability of failure} $$

The compact way to write the function of all possible outcomes, namely the density function, is 

$$f(X=x|p)=f(x|p)=p^x (1-p)^{(1-x)} I_{x \in \{ 0,1 \} }$$

Expected value

$$E[X] = \sum_X xP(X=x)=1 \, (p)+ 0 \, (1-p) =p $$
The variance is

$$Var(X)=p \, (1-p)$$



```{admonition} **Examples in ecommerce**


- Probability that one buyer has to contact one seller

- Probability that one user has to click in one banner

- Probability that one user is going to do a churn


```

Generalization of Bernoulli when we have n repeated trial is the next distribution.


#### Binomial 
The Binonial is the sum of $n$ independent Bernoulli.

$X \sim Bin(n,p)$

$$P(X=x|p)=f(X|p)= \binom{n}{x} p^x \, (1-p)^{n-x}$$

where $\binom{n}{x} = \frac{n!}{x!(n-x)!}$ for $x \in \{0,1,..,n \}$.
The expected value and the variance are:

$$E[X] = np$$

$$Var(X)=np \, (1-p)$$ 


```{admonition} Examples in ecommerce

- Churn rate for one marketplaces in apps

- Return rate

- CTRs

```



## Review Continuous Distributions <a class="anchor" id="section_1_1"></a>

### Uniform 

$X \sim U [0,1]$ with $f(X)$ equals to 1 if $x \in [0,1]$, 0 otherwise :$f(X)= I_{x \in \{ 0,1 \} }$

The expected value is:

$$E[X] = \int_{-\infty}^{\infty} x f(x) \,dx =  \int_{0}^{1} x \cdot 1 \,dx =\int_{0}^{1/2}  x\,dx = 1/2$$

and

$$Var(X)= \frac{1}{12}$$ 


$\textbf{Generalized expression for uniform}$
$X \sim U [\theta_1,\theta_2]$

$f(X|\theta_1,\theta_2)= \frac{1}{\theta_1 - \theta_2}  I_{ \{ \theta_1 \leq x \leq \theta_2 \} }$

where

$$E[X] = \frac{1}{2}(\theta_1+\theta_2)$$

and 

$$V[X] = \frac{1}{12}(\theta_2-\theta_1)^2$$

```{admonition} Examples in ecommerce

- Demand value for a new product 

```

### Normal

$X \sim N(\mu, \sigma^2)$

$$f(X|\mu, \sigma^2)= \frac{1}{ \sqrt{2 \Pi \sigma^2 }} e^{- \frac{1}{2 \sigma^2} (x - \mu)^2}  $$

$$E[X] = \mu $$

$$Var(X)= \sigma^2$$


```{admonition} Examples in ecommerce

- Daily average number of clicks per user

- Daily average ad replies per user

```

### Exponential

$X \sim Exp(\lambda)$

$$
f(X|\lambda)= \lambda e^{-\lambda x}  \tag{for $x \geq 0$}
$$


The expected value and the variance are:

$$E[X] = 1/\lambda $$

$$Var(X)=1/\lambda^2$$


```{admonition} Examples in ecommerce

- Time of one user to return to visit our site

- Time that a seller is going to buy a premium feature

```



### Beta
The beta distribution is used for random variables which take on values between 0 and 1.
For this reason (and other reasons we will see later in the course), the beta distribution is
commonly used to model probabilities.

$$ X \sim Beta(\alpha,\beta)  $$


$$ f(X|\alpha,\beta) = \frac{\Gamma(\alpha +\beta)}{\Gamma(\alpha)\Gamma(\beta)} X^{\alpha-1} (1-X)^{\beta-1} I_{\{0 \lt X \lt 1 \}}$$

$$E[X] = \frac{\alpha}{\alpha +\beta} $$

$$Var(X)= \frac{\alpha \beta}{(\alpha +\beta)^2 (\alpha +\beta+1)} $$

where $\Gamma(.)$ is the gamma function introduced below with the gamma distribution. Note also that
$\alpha > 0$ and $\beta > 0$. The standard Uniform(0, 1) distribution is a special case of the beta
distribution with $\alpha=\beta=1$.




import numpy as np
from scipy.stats import beta
from matplotlib import pyplot as plt
from IPython.display import HTML


#------------------------------------------------------------
# Define the distribution parameters to be plotted
alpha_values = [0.5, 1., 3.0, 0.5]
beta_values = [0.5, 1., 3.0, 1.5]
linestyles = ['-', '--', ':', '-.']
x = np.linspace(0, 1, 1002)[1:-1]

#------------------------------------------------------------
# plot the distributions
fig, ax = plt.subplots(figsize=(5, 3.75))

for a, b, ls in zip(alpha_values, beta_values, linestyles):
    dist = beta(a, b)

    plt.plot(x, dist.pdf(x), ls=ls, c='black',
             label=r'$\alpha=%.1f,\ \beta=%.1f$' % (a, b))

plt.xlim(0, 1)
plt.ylim(0, 3)

plt.xlabel('$x$')
plt.ylabel(r'$p(x|\alpha,\beta)$')
plt.title('Beta Distribution')

plt.legend(loc=0)
plt.show()





### Gamma

Gamma prior $Y \sim \Gamma(\alpha,\beta) \Rightarrow$

$$f(y|\alpha,\beta)=\frac{\beta^\alpha}{\Gamma(\alpha)} \lambda^{\alpha-1} e^{- \beta \lambda} I_{Y \gt 0} $$

$$E[X] = \frac{\alpha}{\beta} $$

$$Var(X)= \frac{\alpha}{\beta ^2} $$

where $\Gamma(.)$ is the gamma function, a generalization of the factorial function which can accept
non-integer arguments. If n is a positive integer, then $\Gamma(n)=(n-1)!$. Note also that $\alpha > 0$
and $\beta > 0$.
The exponential distribution is a special case of the gamma distribution with $\alpha=1$.


import numpy as np
from scipy.stats import gamma
from matplotlib import pyplot as plt



#------------------------------------------------------------
# plot the distributions
k_values = [1, 2, 3, 5]
theta_values = [2, 1, 1, 0.5]
linestyles = ['-', '--', ':', '-.']
x = np.linspace(1E-6, 10, 1000)

#------------------------------------------------------------
# plot the distributions
fig, ax = plt.subplots(figsize=(5, 3.75))

for k, t, ls in zip(k_values, theta_values, linestyles):
    dist = gamma(k, 0, t)
    plt.plot(x, dist.pdf(x), ls=ls, c='black',
             label=r'$\alpha=%.1f,\ \beta=%.1f$' % (k, t))

plt.xlim(0, 10)
plt.ylim(0, 0.45)

plt.xlabel('$x$')
plt.ylabel(r'$p(x|\alpha,\beta)$')
plt.title('Example Gamma Distribution')

plt.legend(loc=0)
plt.show()





Note that in python to print the cumulative value (at p=0.1) for the gamma given $\alpha=6$ and $\beta=81.5$ you can use the following code taking in mind that the parameter $\mathrm{scale}$ is $1/\beta$:


gamma.cdf(0.1,6,scale=1/81.5)


In the next  [section](distributionPython) it is reported the list of main python function to compute the different probabilities



```{toctree}
:hidden:
:titlesonly:


Distribution Manipulation Function in Python
```
