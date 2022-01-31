# Other Prior and Posterior models  <a class="anchor" id="chapter4"></a>


In the previous chapter, we introduced the concept of conjugate family and we saw an example of conjugate distribution: the Beta Prior conjugate with binomial give a Beta Posterior.

In this section we are introducing other conjugate families. We already saw a summary of the conjugate priors families and we go through some of those in this chapter:


| Likelihood  | Prior  | Posterior  |  
|---|---|---|
| Binomial    | Beta | Beta  | 
| Poisson    | Gamma | Gamma  | 
| Exponential     | Gamma | Gamma  | 
| Normal (mean unknown)    | Normal | Normal  | 
| Normal (variance unknown)   | Inverse Gamma | Inverse Gamma  | 
| Normal (variance and mean unknown)  | Normal/Gamma | Normal/Gamma  | 


## Poisson Data
```{admonition} Example : chips into cookies
 The distribution of chips inside each cookies can be approximated to a Poisson distribution.
$Y_i \sim \textrm{Poiss}(\lambda)$.
Typically the Poisson distribution is used to count data unrestricted (for example number of goals in a match). 

The Likelihood distribution for poisson would be:

$$f(y|\lambda)= \frac{\lambda^{\sum y_i} e^{-n \lambda}}{\prod_{i=1}^{n} y_i!} \tag{for $\lambda$ >0}$$

Now the question is : what prio can we use? A convenient choise would be a conjunte function. Which distribution looks like to an exponential to the power of minus a function? That is the Gamma distribution again.
```
Gamma prior $\lambda \sim \Gamma(\alpha,\beta) \Rightarrow$

$$f(\lambda)=\frac{\beta^\alpha}{\Gamma(\alpha)} \lambda^{\alpha-1} e^{- \beta \lambda}\Rightarrow $$

the posterior will looks like

$$ 
f(\lambda|y) \varpropto f(y|\lambda) f(\lambda) \varpropto \lambda^{\sum y_i} e^{-n \lambda} \lambda^{\alpha -1} e^{-\beta \lambda} \varpropto \lambda^{(\alpha +\sum y_i)-1} e^{-(\beta +n)\lambda}
$$

the posterior is $\Gamma(\alpha+\sum y_i, \beta +n)$ 

with the mead of $\Gamma$=$\frac{\alpha}{\beta}$


The posterior mean is 

$$\frac{\alpha+\sum y_i}{\beta +n} = \frac{\beta}{\beta +n}\frac{\alpha}{\beta}+\frac{n}{\beta +n}\frac{\sum y_i}{n}$$

we recognize the first and third term as the weigth that sums to 1 and the second and third element as the prior mean and the data mean. 
The effective sample size here is $\beta$.

The question now is : How we choose $\beta, \alpha$?

Coming back to the example of chips into the cookies: we may have some ideas for the prios. Which stragegy do we use to put information in, then collect data and update it to get the posterior?

We can opt on two strategies:

$\textbf{first strategy to choose the prior poisson distribution}$

Include our belief and knowledge on our prior. We need to find two paremeter $\alpha,\beta$ so we need to specify at least two equation.


First can be the prior mean ($=\frac{\alpha}{\beta}$) : what is our belief on the mean number of chips for cookies


Second what is a reasonable prior standard deviation for this mean: for example if we estimate that the prior mean is to have 12 chips for cookies, do we think we have a standard deviation of 3,4,6? How sure are we? Remember standard dev = $\frac{\sqrt{\alpha}}{\beta}$


Given the previous two info, we can solve to obtain $\alpha,\beta$.


$\textbf{second strategy to choose the prior poisson distribution}$

We express our ignorance with the vague prior: a flat distribution across much of the space parameter. 

Ex. Given a small $\epsilon \Rightarrow \Gamma(\epsilon, \epsilon)$. The mean here is $\epsilon / \epsilon =1$ and the standard deviation is $\frac{\sqrt{\epsilon}}{\epsilon}=\frac{1}{\sqrt{\epsilon}} \Rightarrow $
if $\epsilon$ is small, the standard deviation will be large and so we have a very diffuse prior across the space. 


The mean posterior under this prior will be $ \frac{\epsilon +\sum y_i}{\epsilon +n} \approx \frac{\sum y_i}{ n} $ since $\epsilon$ is small. This shows that the posterior will be strictly driven by the data.


```{admonition} Example


A retailer notices that a certain type of customer tends to call their customer service hotline more often than other customers, so they begin keeping track. They decide a Poisson process model is appropriate for counting calls, with calling rate $\theta$ calls per customer per day.

The model for the total number of calls is then 

$$Y \sim \text{Poisson}(n\cdot t \cdot \theta)$$ 
where $n$ is the number of customers in the group and $t$ is the number of days. That is, if we observe the calls from a group with 24 customers for 5 days, the expected number of calls would be $24\cdot 5\cdot \theta = 120\cdot \theta$.

The likelihood for Y is then $f(y \mid \theta) = \frac{(nt\theta)^y e^{-nt\theta}}{y!} \propto (\theta)^y e^{-nt\theta}$.

This model also has a conjugate gamma prior $\theta \sim \text{Gamma}(a, b)$  which has density (PDF) $f(\theta) = \frac{b^a}{\Gamma(a)} \theta^{a-1} e^{-b\theta} \propto \theta^{a-1} e^{-b\theta}$ 

Following the same procedure of the previous paragraph, it can be shown that the posterior distribution for $\theta$ is $\Gamma(a+y,b+nt)$
```


## Exponential data

```{admonition} Example

 A bus pass every 10 minute. The waiting time (Y) for the next bus is an exponential distribution 
$Y \sim exp(\lambda)$.
The expectation value for $Y$ is $1/\lambda$.


The gamma distribution is a conjugate distribution for the exponential likelihood.

We need to specify a prio for our distribution so a specific value for $\Gamma$. If we expect the bus coming every 10 minutes, that is a rate of 1 over 10 $\Rightarrow$ Prior mean$=(\frac{\alpha}{\beta}$) =1/10.

We need to specify the variabilty also: Remember standard dev = $\frac{\sqrt{\alpha}}{\beta}$


For example: $\Gamma(100,1000)$ will have a mean of $1/10$ and a standard deviation of $1/100$. If we are thinking to the mean plus 2 standard deviation, the current distribution is between $0.1 \pm 0.02$.


Now suppose you waited 12 min ($Y=12$) and the bus arrive. You want to update the posterior for $\lambda$ for how often the bus arrives:

$$f(\lambda|y) \sim f(y|\lambda) f(y) \sim \lambda e^{- \lambda y} \lambda^{\alpha -1} e^{- \beta \lambda} = \lambda^{(\alpha -1)+1} e^{-(\beta + y)\lambda}$$

$$\lambda|y \sim \Gamma(\alpha+1,\beta+y) $$

Including our values for our example : $\lambda|y \sim \Gamma(101,1012) $

Thus the posterior mean is $101/1012=0.0998=1/10.02$.
Just one observation does not change a lot our prior belief and the shift in the posterior is negligible.


Note that in the case of the exponential-gamma model, the sample size is given by $\alpha$ ( and not $\beta$ like in the poisson-gamma model)
```

## Normal data

### known standard deviation, unknown mean

For now, let's suppose the standard deviation is known and we are interested in the mean (this often happens in the manufactoring industry)

$$X_i \sim N(\mu,\sigma_0^2) $$

How we can choose the prior for the mean unknown paramater? It turns out that the Normal is conjugate with himself for the mean so we will specify a normal distribution for the prior of the mean

$$\mu \sim N(m_0,s_0^2 )$$
The posterior will be: $f(\mu|x)=f(x|\mu)f(\mu)$

After some calculation it is obtained that

$$f(\mu|x)=N \left(   \frac{\frac{n \bar{x}}{\sigma_0^2}+ \frac{m_0}{s_0^2}}{\frac{n}{\sigma_0^2}+\frac{1}{s_0^2}},\frac{1}{\frac{n}{\sigma_0^2}+\frac{1}{s_0^2}} \right) $$

We can rewrite the posterior mean as 


$$\frac{n}{n+ \frac{\sigma_0^2}{s_0^2}} \bar{x} + \frac{\frac{\sigma_0^2}{s_0^2}}{n+ \frac{\sigma_0^2}{s_0^2}} m $$

We can see that the posterior mean is a weighted sum of the data mean ($\bar{x}$) and the prior mean ($m$).

The effective sample size is $\frac{\sigma_0^2}{s_0^2}$ (the ratio of the variance of the data $\sigma_0$ and the variance of the prior $s_0$)


We can also consider the case in which both $\mu$ and $\sigma$ are unknown. In this case you can specify a prior for $\mu$ given $\sigma$ following the normal distribution and a prior for $\sigma$ following a $\Gamma$ distribution. If we marginalize by integrating out the $\sigma$, it can be obtained that the posterior for $\mu \mid x$ follows a t distribution. This approach can be extended to the multivariate normal cases.




## Alternative priors

### Non-informative priors

These priors are built based on our ignorance about the prior and with the scope to give the maximum wieght on the posterior based on the data observations.

Coming back to the coin example : $Y_i \sim B(\theta)$. How we minimize our prior information for $\theta$?

Intuitively we can say that all the values of $\theta$ are equally likely : 
$\theta = U[0,1] =\textrm{Beta}(1,1) $.

Remember that the effective sample size of the Beta prior is the sum of the two parameters, in the case of the uniform distribution, it is equal to 2, this is equivalent to data with one head and one tail already in it. So this is not a complete non informative priors. We can think to an even leff informative prior:

$\textrm{Beta}(1/2,1/2)$ so the the effective sample size is 1 or even go to a effective sample size close to 0 with a  prior distribution like $\textrm{Beta}(0.001,0.001)$ . In this case all will be driven by the data.


The limit case would be $\textrm{Beta}(0.,0.) \Rightarrow f(\theta) \propto \theta^{-1} (1-\theta)^{-1}$
The last one is not a proper density, the ingral does not give 1 : this is called as $\textbf{improper prior}$.

However we can use it and we would obtain $f(y \mid \theta) \propto \theta^{y-1} (1- \theta)^{n-y-1} \sim \textrm{Beta}(y,n-y)$. The posterior mean is $y/n$ that we recognize as $\hat{\theta}$ the maximum likelihood for the binomial. 

So that by using an improper prior we got a posterior which gives us a point estimates equals to the frequentist approach which by constrution depend completely from the data. But now, with respect to the frequentist approach, we can go further and build a posterior and a 95$\%$ confidence interval that our $\theta$ will fall there.


As another example of improper priors: let's take the normal distribution with known $\sigma$ and unknown $\mu$. $Y_i \sim N(\mu,\sigma^2)$ and an improper normal distribution can be something like $N(0,100000^2)$ a distribution with spread variance, in the limit $\sigma$ which tends to $\infty$, $f(\mu) \sim 1$. In this case 

$$f(\mu \mid y) = f(y \mid \mu) f(\mu)= exp \left[-\frac{1}{2 \sigma^2} \sum (y_i-\mu)^2 \right] (1) \propto  exp \left[-\frac{1}{2 \sigma^2 /n}  (\mu - \bar{y})^2 \right] \propto N(\bar{y}, \sigma^2/n)   $$
which can be recognized as the likellihood of the normal distribution. Again, even if in this case with improper prior we ended up with the result of the frequentistic approach, the advantage here is that we can build a posterior and a confidence interval for the parameters.


 
If we want to choose an uniform prior, there are different way to define an uniform prior. For example a uniform prior for the $\sigma$ of the normal distribution, can be $f(\sigma^2)=1/\sigma^2$ or $f(\sigma^2)=1$ both are uniform in certain scale and using certain parametrization. Using different uniform prior we will get different posteriors.

The key concept is that uniform priors are not invariant with respect to trasformation. A way to overpass this limitation is by using the $\textbf{Jeffreys Priors}$ which use $f(\theta) \propto \sqrt{I(\theta)}$ where $I(\theta)$ is the Fisher distribution.



