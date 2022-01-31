## Distribution Manipulation Functions in Python

### Common Functionality

The typycal funcions to manipulate distributions are:

| Function           | What it does                                               |
|:------------------ |:---------------------------------------------------------- |
| dnorm(x, mean, sd) | Evaluate the PDF at x (x=mean, sd=standard deviation)      |
| pnorm(q, mean, sd) | Evaluate the CDF at q (x=mean, sd=standard deviation)      |
| qnorm(p, mean, sd) | Evaluate the quantile at p (x=mean, sd=standard deviation) |
| rnorm(n, mean, sd) | Generate n pseudo-random values from distribution .        |

![alt text](fig-1.png "Title")


import scipy.stats as stats

rv_dnorm = stats.norm.pdf(0.6, loc=0, scale=1)
rv_pnorm = stats.norm.cdf(0.6, loc=0, scale=1)
rv_qnorm = stats.norm.ppf(0.6, loc=0, scale=1)
rv_rnorm = stats.norm.rvs(size=10, loc=0, scale=1)

print("rv_dnorm:", rv_dnorm)
print("rv_pnorm:", rv_pnorm)
print("rv_qnorm:", rv_qnorm)
print("rv_rnorm:", rv_rnorm)

rv = stats.norm(loc=0, scale=1)
rv_dnorm = rv.pdf(0.6)
rv_pnorm = rv.cdf(0.6)
rv_qnorm = rv.ppf(0.6)
rv_rnorm = rv.rvs(size=10)

print("rv_dnorm:", rv_dnorm)
print("rv_pnorm:", rv_pnorm)
print("rv_qnorm:", rv_qnorm)
print("rv_rnorm:", rv_rnorm)

### Common Distributions

We are going to see how to do this in python for a superset of distributions which ar available in scipy.stats and listed in the [scipy.stats documentation](https://docs.scipy.org/doc/scipy/reference/stats.html).

* Binomial(n, p)
* Poisson($\lambda$)
* Exponential($\lambda$)
* Gamma($\alpha$, $\beta$)
* Uniform(a, b)
* Beta($\alpha$, $\beta$)
* Normal($\mu$, $\sigma$<sup>2</sup>)
* Student t($\nu$)

Below we will compute some values using scipy.stats functions.

#### Example 1

Suppose X ∼ Binomial(5, 0.6). Evaluate the CDF at x=1, or equivalently P(X ≤ 1) ≈ 0.087. Verify that the quantile value returned at this CDF is the same as our original value of x.

stats.binom.cdf(1, n=5, p=0.6)

stats.binom.ppf(0.08704, n=5, p=0.6)

#### Example 2

Suppose Y ∼ Exp(1). Verify that the range 0.105 < Y ≤ 2.303 contains the middle 80% of the probability mass.

stats.expon.ppf([0.1, 0.9], scale=1)

#### Exercise 1

Let X ∼ Pois(3). Find P(X = 1).

stats.poisson.pmf(1, mu=3)

#### Exercise 2

Let X ∼ Pois(3). Find P(X ≤ 1).

stats.poisson.cdf(1, mu=3)

#### Exercise 3

Let X ∼ Pois(3). Find P(X > 1).

1 - stats.poisson.cdf(1, mu=3)

#### Exercise 4

Let Y ∼ Gamma(2, 1/3). Find P(0.5 < Y < 1.5).

rv = stats.gamma(a=2, scale=3)
rv.cdf(1.5) - rv.cdf(0.5)

#### Exercise 5

Let Z ∼ N(0, 1). Find z such that P(Z < z) = 0.975.

stats.norm.ppf(0.975, loc=0, scale=1)

#### Exercise 6

Let Z ∼ N(0, 1). Find P(−1.96 < Z < 1.96).

rv = stats.norm(loc=0, scale=1)
rv.cdf(1.96) - rv.cdf(-1.96)

#### Exercise 7

Let Z ∼ N(0, 1). Find z such that P(−z < Z < z) = 0.90.

z_range = stats.norm.ppf([0.05, 0.95], loc=0, scale=1)
print(z_range)
print("z:", z_range[1])

