(ctrperplatformexamples)=
## Uncertainty of unique clicks rate in searches per platforms

We cand distinguish between AndroidApp and ResponsiveWeb

We want to compare the performance os the search per different platform.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from operator import itemgetter
%matplotlib inline

### Preparing the dataset

searches = pd.read_csv("../../../bayesian-course/datasets-toys/datasets/daily_click_to_rate_per_searches_type.csv")
searches['uctr'] = searches['searches_with_click'] /searches['searches']

data = searches
data.head()

For every day we know:

- $\theta$ is the probability that one search has one click (this is following a Benoulli($\theta$)
- Random variable $X$ which represnets number of searches with clicks during a day follows a binomial distributions $B(n,\theta)$ where $n$ is the number of total daily searches and $\theta$ is the probability that one search has one click . Then, the unique CTR (uctr) is the expected value of this distribution:

$$
E(X) = \theta = uctr
$$

We want to create a model for the daily unique click rate (uctr) using bayesian statistics.

Bayesian Inference has three steps.

Step 1. [Prior] Choose a PDF to model your parameter $\theta$, aka the prior distribution P($\theta$). This is your best guess about parameters before seeing the data X.

Step 2. [Likelihood] Choose a PDF for P(X|$\theta$). Basically you are modeling how the data X will look like given the parameter $\theta$.

Step 3. [Posterior] Calculate the posterior distribution P($\theta$|X) and pick the $\theta$ that has the highest P($\theta$|X).
And the posterior becomes the new prior. Repeat step 3 as you get more data.


We are going to do the below process per every platform and plot credible intervals over time!

data.head()

### We are going to ompute the credible interval for every day and platform and plot

import scipy.stats as stats
import matplotlib.pyplot as plt
def credible_intervale(dt,df,a,b,platform):
    data_dt = df[(df['date']==dt) & (df['product_type']==platform)].copy().reset_index()
    k = data_dt['searches_with_click']
    n = data_dt['searches']
    cl,cu = stats.beta.ppf(0.025,a+k,b+n-k)[0],stats.beta.ppf(0.975,a+k,b+n-k)[0]
    return cl,cu,data_dt['uctr'][0]

a=1
b=1
dic_web = {}
for dt in data['date']:
    cl,cu,uctr = credible_intervale(dt,data,a,b,"ResponsiveWeb")
    dic_web[dt] = cl,cu,uctr

a=1
b=1
dic_android = {}
for dt in data['date']:
    cl,cu,uctr = credible_intervale(dt,data,a,b,"AndroidApp")
    dic_android[dt] = cl,cu,uctr

resultWeb = pd.DataFrame.from_dict(dic_web).transpose().sort_index()
resultWeb.columns = ["lower",'upper','uctr']
resultWeb

resultAnd = pd.DataFrame.from_dict(dic_android).transpose().sort_index()
resultAnd.columns = ["lower",'upper','uctr']
resultAnd

fig, ax = plt.subplots(figsize=(20,12))
plt.plot(resultWeb.index, resultWeb['uctr'],marker='o')
for lower,upper,y in zip(resultWeb['lower'],resultWeb['upper'],range(resultWeb.shape[0])):
    plt.plot((y,y),(lower,upper),'r_-',color='red')
plt.xticks(range(resultWeb.shape[0]),list(resultWeb.index))
plt.title('Uncertainty of the UCTR in ResponsiveWeb \n (credible intervals)' , fontsize=20)
# Add X and y Label
_ = plt.xlabel("date", fontsize=16)
_ = plt.setp(ax.xaxis.get_majorticklabels(), rotation=-45 ) 

fig, ax = plt.subplots(figsize=(20,12))
plt.plot(resultAnd.index, resultAnd['uctr'],marker='o',color='orange')
for lower,upper,y in zip(resultAnd['lower'],resultAnd['upper'],range(resultAnd.shape[0])):
    plt.plot((y,y),(lower,upper),'g_-',color='red')
plt.xticks(range(resultAnd.shape[0]),list(resultAnd.index))
plt.title('Uncertainty of the UCTR over time \n (credible intervals)' , fontsize=20)
_ = plt.setp(ax.xaxis.get_majorticklabels(), rotation=-45 ) 
# Add X and y Label

fig, ax = plt.subplots(figsize=(20,12))
plt.plot(resultWeb.index, resultWeb['uctr'],marker='o')
for lower,upper,y in zip(resultWeb['lower'],resultWeb['upper'],range(resultWeb.shape[0])):
    plt.plot((y,y),(lower,upper),'r_-',color='red')
plt.xticks(range(resultWeb.shape[0]),list(resultWeb.index))
plt.plot(resultAnd.index, resultAnd['uctr'],marker='o',color='orange')
for lower,upper,y in zip(resultAnd['lower'],resultAnd['upper'],range(resultAnd.shape[0])):
    plt.plot((y,y),(lower,upper),'g_-',color='red')
plt.xticks(range(resultAnd.shape[0]),list(resultAnd.index))
plt.title('Click to rate over time per platform comparison' , fontsize=20)
_ = plt.setp(ax.xaxis.get_majorticklabels(), rotation=-45 ) 
# Add X and y Label