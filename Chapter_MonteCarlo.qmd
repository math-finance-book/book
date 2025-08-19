{{< include macros.qmd >}}

## Monte Carlo Methods {#sec-c:montecarlo} 

```{python}
#| eval: true
#| echo: false

import plotly
from IPython.display import display, HTML

plotly.offline.init_notebook_mode(connected=True)
display(
    HTML(
        '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
    )
)
```


In this chapter, we will introduce a principal numerical method for valuing derivative securities: Monte Carlo. Throughout the chapter, we will assume there is a constant risk-free rate.  The last section, while quite important, could be skimmed on first reading---the rest of the book does not build upon it.

## Introduction to Monte Carlo {#sec-s:mc_europeans}

According to our risk-neutral pricing @eq-riskneutralformula, the value of a security paying an amount $x$ at date $T$ is
$$
\mathrm{e}^{-rT}\mathbb{E}^R[x]\;.
$$ {#eq-montecarlo1}

To estimate this by Monte-Carlo \index{Monte Carlo} means to simulate a sample of values for the random variable
$x$
and to estimate the expectation by averaging the sample values.^[Boyle~[@boyle] introduced Monte-Carlo methods for derivative valuation, including the variance-reduction methods of control variates and antithetic variates to be discussed later].  Of course, for this to work, the sample must be generated from a population having a distribution consistent with the risk-neutral probabilities.

The simplest example is valuing a European option under the Black-Scholes assumptions.  Of course, for calls and puts, this is redundant, because we already have the Black-Scholes formulas.  Nevertheless, we will describe how to do this for the sake of introducing the Monte Carlo method.  In the case of a call option, the random variable $x$ in @eq-montecarlo1 is $\max(0,S_T-K)$.  To simulate a sample of values for this random variable, we need to simulate the terminal stock price $S_T$.  This is easy to do, because, under the Black-Scholes assumptions, the logarithm of $S_T$ is normally distributed under the risk-neutral probability with mean $\log S_0+\nu T$ and variance $\sigma^2T$, where $\nu=r-q-\sigma^2/2$.  Thus, we can simulate values for $\log S_T$ as $\log S_0+\nu T + \sigma\sqrt{T}z$, where $z$ is a standard normal.  We can  average the simulated values of $\max(0,S_T-K)$, or whatever the payoff of the derivative is, and then discount at the risk-free rate to compute the date--0 value of the derivative.  This means that we generate some number $M$ of standard normals $z_i$ and estimate the option value as $\mathrm{e}^{-rT}\bar{x}$, where $\bar{x}$ is the mean of 
$$x_i = \max\left(0,\mathrm{e}^{\log S_0+\nu T + \sigma\sqrt{T}z_i}-K\right)\; .$$
To value options that are path-dependent  we need to simulate the path of the underlying asset price.  

There are two main drawbacks to Monte-Carlo methods.  First, it is difficult (though not impossible) to value early-exercise features.^[Monte-Carlo methods for valuing early exercise include the stochastic mesh method of Broadie and Glasserman [@BG] and the regression method of Longstaff and Schwartz [@LS01].  Glasserman [@Glasserman] provides a good discussion of these methods and the relation between them.]  To value early exercise, we need to know the value at each date if not exercised, to compare to the intrinsic value.  One could consider performing a simulation at each date to calculate the value if not exercised, but this value depends on the option to exercise early at later dates, which cannot be calculated without knowing the value of being able to exercise early at even later dates, etc.  In contrast, the binomial model can easily handle early exercise but cannot easily handle path dependencies.  

The second drawback of Monte Carlo methods is that they can be quite inefficient in terms of computation time (though, as will be explained later, they may be faster than alternative methods for derivatives written on multiple assets).  As in statistics, the standard error of the estimate depends on the sample size.  Specifically, given a random sample $\{x_1,\ldots,x_M\}$ of size $M$ from a population with mean $\mu$ and variance $\sigma^2$, the best estimate of $\mu$ is the sample mean $\bar{x}$, and the standard error of $\bar{x}$ (which means the standard deviation of $\bar{x}$ in repeated samples) is best estimated by
$$
\sqrt{\frac{1}{M(M-1)}\left(\sum_{i=1}^{M} x_i^2-M\bar{x}^2\right)}\;.
$$ {#eq-standarderror}
 \index{standard error}
Recall that $\bar{x}$ plus or minus 1.96 standard errors is a 95\% confidence interval for $\mu$ when the $x_i$ are normally distributed.
In the context of European option valuation, the expression @eq-standarderror gives the  standard error of the estimated option value at maturity, and multiplication of @eq-standarderror by $\mathrm{e}^{-rT}$ gives the standard error of the estimated date--0 option value.

To obtain an estimate with an acceptably small standard error may require a large sample size and hence a relatively large amount of computation time.  The complexities of Monte Carlo methods arise from trying to reduce the required sample size.  Later, we will describe two such methods (antithetic variates and control variates).   For those who want to engage in a more detailed study of Monte Carlo methods, the book of Glasserman [@Glasserman] is highly recommended.  J$\ddot{\text{a}}$ckel [@Jackel] is useful for more advanced readers, and Clewlow and Strickland [@CS] and  Brandimarte [@Brandimarte] are useful references that include computer code.


#### Monte Carlo Valuation of a European Call
We will illustrate Monte Carlo by valuing a European call under the Black-Scholes assumptions.  We will also estimate the delta by each of the methods described in @sec-s:montecarlogreeks1 and @sec-s:montecarlogreeks2.  Of course, we know the call value and its delta from the Black-Scholes formulas, and they can be used to evaluate the accuracy of the Monte Carlo estimates.  In this circumstance, we only need to simulate the price of the underlying at the option maturity rather than the entire path of the price process. Therefore we  set $m=1$. However, we use a large number of paths, $n=10,000$ to get a large sample of terminal stock prices.

```{python}
# Simulate Geometric Brownian Motion
import numpy as np
import matplotlib.pyplot as plt
# number of paths
n = 10000
#number of divisions
m = 1
# Interest rate (We set the drift equal to the interest rate for the risk-neutral probability)
r = 0.1
# Volatility
sig = 0.2
# Initial Stock Price
S0 = 42
# Maturity
T = 0.5
#Strike Price
K=40
# Dividend Yield
q=0.0
# Delta t
dt = T/m
# Drift (nu)
drift = (r-q-0.5*sig**2)
# Volatility
vol = sig * np.sqrt(dt)

t = np.array(range(0,m + 1,1)) * dt

# seed for random generator
seed= 2020
# define a random generator
np.random.seed(seed)
inc = np.zeros(shape = (m + 1, n))
inc[1:] = np.transpose(np.random.normal(loc = 0, scale = vol,size = (n,m)))
St = np.zeros(shape = (m + 1, n))
St = S0 * np.exp(np.cumsum(inc,axis=0) + (drift * t[0:m + 1])[:,None])
St1 = S0 * np.exp(-np.cumsum(inc,axis=0) + (drift * t[0:m + 1])[:,None])

```
As before, this code generates two samples $St$, which adds the simulated standard (zero mean) normal random variable, and $St1$ which subtracts the simulated (zero mean) standard normal random variable.  Each sample produces and estimate for the Black-Scholes European call option.

```{python}
cc=np.maximum(St[m,:]-K,0)
cp = np.mean(cc) * np.exp(-r * T)
cc1=np.maximum(St1[m,:]-K,0)*np.exp(-r * T)
cp1= np.mean(np.maximum(St1[m,:]-K,0)) * np.exp(-r * T)

print('The first sample gives an estimated call price=',cp)
print('The second sample gives an estimated call price=',cp1)
bsc = (cp+cp1)/2
print('The average of the two estimates=',bsc)
```
The true call price is given by

```{python}
from scipy.stats import norm
import numpy as np
from scipy.optimize import minimize, minimize_scalar

def blackscholes(S0, K, r, q, sig, T, call = True):
    '''Calculate option price using B-S formula.
    
    Args:
        S0 (num): initial price of underlying asset.
        K (num): strick price.
        r (num): risk free rate
        q (num): dividend yield
        sig (num): Black-Scholes volatility.
        T (num): maturity.
        call (bool): True returns call price, False returns put price.
        
    Returns:
        num
    '''
    d1 = (np.log(S0/K) + (r -q + sig**2/2) * T)/(sig*np.sqrt(T))
    d2 = d1 - sig*np.sqrt(T)
    if call:
        return np.exp(- q *T) * S0 * norm.cdf(d1,0,1) - K * np.exp(-r * T) * norm.cdf(d2,0, 1) 
    else:
        return -np.exp(-q * T) * S0 * norm.cdf(-d1,0,1) + K * np.exp(-r * T) * norm.cdf(-d2,0, 1)

truebsc=blackscholes(S0, K, r, q, sig, T, call = True)
print('The black scholes fromula=',truebsc)
```
  
  
  Notice that even with 10,000 data points for each sample the individual estimates are not very accurate compared to the exact Black Scoles price.  This is a well known problem that is difficult to estimate the mean, even with a lot of data and is a drawback to Monte Carlo as discussed earlier. However, the average of the two prices is sgnificantly more accurate.  This is an example of an antithetic variable which is discussed later.  One simple intution is the two samples yield negatively correlated errors; if the plus sample is two high, then the minus sample will be too low.  Combined, the simulation error will cancel out.  Another intution is that each individual sample has a wrong estimate of the mean.  However, the combined sample has zero mean by construction.  Therefore combining the samples give the right mean of the simulated standard normal random variable.  Nevertheless, there is still sampling error since we are estimating the mean of the discounted call payoffs, not the mean of the standard normal.  This method and other methods to reduce sampling error are discussed next.

## Antithetic Variates in Monte Carlo

In this and the following section, we will discuss two methods to increase the efficiency of the Monte Carlo method.  These are two of the simplest methods.  They are used extensively, but there are other important methods that are also widely used.  J$\ddot{\text{a}}$ckel [@Jackel] and Glasserman [@Glasserman] provide a wealth of information on this topic.

The Monte Carlo method estimates the mean $\mu$ of a random variable $x$ as the sample average of randomly generated values of $x$.  An antithetic variate \index{antithetic variate} is a random variable $y$ with the same mean as $x$ and a negative correlation with $x$.  It follows that the random variable $z=(x+y)/2$ will have the same mean as $x$ and a lower variance.  Therefore the sample mean of $M$ simulations of $z$ will be an unbiased estimate of $\mu$ and will have a lower standard error than the sample mean of $M$ simulations of $x$.  Thus, we should obtain a more efficient estimator of $\mu$ by simulating $z$ instead of $x$.^[
The negative correlation between $x$ and $y$ is essential for this method to generate a real gain in efficiency.  To generate $M$ simulations of $z$, one must generate $M$ simulations of $x$ and $M$ of $y$, which will generally require about as much computation time as generating $2M$ simulations of $x$.  If $x$ and $y$ were independent, the standard error from $M$ simulations of $z$ would be the same as the standard error from $2M$ simulations of $x$, so using the antithetic variate would be no better than just doubling the sample size for $x$.]

In the context of derivative valuation, the standard application of this idea is to generate two negatively correlated underlying asset prices (or price paths, if the derivative is path dependent).  The terminal value of the derivative written on the first asset serves as $x$ and the terminal value of the derivative written on the second serves as $y$.  Because both asset prices have the same distribution, the means of $x$ and $y$ will be the same, and the discounted mean is the date--0 value of the derivative. 

Consider for example a non-path-dependent option in a world with constant volatility.  In each simulation $i$ ($i=1,\ldots,M$), we would generate a standard normal $Z_i$ and compute
$$
\begin{align*}
\log S_i(T) &= \log S_0 + \left(r-q-\frac{1}{2}\sigma^2\right)T + \sigma\sqrt{T}Z_i\; ,\\
\log S_i'(T) &= \log S_0 + \left(r-q-\frac{1}{2}\sigma^2\right)T - \sigma\sqrt{T}Z_i\;.
\end{align*}
$$
Given the first terminal price, the value of the derivative will be some number $x_i$ and given the second it will be some number $y_i$.  The date--0 value of the derivative is estimated as
$$
\mathrm{e}^{-rT}\frac{1}{M}\sum_{i=1}^M\frac{x_i+y_i}{2}\; .
$$


## Control Variates in Monte Carlo {#sec-s:controlvariates}
\index{control variate}
Another approach to increasing the efficiency of the Monte Carlo method is to adjust the estimated mean (option value) based on the known mean of another related variable.  We can explain this in terms of linear regression in statistics.  Suppose we have a random sample $\{x_1,\ldots,x_M\}$ of a variable $x$ with unknown mean $\mu$, and suppose we have a corresponding sample $\{y_1,\ldots,y_M\}$ of another variable $y$ with known mean $\phi$.  Then an efficient estimate of $\mu$ is $\hat{\mu} = \bar{x} + \hat{\beta} (\phi-\bar{y})$, where $\bar{x}$ and $\bar{y}$ denote the sample means of $x$ and $y$, and where $\hat{\beta}$ is the coefficient of $y$ in the linear regression of $x$ on $y$ (i.e., the estimate of $\beta$ in the linear model $x = \alpha +\beta y + \varepsilon$).  The standard Monte Carlo method, which we have described thus far, simply estimates the mean of $x$ as $\bar{x}$.  The control variate method adjusts the estimate by adding $\hat{\beta} (\phi-\bar{y})$.  To understand this correction, assume for example that the true $\beta$ is positive.  If the random sample is such that $\bar{y}<\phi$, then it must be that small values of $y$ were over-represented in the sample.  Since $x$ and $y$ tend to move up and down together (this is the meaning of a positive $\beta$) it is likely that small values of $x$ were also over-represented in the sample.  Therefore, one should adjust the sample mean of $x$ upwards in order to estimate $\mu$.  The best adjustment will take into account the extent to which small values of $y$ were over-represented (i.e., the difference between $\bar{y}$ and $\phi$) and the strength of the relation between $x$ and $y$ (which the estimate $\hat{\beta}$ represents).  The efficient correction of this sort is also the simplest:  just add $\hat{\beta}(\phi-\bar{y})$ to $\bar{x}$.  In practice, the estimation of $\hat{\beta}$ may be omitted and one may simply take $\hat{\beta}=1$, if the relationship between $x$ and $y$ can be assumed to be one-for-one.  If $\beta$ is to be estimated, the estimate (by ordinary least squares) is
$$\hat{\beta} = \frac{\sum_{i=1}^M x_iy_i - M\bar{x}\bar{y}}{\sum_{i=1}^M y_i^2 - M\bar{y}^2}\; .$$
In general, the correction term $\hat{\beta}(\phi-\bar{y})$ will have a nonzero mean, which introduces a bias in the estimate of $\mu$.  To eliminate the bias, one can compute $\hat{\beta}$ from a pre-sample of $\{x,y\}$ values.  

As an example of a control variate, in our simulation code to estimate the Black Scholes price for a call option we can use the stock price itself.  The known stock price is the input price $S_0$.  The simulation also produces an estimate for the stock price as the dicsounted expected value of the terminal stock price  $\hat{S}=\sum_{i=1}^{n} e^{- r T } S_t(m,i)$ where $S_t(m,i)$ is the $i$th simulated stock price at time $T$.  Theoretically these should be the same umber, but due to error they typically wil not be the same.

```{python}
SS=np.mean(St[m,:])*np.exp(-r*T)
print('The Estimated Stock Price for the first sample is =', SS)
print('The actual stock price should be=', S0)
print('The error is =', S0-SS)
```

The error is $S_0-\hat{S}$ which corresponds to $\phi-y$ above.  We then compute $\hat{\beta}$ and comute the improved estimate
$$ \text{new estimate}= \text{original estimate} +\hat{\beta}(S0-\hat{S}) $$
In the code below we do this procedure for both samples and average the updates.
```{python}
hatbeta= np.cov(St[m,:],cc)[0,1]/np.cov(St[m,],cc)[1,1]
hatbeta1=np.cov(St1[m,:],cc1)[0,1]/np.cov(St1[m,],cc1)[1,1]
correction =hatbeta*(S0-SS)
update=cp + correction
print('hatbeta=',hatbeta)
print('The original estimate for the call price from the first sample=',cp)
print('The original estimate for the call price from the second sample=',cp1)
print('The updated estimate from the first sample is=',update)
SS1=np.mean(St1[m,:])*np.exp(-r*T)
update1=cp1+hatbeta1*(S0-SS1)
print('The updated estimate from the second sample is=',update1)
print('The average of the updated estimates =',(update+update1)/2)

```

We can compare this to the exact Black Scholes formula from before.
```{python}
print('The exact Black Scholes Price is=', truebsc)
```

As another example,  consider the classic case of estimating the value of a discretely-sampled average-price call, using a discretely-sampled geometric-average-price call \index{average-price option} \index{geometric-average option} as a control variate.  Let $\tau$ denote the amount of time that has elapsed since the call was issued and $T$ the amount of time remaining before maturity, so the total maturity of the call is $T+\tau$.  To simplify somewhat,  assume date $0$ is the beginning of a period between observations.  Let $t_1, \ldots, t_N$ denote the remaining sampling dates, with $t_1 = \Delta t$, $t_i-t_{i-1}=\Delta t = T/N$ for each $i$, and $t_N=T$.  We will input the average price $A_0$ computed up to date $0$, assuming this average includes the price $S_0$ at date $0$.  The average price at date $T$ will be 
$$A_T = \frac{\tau}{T+\tau}A_0 + \frac{T}{T+\tau}\left(\frac{\sum_{i=1}^N S_{t_i}}{N}\right)\; .$$
The average-price call pays $\max(0,A_T-K)$ at its maturity $T$, and we can write this as
\begin{align*}
\max(A_T-K,0) &= \max\left(\frac{T}{T+\tau}\left( \frac{\sum_{i=1}^N S_{t_i}}{N}\right) - \left(K - \frac{\tau}{T+\tau}A_0\right), 0\right)\\
&= \frac{T}{T+\tau} \max \left(\frac{\sum_{i=1}^N S_{t_i}}{N} - K^*,0\right)\;,
\end{align*}
where 
$$K^* = \frac{T+\tau}{T}K - \frac{\tau}{T}A_0\; .$$
Therefore, the value at date $0$ of the discretely-sampled average-price call is
$$\frac{T}{T+\tau} \,\mathrm{e}^{-rT} \mathbb{E}^R\left[\max \left(\frac{\sum_{i=1}^N S_{t_i}}{N} - K^*,0\right)\right]\; .$$
In terms of the discussion above, the random variable  the mean of which we want to estimate is
$$x = \mathrm{e}^{-rT}\max \left(\frac{\sum_{i=1}^N S_{t_i}}{N} - K^*,0\right)\; .$$
A random variable $y$ that will be closely correlated to $x$ is
$$y =\mathrm{e}^{-rT}\max \left(\mathrm{e}^{\sum_{i=1}^N \log S_{t_i}/N} - K^*,0\right)\; .$$
The mean $\phi$ of $y$ under the risk-neutral probability is given in the pricing @eq-disc_geom_avg_call. 
We can use the sample mean of $y$ and its known mean $\phi$ to adjust the sample mean of $x$ as an estimator of the value of the average-price call.  Generally, the estimated adjustment coefficient $\hat{\beta}$ will be quite close to 1.  

## Monte Carlo Greeks I: Difference Ratios {#sec-s:montecarlogreeks1}

Greeks can be calculated by Monte Carlo by running the valuation program twice and computing a difference ratio, for example $(C_u-C_d)/(S_u-S_d)$ to estimate a delta.  However, to minimize the error, and minimize the number of computations required, one should use the same set of random draws to estimate the derivative value for different values of the parameter.  For path-independent options (e.g., European puts and calls) under the Black-Scholes assumptions, we only need to generate $S_T$ and then we can compute $S_u(T)$ as $[S_u(0)/S_0] \times S_T$ and $S_d(T)$ as $[S_u(0)/S_0] \times S_T$.  We can estimate standard errors for the Greeks in the same way that we estimate the standard error of the derivative value.  

Actually, there is often a better method available that is just as simple.  This is called pathwise calculation.  We will explain this in the next section.   Here we will describe how to estimate the delta and gamma of a derivative as sample means of difference ratios.

Consider  initial prices for the underlying $S_u>S>S_d$.  Denote the underlying price at the option maturity in a given simulation by $S_u(T)$  when the initial underlying price is $S_u$, by $S_T$  when the initial underlying price is $S$, and by $S_d(T)$  when the initial underlying price is $S_d$.   Under the Black-Scholes assumptions, the logarithm of the stock price at date $T$ starting from the three initial prices $S_d$, $S$ and $S_u$ is

$$
\begin{align*}
\log S_d(T) &= \log S_d + \left(r-q-\frac{1}{2}\sigma^2\right)T + \sigma B_T\; ,\\
\log S_T &= \log S + \left(r-q-\frac{1}{2}\sigma^2\right)T + \sigma B_T\; ,\\
\log S_u(T) &= \log S_u + \left(r-q-\frac{1}{2}\sigma^2\right)T + \sigma B_T\;,
\end{align*}
$$
so 
$$\log S_d(T) = \log S_T + \log S_d - \log S\Longrightarrow S_d(T) = \left(\frac{S_d}{S}\right) S_T\; ,$$
and
$$\log S_u(T) = \log S_T + \log S_u - \log S \Longrightarrow S_u(T) = \left(\frac{S_u}{S}\right) S_T\; .$$
Therefore, under the Black-Scholes assumptions, we only need to simulate $S_T$ and then perform the multiplications indicated above to obtain $S_d(T)$ and $S_u(T)$. 

Consider a particular simulation and let $C_d(T)$ denote the value of the derivative at maturity  when the initial asset price is $S_d$, let  $C_T$ denote the value of the derivative at maturity  when the initial asset price is $S$, and let $C_u(T)$ denote the value of the derivative at maturity  when the initial asset price is $S_u$.  For path-independent derivatives under the Black-Scholes assumptions, these can be computed directly from the simulation of $S_T$ as just described.  However, the following applies to general European derivatives under general assumptions about the underlying asset price (for example, it could follow a GARCH process).

The estimates $C_d$, $C$ and $C_u$ of the date--0 derivative values, for the different initial prices of the underlying, are the discounted sample means of the $C_d(T)$, $C_T$ and $C_u(T)$.
One way to estimate the delta is $(C_u-C_d)/(S_u-S_d)$.  This is a difference of discounted sample means, multiplied by the reciprocal of $S_u-S_d$.  Equivalently, it is the sample mean of the differences $C_u(T)-C_d(T)$, multiplied by $\mathrm{e}^{-rT}/(S_u-S_d)$.  The standard error is
$$
\frac{\mathrm{e}^{-rT}}{S_u-S_d}\sqrt{\frac{1}{M(M-1)}\left(\sum_{i=1}^M \left[C_{ui}(T)-C_{di}(T)\right]^2 - M\left[\overline{C_{u}(T)}-\overline{C_{d}(T)}\right]^2\right)}\; ,
$$ 
\index{standard error}
where the overline denotes the sample mean and where $C_{ui}(T)$ [respectively, $C_{di}(T)$] denotes the value of the derivative at maturity in simulation $i$ when the initial asset price is $S_u$ [respectively, $S_d$].

The corresponding Monte Carlo estimate of the gamma is also a sample mean.  Simple algebra shows that @eq-binomialgamma100 is equivalent to
$$
\Gamma = \frac{2}{(S_u-S)(S_u-S_d)}C_u - \frac{2}{(S_u-S)(S-S_d)}C +\frac{2}{(S-S_d)(S_u-S_d)}C_d\;.
$$ {#eq-binomialgamma200}

Normally one would take $S_u=(1+\alpha)S$ and $S_d = (1-\alpha)S$ for some $\alpha$ (e.g., $\alpha=0.01$).  In this case  @eq-binomialgamma200
simplifies to
$$
\Gamma = \frac{C_u - 2C + C_d}{\alpha^2S^2}\;,
$$ {#eq-binomialgamma300}

and the standard error of the gamma is

$$
\begin{multline*}\frac{\mathrm{e}^{-rT}}{\alpha^2S^2}\sqrt{\frac{1}{M(M-1)}}\\
\times \sqrt{\sum_{i=1}^M \left[C_{ui}(T)-2C_i(T)+C_{di}(T)\right]^2 -M\left[\overline{C_{u}(T)}-2\overline{C_T}+\overline{C_{d}(T)}\right]^2}\; .
\end{multline*}
$$

## Monte Carlo Greeks II: Pathwise Estimates {#sec-s:montecarlogreeks2}
We will examine the bias in the Monte Carlo delta estimate discussed in the preceding section and explain  pathwise estimation of Greeks. By biased, we mean that the expected value of an estimate is different from the true value. \index{bias} It is important to recognize that if a Monte Carlo estimate is biased, then, even if a large number of simulations is used and the standard error is nearly zero, the answer provided by the Monte Carlo method will be incorrect.   For simplicity, consider a European call under the Black-Scholes assumptions.  

The delta estimate we have considered is the discounted sample mean of 
$$
\frac{C_u(T) - C_d(T)}{S_u-S_d}\;.
$$  {#eq-montecarlodelta2}


This ratio takes on one of three values, depending on $S_T$:


- If $S_u(T) \leq K$ then the option is out of the money in both the up and down cases; i.e., 
$$C_u(T) = C_d(T) = 0\; ,$$
so the ratio @eq-montecarlodelta2 is zero.
- If $S_d(T) \geq K$ then the option is in the money in both the up and down cases; i.e.,

$$
\begin{align*} C_u(T) &= S_u(T) - K =\left(\frac{S_u}{S}\right)S_T - K\; ,\\
C_d(T) &= S_d(T) - K = \left(\frac{S_d}{S}\right)S_T - K\;,
\end{align*}
$$

so the ratio  @eq-montecarlodelta2 equals $S_T/S$.
- If $S_u(T) > K > S_d(T)$, then the option is in the money in only the up case; i.e.,
$$
\begin{align*}
C_u(T) &= S_u(T) - K = \left(\frac{S_u}{S}\right)S_T - K\; ,\\
C_d(T) &= 0\;,
\end{align*}
$$
so the ratio @eq-montecarlodelta2 equals 
$$
\frac{\left(\frac{S_u}{S}\right)S_T - K}{S_u-S_d} < \frac{S_T}{S}\; .
$$


The bias is induced by the third case above.  We can see this as follows.  We are trying to estimate
$$
\frac{\partial }{\partial S} \mathrm{e}^{-rT}\mathbb{E}^R \big[\max(0,S_T-K)\big] = \mathrm{e}^{-rT}\mathbb{E}^R  \left[ \frac{\partial }{\partial S} \max(0,S_T-K)\right]\;.
$$ {#eq-montecarlodelta4}

The delta estimate $(C_u-C_d)/(S_u-S_d)$ replaces the mean $\mathbb{E}^R$ with the sample mean and replaces
$$
\frac{\partial }{\partial S} \max(0,S_T-K)
$$ {#eq-montecarlodelta3}

with the ratio @eq-montecarlodelta2.  The derivative @eq-montecarlodelta3 takes on two possible values, depending on $S_T$---we can ignore the case $S_T=K$ because it occurs with zero probability:


- If $S_T < K$, then $\max(0,S_T-K) = 0$ and the derivative is zero.
- If $S_T>K$,  then $\max(0,S_T-K) = S_T-K$ and the derivative equals 
$$\frac{\partial S_T}{\partial S}=\mathrm{e}^{(r-q-\sigma^2/2)T + \sigma B_T} = \frac{S_T}{S}\; .$$

Therefore, the true delta---the expectation @eq-montecarlodelta4---equals^[By changing numeraires, we can show that @eq-montecarlodelta5 equals $\mathrm{e}^{-qT}\mathbb{E}^V[x] = \mathrm{e}^{-qT}\mathrm{N}(d_1)$, as we know from @sec-c:blackscholes is the delta of a European call in the Black-Scholes model (here, as in @sec-c:blackscholes, $V_t=\mathrm{e}^{qt}S_t$ denotes the value of the dividend-reinvested portfolio created from the stock).] 
$$
\mathrm{e}^{-rT}\mathbb{E}^R\left[\frac{S_T}{S} x\right]\;,
$$ {#eq-montecarlodelta5}

where $x$ is the random variable defined as
\begin{equation*}
x =  \begin{cases} 1 & \text{if $S_T>K$}\; ,\\
0 & \text{otherwise}\;.
\end{cases}
\end{equation*}
 On the other hand, our  analysis of the ratio @eq-montecarlodelta2 shows that the expected value of the delta estimate $(C_u-C_d)/(S_u-S_d)$ is
$$
\mathrm{e}^{-rT}\mathbb{E}^R\left[\frac{S_T}{S} y\right] + \mathrm{e}^{-rT}\mathbb{E}^R\left[\frac{S_uS_T-SK}{S(S_u-S_d)}z\right]\;,
$$ {#eq-montecarlodelta6}

where
$$
\begin{align*}
y &=  \begin{cases} 1 & \text{if} S_d(T)>K\; ,\\
0 & \text{otherwise}\;.
\end{cases}
\end{align*}
$$
and
$$
\begin{align*}
z &=  \begin{cases} 1 & \text{if} S_u(T)>K>S_d(T)\; ,\\
0 & \text{otherwise}\;.
\end{cases}
\end{align*}
$$
To contrast @eq-montecarlodelta5 and @eq-montecarlodelta6, note that if $y=1$ then $x=1$, so the term 
$\mathbb{E}^R\left[\frac{S_T}{S} y\right]$ in @eq-montecarlodelta6 is part of @eq-montecarlodelta5.  However, there are two partially offsetting errors in @eq-montecarlodelta6: $z$ sometimes equals one when $x$ is zero, and when both $z$ and $x$ are one, then the factor multiplying $z$ is smaller than the factor multiplying $x$.  In any case, the expected value @eq-montecarlodelta6  is not the same as the true delta @eq-montecarlodelta5.  As noted before, this implies that the delta estimate will be incorrect even if its standard error is zero.  The bias can be made as small as one wishes by taking the magnitude $S_u-S_d$ of the perturbation to be small, but taking the perturbation to be very small will introduce unacceptable roundoff error.

The obvious way to estimate the delta in this situation is simply to compute the discounted sample average of $[S_T/S]x$.  This is called a pathwise estimate \index{pathwise Monte Carlo Greeks} of the delta, because it only uses the sample paths of $S_t$ rather than considering up and down perturbations.  This method is due to Broadie and Glasserman [@bg2]. Because the pathwise estimate is a sample average, its standard error can be computed in the usual way.

To compute pathwise estimates in other models and for other Greeks, we need the Greek to be an expectation as on the right-hand side of @eq-montecarlodelta4.  Additional examples can be found in  Glasserman [@Glasserman] and J$\ddot{\text{a}}$ckel [@Jackel].



## Monte Carlo Models for Path-Dependent Options
A derivative is said to be path dependent \index{path-dependent option} if its value depends on the path of the underlying asset price rather than just on the  price at the time of exercise.  Examples of path-dependent options are lookbacks, barrier options, and Asians.
To value a path-dependent option by Monte Carlo, we need to simulate an approximate path of the stock price.  We do this by considering time periods of length $\Delta t = T/N$ for some integer $N$.  Under the risk-neutral probability, the logarithm of the stock price changes over such a time period by
$$
\Delta \log S = \nu\,\Delta t + \sigma\sqrt{\Delta t}\,z\;,
$$ {#eq-pathdependent}

where $\nu = r-q-\sigma^2/2$ and $z$ is a standard normal.  Given that there are $N$ time periods of length $\Delta t$, we need to generate $N$ standard normals to generate a stock price path.  If we generate $M$ paths to obtain a sample of $M$ option values, then we will need to generate $MN$ standard normals.  


Consider for example a floating-strike lookback call. \index{lookback option} The formula for this option given in @sec-s:lookbacks assumes the minimum stock price is computed over the entire path of the stock price, i.e., with continuous sampling of the stock price.  In practice, the minimum will be computed by recording the price at a discrete number of dates.  We can value the discretely sampled lookback using Monte-Carlo by choosing $\Delta t$ to be the interval of time (e.g., a day or week) at which the price is recorded.  For example, if the contract calls for weekly observation, we will attain maximum precision by setting $N$ to be the number of weeks before the option matures.

For most path dependent options, a possible starting point is to generate an array of $n$ paths but since we want the entire path we choose the number of time steps that is appropriate for our application.  We can use the same code as in @#sec-s:mc_europeans if we are working in a Black Scholes setting.

```{python}
# Simulate Geometric Brownian Motion
import numpy as np
import matplotlib.pyplot as plt
# number of paths
n = 1000
#number of divisions
m = 1000
# Interest rate (We set the drift equal to the interest rate for the risk-neutral probability)
r = 0.1
# Dividend yield
q=0.0
# Volatility
sig = 0.2
# Initial Stock Price
S0 = 42
# Maturity
T = 0.5
#Strike Price
K=40
# Delta t
dt = T/m
# Drift
drift = (r-q-0.5*sig**2)
# Volatility
vol = sig * np.sqrt(dt)

t = np.array(range(0,m + 1,1)) * dt

# seed for random generator
seed= 2024
# define a random generator
np.random.seed(seed)
inc = np.zeros(shape = (m + 1, n))
inc[1:] = np.transpose(np.random.normal(loc = 0, scale = vol,size = (n,m)))
St = np.zeros(shape = (m + 1, n))
St = S0 * np.exp(np.cumsum(inc,axis=0) + (drift * t[0:m + 1])[:,None])
St1 = S0 * np.exp(-np.cumsum(inc,axis=0) + (drift * t[0:m + 1])[:,None])

```

As before this code generates two samples the original and the antithetic.  The output is an array of $n$ sample paths with $m$ time steps.  The sample can also be used to find the value of a floating strike lookback call.

```{python}

def floating_strike_call(S, r, sigma, q, T, SMin):
    d1 = (np.log(S / SMin) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    d2prime = (np.log(SMin / S) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    N1 = norm.cdf(d1)
    N2 = norm.cdf(d2)
    N2prime = norm.cdf(d2prime)
    x = 2 * (r - q) / (sigma ** 2)
    return np.exp(-q * T) * S * N1 - np.exp(-r * T) * SMin * N2 + (1 / x) * (SMin / S) ** x * np.exp(-r * T) * SMin * N2prime - (1 / x) * np.exp(-q * T) * S * (1 - N1)

S = 100
r = 0.05
sigma = 0.2
q = 0.02
T=1

Stmin=St[m:]-np.minimum(np.min(St,axis=0),S0)
St1min=St1[m:]-np.minimum(np.min(St1,axis=0),S0)
floatlkbk=np.exp(-r*T)*np.mean(Stmin)
floatlkbk1=np.exp(-r*T)*np.mean(St1min)

print('The first estimate is=',floatlkbk)
print('The second estimate is=',floatlkbk1)
print('The average estimate is=',(floatlkbk+floatlkbk1)/2)
print('The exact formula is=',floating_strike_call(S0, r, sigma, 0, T, S0))

```

To value the fixed strike lookback call option with time $T$ payoff $\max(\max_{0\le t \le T} S_t.0)$, we simply add the following 
```{python}
Stmax=np.maximum(np.max(St,axis=0)-K,0)
St1max=np.maximum(np.max(St1,axis=0)-K,0)
lookbck = np.exp(-r*T) *np.mean(Stmax)
lookbck1=np.exp(-r * T)*np.mean(St1max)
print('The first estimate is=',lookbck)
print('The second estimate is=',lookbck1)
print('The average estimate is=', (lookbck + lookbck1)/2)

```


Asian and barrier options are also subject to discrete rather than continuous sampling and can be valued by Monte-Carlo in the same way as lookbacks. 


As another example,  consider the classic case of estimating the value of a discretely-sampled average-price call, using a discretely-sampled geometric-average-price call \index{average-price option} \index{geometric-average option} as a control variate.  Let $\tau$ denote the amount of time that has elapsed since the call was issued and $T$ the amount of time remaining before maturity, so the total maturity of the call is $T+\tau$.  To simplify somewhat,  assume date $0$ is the beginning of a period between observations.  Let $t_1, \ldots, t_N$ denote the remaining sampling dates, with $t_1 = \Delta t$, $t_i-t_{i-1}=\Delta t = T/N$ for each $i$, and $t_N=T$.  We will input the average price $A_0$ computed up to date $0$, assuming this average includes the price $S_0$ at date $0$.  The average price at date $T$ will be 
$$
A_T = \frac{\tau}{T+\tau}A_0 + \frac{T}{T+\tau}\left(\frac{\sum_{i=1}^N S_{t_i}}{N}\right)\;.
$$
The average-price call pays $\max(0,A_T-K)$ at its maturity $T$, and we can write this as
\begin{align*}
\max(A_T-K,0) &= \max\left(\frac{T}{T+\tau}\left( \frac{\sum_{i=1}^N S_{t_i}}{N}\right) - \left(K - \frac{\tau}{T+\tau}A_0\right), 0\right)\\
&= \frac{T}{T+\tau} \max \left(\frac{\sum_{i=1}^N S_{t_i}}{N} - K^*,0\right)\;,
\end{align*}
where 
$$
K^* = \frac{T+\tau}{T}K - \frac{\tau}{T}A_0\;.
$$
Therefore, the value at date $0$ of the discretely-sampled average-price call is
$$
\frac{T}{T+\tau} \,\mathrm{e}^{-rT} \mathbb{E}^R\left[\max \left(\frac{\sum_{i=1}^N S_{t_i}}{N} - K^*,0\right)\right]\;.
$$
In terms of the discussion above, the random variable  the mean of which we want to estimate is
$$
x = \mathrm{e}^{-rT}\max \left(\frac{\sum_{i=1}^N S_{t_i}}{N} - K^*,0\right)\;.
$$
A random variable $y$ that will be closely correlated to $x$ is
$$
y =\mathrm{e}^{-rT}\max \left(\mathrm{e}^{\sum_{i=1}^N \log S_{t_i}/N} - K^*,0\right)\;.
$$
The mean $\phi$ of $y$ under the risk-neutral probability is given in the pricing @eq-disc_geom_avg_call. 
We can use the sample mean of $y$ and its known mean $\phi$ to adjust the sample mean of $x$ as an estimator of the value of the average-price call.  Generally, the estimated adjustment coefficient $\hat{\beta}$ will be quite close to 1.

Again we can get a sample of payoffs using our stock price samples.

```{python}
average = np.mean(St,axis=0)
average1 = np.mean(St1,axis=0)
dpayoff=np.exp(-r*T)*np.mean(np.maximum(average-K,0))
dpayoff1=np.exp(-r*T)*np.mean(np.maximum(average1-K,0))
print('The first estimate is=',dpayoff)
print('The second estimate is=',dpayoff1)
print('The average of the estimates=',(dpayoff+dpayoff1)/2)

```

We now construct a control variate, the geometric asian option which has a known formula for its value.

```{python}
def black_scholes_call(S, K, r, sigma, q, T):
    """
    Inputs:
    S = initial stock price
    K = strike price
    r = risk-free rate
    sigma = volatility
    q = dividend yield
    T = time to maturity
    """
    if sigma == 0:
        return max(0, np.exp(-q * T) * S - np.exp(-r * T) * K)
    else:
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        N1 = norm.cdf(d1)
        N2 = norm.cdf(d2)
        return np.exp(-q * T) * S * N1 - np.exp(-r * T) * K * N2

def discrete_geom_average_price_call(S, K, r, sigma, q, T, N):
    dt = T / N
    nu = r - q - 0.5 * sigma ** 2
    a = N * (N + 1) * (2 * N + 1) / 6
    V = np.exp(-r * T) * S * np.exp(((N + 1) * nu / 2 + sigma ** 2 * a / (2 * N ** 2)) * dt)
    sigavg = sigma * np.sqrt(a) / (N ** 1.5)
    return black_scholes_call(V, K, r, sigavg, q, T)

geom=np.exp((np.mean(np.log(St),axis=0)))
geom1=np.exp((np.mean(np.log(St1),axis=0)))
geomavgpo=np.maximum(geom-K,0)
geomavg1po=np.maximum(geom1-K,0)
value=np.mean(geomavgpo)*np.exp(-r*T)
value1=np.mean(geomavg1po)*np.exp(-r*T)
tga=discrete_geom_average_price_call(S0, K, r, sigma, q, T, m)
error =tga-value
error1=tga-value1
print('The estimate from the first sample=',value)
print('The estimate from the second sample=',value1)
print('The average of the two estimates is=',(value+value1)/2)
print('The value from the exact formula=',tga)
print('The error in the first estimate=',error)
print('The error in the second estimate=',error1)

```

Next we estimate the beta.  As discussed before, we could simply set beta=1. Alternatively, if we estimate beta from the  simulated sample, then our update could be biased.  Instead we compute an independent sample from which we estimate beta.  We then estimate the updated estimate for both samples from the formula
$$
 \text{new estimate} = \text{original estimate} + \beta * \text{error}
 $$

```{python}
incpre = np.zeros(shape = (m + 1, n))
incpre[1:] = np.transpose(np.random.normal(loc = 0, scale = vol,size = (n,m)))
Stpre = np.zeros(shape = (m + 1, n))
St1pre=np.zeros(shape = (m + 1, n))
Stpre = S0 * np.exp(np.cumsum(inc,axis=0) + (drift * t[0:m + 1])[:,None])
St1pre = S0 * np.exp(-np.cumsum(inc,axis=0) + (drift * t[0:m + 1])[:,None])

amean=np.mean(Stpre,axis=0)
amean1=np.mean(St1pre,axis=0)
apo = np.maximum(amean-K,0)
a1po=np.maximum(amean1-K,0)
gmean=np.exp(np.mean(np.log(St),axis=0))
g1mean=np.exp(np.mean(np.log(St1),axis=0))
gpo=np.maximum(gmean-K,0)
g1po=np.maximum(g1mean-K,0)
beta=np.cov(gpo,apo)[0,1]/np.cov(gpo,apo)[1,1]
beta1=np.cov(g1po,a1po)[0,1]/np.cov(g1po,a1po)[1,1]
update=dpayoff +beta*error
update1=dpayoff1+beta1*error1

print('The updated estimate for the first sample=',update)
print('The updated value for the second sample=',update1)
print('The average of the updated values is=',(update +update1)/2)
```


## Monte Carlo Valuation of Basket and Spread Options {#sec-montecarlomultiple}

\index{basket option} \index{spread option} \index{Monte Carlo} In this section, we will consider the valuation of European spread and basket options by the Monte Carlo method.  There are no simple formulas for these options. In each simulation, we will generate a terminal price for each of the underlying assets and compute the value of the option at its maturity. 
Discounting the average terminal value gives the estimate of the option value as usual.  

The difference between binomial and Monte Carlo methods for options written on multiple assets can be understood as follows.  Both methods attempt to estimate the discounted expected value of the option (under the risk-neutral probability).  In an $N$--period model, the binomial model produces $N+1$ values for the terminal price of each underlying asset.  Letting $k$ denote the number of underlying assets, this produces $(N+1)^k$ combinations of asset prices.  Of course, each combination has an associated probability.  In contrast, the Monte Carlo method produces $M$ combinations of terminal prices, where $M$ is the number of simulations.  Each combination is given the same weight ($1/M$) when estimating the expected value.  

With a single underlying asset, the binomial model is more efficient, as discussed in @sec-s:introbinomial, because the specifically chosen terminal prices in the binomial model sample the set of possible terminal prices more efficiently than randomly generated terminal prices.  However, this advantage disappears, and the ranking of the methods can be reversed, when there are several underlying assets.  The reason is that many of the $(N+1)^k$ combinations of prices in the binomial model will have very low probabilities.  For example, with two assets that are positively correlated, it is very unlikely that one asset will be at its highest value in the binomial model and the other asset simultaneously at its lowest.  It is computationally wasteful to evaluate the option for such a combination, because the probability-weighted value will be very small and hence contribute little to the estimate of the expected value.  On the other hand, each set of terminal prices generated by the Monte Carlo method will be generated from a distribution having the assumed correlation.  Thus, only relatively likely combinations will typically be generated, and time is not wasted on evaluating unlikely combinations.  However, it should not be concluded that Monte Carlo valuation of a derivative on multiple assets will be quick and easy---even though the computation time required for more underlying assets does not increase as much with Monte Carlo as for binomial models, it can nevertheless be substantial.

To implement Monte Carlo valuation of options on multiple assets, we must first explain how to simulate correlated asset prices.
We can simulate the changes in two Brownian motions $B_1$ and $B_2$ that have correlation $\rho$ by generating two independent standard normals $Z_1$ and $Z_2$ and defining
$$
\Delta B_1 = \sqrt{\Delta t}\,Z_1\;, \qquad \text{and} \qquad \Delta B_2 = \sqrt{\Delta t}\,Z\; ,
$$
where $Z$ is defined as 
$$
Z = \rho Z_1 + \sqrt{1-\rho^2}\,Z_2\;.
$$
The random variable $Z$ is also a standard normal, and the correlation between $Z_1$ and $Z$ is $\rho$.  

```{python}
# Simulate 2 Geometric Brownian Motions
import numpy as np
import matplotlib.pyplot as plt
# number of paths
n = 1000
#number of divisions
m = 1000
# Interest rate (We set the drift equal to the interest rate for the risk-neutral probability)
r = 0.1
# Dividend yield
q1=0.0
q2=0
# Volatility
sig1 = 0.2
sig2=.3
# correlation
rho=0.5
# Initial Stock Price
S0 = 42
V0 = 50
# Maturity
T = 0.5

# Delta t
dt = T/m
# Drift
drift1 = (r-q1-0.5*sig1**2)
drift2 = (r-q2-0.5*sig2**2)
# Volatility
vol = np.sqrt(dt)

t = np.array(range(0,m + 1,1)) * dt

# seed for random generator
seed= 2024
# define a random generator
np.random.seed(seed)
inc = np.zeros(shape = (m + 1, n))
inc[1:] = np.transpose(np.random.normal(loc = 0, scale = vol,size = (n,m)))
inc1 = np.zeros(shape = (m + 1, n))
inc1[1:] = np.transpose(np.random.normal(loc = 0, scale = vol,size = (n,m)))
incr = np.zeros(shape = (m + 1, n))
incr = rho*inc + np.sqrt(1-rho**2)*inc1
```

Thus, we can simulate the changes in the logarithms of two correlated asset prices as
\begin{align*}
\Delta \log S_1 &= \nu_1\Delta t + \sigma_1\sqrt{\Delta t}Z_1 \; ,\\
\Delta \log S_2 &= \nu_2\Delta t + \sigma_2\rho\sqrt{\Delta t}Z_1 + \sigma_2\sqrt{1-\rho^2}\sqrt{\Delta t}Z_2\;,
\end{align*}
where $\nu_i = r-q_1-\sigma_i^2/2$ and the $Z_i$ are independent standard normals. 

```{python}

St1 = np.zeros(shape = (m + 1, n))
St2 = np.zeros(shape = (m + 1, n))
St1 = S0 * np.exp(sig1*np.cumsum(inc,axis=0) + (drift1 * t[0:m + 1])[:,None])
St2 = V0 * np.exp(sig2*np.cumsum(incr,axis=0) + (drift2 * t[0:m + 1])[:,None])

```

We can also construct antithetic variables.

```{python}

St1a = np.zeros(shape = (m + 1, n))
St2a = np.zeros(shape = (m + 1, n))
St1a = S0 * np.exp(-sig1*np.cumsum(inc,axis=0) + (drift1 * t[0:m + 1])[:,None])
St2a = V0 * np.exp(-sig2*np.cumsum(incr,axis=0) + (drift2 * t[0:m + 1])[:,None])

```

Given this sample, we can estimate the value of a best of 2 option with payoff $\max(S_{1T},S_{2T})$.

```{python}
payoff = np.maximum(St1[m,:],St2[m,:])
payoffa = np.maximum(St1a[m,:],St2a[m,:])
value= np.exp(-r*T)*np.mean(payoff)
valuea= np.exp(-r*T)*np.mean(payoffa)

print('The first estmate is =',value)
print('The second estimate is =',valuea)
print('The avergae of the estimates is=',(value+valuea)/2)
```

To generalize this idea to more than two assets, we introduce some additional notation.  The simulation for the case of two assets can be written as


$$
\Delta \log S_1 = \nu_1\Delta t + a_{11}\sqrt{\Delta t}Z_1 + a_{12}\sqrt{\Delta t}Z_2\;,
$$ {#eq-mc_twoa}

$$
\Delta \log S_2 = \nu_2\Delta t + a_{21}\sqrt{\Delta t}Z_1 + a_{22}\sqrt{\Delta t}Z_2\;,
$$ {#eq-mc_twob}



where
$$\begin{array}{rclcrcl}
a_{11}&=&\sigma_1\;, &\qquad & a_{12}&=&0\; ,\\
a_{21}&=&\sigma_2\rho\;, &\qquad & a_{22} &= &\sigma_2\sqrt{1-\rho^2}\;.
\end{array}
$$

These are not the only possible choices for the constants $a_{ij}$.  Given that $Z_1$ and $Z_2$ are independent standard normals, the conditions the $a_{ij}$ must satisfy in order to match the variances $\sigma_i^2\Delta t$ and correlation $\rho$ of the changes in the logarithms are


$$
a_{11}^2+a_{12}^2 =\sigma_1^2\;,
$$ {#eq-a1}

$$
a_{21}^2+a_{22}^2 =\sigma_2^2\;,
$$ {#eq-a2}

$$
a_{11}a_{21}+a_{12}a_{22} = \sigma_1\sigma_2\rho\;.
$$ {#eq-a3}



These three equations in the four coefficients $a_{ij}$ leave one degree of freedom.  We choose to take $a_{12}=0$ and then solve for the other three.

In matrix notation, the system @eq-a1 - @eq-a3 plus the condition $a_{12}=0$ can be written as the equation
$$
\begin{pmatrix}a_{11} & 0 \\a_{21} & a_{22}\end{pmatrix}\begin{pmatrix}a_{11} & 0 \\a_{21} & a_{22}\end{pmatrix}^\top = \begin{pmatrix}\sigma_1^2 & \rho\sigma_1\sigma_2 \\\rho\sigma_1\sigma_2 & \sigma_2^2\end{pmatrix}\; ,
$$
where $^\top$ denotes the matrix transpose.  The matrix on the right hand side is the covariance matrix \index{covariance matrix} of the continuously-compounded annual returns (changes in log asset prices).  Choosing the $a_{ij}$ so that the lower triangular matrix \index{lower triangular matrix}
$$
A \equiv \begin{pmatrix}a_{11} & 0 \\a_{21} & a_{22}\end{pmatrix}
$$
satisfies 
$$
AA^\top = \text{covariance matrix}
$$
is called the \index{Cholesky decomposition}
the Cholesky decomposition of the covariance matrix.  Given any number $L$ of assets, provided none of the assets is redundant (perfectly correlated with a portfolio of the  others), the Cholesky decomposition of the $L\times L$ covariance matrix always exists. An algorithm for computing the Cholesky decomposition in numpy is [np.linalg.cholesky](https://numpy.org/doc/2.2/reference/generated/numpy.linalg.cholesky.html). 

We can use the Cholesky decomposition to perform Monte-Carlo valuation of a basket or spread option.^[For a spread option, take $L=2$, $w_1=1$ and $w_2=-1$.]  If there were some path dependency in the option value, we would simulate the paths of the asset prices as in @eq-mc_twoa - @eq-mc_twob.  However a standard basket option is not path dependent, so we only need to simulate the asset prices at the option maturity date $T$, as in @sec-s:mc_europeans.  The value of a basket call option at its maturity $T$ is 
$$
\max\left(0,\;\sum_{i=1}^L w_iS_i(T)-K\right)\; ,
$$
where $L$ is the number of assets in the basket (portfolio) and $w_i$ is the weight of the $i$--th asset in the basket.
The logarithm of the $i$--th asset price at maturity is simulated as
$$
\log S_i(T) = \log S_i(0) +\nu_iT + \sqrt{T} \sum_{j=1}^L a_{ij}Z_j\; ,
$$
where the $Z_j$ are independent standard normals.  Given the simulated values of the $\log S_i(T)$, the value at maturity of the basket option is readily computed.  The estimate of the date--0 value is then computed as the discounted average of the simulated values at maturity.  

For our two asset example we compute the value of a call opttion on an equally weighted porfotlio.

```{python}
w=0.5
K=45
basketpo=np.maximum(w*St1[m,:]+(1-w)*St2[m,:]-K,0)
basketpoa=np.maximum(w*St1a[m,:]+(1-w)*St2a[m,:]-K,0)
estimate=np.exp(-r*T)*np.mean(basketpo)
estimatea=np.exp(-r*T)*np.mean(basketpoa)
print('The first estimate is =',estimate)
print('The second estimate is =',estimatea)
print('The average of the estimates=',(estimate+estimatea)/2)

```

Below is a three asset basket option whihc uses the numpy cholesky decomposition.  In contrast to the above routine, this routine is does not have the option to  generate the entire path, although this can be easily modeified.

```{python}
import numpy as np
#risk free rate
r=0.1
# number of assets
k=3

# number of paths
n=100000
# Horizon
T=0.5

# Initial price

S0=[42,50,45]

# Basket Weights
w=[.25,.5,.25]

#Strike Price

K=45

#  put in volatilities
sig1=.2
sig2=.3
sig3=.4

#create diagonal
sig=[sig1,sig2,sig3]

S=np.diag(sig)

# drift of log returns

drift= r*np.ones(k) -0.5*np.dot(S@S,np.ones(k))


# correlation matrix

rho=np.array([[1.0, 0.5, 0.3],
                  [0.5, 1.0, 0.2],
                  [0.3, 0.2, 1.0]])
# covariance matrix

V = S@rho@S

# generate uniform n*k normal uncorrelated random variables

seed=2024
np.random.seed(seed)

inc1=np.transpose(np.random.normal(loc = 0, scale = np.sqrt(T),size = (n,k)))
 


# create correlated random variables
Z=np.linalg.cholesky(V)
incr=np.dot(Z,inc1)

print('The sample correlation matrix =',np.corrcoef(incr))
print('The input correlation matrix =',rho)




St = S0 * np.exp(drift *T + np.transpose(incr))
#antithetic sample
St1 = S0 * np.exp( drift * T - np.transpose(incr))

estimate=np.mean(St,axis=0)*np.exp(-r*T)
estimate1=np.mean(St1,axis=0)*np.exp(-r*T)
print('The average discounted stock price averaged over both samples=',(estimate+estimate1)/2)
print('The initial Srock Price input =',S0)

basketpo=np.maximum(w@np.transpose(St)-K,0)
basketpo1=np.maximum(w@np.transpose(St1)-K,0)
value=np.mean(basketpo)*np.exp(-r*T)
value1=np.mean(basketpo1)*np.exp(-r*T)
print('The first sample estimate of the basket option value=',value)
print('The second sample estimate of the basket option value=',value1)
print('The average estimate of the basket option value=',(value+value1)/2)
```

We can generate the entire path of multiple assets to value, for example, lookback options on a basket.  The code below values the same European basket option as above only it calculates $m=2$ time steps;  a lookback can be created by changing the payoffs and increasing $m$.

```{python}
import numpy as np
#number of time steps
m=2
# number of assets
k=3
# number of sample paths
n=100000

#risk free rate
r=0.1



# Horizon
T=0.5
# delta t
dt=T/m
# Initial price

S0=[42,50,45]

#Strike Price

K=45

#  put in volatilities
sig1=.2
sig2=.3
sig3=.4

#create diagonal
sig=[sig1,sig2,sig3]

S=np.diag(sig)


# correlation matrix

rho=np.array([[1.0, 0.5, 0.3],
                  [0.5, 1.0, 0.2],
                  [0.3, 0.2, 1.0]])

# covariance matrix

V = S@rho@S

# drift of log returns
drift= np.array(r*np.ones(k) -0.5*np.dot(S@S,np.ones(k)))*dt

# times vector
t=np.array(range(1,m + 1,1))


driftv = np.transpose(np.kron(drift,t).reshape(3,m))

# generate uniform n(paths)*k(assets)*m(time steps) normal uncorrelated random variables

seed=2024
np.random.seed(seed)

inc=np.random.normal(loc = 0, scale = np.sqrt(dt),size = (n,k,m))

#create correlated random increments
Z=np.linalg.cholesky(V)
# numpy matmul assumes last two define matrix multiplication
incr=np.matmul(Z,inc)



SSt=S0*np.exp(driftv)

# generate returns along path and antithetic path
# first e^{cumsum(increments)} gives e^sigma B_t for different t

Stb = np.exp(np.cumsum(incr[:,:,],axis=2))
Stb1 = np.exp(-np.cumsum(incr[:,:,],axis=2))


#Multiply by S0 e^drift for each t
St=np.multiply(Stb,np.transpose(SSt))
St1=np.multiply(Stb1,np.transpose(SSt))

#Last date returns
Stm=St[:,:,m-1]
Stm1=St1[:,:,m-1]

#define payoff
# Basket Weights
w=[.25,.5,.25]


payoff= np.maximum(np.matmul(Stm,np.transpose(w))-K,0)
payoff1= np.maximum(np.matmul(Stm1,np.transpose(w))-K,0)


value= np.exp(-r*T)*np.mean(payoff)
value1= np.exp(-r*T)*np.mean(payoff1)
print('The estimate for the first sample value=',value)
print('The estimate for the second sample value=',value1)
print('The average estimate for the value=',(value+value1)/2)

```


