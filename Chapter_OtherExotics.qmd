{{< include macros.qmd >}}

# Other Exotics {#sec-c:exotics} 

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


We study six classes of exotic options:

- Barrier options (knock-outs and knock-ins)
- Lookback options
- Compound options
- Options of the max or min of two asset prices
- Forward start options
- Choosers


## Barrier Options {#sec-s:barriers}

\index{barrier option} \index{down-and-out option} \index{down-and-in option} \index{knock-out option} \index{knock-in option} A down-and-out call pays the usual call value at maturity if and only if the stock price does not hit a specified lower bound during the lifetime of the option.  If it does breach the lower barrier, then it is out.  Conversely, a down-and-in call pays off only if the stock price does hit the lower bound.  Up-and-out and up-and-in calls are defined similarly, and there are also put options of this sort.  The out versions are called knock-outs and the in versions are called knock-ins.  

Knock-ins can be priced from knock-outs and vice-versa.  For example, the combination of a down-and-out call and a down-and-in call creates a standard European call, so the value of a down-and-in can be obtained by subtracting the value of a down-and-out from the value of a standard European call.  Likewise, up-and-in calls can be valued by subtracting the value of an up-and-out from the value of a standard European call.  Both knock-outs and knock-ins are of course less expensive than comparable standard options.


We will describe the pricing of a down-and-out call.  The pricing of up-and-out calls and knock-out puts is similar.  Often there are rebates associated with the knocking-out of a barrier option, but we will not include that feature here.  

A down-and-out call provides a hedge against an increase in an asset price, just as does a standard call, for someone who is short the asset.  The difference is that the down-and-out is knocked out when the asset price falls sufficiently.  Presumably this is acceptable to the buyer because the need to hedge against high prices diminishes when the price falls.  In fact, in this circumstance the buyer may want  to establish a new hedge at a lower strike.  However, absent re-hedging at a lower strike, the buyer of a knock-out call obviously faces the risk that the price may reverse course after falling to the knock-out boundary, leading to regret that the option was knocked out.  The rationale for accepting this risk is  that the knock-out is cheaper than a standard call.  Thus, compared to a standard call, a down-and-out call provides cheaper but incomplete insurance.

The combination of a knock-out call and a knock-in call (or knock-out puts) with the same barrier and different strikes creates an option with a strike that is reset when the barrier is hit.  This is a hedge that adjusts automatically to the market.  An example is given in Probs.~\ref{e_standardknockout3} and~\ref{e_standardknockout4}.

#### Down-and-Out Call Payoff
Let $L$ denote the lower barrier for the down-and-out call and assume it has not yet been breached at the valuation date, which we are calling date $0$.  Denote the minimum stock price realized during the remaining life of the contract by $z = \min_{0\leq t\leq T} S_t$.  In practice, this minimum is calculated at discrete dates (for example, based on daily closing prices), but we will assume here that the stock price is monitored continuously for the purpose of calculating the minimum. 

The down-and-out call will pay $\max(0,S_T-K)$ if $z > L$ and 0 otherwise, 
at its maturity T.  
Let
$$
x = \begin{cases} 1 & \text{if $S_T>K$ and $z > L$;,}\\
0 & \text{otherwise;.} \end{cases}
$$
Then the value of the down-and-out call at maturity is
$$
xS_T - xK\; .
$$

#### Numeraires
As in other cases, the value at date $0$ can be written as
$$
\mathrm{e}^{-qT}S_0\times\text{prob}^{V}(x=1) - \mathrm{e}^{-rT}K\times\text{prob}^{R}(x=1)\; ,
$$

where $V_t = \mathrm{e}^{qt}S_t$ and $R_t=\mathrm{e}^{rt}$.

#### Calculating Probabilities
To calculate $\text{prob}^{V}(x=1)$ and $\text{prob}^{R}(x=1)$, we consider two cases.  


1. Suppose $K>L$.  Define
$$
y = \begin{cases} 1 & \text{if $S_T>K$ and $z \leq L$}\\
0 & \text{otherwise\;.} \end{cases}
$$ 

The event $S_T>K$ is equal to the union of the disjoint events $x=1$ and $y=1$.  Therefore,
\begin{align*}
\text{prob}^{V}(x=1) &= \text{prob}^{V}(S_T\!>\!K) - \text{prob}^{V}(y=1)\; ,\\
\text{prob}^{R}(x=1) &= \text{prob}^{R}(S_T\!>\!K) - \text{prob}^{R}(y=1)\;.
\end{align*}

As in the derivation of the Black-Scholes formula, we have
$$
\text{prob}^{V}(S_T\!>\!K) = \mathrm{N}(d_1) \quad \text{and} \quad \text{prob}^{R}(S_T\!>\!K) = \mathrm{N}(d_2)\;,
$$ {#eq-casea2}

where

$$
d_1= \frac{\log\left(\frac{S_0}{K}\right)+\left(r-q+\frac{1}{2}\sigma^2\right)T}{\sigma\sqrt{T}}\; ,\qquad  d_2 = d_1-\sigma\sqrt{T}\;.
$$ {#eq-casea3}

Furthermore , defining 
$$
d_1** = \frac{\log\left(\frac{L^2}{KS_0}\right)+\left(r-q+\frac{1}{2}\sigma^2\right)T}{\sigma\sqrt{T}}\;, \qquad d_2** = d_1**-\sigma\sqrt{T}\;,
$$ {#eq-casea5}

it can be shown that


$$
\text{prob}^{V}(y=1) = \left(\frac{L}{S_0}\right)^{2\left(r-q+\frac{1}{2}\sigma^2\right)/\sigma^2}\mathrm{N}(d_1**)\;,
$$ {#eq-casea4}

$$
\text{prob}^{R}(y=1) = \left(\frac{L}{S_0}\right)^{2\left(r-q-\frac{1}{2}\sigma^2\right)/\sigma^2}\mathrm{N}(d_2**)\;.
$$ {#eq-casea41}





2. Suppose $K \leq L$.  Then the condition $S_T>K$ in the definition of the event $x=1$ is redundant: if $z > L \geq K$, then it is necessarily true that $S_T>K$.  Therefore, the probability (under either numeraire) of the event $x=1$ is the probability that $z > L$.  Define
$$y = \begin{cases} 1 & \text{if $S_T>L$ and $z \leq L$;,}\\
0 & \text{otherwise;.} \end{cases}
$$

The event $S_T>L$ is the union of the disjoint events $x=1$ and $y=1$.  Therefore, as in the previous case (but now with $K$ replaced by $L$),
\begin{align*}
\text{prob}^{V}(x=1) &= \text{prob}^{V}(S_T\!>\!L) - \text{prob}^{V}(y=1)\; ,\\
\text{prob}^{R}(x=1) &= \text{prob}^{R}(S_T\!>\!L) - \text{prob}^{R}(y=1)\;.
\end{align*}
Also as before, we know that
$$
\text{prob}^{V}(S_T\!>\!L) = \mathrm{N}(d_1) \quad \text{and} \quad \text{prob}^{R}(S_T\!>\!L) = \mathrm{N}(d_2)\;,
$$ {#eq-caseb2}

where now

$$
d_1= \frac{\log\left(\frac{S_0}{L}\right)+\left(r-q+\frac{1}{2}\sigma^2\right)T}{\sigma\sqrt{T}}\; ,\qquad  d_2 = d_1-\sigma\sqrt{T}\;.
$$ {#eq-caseb3}

Moreover, $\text{prob}^{V}(y=1)$ and $\text{prob}^{R}(y=1)$ are given by @eq-casea4 - @eq-casea41 but with $K$ replaced by $L$, which means that
$$
d_1** = \frac{\log\left(\frac{L}{S_0}\right)+\left(r-q+\frac{1}{2}\sigma^2\right)T}{\sigma\sqrt{T}}\;, \qquad d_2**= d_1** - \sigma\sqrt{T}\;.
$$ {#eq-caseb5}





#### Down-and-Out Call Pricing Formula


::: Rule

The value of a continuously-sampled down-and-out call option with barrier $L$ is 

$$
\begin{multline}
\mathrm{e}^{-qT}S_0\left[\mathrm{N}(d_1)-\left(\frac{L}{S_0}\right)^{2\left(r-q+\frac{1}{2}\sigma^2\right)/\sigma^2}\mathrm{N}(d_1**)\right]\\ - \mathrm{e}^{-rT}K\left[\mathrm{N}(d_2) - \left(\frac{L}{S_0}\right)^{2\left(r-q-\frac{1}{2}\sigma^2\right)/\sigma^2}\mathrm{N}(d_2**)\right]\;,
\end{multline} 
$$ {#eq-downout100}

where 


\renewcommand{\labelenumi}{(\arabic{enumi})}
1. if $K>L$, $d_1$, $d_2$ , $d_1**$ and $d_2**$ are defined in @eq-casea3 - @eq-casea5,
2. if $K\leq L$, $d_1$, $d_2$, $d_1**$ and $d_2**$ are defined in @eq-caseb3 - @eq-caseb5.



:::

The following code computes the price of a down and out call option.

```{python}

#| code-fold: true
#| label: Down-And_Out_Call

import numpy as np
from scipy.stats import norm

def down_and_out_call(S, K, r, sigma, q, T, Barrier):
    if K > Barrier:
        a = S / K
        b = Barrier * Barrier / (K * S)
    else:
        a = S / Barrier
        b = Barrier / S

    d1 = (np.log(a) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    d1prime = (np.log(b) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2prime = d1prime - sigma * np.sqrt(T)
    N1 = norm.cdf(d1)
    N2 = norm.cdf(d2)
    N1prime = norm.cdf(d1prime)
    N2prime = norm.cdf(d2prime)
    x = 1 + 2 * (r - q) / (sigma ** 2)
    y = x - 2
    q1 = N1 - (Barrier / S) ** x * N1prime
    q2 = N2 - (Barrier / S) ** y * N2prime

    return np.exp(-q * T) * S * q1 - np.exp(-r * T) * K * q2

S=100
K=120
r=0.05
sigma=0.2
q=0
T=1
print("Down and Out Call:", down_and_out_call(S, K, r, sigma, q, T, 80))

```



## Lookbacks {#sec-s:lookbacks}

\index{lookback option} \index{floating-strike lookback option} \index{fixed-strike lookback option} A floating-strike lookback call pays the difference between the asset price at maturity and the minimum price realized during the life of the contract.  A floating-strike lookback put pays the difference between the maximum price over the life of the contract and the price at maturity.  Thus, the floating-strike lookback call allows one to buy the asset at its minimum price, and the floating-strike lookback put allows one to sell the asset at its maximum price.   Of course, one pays upfront for this opportunity to time the market.  These options were first discussed by Goldman, Sosin and Gatto [@GSG].

A fixed-strike lookback put pays the difference between a fixed strike price and the minimum price during the lifetime of the contract.   Thus, a fixed-strike lookback put and a floating-strike lookback call are similar in one respect:  both enable one to buy the asset at its minimum price.  However, the put allows one to sell the asset at a fixed price whereas the call allows one to sell it at the terminal asset price.  A fixed-strike lookback call pays the difference between the maximum price and a fixed strike price and is similar to a floating-strike lookback put in the sense that both enable one to sell the asset at its maximum price.  Fixed-strike lookback options were first discussed by Conze and Viswanathan [@CV].
We will discuss the valuation of floating-strike lookback calls. As in the discussion of barrier options, we will assume that the price is continuously sampled for the purpose of computing the minimum.

#### Floating-Strike Lookback Call Payoff

As in the previous section, let $z$ denote the minimum stock price realized over the remaining lifetime of the contract.  This is not necessarily the minimum stock price realized during the entire lifetime of the contract.  Let $S_{\min}$ denote the minimum stock price realized during the lifetime of the contract up to and including date $0$, which is the date at which we are valuing the contract.  The minimum stock price during the entire lifetime of the contract will be the smaller of $z$ and $S_{\text{min}}$.  The payoff of the floating strike lookback call is $S_T - \min\left(z, S_{\text{min}}\right)$.




#### Calculations
The value at date $0$ of the piece $S_T$ is simply $\mathrm{e}^{-qT}S_0$.  It can be shown (see, e.g., Musiela and Rutkowski [@MR] for the details) that the value at date $0$ of receiving 
$$
\min(z, S_{\text{min}})
$$
at date $T$ is
\begin{multline*}
\mathrm{e}^{-rT}S_{\text{min}}\mathrm{N}(d_2) -\frac{\sigma^2}{2(r-q)}\left(\frac{S_{\text{min}}}{S_0}\right)^{2(r-q)/\sigma^2}\mathrm{e}^{-rT}S_0\mathrm{N}(d_2**) \\
+\left(1+ \frac{\sigma^2}{2(r-q)}\right)\mathrm{e}^{-qT}S_0\mathrm{N}(-d_1)\;.
\end{multline*}
where


$$
d_1 = \frac{\log\left(\frac{S_0}{S_{\text{min}}}\right)+\left(r-q+\frac{1}{2}\sigma^2\right)T}{\sigma\sqrt{T}}\; , \qquad d_2 = d_1 - \sigma\sqrt{T}\;,
$$ {#eq-fslc100a}

$$
d_1** = \frac{\log\left(\frac{S_{\text{min}}}{S_0}\right)+\left(r-q+\frac{1}{2}\sigma^2\right)T}{\sigma\sqrt{T}}\;, \qquad d_2**=d_1** - \sigma\sqrt{T} \;.
$$ {#eq-fslc100c}


Using the fact that $[1-\mathrm{N}(-d_1)]\mathrm{e}^{-qT}S_0=\mathrm{e}^{-qT}S_0\mathrm{N}(d_1)$, this implies:

#### Floating-Strike Lookback Call Pricing Formula

::: Rule

The value at date $0$ of a continuously-sampled floating-strike lookback call, given that the minimum price during the lifetime of the contract through date $0$ is $S_{\text{min}}$ and the remaining time to maturity is $T$, is

$$
\begin{multline}
\mathrm{e}^{-qT}S_0\mathrm{N}(d_1)-\mathrm{e}^{-rT}S_{\text{min}}\mathrm{N}(d_2) \\+\frac{\sigma^2}{2(r-q)}\left(\frac{S_{\text{min}}}{S_0}\right)^{2(r-q)/\sigma^2}\mathrm{e}^{-rT}S_0\mathrm{N}(d_2**) \\
-\frac{\sigma^2}{2(r-q)}\mathrm{e}^{-qT}S_0\mathrm{N}(-d_1)\;,
\end{multline}
$$ {#eq-fslc100}

where $d_1$, $d_2$, and $d_2**$ are defined in @eq-fslc100a - @eq-fslc100c.
:::

The following program calculates the price of a floating strike lookback option. 

```{python}

#| code-fold: true
#| label: Floating_Strike_Lookback

import numpy as np
from scipy.stats import norm

def floating_strike_call(S, r, sigma, q, T, SMin):
    d1 = (np.log(S / SMin) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    d2prime = (np.log(SMin / S) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    N1 = norm.cdf(d1)
    N2 = norm.cdf(d2)
    N2prime = norm.cdf(d2prime)
    x = 2 * (r - q) / (sigma ** 2)
    return np.exp(-q * T) * S * N1 - np.exp(-r * T) * SMin * N2 + (1 / x) * (SMin / S) ** x * np.exp(-r * T) * SMin * N2prime - (1 / x) * np.exp(-q * T) * S * (1 - N1)

# Example usage

S = 100
r = 0.05
sigma = 0.2
q = 0.02
T=1


print("Floating Strike Call:", floating_strike_call(S, r, sigma, q, T, 90))

```

## Compound Options

A compound option \index{compound option} is an option on an option, for example a call option on a call option or a call on a put.   These options are useful for hedging when there is some uncertainty about the need for hedging which may be resolved by the exercise date of the compound option.  As speculative trades, they have the benefit of higher leverage than ordinary options.  These options were first discussed by Geske [@Geske].

#### Call-on-a-Call Payoff

Let the underlying call option have exercise price $K**$ and maturity $T^*$.  Consider an option maturing at $T<T^*$ to purchase the underlying call at price $K$.

Let $C(t,S)$ denote the value at date $t$ of the underlying call when the stock price is $S$ (i.e., $C$ is the Black-Scholes formula).    It is of course rational to exercise the compound call at date $T$ if the value of the underlying call exceeds $K$; i.e., if $C(T,S_T)>K$.  Let $S^*$ denote the critical price such that $C(T,S^*)=K$.  To calculate $S^*$, we need to solve

``` Black_Scholes_Call(S*,Kprime,r,sigma,q,Tprime-T) = K ```

for $S^*$.  We can do this by bisection or one of the other methods mentioned in @sec-s:impliedvolatility.
It is rational to exercise the compound option 
when $S_T > S^*$.  

When $S_T > S^*$,  exercise of the compound option generates a cash flow of $-K$ at date $T$.  There is a cash flow (of $S_{T^*}-K**$) at date $T^*$ only if the compound call is exercised and the underlying call finishes in the money.  This is equivalent to:
$$
S_T > S^* \quad\text{and}\quad S_{T^*}>K**\;.
$$ {#eq-compound1}

Let
$$
\begin{align*}
x&= \begin{cases} 1 &\text{if $S_T>S^*$;,}\\
                               0 & \text{otherwise;,}
        \end{cases}\\
y&= \begin{cases} 1 &\text{if $S_T>S^*$ and $S_{T^*}>K**$;,}\\
                               0 & \text{otherwise;.}
        \end{cases} 
\end{align*} 
$$
The cash flows of the compound option are $-xK$ at date $T$ and $yS_{T^*}-yK**$ at date $T^*$.  We can value the compound option at date $0$ by valuing these separate cash flows.

The cash flow $-xK$ is the cash flow from being short $K$ digital options on the underlying asset with strike price $S^*$ and maturity $T$.  Therefore the value at date $0$ of this cash flow is $-\mathrm{e}^{-rT}K\mathrm{N}(d_2)$, where
$$
d_1 = \frac{\log\left(\frac{S_0}{S^*}\right)+\left(r-q+\frac{1}{2}\sigma^2\right)T}{\sigma\sqrt{T}},  \qquad d_2 = d_1-\sigma\sqrt{T}\;.
$$ {#eq-calloncalld1d2}



#### Numeraires
The payoffs $yS_T$ and $yK**$ are similar to share digitals and digitals, respectively, except that the event $y=1$ is more complex than we have previously encountered.  However, we know from the analysis of share digitals and digitals that the values at date $0$ of these payoffs are 
$$
\mathrm{e}^{-q T^*}S_0\times\text{prob}^V\!(y=1) \quad \text{and}\quad \mathrm{e}^{-rT^*}K**\times\text{prob}^R(y=1)\; ,
$$
where $V_t=\mathrm{e}^{qt}S_t$ and $R_t=\mathrm{e}^{rt}$.  

#### Calculating Probabilities

We will calculate the two probabilities in terms of the bivariate normal distribution function.  


1. The event $y=1$ is equivalent to
$$
\log S_0 + \left(r-q+\frac{1}{2}\sigma^2\right)T+\sigma B^*_T > \log S^*
$$
and
$$
\log S_0 + \left(r-q+\frac{1}{2}\sigma^2\right)T^*+\sigma B^*_{T^*} > \log K**\; ,
$$
where $B^*$ is a Brownian motion when the underlying asset ($V$) is used as the numeraire. These conditions can be rearranged as

$$-\frac{B^*_T}{\sqrt{T}}<d_1 \quad \text{and} \quad - \frac{B^*_{T^*}}{\sqrt{T^*}}<d_1**\;,
$$ {#eq-new10}

where $d_1$ is defined in @eq-calloncalld1d2, and
$$
d_1** = \frac{\log\left(\frac{S_0}{K**}\right)+\left(r-q+\frac{1}{2}\sigma^2\right)T^*}{\sigma\sqrt{T^*}}\;,
 \qquad d_2**=d_1**-\sigma\sqrt{T^*}\;.
$$ {#eq-callcallds}



The two standard normal variables on the left-hand sides in @eq-new10 have a covariance equal to
$$
\frac{1}{\sqrt{TT^*}}\mathrm{cov}(B_T,B_{T^*}) = \frac{1}{\sqrt{TT^*}}\mathrm{cov}(B_T,B_T) = \sqrt{\frac{T}{T^*}}\; ,
$$

the first equality following from the fact that $B_T$ is independent of $B_{T^*}-B_T$ and the second from the fact that the covariance of a random variable with itself is its variance.
Hence, $\text{prob}^V\!(y=1)$ is the probability that $a\leq d_1$ and $b\leq d_1**$, where $a$ and $b$ are standard normal random variables with covariance (= correlation coefficient) of $\sqrt{T/T^*}$.  We will write this probability as $\mathrm{M}\!\left(d_1,d_1**,\sqrt{T/T^*}\right)$.  A program to approximate the bivariate normal \index{bivariate normal distribution function} distribution function $\mathrm{M}$ is provided later.

2. The calculation for $\text{prob}^R(y=1)$ is similar.  The event $y=1$
is equivalent to
$$
\log S_0 + \left(r-q+\frac{1}{2}\sigma^2\right)T+\sigma B^*_T > \log S^*\;,
$$
and
$$
\log S_0 + \left(r-q+\frac{1}{2}\sigma^2\right)T^*+\sigma B^*_{T^*} > \log K**\; ,
$$
where $B^*$ now denotes a Brownian motion under the risk-neutral probability.  These are equivalent to
$$
-\frac{B^*_T}{\sqrt{T}}<d_2 \quad \text{and} \quad - \frac{B^*_{T^*}}{\sqrt{T^*}} < d_2**\;.
$$ {#eq-new11000}

Hence, $\text{prob}^R(y=1)=\mathrm{M}\!\left(d_2,d_2**,\sqrt{T/T^*}\right)$.  


#### Call-on-a-Call Pricing Formula

We conclude:


::: Rule

The value of a call on a call is

$$
\begin{multline}
-\mathrm{e}^{-rT}K\mathrm{N}(d_2) + \mathrm{e}^{-q T^*}S_0\mathrm{M}\!\left(d_1,d_1**,\sqrt{T/T^*}\right)
 \\- \mathrm{e}^{-rT^*}K**\mathrm{M}\!\left(d_2,d_2**,\sqrt{T/T^*}\right)\;,
 \end{multline}
$$ {#eq-marketmodel5}

 where $d_1$ and  $d_2$ are defined in @eq-calloncalld1d2 and $d_1**$ and $d_2**$ are defined in @eq-callcallds.
:::


#### Put-Call Parity\index{put-call parity}

European compound options with the same underlyings and strikes satisfy put-call parity in the usual way:
$$
\text{Cash} + \text{Call} = \text{Underlying} + \text{Put}\; .
$$

The portfolio on each side of this equation gives the owner the maximum of the strike and the value of the underlying at the option maturity.  In the case of options on calls, put-call parity is specifically
\begin{multline*}
\mathrm{e}^{-rT}K + \text{Value of call on call} \\= \text{Value of underlying call} + \text{Value of put on call}\; ,
\end{multline*}
where $K$ is the strike price of the compound options and $T$ is their maturity date.  Likewise, for options on puts, we have
\begin{multline*}
\mathrm{e}^{-rT}K + \text{Value of call on put} \\= \text{Value of underlying put} + \text{Value of put on put}\; .
\end{multline*}
Thus, the value of a put on a call can be derived from the value of a call on a call.  The value of a put on a put can be derived from the value of a call on a put, which we will now consider.

#### Call-on-a-Put Pricing Formula

Consider a call option maturing at $T$ with strike $K$ with the underlying being a put option with strike $K**$ and maturity $T^*>T$.  The underlying of the put is the asset with price $S$ and constant volatility $\sigma$.  The call on the put will never be in the money at $T$ and hence is worthless if $K> \mathrm{e}^{-r(T^*-T)}K**$, because the maximum possible value of the put option at date $T$ is $\mathrm{e}^{-r(T^*-T)}K**$.  So assume $K< \mathrm{e}^{-r(T^*-T)}K**$.  

Let $S^*$ again denote the critical value of the stock price such that the call is at the money at date $T$ when $S_T=S^*$.  This means that $S^*$ solves

```Black_Scholes_Put(S*,Kprime,r,sigma,q,Tprime-T) = K.```
 
We leave it as an exercise to confirm the following.


::: Rule

The value of a call on a put is

$$
\begin{multline}
-\mathrm{e}^{-rT}K\mathrm{N}(-d_2) + \mathrm{e}^{-rT^*}K**\mathrm{M}\!\left(-d_2,-d_2**,\sqrt{T/T^*}\right) \\- \mathrm{e}^{-q T^*}S_0\mathrm{M}\!\left(-d_1,-d_1**,\sqrt{T/T^*}\right) \;,
 \end{multline}
$$ {#eq-callonaput}

 where $d_1$ and  $d_2$ are defined in @eq-calloncalld1d2 and $d_1**$ and $d_2**$ are defined in @eq-callcallds.
:::

We will use bisection to find the critical price $S^*$.  We can use $e^{q(T^*-T)}(K+K**)$ as an upper bound for $S^*$ and 0 as a lower bound.^[We set the value of the call to be zero when the stock price is zero.  The upper bound works because (by put-call parity and the fact that the put value is nonnegative) $C(T,S) \geq e^{-q(T^*-T)}S-e^{-r(T^*-T)}K**$.  Therefore, when $S = e^{q(T^*-T)}(K+K**)$, we have 
$C(T,S) \geq K + K** - e^{-r(T^*-T)}K** > K$.
}  
The following uses $10^{-6}$ as the error tolerance in the bisection.

```{python}

#| code-fold: true
#| label: call_on_call

import numpy as np

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


def call_on_call(S, Kc, Ku, r, sigma, q, Tc, Tu):
    tol = 1e-6
    lower = 0
    upper = np.exp(q * (Tu - Tc)) * (Kc + Ku)
    guess = 0.5 * lower + 0.5 * upper
    flower = -Kc
    fupper = black_scholes_call(upper, Ku, r, sigma, q, Tu - Tc) - Kc
    fguess = black_scholes_call(guess, Ku, r, sigma, q, Tu - Tc) - Kc
    while upper - lower > tol:
        if fupper * fguess < 0:
            lower = guess
            flower = fguess
            guess = 0.5 * lower + 0.5 * upper
            fguess = black_scholes_call(guess, Ku, r, sigma, q, Tu - Tc) - Kc
        else:
            upper = guess
            fupper = fguess
            guess = 0.5 * lower + 0.5 * upper
            fguess = black_scholes_call(guess, Ku, r, sigma, q, Tu - Tc) - Kc
    Sstar = guess

    d1 = (np.log(S / Sstar) + (r - q + sigma ** 2 / 2) * Tc) / (sigma * np.sqrt(Tc))
    d2 = d1 - sigma * np.sqrt(Tc)
    d1prime = (np.log(S / Ku) + (r - q + sigma ** 2 / 2) * Tu) / (sigma * np.sqrt(Tu))
    d2prime = d1prime - sigma * np.sqrt(Tu)
    rho = np.sqrt(Tc / Tu)
    N2 = norm.cdf(d2)
    M1 = binormal_prob(d1, d1prime, rho)
    M2 = binormal_prob(d2, d2prime, rho)

    return -np.exp(-r * Tc) * Kc * N2 + np.exp(-q * Tu) * S * M1 - np.exp(-r * Tu) * Ku * M2

# Example usage
S = 100
Kc = 10
Ku = 100
r = 0.05
sigma = 0.2
q = 0.02
Tc = 0.5
Tu = 1


# print("Call on Call:", call_on_call(S, Kc, Ku, r, sigma, q, Tc, Tu))
```


The implementation of the call-on-a-put formula is of course very similar to that of a call-on-a-call.  One difference is that there is no obvious upper bound for $S^*$, so we start with $2K**$ (= `2*K2`) and double this until the value of the put is below $K$.  We can take 0 again to be the lower bound.  Recall that we assume $K<\mathrm{e}^{-r(T^*-T)}K**$ and the right-hand side of this is the value of the put at date $T$ when $S_T=0$.

```{python}

#| code-fold: true
#| label: Call_On_Put

def black_scholes_put(S, K, r, sigma, q, T):
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
        return max(0, np.exp(-r * T) * K - np.exp(-q * T) * S)
    else:
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        N1 = norm.cdf(-d1)
        N2 = norm.cdf(-d2)
        return np.exp(-r * T) * K * N2 - np.exp(-q * T) * S * N1

def call_on_put(S, Kc, Ku, r, sigma, q, Tc, Tu):
    tol = 1e-6
    lower = 0
    flower = np.exp(-r * (Tu - Tc)) * Ku - Kc
    upper = 2 * Ku
    fupper = black_scholes_put(upper, Ku, r, sigma, q, Tu - Tc) - Kc
    while fupper > 0:
        upper *= 2
        fupper = black_scholes_put(upper, Ku, r, sigma, q, Tu - Tc) - Kc

    guess = 0.5 * lower + 0.5 * upper
    fguess = black_scholes_put(guess, Ku, r, sigma, q, Tu - Tc) - Kc
    while upper - lower > tol:
        if fupper * fguess < 0:
            lower = guess
            flower = fguess
            guess = 0.5 * lower + 0.5 * upper
            fguess = black_scholes_put(guess, Ku, r, sigma, q, Tu - Tc) - Kc
        else:
            upper = guess
            fupper = fguess
            guess = 0.5 * lower + 0.5 * upper
            fguess = black_scholes_put(guess, Ku, r, sigma, q, Tu - Tc) - Kc
    Sstar = guess

    d1 = (np.log(S / Sstar) + (r - q + sigma ** 2 / 2) * Tc) / (sigma * np.sqrt(Tc))
    d2 = d1 - sigma * np.sqrt(Tc)
    d1prime = (np.log(S / Ku) + (r - q + sigma ** 2 / 2) * Tu) / (sigma * np.sqrt(Tu))
    d2prime = d1prime - sigma * np.sqrt(Tu)
    rho = np.sqrt(Tc / Tu)
    N2 = norm.cdf(-d2)
    M1 = binormal_prob(-d1, -d1prime, rho)
    M2 = binormal_prob(-d2, -d2prime, rho)

    return -np.exp(-r * Tc) * Kc * N2 + np.exp(-r * Tu) * Ku * M2 - np.exp(-q * Tu) * S * M1
# Example usage
S = 100
Kc = 10
Ku = 100
r = 0.05
sigma = 0.2
q = 0.02
Tc = 0.5
Tu = 1

# print("Call on Put:", call_on_put(S, Kc, Ku, r, sigma, q, Tc, Tu))

```


## Options on the Max or Min

We will consider here an option written on the maximum or minimum of two asset prices; for example, a call on the maximum pays
$$
\max(0,\max(S_1(T), S_2(T))-K) = \max(0,S_1(T)-K,S_2(T)-K)
$$
at maturity $T$.  There are also call options on $\min(S_1(T), S_2(T))$ and put options on the maximum and minimum of two (or more) asset prices.  Pricing formulas for these options are due to Stulz [@Stulz], who also discusses applications.  We will assume the two assets have constant dividend yields $q_i$, constant volatilities $\sigma_i$, and a constant correlation $\rho$.

\vfil\eject
#### Call-on-the-Max Payoff
To value the above option, define the random variables:
\begin{align*}
x&= \begin{cases} 1 & \text{if $S_1(T)>S_2(T)$ and $S_1(T)>K$}\; ,\\
0 & \text{otherwise}\;, \end{cases}\\
y&= \begin{cases} 1 & \text{if $S_2(T)> S_1(T)$ and $S_2(T)>K$}\; ,\\
0 & \text{otherwise}\;, \end{cases}\\
z&= \begin{cases} 1 & \text{if $S_1(T) > K$ or $S_2(T)> K$}\; ,\\
0 & \text{otherwise}\;. \end{cases}
\end{align*}
Then the value of the option at maturity is
$$
xS_1(T) + yS_2(T) - zK\; .
$$

#### Numeraires
Consider numeraires  $V_1(t) = \mathrm{e}^{q_1t}S_1(T)$,  $V_2(t)=\mathrm{e}^{q_2t}S_2(T)$, and $R_t=\mathrm{e}^{rt}$.
By familiar arguments, the value of the option at date $0$ is
\begin{multline*}
\mathrm{e}^{-q_1T}S_1(0)\times\text{prob}^{V_1}(x=1) + \mathrm{e}^{-q_2T}S_2(0)\times\text{prob}^{V_2}(y=1) \\- \mathrm{e}^{-rT}K\times\text{prob}^R(z=1)\; .
\end{multline*}

#### Calculating Probabilities


1.
We will begin by calculating $\text{prob}^{V_1}(x=1)$.  From the second and third examples in @sec-s:girsanov, the asset prices satisfy
\begin{align*}
\frac{\mathrm{d}  S_1}{S_1} &= (r-q_1+\sigma^2_1)\mathrm{d}   t + \sigma_1\mathrm{d}   B^*_{1}\; ,\\
\frac{\mathrm{d}  S_2}{S_2} &= (r-q_2+\rho\sigma_1\sigma_2)\mathrm{d}   t + \sigma_2\mathrm{d}   B^*_{2}\;,
\end{align*}
where $B^*_{1}$ and $B^*_{2}$ are Brownian motions when we use $V_1$ as the numeraire.
Thus,
$$
\begin{align*}
\log S_1(T) &= \log S_1(0) + \left(r-q_1+\frac{1}{2}\sigma_1^2\right)T +\sigma_1B^*_{1}(T)\; ,\\
\log S_2(T) &= \log S_2(0) + \left(r-q_2+\rho\sigma_1\sigma_2-\frac{1}{2}\sigma_2^2\right)T +\sigma_2B^*_{2}(T)\;.
\end{align*}
$$
The condition $\log S_1(T) > \log K$ is therefore equivalent to 

$$
  -\frac{1}{\sqrt{T}}B^*_{1}(T) < d_{11}\;,
$$ {#eq-max1}

and the condition $\log S_1(T)>\log S_2(T)$ is equivalent to
$$
\frac{\sigma_2B^*_{2}(T)-\sigma_1B^*_{1}(T)}{\sigma\sqrt{T}} < d_1\;,
$$ {#eq-max2}


where
$$
\sigma =\sqrt{\sigma_1^2-2\rho\sigma_1\sigma_2+\sigma_2^2}\;,
$$ {#eq-max2a}

and

$$
d_1 = \frac{\log\left(\frac{S_1(0)}{S_2(0)}\right)+\left(q_2-q_1+\frac{1}{2}\sigma^2\right)T}{\sigma\sqrt{T}}\;, \qquad d_2 = d_1 - \sigma\sqrt{T}\;,
$$ {#eq-max4}

$$
d_{11}=\frac{\log\left(\frac{S_1(0)}{K}\right)+\left(r-q_1+\frac{1}{2}\sigma_1^2\right)T}{\sigma_1\sqrt{T}}\;, \qquad d_{12} = d_{11} - \sigma_1\sqrt{T}\;.
$$ {#eq-max3}


 The random variables on the left-hand sides of @eq-max1 - @eq-max2 have standard normal distributions and their correlation is
$$
\rho_1 = \frac{\sigma_1-\rho\sigma_2}{\sigma}\; .
$$
Therefore,
\begin{equation*}
\text{prob}^{V_1}(x=1) = \mathrm{M}(d_{11},d_1,\rho_1)\;,
\end{equation*}
where $\mathrm{M}$ again denotes the bivariate normal distribution function.

2. The probability $\text{prob}^{V_2}(y=1)$ is exactly symmetric to $\text{prob}^{V_1}(x=1)$, with the roles of $S_1$ and $S_2$ interchanged. Note that the mirror image of $d_1$ defined in @eq-max4 is
$$
\frac{\log\left(\frac{S_2(0)}{S_1(0)}\right)+\left(q_1-q_2+\frac{1}{2}\sigma^2\right)T}{\sigma\sqrt{T}}\; ,
$$
which equals $-d_2$.
Therefore,
\begin{equation*}
\text{prob}^{V_2}(y=1) = \mathrm{M}(d_{21},-d_2,\rho_2)\;,
\end{equation*}
where
$$
d_{21}=\frac{\log\left(\frac{S_2(0)}{K}\right)+\left(r-q_2+\frac{1}{2}\sigma_2^2\right)T}{\sigma_2\sqrt{T}},\qquad d_{22} = d_{21}-\sigma_2\sqrt{T}\;,
$$ {#eq-max31}

and
$$
\rho_2 = \frac{\sigma_2-\rho\sigma_1}{\sigma}\; .
$$
3. As usual, we have

$$
\begin{align*}
\log S_1(T) &= \log S_1(0) + \left(r-q_1-\frac{1}{2}\sigma_1^2\right)T +\sigma_1B^*_{1}(T)\; ,\\
\log S_2(T) &= \log S_2(0) + \left(r-q_2-\frac{1}{2}\sigma_2^2\right)T +\sigma_2B^*_{2}(T)\;,
\end{align*}
$$

where $B^*_{1}$ and $B^*_{2}$ now denote Brownian motions under the risk-neutral probability.  The event $z=1$ is the complement of the event
$$
S_1(T)\leq K \quad \text{and} \quad S_2(T)\leq K\; ,
$$
which is equivalent to 

$$
\frac{1}{\sqrt{T}}B^*_{1}(T) < -d_{12}\;,
$$ {#eq-max32}

and
$$
\frac{1}{\sqrt{T}}B^*_{2}(T) < -d_{22}\;.
$$ {#eq-max42}



The random variables on the left-hand sides of @eq-max32 and @eq-max42 are standard normals and have correlation $\rho$.
Therefore,
\begin{equation*}
\text{prob}^{R}(z=1) = 1- \mathrm{M}(-d_{12},-d_{22},\rho)\;.
\end{equation*}



#### Call-on-the-Max Pricing Formula


::: Rule

The value of a call option on the maximum of two risky asset prices with volatilities $\sigma_1$ and $\sigma_2$ and correlation $\rho$ is

$$
\begin{multline}
\mathrm{e}^{-q_1T}S_1(0)\mathrm{M}\!\left(d_{11},d_1,\frac{\sigma_1-\rho\sigma_2}{\sigma}\right) + \mathrm{e}^{-q_2T}S_2(0)\mathrm{M}\!\left(d_{21},-d_2,\frac{\sigma_2-\rho\sigma_1}{\sigma}\right)\\+ \mathrm{e}^{-rT}K\mathrm{M}(-d_{12},-d_{22},\rho) - \mathrm{e}^{-rT}K\;,
\end{multline}
$$ {#eq-callonmaxformula}

where $\sigma$ is defined in @eq-max2a and  $d_1$, $d_2$, $d_{11}$, $d_{12}$, $d_{21}$ and $d_{22}$ are defined in @eq-max4 - @eq-max3.
:::

The following code shows how to compute the price of a Call on the Max.

```{python}

#| code-fold: true
#| label: Call_On_Max

def call_on_max(S1, S2, K, r, sig1, sig2, rho, q1, q2, T):
    sigma = np.sqrt(sig2 ** 2 - 2 * rho * sig1 * sig2 + sig1 ** 2)
    d1 = (np.log(S1 / S2) + (q2 - q1 + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    d11 = (np.log(S1 / K) + (r - q1 + sig1 ** 2 / 2) * T) / (sig1 * np.sqrt(T))
    d12 = d11 - sig1 * np.sqrt(T)
    d21 = (np.log(S2 / K) + (r - q2 + sig2 ** 2 / 2) * T) / (sig2 * np.sqrt(T))
    d22 = d21 - sig2 * np.sqrt(T)
    rho1 = (sig1 - rho * sig2) / sigma
    rho2 = (sig2 - rho * sig1) / sigma
    M1 = binormal_prob(d11, d1, rho1)
    M2 = binormal_prob(d21, -d2, rho2)
    M3 = binormal_prob(-d12, -d22, rho)

    return np.exp(-q1 * T) * S1 * M1 + np.exp(-q2 * T) * S2 * M2 + np.exp(-r * T) * K * M3 - np.exp(-r * T) * K
# Example usage

S1 = 100
S2= 100
K = 100
r = 0.05
sigma1 = 0.2
sigma2 = 0.2
q1 = 0.02
q2= 0.01
T=1
rho= 0.1

# print("Call on Max:", call_on_max(S1, S2, K, r, sigma1, sigma2, rho, q1, q2, T))
```


## Forward-Start Options
A forward-start option \index{forward-start option} is an option for which the strike price is set equal to the stock price at some later date.  In essence, it is issued at the later date, with the strike price set at the money.  For example, an executive may know that he is to be given an option grant at some later date with the strike price set equal to the stock price at that date.

#### Forward-Start Call Payoff

A forward-start call is defined by its maturity date $T^*$ and the date $T<T^*$ at which the strike price is set.  The value of a forward-start call at maturity is
$$\max(0,S_{T^*}-S_T)\; .$$
Let
$$x= \begin{cases} 1 &\text{if $S_{T^*}>S_T$;,}\\
                               0 & \text{otherwise;.}
        \end{cases}
$$
Then, the value of the call at maturity can be written as 
$$
xS_{T^*}-xS_T\; .
$$

#### Numeraires


1. Use $V_t=\mathrm{e}^{qt}S_t$ as numeraire to price the payoff $xS_{T^*}$.  From the fundamental pricing @eq-formula, the value at date $0$ is
$$\mathrm{e}^{-qT^*}S_0\mathbb{E}^V[x] = \mathrm{e}^{-qT^*}S_0\times \text{prob}^V\!(S_{T^*}>S_T)\; .$$
2. To price the payoff $xS_T$, use the following portfolio as numeraire:^[We are going to use equation~@eq-probSnumeraire at date $T^*$ to define the probabilities, because it will not be known until date $T^*$ whether the event $S_{T^*}>S_T$ is true.  Thus, we need the price of a numeraire asset at date $T^*$.  We would like this price to be a constant times $S_T$, which is what we will obtain.  An equivalent numeraire is to make a smaller investment in the same portfolio: start with $\mathrm{e}^{-r(T^*-T)-qT}$ shares.  This results in a final value of $S_T$ at date $T^*$.  As will be seen, this is useful for deriving the put-call parity relation for forward-start options.]purchase $\mathrm{e}^{-qT}$ shares of the stock at date $0$ and reinvest dividends until date $T$.  This will result in the ownership of one share at date $T$, worth $S_T$ dollars.  At date $T$, sell the share and invest the proceeds in the risk-free asset and hold this position until date $T^*$.  At date $T^*$, the portfolio will be worth $\mathrm{e}^{r(T^*-T)}S_T$.  Let $Z_t$ denote the value of this portfolio for each $0\leq t\leq T^*$.  The fundamental pricing @eq-formula implies that the value of receiving $xS_T$ at date $T^*$ is
\begin{align*}
Z_0\mathbb{E}^Z\left[ \frac{xS_T}{Z_{T^*}}\right] &=
\mathrm{e}^{-qT}S_0\mathbb{E}^Z\left[ \frac{xS_T}{\mathrm{e}^{r(T^*-T)}S_T}\right]\\&= \mathrm{e}^{-qT-r(T^*-T)}S_0\mathbb{E}^Z[x] \\&= \mathrm{e}^{-qT-r(T^*-T)}S_0 \times\text{prob}^Z(S_{T^*}>S_T)\;.
\end{align*}




#### Calculating Probabilities



1. As in the case of a share digital, we know that
$$ 
\log S_t = \log S_0 + \left(r-q +\frac{1}{2}\sigma^2\right)t + \sigma B^*_t
$$
for all $t>0$, where $B^*$ is a Brownian motion when $V$ is used as the numeraire.  Taking $t=T^*$ and $t=T$ and subtracting yields
$$\log S_{T^*}-\log S_T = \left(r-q +\frac{1}{2}\sigma^2\right)(T^*-T) + \sigma \left[B^*_{T^*}-B^*_T\right]\; .$$
Hence, $S_{T^*}>S_T$ if and only if
$$-\frac{B^*_{T^*}-B^*_T}{\sqrt{T^*-T}} < \frac{\left(r-q +\frac{1}{2}\sigma^2\right)(T^*-T)}{\sigma\sqrt{T^*-T}}\; .$$
The random variable on the left hand side is a standard normal, so
$$ 
\text{prob}^V\!(S_{T^*}>S_T) = \mathrm{N}(d_1)\; ,
$$
where

$$
d_1 = \frac{\left(r-q +\frac{1}{2}\sigma^2\right)(T^*-T)}{\sigma\sqrt{T^*-T}} =\frac{\left(r-q +\frac{1}{2}\sigma^2\right)\sqrt{T^*-T}}{\sigma}\;.
$$ {#eq-forwardstart_d1}


2. To calculate the probability $\text{prob}^Z(S_{T^*}>S_T)$, note that between $T$ and $T^*$, the portfolio with price $Z$ earns the risk-free rate $r$.  The same argument presented in @sec-s:girsanov shows that between $T$ and $T^*$ we have
$$
\frac{\mathrm{d}  S}{S} = (r-q)\mathrm{d}   t + \sigma\mathrm{d}   B^*\; ,
$$
where now $B^*$ denotes a Brownian motion when $Z$ is used as the numeraire.   This implies as usual that
$$
 \mathrm{d} \log S = \left(r-q-\frac{1}{2}\sigma^2\right)\mathrm{d}   t + \sigma\mathrm{d}   B^*\; ,
 $$
which means that
$$
\log S_{T^*} - \log S_T = \left(r-q-\frac{1}{2}\sigma^2\right)(T^*-T) + \sigma(B^*_{T^*}-B^*_T)\; .
$$
Hence, $S_{T^*}>S_T$ if and only if
$$
-\frac{B^*_{T^*}-B^*_T}{\sqrt{T^*-T}} < \frac{\left(r-q -\frac{1}{2}\sigma^2\right)(T^*-T)}{\sigma\sqrt{T^*-T}}\; .
$$
As before, the random variable on the left hand side is a standard normal, so
$$
\text{prob}^Z(S_{T^*}>S_T) = \mathrm{N}(d_2)\; ,
$$
where
$$
d_2 = \frac{\left(r-q -\frac{1}{2}\sigma^2\right)\sqrt{T^*-T}}{\sigma}=d_1-\sigma\sqrt{T^*-T}\;.
$$ {#eq-forwardstart_d2}




#### Forward-Start Call Pricing Formula
Combining these results, we have:

::: Rule

The value of a forward-start call at date $0$ is
$$
\mathrm{e}^{-qT^*}S_0\mathrm{N}(d_1) - \mathrm{e}^{-qT-r(T^*-T)}S_0\mathrm{N}(d_2)\;,
$$ {#eq-fstrikecall}

where $d_1$ and $d_2$ are defined in @eq-forwardstart_d1 - @eq-forwardstart_d2.
:::


#### Put-Call Parity\index{put-call parity}

\next Forward-strike calls and puts satisfy a somewhat unusual form of put-call parity.  The usual put-call parity is of the form:
$$
\text{Call} \;+\; \text{Cash} \quad = \quad \text{Put} \;+ \;\text{Underlying}\; .
$$

The amount of cash is the amount that will accumulate to the exercise price at maturity; i.e., it is $\mathrm{e}^{-rT^*}K$.  For forward-start calls and puts, the effective exercise price is $S_T$, which is not known at date $0$.  However, the portfolio used as numeraire to value the second part of the payoff will be worth $\mathrm{e}^{r(T^*-T)}S_T$ at date $T^*$, and by following the same strategy but starting with $\mathrm{e}^{-r(T^*-T)-qT}$ instead of $\mathrm{e}^{-qT}$ shares, we will have $S_T$ dollars at date $T^*$.  The date--0 value of this portfolio should replace Cash in the above.  Thus:

::: Rule

Put-call parity for forward-start calls and puts is
$$
\text{Call Price} \;+\; \mathrm{e}^{-r(T^*-T)-qT}S_0 = \text{Put Price} \;+\; \mathrm{e}^{-qT^*}S_0\;.
$$ {#eq-fstrikeparity}

The new features in the option pricing formulas in this chapter are the use of the bivariate normal distribution function and sometimes the need to compute a critical (at-the-money) value of the underlying asset price.  We will compute the critical values by bisection, in the same way that we computed implied volatilities for the Black-Scholes formula in @sec-c:blackscholes.

The following is a fast approximation of the bivariate \index{bivariate normal distribution function} cumulative normal distribution function, accurate to six decimal places, due to Drezner [@Drezner]).  For given numbers $a$ and $b$, this function gives the probability that $\xi_1<a$ and $\xi_2<b$ where $\xi_1$ and $\xi_2$ are standard normal random variables with a given correlation $\rho$, which we must input.  

```{python}

#| code-fold: true
#| label: Binormal_Prob

import numpy as np
from scipy.stats import norm

def binormal_prob(a, b, rho):
    x = np.array([0.24840615, 0.39233107, 0.21141819, 0.03324666, 0.00082485334])
    y = np.array([0.10024215, 0.48281397, 1.0609498, 1.7797294, 2.6697604])
    a1 = a / np.sqrt(2 * (1 - rho ** 2))
    b1 = b / np.sqrt(2 * (1 - rho ** 2))
    if a <= 0 and b <= 0 and rho <= 0:
        total_sum = 0
        for i in range(5):
            for j in range(5):
                z1 = a1 * (2 * y[i] - a1)
                Z2 = b1 * (2 * y[j] - b1)
                z3 = 2 * rho * (y[i] - a1) * (y[j] - b1)
                total_sum += x[i] * x[j] * np.exp(z1 + Z2 + z3)
        return total_sum * np.sqrt(1 - rho ** 2) / np.pi
    elif a <= 0 and b >= 0 and rho >= 0:
        return norm.cdf(a) - binormal_prob(a, -b, -rho)
    elif a >= 0 and b <= 0 and rho >= 0:
        return norm.cdf(b) - binormal_prob(-a, b, -rho)
    elif a >= 0 and b >= 0 and rho <= 0:
        total_sum = norm.cdf(a) + norm.cdf(b)
        return total_sum - 1 + binormal_prob(-a, -b, rho)
    elif a * b * rho > 0:
        rho1 = (rho * a - b) * np.sign(a) / np.sqrt(a ** 2 - 2 * rho * a * b + b ** 2)
        rho2 = (rho * b - a) * np.sign(b) / np.sqrt(a ** 2 - 2 * rho * a * b + b ** 2)
        Delta = (1 - np.sign(a) * np.sign(b)) / 4
        return binormal_prob(a, 0, rho1) + binormal_prob(b, 0, rho2) - Delta
# print("BiNormalProb:", binormal_prob(0.1, 0.2, 0.3))
```

\noindent Notice that this function calls itself.  This is an example of recursion.


The forward-start call pricing formula is of the same form as the Black-Scholes, Margrabe, Black, and Merton formulas.  We can compute it with our `Generic_Option` pricing function.

```{python}

#| code-fold: true
#| label: forward_start_call

def generic_option(P1, P2, sigma, T):
    """
    Inputs:
    P1 = present value of asset to be received
    P2 = present value of asset to be delivered
    sigma = volatility
    T = time to maturity
    """
    x = (np.log(P1 / P2) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    y = x - sigma * np.sqrt(T)
    N1 = norm.cdf(x)
    N2 = norm.cdf(y)
    return P1 * N1 - P2 * N2

def forward_start_call(S, r, sigma, q, Tset, TCall):
    P1 = np.exp(-q * TCall) * S
    P2 = np.exp(-q * Tset - r * (TCall - Tset)) * S
    return generic_option(P1, P2, sigma, TCall - Tset)

# Example usage
S = 100
K = 100
r = 0.05
sigma = 0.2
q = 0.02
T = 1
Div = 5
TDiv = 0.5
TCall = 1
N = 10

# print("Forward Start Call:", forward_start_call(S, r, sigma, q, 0.5, TCall))
```



:::




## Choosers {#sec-s:choosers}
A chooser option \index{chooser} allows the holder to choose whether the option will be a put or call at some fixed date before the option maturity.  Let $T$ denote the date at which the choice is made, $T_c$ the date at which the call expires, $T_p$ the date at which the put expires, $K_c$ the exercise price of the call, and $K_p$ the exercise price of the put, where $0<T<T_c$ and $0<T<T_p$.  A simple chooser has $T_c=T_p$ and $K_c=K_p$.  A chooser is similar in spirit to a straddle: it is a bet on volatility without making a bet on direction.  A simple chooser must be cheaper than a straddle with the same exercise price and maturity $T^*=T_c=T_p$, because a straddle is always in the money at maturity, whereas a simple chooser has the same value as the straddle if it is in the money but is only in the money at $T^*$ when the choice made at $T$ turns out to have been the best one.  

#### Chooser Payoff
The value of the chooser at date $T$ will be the larger of the call and put prices.  Let $S^*$ denote the stock price at which the call and put have the same value.  We can find $S^*$ by solving

```Black_Scholes_Call(S*,Kc,r,sigma,q,Tc-T) = Black_Scholes_Put(S*,Kp,r,sigma,q,Tp-T).```                           


\noindent For a simple chooser with $K_c=K_p=K$ and $T_c=T_p=T^*$, we can find $S^*$ from the put-call parity relation at $T$, leading to $S^*=\mathrm{e}^{(q-r)(T^*-T)}K.$

The call will be chosen when $S_T>S^*$ and it finishes in the money if $S(T_c)>K_c$ at date $T_c$, so the payoff of the chooser is $S(T_c)-K_c$ when 
$$S_T>S^* \quad \text{and}\quad S(T_c)>K_c\;.
$$
The payoff is $K_p-S(T_p)$ at date $T_p$ when 
$$
S_T<S^* \quad \text{and}\quad S(T_p)<K_p\;.
$$
Let 
$$
\begin{align*}
x&= \begin{cases} 1 &\text{if $S_T>S^* $ and $S(T_c)>K_c$;,}\\
                               0 & \text{otherwise;.}
        \end{cases}\\
y&= \begin{cases} 1 &\text{if $S_T<S^*$ and $S(T_p)<K_p$;,}\\
                               0 & \text{otherwise;.}
        \end{cases} 
\end{align*}
$$
Then the payoff of the chooser is $xS(T_c)-xK_c$ at date $T_c$ and $yK_p-yS(T_p)$ at date $T_p$.

#### Numeraires
As in the analysis of compound options, the value of the chooser at date $0$ must be 

$$
\begin{multline}
\mathrm{e}^{-q T_c}S_0\times\text{prob}^V\!(x=1) \;- \;\mathrm{e}^{-rT_c}K_c\times\text{prob}^R(x=1)
\\+\; \mathrm{e}^{-rT_p}K_p\times\text{prob}^R(y=1) \;- \;\mathrm{e}^{-q T_p}S_0\times\text{prob}^V\!(y=1)\;,
\end{multline}
$$ {#eq-chooser1}

where we use $V_t=\mathrm{e}^{qt}S_t$ and $R_t=\mathrm{e}^{rt}$ as numeraires. 

#### Chooser Pricing Formula
@eq-chooser1 and calculations similar to those of the previous two sections lead us to:


::: Rule

The value of a chooser option is 

$$
\begin{multline}
\mathrm{e}^{-q T_c}S_0\mathrm{M}\!\left(d_1,d_{1c},\sqrt{T/T_c}\right) - \mathrm{e}^{-rT_c}K_c\mathrm{M}\!\left(d_2,d_{2c},\sqrt{T/T_c}\right)\\ +\mathrm{e}^{-rT_p}K_p\mathrm{M}\!\left(-d_2,-d_{2p} , \sqrt{T/T_p}\right) \\- \mathrm{e}^{-q T_p}S_0\mathrm{M}\!\left(-d_1 ,-d_{1p} ,\sqrt{T/T_p}\right)\;,
\end{multline}
$$ {#eq-chooser2}

where
\begin{equation*}
\begin{array}{ll}
d_1 = \frac{\log\left(\frac{S_0}{S^*}\right)+\left(r-q+\frac{1}{2}\sigma^2\right)T}{\sigma\sqrt{T}}\;,
& \qquad d_2=d_1-\sigma\sqrt{T}\;,\\
&\\
d_{1c} = \frac{\log\left(\frac{S_0}{K_c}\right)+\left(r-q+\frac{1}{2}\sigma^2\right)T_c}{\sigma\sqrt{T_c}}\;,
& \qquad d_{2c}=d_{1c}-\sigma\sqrt{T_c}\;,\\
&\mathrm{d}_{1p} = \frac{\log\left(\frac{S_0}{K_p}\right)+\left(r-q+\frac{1}{2}\sigma^2\right)T_p}{\sigma\sqrt{T_p}}\;,
& \qquad d_{2p}=d_{1p}-\sigma\sqrt{T_p}\;.
\end{array}
\end{equation*}
:::

To implement the bisection to compute $S^*$, we can take zero as a lower bound and $K_c+K_p$ as an upper bound.^[We take the call value to be zero and the put value to be $\mathrm{e}^{-r(T_p-T)}K_p$ at date $T$ when the stock price is zero.  To see why the upper bound works, note that when the stock price is $S$ at date $T$,  the call is worth at least $S^*-K_c$ and the put is worth no more than $K_p$; i.e, $C \geq S-K_c$ and $P \leq K_p$.  Therefore, $C-P \geq S-K_c-K_p$.  Hence when $S=K_c+K_p$, we have $C-P\geq 0$. }

```{python}

#| code-fold: true
#| label: Choosers_Option

def chooser(S, Kc, Kp, r, sigma, q, T, Tc, Tp):
    tol = 1e-6
    lower = 0
    upper = np.exp(q * Tc) * (Kc + Kp)
    guess = 0.5 * Kc + 0.5 * Kp
    flower = -np.exp(-r * (Tp - T)) * Kp
    fupper = black_scholes_call(upper, Kc, r, sigma, q, Tc - T) - black_scholes_put(upper, Kp, r, sigma, q, Tp - T)
    fguess = black_scholes_call(guess, Kc, r, sigma, q, Tc - T) - black_scholes_put(guess, Kp, r, sigma, q, Tp - T)
    while upper - lower > tol:
        if fupper * fguess < 0:
            lower = guess
            flower = fguess
            guess = 0.5 * lower + 0.5 * upper
            fguess = black_scholes_call(guess, Kc, r, sigma, q, Tc - T) - black_scholes_put(guess, Kp, r, sigma, q, Tp - T)
        else:
            upper = guess
            fupper = fguess
            guess = 0.5 * lower + 0.5 * upper
            fguess = black_scholes_call(guess, Kc, r, sigma, q, Tc - T) - black_scholes_put(guess, Kp, r, sigma, q, Tp - T)
    Sstar = guess

    d1 = (np.log(S / Sstar) + (r - q + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    d1c = (np.log(S / Kc) + (r - q + sigma ** 2 / 2) * Tc) / (sigma * np.sqrt(Tc))
    d2c = d1c - sigma * np.sqrt(Tc)
    d1p = (np.log(S / Kp) + (r - q + sigma ** 2 / 2) * Tp) / (sigma * np.sqrt(Tp))
    d2p = d1p - sigma * np.sqrt(Tp)
    rhoc = np.sqrt(T / Tc)
    rhop = np.sqrt(T / Tp)
    M1c = binormal_prob(d1, d1c, rhoc)
    M2c = binormal_prob(d2, d2c, rhoc)
    M1p = binormal_prob(-d1, -d1p, rhop)
    M2p = binormal_prob(-d2, -d2p, rhop)

    return np.exp(-q * Tc) * S * M1c - np.exp(-r * Tc) * Kc * M2c + np.exp(-r * Tp) * Kp * M2p - np.exp(-q * Tp) * S * M1p

# Example usage
S = 100
Kc = 80
Kp = 80
r = 0.05
sigma = 0.2
q = 0.02
Tc = 1.5
Tp = 1.5
T=1

# print("Chooser Option:", chooser(S, Kc, Kp, r, sigma, q, T, Tc, Tp))

```




## Exercises

::: {#exr-exotics1}
 Intuitively, the value of a forward-start call option should be lower the closer is the date $T$ at which the strike is set to the date $T^*$ at which the option matures, because then the option has less time to maturity after being created at $T$.  Create a Python program to confirm this.  Allow the user to input $S$, $r$, $\sigma$, $q$, and $T^*$.  Compute and plot the value of the option for $T=0.1T^*$, $T=0.2T^*$, \ldots, $T=0.9T^*$.
:::
::: {#exr-exotics2}
 Create a Python program to demonstrate the additional leverage of a call-on-a-call relative to a standard call.  Allow the user to input $S$, $r$, $\sigma$, $q$, and $T^*$.  Use the `Black-Scholes_Call` function to compute and output the value $C$ of a European call with strike $K**=S$ (i.e., the call is at the money) and maturity $T^*$.  Use the `Call_on_Call` function to compute and output the value of a call option on the call with strike $K=C$ (i.e., the call-on-a-call is at the money) and maturity $T=0.5T^*$.  Compute the percentage returns the standard European call and the call-on-a-call would experience if the stock price $S$ instantaneously increased by 10\%.
:::
::: {#exr-exotics3}
 Create a Python program to illustrate the early exercise premium for an American call on a stock paying a discrete dividend.  Allow the user to input $S$, $r$, $\sigma$, and $T^*$.  Take the date of the dividend payment to be $T=0.5T^*$ and take the strike price to be $K=S$.  The value of a European call is given by the Black-Scholes formula with $S-\mathrm{e}^{-rT}D$ being the initial asset price and $q=0$ being the constant dividend yield.  Use the function `American_Call_Dividend` to compute the value of an American call for dividends $D=.1S$, \ldots $D=.9S$.  Subtract the value of the European call with the same dividend to obtain the early exercise premium.  Plot the early exercise premium against the dividend $D$.
:::
::: {#exr-exotics4}
 Create a Python function to value a simple chooser (a chooser option in which $K_c=K_p$ and $T_c=T_p$) using put-call parity to compute $S^*$ as mentioned in @sec-s:choosers.  Verify that the function gives the same result as the function `Chooser`.
:::
::: {#exr-exotics5}
 Create a Python code to compare the cost of a simple chooser to that of a straddle (straddle = call + put with same strike and maturity).  Allow the user to input $S$, $r$, $\sigma$, $q$, and $T^*$.  Take the time to maturity of the underlying call and put to be $T^*$ for both the chooser and the straddle.  Take the strike prices to be $K=S$.  Take the time the choice must be made for the chooser to be $T=0.5T^*$.  Compute the cost of the chooser and the cost of the straddle.
:::
::: {#exr-exotics6}
 A stock has fallen in price and you are attempting to persuade a client that it is now a good buy.  The client believes it may fall further before bouncing back and hence is inclined to postpone a decision.  To convince the client to buy now, you offer to deliver the stock to him at the end of two months at which time he will pay you the lowest price it trades during the two months plus a fee for your costs.  The stock is not expected to pay a dividend during the next two months.  Assuming the stock actually satisfies the Black-Scholes assumptions, find a formula for the minimum fee that you would require.  (Hint:  It is almost in @sec-s:lookbacks.)  Create a Python program allowing the user to input $S$, $r$, and $\sigma$  and computing the minimum fee.
:::
::: {#exr-e_standardknockout}
  Suppose you must purchase 100 units of an asset at the end of a year.  Create a Python program simulating the asset price and comparing the quality of the following hedges (assuming 100 contracts of each):



1. a standard European call,
2. a down-and-out call in which the knock-out barrier is 10\% below the current price of the asset.


Take both options to be at the money at the beginning of the year.  Allow the user to input $S$, $r$, $\sigma$ and $q$.   Generate 500 simulated end-of-year costs (net of the option values at maturity) for each hedging strategy and create histogram charts to visually compare the hedges.  Note: to create histograms, you will need the Data Analysis add-in, which may be need to be loaded (click Tools/Add Ins).
:::
::: {#exr-e_standardknockout2}

Compute the prices of the options in the previous exercise.  Modify the simulations to compare the end-of-year costs including the costs of the options, adding interest on the option prices to put everything on an end-of-year basis.
:::
::: {#exr-e_standardknockout3}
 
Modify @exr-e_standardknockout by including a third hedge: a combination of a down-and-out call as in part (b) of @exr-e_standardknockout and a down-and-in call with knockout barrier and strike 10\% below the current price of the asset.  Note that this combination forms a call option with strike that is reset when the underlying asset price hits a barrier.
:::
::: {#exr-e_standardknockout4}

Modify @exr-e_standardknockout2 by including the hedge in @exr-e_standardknockout3.  Value the down-and-in call using the function `Down_And_Out_Call` and the fact that a down-and-out and down-and-in with the same strikes and barriers form a standard option. 
:::
::: {#exr-e_averagehedge}
  Each week you purchase 100 units of an asset, and you want to hedge your total quarterly (13-week) cost.  Create a Python program simulating the asset price and comparing the quality of the following hedges: 



1. a standard European call maturing at the end of the quarter ($T=0.25$) on 1300 units of the asset, 
2. 13 call options maturing at the end of each week of the quarter, each written on 100 units of the asset, and 
3. a discretely sampled average-price call with maturity $T=0.25$ written on 1300 units of the asset, where the sampling is at the end of each week.  
4. a discretely sampled geometric-average-price call with maturity $T=0.25$ written on 1300 units of the asset, where the sampling is at the end of each week.


Allow the user to input $S$, $r$, $\sigma$ and $q$.  Assume all of the options are at the money at the beginning of the quarter ($K=S$).  Compare the hedges as in @exr-e_standardknockout.
:::
::: {#exr-e_averagehedge2}
 In the setting of the previous problem, compute the prices of the options in parts (a), (b) and (d).  Modify the simulations in the previous problem to compare the end-of-quarter costs including the costs of the options (adding interest on the option prices to put everything on an end-of-quarter basis).
:::
::: {#exr-exotics13}
 Using the put-call parity relation, derive a formula for the value of a forward-start put.
:::
::: {#exr-exotics14}
 Derive @eq-callonaput for the value of a call on a put.
:::
::: {#exr-exotics15}
 Complete the derivation of @eq-chooser2 for the value of a chooser option.
:::
::: {#exr-exotics16}
 Derive a formula for the value of a put option on the maximum of two risky asset prices.
:::
::: {#exr-exotics17}
 Using the result of the preceding exercise and Margrabe's formula, verify that calls and puts (having the same strike $K$ and maturity $T$) on the maximum of two risky asset prices satisfy the following put-call parity relation:
\begin{multline*}
\mathrm{e}^{-rT}K + \text{Value of call on max} \\
= \mathrm{e}^{-q_2T}S_2(0) + \text{Value of option to exchange asset 2 for asset 1} \\+ \text{Value of put on max}\;.
\end{multline*}
:::

