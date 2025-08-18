{{< include macros.qmd >}}


## Geometric Brownian Motion {#sec-c:geometricbrownianmotion}


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


A random variable $\tilde{y}$ is lognormally distributed if it can be written as $\tilde{y}=  \mathrm{e}^{\tilde{x}}$ where $\tilde{x}$  has a normal distribution.  Another way of saying this is that $\log \tilde y$ is normally distributed.  We denote the expected value (mean) of any random variable using the symbol $\mathbb{E}$.  The mean of a lognormal random variable is given by the following.

::: Principle
If $\tilde y = \tilde{x}$ where $\tilde x$ is normally distributed with mean $m$ and standard deviation $s$, then 
$$\mathbb{E}[\tilde{y}]=e^{m +  s^2/2};.
$$ {#eq-lognormal}
:::
An example of the distribution of a lognormal random variable is shown in @fig-lognormal.

```{python}
#| label: fig-lognormal
#| fig-cap: "The distribution of the random variable y = exp(x) where x has the standard normal distribution."
import numpy as np
from scipy.stats import lognorm
import plotly.graph_objects as go

# Generate x values
x = np.linspace(0, 5, 1000)

# Lognormal density (s=1, loc=0, scale=1 gives standard lognormal)
pdf = lognorm.pdf(x, s=1, loc=0, scale=1)

# Create plot
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=x,
        y=pdf,
        mode='lines',
        name='Lognormal density',
        hovertemplate='x = %{x:.2f}<br>density = %{y:.2f}<extra></extra>'
    )
)

fig.update_layout(
    showlegend=False,
    xaxis_title='',
    yaxis_title='Lognormal density',
    template='plotly_white',
    height=300,
    autosize=True
)

fig.show()
#| out-width: "100%"
```

An important stochastic process is geometric Brownian motion given by
$$
S_t=S_0\mathrm{e}^{\mu t- \sigma^2 t/2 + \sigma B_t}
$$ {#eq-exponential1}
for constants $\mu$ and $\sigma$, where $B$ is a Brownian motion.  For each time $t$, the random variable $S_t$ in @eq-exponential1 is a lognormal random variable. Ito's formula for exponentials of Ito processes implies
$$
\frac{\mathrm{d}  S}{S} = \mu\mathrm{d}   t+\sigma\mathrm{d}   B\;.
$$ {#eq-Y}
When we see an equation of the form @eq-Y, we should recognize @eq-exponential1 as the solution. 

The process $S$ is called a geometric Brownian motion.  We interpret @eq-Y as stating that $\mu\mathrm{d}   t$ is the expected rate of change of $S$ and $\sigma^2\mathrm{d}   t$ is the variance of the rate of change in an instant $\mathrm{d}  t$.  We call $\mu$ the **drift** and $\sigma$ the **volatility**.  The geometric Brownian motion grows at the average rate of $\mu$, in the sense that $\mathbb{E}[S_t] = \mathrm{e}^{\mu t}S_0$.  This can be verified with the aid of @eq-lognormal.

If $\mu=0$, then $\mathbb{E}[S_t] = S_0$.  In fact, a geometric Brownian motion $S$ as in @eq-exponential1 and @eq-Y is a martingale when $\mu=0$.  It is an example of what is called an **exponential martingale**.  An exponential is always positive, so an exponential martingale is always positive.

Taking the natural logarithm of @eq-exponential1 gives an equivalent form of the solution:
$$
\log S_t= \log S_0+\left(\mu -\frac{1}{2}\sigma^2\right)t + \sigma B_t\;.
$$ {#eq-exponential2}
  This shows that $\log S_t - \log S_0$ is a $(\mu-\sigma^2/2,\sigma)$--Brownian motion.  Given information at time $t$, the logarithm of $S(u)$ for $u>t$ is normally \index{lognormal distribution}distributed with mean $(u-t)(\mu-\sigma^2/2)$ and variance $(u-t)\sigma^2$.  Because $S$ is the exponential of its logarithm, $S$ can never be negative.  For this reason, a geometric Brownian motion is a better model for stock prices than is a Brownian motion.  The differential of @eq-exponential2 is
$$
\mathrm{d}  \log S_t = \left(\mu -\frac{1}{2}\sigma^2\right)\mathrm{d}   t+ \sigma\mathrm{d}   B_t\;.
$$ {#eq-exponential3}

We can summarize this discussion as follows.

::: Principle
The  equation 
$$\frac{\mathrm{d}  S}{S} = \mu\mathrm{d}   t+\sigma\mathrm{d}   B$$
is equivalent to the equation
$$\mathrm{d}  \log S = \left(\mu -\frac{1}{2}\sigma^2\right)\mathrm{d}   t+ \sigma\mathrm{d}   B\; .$$
The solution of both equations is @eq-exponential1 or the equivalent @eq-exponential2.
:::

We can simulate a path of a geometric Brownian motion $S$ by first simulating $\log S$ and then computing the exponential.  We simulate the changes $\Delta \log S$ as normally distributed random variables with mean equal to $(\mu-\sigma^2/2)\Delta t$ and variance equal to $\sigma^2\Delta t$.  We could also simulate the changes $\Delta \log S$ using a binomial model as in @sec-s:brownian_binomial.

```{python}
#| label: fig-gbm-sim
#| fig-cap: "A path of a simulated geometric Brownian motion with mu=0.1, sigma=0.3, and 1,000 steps."
import numpy as np
import plotly.graph_objects as go

n = 1000     # number of divisions
mu = 0.1    # drift
sigma = 0.3 # volatility
S0 =  100   # initial value
T = 1       # length of simulation
dt = T / n  # Delta t
steps = np.random.normal(
    loc = (mu-0.5*sigma**2)*dt, 
    scale = sigma*np.sqrt(dt),
    size = n
)
logS = np.empty(n+1)
logS[0] = np.log(S0)
logS[1:] = logS[0] + np.cumsum(steps)
S = np.exp(logS)

fig = go.Figure(
    go.Scatter(
        x=np.arange(0, T+dt, dt),
        y=S,
        mode="lines",
        hovertemplate="t = %{x:.2f}<br>S = %{y:.2f}<extra></extra>", #
    )
)

fig.update_layout(
    showlegend=False,
    xaxis_title="Time",
    yaxis_title="Simulated GBM",
    template="plotly_white",
    height=300,
    autosize=True
)

fig.show()
#| out-width: "100%"
```

## Continuously Compounded Returns

Given a rate of return $r$ over a time period of $\Delta t$ years, the annualized continuously compounded return is defined to be the number $x$ such that $\mathrm{e}^{x\Delta t} = 1+r$; equivalently, $x = \log(1+r) / \Delta t$.  We frequently take **annualized** for granted and just say **continuously compounded return.**  As an example, if you earn 1\% in a month, then the continuously compounded return is $\log (1.01) / (1/12) = 0.1194$.  

The reason for the name **continuously compounded** is that compounding a monthly rate of interest of $0.1194/12$ for a total of $n$ times during a month produces a total return over the month of 
$$\left(1 + \frac{0.1194/12}{n}\right)^ n - 1\,,$$
which converges to $0.01$ as $n \rightarrow \infty$.  Thus, compounding an infinite number of times during a month at an annual rate of $0.1194$ (equal to a monthly rate of $0.1194/12$) is equivalent to the actual 1\% return.^[ The reason for considering this concept is that compound returns like $(1+r_1)(1+r_2)$ are simpler to analyze in some contexts as $\mathrm{e}^{x_1+x_2}$.]

Given a dividend-reinvested asset price $S$, the rate of return from date $t_1$ to $t_2$ is $S_{t_2}/S_{t_1} - 1$.  Denote this rate of return by $r$ and defined the corresponding continuously compounded return
$$x = \log(1+r) = \log\left(\frac{S_{t_2}}{S_{t_1}}\right) = \log S_{t_2} - \log S_{t_1}\,.$$
Using @eq-exponential2, we see that the continuously compounded return is 
$$
x = \log S_{t_2} - \log S_{t_1} = \left(\mu -\frac{1}{2}\sigma^2\right)(t_2-t_1)+ \sigma\,(B_{t_2} - B_{t_1})\;.
$$ {#eq-exponential11}
Thus, from the vantage point of date $t_1$, the continuously compounded return is normally distributed with mean
$$\left(\mu -\frac{1}{2}\sigma^2\right)\Delta t$$
and variance $\sigma^2 \Delta t$, where we define $\Delta t = t_2-t_1$.  Given historical data on rates of return, the parameters $\mu$ and $\sigma$ can be estimated by standard methods.


## Tail Probabilities of Geometric Brownian Motions {#sec-tailprobs}

For each of the numeraires discussed in the previous section, we have
$$
\mathrm{d}  \log S = \alpha\mathrm{d}   t + \sigma\mathrm{d}   B\;,
$$ {#eq-tailprob1}

for some $\alpha$ and $\sigma$, where $B$ is a Brownian motion under the probability measure associated with the numeraire.  Specifically, $\sigma=\sigma$, $B=B^*$, and



\renewcommand{\labelenumi}{(\arabic{enumi})}
1. for the risk-neutral probability, $\alpha = r-q-\sigma^2/2$,
2. when $\mathrm{e}^{qt}S_t$ is the numeraire, $\alpha = r-q +\sigma^2/2$,
3. when another risky asset price $Y$ is the numeraire, $\alpha = r-q+\rho\sigma\phi-\sigma^2/2$.



We will assume in this section that $\alpha$ and $\sigma$ are constants.
The essential calculation in pricing options is to compute $\text{prob}(S_t>K)$ and $\text{prob}(S_t<K)$ for a constant $K$ (the strike price of an option), where $\text{prob}$ denotes the probabilities at date $0$ (the date we are pricing an option) associated with a particular numeraire.  

@eq-tailprob1 gives us
$$\log S_t = \log S_0 + \alpha T + \sigma B_t\; .$$
Given this, we deduce

$$ 
S_t > K  \quad\Longleftrightarrow\quad \log S_t > \log K
$$
$$
 \quad\Longleftrightarrow\quad \sigma B_t > \log K - \log S_0-\alpha T
$$
$$
 \quad\Longleftrightarrow\quad \frac{B_t}{\sqrt{T}} > \frac{\log K - \log S_0-\alpha T}{\sigma\sqrt{T}}
$$
$$
 \quad\Longleftrightarrow\quad -\frac{B_t}{\sqrt{T}} < \frac{\log S_0-\log K + \alpha T}{\sigma\sqrt{T}}
$$
$$
 \quad\Longleftrightarrow\quad -\frac{B_t}{\sqrt{T}} < \frac{\log \left(\frac{S_0}{K}\right) + \alpha T}{\sigma\sqrt{T}}\;.
$$ {#eq-tailprob2}


The random variable on the left-hand side of @eq-tailprob2 has the standard normal distribution---it is normally distributed with mean equal to zero and variance equal to one.  As is customary, we will denote the probability that a standard normal is less than some number $d$ as $\mathrm{N}(d)$.  We conclude:

::: Rule
Assume $\mathrm{d}  \log S = \alpha\mathrm{d}   t + \sigma\mathrm{d}   B$, where $B$ is a Brownian motion.  Then, for any number $K$,
$$
\text{prob}(S_t>K) = \mathrm{N}(d)\;,
$$ {#eq-tailprob01}
where
$$
d = \frac{\log \left(\frac{S_0}{K}\right) + \alpha T}{\sigma\sqrt{T}}\;.
$$ {#eq-tailprob3}
:::


The  probability $\text{prob}(S_t<K)$ can be calculated similarly, but the simplest way to derive it is to note that the events $S_t>K$ and $S_t<K$ are complementary---their probabilities sum to one (the event $S_t=K$ having zero probability).  Therefore $\text{prob}(S_t<K) = 1-\mathrm{N}(d)$.  This is the probability that a standard normal is greater than $d$, and by virtue of the symmetry of the standard normal distribution, it equals the probability that a standard normal is less than $-d$.  Therefore, we have:

::: Rule
Assume $\mathrm{d}  \log S = \alpha\mathrm{d}   t + \sigma\mathrm{d}   B$, where $B$ is a Brownian motion.  Then, for any number $K$,
$$
\text{prob}(S_t<K) = \mathrm{N}(-d)\;,
$$ {#eq-tailprob02}
where $d$ is defined in @eq-tailprob3.
:::

## Exercises

::: {#exr-gbm1}
Use Ito's Lemma to derive the stochastic differential equation for $S_t^2$.  Show that $S(t)^2$ is a geometric Brownian motion if $\mu$ and $\sigma$ are constants and find $\mathbb{E}[S(t)^2]$.
::: 

::: {#exr-gbm2} 
Let
\begin{align*}
\frac{\mathrm{d} S_1}{S_1} &= \mu_1\mathrm{d} t + \sigma_1 \mathrm{d} B_1\\
\frac{\mathrm{d} S_2}{S_2} &= \mu_1\mathrm{d} t + \sigma_2 \mathrm{d} B_2
\end{align*}
where the $\mu_i$ and $\sigma_i$ are constants and $B_1$ and $B_2$ are Brownian motions with constant correlation $\rho$.  

1. Define $Y=S_1S_2$.  Show that $Y$ is a geometric Brownian motion and calculate its drift and volatility.
2. Repeat for $Y=S_1/S_2$.  

Hint: use the facts $\mathrm{e}^{x+y}=\mathrm{e}^x \times \mathrm{e}^y$ and $\mathrm{e}^x/\mathrm{e}^y = \mathrm{e}^{x-y}$.
::: 
:




