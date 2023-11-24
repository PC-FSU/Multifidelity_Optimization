# Multifidelity Optimization

This is project for multifidelity optimization, where we assume, we have a simulation model with tunable fidelity or a set of models with varying fidelity and are interested in optimizing an objective function at the highest fidelity, while keeping the overall computational cost at a minimum. In other words, we seek to optimize the objective function at a cost cheaper than optimizing it solely with the highest fidelity model. We propose a framework that uses Gaussian process (GP) regression to learn the mapping between the system output and inputs augmented with the fidelity parameters, from a few evaluations of a subset of the available models. The GP model—along with a Bayesian framework---provides a sequential decision-making setup, that is leveraged to optimize the objective function at the highest fidelity, with only cheaper low-fidelity evaluations.

# Introduction

| **Problem Objective** | Find a global optimum at highest fidelity, $\underset{x}{max} 𝑓(𝑥, 𝑠=1)$ or $\underset{x}{min} 𝑓(𝑥, 𝑠=1)$. |
|-----------------------|---------------------------------------------------------|
| **Data Available**    | $𝑓(𝑥,𝑠)$ where $𝑥 \in \mathcal{R}^𝑑$ (design space), and $𝑠 \in [0,1]$ (fidelity space). |
| **Cost Model**        | Monotonically increasing function $𝑐(𝑠)$. |
| **Method Used**       | Gaussian Process based Bayesian Optimization with an acquisition function $\alpha$. |
|**Details**: |The acquisition function $\alpha$ determines where to evaluate the $𝑓(𝑥,𝑠)$ to achieve maximum information of $𝑓(𝑥,𝑠=1)$. |


### ``How does a single multifidelity bayesian optimization loop look likes?``:

Given: Gaussian Process Model $M$, Acquisition function $\alpha$, $f(x,s)$, cost model $c(s)$, Data $D=(x_n, y_n, s_n)$
1. Repeat: $n+1,...,N$ do:

   a. Select new $x_{n+1}, s_{n+1}= \underset{\{x,s\}}{ argmax} \, \frac{\alpha(x,s;D)}{c(s)}$

   b. $y_{n+1} = f(x_{n+1}, s_{n+1})$

   c. Augment Data $ D = {\{D, (x_{n+1}, y_{n+1}, s_{n+1})}\}$

   d. Update GP model, $M$.


# Acquisition function
## A short theoretical primer about acquisition function.

Acquisition functions are mathematical techniques that guide the exploration of the parameter space during Bayesian optimization. Consider a function $f : X \rightarrow \mathbb{R}$ where we aim to find its minimum within the domain $x \in X$.

$$
x_* = \underset{x \in X}{\text{argmin}} \, f(x)
$$

Typically, if the functional form of $f$ is known, an optimization routine can be applied to find its minimum. However, when $f$ is unknown, we approximate it using a surrogate that is easy to interpret and construct. In Bayesian optimization, $f$ is usually modeled by a Gaussian Process Regression (GPR). GPR assumes a Gaussian prior over $f$:

$$
p(f) = \mathcal{N}(f; \mu, K)
$$

Given $n$ observations of $f$, $\mathcal{D} = (X, Y) = (\{x_1,\cdots,x_n\},\{y_1,\cdots, y_n\})$, a positive semi-definite $n \times n$ matrix $K_{n \times n}$ can be defined using the kernel function $k$, where $[K]_{ij} = k(x_i,x_j)$. Conditioning $p(f)$ on $\mathcal{D}$ results in:

$$
P(f|\mathcal{D})  = \mathcal{N}(f; \mu_{f|\mathcal{D}}, K_{f|\mathcal{D}})
$$

Now, having a functional form of $f$ and an understanding of how to model it, the task is to use GPR for Bayesian optimization. As a Gaussian Process is defined by a mean function $\mu$ and covariance $K$, acquisition functions cleverly utilize these values to intelligently select the location for the next observation. 

<hr>

``How to fit a GP?``

To train a Gaussian Process Regression (GPR), we minimize the negative log marginal likelihood, given by:

$$
\mathcal{L}(\theta) = -\log p(Y|X,\theta) = \frac{1}{2} Y^T (K_{n \times n} + \sigma^2_\epsilon  I_{n \times n})^{-1} Y + \frac{1}{2} \log \left| K_{n \times n} + \sigma^2_\epsilon I_{n \times n} \right| + \frac{n}{2} \log 2\pi.
$$

Once the best hyperparameters $\hat{\theta}$ are found, inference on a new point $x_*$ can be made. The advantage of GPR is that everything is Gaussian, allowing closed-form evaluation. The predictive distribution $p(y_*) = \mathcal{N}(\mu_*,  \sigma^2_{*}|\mathcal{D},x_*)$ is also Gaussian.

<hr>


### `Probability of improvement`

The easiest acquisition function designed for Bayesian optimization was the probability of improvement. Let $f_0 = \min f$ represent the minimal value of $f$ observed thus far. The probability of improvement assesses $f$ at the point most likely to improve this value. It corresponds to the utility function $u(x)$ associated with evaluating $f$ at a given point $x$:

$$
u(x) = \begin{cases} 
      0 & \text{if } f(x) > f_0 \\
      1 & \text{if } f(x) \leq f_0 
   \end{cases}
$$

Here, a unit reward is received if $f(x)$ turns out to be less than or equal to $f_0$, and no reward is given otherwise. The probability of improvement acquisition function is then the expected utility as a function of $x$:

$$
\text{PI}(x) = E[u(x) | x, D] = \int_{-\infty}^{f_0} N\left(f; \mu(x), K(x, x)\right) df =  \Phi(f_0; \mu(x), K(x,x))
$$

For 1-D case this simplifies to:
$$
\text{PI}(x) = E[u(x) | x, D] = \Phi\left(\frac{f_0 - \mu(x)}{\sigma(x)}\right)
$$

Here $\Phi$ represent the $CDF$ of standard normal. The point with the highest probability of improvement (maximal expected utility) is selected, which represents the Bayes action under this loss function.

### `Expected improvement`

The loss function associated with probability of improvement is somewhat odd: we get a reward for improving upon the current minimum independent of the size of the improvement! This can sometimes lead to odd behavior, and in practice can get stuck in local optima and under-explore globally.

An alternative acquisition function that does account for the size of the improvement is expected improvement. Again, suppose that $f_0$ is the minimal value of $f$ observed so far. Expected improvement evaluates $f$ at the point that, in expectation, improves upon $f_0$ the most. This corresponds to the following utility function:
$$
u(x) = \max(0, f_0 - f(x)).
$$

Here, we receive a reward equal to the “improvement” $f_0 - f(x)$ if $f(x)$ turns out to be less than $f_0$, and no reward is given otherwise. The expected improvement acquisition function is then the expected utility as a function of $x$:
$$
\text{EI}(x) = E[u(x) | x, D] = \int_{-\infty}^{f_0} (f_0 - f) N(f; \mu(x), K(x, x)) df \\
= (f_0 - \mu(x)) \Phi(f_0; \mu(x), K(x,x)) + K(x, x)N(f_0; \mu(x), K(x, x)).
$$

For a 1-D case the equation simplifies to 
$$
\text{EI}(x) = E[u(x) | x, D] = (f_0 - \mu(x)) \Phi\left(\frac{f_0 - \mu(x)}{\sigma(x)}\right) + \sigma(x)\phi\left(\frac{f_0 - \mu(x)}{\sigma(x)}\right)
$$

Here $ \Phi$ and $\phi$ represent the $CDF$, and $PDF$ of standard normal. The point with the highest expected improvement (the maximal expected utility) is selected. The expected improvement has two components. The first can be increased by reducing the mean function $\mu(x)$. The second can be increased by increasing the variance $K(x, x)$. These two terms can be interpreted as explicitly encoding a tradeoff between exploitation (evaluating at points with low mean) and exploration (evaluating at points with high uncertainty).

Here's a short snippet to calculate the EI acquisition function.

```python
# Define the acquisition function (Expected Improvement)
def expected_improvement(X, gaussian_process, evaluated_loss, xi=0.01):
    mean, std = gaussian_process.predict(X, return_std=True)
    Z = (evaluated_loss - mean - xi) / std
    return (evaluated_loss - mean - xi) * norm.cdf(Z) + std * norm.pdf(Z)
```

![Alt text](C:\Users\18503\Internship\Intern_2021\module\fig\EI.jpeg)

The figure illustrates that EI is highest around the function's minimum, concentrating the search in the area of optimal performance. To balance this bias, introducing a trade-off value $\xi$ is crucial. In Expected Improvement, this value determines the sacrificed performance (in original units) when assessing improvement, serving as a compromise between exploration and exploitation.


 
### `Entropy search`


Entropy can be interpreted as a measure of uncertainty or lack of predictability associated with a random variable. For example, consider a sequence of symbols $c_n \sim p$ generated from a distribution $p$. If $p$ has high entropy, predicting the value of each observation $c_n$ becomes challenging. Uniform distributions have maximum entropy, whereas Dirac delta functions have minimum entropy. Here, we seek to minimize the uncertainty we have in the location of the optimal value $x_* = \arg \min_{x \in X} f(x)$. Since we have defined a distribution over $f$, we can always  induces a distribution over $x_*$, $p(x_*| D)$. Unfortunately, there is no closed-form expression for this distribution.

Entropy search seeks to evaluate points so as to minimize the entropy of the induced distribution $p(x_*| D)$. Here the utility function is the reduction in this entropy given a new measurement at $x, f(x)$:

$$ u(x) = H[p(x_* | D)] - H[p(x_* |D \cup \{x, f(x)\})] $$


As in probability of improvement and expected improvement, we may build an acquisition function by evaluating the expected utility provided by evaluating $f$ at a point $x$. Due to the nature of the distribution $p(x_*| D)$, this is somewhat complicated, and a series of approximations must be made for both LHS and RHS term. Please look at [link1](https://botorch.org/tutorials/max_value_entropy) [link2](https://gregorygundersen.com/blog/2020/10/28/predictive-entropy-search/) to see how max value is calculated. 

### `Knowledge Gradient`

please look at the orignal article here [link](https://tiao.io/post/an-illustrated-guide-to-the-knowledge-gradient-acquisition-function/). The article is very well written, i am borrowing/copying this material.

This is not the complete knowledge gradient method, but serve a purpose of understanding what's happening under the hood. It only showed a (naïve) approach to calculating the KG at a given location. Suffice it to say, there is still quite a gap between this and being able to efficiently minimize KG within a sequential decision-making algorithm. For a guide on incorporating KG in a modular and fully-fledged framework for BO please look at [link](https://botorch.org/tutorials/one_shot_kg)


Essentially, the steps to implement KG are:

- Find the maximum of the mean of the posterior predictive distribution from the Gaussian Process (GP) model. This is the mean $ \mu_n(x) = E[y|x,D] = \mu(x; D_n) $, and its maximum is denoted as $ \tau_n(D_n) = \max(\mu_n(x)) $.
- Augment the dataset by adding a new point: define the new dataset $ D_{n+1} = \{D_n \cup \{x_{n+1}, y_{n+1}\}\} $.
- Find the maximum of the posterior predictive mean of the GP model using the dataset $ D_n $, $ \tau_{n+1}(D_{n+1}) = \max(\mu_{n+1}(x)) $.
- Define the acquisition function as $ \alpha(x;D_n) = \mathbb{E}_{p(y|x,D_n)}[\tau_n - \tau_{n+1}] $.
- As the acquisition function can't be solved analytically, use Monte Carlo simulation. Approximate $ \alpha(x;D_n) $ as: 
  $$ \alpha(x;D_n) \approx \frac{1}{M} \left[ \sum_{i=1}^{M} \tau_n - \tau_{n+1}^{(m)} \right] $$
  Here, $ \tau_{n+1}^{(m)} = \tau(D_{n+1}^{(m)}) $ and $ D_{n+1}^{(m)} = \{D_n \cup \{x_{n+1},y_{n+1}^{(m)}\}\} $. Multiple $ y $ samples are obtained for a single $ x $, and the $ x $ is fixed initially.
- This approximation to the knowledge gradient is essentially the average difference between the predictive minimum values based on simulation-augmented data $ \tau_{n+1}^{(m)} $ and that based on observed data $ \tau_n $ across $ M $ simulations.

`One dimensional example`
Synthetic function defined by
$$f(x) = sin(3x) + x^2 - 0.7x$$ 
We generate $n=10$ observations at locations sampled uniformly at random. The true function, and the set of noisy observations $D_n$ are visualized in the figure below: ** ***image******


Posterior predictive distribution.The posterior predictive $p(y|x,D_n)$ is visualized in the figure below. In particular, the predictive mean $\mu_n(x)$ is represented by the solid orange curve. Clearly, this is a poor fit to the data and a uncalibrated estimation of the predictive uncertainly. ********image *****


Step 1

STEP 2

STEP 3

STEP 4


### Continuous fidelity knowledge gradient












