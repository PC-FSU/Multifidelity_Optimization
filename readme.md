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

   c. Augment Data $D = {\{D, (x_{n+1}, y_{n+1}, s_{n+1})}\}$

   d. Update GP model, $M$.

# Acquisition function
`The next section is a short theoretical primer about acquisition function. Feel free to skip this, keeping in mind that "Acquisition functions are mathematical techniques that guide the exploration of the parameter space during Bayesian optimization."` You can find the PDF [here](https://github.com/PC-FSU/Multifidelity_Optimization/blob/main/Acquisition_Function.pdf)

![Screenshot](https://github.com/PC-FSU/Multifidelity_Optimization/blob/main/readme_figs/Acquisition_Function_page-0001.jpg)
![Screenshot](https://github.com/PC-FSU/Multifidelity_Optimization/blob/main/readme_figs/Acquisition_Function_page-0002.jpg)
![Screenshot](https://github.com/PC-FSU/Multifidelity_Optimization/blob/main/readme_figs/Acquisition_Function_page-0003.jpg)
![Screenshot](https://github.com/PC-FSU/Multifidelity_Optimization/blob/main/readme_figs/Acquisition_Function_page-0004.jpg)
![Screenshot](https://github.com/PC-FSU/Multifidelity_Optimization/blob/main/readme_figs/Acquisition_Function_page-0005.jpg)
![Screenshot](https://github.com/PC-FSU/Multifidelity_Optimization/blob/main/readme_figs/Acquisition_Function_page-0006.jpg)
![Screenshot](https://github.com/PC-FSU/Multifidelity_Optimization/blob/main/readme_figs/Acquisition_Function_page-0007.jpg)
![Screenshot](https://github.com/PC-FSU/Multifidelity_Optimization/blob/main/readme_figs/Acquisition_Function_page-0008.jpg)


# Test functions












