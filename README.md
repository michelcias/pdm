# pdm: Polynomial Dynamic Models

<!-- badges: start -->
<!-- badges: end -->

## Overview

**pdm** is an R package designed for efficient Bayesian inference in polynomial dynamic linear models (DLMs).

Unlike standard DLM packages that often focus primarily on Gaussian data, **pdm** provides a unified framework for non-Gaussian dynamic modeling. It leverages a high-performance C backend to implement efficient Gibbs samplers and adaptive Metropolis-Hastings algorithms, enabling fast estimation even for complex state structures.

A key methodological feature of **pdm** is its efficient treatment of higher-order states as hyperparameters, leveraging the Markovian structure of the models.

## Key Features

- ⚡ **High-Performance C Backend**: Fast MCMC sampling via optimized compiled code.
- 🎯 **Non-Gaussian Support**: Native support for Binomial, Poisson, Probit-Bernoulli, and Normal Mixture models.
- 🔧 **Adaptive Algorithms**: Automatic tuning of Metropolis-Hastings proposals for non-Gaussian models (no manual tuning required).
- 📈 **Polynomial Structures**: Built-in support for Local Level, Local Trend, and Local Acceleration states.
- 📊 **Comprehensive Diagnostics**: S3 methods (`plot`, `summary`, `print`) for immediate analysis of convergence and state trajectories.

## Installation

The package is under active development. You can install it directly from GitHub:

```r
# Install devtools if needed
# install.packages("devtools")

devtools::install_github("michelcias/pdm")
```

## Quick Start

Here is a simple example fitting a Local Level model to Gaussian data.

```r
library(pdm)

# 1. Simulate data (Random Walk + Noise)
set.seed(123)
y <- cumsum(rnorm(100)) + rnorm(100, sd = 0.5)

# 2. Fit the model
# The package uses efficient defaults for priors and tuning
fit <- mcmc_normal_locallevel(y, n_iter = 2000, burnin = 1000)

# 3. View summary statistics
print(fit)

# 4. Visualize the estimated states
plot(fit, type = "states")

# 5. Check MCMC convergence
plot(fit, type = "mcmc")
```

## Implemented Models

The package organizes models by **Observation Family** (the distribution of $y_t$) and **State Structure** (the evolution of $\theta_t$).

Function names follow the pattern: `mcmc_{family}_{structure}()`.

### 1. Normal (Gaussian) Models

$$y_t \sim \mathcal{N}(\theta_{t,1}, V).$$

| Structure | Function |
|-----------|----------|
| Local Level | `mcmc_normal_locallevel()` |
| Local Trend | `mcmc_normal_localtrend()` |
| Local Acceleration | `mcmc_normal_localacceleration()` |

### 2. Binomial Models

$$y_t \sim \text{Binomial}(n_{\text{trials}}, \alpha_t),$$

where $\alpha_t = \text{logit}^{-1}(\theta_{t,1}).$

| Structure | Function |
|-----------|----------|
| Local Level | `mcmc_binomial_locallevel()` |
| Local Trend | `mcmc_binomial_localtrend()` |
| Local Acceleration | `mcmc_binomial_localacceleration()` |

### 3. Poisson Models

$$y_t \sim \text{Poisson}(\alpha_t),$$

where $\alpha_t = \exp(\theta_{t,1}).$

| Structure | Function |
|-----------|----------|
| Local Level | `mcmc_poisson_locallevel()` |
| Local Trend | `mcmc_poisson_localtrend()` |
| Local Acceleration | `mcmc_poisson_localacceleration()` |

### 4. Probit-Bernoulli Models

$$y_t \sim \text{Bernoulli}(\alpha_t),$$

where $\alpha_t = \Phi(\theta_{t,1})$ uses the standard normal CDF.

| Structure | Function |
|-----------|----------|
| Local Level | `mcmc_probit_bernoulli_locallevel()` |
| Local Trend | `mcmc_probit_bernoulli_localtrend()` |
| Local Acceleration | `mcmc_probit_bernoulli_localacceleration()` |

### 5. Normal Mixture Models (2 Components)

Modeling data from two switching regimes. The observation $y_t$ is drawn from one of two Gaussian components depending on a latent indicator $z_t \in \{0, 1\}$:

$$
y_t \mid z_t \sim \mathcal{N}(\mu_{z_t+1}, \phi_{z_t+1}^{-1}), \quad z_t \in \{0, 1\}.
$$

The success probability of $z_t$ evolves dynamically:

$$P(z_t = 1) = \alpha_t = g(\theta_{t,1}), \quad g \in \{\text{logit}^{-1}, \Phi\}.$$

**Label Switching**: To ensure model identifiability and handle the label switching problem, the constraint $\mu_1 < \mu_2$ is enforced during the sampling process.

| Structure | Function |
|-----------|----------|
| Local Level | `mcmc_normal_mixture_locallevel()` |
| Local Trend | `mcmc_normal_mixture_localtrend()` |
| Local Acceleration | `mcmc_normal_mixture_localacceleration()` |

## Methodological Details

The polynomial dynamic model framework and the dynamic mixture approach implemented in this package are based on the methodology proposed by **Montoril et al. (2021)**.

### State Evolution Structures

The latent states evolve according to polynomial structures defined by the following stochastic difference equations.

#### Local Level:

$$\theta_{t,1} = \theta_{t-1,1} + u_{t,1}, \quad u_{t,1} \sim \mathcal{N}(0, W_1).$$

#### Local Trend:

$$\begin{aligned}
\theta_{t,1} &= \theta_{t-1,1} + \theta_{t-1,2} + u_{t,1}, & u_{t,1} &\sim \mathcal{N}(0, W_1), \\
\theta_{t,2} &= \theta_{t-1,2} + u_{t,2}, & u_{t,2} &\sim \mathcal{N}(0, W_2).
\end{aligned}$$

#### Local Acceleration:

$$\begin{aligned}
\theta_{t,1} &= \theta_{t-1,1} + \theta_{t-1,2} + u_{t,1}, & u_{t,1} &\sim \mathcal{N}(0, W_1), \\
\theta_{t,2} &= \theta_{t-1,2} + \theta_{t-1,3} + u_{t,2}, & u_{t,2} &\sim \mathcal{N}(0, W_2), \\
\theta_{t,3} &= \theta_{t-1,3} + u_{t,3}, & u_{t,3} &\sim \mathcal{N}(0, W_3).
\end{aligned}$$

### MCMC Algorithms

The package implements hybrid MCMC strategies to ensure efficiency and convergence:

- **Normal Models**: Gibbs samplers with conjugate full conditional distributions for all parameters.
- **Probit Models**: Exact Gibbs sampling via data augmentation (Albert & Chib, 1993).
- **Non-Gaussian Models (Binomial/Poisson/Logit-Mixture)**:
  - Uses Adaptive Metropolis-Hastings steps for the latent states.
  - Implements the diminishing adaptation scheme of Roberts & Rosenthal (2009).
  - **User Benefit**: The proposal variances are automatically tuned throughout the MCMC chain to converge towards a target acceptance proportion (user-defined), removing the need for manual parameter tuning.

### Diagnostic Tools

The package provides a unified `plot()` method with the `type` argument:

- `plot(fit, type = "states")`: Posterior medians and credible intervals for $\theta_t$.
- `plot(fit, type = "alpha")`: Transformed parameters (probabilities or rates) on the observation scale.
- `plot(fit, type = "mcmc")`: Trace plots, ACF, and density estimates for convergence checks.
- `plot(fit, type = "acceptance")`: (For adaptive models) Tracks the evolution of acceptance rates over time.

## References

- Montoril, M. H., Correia, L. T., & Migon, H. S. (2021). Bayesian estimation of dynamic weights in Gaussian mixture models. *arXiv preprint arXiv:2104.03395*. https://arxiv.org/abs/2104.03395
- West, M., & Harrison, J. (1997). *Bayesian Forecasting and Dynamic Models* (2nd ed.). Springer.
- Petris, G., Petrone, S., & Campagnoli, P. (2009). *Dynamic Linear Models with R*. Springer.
- Roberts, G. O., & Rosenthal, J. S. (2009). Examples of adaptive MCMC. *Journal of Computational and Graphical Statistics*, 18(2), 349-367.
- Albert, J. H., & Chib, S. (1993). Bayesian analysis of binary and polychotomous response data. *JASA*, 88(422), 669-679.

## Contributing

This package is under active development. Contributions, bug reports, and feature requests are welcome via [GitHub Issues](https://github.com/michelcias/pdm/issues).

## License

GPL (>= 3)

## Author

**Michel H. Montoril**

GitHub: [@michelcias](https://github.com/michelcias)
