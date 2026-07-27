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
# The MCMC run is described by three numbers: iterations discarded, the
# thinning interval, and how many draws to keep after thinning. The prior on
# the initial state has to be given; the precision priors default to
# data-scaled Half-Cauchys if you leave them out.
fit <- mcmc_normal_locallevel(
  y,
  burnin             = 1000,
  thinning           = 5,
  n_draws            = 1000,
  prior_theta01_mean = y[1],
  prior_theta01_prec = 1 / var(y)
)

# 3. View summary statistics
print(fit)

# 4. Visualize the estimated states
plot(fit, type = "states")

# 5. Look at the chain
plot(fit, type = "mcmc")
```

### Checking convergence

A single chain can only be checked against itself. `mcmc_convergence()` reports
effective sample size and, when `coda` is installed, the Geweke and
Heidelberger–Welch tests:

```r
mcmc_convergence(fit)
```

To find out whether the sampler agrees with *itself from a different starting
point* — which is what catches a run trapped in one region of the posterior —
fit several chains and read the Gelman–Rubin statistic:

```r
fits <- mcmc_normal_locallevel(
  y,
  burnin             = 1000,
  thinning           = 5,
  n_draws            = 1000,
  prior_theta01_mean = y[1],
  prior_theta01_prec = 1 / var(y),
  chains             = 4   # each chain draws its own starting values from the priors
)

mcmc_convergence(fits)   # rank-normalized split-R-hat, bulk and tail ESS
summary(fits)            # pools the chains, and warns first if they disagree
```

Every `mcmc_*()` sampler accepts `chains`, and `chains = 1` (the default)
returns exactly what it always did. `seed` acts as a master seed, so a
multi-chain fit reproduces whether or not it is run in parallel.

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

### Numerical Safeguards

MCMC samplers are only as robust as their weakest arithmetic step: a **single**
degenerate draw can silently poison an otherwise correct chain. A sampled
precision that underflows to `0` turns into an infinite variance and a zero
Cholesky pivot; a latent variable drawn many standard deviations into a tail can
evaluate to `±Inf`; a link that saturates to exactly `0` or `1` flattens the
likelihood and lets an unidentified state wander off. Once an `Inf`/`NaN` enters
the state vector it propagates to every subsequent iteration, and stalling
behaviour (a state that drifts away while its innovation precision collapses)
degrades convergence without ever raising an error.

The C backend therefore applies a small number of guards. They share a common
design principle: **each one only acts where the ordinary floating-point
evaluation of the model has already lost the information the sampler needs**, so
on the identifiable part of the parameter space they leave the target
distribution unchanged and do not affect the underlying theory. In well-behaved
problems they never trigger.

- **Strictly positive precisions** (`rgamma_positive`, `src/utils.c`). Every
  sampled precision ($V^{-1}$, $W_k^{-1}$, $\phi_k$) is floored at machine
  epsilon (`DBL_EPSILON` $\approx 2.2\times10^{-16}$). With small shape
  parameters `rgamma` can return a subnormal value or exactly `0`, which would
  produce an infinite variance $1/\text{prec}$, an infinite standard deviation
  $\sqrt{1/\text{prec}}$, or a zero pivot in the state update. The floor
  corresponds to an astronomically large variance ($\approx 4.5\times10^{15}$),
  so it is statistically inert — it only
  replaces a value that carries no usable information — while keeping every
  downstream computation finite. The negated comparison also traps `NaN`.

- **Finite truncated-normal draws** (`rtruncnorm`, `src/generate_alpha_binomial.c`).
  The Albert–Chib latent variables are drawn by inverse-CDF. When the mean sits
  many standard deviations from the truncation point, `pnorm` rounds to exactly
  `0` or `1` and `qnorm` would return `±Inf`. One-sided truncations are hence
  computed on the *survival* scale (the tail that stays away from `1`) to avoid
  catastrophic cancellation, and the cumulative probabilities are confined to
  `[1e-300, 1 - DBL_EPSILON/2]`. This handles truncations up to $\sim 36$
  standard deviations while leaving typical draws unchanged.

- **Probabilities bounded away from $0$ and $1$** (`clamp_link_alpha`,
  `src/link_guard.h`). Every probability produced by an inverse link is confined
  to $[2\times10^{-16},\, 1 - 2.3\times10^{-16}]$ at the point where it is
  computed. An exact `0` or `1` is not merely extreme: it makes the log-likelihood
  $-\infty$, and it lets the mixture indicator sampler take a deterministic
  branch from which no draw can return. The bounds sit at the numerical
  saturation point of the logit transform, so on any value a well-behaved chain
  visits the guard is inert. **It never touches the latent state**: $\theta$ is
  stored exactly as drawn, and only the derived $\alpha_t$ is protected.

- **Latent-state saturation clamps** (`clamp_probit_state` in
  `src/generate_alpha_binomial.c`; `clamp_logit_state` in `src/cwmh_binomial.c`).
  This guard addresses a different failure, one the probability bound above
  cannot reach. When $\alpha_t = g(\theta_{t,1})$ sits at the boundary over a
  stretch — common with well-separated or segmented data, e.g. aCGH copy-number
  profiles — the likelihood is flat in $\theta_{t,1}$, which is then unidentified
  by the data. The sampler random-walks into the tail, the squared innovations
  $\sum_t (\theta_{t,1} - \theta_{t-1,1})^2$ inflate, and since that sum is the
  sufficient statistic in the update for $W_1^{-1}$, the innovation precision is
  dragged toward zero — which widens the next proposal, which lets the state
  wander further. A positive-feedback loop that stalls the chain without ever
  raising an error.

  Both links are clamped at the **same** bound, $|\theta_{t,1}| \le 36$, even
  though they saturate at very different points ($\Phi$ reaches exactly `1` in
  double precision at $\theta \approx 8.3$; the inverse logit only at
  $\approx 36.7$). The common bound is deliberate. Those clamped states are what
  feed the draw of $W_1^{-1}$ through the sum above, so a per-link bound would
  make the innovation precision's scale depend on which link was chosen rather
  than on the data — the two samplers would no longer be measuring $W_1$ on
  comparable terms.

  This splits the work cleanly between the two guards. For probit in the band
  $8.3 < |\theta| \le 36$, $\alpha_t$ is already numerically at the boundary, so
  it is the probability bound that keeps the likelihood evaluable; the state
  clamp is doing nothing there but standing ready as the brake on the runaway.
  In well-identified problems $|\theta_{t,1}|$ stays far below $36$ and neither
  guard engages.

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
