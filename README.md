# pdm: Polynomial Dynamic Models

<!-- badges: start -->
[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
<!-- badges: end -->

## Overview

**pdm** is an R package for Bayesian inference in **polynomial dynamic linear models** (DLMs) using MCMC methods. The package implements efficient Gibbs samplers and adaptive Metropolis-Hastings algorithms for a variety of observation models and polynomial state structures.

### Key Features

- 🎯 **Multiple observation families**: Normal (Gaussian), Binomial, Poisson, Probit-Bernoulli, Normal Mixture (2 components)
- 📈 **Polynomial state structures**: Local level, local trend, local acceleration
- ⚡ **Optimized C backend**: Fast MCMC sampling via compiled code
- 📊 **Comprehensive diagnostics**: Built-in trace plots, ACF, running means, credible intervals
- 🔧 **Adaptive algorithms**: Automatic tuning for non-Gaussian models
- 📦 **S3 methods**: `plot()`, `summary()`, and `print()` for all models

---

## Installation

The package is under active development. Install directly from GitHub:

```r
# Install devtools if needed
# install.packages("devtools")

devtools::install_github("michelcias/pdm")
```

---

## Implemented Models

### **Normal (Gaussian) Models**

Observation model: $y_t \sim \mathcal{N}(\theta_{t,1}, V)$

| State Structure | Function |
|----------------|----------|
| Local Level | `mcmc_normal_locallevel()` |
| Local Trend | `mcmc_normal_localtrend()` |
| Local Acceleration | `mcmc_normal_localacceleration()` |

---

For all non-Gaussian models below, the latent state $\theta_{t,1}$ (and higher-order states for trend/acceleration models) evolves polynomially, and $\alpha_t$ is the observation-level parameter obtained via a link function:

$$\alpha_t = g(\theta_{t,1}),$$

where $g(\cdot)$ is the link function specific to each observation family.

### **Binomial Models**

Observation model: $y_t \sim \text{Binomial}(n_{\text{trials}}, \alpha_t)$  
Link function: $\alpha_t = \text{logit}^{-1}(\theta_{t,1})$ \ \ (success probability)

| State Structure | Function |
|----------------|----------|
| Local Level | `mcmc_binomial_locallevel()` |
| Local Trend | `mcmc_binomial_localtrend()` |
| Local Acceleration | `mcmc_binomial_localacceleration()` |

### **Poisson Models**

Observation model: $y_t \sim \text{Poisson}(\alpha_t)$  
Link function: $\alpha_t = \exp(\theta_{t,1})$ \ \ (rate parameter)

| State Structure | Function |
|----------------|----------|
| Local Level | `mcmc_poisson_locallevel()` |
| Local Trend | `mcmc_poisson_localtrend()` |
| Local Acceleration | `mcmc_poisson_localacceleration()` |

### **Probit-Bernoulli Models**

Observation model: $y_t \sim \text{Bernoulli}(\alpha_t)$  
Link function: $\alpha_t = \Phi(\theta_{t,1})$ \ \ (success probability via probit link)

| State Structure | Function |
|----------------|----------|
| Local Level | `mcmc_probit_bernoulli_locallevel()` |
| Local Trend | `mcmc_probit_bernoulli_localtrend()` |
| Local Acceleration | `mcmc_probit_bernoulli_localacceleration()` |

---

### **Normal Mixture Models (2 Components)**

Observation model: 

$$y_t \sim \alpha_t \cdot \mathcal{N}(\mu_1, \phi_1^{-1}) + (1 - \alpha_t) \cdot \mathcal{N}(\mu_2, \phi_2^{-1})$$

The latent state $\theta_{t,1}$ (and higher-order states for trend/acceleration models) evolves polynomially, and $\alpha_t$ (mixture weight) is obtained via:

- **Logit link**: $\alpha_t = \text{logit}^{-1}(\theta_{t,1})$
- **Probit link**: $\alpha_t = \Phi(\theta_{t,1})$

| State Structure | Function |
|----------------|----------|
| Local Level | `mcmc_normal_mixture_locallevel()` |
| Local Trend | `mcmc_normal_mixture_localtrend()` |
| Local Acceleration | `mcmc_normal_mixture_localacceleration()` |

Each function supports both `link = "logit"` and `link = "probit"`.

---

## Methodological Details

### **State Evolution**

All models share the polynomial state-space structure. The state vector at time $t$ is denoted by $\boldsymbol{\theta}_t = (\theta_{t,1}, \theta_{t,2}, \theta_{t,3})^\top$, where each component evolves according to the model order:

#### **Local Level**

$$
\theta_{t,1} = \theta_{t-1,1} + u_{t,1}, \quad u_{t,1} \sim \mathcal{N}(0, W_1)
$$

- $\theta_{t,1}$: latent level state
- $W_1$: level innovation variance

---

#### **Local Trend**

$$
\begin{aligned}
\theta_{t,1} &= \theta_{t-1,1} + \theta_{t-1,2} + u_{t,1}, \quad &u_{t,1} &\sim \mathcal{N}(0, W_1) \\
\theta_{t,2} &= \theta_{t-1,2} + u_{t,2}, \quad &u_{t,2} &\sim \mathcal{N}(0, W_2)
\end{aligned}
$$

- $\theta_{t,1}$: latent level state
- $\theta_{t,2}$: latent trend state
- $W_1, W_2$: level and trend innovation variances

---

#### **Local Acceleration**

$$
\begin{aligned}
\theta_{t,1} &= \theta_{t-1,1} + \theta_{t-1,2} + u_{t,1}, \quad &u_{t,1} &\sim \mathcal{N}(0, W_1) \\
\theta_{t,2} &= \theta_{t-1,2} + \theta_{t-1,3} + u_{t,2}, \quad &u_{t,2} &\sim \mathcal{N}(0, W_2) \\
\theta_{t,3} &= \theta_{t-1,3} + u_{t,3}, \quad &u_{t,3} &\sim \mathcal{N}(0, W_3)
\end{aligned}
$$

- $\theta_{t,1}$: latent level state
- $\theta_{t,2}$: latent trend state  
- $\theta_{t,3}$: latent acceleration state
- $W_1, W_2, W_3$: level, trend, and acceleration innovation variances

---

### **Observation Models**

#### **Normal (Gaussian)**

$$
y_t \sim \mathcal{N}(\theta_{t,1}, V)
$$

where $V$ is the observation variance.

---

#### **Non-Gaussian Models**

For non-Gaussian observation families, we introduce a link function $g(\cdot)$ that maps the latent state to the observation parameter $\alpha_t$:

$$
\alpha_t = g(\theta_{t,1})
$$

**Binomial (Logit Link):**
$$
\begin{aligned}
y_t &\sim \text{Binomial}(n, \alpha_t) \\
\alpha_t &= \text{logit}^{-1}(\theta_{t,1}) = \frac{\exp(\theta_{t,1})}{1 + \exp(\theta_{t,1})}
\end{aligned}
$$

**Poisson (Log Link):**
$$
\begin{aligned}
y_t &\sim \text{Poisson}(\alpha_t) \\
\alpha_t &= \exp(\theta_{t,1}) 
\end{aligned}
$$

**Probit-Bernoulli (Probit Link):**
$$
\begin{aligned}
y_t &\sim \text{Bernoulli}(\alpha_t) \\
\alpha_t &= \Phi(\theta_{t,1}) 
\end{aligned}
$$

where $\Phi(\cdot)$ is the standard normal cumulative distribution function.

---

#### **Normal Mixture Models (2 Components)**

$$
y_t \sim \alpha_t \cdot \mathcal{N}(\mu_1, \phi_1^{-1}) + (1 - \alpha_t) \cdot \mathcal{N}(\mu_2, \phi_2^{-1})
$$

where:

- $\mu_1, \mu_2$: component means
- $\phi_1, \phi_2$: component precisions
- $\alpha_t \in [0,1]$: time-varying mixture weight

The mixture weight $\alpha_t$ is obtained via a link function:

**Logit Link:**
$$
\alpha_t = \text{logit}^{-1}(\theta_{t,1}) = \frac{\exp(\theta_{t,1})}{1 + \exp(\theta_{t,1})}
$$

**Probit Link:**
$$
\alpha_t = \Phi(\theta_{t,1})
$$

---

### **Prior Distributions**

All models employ conjugate or semi-conjugate priors to facilitate efficient Gibbs sampling:

#### **Initial States**
$$
\begin{aligned}
\theta_{0,1} &\sim \mathcal{N}(\mu_{01}, \tau_{01}^{-1}) \\
\theta_{0,2} &\sim \mathcal{N}(\mu_{02}, \tau_{02}^{-1}) \quad \text{(trend and acceleration models)} \\
\theta_{0,3} &\sim \mathcal{N}(\mu_{03}, \tau_{03}^{-1}) \quad \text{(acceleration models)}
\end{aligned}
$$

#### **Innovation Precisions**
$$
W_i^{-1} \sim \text{Gamma}(\nu_i, \eta_i), \quad i = 1, 2, 3
$$

#### **Observation Precision (Normal Models)**
$$
V^{-1} \sim \text{Gamma}(\nu_V, \eta_V)
$$

#### **Mixture Component Parameters (Mixture Models)**
$$
\begin{aligned}
\mu_1, \mu_2 &\sim \mathcal{N}(\mu_0, \tau_0^{-1}) \\
\phi_1, \phi_2 &\sim \text{Gamma}(\nu_\phi, \eta_\phi)
\end{aligned}
$$

---

### **MCMC Algorithms**

#### **Normal Models**
Gibbs sampler with conjugate full conditional distributions for all parameters.

#### **Non-Gaussian Models**
Adaptive Metropolis-Hastings algorithm with:

- **Component-wise sampling** of latent states $\theta_{t,1}$ (and $\theta_{t,2}$, $\theta_{t,3}$ for higher-order models)
- **Automatic proposal tuning** via diminishing adaptation:
  $$
  \sigma_t^{(\text{prop})} = \sigma_{t-1}^{(\text{prop})} \cdot \exp\left(\frac{\gamma_0}{t^\kappa} \cdot (a_t - a^*)\right)
  $$
  where:
  - $a_t$: current acceptance proportion
  - $a^*$: target acceptance rate (default: 0.44)
  - $\gamma_0$: base adaptation rate
  - $\kappa$: decay exponent (default: 0.6)

- **Conjugate updates** for innovation precisions $W_i^{-1}$ via Gamma full conditionals

---

### **Diagnostic Tools**

Each model includes comprehensive `plot()` methods with multiple visualization types:

#### **MCMC Diagnostics** (`type = "mcmc"`)
- Trace plots for visual convergence assessment
- Autocorrelation functions (ACF) to detect serial dependence
- Running means to check stability
- Posterior density estimates

#### **State Trajectories** (`type = "states"`)
- Posterior median of $\theta_{t,1}$ (and $\theta_{t,2}$, $\theta_{t,3}$ for higher-order models)
- Credible intervals (default: 95%)
- Optional overlay of true state values (for simulation studies)

#### **Observation-Level Parameters** (`type = "alpha"`)
- Time evolution of $\alpha_t$ (rates, probabilities, or mixture weights)
- Credible bands
- Optional overlay of observed data

#### **Acceptance Rates** (`type = "acceptance"`, non-Gaussian models only)
- Metropolis-Hastings acceptance proportions over iterations
- Target acceptance rate reference line
- Diagnostic for proposal tuning effectiveness

---

## Development Status

| Component | Status |
|-----------|--------|
| Core C implementations | ✅ Complete |
| R interface functions | ✅ Complete |
| S3 methods (plot/summary/print) | ✅ Complete |
| Documentation | ✅ Complete |
| Unit tests | 🚧 In progress |
| Vignettes | 📅 Planned |
| CRAN submission | 📅 Future |

---

## Roadmap

### **Short-term**
- [ ] Complete unit test coverage
- [ ] Add model comparison tools (DIC, WAIC)
- [ ] Vignettes with real-world examples

### **Medium-term**
- [ ] Seasonal components
- [ ] Regression covariates
- [ ] Additional mixture components (k > 2)

### **Long-term**
- [ ] Explore generalization to broader Bayesian dynamic model framework
- [ ] Potential new package (`bdm`) for non-polynomial structures (decision pending)

See the [project issues](https://github.com/michelcias/pdm/issues) for detailed discussions.

---

## References

### **Theoretical Foundations**

- Montoril, M. H., Correia, L. T., & Migon, H. S. (2021). Bayesian estimation of dynamic weights in Gaussian mixture models. *arXiv preprint* arXiv:2104.03395. [https://arxiv.org/abs/2104.03395](https://arxiv.org/abs/2104.03395)
- West, M., & Harrison, J. (1997). *Bayesian Forecasting and Dynamic Models* (2nd ed.). Springer.
- Petris, G., Petrone, S., & Campagnoli, P. (2009). *Dynamic Linear Models with R*. Springer.

### **Mixture Models with Dynamic Weights**

- Montoril, M. H., Correia, L. T., & Migon, H. S. (2021). Bayesian estimation of dynamic weights in Gaussian mixture models. *arXiv preprint* arXiv:2104.03395. [https://arxiv.org/abs/2104.03395](https://arxiv.org/abs/2104.03395)

### **Adaptive MCMC**

- Roberts, G. O., & Rosenthal, J. S. (2009). Examples of adaptive MCMC. *Journal of Computational and Graphical Statistics*, 18(2), 349-367.

### **Data Augmentation (Probit Model)**

- Albert, J. H., & Chib, S. (1993). Bayesian analysis of binary and polychotomous response data. *Journal of the American Statistical Association*, 88(422), 669-679.

---

## Contributing

This package is under active development. Contributions, bug reports, and feature requests are welcome via [GitHub Issues](https://github.com/michelcias/pdm/issues).

---

## License

GPL (>= 3)

---

## Author

**Michel H. Montoril**  
GitHub: [@michelcias](https://github.com/michelcias)
