#' @title Gibbs Sampler for a Poisson Mixture Model with Local-Trend Mixture Weights
#'
#' @description Runs a Gibbs sampler for a two-component Poisson mixture model
#'   with time-varying mixture weights governed by a local-trend dynamic
#'   structure. Supports both logit and probit link functions.
#'
#' @details The model is defined as:
#' \deqn{
#' \begin{aligned}
#' y_t | z_t, \lambda &\sim \text{Poisson}(z_t \cdot \lambda_2 + (1 - z_t) \cdot \lambda_1), \\
#' z_t | \alpha_t &\sim \text{Bernoulli}(\alpha_t), \\
#' \alpha_t &= T^{-1}(\theta_{t,1}), \\
#' \theta_{t,1} &= \theta_{t-1,1} + \theta_{t-1,2} + u_{t,1}, & u_{t,1} \sim N(0, W_1), \\
#' \theta_{t,2} &= \theta_{t-1,2} + u_{t,2}, & u_{t,2} \sim N(0, W_2),
#' \end{aligned}
#' }
#' where \eqn{t = 1, 2, \ldots, n}, \eqn{n} is the number of observations,
#' \eqn{z_t \in \{0, 1\}} are latent indicators (0 = component 1, 1 = component 2),
#' and \eqn{\lambda = (\lambda_1, \lambda_2)'} are the component rates.
#'
#' \strong{Which quantity the link carries.} This family and
#' \code{\link{mcmc_poisson_localtrend}} both put a Poisson observation on a
#' dynamic model, and they are not variants of one another. There the latent
#' state drives the \emph{rate}, through a log link, and every observation comes
#' from the same evolving Poisson. Here it drives the \emph{mixture weight}: the
#' two rates are constants, and what moves through time is the probability that
#' an observation comes from the high-rate component rather than the low-rate
#' one. Use this one when the counts look like a regime-switching series --
#' quiet stretches interrupted by busy ones -- rather than one rate drifting.
#'
#' The link function \eqn{T^{-1}} determines how the latent state \eqn{\theta_{t,1}}
#' maps to the mixture weight \eqn{\alpha_t}:
#'
#' \strong{Logit Link:}
#' \deqn{\alpha_t = \frac{\exp(\theta_{t,1})}{1 + \exp(\theta_{t,1})}}
#' Uses component-wise Metropolis-Hastings with adaptive tuning. Allows optional
#' diagnostics (log_sigma, accept_prop) to monitor proposal adaptation and
#' acceptance rates.
#'
#' \strong{Probit Link:}
#' \deqn{\alpha_t = \Phi(\theta_{t,1})}
#' where \eqn{\Phi(\cdot)} is the standard normal CDF. Uses Albert-Chib (1993)
#' data augmentation for efficient Gibbs sampling. Since this is a pure Gibbs
#' sampler, the acceptance rate is always 1.0.
#'
#' \strong{Prior Distributions:}
#'
#' \emph{Component rates:}
#' \deqn{\lambda_1 \sim \text{Gamma}(a_{01}, b_{01}), \quad
#'       \lambda_2 \sim \text{Gamma}(a_{02}, b_{02}),}
#' in the shape-rate parameterization. The Gamma is conjugate for a Poisson
#' rate, so both components are drawn exactly, with no tuning and no auxiliary
#' variable. Unlike the Gaussian mixture's component precisions, the rates admit
#' no Half-t option: that prior exists to tame the likelihood singularity at zero
#' variance, and a Poisson rate has no such singularity to tame.
#'
#' \emph{Initial states:}
#' \deqn{\theta_{0,1} \sim N(\mu_{\theta_{01}}, \tau_{\theta_{01}}^{-1}), \quad
#'       \theta_{0,2} \sim N(\mu_{\theta_{02}}, \tau_{\theta_{02}}^{-1}).}
#'
#' \emph{State innovation precisions:}
#'
#' Each of \eqn{W_1^{-1}} and \eqn{W_2^{-1}} may take one of two priors, chosen
#' independently through the corresponding `*_type` argument:
#'
#' \emph{(a) Gamma prior on the precision} (conjugate):
#' \deqn{W_1^{-1} \sim \text{Gamma}(\nu_1, \eta_1), \quad
#'       W_2^{-1} \sim \text{Gamma}(\nu_2, \eta_2).}
#'
#' \emph{(b) Half-t prior on the standard deviation} (Gelman, 2006; the default):
#' \deqn{\sqrt{W_1} \sim \text{Half-}t(\nu_1, A_1), \quad
#'       \sqrt{W_2} \sim \text{Half-}t(\nu_2, A_2),}
#' where \eqn{A > 0} is a scale hyperparameter and \eqn{\nu > 0} the degrees of
#' freedom; \eqn{\nu = 1} gives the Half-Cauchy prior (also selectable with
#' `type = "halfcauchy"`), and larger \eqn{\nu} approaches a Half-Normal. The
#' Half-t prior is represented by the inverse-gamma scale mixture of Wand et al.
#' (2011), which keeps every Gibbs update closed-form via a single auxiliary
#' variable per precision.
#'
#' The prior hyperparameters correspond to function arguments as follows:
#' \tabular{cc}{
#'   \strong{Hyperparameter} \tab \strong{Function Argument} \cr
#'   \eqn{a_{01}} (shape) \tab `prior_lambda01_shape` \cr
#'   \eqn{b_{01}} (rate) \tab `prior_lambda01_rate` \cr
#'   \eqn{a_{02}} (shape) \tab `prior_lambda02_shape` \cr
#'   \eqn{b_{02}} (rate) \tab `prior_lambda02_rate` \cr
#'   \eqn{\mu_{\theta_{01}}} \tab `prior_theta01_mean` \cr
#'   \eqn{\tau_{\theta_{01}}} \tab `prior_theta01_prec` \cr
#'   \eqn{\mu_{\theta_{02}}} \tab `prior_theta02_mean` \cr
#'   \eqn{\tau_{\theta_{02}}} \tab `prior_theta02_prec` \cr
#'   \eqn{\nu_1} (shape), \eqn{A_1} (scale) \tab `prior_prec1_shape`, `prior_prec1_scale` \cr
#'   \eqn{\eta_1} (rate), \eqn{\nu_1} (df) \tab `prior_prec1_rate`, `prior_prec1_df` \cr
#'   \eqn{\nu_2} (shape), \eqn{A_2} (scale) \tab `prior_prec2_shape`, `prior_prec2_scale` \cr
#'   \eqn{\eta_2} (rate), \eqn{\nu_2} (df) \tab `prior_prec2_rate`, `prior_prec2_df`
#' }
#'
#' \strong{Label Switching:}
#' To ensure identifiability, the constraint \eqn{\lambda_1 < \lambda_2} is
#' enforced by swapping components when necessary during sampling. Component 1
#' is therefore the low-rate component by construction.
#'
#' \strong{Adaptive Metropolis-Hastings Algorithm (Logit Link Only):}
#'
#' When using the logit link, the algorithm employs component-wise Metropolis-Hastings
#' for sampling the latent states \eqn{\theta_{t,1}}, with adaptive proposal tuning
#' based on acceptance proportions.
#'
#' The adaptive algorithm uses a diminishing adaptation schedule that ensures
#' theoretical convergence guarantees (Roberts and Rosenthal, 2007). Adaptation
#' occurs every `lag_update` iterations (e.g., at MCMC iterations 50, 100, 150, ...
#' if `lag_update = 50`), and only after sufficient history has been accumulated
#' (iteration >= `lag_update`).
#'
#' At each adaptation point (MCMC iteration \eqn{m}), for each time point \eqn{t},
#' the proposal log-scale standard deviation is updated according to:
#'
#' \deqn{\log(\sigma_t) \leftarrow \log(\sigma_t) + \text{sign}(\hat{p}_t - p^*) \cdot \gamma_m \cdot \mathbb{1}_{\{|\hat{p}_t - p^*| > \tau\}},}
#'
#' where:
#' \itemize{
#'   \item \eqn{\gamma_m} is the diminishing step size at MCMC iteration \eqn{m}, computed as
#'     \deqn{\gamma_m = \min\left(\text{max\_step\_size}, \frac{\text{base\_adaptation\_rate}}{m^\xi}\right);}
#'   \item \eqn{\xi} is the decay exponent (`decay_exponent`) applied to the MCMC iteration number;
#'   \item \eqn{\hat{p}_t} is the empirical acceptance proportion at time point \eqn{t}
#'     over the last `lag_update` MCMC iterations;
#'   \item \eqn{p^*} is the target acceptance rate (`target_acceptance`);
#'   \item \eqn{\tau} is the minimum deviation threshold (`min_deviation_threshold`);
#'   \item \eqn{\mathbb{1}_{\{|\hat{p}_t - p^*| > \tau\}}} is an indicator function that equals 1 when
#'     the condition is true, 0 otherwise.
#' }
#'
#' The update only occurs if the absolute deviation exceeds the threshold:
#' \deqn{|\hat{p}_t - p^*| > \tau.}
#'
#' This threshold-based approach prevents spurious updates due to random fluctuations.
#'
#' \strong{Recommended Settings:}
#'
#' For most applications:
#' \itemize{
#'   \item `lag_update = 50`: Provides good balance between stability and responsiveness;
#'   \item `max_step_size = 0.1`: Conservative adjustment rate;
#'   \item `base_adaptation_rate = 1.0`: Moderate adaptation speed;
#'   \item `decay_exponent = 0.6`: Standard diminishing adaptation;
#'   \item `target_acceptance = 0.44`: Theoretically optimal for univariate proposals;
#'   \item `min_deviation_threshold = NULL`: Uses practical default of 1/lag_update.
#' }
#'
#' \strong{Progress Bar:}
#' When `verbose = TRUE`, a visual progress bar is displayed showing
#' \itemize{
#' \item{Progress bar with adaptive update frequency (based on `bar_width`)}
#' \item{Elapsed time in HH:MM:SS format}
#' \item{Estimated remaining time in HH:MM:SS format}
#' }
#'
#' Burn-in and thinning are applied so that exactly `n_draws` posterior
#' samples are returned.
#'
#'
#' \strong{Starting values:}
#'
#' By default the chain starts from a draw from each parameter's own prior. That
#' is what makes several chains disperse, and it is what `chains > 1` relies on,
#' but it can also start the chain a long way from the bulk of the posterior --
#' a Half-Cauchy draw for a standard deviation is occasionally enormous -- and
#' burn-in then pays for it. `init` pins any subset of the parameters to values
#' of your choosing; everything left out is still drawn from its prior.
#'
#' The names are the fitted object's own components, which is also the
#' vocabulary the `plot()` method's `true_values` argument uses, so one name
#' means one parameter across the package. `init` covers the scalar parameters
#' only. Two components of this model are deliberately left out: the latent
#' trajectories and the `alpha` weights derived from them, and the indicators
#' `z`. `z` is drawn as `Bernoulli(0.5)` inside the sampler and stays there,
#' because it is a per-observation vector rather than a scalar, and because that
#' draw is a source of the between-chain dispersion
#' \code{\link{mcmc_convergence}} needs.
#'
#' This model identifies its components only up to their order, so the sampler
#' requires \eqn{\lambda_1 \leq \lambda_2}. When both rates are drawn and come
#' out the wrong way round they are simply relabelled. When you pin either one
#' through `init` and the resolved pair violates the order, that is an error
#' instead: relabelling would move a value you asked for into the other
#' component.
#' @param y Numeric vector of observed counts (length \eqn{n}). Every element
#'   must be a non-negative integer.
#' @param link Character string specifying the link function. Must be either
#'   `"logit"` or `"probit"`. Default is `"logit"`.
#' @param burnin Integer \eqn{\geq 0}, number of burn-in iterations.
#' @param thinning Integer \eqn{\geq 1}, thinning interval.
#' @param n_draws Integer \eqn{\geq 1}, number of posterior samples to retain.
#' @param prior_lambda01_shape Numeric > 0, shape \eqn{a_{01}} of the Gamma prior
#'   for the rate of component 1 (\eqn{\lambda_1}). Default is 2, which gives a
#'   prior with a finite mode and a moderate tail; the shape controls how tightly
#'   the component is pinned to the target implied by the rate.
#' @param prior_lambda01_rate Numeric > 0, rate \eqn{b_{01}} of the Gamma prior
#'   for \eqn{\lambda_1}. If `NULL` (default), set to
#'   `prior_lambda01_shape / max(quantile(y, 0.25), 0.5)`, so the prior mean sits
#'   at the lower quartile of the counts. The components are identified by the
#'   constraint \eqn{\lambda_1 < \lambda_2}, which the sampler enforces by
#'   relabelling after each draw, so component 1 is the low-rate one by
#'   construction; priors placing component 1 above component 2 ask for the
#'   reverse and raise a warning.
#' @param prior_lambda02_shape Numeric > 0, shape \eqn{a_{02}} of the Gamma prior
#'   for the rate of component 2 (\eqn{\lambda_2}). Default is 2.
#' @param prior_lambda02_rate Numeric > 0, rate \eqn{b_{02}} of the Gamma prior
#'   for \eqn{\lambda_2}. If `NULL` (default), set to
#'   `prior_lambda02_shape / max(quantile(y, 0.75), 0.5)`, so the prior mean sits
#'   at the upper quartile of the counts.
#' @param prior_theta01_mean Numeric, prior mean for the initial level state
#'   \eqn{\theta_{0,1}}. Default is 0.
#' @param prior_theta01_prec Numeric > 0, prior precision (inverse variance) for
#'   \eqn{\theta_{0,1}}. Default is 0.01 (vague prior).
#' @param prior_theta02_mean Numeric, prior mean for the initial trend state
#'   \eqn{\theta_{0,2}}. Default is 0.
#' @param prior_theta02_prec Numeric > 0, prior precision (inverse variance) for
#'   \eqn{\theta_{0,2}}. Default is 0.01 (vague prior).
#' @param prior_prec1_type Character, prior on the level innovation
#'   precision \eqn{1/W_1}: `"halfcauchy"` (default) or `"halft"` for a Half-t /
#'   Half-Cauchy prior on \eqn{\sqrt{W_1}} (Gelman, 2006), or `"gamma"` for a
#'   Gamma prior on the precision. A call that supplies
#'   `prior_prec1_shape`/`prior_prec1_rate` without naming a type is read as
#'   `"gamma"`.
#' @param prior_prec1_shape Numeric > 0, shape parameter of the Gamma prior for
#'   the level innovation precision \eqn{1/W_1}. Default is 0.01 (vague prior).
#'   Used only when `prior_prec1_type = "gamma"`.
#' @param prior_prec1_rate Numeric > 0, rate parameter of the Gamma prior for
#'   \eqn{1/W_1}. Default is 0.01 (vague prior). Used only when
#'   `prior_prec1_type = "gamma"`.
#' @param prior_prec1_scale Numeric > 0, scale \eqn{A} of the Half-t prior for
#'   \eqn{\sqrt{W_1}}. Default `2`, on the scale of the link -- a fixed value
#'   is enough here, unlike the Gaussian family, because the link scale carries
#'   no arbitrary units. Ignored when `prior_prec1_type = "gamma"`.
#' @param prior_prec1_df Numeric > 0, Half-t df \eqn{\nu_1} for \eqn{\sqrt{W_1}}.
#'   Default `1` (Half-Cauchy).
#' @param prior_prec2_type Character, prior on the trend innovation precision
#'   \eqn{1/W_2} (`"halfcauchy"` default, or `"halft"` / `"gamma"`).
#' @param prior_prec2_shape Numeric > 0, shape parameter of the Gamma prior for
#'   the trend innovation precision \eqn{1/W_2}. Default is 0.01 (vague prior).
#'   Used only when `prior_prec2_type = "gamma"`.
#' @param prior_prec2_rate Numeric > 0, rate parameter of the Gamma prior for
#'   \eqn{1/W_2}. Default is 0.01 (vague prior). Used only when
#'   `prior_prec2_type = "gamma"`.
#' @param prior_prec2_scale Numeric > 0, Half-t scale for \eqn{\sqrt{W_2}}.
#'   Default `2`. Ignored when `prior_prec2_type = "gamma"`.
#' @param prior_prec2_df Numeric > 0, Half-t df \eqn{\nu_2} for \eqn{\sqrt{W_2}}.
#'   Default `1` (Half-Cauchy).
#' @param lag_update Integer \eqn{\geq 1}, adaptation frequency (sliding window size) for
#'   computing acceptance proportions (logit link only). Adaptation occurs at MCMC iterations
#'   \eqn{m = k \cdot \text{lag\_update}} for \eqn{k = 1, 2, 3, \ldots}. Default is 50.
#'   Ignored when `link = "probit"`.
#' @param max_step_size Numeric > 0, maximum allowed change in log-scale proposal variance
#'   per adaptation step (logit link only). Prevents extreme adjustments. Default is 0.1.
#'   Ignored when `link = "probit"`.
#' @param base_adaptation_rate Numeric > 0, base rate controlling adaptation speed
#'   (logit link only). Higher values lead to faster but potentially less stable adaptation.
#'   Default is 1.0. Ignored when `link = "probit"`.
#' @param decay_exponent Numeric > 0, exponent controlling diminishing adaptation
#'   rate (logit link only). The adaptation rate decays as \eqn{m^{-\text{decay\_exponent}}}
#'   to satisfy diminishing adaptation conditions. Must be in (0.5, 1] for theoretical
#'   convergence guarantees. Default is 0.6. Ignored when `link = "probit"`.
#' @param target_acceptance Numeric in (0, 1), target acceptance rate for the
#'   Metropolis-Hastings algorithm (logit link only). The value 0.44 is theoretically
#'   optimal for univariate random-walk proposals. Default is 0.44. Ignored when
#'   `link = "probit"`.
#' @param min_deviation_threshold Numeric \eqn{\geq 0} or `NULL`, minimum absolute deviation
#'   from `target_acceptance` required to trigger adaptation (logit link only). If `NULL`
#'   (default), uses practical threshold of `1.0/lag_update`. Set to `0.0` for maximum
#'   sensitivity (adapt for any deviation). Larger values make adaptation more conservative.
#'   Ignored when `link = "probit"`.
#' @param return_log_sigma Logical, whether to return proposal scale diagnostics
#'   (log-scale proposal standard deviations) for logit link only. Useful for diagnosing
#'   adaptation behavior. Default is `FALSE`. Ignored when `link = "probit"`.
#' @param return_accept_prop Logical, whether to return acceptance proportion diagnostics
#'   over iterations for logit link only. Useful for diagnosing MCMC mixing. Default is
#'   `FALSE`. Ignored when `link = "probit"`.
#' @param verbose Logical, whether to display a progress bar during sampling. Default is `FALSE`.
#' @param bar_width Integer in \[10, 120\], width of the progress bar when `verbose = TRUE`.
#'   Default is `60`.
#' @param seed Optional integer used to set the random number generator seed for
#'   reproducibility. Default is `NULL` (no seed set).
#'   When `chains > 1` it acts as a master seed from which each chain's own
#'   seed is drawn.
#' @param chains Integer \eqn{\geq 1}, number of independent chains to run.
#'   Default is `1`, which returns a single fitted object exactly as before.
#'   With `chains > 1` the sampler is re-run once per chain -- each starting
#'   from its own values drawn from the priors -- and an object of class
#'   `"pdm_mcmc_list"` is returned, which \code{\link{mcmc_convergence}} turns
#'   into R-hat and effective sample sizes. The progress bar is disabled in this
#'   case, since several bars sharing one console interleave.
#' @param parallel Logical, whether to run the chains through
#'   `parallel::mclapply()`. Used only when `chains > 1`. The result does not
#'   depend on this setting: every chain receives an explicit seed, so a given
#'   `seed` reproduces the same output sequentially or in parallel. Default is
#'   `FALSE`.
#' @param init Optional named list of starting values, or `NULL` (the default)
#'   to draw every one of them from its prior, as the sampler has always done.
#'   The names this model accepts are `lambda_1`, `lambda_2`, `theta_01`,
#'   `theta_02`, `prec_theta1` and `prec_theta2`;
#'   any subset may be given, and rates and precisions must be positive. An
#'   unrecognised name is an error rather than being ignored, so a misspelling
#'   cannot pass for a starting value that quietly had no effect. With
#'   `chains > 1` pass a list of `chains` such lists, one per chain: a single
#'   starting point shared by every chain removes the dispersion the
#'   Gelman-Rubin statistic in \code{\link{mcmc_convergence}} is computed from,
#'   so it is refused.
#'
#' @return A list with components:
#' \describe{
#'   \item{\code{lambda_1}}{Numeric vector of length `n_draws` of posterior samples
#'     for the rate of component 1 (\eqn{\lambda_1}). Component 1 is defined as the
#'     component with the smaller rate due to the label switching constraint.}
#'   \item{\code{lambda_2}}{Numeric vector of length `n_draws` of posterior samples
#'     for the rate of component 2 (\eqn{\lambda_2}).}
#'   \item{\code{theta_1}}{Numeric matrix \eqn{[n_{draws} \times n]} of posterior
#'     samples for the latent level state \eqn{\theta_{t,1}}.}
#'   \item{\code{theta_2}}{Numeric matrix \eqn{[n_{draws} \times n]} of posterior
#'     samples for the latent trend state \eqn{\theta_{t,2}}.}
#'   \item{\code{theta_01}}{Numeric vector of length `n_draws` of posterior samples
#'     for the initial level state \eqn{\theta_{0,1}}.}
#'   \item{\code{theta_02}}{Numeric vector of length `n_draws` of posterior samples
#'     for the initial trend state \eqn{\theta_{0,2}}.}
#'   \item{\code{prec_theta1}}{Numeric vector of length `n_draws` of posterior samples
#'     for the level innovation precision \eqn{1/W_1}.}
#'   \item{\code{prec_theta2}}{Numeric vector of length `n_draws` of posterior samples
#'     for the trend innovation precision \eqn{1/W_2}.}
#'   \item{\code{alpha}}{Numeric matrix \eqn{[n_{draws} \times n]} of posterior
#'     samples for the mixture weights \eqn{\alpha_t}. Convergence of \eqn{\alpha_t} is already covered by the
#'     `theta_1[t=...]` rows of \code{\link{mcmc_convergence}} on a
#'     multi-chain fit, which need not be supplemented by a diagnostic
#'     computed on `alpha` itself; see the "Link families" section of
#'     \code{\link{mcmc_convergence.pdm_mcmc_list}} for why, and for why
#'     the same does not hold of the single-chain diagnostics.}
#'   \item{\code{z}}{Numeric matrix \eqn{[n_{draws} \times n]} of posterior samples
#'     for the latent component indicators \eqn{z_t}.}
#'   \item{\code{log_sigma}}{(Logit link only, optional) Numeric matrix of proposal
#'     log standard deviations used in the adaptive Metropolis-Hastings steps.}
#'   \item{\code{accept_prop}}{(Logit link only, optional) Numeric matrix of proposal
#'     acceptance proportions accumulated during adaptive tuning.}
#' }
#'
#' When `chains > 1` the return value is instead an object of class
#' `c("pdm_mcmc_list", "list")` holding one such fit per chain; see
#' \code{\link{print.pdm_mcmc_list}}.
#'
#'
#' The object also carries `pdm_version`, `seed` and `priors` as attributes:
#' the version that produced it, the seed as supplied, and every prior after
#' resolution -- including the ones derived from the data, which never appear in
#' the call. Splicing `attr(fit, "priors")` back into a fresh call reproduces the
#' fit on any later version, whatever the defaults have become.
#' @examples
#' ## Description
#' # This example demonstrates how to:
#' # 1. Simulate counts from a Poisson mixture with drifting mixture weights
#' # 2. Use `mcmc_poisson_mixture_localtrend` to estimate rates and latent states
#' # 3. Compare logit and probit link functions
#' # 4. Perform a detailed posterior analysis with visualizations
#' # 5. Set a seed for reproducibility
#' # 6. Use a progress bar for monitoring MCMC execution
#'
#' ## Simulation of data
#' n <- 200  # Number of observations to simulate
#'
#' # Use a fixed seed for data simulation
#' set.seed(123)
#'
#' # Generate true mixture weights with a slow drift from low to high
#' grid_vals <- seq_len(n) / n
#' alpha_true <- plogis(-3 + 6 * grid_vals)
#'
#' # Generate latent component indicators
#' z_true <- rbinom(n, size = 1, prob = alpha_true)
#'
#' # True component rates for simulation
#' lambda_1_true <- 2
#' lambda_2_true <- 10
#'
#' # Generate observations from the Poisson mixture
#' lambda_y <- (1 - z_true) * lambda_1_true + z_true * lambda_2_true
#' y <- rpois(n, lambda = lambda_y)
#'
#' ## Running the Gibbs sampler with logit link
#' # NOTE: these MCMC controls are sized to keep the example quick, and are
#' # not guaranteed to be enough for convergence -- how much burn-in and
#' # thinning a fit needs depends on the data and on the family. For real
#' # work, fit with chains > 1 and read mcmc_convergence() before trusting
#' # any summary.
#' out_logit <- mcmc_poisson_mixture_localtrend(
#'   y,
#'   link                    = "logit",
#'   burnin                  = 1000,
#'   thinning                = 10,
#'   n_draws                 = 500,
#'   prior_lambda01_shape    = 2,
#'   prior_lambda01_rate     = NULL,
#'   prior_lambda02_shape    = 2,
#'   prior_lambda02_rate     = NULL,
#'   prior_theta01_mean      = 0,
#'   prior_theta01_prec      = 1,
#'   prior_theta02_mean      = 0,
#'   prior_theta02_prec      = 1,
#'   prior_prec1_shape       = 100,
#'   prior_prec1_rate        = 1,
#'   prior_prec2_shape       = 100,
#'   prior_prec2_rate        = 1,
#'   lag_update              = 50,
#'   max_step_size           = 0.1,
#'   base_adaptation_rate    = 1.0,
#'   decay_exponent          = 0.6,
#'   target_acceptance       = 0.44,
#'   min_deviation_threshold = NULL,
#'   return_log_sigma        = FALSE,
#'   return_accept_prop      = FALSE,
#'   verbose                 = TRUE,
#'   bar_width               = 60,
#'   seed                    = 456
#' )
#'
#' ## Running the Gibbs sampler with probit link
#' out_probit <- mcmc_poisson_mixture_localtrend(
#'   y,
#'   link               = "probit",
#'   burnin             = 1000,
#'   thinning           = 10,
#'   n_draws            = 500,
#'   prior_theta01_mean = 0,
#'   prior_theta01_prec = 1,
#'   prior_theta02_mean = 0,
#'   prior_theta02_prec = 1,
#'   prior_prec1_shape  = 100,
#'   prior_prec1_rate   = 1,
#'   prior_prec2_shape  = 100,
#'   prior_prec2_rate   = 1,
#'   verbose            = TRUE,
#'   bar_width          = 60,
#'   seed               = 789
#' )
#'
#' ## Alternative: weakly-informative Half-Cauchy priors on both innovations
#' # Both weight-innovation SDs live on the probit latent scale, so a scale of
#' # about 1 is weakly informative there. The component rates keep their
#' # conjugate Gamma priors, which have no Half-t option.
#' # This fit also pins starting values through `init`. Two things are worth
#' # noting. The component rates must be supplied in order -- the model
#' # identifies them only up to their labelling, and `lambda_1` above
#' # `lambda_2` is an error rather than a silent relabelling. And `theta_01` is
#' # the initial weight on the linear-predictor scale, not a probability: the
#' # weight the chain starts from is link(theta_01), so 0 means an even split.
#' out_hc <- mcmc_poisson_mixture_localtrend(
#'   y,
#'   link               = "probit",
#'   burnin             = 1000,
#'   thinning           = 10,
#'   n_draws            = 500,
#'   prior_theta01_mean = 0,
#'   prior_theta01_prec = 1,
#'   prior_theta02_mean = 0,
#'   prior_theta02_prec = 1,
#'   prior_prec1_type   = "halfcauchy",
#'   prior_prec1_scale  = 1,
#'   prior_prec2_type   = "halfcauchy",
#'   prior_prec2_scale  = 1,
#'   init               = list(lambda_1 = quantile(y, 0.25),
#'                             lambda_2 = quantile(y, 0.75),
#'                             theta_01 = 0),
#'   verbose            = FALSE,
#'   seed               = 789
#' )
#' # Where the chain actually started is recorded on the fit:
#' attr(out_hc, "init")
#'
#' ## Posterior analysis and visualization
#' # Use the plot method for comprehensive diagnostics
#' \donttest{
#'   # --- Logit-link fit (`out_logit`) ---
#'   # Complete dashboard with all diagnostics
#'   plot(out_logit, type = "all")
#'
#'   # Individual diagnostic types
#'   plot(out_logit, type = "mcmc", which = 1:2)   # Component rates
#'   plot(out_logit, type = "mcmc", which = 3:6)   # theta_0 and W^-1
#'   plot(out_logit, type = "params")              # Mixture components
#'   plot(out_logit, type = "states")              # Dynamic level and trend
#'   plot(out_logit, type = "alpha")               # Mixture weights
#'
#'   # Plot with true values for comparison
#'   plot(out_logit, type = "params",
#'        true_values = list(lambda_1 = lambda_1_true, lambda_2 = lambda_2_true))
#'   plot(out_logit, type = "alpha",
#'        true_values = list(alpha = alpha_true, z = z_true))
#'
#'   # --- Probit-link fit (`out_probit`) ---
#'   # The same diagnostics for the probit-link fit, for comparison.
#'   plot(out_probit, type = "all")
#'
#'   plot(out_probit, type = "params")             # Mixture components
#'   plot(out_probit, type = "states")             # Dynamic level and trend
#'   plot(out_probit, type = "alpha",
#'        true_values = list(alpha = alpha_true, z = z_true))
#'
#'   # --- Half-Cauchy-prior fit (`out_hc`) ---
#'   # The same diagnostics for the weakly-informative Half-Cauchy fit.
#'   plot(out_hc, type = "all")
#'
#'   plot(out_hc, type = "alpha",
#'        true_values = list(alpha = alpha_true, z = z_true))
#' }
#'
#' @references
#' Albert, J. H., & Chib, S. (1993). Bayesian analysis of binary and polychotomous
#' response data. \emph{Journal of the American Statistical Association}, 88(422),
#' 669-679. \doi{10.1080/01621459.1993.10476321}
#'
#' Gelman, A. (2006). Prior distributions for variance parameters in
#' hierarchical models. \emph{Bayesian Analysis}, 1(3), 515-534.
#'
#' Montoril, M. H., Correia, L. T., & Migon, H. S. (2021). Bayesian estimation of
#' dynamic weights in Gaussian mixture models. arXiv:2104.03395.
#'
#' Roberts, G. O., & Rosenthal, J. S. (2001). Optimal scaling for various
#' Metropolis-Hastings algorithms. \emph{Statistical Science}, 16(4), 351-367.
#'
#' Roberts, G. O., & Rosenthal, J. S. (2007). Coupling and ergodicity of adaptive MCMC.
#' \emph{Journal of Applied Probability}, 44(2), 458-475.
#'
#' Roberts, G. O., & Rosenthal, J. S. (2009). Examples of adaptive MCMC.
#' \emph{Journal of Computational and Graphical Statistics}, 18(2), 349-367.
#' \doi{10.1198/jcgs.2009.06134}
#'
#' Wand, M. P., Ormerod, J. T., Padoan, S. A., & Fruhwirth, R. (2011). Mean field
#' variational Bayes for elaborate distributions. \emph{Bayesian Analysis},
#' 6(4), 847-900.
#'
#' @seealso
#'   \code{\link{plot.poisson_mixture_localtrend}},
#'   \code{\link{print.poisson_mixture_localtrend}},
#'   \code{\link{summary.poisson_mixture_localtrend}} for methods on the fitted object;
#'   \code{\link{mcmc_poisson_mixture_locallevel}} and
#'   \code{\link{mcmc_poisson_mixture_localacceleration}} for the other dynamic orders.
#'
#' @export
mcmc_poisson_mixture_localtrend <- function(y,
                                            link = c("logit", "probit"),
                                            burnin,
                                            thinning,
                                            n_draws,
                                            prior_lambda01_shape = 2,
                                            prior_lambda01_rate = NULL,
                                            prior_lambda02_shape = 2,
                                            prior_lambda02_rate = NULL,
                                            prior_theta01_mean = 0,
                                            prior_theta01_prec = 0.01,
                                            prior_theta02_mean = 0,
                                            prior_theta02_prec = 0.01,
                                            prior_prec1_shape = 0.01,
                                            prior_prec1_rate = 0.01,
                                            prior_prec2_shape = 0.01,
                                            prior_prec2_rate = 0.01,
                                            lag_update = 50,
                                            max_step_size = 0.1,
                                            base_adaptation_rate = 1.0,
                                            decay_exponent = 0.6,
                                            target_acceptance = 0.44,
                                            min_deviation_threshold = NULL,
                                            return_log_sigma = FALSE,
                                            return_accept_prop = FALSE,
                                            verbose = FALSE,
                                            bar_width = 60,
                                            seed = NULL,
                                            prior_prec1_type = c("halfcauchy", "gamma", "halft"),
                                            prior_prec1_scale = 2,
                                            prior_prec1_df = 1,
                                            prior_prec2_type = c("halfcauchy", "gamma", "halft"),
                                            prior_prec2_scale = 2,
                                            prior_prec2_df = 1,
                                            chains = 1,
                                            parallel = FALSE,
                                            init = NULL) {

  # --- Multi-chain dispatch ---
  # Re-issues this same call once per chain, each with its own seed and its own
  # entry of `init`, and returns the collection. Kept at the very top so
  # `match.call()` captures the call exactly as the user wrote it.
  validate_chains_args(chains, parallel)
  if (chains > 1) {
    return(run_chains(match.call(), parent.frame(),
                      as.integer(chains), seed, parallel, init))
  }

  # Record whether each df was explicitly set (used to flag a contradictory
  # `type = "halfcauchy"` + `df != 1`). `missing()` must be read before use.
  prec1_df_user_set <- !missing(prior_prec1_df)
  prec2_df_user_set <- !missing(prior_prec2_df)
  # Whether the type was named decides how a bare shape/rate pair is read
  # (see infer_gamma_from_hyperparams() in R/innovation_priors.R).
  prec1_type_user_set <- !missing(prior_prec1_type)
  prec2_type_user_set <- !missing(prior_prec2_type)

  prior_prec1_type <- match.arg(prior_prec1_type)
  prior_prec2_type <- match.arg(prior_prec2_type)

  prior_prec1_type <- infer_gamma_from_hyperparams(
    prior_prec1_type, prec1_type_user_set,
    !missing(prior_prec1_shape) || !missing(prior_prec1_rate)
  )
  prior_prec2_type <- infer_gamma_from_hyperparams(
    prior_prec2_type, prec2_type_user_set,
    !missing(prior_prec2_shape) || !missing(prior_prec2_rate)
  )
  # --- Input Validation ---

  # Validate y
  if (!is.numeric(y)) {
    stop("`y` must be a numeric vector")
  }
  if (!all(is.finite(y))) {
    stop("`y` must contain only finite numeric values (no NA, NaN, Inf)")
  }
  # Validate Poisson constraints (non-negative integers)
  if (any(y < 0 | y != floor(y))) {
    stop("`y` values must be non-negative integers")
  }
  if (length(y) < 3) {
    stop("`y` must have at least 3 observations")
  }

  # Validate and match link argument
  link <- match.arg(link)

  # Validate MCMC control parameters
  if (!is.numeric(burnin) || length(burnin) != 1 || burnin < 0 ||
      burnin != floor(burnin)) {
    stop("`burnin` must be a single non-negative integer")
  }
  if (!is.numeric(thinning) || length(thinning) != 1 || thinning < 1 ||
      thinning != floor(thinning)) {
    stop("`thinning` must be a single positive integer")
  }
  if (!is.numeric(n_draws) || length(n_draws) != 1 || n_draws < 1 ||
      n_draws != floor(n_draws)) {
    stop("`n_draws` must be a single positive integer")
  }

  # Validate the component shapes before they are used to derive the rates.
  if (!is.numeric(prior_lambda01_shape) || length(prior_lambda01_shape) != 1 ||
      prior_lambda01_shape <= 0) {
    stop("`prior_lambda01_shape` must be a single positive numeric value")
  }
  if (!is.numeric(prior_lambda02_shape) || length(prior_lambda02_shape) != 1 ||
      prior_lambda02_shape <= 0) {
    stop("`prior_lambda02_shape` must be a single positive numeric value")
  }

  # Data-scaled defaults for the component rates: centre component 1 on the
  # lower quartile of the counts and component 2 on the upper one (see
  # poisson_component_rate() in R/mixture_priors.R).
  lambda_targets <- poisson_component_targets(y)
  if (is.null(prior_lambda01_rate)) {
    prior_lambda01_rate <- poisson_component_rate(lambda_targets[1L],
                                                  prior_lambda01_shape)
  }
  if (is.null(prior_lambda02_rate)) {
    prior_lambda02_rate <- poisson_component_rate(lambda_targets[2L],
                                                  prior_lambda02_shape)
  }

  if (!is.numeric(prior_lambda01_rate) || length(prior_lambda01_rate) != 1 ||
      prior_lambda01_rate <= 0) {
    stop("`prior_lambda01_rate` must be a single positive numeric value")
  }
  if (!is.numeric(prior_lambda02_rate) || length(prior_lambda02_rate) != 1 ||
      prior_lambda02_rate <= 0) {
    stop("`prior_lambda02_rate` must be a single positive numeric value")
  }

  # The components are identified by the constraint lambda_1 < lambda_2; warn if
  # the priors ask for the reverse (see R/mixture_priors.R). Checked after the
  # NULL defaults are resolved, so the data-driven quantiles are covered too.
  check_mixture_lambda_priors(prior_lambda01_shape, prior_lambda01_rate,
                              prior_lambda02_shape, prior_lambda02_rate)

  # Validate dynamic state prior parameters
  if (!is.numeric(prior_theta01_mean) || length(prior_theta01_mean) != 1) {
    stop("`prior_theta01_mean` must be a single numeric value")
  }
  if (!is.numeric(prior_theta01_prec) || length(prior_theta01_prec) != 1 ||
      prior_theta01_prec <= 0) {
    stop("`prior_theta01_prec` must be a single positive numeric value")
  }
  if (!is.numeric(prior_theta02_mean) || length(prior_theta02_mean) != 1) {
    stop("`prior_theta02_mean` must be a single numeric value")
  }
  if (!is.numeric(prior_theta02_prec) || length(prior_theta02_prec) != 1 ||
      prior_theta02_prec <= 0) {
    stop("`prior_theta02_prec` must be a single positive numeric value")
  }
  # Resolve the state innovation precision priors (Gamma or Half-t).
  prec1_prior <- resolve_prec_prior(
    prior_prec1_type, prior_prec1_shape, prior_prec1_rate,
    prior_prec1_scale, prior_prec1_df, prec1_df_user_set, "prior_prec1"
  )
  prec2_prior <- resolve_prec_prior(
    prior_prec2_type, prior_prec2_shape, prior_prec2_rate,
    prior_prec2_scale, prior_prec2_df, prec2_df_user_set, "prior_prec2"
  )

  # Validate MCMC adaptation parameters
  if (!is.numeric(lag_update) || length(lag_update) != 1 || lag_update < 1 ||
      lag_update != floor(lag_update)) {
    stop("`lag_update` must be a single positive integer")
  }
  if (!is.numeric(max_step_size) || length(max_step_size) != 1 ||
      max_step_size <= 0) {
    stop("`max_step_size` must be a single positive numeric value")
  }
  if (!is.numeric(base_adaptation_rate) || length(base_adaptation_rate) != 1 ||
      base_adaptation_rate <= 0) {
    stop("`base_adaptation_rate` must be a single positive numeric value")
  }
  if (!is.numeric(decay_exponent) || length(decay_exponent) != 1 ||
      decay_exponent <= 0) {
    stop("`decay_exponent` must be a single positive numeric value")
  }
  if (!is.numeric(target_acceptance) || length(target_acceptance) != 1 ||
      target_acceptance <= 0 || target_acceptance >= 1) {
    stop("`target_acceptance` must be a single numeric value in (0, 1)")
  }

  # Validate min_deviation_threshold parameter
  if (is.null(min_deviation_threshold)) {
    # Compute practical default threshold
    min_deviation_threshold <- 1.0 / lag_update
  } else {
    if (!is.numeric(min_deviation_threshold) || length(min_deviation_threshold) != 1 ||
        min_deviation_threshold < 0) {
      stop("`min_deviation_threshold` must be a single non-negative numeric value or NULL")
    }
  }

  # Validate logical parameters
  if (!is.logical(return_log_sigma) || length(return_log_sigma) != 1) {
    stop("`return_log_sigma` must be a single logical value")
  }
  if (!is.logical(return_accept_prop) || length(return_accept_prop) != 1) {
    stop("`return_accept_prop` must be a single logical value")
  }

  # Validate verbose parameter
  if (!is.logical(verbose) || length(verbose) != 1) {
    stop("`verbose` must be a single logical value")
  }

  # Validate bar_width parameter
  if (!is.numeric(bar_width) || length(bar_width) != 1 ||
      bar_width < 10 || bar_width > 120 || bar_width != floor(bar_width)) {
    stop("`bar_width` must be a single integer in [10, 120]")
  }

  # Validate and set seed if provided
  if (!is.null(seed)) {
    if (!is.numeric(seed) || length(seed) != 1 || seed != floor(seed)) {
      stop("`seed` must be a single integer value")
    }
    set.seed(seed)
  }

  # Issue warning if diagnostic flags are set for probit link (they will be ignored)
  if (link == "probit") {
    if (return_log_sigma) {
      warning("Argument `return_log_sigma` is ignored when link = 'probit'")
    }
    if (return_accept_prop) {
      warning("Argument `return_accept_prop` is ignored when link = 'probit'")
    }
  }

  # --- End Input Validation ---

  # --- Starting values ---
  # Validated and resolved in R (see R/init_values.R). The component rates go
  # through `precs` rather than `states`: what that branch does is draw from a
  # Gamma, which is exactly the rates' prior. `ordered_components` then carries
  # the lambda_1 <= lambda_2 relabelling on that same pair.
  init_state <- resolve_init(
    init,
    states = list(theta_01 = list(mean = prior_theta01_mean,
                                  prec = prior_theta01_prec),
                  theta_02 = list(mean = prior_theta02_mean,
                                  prec = prior_theta02_prec)),
    precs  = list(lambda_1    = list(code  = 0L,
                                     shape = prior_lambda01_shape,
                                     rate  = prior_lambda01_rate),
                  lambda_2    = list(code  = 0L,
                                     shape = prior_lambda02_shape,
                                     rate  = prior_lambda02_rate),
                  prec_theta1 = prec1_prior,
                  prec_theta2 = prec2_prior),
    ordered_components = list(precs = c("lambda_1", "lambda_2"))
  )

  result <- .Call(
    "_pdm_C_MCMC_poisson_mixture_localtrend",
    as.numeric(y),
    as.character(link),
    as.integer(burnin),
    as.integer(thinning),
    as.integer(n_draws),
    as.numeric(prior_lambda01_shape),
    as.numeric(prior_lambda01_rate),
    as.numeric(prior_lambda02_shape),
    as.numeric(prior_lambda02_rate),
    as.numeric(prior_theta01_mean),
    as.numeric(prior_theta01_prec),
    as.numeric(prior_theta02_mean),
    as.numeric(prior_theta02_prec),
    as.integer(prec1_prior$code),
    as.numeric(prec1_prior$shape),
    as.numeric(prec1_prior$rate),
    as.numeric(prec1_prior$scale),
    as.numeric(prec1_prior$df),
    as.integer(prec2_prior$code),
    as.numeric(prec2_prior$shape),
    as.numeric(prec2_prior$rate),
    as.numeric(prec2_prior$scale),
    as.numeric(prec2_prior$df),
    as.integer(lag_update),
    as.numeric(max_step_size),
    as.numeric(base_adaptation_rate),
    as.numeric(decay_exponent),
    as.numeric(target_acceptance),
    as.numeric(min_deviation_threshold),
    as.logical(return_log_sigma),
    as.logical(return_accept_prop),
    as.numeric(init_state$values),
    as.logical(verbose),
    as.integer(bar_width)
  )

  result <- new_poisson_mixture_localtrend(
    result = result,
    link = link,
    n_obs = length(y),
    n_draws = n_draws,
    burnin = burnin,
    thinning = thinning,
    y = y
  )

  result <- validate_poisson_mixture_localtrend(result)

  # Record the priors used on the component rates lambda_k and the state
  # innovation precisions (auxiliary Half-t variables are nuisance parameters
  # and are intentionally not returned).
  attr(result, "prior_lambda_1")    <- list(shape = prior_lambda01_shape,
                                            rate  = prior_lambda01_rate)
  attr(result, "prior_lambda_2")    <- list(shape = prior_lambda02_shape,
                                            rate  = prior_lambda02_rate)
  attr(result, "prior_prec_theta1") <- prec1_prior[c("type", "shape", "rate", "scale", "df")]
  attr(result, "prior_prec_theta2") <- prec2_prior[c("type", "shape", "rate", "scale", "df")]

  # Record the RWMH target acceptance rate so plot(type = "acceptance") can draw
  # the correct reference line (logit link only; unused under the probit sampler).
  attr(result, "target_acceptance") <- as.numeric(target_acceptance)

  # Record what produced this fit: version, seed and every resolved prior
  # (see R/provenance.R). Must come after all defaults are filled in.
  result <- record_provenance(result, seed, init_state$init)

  return(result)
}
