#' @title Gibbs Sampler for a Local-Level Poisson Dynamic Model
#'
#' @description Runs a Gibbs sampler for the local-level Poisson dynamic model
#'   with log link.
#'
#' @details The model is defined as:
#' \deqn{
#' \begin{aligned}
#' y_t &\sim \text{Poisson}(\alpha_t), \\
#' \alpha_t &= \exp(\theta_{t,1}), \\
#' \theta_{t,1} &= \theta_{t-1,1} + u_{t,1}, & u_{t,1} \sim N(0, W_1),
#' \end{aligned}
#' }
#' where \eqn{t = 1, 2, \ldots, n} and \eqn{n} is the number of observations.
#'
#' The log link function is defined as
#' \eqn{\exp(x)}, ensuring that \eqn{\alpha_t > 0} as required for Poisson rates.
#'
#' \strong{Prior Distributions: }
#'
#' The following conjugate and semi-conjugate prior distributions are employed:
#'
#' \emph{Initial States:}
#' \deqn{
#' \begin{aligned}
#' \theta_{0,1} &\sim N(\mu_{01}, \tau_{01}^{-1}).
#' \end{aligned}
#' }
#'
#' \emph{Precision Parameters:}
#'
#' The innovation precision \eqn{W_1^{-1}} may take one of two priors, chosen
#' through `prior_prec1_type`:
#'
#' \emph{(a) Gamma prior on the precision} (default, semi-conjugate):
#' \deqn{W_1^{-1} \sim \text{Gamma}(\nu_1, \eta_1).}
#'
#' \emph{(b) Half-t prior on the standard deviation} (Gelman, 2006):
#' \deqn{\sqrt{W_1} \sim \text{Half-}t(\nu_1, A_1),}
#' where \eqn{A_1 > 0} is a scale hyperparameter and \eqn{\nu_1 > 0} the degrees
#' of freedom. Setting \eqn{\nu_1 = 1} gives the Half-Cauchy prior (also
#' selectable directly with `type = "halfcauchy"`); larger \eqn{\nu_1} approaches
#' a Half-Normal. The Half-t prior is represented by the inverse-gamma scale
#' mixture of Wand et al. (2011), which keeps the innovation-precision Gibbs
#' update closed-form via a single auxiliary variable. This Poisson model has no
#' Gaussian observation variance, so there is no observation precision \eqn{1/V}.
#'
#' The Gamma shape/rate arguments are required only when
#' `prior_prec1_type = "gamma"`; the Half-t scale/df arguments only when it is
#' `"halft"` or `"halfcauchy"`. The prior choice is resolved once, before the
#' sampler runs, and never re-evaluated inside the MCMC loop.
#'
#' The prior hyperparameters correspond to function arguments as follows:
#' \tabular{cc}{
#'   \strong{Hyperparameter} \tab \strong{Function Argument} \cr
#'   \eqn{\mu_{01}} \tab `prior_theta01_mean` \cr
#'   \eqn{\tau_{01}} \tab `prior_theta01_prec` \cr
#'   \eqn{\nu_1} (Gamma shape) \tab `prior_prec1_shape` \cr
#'   \eqn{\eta_1} (Gamma rate) \tab `prior_prec1_rate` \cr
#'   \eqn{A_1} (Half-t scale) \tab `prior_prec1_scale` \cr
#'   \eqn{\nu_1} (Half-t df) \tab `prior_prec1_df`
#' }
#'
#' \strong{Adaptive Metropolis-Hastings Algorithm:}
#'
#' Due to the non-linear observation model with log link, the algorithm
#' employs component-wise Metropolis-Hastings for sampling the latent states
#' \eqn{\theta_{t,1}}, with adaptive proposal tuning based on acceptance proportions.
#' The innovation precision is sampled from its conjugate Gamma posterior.
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
#' \strong{Parameter Interactions:}
#'
#' The adaptive tuning parameters interact as follows:
#' \itemize{
#'   \item \strong{lag_update}:  Controls both the sliding window size for computing acceptance
#'     proportions AND the frequency of adaptation. Adaptation occurs at MCMC iterations
#'     \eqn{m = k \cdot \text{lag\_update}} for \eqn{k = 1, 2, 3, \ldots}. Larger values
#'     provide more stable estimates but slower adaptation. Common choices:  50-200 iterations.
#'   \item \strong{target_acceptance}:  Optimal acceptance rate for the Metropolis-Hastings
#'     algorithm. The value 0.44 is theoretically optimal for univariate random-walk proposals
#'     (Roberts and Rosenthal, 2001). Each time point \eqn{t} has its own acceptance rate
#'     \eqn{\hat{p}_t}.
#'   \item \strong{min_deviation_threshold}:  Minimum deviation \eqn{|\hat{p}_t - p^*|}
#'     required to trigger adaptation for time point \eqn{t}. The default `NULL` uses the
#'     practical threshold \eqn{1/\text{lag\_update}}, corresponding to one additional
#'     acceptance/rejection in the sliding window.
#'   \item \strong{max_step_size}: Maximum allowed change in log-scale proposal variance
#'     per adaptation step. Prevents extreme adjustments. Common choices: 0.01-0.1.
#'   \item \strong{base_adaptation_rate}: Controls the overall speed of adaptation before
#'     decay is applied. Higher values lead to faster but potentially less stable adaptation.
#'     Common choices: 0.1-10.0.
#'   \item \strong{decay_exponent}: Controls how quickly the adaptation step size diminishes
#'     over MCMC iterations. As the algorithm runs, \eqn{\gamma_m} decreases according to
#'     \eqn{m^{-\xi}}. Must be in (0.5, 1] for theoretical convergence guarantees.
#'     Common choices: 0.5-0.8.
#' }
#'
#' \strong{Example of Adaptation Schedule:}
#'
#' With `lag_update = 50`, `base_adaptation_rate = 1.0`, `decay_exponent = 0.6`,
#' and `max_step_size = 0.1`:
#' \itemize{
#'   \item At MCMC iteration 50:  \eqn{\gamma_{50} = \min(0.1, 1.0/50^{0.6}) \approx 0.0875}.
#'   \item At MCMC iteration 100: \eqn{\gamma_{100} = \min(0.1, 1.0/100^{0.6}) \approx 0.0631}.
#'   \item At MCMC iteration 1000: \eqn{\gamma_{1000} = \min(0.1, 1.0/1000^{0.6}) \approx 0.0158}.
#' }
#'
#' This ensures that adaptation becomes increasingly conservative as the chain progresses,
#' satisfying theoretical requirements for ergodicity.
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
#' Burn-in and thinning are applied so that exactly `n_chain` posterior samples
#' are returned.
#'
#' @param y Numeric vector of observed Poisson counts (length \eqn{n}). Each
#'   element must be a non-negative integer.
#' @param burnin Integer \eqn{\geq 0}, number of burn-in iterations.
#' @param thinning Integer \eqn{\geq 1}, thinning interval.
#' @param n_chain Integer \eqn{\geq 1}, number of posterior samples to retain.
#' @param prior_theta01_mean Numeric, prior mean for the initial state
#'   \eqn{\theta_{0,1}}.
#' @param prior_theta01_prec Numeric > 0, prior precision (inverse variance)
#'   for \eqn{\theta_{0,1}}.
#' @param prior_prec1_type Character, prior on the level innovation
#'   precision \eqn{1/W_1}: `"halfcauchy"` (default) or `"halft"` for a Half-t /
#'   Half-Cauchy prior on \eqn{\sqrt{W_1}} (Gelman, 2006), or `"gamma"` for a
#'   Gamma prior on the precision. The default changed in 0.5-0; a call that
#'   supplies `prior_prec1_shape`/`prior_prec1_rate` without naming a type is
#'   still read as `"gamma"`, so existing code keeps working.
#' @param prior_prec1_shape Numeric > 0, shape parameter of the Gamma prior
#'   for \eqn{1/W_1}. Required (and used) only when `prior_prec1_type = "gamma"`.
#' @param prior_prec1_rate Numeric > 0, rate parameter of the Gamma prior
#'   for \eqn{1/W_1}. Required (and used) only when `prior_prec1_type = "gamma"`.
#' @param prior_prec1_scale Numeric > 0, scale \eqn{A} of the Half-t prior for
#'   \eqn{\sqrt{W_1}}. Default `2`, on the scale of the link -- a fixed value
#'   is enough here, unlike the Gaussian family, because the link scale carries
#'   no arbitrary units. Ignored when `prior_prec1_type = "gamma"`.
#' @param prior_prec1_df Numeric > 0, degrees of freedom \eqn{\nu_1} of the Half-t
#'   prior for \eqn{\sqrt{W_1}}. Default `1` (Half-Cauchy). Must equal `1` when
#'   `prior_prec1_type = "halfcauchy"`.
#' @param lag_update Integer \eqn{\geq 1}, adaptation frequency (sliding window size) for
#'   computing acceptance proportions. Default is 50.
#' @param max_step_size Numeric > 0, maximum allowed change in log-scale proposal variance
#'   per adaptation step. Default is 0.1.
#' @param base_adaptation_rate Numeric > 0, base rate controlling adaptation speed.
#'   Default is 1.0.
#' @param decay_exponent Numeric in (0.5, 1], exponent controlling diminishing adaptation
#'   rate. Default is 0.6.
#' @param target_acceptance Numeric in (0,1), target acceptance proportion for
#'   Metropolis-Hastings proposals. Default is 0.44 (theoretically optimal).
#' @param min_deviation_threshold Numeric \eqn{\geq 0} or `NULL`, minimum absolute deviation
#'   from `target_acceptance` required to trigger adaptation. If `NULL` (default),
#'   uses practical threshold of `1.0/lag_update`. Set to `0.0` for maximum sensitivity
#'   (adapt for any deviation). Larger values make adaptation more conservative.
#' @param return_log_sigma Logical, whether to return proposal scale diagnostics
#'   (log-scale proposal standard deviations). Default is `FALSE`.
#' @param return_accept_prop Logical, whether to return acceptance proportion diagnostics
#'   over iterations. Default is `FALSE`.
#' @param verbose Logical, whether to display a progress bar during sampling. Default is `TRUE`.
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
#'
#' @return An object of class `c("poisson_locallevel", "pdm_mcmc", "list")`
#'   with components:
#' \describe{
#'   \item{\code{theta_1}}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior samples for \eqn{\theta_{t,1}}.}
#'   \item{\code{theta_01}}{Numeric vector of length `n_chain` of posterior samples for \eqn{\theta_{0,1}}.}
#'   \item{\code{prec_theta1}}{Numeric vector of length `n_chain` of posterior samples for \eqn{1/W_1}.}
#'   \item{\code{alpha}}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior samples for \eqn{\alpha_t} (rates).}
#'   \item{\code{log_sigma}}{(Optional) Numeric matrix \eqn{[n_{chain} \times n]} of proposal scale diagnostics
#'     (if `return_log_sigma = TRUE`).}
#'   \item{\code{accept_prop}}{(Optional) Numeric matrix \eqn{[n_{chain} \times n]} of acceptance proportion
#'     diagnostics (if `return_accept_prop = TRUE`).}
#' }
#'
#' When `chains > 1` the return value is instead an object of class
#' `c("pdm_mcmc_list", "list")` holding one such fit per chain; see
#' \code{\link{print.pdm_mcmc_list}}.
#'
#' @references
#' Gelman, A. (2006). Prior distributions for variance parameters in
#' hierarchical models. \emph{Bayesian Analysis}, 1(3), 515-534.
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
#' @examples
#' ## Description
#' # This example demonstrates how to:
#' # 1. Simulate data from a local-level Poisson dynamic model
#' # 2. Use `mcmc_poisson_locallevel` to estimate parameters and latent states
#' # 3. Perform a detailed posterior analysis with visualizations
#' # 4. Set a seed for reproducibility
#'
#' ## Simulation of data
#' set.seed(123)
#' n <- 200        # Number of observations to simulate
#'
#' # Generate true rates (ensuring positive values)
#' alpha_true <- exp(sin(2 * pi * seq_len(n) / n) + 1)
#'
#' # Generate Poisson observations
#' y <- rpois(n, lambda = alpha_true)
#'
#' ## Running the Gibbs sampler
#' # Run the Gibbs sampler with specified priors and a seed
#' out <- mcmc_poisson_locallevel(
#'   y,
#'   burnin                  = 1000,
#'   thinning                = 20,
#'   n_chain                 = 500,
#'   prior_theta01_mean      = 0,
#'   prior_theta01_prec      = 1,
#'   prior_prec1_shape       = 100,
#'   prior_prec1_rate        = 1,
#'   lag_update              = 50,
#'   max_step_size           = 0.1,
#'   base_adaptation_rate    = 1,
#'   decay_exponent          = 0.6,
#'   target_acceptance       = 0.44,
#'   min_deviation_threshold = NULL,  # Uses practical default:  1.0/50 = 0.02
#'   return_log_sigma        = FALSE,
#'   return_accept_prop      = TRUE,
#'   verbose                 = TRUE,  # Enable progress bar
#'   bar_width               = 60,    # Progress bar width
#'   seed                    = 456
#' )
#'
#' ## Alternative: weakly-informative Half-Cauchy prior (Gelman, 2006)
#' # Replace the Gamma prior on the innovation precision with a Half-Cauchy on
#' # the innovation SD sqrt(W[1]); the Gamma shape/rate are then unused.
#' out_hc <- mcmc_poisson_locallevel(
#'   y,
#'   burnin             = 1000,
#'   thinning           = 20,
#'   n_chain            = 500,
#'   prior_theta01_mean = 0,
#'   prior_theta01_prec = 1,
#'   prior_prec1_type   = "halfcauchy",  # Half-Cauchy on sqrt(W[1])
#'   prior_prec1_scale  = 1,             # scale A_1 > 0
#'   verbose            = FALSE,
#'   seed               = 456
#' )
#'
#' ## Posterior analysis and visualization
#' # Use the plot method for comprehensive diagnostics
#' \donttest{
#'   # Complete dashboard with all diagnostics
#'   plot(out, type = "all")
#'
#'   # Individual diagnostic types
#'   plot(out, type = "mcmc", which = 1)      # Parameter diagnostics
#'   plot(out, type = "states")               # Dynamic states
#'   plot(out, type = "alpha")                # Poisson rates
#'   plot(out, type = "acceptance")           # Acceptance rates
#'
#'   # Plot with true alpha for comparison
#'   plot(out, type = "alpha", true_values = list(alpha = alpha_true))
#' }
#'
#' @seealso
#'   \code{\link{plot.poisson_locallevel}},
#'   \code{\link{print.poisson_locallevel}},
#'   \code{\link{summary.poisson_locallevel}} for methods on the fitted object;
#'   \code{\link{mcmc_poisson_localtrend}} and
#'   \code{\link{mcmc_poisson_localacceleration}} for the other dynamic orders.
#'
#' @export
mcmc_poisson_locallevel <- function(y,
                                    burnin,
                                    thinning,
                                    n_chain,
                                    prior_theta01_mean,
                                    prior_theta01_prec,
                                    prior_prec1_shape = NULL,
                                    prior_prec1_rate = NULL,
                                    lag_update = 50,
                                    max_step_size = 0.1,
                                    base_adaptation_rate = 1.0,
                                    decay_exponent = 0.6,
                                    target_acceptance = 0.44,
                                    min_deviation_threshold = NULL,
                                    return_log_sigma = FALSE,
                                    return_accept_prop = FALSE,
                                    verbose = TRUE,
                                    bar_width = 60,
                                    seed = NULL,
                                    prior_prec1_type = c("halfcauchy", "gamma", "halft"),
                                    prior_prec1_scale = 2,
                                    prior_prec1_df = 1,
                                    chains = 1,
                                    parallel = FALSE) {

  # --- Multi-chain dispatch ---
  # Re-issues this same call once per chain, each with its own seed, and returns
  # the collection. Kept at the very top so `match.call()` captures the call
  # exactly as the user wrote it.
  validate_chains_args(chains, parallel)
  if (chains > 1) {
    return(run_chains(match.call(), parent.frame(),
                      as.integer(chains), seed, parallel))
  }

  # `missing()` must be read before the argument is touched.
  prec1_df_user_set <- !missing(prior_prec1_df)
  # Whether the type was named decides how a bare shape/rate pair is read
  # (see infer_gamma_from_hyperparams() in R/innovation_priors.R).
  prec1_type_user_set <- !missing(prior_prec1_type)

  prior_prec1_type  <- match.arg(prior_prec1_type)

  prior_prec1_type <- infer_gamma_from_hyperparams(
    prior_prec1_type, prec1_type_user_set,
    !missing(prior_prec1_shape) || !missing(prior_prec1_rate)
  )
  # --- Input Validation ---
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

  if (! is.numeric(burnin) || length(burnin) != 1 || burnin < 0 ||
      burnin != floor(burnin)) {
    stop("`burnin` must be a single non-negative integer")
  }
  if (!is.numeric(thinning) || length(thinning) != 1 || thinning < 1 ||
      thinning != floor(thinning)) {
    stop("`thinning` must be a single positive integer")
  }
  if (!is.numeric(n_chain) || length(n_chain) != 1 || n_chain < 1 ||
      n_chain != floor(n_chain)) {
    stop("`n_chain` must be a single positive integer")
  }
  if (!is.numeric(prior_theta01_mean) || length(prior_theta01_mean) != 1) {
    stop("`prior_theta01_mean` must be a single numeric value")
  }
  if (!is.numeric(prior_theta01_prec) || length(prior_theta01_prec) != 1 ||
      prior_theta01_prec <= 0) {
    stop("`prior_theta01_prec` must be a single positive numeric value")
  }
  # Resolve the innovation precision prior (Gamma or Half-t); see R/prec_prior.R.
  prec1_prior <- resolve_prec_prior(
    prior_prec1_type, prior_prec1_shape, prior_prec1_rate,
    prior_prec1_scale, prior_prec1_df, prec1_df_user_set, "prior_prec1"
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
    stop("`target_acceptance` must be a single numeric value in (0,1)")
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
  if (!is.logical(verbose) || length(verbose) != 1) {
    stop("`verbose` must be a single logical value")
  }
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
  # --- End Input Validation ---

  # Call the C function
  result <- .Call(
    "_pdm_C_MCMC_log_poisson_locallevel",
    as.numeric(y),
    as.integer(burnin),
    as.integer(thinning),
    as.integer(n_chain),
    as.numeric(prior_theta01_mean),
    as.numeric(prior_theta01_prec),
    as.integer(prec1_prior$code),
    as.numeric(prec1_prior$shape),
    as.numeric(prec1_prior$rate),
    as.numeric(prec1_prior$scale),
    as.numeric(prec1_prior$df),
    as.integer(lag_update),
    as.numeric(max_step_size),
    as.numeric(base_adaptation_rate),
    as.numeric(decay_exponent),
    as.numeric(target_acceptance),
    as.numeric(min_deviation_threshold),
    as.logical(return_log_sigma),
    as.logical(return_accept_prop),
    as.logical(verbose),
    as.integer(bar_width)
  )

  result <- new_poisson_locallevel(
    result = result,
    n_obs = length(y),
    n_chain = n_chain,
    burnin = burnin,
    thinning = thinning,
    y = y,
    target_acceptance = target_acceptance
  )

  result <- validate_poisson_locallevel(result)

  # Record the prior used on the innovation precision (auxiliary Half-t variable
  # is a nuisance parameter and is intentionally not returned).
  attr(result, "prior_prec_theta1") <- prec1_prior[c("type", "shape", "rate", "scale", "df")]

  return(result)
}
