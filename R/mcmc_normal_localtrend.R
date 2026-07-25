#' @title Gibbs Sampler for a Local-Trend Dynamic Model
#'
#' @description Runs a Gibbs sampler for the local trend dynamic model.
#'
#' @details The model is defined as:
#' \deqn{
#' \begin{aligned}
#' y_t &= \theta_{t,1} + \epsilon_t,                          & \epsilon_t  & \sim N(0, V), \\
#' \theta_{t,1} &= \theta_{t-1,1} + \theta_{t-1,2} + u_{t,1}, & u_{t,1}     & \sim N(0, W_1), \\
#' \theta_{t,2} &= \theta_{t-1,2} + u_{t,2},                  & u_{t,2}     & \sim N(0, W_2),
#' \end{aligned}
#' }
#' where \eqn{t = 1, 2, \ldots, n} and \eqn{n} is the number of observations.
#'
#'
#' \strong{Prior Distributions:}
#'
#' The following conjugate prior distributions are employed to ensure computational
#' tractability and closed-form posterior updates in the Gibbs sampler:
#'
#' \emph{Initial States:}
#' \deqn{
#' \begin{aligned}
#' \theta_{0,1} &\sim N(\mu_{01}, \tau_{01}^{-1}), \\
#' \theta_{0,2} &\sim N(\mu_{02}, \tau_{02}^{-1}).
#' \end{aligned}
#' }
#'
#' \emph{Precision Parameters:}
#'
#' Each precision (the innovation precisions \eqn{W_1^{-1}}, \eqn{W_2^{-1}} and
#' the observation precision \eqn{V^{-1}}) may be given one of two priors, chosen
#' independently through `prior_prec1_type`, `prior_prec2_type` and
#' `prior_prec_y_type`:
#'
#' \emph{(a) Gamma prior on the precision} (default, conjugate):
#' \deqn{
#' \begin{aligned}
#' W_1^{-1} &\sim \text{Gamma}(\nu_1, \eta_1), \\
#' W_2^{-1} &\sim \text{Gamma}(\nu_2, \eta_2), \\
#' V^{-1} &\sim \text{Gamma}(\nu_V, \eta_V).
#' \end{aligned}
#' }
#'
#' \emph{(b) Half-t prior on the standard deviation} (Gelman, 2006):
#' \deqn{
#' \sqrt{W_1} \sim \text{Half-}t(\nu_1, A_1), \quad
#' \sqrt{W_2} \sim \text{Half-}t(\nu_2, A_2), \quad
#' \sqrt{V} \sim \text{Half-}t(\nu_V, A_V),
#' }
#' where \eqn{A > 0} is a scale hyperparameter and \eqn{\nu > 0} the degrees of
#' freedom. Setting \eqn{\nu = 1} gives the Half-Cauchy prior (also selectable
#' directly with `type = "halfcauchy"`); larger \eqn{\nu} approaches a
#' Half-Normal. The Half-t prior is represented by the inverse-gamma scale
#' mixture of Wand et al. (2011), which keeps every Gibbs update closed-form via
#' a single auxiliary variable per precision.
#'
#' The Gamma parameterization uses shape-rate, where \eqn{E(X) = \nu/\eta} and
#' \eqn{\text{Var}(X) = \nu/\eta^2}. The Gamma shape/rate arguments are required
#' only when the corresponding `_type` is `"gamma"`; the Half-t scale/df
#' arguments are required only when it is `"halft"` or `"halfcauchy"`. The prior
#' choice is resolved once, before the sampler runs, and never re-evaluated
#' inside the MCMC loop.
#'
#' Burn-in and thinning are applied so that exactly `n_chain` posterior samples
#' are returned.
#'
#' @param y Numeric vector of observations (length \eqn{n}). Must contain only finite values.
#' @param burnin Integer \eqn{\geq 0}, number of burn-in iterations.
#' @param thinning Integer \eqn{\geq 1}, thinning interval.
#' @param n_chain Integer \eqn{\geq 1}, number of posterior samples to retain.
#' @param prior_theta01_mean Numeric, prior mean for the initial level \eqn{\theta_{0,1}}.
#' @param prior_theta01_prec Numeric > 0, prior precision for \eqn{\theta_{0,1}}.
#' @param prior_theta02_mean Numeric, prior mean for the initial trend \eqn{\theta_{0,2}}.
#' @param prior_theta02_prec Numeric > 0, prior precision for \eqn{\theta_{0,2}}.
#' @param prior_prec1_type Character, prior on the level innovation precision
#'   \eqn{1/W_1}: `"gamma"` (default) for a Gamma prior on the precision, or
#'   `"halft"` / `"halfcauchy"` for a Half-t / Half-Cauchy prior on
#'   \eqn{\sqrt{W_1}} (Gelman, 2006). `"halfcauchy"` is Half-t with `df = 1`.
#' @param prior_prec1_shape Numeric > 0, shape parameter \eqn{\nu_1} of the Gamma
#'   prior for \eqn{1/W_1}. Required (and used) only when `prior_prec1_type = "gamma"`.
#' @param prior_prec1_rate Numeric > 0, rate parameter \eqn{\eta_1} of the Gamma
#'   prior for \eqn{1/W_1}. Required (and used) only when `prior_prec1_type = "gamma"`.
#' @param prior_prec1_scale Numeric > 0, scale \eqn{A_1} of the Half-t prior for
#'   \eqn{\sqrt{W_1}}. Required when `prior_prec1_type` is `"halft"` or
#'   `"halfcauchy"`; ignored otherwise.
#' @param prior_prec1_df Numeric > 0, degrees of freedom \eqn{\nu_1} of the Half-t
#'   prior for \eqn{\sqrt{W_1}}. Default `1` (Half-Cauchy). Must equal `1` when
#'   `prior_prec1_type = "halfcauchy"`.
#' @param prior_prec2_type Character, prior on the trend innovation precision
#'   \eqn{1/W_2}: `"gamma"` (default), or `"halft"` / `"halfcauchy"` for a Half-t
#'   / Half-Cauchy prior on \eqn{\sqrt{W_2}}.
#' @param prior_prec2_shape Numeric > 0, shape parameter \eqn{\nu_2} of the Gamma
#'   prior for \eqn{1/W_2}. Required (and used) only when `prior_prec2_type = "gamma"`.
#' @param prior_prec2_rate Numeric > 0, rate parameter \eqn{\eta_2} of the Gamma
#'   prior for \eqn{1/W_2}. Required (and used) only when `prior_prec2_type = "gamma"`.
#' @param prior_prec2_scale Numeric > 0, scale \eqn{A_2} of the Half-t prior for
#'   \eqn{\sqrt{W_2}}. Required when `prior_prec2_type` is `"halft"` or
#'   `"halfcauchy"`; ignored otherwise.
#' @param prior_prec2_df Numeric > 0, degrees of freedom \eqn{\nu_2} of the Half-t
#'   prior for \eqn{\sqrt{W_2}}. Default `1` (Half-Cauchy). Must equal `1` when
#'   `prior_prec2_type = "halfcauchy"`.
#' @param prior_prec_y_type Character, prior on the observation precision
#'   \eqn{1/V}: `"gamma"` (default), or `"halft"` / `"halfcauchy"` for a Half-t /
#'   Half-Cauchy prior on \eqn{\sqrt{V}}.
#' @param prior_prec_y_shape Numeric > 0, shape parameter \eqn{\nu_V} of the Gamma
#'   prior for \eqn{1/V}. Required (and used) only when `prior_prec_y_type = "gamma"`.
#' @param prior_prec_y_rate Numeric > 0, rate parameter \eqn{\eta_V} of the Gamma
#'   prior for \eqn{1/V}. Required (and used) only when `prior_prec_y_type = "gamma"`.
#' @param prior_prec_y_scale Numeric > 0, scale \eqn{A_V} of the Half-t prior for
#'   \eqn{\sqrt{V}}. Required when `prior_prec_y_type` is `"halft"` or
#'   `"halfcauchy"`; ignored otherwise.
#' @param prior_prec_y_df Numeric > 0, degrees of freedom \eqn{\nu_V} of the
#'   Half-t prior for \eqn{\sqrt{V}}. Default `1` (Half-Cauchy). Must equal `1`
#'   when `prior_prec_y_type = "halfcauchy"`.
#' @param verbose Logical, whether to display a progress bar during sampling. Default is `FALSE`.
#' @param bar_width Integer in \[10, 120\], width of the progress bar when `verbose = TRUE`. Default is `60`.
#' @param seed Optional integer used to set the random number generator seed.
#'   Default is `NULL`, which does not set the seed. When `chains > 1` it acts
#'   as a master seed from which each chain's own seed is drawn.
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
#' @return A list with components:
#' \describe{
#'   \item{\code{theta_1}}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior samples for the level state \eqn{\theta_{t,1}}.}
#'   \item{\code{theta_2}}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior samples for the trend state \eqn{\theta_{t,2}}.}
#'   \item{\code{theta_01}}{Numeric vector of length `n_chain` for the initial level \eqn{\theta_{0,1}}.}
#'   \item{\code{theta_02}}{Numeric vector of length `n_chain` for the initial trend \eqn{\theta_{0,2}}.}
#'   \item{\code{prec_theta1}}{Numeric vector of length `n_chain` for the level innovation precision \eqn{1/W_1}.}
#'   \item{\code{prec_theta2}}{Numeric vector of length `n_chain` for the trend innovation precision \eqn{1/W_2}.}
#'   \item{\code{prec_y}}{Numeric vector of length `n_chain` for the data precision \eqn{1/V}.}
#' }
#'
#' When `chains > 1` the return value is instead an object of class
#' `c("pdm_mcmc_list", "list")` holding one such fit per chain; see
#' \code{\link{print.pdm_mcmc_list}}.
#'
#' @examples
#' ## Description
#' # This example demonstrates how to:
#' # 1. Simulate data from a local trend dynamic model
#' # 2. Use `mcmc_normal_localtrend` to estimate parameters and latent states
#' # 3. Perform a detailed posterior analysis with visualizations
#' # 4. Set a seed for reproducibility
#'
#' ## Simulation of data
#' n <- 200  # Number of observations to simulate
#'
#' # True parameters for simulation:
#' theta01_true <- 10        # Initial level (theta[0,1])
#' theta02_true <- 0.5       # Initial trend (theta[0,2])
#' prec1_true   <- 1 / 0.10  # Level innovation precision (1/W[1])
#' prec2_true   <- 1 / 0.01  # Trend innovation precision (1/W[2])
#' prec_y_true  <- 1 / 1.00  # Observation precision (1/V)
#'
#' # Use a fixed seed for data simulation
#' set.seed(123)
#'
#' # Generate noise terms:
#' u1      <- rnorm(n, sd = sqrt(1 / prec1_true))  # Level noise
#' u2      <- rnorm(n, sd = sqrt(1 / prec2_true))  # Trend noise
#' epsilon <- rnorm(n, sd = sqrt(1 / prec_y_true)) # Observation noise
#'
#' # Simulate latent states and observations:
#' theta1_true    <- numeric(n)
#' theta2_true    <- numeric(n)
#' theta2_true[1] <- theta02_true + u2[1]
#' theta1_true[1] <- theta01_true + theta02_true + u1[1]
#' for (t in 2:n) {
#'   theta2_true[t] <- theta2_true[t-1] + u2[t]
#'   theta1_true[t] <- theta1_true[t-1] + theta2_true[t-1] + u1[t]
#' }
#' y <- theta1_true + epsilon # Observed data
#'
#' ## Running the Gibbs sampler
#' # Run the Gibbs sampler with specified priors and a seed
#' out <- mcmc_normal_localtrend(
#'   y,
#'   burnin             = 1000,
#'   thinning           = 20,
#'   n_chain            = 500,
#'   prior_theta01_mean = y[1] / 2,
#'   prior_theta01_prec = 1 / var(y),
#'   prior_theta02_mean = y[1] / 2,
#'   prior_theta02_prec = 1 / var(y),
#'   prior_prec1_shape  = 1e-1,
#'   prior_prec1_rate   = 1e-1,
#'   prior_prec2_shape  = 1e-2,
#'   prior_prec2_rate   = 1e-2,
#'   prior_prec_y_shape = 1e-1,
#'   prior_prec_y_rate  = 1e-1,
#'   verbose            = TRUE,
#'   bar_width          = 60,
#'   seed               = 456
#' )
#'
#' ## Alternative: weakly-informative Half-Cauchy priors (Gelman, 2006)
#' # Put a Half-Cauchy prior on the level and trend innovation SDs sqrt(W[1]),
#' # sqrt(W[2]), keeping a Gamma prior on the observation precision 1/V. Each
#' # precision's prior is chosen independently via its `*_type` argument.
#' out_hc <- mcmc_normal_localtrend(
#'   y,
#'   burnin             = 1000,
#'   thinning           = 20,
#'   n_chain            = 500,
#'   prior_theta01_mean = y[1] / 2,
#'   prior_theta01_prec = 1 / var(y),
#'   prior_theta02_mean = y[1] / 2,
#'   prior_theta02_prec = 1 / var(y),
#'   prior_prec1_type   = "halfcauchy",
#'   prior_prec1_scale  = 1,     # sqrt(W[1])
#'   prior_prec2_type   = "halfcauchy",
#'   prior_prec2_scale  = 1,     # sqrt(W[2])
#'   prior_prec_y_shape = 1e-1,
#'   prior_prec_y_rate  = 1e-1,  # 1/V keeps Gamma
#'   seed               = 456
#' )
#'
#' ## Posterior analysis and visualization
#' # Use the plot method for comprehensive diagnostics
#' \donttest{
#'   # --- Gamma-prior fit (`out`) ---
#'   # Complete dashboard with all diagnostics
#'   plot(out, type = "all")
#'
#'   # Individual diagnostic types
#'   plot(out, type = "mcmc", which = 1)      # Observation precision (V^-1)
#'   plot(out, type = "mcmc", which = 2:3)    # Initial states (theta_0)
#'   plot(out, type = "mcmc", which = 4:5)    # Innovation precisions (W^-1)
#'   plot(out, type = "states")               # Dynamic states
#'
#'   # Plot with true states for comparison
#'   plot(out, type = "states",
#'        true_values = list(theta_1 = theta1_true, theta_2 = theta2_true))
#'
#'   # --- Half-Cauchy-prior fit (`out_hc`) ---
#'   # The same diagnostics for the weakly-informative Half-Cauchy fit.
#'   plot(out_hc, type = "all")
#'
#'   plot(out_hc, type = "mcmc", which = 2:3)  # Initial states (theta_0)
#'   plot(out_hc, type = "states")             # Dynamic states
#'
#'   plot(out_hc, type = "states",
#'        true_values = list(theta_1 = theta1_true, theta_2 = theta2_true))
#' }
#'
#' @references
#' Gelman, A. (2006). Prior distributions for variance parameters in
#' hierarchical models. \emph{Bayesian Analysis}, 1(3), 515-534.
#'
#' Wand, M. P., Ormerod, J. T., Padoan, S. A., & Fruhwirth, R. (2011). Mean field
#' variational Bayes for elaborate distributions. \emph{Bayesian Analysis},
#' 6(4), 847-900.
#'
#' @seealso
#'   \code{\link{plot.normal_localtrend}},
#'   \code{\link{print.normal_localtrend}},
#'   \code{\link{summary.normal_localtrend}} for methods on the fitted object;
#'   \code{\link{mcmc_normal_locallevel}} and
#'   \code{\link{mcmc_normal_localacceleration}} for the other dynamic orders.
#'
#' @export
#'
mcmc_normal_localtrend <- function(y,
                                   burnin,
                                   thinning,
                                   n_chain,
                                   prior_theta01_mean,
                                   prior_theta01_prec,
                                   prior_theta02_mean,
                                   prior_theta02_prec,
                                   prior_prec1_shape = NULL,
                                   prior_prec1_rate = NULL,
                                   prior_prec2_shape = NULL,
                                   prior_prec2_rate = NULL,
                                   prior_prec_y_shape = NULL,
                                   prior_prec_y_rate = NULL,
                                   verbose = FALSE,
                                   bar_width = 60,
                                   seed = NULL,
                                   prior_prec1_type = c("gamma", "halfcauchy", "halft"),
                                   prior_prec1_scale = NULL,
                                   prior_prec1_df = 1,
                                   prior_prec2_type = c("gamma", "halfcauchy", "halft"),
                                   prior_prec2_scale = NULL,
                                   prior_prec2_df = 1,
                                   prior_prec_y_type = c("gamma", "halfcauchy", "halft"),
                                   prior_prec_y_scale = NULL,
                                   prior_prec_y_df = 1,
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

  # Record whether the user explicitly set each df (used to flag a contradictory
  # `type = "halfcauchy"` + `df != 1`). `missing()` must be read before the
  # arguments are touched.
  prec1_df_user_set  <- !missing(prior_prec1_df)
  prec2_df_user_set  <- !missing(prior_prec2_df)
  prec_y_df_user_set <- !missing(prior_prec_y_df)

  prior_prec1_type  <- match.arg(prior_prec1_type)
  prior_prec2_type  <- match.arg(prior_prec2_type)
  prior_prec_y_type <- match.arg(prior_prec_y_type)
  # --- Input Validation ---
  if (!is.numeric(y)) {
    stop("`y` must be a numeric vector")
  }
  if (!all(is.finite(y))) {
    stop("`y` must contain only finite numeric values (no NA, NaN, Inf)")
  }
  if (!is.numeric(burnin) || length(burnin) != 1 || burnin < 0 || burnin != floor(burnin)) {
    stop("`burnin` must be a single non-negative integer")
  }
  if (!is.numeric(thinning) || length(thinning) != 1 || thinning < 1 || thinning != floor(thinning)) {
    stop("`thinning` must be a single positive integer")
  }
  if (!is.numeric(n_chain) || length(n_chain) != 1 || n_chain < 1 || n_chain != floor(n_chain)) {
    stop("`n_chain` must be a single positive integer")
  }
  # Priors for theta_01
  if (!is.numeric(prior_theta01_mean) || length(prior_theta01_mean) != 1) {
    stop("`prior_theta01_mean` must be a single numeric value")
  }
  if (!is.numeric(prior_theta01_prec) || length(prior_theta01_prec) != 1 || prior_theta01_prec <= 0) {
    stop("`prior_theta01_prec` must be a single positive numeric value")
  }
  # Priors for theta_02
  if (!is.numeric(prior_theta02_mean) || length(prior_theta02_mean) != 1) {
    stop("`prior_theta02_mean` must be a single numeric value")
  }
  if (!is.numeric(prior_theta02_prec) || length(prior_theta02_prec) != 1 || prior_theta02_prec <= 0) {
    stop("`prior_theta02_prec` must be a single positive numeric value")
  }
  # --- Resolve each precision prior (Gamma or Half-t) ---
  # Validation and the "halfcauchy" -> Half-t(df = 1) normalisation are handled
  # by the shared internal helper `resolve_prec_prior()` (see R/prec_prior.R), so
  # the C sampler receives an already-decided integer code (0 = Gamma, 1 = Half-t)
  # plus finite numeric hyperparameters. The prior choice is never re-decided
  # inside the MCMC loop.
  prec1_prior <- resolve_prec_prior(
    prior_prec1_type, prior_prec1_shape, prior_prec1_rate,
    prior_prec1_scale, prior_prec1_df, prec1_df_user_set, "prior_prec1"
  )
  prec2_prior <- resolve_prec_prior(
    prior_prec2_type, prior_prec2_shape, prior_prec2_rate,
    prior_prec2_scale, prior_prec2_df, prec2_df_user_set, "prior_prec2"
  )
  prec_y_prior <- resolve_prec_prior(
    prior_prec_y_type, prior_prec_y_shape, prior_prec_y_rate,
    prior_prec_y_scale, prior_prec_y_df, prec_y_df_user_set, "prior_prec_y"
  )

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

  # Call the C function. Argument order groups each precision's prior spec
  # (type code, Gamma shape/rate, Half-t scale/df) and must match the C signature
  # in `src/mcmc_normal_localtrend.c`.
  result <- .Call(
    "_pdm_C_MCMC_normal_localtrend",
    as.numeric(y),
    as.integer(burnin),
    as.integer(thinning),
    as.integer(n_chain),
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
    as.integer(prec_y_prior$code),
    as.numeric(prec_y_prior$shape),
    as.numeric(prec_y_prior$rate),
    as.numeric(prec_y_prior$scale),
    as.numeric(prec_y_prior$df),
    as.logical(verbose),
    as.integer(bar_width)
  )

  result <- new_normal_localtrend(
    result = result,
    n_obs = length(y),
    n_chain = n_chain,
    burnin = burnin,
    thinning = thinning,
    y = y
  )

  result <- validate_normal_localtrend(result)

  # Record the priors actually used on each precision (for reproducibility /
  # downstream reporting). The auxiliary Half-t variables are nuisance
  # parameters and are intentionally not returned.
  attr(result, "prior_prec_theta1") <- prec1_prior[c("type", "shape", "rate", "scale", "df")]
  attr(result, "prior_prec_theta2") <- prec2_prior[c("type", "shape", "rate", "scale", "df")]
  attr(result, "prior_prec_y")      <- prec_y_prior[c("type", "shape", "rate", "scale", "df")]

  return(result)
}
