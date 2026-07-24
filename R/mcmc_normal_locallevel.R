#' @title Gibbs Sampler for a Local-Level Dynamic Model
#'
#' @description Runs a Gibbs sampler for the local-level dynamic model.
#'
#' @details The model is defined as:
#' \deqn{
#' \begin{aligned}
#' y_t &= \theta_{t,1} + e_t,                 & e_t     & \sim N(0, V),  \\
#' \theta_{t,1} &= \theta_{t-1,1} + u_{t,1},  & u_{t,1} & \sim N(0, W_1),
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
#' \theta_{0,1} &\sim N(\mu_{01}, \tau_{01}^{-1}).
#' \end{aligned}
#' }
#'
#' \emph{Precision Parameters:}
#'
#' Each precision (the innovation precision \eqn{W_1^{-1}} and the observation
#' precision \eqn{V^{-1}}) may be given one of two priors, chosen independently
#' through `prior_prec1_type` and `prior_prec_y_type`:
#'
#' \emph{(a) Gamma prior on the precision} (default, conjugate):
#' \deqn{
#' \begin{aligned}
#' W_1^{-1} &\sim \text{Gamma}(\nu_1, \eta_1), \\
#' V^{-1} &\sim \text{Gamma}(\nu_V, \eta_V).
#' \end{aligned}
#' }
#'
#' \emph{(b) Half-t prior on the standard deviation} (Gelman, 2006):
#' \deqn{
#' \sqrt{W_1} \sim \text{Half-}t(\nu_1, A_1), \qquad
#' \sqrt{V} \sim \text{Half-}t(\nu_V, A_V),
#' }
#' where \eqn{A > 0} is a scale hyperparameter and \eqn{\nu > 0} the degrees of
#' freedom. Setting \eqn{\nu = 1} gives the Half-Cauchy prior (also selectable
#' directly with `type = "halfcauchy"`); larger \eqn{\nu} approaches a
#' Half-Normal. The Half-t prior is represented by the inverse-gamma scale
#' mixture of Wand et al. (2011), which keeps every Gibbs update closed-form via
#' a single auxiliary variable per precision.
#'
#' The prior hyperparameters correspond to function arguments as follows:
#' \tabular{cc}{
#'   \strong{Hyperparameter} \tab \strong{Function Argument} \cr
#'   \eqn{\mu_{01}} \tab `prior_theta01_mean` \cr
#'   \eqn{\tau_{01}} \tab `prior_theta01_prec` \cr
#'   \eqn{\nu_1} (Gamma shape) \tab `prior_prec1_shape` \cr
#'   \eqn{\eta_1} (Gamma rate) \tab `prior_prec1_rate` \cr
#'   \eqn{A_1} (Half-t scale) \tab `prior_prec1_scale` \cr
#'   \eqn{\nu_1} (Half-t df) \tab `prior_prec1_df` \cr
#'   \eqn{\nu_V} (Gamma shape) \tab `prior_prec_y_shape` \cr
#'   \eqn{\eta_V} (Gamma rate) \tab `prior_prec_y_rate` \cr
#'   \eqn{A_V} (Half-t scale) \tab `prior_prec_y_scale` \cr
#'   \eqn{\nu_V} (Half-t df) \tab `prior_prec_y_df`
#' }
#'
#' The Gamma parameterization uses shape-rate, where \eqn{E(X) = \nu/\eta} and
#' \eqn{\text{Var}(X) = \nu/\eta^2}. The Gamma shape/rate arguments are required
#' only when the corresponding `_type` is `"gamma"`; the Half-t scale/df
#' arguments are required only when it is `"halft"` or `"halfcauchy"`. The prior
#' choice is resolved once, before the sampler runs, and never re-evaluated
#' inside the MCMC loop.
#'
#' Burn‐in and thinning are applied so that exactly `n_chain`
#' posterior samples are returned.
#'
#' @param y Numeric vector of observations (length \eqn{n}). Must contain only finite values.
#' @param burnin Integer \eqn{\geq 0}, number of burn-in iterations.
#' @param thinning Integer \eqn{\geq 1}, thinning interval.
#' @param n_chain Integer \eqn{\geq 1}, number of posterior samples to retain.
#' @param prior_theta01_mean Numeric, prior mean for the initial state \eqn{\theta_{0,1}}.
#' @param prior_theta01_prec Numeric > 0, prior precision (inverse variance) for \eqn{\theta_{0,1}}.
#' @param prior_prec1_type Character, prior on the innovation precision
#'   \eqn{1/W_1}: `"gamma"` (default) for a Gamma prior on the precision, or
#'   `"halft"` / `"halfcauchy"` for a Half-t / Half-Cauchy prior on the innovation
#'   standard deviation \eqn{\sqrt{W_1}} (Gelman, 2006). `"halfcauchy"` is Half-t
#'   with `df = 1`.
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
#' @param prior_prec_y_type Character, prior on the observation precision
#'   \eqn{1/V}: `"gamma"` (default), or `"halft"` / `"halfcauchy"` for a Half-t /
#'   Half-Cauchy prior on the observation standard deviation \eqn{\sqrt{V}}.
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
#'   Default is `NULL`, which does not set the seed.
#'
#' @return An object of class `c("normal_locallevel", "pdm_mcmc", "list")`
#'   containing the following components:
#'   \describe{
#'     \item{\code{theta_1}}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior samples for the latent state \eqn{\theta_{t,1}}.}
#'     \item{\code{theta_01}}{Numeric vector of length `n_chain` of posterior samples for the initial state \eqn{\theta_{0,1}}.}
#'     \item{\code{prec_theta1}}{Numeric vector of length `n_chain` of posterior samples for the innovation precision \eqn{1/W_1}.}
#'     \item{\code{prec_y}}{Numeric vector of length `n_chain` of posterior samples for the data precision \eqn{1/V}.}
#'   }
#'   Metadata about the MCMC run (burn-in, thinning, number of retained
#'   samples, and original data) are stored as attributes to facilitate S3
#'   method dispatch.
#'
#' @examples
#' ## Description
#' # This example demonstrates how to:
#' # 1. Simulate data from a local-level dynamic model
#' # 2. Use `mcmc_normal_locallevel` to estimate parameters and latent states
#' # 3. Perform a detailed posterior analysis with visualizations
#' # 4. Set a seed for reproducibility
#'
#' ## Simulation of data
#' n <- 1000  # Number of observations to simulate
#'
#' # True parameters for simulation:
#' theta0_true <- 10  # Initial state (theta[01])
#' prec1_true <- 1    # Innovation precision (1/W[1])
#' prec_y_true <- 5   # Observation precision (1/V)
#'
#' # Use a fixed seed for data simulation
#' set.seed(123)
#'
#' # Generate noise terms:
#' u1 <- rnorm(n, sd = sqrt(1/prec1_true))  # Evolution noise (u1[t])
#' e  <- rnorm(n, sd = sqrt(1/prec_y_true)) # Observation noise (e[t])
#'
#' # Simulate latent states and observations:
#' theta1_true <- cumsum(c(theta0_true, u1))[-1]  # theta[t1] series
#' y <- theta1_true + e                           # Observed data (y[t])
#'
#' ## Running the Gibbs sampler
#' # Run the Gibbs sampler with specified priors and a seed
#' out <- mcmc_normal_locallevel(
#'   y,
#'   burnin             = 1000,
#'   thinning           = 10,
#'   n_chain            = 1000,
#'   prior_theta01_mean = y[1],
#'   prior_theta01_prec = 1 / var(y),
#'   prior_prec1_shape  = 1e-2,
#'   prior_prec1_rate   = 1e-2,
#'   prior_prec_y_shape = 1e-2,
#'   prior_prec_y_rate  = 1e-2,
#'   verbose            = TRUE,
#'   bar_width          = 60,
#'   seed               = 456
#' )
#'
#' ## Alternative: weakly-informative Half-Cauchy prior (Gelman, 2006)
#' # Any precision can instead take a Half-t / Half-Cauchy prior on its standard
#' # deviation. Here the innovation SD sqrt(W[1]) gets a Half-Cauchy(scale = 1)
#' # prior (prior_prec1_type = "halfcauchy"), while the observation precision 1/V
#' # keeps its Gamma prior. The Gamma shape/rate for W[1] are then unused.
#' out_hc <- mcmc_normal_locallevel(
#'   y,
#'   burnin             = 1000,
#'   thinning           = 10,
#'   n_chain            = 1000,
#'   prior_theta01_mean = y[1],
#'   prior_theta01_prec = 1 / var(y),
#'   prior_prec1_type   = "halfcauchy",  # Half-Cauchy on sqrt(W[1])
#'   prior_prec1_scale  = 1,             # scale A_1 > 0
#'   prior_prec_y_shape = 1e-2,          # 1/V keeps the Gamma prior
#'   prior_prec_y_rate  = 1e-2,
#'   seed               = 456
#' )
#' # Use prior_prec1_type = "halft" with prior_prec1_df > 1 for a general Half-t.
#'
#' ## Posterior analysis and visualization
#' # Use the plot method for comprehensive diagnostics
#' \donttest{
#'   # --- Gamma-prior fit (`out`) ---
#'   # Complete dashboard with all diagnostics
#'   plot(out, type = "all")
#'
#'   # Individual diagnostic types
#'   plot(out, type = "mcmc", which = 1)      # Observation precision
#'   plot(out, type = "mcmc", which = 2:3)    # theta_01 and W_1^-1
#'   plot(out, type = "states")               # Dynamic level state
#'
#'   # Plot with the true level state for comparison
#'   plot(out, type = "states", true_values = list(theta_1 = theta1_true))
#'
#'   # --- Half-Cauchy-prior fit (`out_hc`) ---
#'   # The same diagnostics for the weakly-informative Half-Cauchy fit.
#'   plot(out_hc, type = "all")
#'
#'   plot(out_hc, type = "states")             # Dynamic level state
#'
#'   plot(out_hc, type = "states", true_values = list(theta_1 = theta1_true))
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
#'   \code{\link{plot.normal_locallevel}},
#'   \code{\link{print.normal_locallevel}},
#'   \code{\link{summary.normal_locallevel}} for methods on the fitted object;
#'   \code{\link{mcmc_normal_localtrend}} and
#'   \code{\link{mcmc_normal_localacceleration}} for the other dynamic orders.
#'
#' @export
mcmc_normal_locallevel <- function(y,
                                   burnin,
                                   thinning,
                                   n_chain,
                                   prior_theta01_mean,
                                   prior_theta01_prec,
                                   prior_prec1_shape = NULL,
                                   prior_prec1_rate = NULL,
                                   prior_prec_y_shape = NULL,
                                   prior_prec_y_rate = NULL,
                                   verbose = FALSE,
                                   bar_width = 60,
                                   seed = NULL,
                                   prior_prec1_type = c("gamma", "halfcauchy", "halft"),
                                   prior_prec1_scale = NULL,
                                   prior_prec1_df = 1,
                                   prior_prec_y_type = c("gamma", "halfcauchy", "halft"),
                                   prior_prec_y_scale = NULL,
                                   prior_prec_y_df = 1) {

  # Record whether the user explicitly set the degrees of freedom (used to flag
  # a contradictory `type = "halfcauchy"` + `df != 1`). `missing()` must be read
  # before these arguments are touched.
  prec1_df_user_set  <- !missing(prior_prec1_df)
  prec_y_df_user_set <- !missing(prior_prec_y_df)

  prior_prec1_type  <- match.arg(prior_prec1_type)
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
  if (!is.numeric(prior_theta01_mean) || length(prior_theta01_mean) != 1) {
    stop("`prior_theta01_mean` must be a single numeric value")
  }
  if (!is.numeric(prior_theta01_prec) || length(prior_theta01_prec) != 1 || prior_theta01_prec <= 0) {
    stop("`prior_theta01_prec` must be a single positive numeric value")
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
  # in `src/mcmc_normal_locallevel.c`.
  result <- .Call(
    "_pdm_C_MCMC_normal_locallevel",
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
    as.integer(prec_y_prior$code),
    as.numeric(prec_y_prior$shape),
    as.numeric(prec_y_prior$rate),
    as.numeric(prec_y_prior$scale),
    as.numeric(prec_y_prior$df),
    as.logical(verbose),
    as.integer(bar_width)
  )

  result <- new_normal_locallevel(
    result = result,
    n_obs = length(y),
    n_chain = n_chain,
    burnin = burnin,
    thinning = thinning,
    y = y
  )

  result <- validate_normal_locallevel(result)

  # Record the priors actually used on the innovation and observation precisions
  # (for reproducibility / downstream reporting). The auxiliary Half-t variable
  # is a nuisance parameter and is intentionally not returned.
  attr(result, "prior_prec_theta1") <- prec1_prior[c("type", "shape", "rate", "scale", "df")]
  attr(result, "prior_prec_y")      <- prec_y_prior[c("type", "shape", "rate", "scale", "df")]

  return(result)
}
