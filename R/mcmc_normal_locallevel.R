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
#' \emph{(a) Gamma prior on the precision} (conjugate; the default until 0.4-0):
#' \deqn{
#' \begin{aligned}
#' W_1^{-1} &\sim \text{Gamma}(\nu_1, \eta_1), \\
#' V^{-1} &\sim \text{Gamma}(\nu_V, \eta_V).
#' \end{aligned}
#' }
#'
#' \emph{(b) Half-t prior on the standard deviation} (Gelman, 2006; the
#' default since 0.4-0, with a data-scaled \eqn{A}):
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
#' @param prior_prec1_type Character, prior on the level innovation precision
#'   \eqn{1/W_1}: `"halfcauchy"` (default) or `"halft"` for a Half-t /
#'   Half-Cauchy prior on the corresponding standard deviation (Gelman, 2006),
#'   or `"gamma"` for a Gamma prior on the precision. The default changed in
#'   0.4-0; a call that supplies `prior_prec1_shape`/`prior_prec1_rate`
#'   without naming a type is still read as `"gamma"`, so existing code keeps
#'   working.
#' @param prior_prec1_shape Numeric > 0, shape parameter \eqn{\nu_1} of the Gamma
#'   prior for \eqn{1/W_1}. Required (and used) only when `prior_prec1_type = "gamma"`.
#' @param prior_prec1_rate Numeric > 0, rate parameter \eqn{\eta_1} of the Gamma
#'   prior for \eqn{1/W_1}. Required (and used) only when `prior_prec1_type = "gamma"`.
#' @param prior_prec1_scale Numeric > 0, scale \eqn{A} of the Half-t prior for
#'   \eqn{\sqrt{W_1}}. If `NULL` (default), derived from the data as
#'   `sd(diff(y)) / (2 * sqrt(2))`, which makes the fit invariant to
#'   the units of `y`. Ignored when `prior_prec1_type = "gamma"`.
#' @param prior_prec1_df Numeric > 0, degrees of freedom \eqn{\nu_1} of the Half-t
#'   prior for \eqn{\sqrt{W_1}}. Default `1` (Half-Cauchy). Must equal `1` when
#'   `prior_prec1_type = "halfcauchy"`.
#' @param prior_prec_y_type Character, prior on the observation precision
#'   \eqn{1/V}: `"halfcauchy"` (default) or `"halft"` for a Half-t /
#'   Half-Cauchy prior on the corresponding standard deviation (Gelman, 2006),
#'   or `"gamma"` for a Gamma prior on the precision. The default changed in
#'   0.4-0; a call that supplies `prior_prec_y_shape`/`prior_prec_y_rate`
#'   without naming a type is still read as `"gamma"`, so existing code keeps
#'   working.
#' @param prior_prec_y_shape Numeric > 0, shape parameter \eqn{\nu_V} of the Gamma
#'   prior for \eqn{1/V}. Required (and used) only when `prior_prec_y_type = "gamma"`.
#' @param prior_prec_y_rate Numeric > 0, rate parameter \eqn{\eta_V} of the Gamma
#'   prior for \eqn{1/V}. Required (and used) only when `prior_prec_y_type = "gamma"`.
#' @param prior_prec_y_scale Numeric > 0, scale \eqn{A} of the Half-t prior for
#'   \eqn{\sqrt{V}}. If `NULL` (default), derived from the data as
#'   `sd(y)`, which makes the fit invariant to
#'   the units of `y`. Ignored when `prior_prec_y_type = "gamma"`.
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
#'   When `chains > 1` the return value is instead an object of class
#'   `c("pdm_mcmc_list", "list")` holding one such fit per chain; see
#'   \code{\link{print.pdm_mcmc_list}}.
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
#' # 1. Simulate data from a local-level dynamic model
#' # 2. Use `mcmc_normal_locallevel` to estimate parameters and latent states
#' # 3. Perform a detailed posterior analysis with visualizations
#' # 4. Set a seed for reproducibility
#'
#' ## Simulation of data
#' n <- 200   # Number of observations to simulate
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
#'   n_chain            = 500,
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
#'   n_chain            = 500,
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
                                   prior_prec1_type = c("halfcauchy", "gamma", "halft"),
                                   prior_prec1_scale = NULL,
                                   prior_prec1_df = 1,
                                   prior_prec_y_type = c("halfcauchy", "gamma", "halft"),
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

  # Record whether the user explicitly set the degrees of freedom (used to flag
  # a contradictory `type = "halfcauchy"` + `df != 1`). `missing()` must be read
  # before these arguments are touched.
  prec1_df_user_set  <- !missing(prior_prec1_df)
  prec_y_df_user_set <- !missing(prior_prec_y_df)

  # Same reason: whether the type was named decides how a bare shape/rate pair
  # is read (see infer_gamma_from_hyperparams() in R/innovation_priors.R).
  prec1_type_user_set  <- !missing(prior_prec1_type)
  prec_y_type_user_set <- !missing(prior_prec_y_type)

  prior_prec1_type  <- match.arg(prior_prec1_type)
  prior_prec_y_type <- match.arg(prior_prec_y_type)

  prior_prec1_type <- infer_gamma_from_hyperparams(
    prior_prec1_type, prec1_type_user_set,
    !missing(prior_prec1_shape) || !missing(prior_prec1_rate)
  )
  prior_prec_y_type <- infer_gamma_from_hyperparams(
    prior_prec_y_type, prec_y_type_user_set,
    !missing(prior_prec_y_shape) || !missing(prior_prec_y_rate)
  )
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
  # --- Data-scaled Half-t defaults ---
  # A scale left NULL is filled in from the data (see R/innovation_priors.R),
  # which is what makes the fit invariant to the units of `y`.
  if (prior_prec1_type != "gamma" && is.null(prior_prec1_scale)) {
    prior_prec1_scale <- innovation_prior_scale(y, 1)
  }
  if (prior_prec_y_type != "gamma" && is.null(prior_prec_y_scale)) {
    prior_prec_y_scale <- observation_prior_scale(y)
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

  # Record what produced this fit: version, seed and every resolved prior
  # (see R/provenance.R). Must come after all defaults are filled in.
  result <- record_provenance(result, seed)

  return(result)
}
