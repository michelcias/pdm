#' @title Gibbs Sampler for a Local-Trend Bernoulli Dynamic Model with Probit Link
#'
#' @description Runs a Gibbs sampler for the local-trend Bernoulli dynamic model
#'   with probit link using Albert-Chib data augmentation.
#'
#' @details The model is defined as:
#' \deqn{
#' \begin{aligned}
#' y_t &\sim \text{Bernoulli}(\alpha_t), \\
#' \alpha_t &= \Phi(\theta_{t,1}), \\
#' \theta_{t,1} &= \theta_{t-1,1} + \theta_{t-1,2} + u_{t,1}, & u_{t,1} \sim N(0, W_1), \\
#' \theta_{t,2} &= \theta_{t-1,2} + u_{t,2},                  & u_{t,2} \sim N(0, W_2),
#' \end{aligned}
#' }
#' where \eqn{t = 1, 2, \ldots, n}, \eqn{n} is the number of observations, and
#' \eqn{\Phi(\cdot)} denotes the standard normal cumulative distribution function.
#'
#' The probit link function is defined as \eqn{\alpha_t = \Phi(\theta_{t,1})}.
#'
#' \strong{Prior Distributions:}
#'
#' The following conjugate prior distributions are employed:
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
#' Each innovation precision (\eqn{W_1^{-1}}, \eqn{W_2^{-1}}) may take one of two
#' priors, chosen independently through `prior_prec1_type` and `prior_prec2_type`:
#'
#' \emph{(a) Gamma prior on the precision} (default, semi-conjugate):
#' \deqn{W_1^{-1} \sim \text{Gamma}(\nu_1, \eta_1), \qquad
#'       W_2^{-1} \sim \text{Gamma}(\nu_2, \eta_2).}
#'
#' \emph{(b) Half-t prior on the standard deviation} (Gelman, 2006):
#' \deqn{\sqrt{W_1} \sim \text{Half-}t(\nu_1, A_1), \qquad
#'       \sqrt{W_2} \sim \text{Half-}t(\nu_2, A_2),}
#' where \eqn{A > 0} is a scale hyperparameter and \eqn{\nu > 0} the degrees of
#' freedom. Setting \eqn{\nu = 1} gives the Half-Cauchy prior (also selectable
#' directly with `type = "halfcauchy"`); larger \eqn{\nu} approaches a
#' Half-Normal. The Half-t prior is represented by the inverse-gamma scale
#' mixture of Wand et al. (2011), which keeps each innovation-precision Gibbs
#' update closed-form via a single auxiliary variable. This probit-Bernoulli
#' model has no Gaussian observation variance, so there is no observation
#' precision \eqn{1/V}.
#'
#' The Gamma shape/rate arguments are required only when the corresponding
#' `_type` is `"gamma"`; the Half-t scale/df arguments only when it is `"halft"`
#' or `"halfcauchy"`. The prior choice is resolved once, before the sampler runs,
#' and never re-evaluated inside the MCMC loop.
#'
#' The prior hyperparameters correspond to function arguments as follows:
#' \tabular{cc}{
#'   \strong{Hyperparameter} \tab \strong{Function Argument} \cr
#'   \eqn{\mu_{01}} \tab `prior_theta01_mean` \cr
#'   \eqn{\tau_{01}} \tab `prior_theta01_prec` \cr
#'   \eqn{\mu_{02}} \tab `prior_theta02_mean` \cr
#'   \eqn{\tau_{02}} \tab `prior_theta02_prec` \cr
#'   \eqn{\nu_1} (Gamma shape), \eqn{A_1} (Half-t scale) \tab `prior_prec1_shape`, `prior_prec1_scale` \cr
#'   \eqn{\eta_1} (Gamma rate), \eqn{\nu_1} (Half-t df) \tab `prior_prec1_rate`, `prior_prec1_df` \cr
#'   \eqn{\nu_2} (Gamma shape), \eqn{A_2} (Half-t scale) \tab `prior_prec2_shape`, `prior_prec2_scale` \cr
#'   \eqn{\eta_2} (Gamma rate), \eqn{\nu_2} (Half-t df) \tab `prior_prec2_rate`, `prior_prec2_df`
#' }
#'
#' The algorithm employs the Albert-Chib (1993) latent variable augmentation
#' scheme to obtain fully conjugate Gibbs updates for all parameters. This approach
#' introduces auxiliary Gaussian latent variables whose signs correspond to the
#' observed binary outcomes, yielding closed-form conditional distributions for
#' the latent states \eqn{\theta_{t,1}} and \eqn{\theta_{t,2}}, the initial states
#' \eqn{\theta_{0,1}} and \eqn{\theta_{0,2}}, and the innovation precisions
#' \eqn{1/W_1} and \eqn{1/W_2}.
#'
#' Since this is a pure Gibbs sampler with no Metropolis-Hastings steps, the
#' acceptance rate is always 1.0, eliminating the need for proposal tuning
#' parameters.
#'
#' \strong{Progress Bar:}
#' When `verbose = TRUE`, a visual progress bar is displayed showing
#' \itemize{
#' \item{Progress bar with adaptive update frequency (based on `bar_width`)}
#' \item{Elapsed time in HH:MM:SS format}
#' \item{Estimated remaining time in HH:MM:SS format}
#' }
#'
#' The progress bar update frequency is automatically calculated as approximately
#' one update per bar segment, ensuring smooth visual feedback with minimal
#' performance overhead (~0.01% for typical runs).
#'
#' Burn-in and thinning are applied so that exactly `n_draws` posterior
#' samples are returned.
#'
#' @param y Numeric vector of observed Bernoulli outcomes (length \eqn{n}). Each
#'   element must be either 0 or 1.
#' @param burnin Integer \eqn{\geq 0}, number of burn-in iterations.
#' @param thinning Integer \eqn{\geq 1}, thinning interval.
#' @param n_draws Integer \eqn{\geq 1}, number of posterior samples to retain.
#' @param prior_theta01_mean Numeric, prior mean for the initial state \eqn{\theta_{0,1}}.
#' @param prior_theta01_prec Numeric > 0, prior precision (inverse variance) for \eqn{\theta_{0,1}}.
#' @param prior_theta02_mean Numeric, prior mean for the initial state \eqn{\theta_{0,2}}.
#' @param prior_theta02_prec Numeric > 0, prior precision (inverse variance) for \eqn{\theta_{0,2}}.
#' @param prior_prec1_type Character, prior on the level innovation
#'   precision \eqn{1/W_1}: `"halfcauchy"` (default) or `"halft"` for a Half-t /
#'   Half-Cauchy prior on \eqn{\sqrt{W_1}} (Gelman, 2006), or `"gamma"` for a
#'   Gamma prior on the precision. The default changed in 0.5-0; a call that
#'   supplies `prior_prec1_shape`/`prior_prec1_rate` without naming a type is
#'   still read as `"gamma"`, so existing code keeps working.
#' @param prior_prec1_shape Numeric > 0, shape parameter of the Gamma prior for \eqn{1/W_1}. Required (and used) only when `prior_prec1_type = "gamma"`.
#' @param prior_prec1_rate Numeric > 0, rate parameter of the Gamma prior for \eqn{1/W_1}. Required (and used) only when `prior_prec1_type = "gamma"`.
#' @param prior_prec1_scale Numeric > 0, scale \eqn{A} of the Half-t prior for
#'   \eqn{\sqrt{W_1}}. Default `2`, on the scale of the link -- a fixed value
#'   is enough here, unlike the Gaussian family, because the link scale carries
#'   no arbitrary units. Ignored when `prior_prec1_type = "gamma"`.
#' @param prior_prec1_df Numeric > 0, degrees of freedom \eqn{\nu_1} of the Half-t
#'   prior for \eqn{\sqrt{W_1}}. Default `1` (Half-Cauchy).
#' @param prior_prec2_type Character, prior on the trend innovation
#'   precision \eqn{1/W_2}: `"halfcauchy"` (default) or `"halft"` for a Half-t /
#'   Half-Cauchy prior on \eqn{\sqrt{W_2}} (Gelman, 2006), or `"gamma"` for a
#'   Gamma prior on the precision. The default changed in 0.5-0; a call that
#'   supplies `prior_prec2_shape`/`prior_prec2_rate` without naming a type is
#'   still read as `"gamma"`, so existing code keeps working.
#' @param prior_prec2_shape Numeric > 0, shape parameter of the Gamma prior for \eqn{1/W_2}. Required (and used) only when `prior_prec2_type = "gamma"`.
#' @param prior_prec2_rate Numeric > 0, rate parameter of the Gamma prior for \eqn{1/W_2}. Required (and used) only when `prior_prec2_type = "gamma"`.
#' @param prior_prec2_scale Numeric > 0, scale \eqn{A} of the Half-t prior for
#'   \eqn{\sqrt{W_2}}. Default `2`, on the scale of the link -- a fixed value
#'   is enough here, unlike the Gaussian family, because the link scale carries
#'   no arbitrary units. Ignored when `prior_prec2_type = "gamma"`.
#' @param prior_prec2_df Numeric > 0, degrees of freedom \eqn{\nu_2} of the Half-t
#'   prior for \eqn{\sqrt{W_2}}. Default `1` (Half-Cauchy).
#' @param verbose Logical, whether to display a progress bar during sampling. Default is `FALSE`.
#' @param bar_width Integer in \[10, 120\], width of the progress bar when `verbose = TRUE`. Default is `60`.
#' @param seed Optional integer used to set the random number generator seed.
#'   Default is `NULL`, which does not set the seed.
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
#' @return A list with components:
#' \describe{
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
#'     samples for the Bernoulli probabilities \eqn{\alpha_t}.}
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
#' # 1. Simulate data from a sinusoidal probability pattern
#' # 2. Use `mcmc_probit_bernoulli_localtrend` to estimate parameters and latent states
#' # 3. Perform a detailed posterior analysis with visualizations
#' # 4. Set a seed for reproducibility
#' # 5. Use progress bar for monitoring MCMC execution
#'
#' ## Simulation of data
#' n <- 200  # Number of observations to simulate
#'
#' # Use a fixed seed for data simulation
#' set.seed(123)
#'
#' # Generate true success probabilities following a sinusoidal pattern
#' grid_vals <- seq_len(n) / n
#' alpha_true <- (sin(2 * pi * grid_vals) + sin(4 * pi * grid_vals) + 2) / 4
#'
#' # Generate Bernoulli observations
#' y <- rbinom(n, size = 1, prob = alpha_true)
#'
#' ## Running the Gibbs sampler with progress bar
#' # NOTE: these MCMC controls are sized to keep the example quick, and are
#' # not guaranteed to be enough for convergence -- how much burn-in and
#' # thinning a fit needs depends on the data and on the family. For real
#' # work, fit with chains > 1 and read mcmc_convergence() before trusting
#' # any summary.
#' out <- mcmc_probit_bernoulli_localtrend(
#'   y,
#'   burnin             = 1000,
#'   thinning           = 20,
#'   n_draws            = 500,
#'   prior_theta01_mean = 0,
#'   prior_theta01_prec = 1,
#'   prior_theta02_mean = 0,
#'   prior_theta02_prec = 1,
#'   prior_prec1_shape  = 100,
#'   prior_prec1_rate   = 1,
#'   prior_prec2_shape  = 400,
#'   prior_prec2_rate   = 1,
#'   verbose            = TRUE,
#'   bar_width          = 60,
#'   seed               = 456
#' )
#'
#' ## Alternative: weakly-informative Half-Cauchy priors (Gelman, 2006)
#' # Put a Half-Cauchy prior on the level and trend innovation SDs sqrt(W[1]),
#' # sqrt(W[2]). Each precision's prior is chosen independently via its `*_type`.
#' out_hc <- mcmc_probit_bernoulli_localtrend(
#'   y,
#'   burnin             = 1000,
#'   thinning           = 20,
#'   n_draws            = 500,
#'   prior_theta01_mean = 0,
#'   prior_theta01_prec = 1,
#'   prior_theta02_mean = 0,
#'   prior_theta02_prec = 1,
#'   prior_prec1_type   = "halfcauchy",
#'   prior_prec1_scale  = 1,  # sqrt(W[1])
#'   prior_prec2_type   = "halfcauchy",
#'   prior_prec2_scale  = 1,  # sqrt(W[2])
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
#'   plot(out, type = "mcmc", which = 1:2)    # Initial states (theta_0)
#'   plot(out, type = "mcmc", which = 3:4)    # Innovation precisions (W^-1)
#'   plot(out, type = "states")               # Dynamic states
#'   plot(out, type = "alpha")                # Bernoulli probabilities
#'
#'   # Plot with true probabilities for comparison
#'   plot(out, type = "alpha", true_values = list(alpha = alpha_true))
#'
#'   # --- Half-Cauchy-prior fit (`out_hc`) ---
#'   # The same diagnostics for the weakly-informative Half-Cauchy fit.
#'   plot(out_hc, type = "all")
#'
#'   plot(out_hc, type = "states")            # Dynamic states
#'
#'   plot(out_hc, type = "alpha", true_values = list(alpha = alpha_true))
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
#' Wand, M. P., Ormerod, J. T., Padoan, S. A., & Fruhwirth, R. (2011). Mean field
#' variational Bayes for elaborate distributions. \emph{Bayesian Analysis},
#' 6(4), 847-900.
#'
#' @seealso
#'   \code{\link{plot.probit_bernoulli_localtrend}},
#'   \code{\link{print.probit_bernoulli_localtrend}},
#'   \code{\link{summary.probit_bernoulli_localtrend}} for methods on the fitted object;
#'   \code{\link{mcmc_probit_bernoulli_locallevel}} and
#'   \code{\link{mcmc_probit_bernoulli_localacceleration}} for the other dynamic orders.
#'
#' @export
mcmc_probit_bernoulli_localtrend <- function(y,
                                             burnin,
                                             thinning,
                                             n_draws,
                                             prior_theta01_mean,
                                             prior_theta01_prec,
                                             prior_theta02_mean,
                                             prior_theta02_prec,
                                             prior_prec1_shape = NULL,
                                             prior_prec1_rate = NULL,
                                             prior_prec2_shape = NULL,
                                             prior_prec2_rate = NULL,
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

  # `missing()` must be read before the arguments are touched.
  prec1_df_user_set <- !missing(prior_prec1_df)
  prec2_df_user_set <- !missing(prior_prec2_df)
  # Whether the type was named decides how a bare shape/rate pair is read
  # (see infer_gamma_from_hyperparams() in R/innovation_priors.R).
  prec1_type_user_set <- !missing(prior_prec1_type)
  prec2_type_user_set <- !missing(prior_prec2_type)

  prior_prec1_type  <- match.arg(prior_prec1_type)
  prior_prec2_type  <- match.arg(prior_prec2_type)

  prior_prec1_type <- infer_gamma_from_hyperparams(
    prior_prec1_type, prec1_type_user_set,
    !missing(prior_prec1_shape) || !missing(prior_prec1_rate)
  )

  prior_prec2_type <- infer_gamma_from_hyperparams(
    prior_prec2_type, prec2_type_user_set,
    !missing(prior_prec2_shape) || !missing(prior_prec2_rate)
  )
  # --- Input Validation ---
  if (!is.numeric(y)) {
    stop("`y` must be a numeric vector")
  }
  if (!all(is.finite(y))) {
    stop("`y` must contain only finite numeric values (no NA, NaN, Inf)")
  }

  # Validate Bernoulli support
  if (!all(y %in% c(0, 1))) {
    stop("`y` values must be either 0 or 1 (Bernoulli outcomes)")
  }

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

  # Resolve each innovation precision prior (Gamma or Half-t); see R/prec_prior.R.
  prec1_prior <- resolve_prec_prior(
    prior_prec1_type, prior_prec1_shape, prior_prec1_rate,
    prior_prec1_scale, prior_prec1_df, prec1_df_user_set, "prior_prec1"
  )
  prec2_prior <- resolve_prec_prior(
    prior_prec2_type, prior_prec2_shape, prior_prec2_rate,
    prior_prec2_scale, prior_prec2_df, prec2_df_user_set, "prior_prec2"
  )

  # Validate logical parameters
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
    "_pdm_C_MCMC_probit_bernoulli_localtrend",
    as.numeric(y),
    as.integer(burnin),
    as.integer(thinning),
    as.integer(n_draws),
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
    as.logical(verbose),
    as.integer(bar_width)
  )

  result <- new_probit_bernoulli_localtrend(
    result = result,
    n_obs = length(y),
    n_draws = n_draws,
    burnin = burnin,
    thinning = thinning,
    y = y
  )

  result <- validate_probit_bernoulli_localtrend(result)

  # Record the priors used on each innovation precision (auxiliary Half-t
  # variables are nuisance parameters and are intentionally not returned).
  attr(result, "prior_prec_theta1") <- prec1_prior[c("type", "shape", "rate", "scale", "df")]
  attr(result, "prior_prec_theta2") <- prec2_prior[c("type", "shape", "rate", "scale", "df")]

  # Record what produced this fit: version, seed and every resolved prior
  # (see R/provenance.R). Must come after all defaults are filled in.
  result <- record_provenance(result, seed)

  return(result)
}
