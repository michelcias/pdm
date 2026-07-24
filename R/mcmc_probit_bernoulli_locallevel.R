#' @title Gibbs Sampler for a Local-Level Bernoulli Dynamic Model with Probit Link
#'
#' @description Runs a Gibbs sampler for the local-level Bernoulli dynamic model
#'   with probit link using Albert-Chib data augmentation.
#'
#' @details The model is defined as:
#' \deqn{
#' \begin{aligned}
#' y_t &\sim \text{Bernoulli}(\alpha_t), \\
#' \alpha_t &= \Phi(\theta_{t,1}), \\
#' \theta_{t,1} &= \theta_{t-1,1} + u_{t,1}, & u_{t,1} \sim N(0, W_1),
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
#' update closed-form via a single auxiliary variable. This probit-Bernoulli
#' model has no Gaussian observation variance, so there is no observation
#' precision \eqn{1/V}.
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
#' The algorithm employs the Albert-Chib (1993) latent variable augmentation
#' scheme to obtain fully conjugate Gibbs updates for all parameters. This approach
#' introduces auxiliary Gaussian latent variables whose signs correspond to the
#' observed binary outcomes, yielding closed-form conditional distributions for
#' the latent states \eqn{\theta_{t,1}}, the initial state \eqn{\theta_{0,1}},
#' and the innovation precision \eqn{1/W_1}.
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
#' Burn-in and thinning are applied so that exactly `n_chain` posterior
#' samples are returned.
#'
#' @param y Numeric vector of observed Bernoulli outcomes (length \eqn{n}). Each
#'   element must be either 0 or 1.
#' @param burnin Integer \eqn{\geq 0}, number of burn-in iterations.
#' @param thinning Integer \eqn{\geq 1}, thinning interval.
#' @param n_chain Integer \eqn{\geq 1}, number of posterior samples to retain.
#' @param prior_theta01_mean Numeric, prior mean for the initial state
#'   \eqn{\theta_{0,1}}.
#' @param prior_theta01_prec Numeric > 0, prior precision (inverse variance)
#'   for \eqn{\theta_{0,1}}.
#' @param prior_prec1_type Character, prior on the innovation precision
#'   \eqn{1/W_1}: `"gamma"` (default) for a Gamma prior on the precision, or
#'   `"halft"` / `"halfcauchy"` for a Half-t / Half-Cauchy prior on
#'   \eqn{\sqrt{W_1}} (Gelman, 2006). `"halfcauchy"` is Half-t with `df = 1`.
#' @param prior_prec1_shape Numeric > 0, shape parameter of the Gamma prior
#'   for \eqn{1/W_1}. Required (and used) only when `prior_prec1_type = "gamma"`.
#' @param prior_prec1_rate Numeric > 0, rate parameter of the Gamma prior
#'   for \eqn{1/W_1}. Required (and used) only when `prior_prec1_type = "gamma"`.
#' @param prior_prec1_scale Numeric > 0, scale \eqn{A_1} of the Half-t prior for
#'   \eqn{\sqrt{W_1}}. Required when `prior_prec1_type` is `"halft"` or
#'   `"halfcauchy"`; ignored otherwise.
#' @param prior_prec1_df Numeric > 0, degrees of freedom \eqn{\nu_1} of the Half-t
#'   prior for \eqn{\sqrt{W_1}}. Default `1` (Half-Cauchy). Must equal `1` when
#'   `prior_prec1_type = "halfcauchy"`.
#' @param verbose Logical, whether to display a progress bar during sampling. Default is `FALSE`.
#' @param bar_width Integer in \[10, 120\], width of the progress bar when `verbose = TRUE`. Default is `60`.
#' @param seed Optional integer used to set the random number generator seed.
#'   Default is `NULL`, which does not set the seed.
#'
#' @return A list with components:
#' \describe{
#'   \item{\code{theta_1}}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior
#'     samples for the latent state \eqn{\theta_{t,1}}.}
#'   \item{\code{theta_01}}{Numeric vector of length `n_chain` of posterior samples
#'     for the initial state \eqn{\theta_{0,1}}.}
#'   \item{\code{prec_theta1}}{Numeric vector of length `n_chain` of posterior samples
#'     for the innovation precision \eqn{1/W_1}.}
#'   \item{\code{alpha}}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior
#'     samples for the Bernoulli probabilities \eqn{\alpha_t}.}
#' }
#'
#' @examples
#' ## Description
#' # This example demonstrates how to:
#' # 1. Simulate data from a sinusoidal probability pattern
#' # 2. Use `mcmc_probit_bernoulli_locallevel` to estimate parameters and latent states
#' # 3. Perform a detailed posterior analysis with visualizations
#' # 4. Set a seed for reproducibility
#'
#' ## Simulation of data
#' n <- 500  # Number of observations to simulate
#'
#' # Use a fixed seed for data simulation
#' set.seed(123)
#'
#' # Generate true success probabilities following a sinusoidal pattern
#' alpha_true <- (sin(2 * pi * seq_len(n) / n) + 2) / 4
#'
#' # Generate Bernoulli observations
#' y <- rbinom(n, size = 1, prob = alpha_true)
#'
#' ## Running the Gibbs sampler
#' out <- mcmc_probit_bernoulli_locallevel(
#'   y,
#'   burnin             = 1000,
#'   thinning           = 50,
#'   n_chain            = 1000,
#'   prior_theta01_mean = 0,
#'   prior_theta01_prec = 1,
#'   prior_prec1_shape  = 100,
#'   prior_prec1_rate   = 1,
#'   verbose            = TRUE,
#'   bar_width          = 60,
#'   seed               = 456
#' )
#'
#' ## Alternative: weakly-informative Half-Cauchy prior (Gelman, 2006)
#' # Replace the Gamma prior on the innovation precision with a Half-Cauchy on
#' # the innovation SD sqrt(W[1]); the Gamma shape/rate are then unused.
#' out_hc <- mcmc_probit_bernoulli_locallevel(
#'   y,
#'   burnin             = 1000,
#'   thinning           = 50,
#'   n_chain            = 1000,
#'   prior_theta01_mean = 0,
#'   prior_theta01_prec = 1,
#'   prior_prec1_type   = "halfcauchy",  # Half-Cauchy on sqrt(W[1])
#'   prior_prec1_scale  = 1,             # scale A_1 > 0
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
#'   plot(out, type = "mcmc", which = 1)      # Initial level (theta_01)
#'   plot(out, type = "mcmc", which = 2)      # Innovation precision (W_1^-1)
#'   plot(out, type = "states")               # Dynamic level state
#'   plot(out, type = "alpha")                # Bernoulli probabilities
#'
#'   # Plot with true probabilities for comparison
#'   plot(out, type = "alpha", true_values = list(alpha = alpha_true))
#'
#'   # --- Half-Cauchy-prior fit (`out_hc`) ---
#'   # The same diagnostics for the weakly-informative Half-Cauchy fit.
#'   plot(out_hc, type = "all")
#'
#'   plot(out_hc, type = "states")            # Dynamic level state
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
#'   \code{\link{plot.probit_bernoulli_locallevel}},
#'   \code{\link{print.probit_bernoulli_locallevel}},
#'   \code{\link{summary.probit_bernoulli_locallevel}} for methods on the fitted object;
#'   \code{\link{mcmc_probit_bernoulli_localtrend}} and
#'   \code{\link{mcmc_probit_bernoulli_localacceleration}} for the other dynamic orders.
#'
#' @export
mcmc_probit_bernoulli_locallevel <- function(y,
                                             burnin,
                                             thinning,
                                             n_chain,
                                             prior_theta01_mean,
                                             prior_theta01_prec,
                                             prior_prec1_shape = NULL,
                                             prior_prec1_rate = NULL,
                                             verbose = FALSE,
                                             bar_width = 60,
                                             seed = NULL,
                                             prior_prec1_type = c("gamma", "halfcauchy", "halft"),
                                             prior_prec1_scale = NULL,
                                             prior_prec1_df = 1) {

  # `missing()` must be read before the argument is touched.
  prec1_df_user_set <- !missing(prior_prec1_df)
  prior_prec1_type  <- match.arg(prior_prec1_type)
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
    "_pdm_C_MCMC_probit_bernoulli_locallevel",
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
    as.logical(verbose),
    as.integer(bar_width)
  )

  result <- new_probit_bernoulli_locallevel(
    result = result,
    n_obs = length(y),
    n_chain = n_chain,
    burnin = burnin,
    thinning = thinning,
    y = y
  )

  result <- validate_probit_bernoulli_locallevel(result)

  # Record the prior used on the innovation precision (auxiliary Half-t variable
  # is a nuisance parameter and is intentionally not returned).
  attr(result, "prior_prec_theta1") <- prec1_prior[c("type", "shape", "rate", "scale", "df")]

  return(result)
}
