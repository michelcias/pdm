#' @title Gibbs Sampler for a Local-Trend Binomial Dynamic Model
#'
#' @description Runs a Gibbs sampler for the local-trend binomial dynamic model
#'   with logit link.
#'
#' @details The model is defined as:
#' \deqn{
#' \begin{aligned}
#' y_t &\sim \text{Binomial}(n_{trials}, \alpha_t), \
#' \alpha_t &= \text{logit}^{-1}(\theta_{t,1}), \
#' \theta_{t,1} &= \theta_{t-1,1} + \theta_{t-1,2} + u_{t,1}, \quad u_{t,1} \sim N(0, W_1), \
#' \theta_{t,2} &= \theta_{t-1,2} + u_{t,2}, \quad u_{t,2} \sim N(0, W_2),
#' \end{aligned}
#' }
#' where \eqn{t = 1, 2, \ldots, n} and \eqn{n} is the number of observations.
#'
#' The logit link function is defined as
#' \eqn{\text{logit}^{-1}(x) = \frac{e^x}{1 + e^x}}.
#'
#' \strong{Prior Distributions:}
#'
#' \emph{Initial States:}
#' \deqn{
#' \begin{aligned}
#' \theta_{0,1} &\sim N(\mu_{01}, \tau_{01}^{-1}), \
#' \theta_{0,2} &\sim N(\mu_{02}, \tau_{02}^{-1}).
#' \end{aligned}
#' }
#'
#' \emph{Precision Parameters:}
#' \deqn{
#' \begin{aligned}
#' W_1^{-1} &\sim \text{Gamma}(\nu_1, \eta_1), \
#' W_2^{-1} &\sim \text{Gamma}(\nu_2, \eta_2).
#' \end{aligned}
#' }
#'
#' The algorithm employs component-wise Metropolis-Hastings for sampling the latent states
#' \eqn{\theta_{t,1}} and \eqn{\theta_{t,2}}, with adaptive proposal tuning based on
#' acceptance rates. Innovation precisions are sampled from conjugate Gamma posteriors.
#'
#' Burn‐in and thinning are applied so that exactly `n_chain` posterior samples are returned.
#'
#' @param y Numeric vector of observed binomial counts (length \eqn{n}). Each
#'   element must satisfy \eqn{0 \leq y_t \leq n_{trials}}.
#' @param n_trials Numeric scalar > 0, number of trials for each binomial observation.
#' @param burnin Integer \eqn{\geq 0}, number of burn-in iterations.
#' @param thinning Integer \eqn{\geq 1}, thinning interval.
#' @param n_chain Integer \eqn{\geq 1}, number of posterior samples to retain.
#' @param prior_theta01_mean Numeric, prior mean for the initial state \eqn{\theta_{0,1}}.
#' @param prior_theta01_prec Numeric > 0, prior precision (inverse variance) for \eqn{\theta_{0,1}}.
#' @param prior_theta02_mean Numeric, prior mean for the initial state \eqn{\theta_{0,2}}.
#' @param prior_theta02_prec Numeric > 0, prior precision (inverse variance) for \eqn{\theta_{0,2}}.
#' @param prior_prec1_shape Numeric > 0, shape parameter of the Gamma prior for \eqn{1/W_1}.
#' @param prior_prec1_rate Numeric > 0, rate parameter of the Gamma prior for \eqn{1/W_1}.
#' @param prior_prec2_shape Numeric > 0, shape parameter of the Gamma prior for \eqn{1/W_2}.
#' @param prior_prec2_rate Numeric > 0, rate parameter of the Gamma prior for \eqn{1/W_2}.
#' @param lag_update Integer \eqn{\geq 1}, adaptation frequency for Metropolis-Hastings proposals (iterations).
#' @param max_step_size Numeric > 0, maximum proposal step size for adaptive algorithm.
#' @param base_adaptation_rate Numeric > 0, base adaptation rate for proposal scaling.
#' @param decay_exponent Numeric > 0, adaptation decay exponent for diminishing adaptation.
#' @param target_acceptance Numeric in (0,1), target acceptance rate for Metropolis-Hastings.
#' @param return_log_sigma Logical, whether to return proposal scale diagnostics. Default is `FALSE`.
#' @param return_accrate Logical, whether to return acceptance rate diagnostics. Default is `FALSE`.
#' @param seed Optional integer used to set the random number generator seed. Default is `NULL`.
#'
#' @return A list with components:
#' \describe{
#'   \item{`theta_1`}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior samples for \eqn{\theta_{t,1}}.}
#'   \item{`theta_2`}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior samples for \eqn{\theta_{t,2}}.}
#'   \item{`theta_01`}{Numeric vector of length `n_chain` of posterior samples for \eqn{\theta_{0,1}}.}
#'   \item{`theta_02`}{Numeric vector of length `n_chain` of posterior samples for \eqn{\theta_{0,2}}.}
#'   \item{`prec_1`}{Numeric vector of length `n_chain` of posterior samples for \eqn{1/W_1}.}
#'   \item{`prec_2`}{Numeric vector of length `n_chain` of posterior samples for \eqn{1/W_2}.}
#'   \item{`alpha`}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior samples for \eqn{\alpha_t}.}
#'   \item{`log_sigma`}{Numeric matrix \eqn{[n_{chain} \times n]} of proposal scale diagnostics (if requested).}
#'   \item{`accrate`}{Numeric matrix \eqn{[n_{chain} \times n]} of acceptance rate diagnostics (if requested).}
#' }
#'
#' @seealso \link[pdm]{mcmc_localtrend}
#' @export
mcmc_binomial_localtrend <- function(y,
                                     n_trials,
                                     burnin,
                                     thinning,
                                     n_chain,
                                     prior_theta01_mean,
                                     prior_theta01_prec,
                                     prior_theta02_mean,
                                     prior_theta02_prec,
                                     prior_prec1_shape,
                                     prior_prec1_rate,
                                     prior_prec2_shape,
                                     prior_prec2_rate,
                                     lag_update = 50,
                                     max_step_size = 2.0,
                                     base_adaptation_rate = 0.01,
                                     decay_exponent = 0.6,
                                     target_acceptance = 0.44,
                                     return_log_sigma = FALSE,
                                     return_accrate = FALSE,
                                     seed = NULL) {

  # --- Input Validation ---
  if (!is.numeric(y)) stop("`y` must be a numeric vector")
  if (!all(is.finite(y))) stop("`y` must contain only finite numeric values (no NA, NaN, Inf)")
  if (!is.numeric(n_trials) || length(n_trials) != 1 || n_trials <= 0) {
    stop("`n_trials` must be a single positive numeric value")
  }
  if (any(y < 0 | y > n_trials)) stop("`y` values must satisfy 0 <= y <= n_trials")

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
  if (!is.numeric(prior_theta02_mean) || length(prior_theta02_mean) != 1) {
    stop("`prior_theta02_mean` must be a single numeric value")
  }
  if (!is.numeric(prior_theta02_prec) || length(prior_theta02_prec) != 1 || prior_theta02_prec <= 0) {
    stop("`prior_theta02_prec` must be a single positive numeric value")
  }

  if (!is.numeric(prior_prec1_shape) || length(prior_prec1_shape) != 1 || prior_prec1_shape <= 0) {
    stop("`prior_prec1_shape` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec1_rate) || length(prior_prec1_rate) != 1 || prior_prec1_rate <= 0) {
    stop("`prior_prec1_rate` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec2_shape) || length(prior_prec2_shape) != 1 || prior_prec2_shape <= 0) {
    stop("`prior_prec2_shape` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec2_rate) || length(prior_prec2_rate) != 1 || prior_prec2_rate <= 0) {
    stop("`prior_prec2_rate` must be a single positive numeric value")
  }

  if (!is.numeric(lag_update) || length(lag_update) != 1 || lag_update < 1 || lag_update != floor(lag_update)) {
    stop("`lag_update` must be a single positive integer")
  }
  if (!is.numeric(max_step_size) || length(max_step_size) != 1 || max_step_size <= 0) {
    stop("`max_step_size` must be a single positive numeric value")
  }
  if (!is.numeric(base_adaptation_rate) || length(base_adaptation_rate) != 1 || base_adaptation_rate <= 0) {
    stop("`base_adaptation_rate` must be a single positive numeric value")
  }
  if (!is.numeric(decay_exponent) || length(decay_exponent) != 1 || decay_exponent <= 0) {
    stop("`decay_exponent` must be a single positive numeric value")
  }
  if (!is.numeric(target_acceptance) || length(target_acceptance) != 1 ||
      target_acceptance <= 0 || target_acceptance >= 1) {
    stop("`target_acceptance` must be a single numeric value in (0,1)")
  }

  if (!is.logical(return_log_sigma) || length(return_log_sigma) != 1) {
    stop("`return_log_sigma` must be a single logical value")
  }
  if (!is.logical(return_accrate) || length(return_accrate) != 1) {
    stop("`return_accrate` must be a single logical value")
  }

  if (!is.null(seed)) {
    if (!is.numeric(seed) || length(seed) != 1 || seed != floor(seed)) {
      stop("`seed` must be a single integer value")
    }
    set.seed(seed)
  }
  # --- End Input Validation ---

  # Call the C function
  .Call(
    "_pdm_C_MCMC_logit_binomial_localtrend",
    as.numeric(y),
    as.numeric(n_trials),
    as.integer(burnin),
    as.integer(thinning),
    as.integer(n_chain),
    as.numeric(prior_theta01_mean),
    as.numeric(prior_theta01_prec),
    as.numeric(prior_theta02_mean),
    as.numeric(prior_theta02_prec),
    as.numeric(prior_prec1_shape),
    as.numeric(prior_prec1_rate),
    as.numeric(prior_prec2_shape),
    as.numeric(prior_prec2_rate),
    as.integer(lag_update),
    as.numeric(max_step_size),
    as.numeric(base_adaptation_rate),
    as.numeric(decay_exponent),
    as.numeric(target_acceptance),
    as.logical(return_log_sigma),
    as.logical(return_accrate)
  )
}