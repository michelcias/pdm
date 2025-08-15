#' @title Gibbs Sampler for a Local-Level Binomial Dynamic Model
#'
#' @description Runs a Gibbs sampler for the local-level binomial dynamic model
#'   with logit link.
#'
#' @details The model is defined as:
#' \deqn{
#' \begin{aligned}
#' y_t &\sim \text{Binomial}(n_{trials}, \alpha_t), \\
#' \alpha_t &= \text{logit}^{-1}(\theta_{t,1}), \\
#' \theta_{t,1} &= \theta_{t-1,1} + u_{t,1}, \quad u_{t,1} \sim N(0, W_1),
#' \end{aligned}
#' }
#' where \eqn{t = 1, 2, \ldots, n} and \eqn{n} is the number of observations.
#'
#' The logit link function is defined as
#' \eqn{\text{logit}^{-1}(x) = \frac{e^x}{1 + e^x}}.
#'
#' \strong{Prior Distributions:}
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
#' \deqn{
#' \begin{aligned}
#' W_1^{-1} &\sim \text{Gamma}(\nu_1, \eta_1).
#' \end{aligned}
#' }
#'
#' The prior hyperparameters correspond to function arguments as follows:
#' \tabular{cc}{
#'   \strong{Hyperparameter} \tab \strong{Function Argument} \cr
#'   \eqn{\mu_{01}} \tab `prior_theta01_mean` \cr
#'   \eqn{\tau_{01}} \tab `prior_theta01_prec` \cr
#'   \eqn{\nu_1} \tab `prior_prec1_shape` \cr
#'   \eqn{\eta_1} \tab `prior_prec1_rate`
#' }
#'
#' Due to the non-linear observation model with logit link, the algorithm
#' employs component-wise Metropolis-Hastings for sampling the latent states
#' \eqn{\theta_{t,1}}, with adaptive proposal tuning based on acceptance rates.
#' The innovation precision is sampled from its conjugate Gamma posterior.
#'
#' Burn‐in and thinning are applied so that exactly `n_chain` posterior
#' samples are returned.
#'
#' @param y Numeric vector of observed binomial counts (length \eqn{n}). Each
#'   element must satisfy \eqn{0 \leq y_t \leq n_{trials}}.
#' @param n_trials Numeric scalar > 0, number of trials for each binomial
#'   observation.
#' @param burnin Integer \eqn{\geq 0}, number of burn-in iterations.
#' @param thinning Integer \eqn{\geq 1}, thinning interval.
#' @param n_chain Integer \eqn{\geq 1}, number of posterior samples to retain.
#' @param prior_theta01_mean Numeric, prior mean for the initial state
#'   \eqn{\theta_{0,1}}.
#' @param prior_theta01_prec Numeric > 0, prior precision (inverse variance)
#'   for \eqn{\theta_{0,1}}.
#' @param prior_prec1_shape Numeric > 0, shape parameter of the Gamma prior
#'   for the innovation precision \eqn{1/W_1}.
#' @param prior_prec1_rate Numeric > 0, rate parameter of the Gamma prior
#'   for \eqn{1/W_1}.
#' @param lag_update Integer \eqn{\geq 1}, adaptation frequency for
#'   Metropolis-Hastings proposals (iterations).
#' @param max_step_size Numeric > 0, maximum proposal step size for adaptive
#'   algorithm.
#' @param base_adaptation_rate Numeric > 0, base adaptation rate for proposal
#'   scaling.
#' @param decay_exponent Numeric > 0, adaptation decay exponent for diminishing
#'   adaptation.
#' @param target_acceptance Numeric in (0,1), target acceptance rate for
#'   Metropolis-Hastings.
#' @param return_log_sigma Logical, whether to return proposal scale
#'   diagnostics. Default is `FALSE`.
#' @param return_accrate Logical, whether to return acceptance rate diagnostics.
#'   Default is `FALSE`.
#' @param seed Optional integer used to set the random number generator seed.
#'   Default is `NULL`, which does not set the seed.
#'
#' @return A list with components:
#' \describe{
#'   \item{`theta_1`}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior
#'     samples for the latent state \eqn{\theta_{t,1}}.}
#'   \item{`theta_01`}{Numeric vector of length `n_chain` of posterior samples
#'     for the initial state \eqn{\theta_{0,1}}.}
#'   \item{`prec_1`}{Numeric vector of length `n_chain` of posterior samples
#'     for the innovation precision \eqn{1/W_1}.}
#'   \item{`alpha`}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior
#'     samples for the success probabilities \eqn{\alpha_t}.}
#'   \item{`log_sigma`}{Numeric matrix \eqn{[n_{chain} \times n]} of proposal
#'     scale diagnostics (only if `return_log_sigma = TRUE`).}
#'   \item{`accrate`}{Numeric matrix \eqn{[n_{chain} \times n]} of acceptance
#'     rate diagnostics (only if `return_accrate = TRUE`).}
#' }
#'
#' @examples
#' ## Description
#' # This example demonstrates how to:
#' # 1. Simulate data from a local-level binomial dynamic model
#' # 2. Use `mcmc_binomial_locallevel` to estimate parameters and latent states
#' # 3. Perform a detailed posterior analysis with visualizations
#' # 4. Set a seed for reproducibility
#'
#' ## Simulation of data
#' n <- 500        # Number of observations to simulate
#' n_trials <- 20  # Number of binomial trials
#'
#' # True parameters for simulation:
#' theta0_true <- 0.5     # Initial state (theta[01]) on logit scale
#' prec1_true <- 100      # Innovation precision (1/W[1])
#'
#' # Use a fixed seed for data simulation
#' set.seed(123)
#'
#' # Generate noise terms:
#' u1 <- rnorm(n, sd = sqrt(1/prec1_true))  # Evolution noise (u1[t])
#'
#' # Simulate latent states and observations:
#' theta1_true <- cumsum(c(theta0_true, u1))[-1]  # theta[t1] series on logit scale
#' alpha_true <- plogis(theta1_true)              # Success probabilities
#' y <- rbinom(n, size = n_trials, prob = alpha_true)  # Observed binomial counts
#'
#' ## Running the Gibbs sampler
#' # Run the Gibbs sampler with specified priors and a seed
#' out <- mcmc_binomial_locallevel(
#'   y,
#'   n_trials             = n_trials,
#'   burnin               = 1000,
#'   thinning             = 50,
#'   n_chain              = 1000,
#'   prior_theta01_mean   = 0,
#'   prior_theta01_prec   = 1,
#'   prior_prec1_shape    = 100,
#'   prior_prec1_rate     = 1,
#'   lag_update           = 50,
#'   max_step_size        = 0.1,
#'   base_adaptation_rate = 1,
#'   decay_exponent       = 0.6,
#'   target_acceptance    = 0.44,
#'   return_log_sigma     = FALSE,
#'   return_accrate       = TRUE,
#'   seed                 = 456
#' )
#'
#' ## Posterior analysis and visualization
#' # The following plots show how to analyze the posterior distributions.
#' # Point estimates are based on the median of posterior samples.
#' \dontrun{
#'   # --- 0. Plot the simulated data ---
#'   plot(
#'     y,
#'     main = "Simulated binomial counts",
#'     ylab = expression(y[t]),
#'     xlab = "t",
#'     type = "o",
#'     pch = 16
#'   )
#'
#'   acc <- out$accrate
#'   min_acc <- apply(X = acc, MARGIN = 2, min)
#'   max_acc <- apply(X = acc, MARGIN = 2, max)
#'   med_acc <- apply(X = acc, MARGIN = 2, median)
#'   sum_acc <- cbind(min_acc, med_acc, max_acc)
#'   matplot(sum_acc, type = "l")
#'   abline(h = 0.44)
#'
#'   # --- 1. Latent State (theta[t1]) on logit scale ---
#'
#'   # Plot true and estimated (median) latent state
#'   theta_1_estimate <- apply(X = out$theta_1, MARGIN = 2, FUN = median)
#'   range_theta_1 <- range(theta_1_estimate, theta1_true)
#'   r1_theta1 <- range_theta_1[1] - 0.1 * diff(range_theta_1)
#'   r2_theta1 <- range_theta_1[2] + 0.1 * diff(range_theta_1)
#'
#'   plot.ts(
#'     theta1_true,
#'     col = "red",
#'     type = "l",
#'     xlab = "t",
#'     ylim = c(r1_theta1, r2_theta1),
#'     lty = 2,
#'     ylab = expression(theta["t,1"]),
#'     main = "Latent state (logit scale)"
#'   )
#'   points(theta_1_estimate, type = "l")
#'   legend(
#'     "topright",
#'     legend = c(expression(theta["t,1"]), expression(hat(theta)["t,1"])),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     bty = "n"
#'   )
#'
#'   # --- 2. Success Probabilities (alpha[t]) ---
#'
#'   # Plot true and estimated (median) success probabilities
#'   alpha_estimate <- apply(X = out$alpha, MARGIN = 2, FUN = median)
#'
#'   plot.ts(
#'     alpha_true,
#'     col = "red",
#'     type = "l",
#'     xlab = "t",
#'     ylim = c(0, 1),
#'     lty = 2,
#'     ylab = expression(alpha[t]),
#'     main = "Success probabilities"
#'   )
#'   points(alpha_estimate, type = "l")
#'   legend(
#'     "topright",
#'     legend = c(expression(alpha[t]), expression(hat(alpha)[t])),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     bty = "n"
#'   )
#'
#'   # --- 3. Initial State (theta[01]) ---
#'
#'   # Trace plot for theta[01]
#'   range_theta_01 <- range(out$theta_01)
#'   r1_theta01 <- range_theta_01[1] - 0.1 * diff(range_theta_01)
#'   r2_theta01 <- range_theta_01[2] + 0.1 * diff(range_theta_01)
#'
#'   plot.ts(
#'     out$theta_01,
#'     ylab = expression(theta["0,1"]),
#'     main = "Trace plot of initial state",
#'     xlab = "Iterations",
#'     col = "gray",
#'     ylim = c(r1_theta01, r2_theta01)
#'   )
#'   abline(
#'     h = c(theta0_true, median(out$theta_01)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'   legend(
#'     "topright",
#'     legend = c(expression(theta["0,1"]), expression(hat(theta)["0,1"])),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     bty = "n",
#'     lwd = 2
#'   )
#'
#'   # --- 4. Innovation Precision (1/W[1]) ---
#'
#'   # Trace plot for 1/W[1]
#'   range_prec_1 <- range(out$prec_1)
#'   r1_prec1 <- range_prec_1[1] - 0.1 * diff(range_prec_1)
#'   r2_prec1 <- range_prec_1[2] + 0.1 * diff(range_prec_1)
#'
#'   plot.ts(
#'     out$prec_1,
#'     ylab = expression(1/W[1]),
#'     main = "Trace plot of innovation precision",
#'     xlab = "Iterations",
#'     col = "gray",
#'     ylim = c(r1_prec1, r2_prec1)
#'   )
#'   abline(
#'     h = c(prec1_true, median(out$prec_1)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'   legend(
#'     "topright",
#'     legend = c(expression(W[1]^-1), expression(hat(W)[1]^-1)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     bty = "n",
#'     lwd = 2
#'   )
#' }
#'
#' @seealso \link[pdm]{mcmc_locallevel}
#' @export
mcmc_binomial_locallevel <- function(y,
                                     n_trials,
                                     burnin,
                                     thinning,
                                     n_chain,
                                     prior_theta01_mean,
                                     prior_theta01_prec,
                                     prior_prec1_shape,
                                     prior_prec1_rate,
                                     lag_update = 50,
                                     max_step_size = 2.0,
                                     base_adaptation_rate = 0.01,
                                     decay_exponent = 0.6,
                                     target_acceptance = 0.44,
                                     return_log_sigma = FALSE,
                                     return_accrate = FALSE,
                                     seed = NULL) {
  # --- Input Validation ---
  if (!is.numeric(y)) {
    stop("`y` must be a numeric vector")
  }
  if (!all(is.finite(y))) {
    stop("`y` must contain only finite numeric values (no NA, NaN, Inf)")
  }
  if (!is.numeric(n_trials) || length(n_trials) != 1 || n_trials <= 0) {
    stop("`n_trials` must be a single positive numeric value")
  }

  # Validate binomial constraints
  if (any(y < 0 | y > n_trials)) {
    stop("`y` values must satisfy 0 <= y <= n_trials")
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
  if (!is.numeric(prior_prec1_shape) || length(prior_prec1_shape) != 1 ||
      prior_prec1_shape <= 0) {
    stop("`prior_prec1_shape` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec1_rate) || length(prior_prec1_rate) != 1 ||
      prior_prec1_rate <= 0) {
    stop("`prior_prec1_rate` must be a single positive numeric value")
  }

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

  # Validate logical parameters
  if (!is.logical(return_log_sigma) || length(return_log_sigma) != 1) {
    stop("`return_log_sigma` must be a single logical value")
  }
  if (!is.logical(return_accrate) || length(return_accrate) != 1) {
    stop("`return_accrate` must be a single logical value")
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
  .Call(
    "_pdm_C_MCMC_logit_binomial_locallevel",
    as.numeric(y),
    as.numeric(n_trials),
    as.integer(burnin),
    as.integer(thinning),
    as.integer(n_chain),
    as.numeric(prior_theta01_mean),
    as.numeric(prior_theta01_prec),
    as.numeric(prior_prec1_shape),
    as.numeric(prior_prec1_rate),
    as.integer(lag_update),
    as.numeric(max_step_size),
    as.numeric(base_adaptation_rate),
    as.numeric(decay_exponent),
    as.numeric(target_acceptance),
    as.logical(return_log_sigma),
    as.logical(return_accrate)
  )
}
