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
#' @param prior_prec1_shape Numeric > 0, shape parameter of the Gamma prior for the level innovation precision \eqn{1/W_1}.
#' @param prior_prec1_rate Numeric > 0, rate parameter of the Gamma prior for \eqn{1/W_1}.
#' @param prior_prec2_shape Numeric > 0, shape parameter of the Gamma prior for the trend innovation precision \eqn{1/W_2}.
#' @param prior_prec2_rate Numeric > 0, rate parameter of the Gamma prior for \eqn{1/W_2}.
#' @param prior_prec_y_shape Numeric > 0, shape parameter of the Gamma prior for the data precision \eqn{1/V}.
#' @param prior_prec_y_rate Numeric > 0, rate parameter of the Gamma prior for \eqn{1/V}.
#' @param seed Optional integer used to set the random number generator seed.
#'   Default is \code{NULL}, which does not set the seed.
#'
#' @return A list with components:
#' \describe{
#'   \item{`theta_1`}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior samples for the level state \eqn{\theta_{t,1}}.}
#'   \item{`theta_2`}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior samples for the trend state \eqn{\theta_{t,2}}.}
#'   \item{`theta_01`}{Numeric vector of length `n_chain` for the initial level \eqn{\theta_{0,1}}.}
#'   \item{`theta_02`}{Numeric vector of length `n_chain` for the initial trend \eqn{\theta_{0,2}}.}
#'   \item{`prec_1`}{Numeric vector of length `n_chain` for the level innovation precision \eqn{1/W_1}.}
#'   \item{`prec_2`}{Numeric vector of length `n_chain` for the trend innovation precision \eqn{1/W_2}.}
#'   \item{`prec_y`}{Numeric vector of length `n_chain` for the data precision \eqn{1/V}.}
#' }
#'
#' @seealso \link[pdm]{mcmc_locallevel}
#' @export
#'
#' @examples
#' ## Description
#' # This example demonstrates how to:
#' # 1. Simulate data from a local trend dynamic model
#' # 2. Use `mcmc_localtrend` to estimate parameters and latent states
#' # 3. Visualize posterior results
#'
#' ## Simulation of data
#' set.seed(123)
#' n <- 150
#'
#' # True parameters
#' theta01_true <- 10
#' theta02_true <- 0.5
#' prec1_true <- 1 / 0.1
#' prec2_true <- 1 / 0.01
#' prec_y_true <- 1 / 0.5
#'
#' # Generate noise terms
#' omega1 <- rnorm(n, sd = sqrt(1 / prec1_true))
#' omega2 <- rnorm(n, sd = sqrt(1 / prec2_true))
#' epsilon <- rnorm(n, sd = sqrt(1 / prec_y_true))
#'
#' # Simulate latent states and observations
#' theta1_true <- numeric(n)
#' theta2_true <- numeric(n)
#' theta2_true[1] <- theta02_true + omega2[1]
#' theta1_true[1] <- theta01_true + theta02_true + omega1[1]
#' for (t in 2:n) {
#'   theta2_true[t] <- theta2_true[t-1] + omega2[t]
#'   theta1_true[t] <- theta1_true[t-1] + theta2_true[t-1] + omega1[t]
#' }
#' y <- theta1_true + epsilon
#'
#' ## Running the Gibbs sampler
#' out <- mcmc_localtrend(
#'   y,
#'   burnin = 2000,
#'   thinning = 20,
#'   n_chain = 1000,
#'   prior_theta01_mean = 0,
#'   prior_theta01_prec = 1e-4,
#'   prior_theta02_mean = 0,
#'   prior_theta02_prec = 1e-4,
#'   prior_prec1_shape = 1e-3,
#'   prior_prec1_rate = 1e-3,
#'   prior_prec2_shape = 1e-3,
#'   prior_prec2_rate = 1e-3,
#'   prior_prec_y_shape = 1e-3,
#'   prior_prec_y_rate = 1e-3,
#'   seed = 456
#' )
#'
#' ## Posterior analysis
#' # You can now analyze the `out` object, for example, by plotting the median
#' # of the posterior samples for the latent states against the true values.
#' \dontrun{
#'   theta1_est <- apply(out$theta_1, 2, median)
#'   theta2_est <- apply(out$theta_2, 2, median)
#'
#'   # Plot level
#'   plot.ts(theta1_true, col = "red", ylab = expression(theta[t1]), main = "Latent Level")
#'   lines(theta1_est, col = "black")
#'   legend("topleft", legend = c("True", "Estimated"), col = c("red", "black"), lty = 1)
#'
#'   # Plot trend
#'   plot.ts(theta2_true, col = "red", ylab = expression(theta[t2]), main = "Latent Trend")
#'   lines(theta2_est, col = "black")
#'   legend("topleft", legend = c("True", "Estimated"), col = c("red", "black"), lty = 1)
#' }
#'
mcmc_localtrend <- function(y,
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
                            prior_prec_y_shape,
                            prior_prec_y_rate,
                            seed = NULL) {
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
  # Priors for prec_1
  if (!is.numeric(prior_prec1_shape) || length(prior_prec1_shape) != 1 || prior_prec1_shape <= 0) {
    stop("`prior_prec1_shape` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec1_rate) || length(prior_prec1_rate) != 1 || prior_prec1_rate <= 0) {
    stop("`prior_prec1_rate` must be a single positive numeric value")
  }
  # Priors for prec_2
  if (!is.numeric(prior_prec2_shape) || length(prior_prec2_shape) != 1 || prior_prec2_shape <= 0) {
    stop("`prior_prec2_shape` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec2_rate) || length(prior_prec2_rate) != 1 || prior_prec2_rate <= 0) {
    stop("`prior_prec2_rate` must be a single positive numeric value")
  }
  # Priors for prec_y
  if (!is.numeric(prior_prec_y_shape) || length(prior_prec_y_shape) != 1 || prior_prec_y_shape <= 0) {
    stop("`prior_prec_y_shape` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec_y_rate) || length(prior_prec_y_rate) != 1 || prior_prec_y_rate <= 0) {
    stop("`prior_prec_y_rate` must be a single positive numeric value")
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
    "_pdm_C_MCMC_localtrend",
    as.numeric(y),
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
    as.numeric(prior_prec_y_shape),
    as.numeric(prior_prec_y_rate)
  )
}
