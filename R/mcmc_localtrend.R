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
#' @examples
#' ## Description
#' # This example demonstrates how to:
#' # 1. Simulate data from a local trend dynamic model
#' # 2. Use `mcmc_localtrend` to estimate parameters and latent states
#' # 3. Perform a detailed posterior analysis with visualizations
#' # 4. Set a seed for reproducibility
#'
#' ## Simulation of data
#' n <- 150 # Number of observations to simulate
#'
#' # True parameters for simulation:
#' theta01_true <- 10      # Initial level (theta[0,1])
#' theta02_true <- 0.5     # Initial trend (theta[0,2])
#' prec1_true <- 1 / 0.1   # Level innovation precision (1/W[1])
#' prec2_true <- 1 / 0.01  # Trend innovation precision (1/W[2])
#' prec_y_true <- 1 / 0.5  # Observation precision (1/V)
#'
#' # Use a fixed seed for data simulation
#' set.seed(123)
#'
#' # Generate noise terms:
#' omega1 <- rnorm(n, sd = sqrt(1 / prec1_true))  # Level noise
#' omega2 <- rnorm(n, sd = sqrt(1 / prec2_true))  # Trend noise
#' epsilon <- rnorm(n, sd = sqrt(1 / prec_y_true)) # Observation noise
#'
#' # Simulate latent states and observations:
#' theta1_true <- numeric(n)
#' theta2_true <- numeric(n)
#' theta2_true[1] <- theta02_true + omega2[1]
#' theta1_true[1] <- theta01_true + theta02_true + omega1[1]
#' for (t in 2:n) {
#'   theta2_true[t] <- theta2_true[t-1] + omega2[t]
#'   theta1_true[t] <- theta1_true[t-1] + theta2_true[t-1] + omega1[t]
#' }
#' y <- theta1_true + epsilon # Observed data
#'
#' ## Running the Gibbs sampler
#' # Run the Gibbs sampler with specified priors and a seed
#' out <- mcmc_localtrend(
#'   y,
#'   burnin = 2000,
#'   thinning = 20,
#'   n_chain = 1000,
#'   prior_theta01_mean = y[1],
#'   prior_theta01_prec = 1/var(y),
#'   prior_theta02_mean = 0,
#'   prior_theta02_prec = 1e-2,
#'   prior_prec1_shape = 1e-1,
#'   prior_prec1_rate = 1e-1,
#'   prior_prec2_shape = 1e-1,
#'   prior_prec2_rate = 1e-1,
#'   prior_prec_y_shape = 1e-1,
#'   prior_prec_y_rate = 1e-1,
#'   seed = 456
#' )
#'
#' ## Posterior analysis and visualization
#' # The following plots show how to analyze the posterior distributions.
#' # Point estimates are based on the median of posterior samples.
#' \dontrun{
#'   # --- 0. Plot the simulated data ---
#'   plot.ts(
#'     y,
#'     main = "Simulated Data",
#'     ylab = expression(y[t]),
#'     xlab = "t"
#'   )
#'
#'   # --- 1. Latent Level (theta[t,1]) ---
#'   theta1_est <- apply(out$theta_1, 2, median)
#'   plot.ts(
#'     theta1_true, col = "red", lty = 2,
#'     ylab = expression(theta[t1]), main = "Latent Level State",
#'     ylim = range(c(theta1_true, theta1_est))
#'   )
#'   lines(theta1_est, col = "black")
#'   legend(
#'     "topleft", bty = "n", lty = c(2, 1),
#'     legend = c("True", "Estimated"), col = c("red", "black")
#'   )
#'
#'   # --- 2. Latent Trend (theta[t,2]) ---
#'   theta2_est <- apply(out$theta_2, 2, median)
#'   plot.ts(
#'     theta2_true, col = "red", lty = 2,
#'     ylab = expression(theta[t2]), main = "Latent Trend State",
#'     ylim = range(c(theta2_true, theta2_est))
#'   )
#'   lines(theta2_est, col = "black")
#'   legend(
#'     "topleft", bty = "n", lty = c(2, 1),
#'     legend = c("True", "Estimated"), col = c("red", "black")
#'   )
#'
#'   # --- 3. Initial Level (theta[0,1]) ---
#'   # Trace plot
#'   plot.ts(
#'     out$theta_01, col = "gray", xlab = "Iterations",
#'     ylab = expression(theta["01"]), main = "Trace Plot of Initial Level",
#'     ylim = range(c(out$theta_01, theta01_true))
#'   )
#'   abline(h = c(theta01_true, median(out$theta_01)), col = c("red", "black"), lty = c(2, 1), lwd = 2)
#'   legend("topright", bty = "n", legend = c("True", "Median"), col = c("red", "black"), lty = c(2, 1), lwd = 2)
#'
#'   # Density plot
#'   plot(density(out$theta_01), main = "Posterior Density of Initial Level", xlab = expression(theta["01"]))
#'   abline(v = c(theta01_true, median(out$theta_01)), col = c("red", "black"), lty = c(2, 1), lwd = 2)
#'   legend("topright", bty = "n", legend = c("True", "Median"), col = c("red", "black"), lty = c(2, 1), lwd = 2)
#'
#'   # --- 4. Initial Trend (theta[0,2]) ---
#'   # Trace plot
#'   plot.ts(
#'     out$theta_02, col = "gray", xlab = "Iterations",
#'     ylab = expression(theta["02"]), main = "Trace Plot of Initial Trend",
#'     ylim = range(c(out$theta_02, theta02_true))
#'   )
#'   abline(h = c(theta02_true, median(out$theta_02)), col = c("red", "black"), lty = c(2, 1), lwd = 2)
#'   legend("topright", bty = "n", legend = c("True", "Median"), col = c("red", "black"), lty = c(2, 1), lwd = 2)
#'
#'   # Density plot
#'   plot(density(out$theta_02), main = "Posterior Density of Initial Trend", xlab = expression(theta["02"]))
#'   abline(v = c(theta02_true, median(out$theta_02)), col = c("red", "black"), lty = c(2, 1), lwd = 2)
#'   legend("topright", bty = "n", legend = c("True", "Median"), col = c("red", "black"), lty = c(2, 1), lwd = 2)
#'
#'   # --- 5. Level Precision (1/W_1) ---
#'   # Trace plot
#'   plot.ts(
#'     out$prec_1, col = "gray", xlab = "Iterations",
#'     ylab = expression(1/W[1]), main = "Trace Plot of Level Precision",
#'     ylim = range(c(out$prec_1, prec1_true))
#'   )
#'   abline(h = c(prec1_true, median(out$prec_1)), col = c("red", "black"), lty = c(2, 1), lwd = 2)
#'   legend("topright", bty = "n", legend = c("True", "Median"), col = c("red", "black"), lty = c(2, 1), lwd = 2)
#'
#'   # Density plot
#'   plot(density(out$prec_1), main = "Posterior Density of Level Precision", xlab = expression(1/W[1]))
#'   abline(v = c(prec1_true, median(out$prec_1)), col = c("red", "black"), lty = c(2, 1), lwd = 2)
#'   legend("topright", bty = "n", legend = c("True", "Median"), col = c("red", "black"), lty = c(2, 1), lwd = 2)
#'
#'   # --- 6. Trend Precision (1/W_2) ---
#'   # Trace plot
#'   plot.ts(
#'     out$prec_2, col = "gray", xlab = "Iterations",
#'     ylab = expression(1/W[2]), main = "Trace Plot of Trend Precision",
#'     ylim = range(c(out$prec_2, prec2_true))
#'   )
#'   abline(h = c(prec2_true, median(out$prec_2)), col = c("red", "black"), lty = c(2, 1), lwd = 2)
#'   legend("topright", bty = "n", legend = c("True", "Median"), col = c("red", "black"), lty = c(2, 1), lwd = 2)
#'
#'   # Density plot
#'   plot(density(out$prec_2), main = "Posterior Density of Trend Precision", xlab = expression(1/W[2]))
#'   abline(v = c(prec2_true, median(out$prec_2)), col = c("red", "black"), lty = c(2, 1), lwd = 2)
#'   legend("topright", bty = "n", legend = c("True", "Median"), col = c("red", "black"), lty = c(2, 1), lwd = 2)
#'
#'   # --- 7. Observation Precision (1/V) ---
#'   # Trace plot
#'   plot.ts(
#'     out$prec_y, col = "gray", xlab = "Iterations",
#'     ylab = expression(1/V), main = "Trace Plot of Observation Precision",
#'     ylim = range(c(out$prec_y, prec_y_true))
#'   )
#'   abline(h = c(prec_y_true, median(out$prec_y)), col = c("red", "black"), lty = c(2, 1), lwd = 2)
#'   legend("topright", bty = "n", legend = c("True", "Median"), col = c("red", "black"), lty = c(2, 1), lwd = 2)
#'
#'   # Density plot
#'   plot(density(out$prec_y), main = "Posterior Density of Observation Precision", xlab = expression(1/V))
#'   abline(v = c(prec_y_true, median(out$prec_y)), col = c("red", "black"), lty = c(2, 1), lwd = 2)
#'   legend("topright", bty = "n", legend = c("True", "Median"), col = c("red", "black"), lty = c(2, 1), lwd = 2)
#' }
#'
#' @seealso \link[pdm]{mcmc_locallevel}
#' @export
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
