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
#' Burn‐in and thinning are applied so that exactly \code{n_chain}
#' posterior samples are returned.
#'
#' @param y Numeric vector of observations (length \eqn{n}). Must contain only finite values.
#' @param burnin Integer \eqn{\geq 0}, number of burn-in iterations.
#' @param thinning Integer \eqn{\geq 1}, thinning interval.
#' @param n_chain Integer \eqn{\geq 1}, number of posterior samples to retain.
#' @param prior_theta01_mean Numeric, prior mean for the initial state \eqn{\theta_{0,1}}.
#' @param prior_theta01_prec Numeric > 0, prior precision (inverse variance) for \eqn{\theta_{0,1}}.
#' @param prior_prec1_shape Numeric > 0, shape parameter of the Gamma prior for the innovation precision \eqn{1/W_1}.
#' @param prior_prec1_rate Numeric > 0, rate parameter of the Gamma prior for \eqn{1/W_1}.
#' @param prior_prec_y_shape Numeric > 0, shape parameter of the Gamma prior for the data precision \eqn{1/V}.
#' @param prior_prec_y_rate Numeric > 0, rate parameter of the Gamma prior for \eqn{1/V}.
#' @param seed Optional integer used to set the random number generator seed.
#'   Default is \code{NULL}, which does not set the seed.
#'
#' @return A list with components:
#' \describe{
#'   \item{`theta_1`}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior samples for the latent state \eqn{\theta_{t,1}}.}
#'   \item{`theta_01`}{Numeric vector of length `n_chain` of posterior samples for the initial state \eqn{\theta_{0,1}}.}
#'   \item{`prec_1`}{Numeric vector of length `n_chain` of posterior samples for the innovation precision \eqn{1/W_1}.}
#'   \item{`prec_y`}{Numeric vector of length `n_chain` of posterior samples for the data precision \eqn{1/V}.}
#' }
#'
#' @examples
#' ## Description
#' # This example demonstrates how to:
#' # 1. Simulate data from a local-level dynamic model
#' # 2. Use `mcmc_locallevel` to estimate parameters and latent states
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
#' u <- rnorm(n, sd = sqrt(1/prec1_true))  # Evolution noise (u[t])
#' e <- rnorm(n, sd = sqrt(1/prec_y_true)) # Observation noise (e[t])
#'
#' # Simulate latent states and observations:
#' theta1_true <- cumsum(c(theta0_true, u))[-1]  # theta[t1] series
#' y <- theta1_true + e                          # Observed data (y[t])
#'
#' ## Running the Gibbs sampler
#' # Run the Gibbs sampler with specified priors and a seed
#' out <- mcmc_locallevel(
#'   y,
#'   burnin = 1000,
#'   thinning = 10,
#'   n_chain = 1000,
#'   prior_theta01_mean = y[1],
#'   prior_theta01_prec = 1/var(y),
#'   prior_prec1_shape = 1e-2,
#'   prior_prec1_rate = 1e-2,
#'   prior_prec_y_shape = 1e-2,
#'   prior_prec_y_rate = 1e-2,
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
#'     main = "Simulated data",
#'     ylab = expression(y[t]),
#'     xlab = "t"
#'   )
#'
#'   # --- 1. Latent State (theta[t1]) ---
#'
#'   # Visualize trajectories from the first few posterior samples
#'   num_traj_to_plot <- 20
#'   matplot(
#'     t(out$theta_1[1:num_traj_to_plot, ]),
#'     type = "l",
#'     lty = 1,
#'     col = grDevices::rainbow(num_traj_to_plot, alpha = 0.5),
#'     xlab = "t",
#'     ylab = expression(theta[t1]),
#'     main = "Sampled trajectories for latent state"
#'   )
#'
#'   # Plot true and estimated (median) latent state
#'   theta_1_estimate <- apply(X = out$theta_1, MARGIN = 2, FUN = median)
#'   range_theta_1 <- range(theta_1_estimate, theta1_true)
#'   r1_theta1 <- range_theta_1[1]
#'   r2_theta1 <- range_theta_1[2] + 0.2 * diff(range_theta_1)
#'
#'   plot.ts(
#'     theta1_true,
#'     col = "red",
#'     type = "l",
#'     xlab = "t",
#'     ylim = c(r1_theta1, r2_theta1),
#'     lty = 2,
#'     ylab = expression(theta[t1]),
#'     main = "Latent state"
#'   )
#'   points(theta_1_estimate, type = "l")
#'   legend(
#'     "topright",
#'     legend = c(expression(theta[t1]), expression(hat(theta)[t1])),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     bty = "n"
#'   )
#'
#'   # --- 2. Initial State (theta[01]) ---
#'
#'   # Trace plot for theta[01]
#'   range_theta_01 <- range(out$theta_01)
#'   r1_theta01 <- range_theta_01[1]
#'   r2_theta01 <- range_theta_01[2] + 0.2 * diff(range_theta_01)
#'
#'   plot.ts(
#'     out$theta_01,
#'     ylab = expression(theta["01"]),
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
#'     legend = c(expression(theta["01"]), expression(hat(theta)["01"])),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     bty = "n",
#'     lwd = 2
#'   )
#'
#'   # Density estimate for theta[01]
#'   plot(
#'     density(out$theta_01),
#'     main = "Posterior density estimate of initial state",
#'     xlab = expression(theta["01"]),
#'     ylab = "Density",
#'     lwd = 2
#'   )
#'   abline(
#'     v = c(theta0_true, median(out$theta_01)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'   legend(
#'     "topright",
#'     legend = c(expression(theta["01"]), expression(hat(theta)["01"])),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     bty = "n",
#'     lwd = 2
#'   )
#'
#'   # --- 3. Evolution Precision (1/W[1]) ---
#'
#'   # Traceplot for 1/W[1]
#'   range_prec_1 <- range(out$prec_1)
#'   r1_prec1 <- range_prec_1[1]
#'   r2_prec1 <- range_prec_1[2] + 0.2 * diff(range_prec_1)
#'
#'   plot.ts(
#'     out$prec_1,
#'     ylab = expression(1/W[1]),
#'     main = "Trace plot of evolution precision",
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
#'
#'   # Density estimate for 1/W[1]
#'   plot(
#'     density(out$prec_1),
#'     main = "Posterior density estimate of evolution precision",
#'     xlab = expression(W[1]^-1),
#'     ylab = "Density",
#'     lwd = 2
#'   )
#'   abline(
#'     v = c(prec1_true, median(out$prec_1)),
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
#'
#'   # --- 4. Observation Precision (1/V) ---
#'
#'   # Traceplot for 1/V
#'   range_prec_y <- range(out$prec_y)
#'   r1_precy <- range_prec_y[1]
#'   r2_precy <- range_prec_y[2] + 0.2 * diff(range_prec_y)
#'
#'   plot.ts(
#'     out$prec_y,
#'     ylab = expression(1/V),
#'     main = "Trace plot of observation precision",
#'     xlab = "Iterations",
#'     col = "gray",
#'     ylim = c(r1_precy, r2_precy)
#'   )
#'   abline(
#'     h = c(prec_y_true, median(out$prec_y)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'   legend(
#'     "topright",
#'     legend = c(expression(V^-1), expression(hat(V)^-1)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     bty = "n",
#'     lwd = 2
#'   )
#'
#'   # Density estimate for 1/V
#'   plot(
#'     density(out$prec_y),
#'     main = "Posterior density estimate of observation precision",
#'     xlab = expression(V^-1),
#'     ylab = "Density",
#'     lwd = 2
#'   )
#'   abline(
#'     v = c(prec_y_true, median(out$prec_y)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'   legend(
#'     "topright",
#'     legend = c(expression(V^-1), expression(hat(V)^-1)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     bty = "n",
#'     lwd = 2
#'   )
#' }
#'
#' @seealso \link[pdm]{mcmc_localtrend}
#' @export
mcmc_locallevel <- function(y,
                            burnin,
                            thinning,
                            n_chain,
                            prior_theta01_mean,
                            prior_theta01_prec,
                            prior_prec1_shape,
                            prior_prec1_rate,
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
  if (!is.numeric(prior_theta01_mean) || length(prior_theta01_mean) != 1) {
    stop("`prior_theta01_mean` must be a single numeric value")
  }
  if (!is.numeric(prior_theta01_prec) || length(prior_theta01_prec) != 1 || prior_theta01_prec <= 0) {
    stop("`prior_theta01_prec` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec1_shape) || length(prior_prec1_shape) != 1 || prior_prec1_shape <= 0) {
    stop("`prior_prec1_shape` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec1_rate) || length(prior_prec1_rate) != 1 || prior_prec1_rate <= 0) {
    stop("`prior_prec1_rate` must be a single positive numeric value")
  }
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
    "_pdm_C_MCMC_locallevel",
    as.numeric(y),
    as.integer(burnin),
    as.integer(thinning),
    as.integer(n_chain),
    as.numeric(prior_theta01_mean),
    as.numeric(prior_theta01_prec),
    as.numeric(prior_prec1_shape),
    as.numeric(prior_prec1_rate),
    as.numeric(prior_prec_y_shape),
    as.numeric(prior_prec_y_rate)
  )
}
