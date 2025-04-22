#' Gibbs Sampler for a Local‐Level Dynamic Model
#'
#' Runs a Gibbs sampler for the local‐level dynamic model:
#' \deqn{
#' \begin{aligned}
#' y_t &= \theta_{t1} + e_t, \quad e_t \sim N(0, V), \\
#' \theta_{t1} &= \theta_{(t-1)1} + u_t, \quad u_t \sim N(0, W_1). \\
#' \end{aligned}
#' }
#'
#' Burn‐in and thinning are applied so that exactly \code{n_chain}
#' posterior samples are returned.
#'
#' @param y Numeric vector of observations (length \eqn{n}).
#' @param burnin Integer \eqn{\geq 0}, number of burn-in iterations.
#' @param thinning Integer \eqn{\geq 1}, thinning interval.
#' @param n_chain Integer \eqn{\geq 1}, number of posterior samples to retain.
#' @param prior_theta01_mean Numeric, prior mean for the initial state \eqn{\theta_{01}}.
#' @param prior_theta01_prec Numeric, prior precision (inverse variance) for \eqn{\theta_{01}}.
#' @param prior_prec1_shape Numeric, shape parameter of the Gamma prior for the innovation precision \eqn{1/W_1}.
#' @param prior_prec1_rate Numeric, rate parameter of the Gamma prior for \eqn{1/W_1}.
#' @param prior_prec_y_shape Numeric, shape parameter of the Gamma prior for the data precision \eqn{1/V}.
#' @param prior_prec_y_rate Numeric, rate parameter of the Gamma prior for \eqn{1/V}.
#'
#' @return A \code{list} with components:
#' \describe{
#'   \item{\code{theta_1}}{Numeric matrix \eqn{[n_{\text{chain}} \times n]} of latent‐state samples.}
#'   \item{\code{theta_01}}{Numeric vector of length \code{n_chain} containing
#'   posterior samples of the initial state \eqn{\theta_{01}}}
#'   \item{\code{prec_1}}{Numeric vector of length \code{n_chain} containing
#'   posterior samples of the innovation precision \eqn{1/W_1}.}
#'   \item{\code{prec_y}}{Numeric vector of length \code{n_chain} containing
#'   posterior samples of the observation precision \eqn{1/V}.}
#' }
#'
#' @examples
#' ## Description
#' # This example demonstrates how to:
#' # 1. Simulate data from a local-level dynamic model
#' # 2. Use \code{mcmc_locallevel} to estimate latent states
#' # 3. Visualize posterior results
#'
#' ## Simulation of Data
#' n <- 1000  # Number of observations to simulate
#'
#' # True parameters for simulation:
#' theta0_true <- 10  # Initial state \eqn{\theta_{01}}
#' prec1_true <- 1    # Innovation precision \eqn{1/W_1}
#' prec_y_true <- 5   # Observation precision \eqn{1/V}
#'
#' set.seed(123)  # For reproducibility
#'
#' # Generate noise terms:
#' u <- rnorm(n, sd = sqrt(1/prec1_true))  # Evolution noise \eqn{u_t}
#' e <- rnorm(n, sd = sqrt(1/prec_y_true)) # Observation noise \eqn{e_t}
#'
#' # Simulate latent states and observations:
#' theta1_true <- cumsum(c(theta0_true, u))[-1]  # \eqn{\theta_{t1}} series
#' y <- theta1_true + e                          # Observed data \eqn{y_t}
#'
#' # Plot the simulated data
#' \dontrun{
#' plot.ts(y, main = "Simulated Data", ylab = expression(y[t]), xlab = "t")
#' }
#'
#' ## Running the Gibbs Sampler
#'
#' # Run the Gibbs sampler with specified priors
#' out <- mcmc_locallevel(
#'   y,
#'   burnin               = 1000,          # Number of burn-in iterations
#'   thinning             = 10,            # Thinning interval
#'   n_chain              = 1000,          # Number of posterior samples
#'   prior_theta01_mean   = y[1],          # Prior mean for \eqn{\theta_{01}}
#'   prior_theta01_prec   = 1/var(y),      # Prior precision for \eqn{\theta_{01}}
#'   prior_prec1_shape    = 1e-2,          # Shape parameter for \eqn{1/W_1}
#'   prior_prec1_rate     = 1e-2,          # Rate parameter for \eqn{1/W_1}
#'   prior_prec_y_shape   = 1e-2,          # Shape parameter for \eqn{1/V}
#'   prior_prec_y_rate    = 1e-2           # Rate parameter for \eqn{1/V}
#' )
#'
#' ## Posterior Analysis and Visualization
#'
#' # Estimate the latent state (\eqn{\theta_1}) using the median of posterior samples
#' theta_1_estimate <- apply(X = out$theta_1, MARGIN = 2, FUN = median)
#'
#' # Plot the true latent state and its posterior estimate
#' range_theta_1 <- range(theta_1_estimate)
#' r1_theta1 <- range_theta_1[1]; r2_theta1 <- range_theta_1[2] + 0.2*diff(range_theta_1)
#' \dontrun{
#' plot.ts(theta1_true, col = "red", type = "l", xlab = "t",
#'         ylim = c(r1_theta1, r2_theta_1), lty = 2,
#'         ylab = expression(theta[t1]), main = "Estimate of the Latent State")
#' points(theta_1_estimate, type = "l")
#' legend("topright", legend = c(expression(theta[t1]), expression(hat(theta)[t1])),
#'        col = c("red", "black"), lty = c(2, 1), bty = "n")
#' }
#'
#' # Trace plot for the initial state (\eqn{\theta_{01}})
#' range_theta_01 <- range(out$theta_01)
#' r1_theta01 <- range_theta_01[1]; r2_theta01 <- range_theta_01[2] + 0.2*diff(range_theta_01)
#' \dontrun{
#' plot.ts(out$theta_01, ylab = expression(theta["01"]), main = "Trace Plot of Initial State",
#'         xlab = "Iterations", col = "gray", ylim = c(r1_theta01, r2_theta01))
#' abline(h = c(theta0_true, median(out$theta_01)), col = c("red", "black"), lty = c(2, 1), lwd = 2)
#' legend("topright", legend = c(expression(theta["01"]), expression(hat(theta)["01"])),
#'        col = c("red", "black"), lty = c(2, 1), bty = "n", lwd = 2)
#' }
#'
#' # Density estimate for \eqn{\theta_{01}}
#' \dontrun{
#' plot(density(out$theta_01), main = "Posterior Density Estimate of Initial State",
#'      xlab = expression(theta["01"]), ylab = "Density", lwd = 2)
#' abline(v = c(theta0_true, median(out$theta_01)), col = c("red", "black"),
#'        lty = c(2, 1), lwd = 2)
#' legend("topright", legend = c(expression(theta["01"]), expression(hat(theta)["01"])),
#'        col = c("red", "black"), lty = c(2, 1), bty = "n", lwd = 2)
#' }
#'
#' # Traceplot for the precision of evolution (\eqn{1/W_1})
#' range_prec_1 <- range(out$prec_1)
#' r1_prec1 <- range_prec_1[1]; r2_prec1 <- range_prec_1[2] + 0.2*diff(range_prec_1)
#' \dontrun{
#' plot.ts(out$prec_1, ylab = expression(1/W[1]), main = "Trace Plot of Evolution Precision",
#'         xlab = "Iterations", col = "gray", ylim = c(r1_prec1, r2_prec1))
#' abline(h = c(prec1_true, median(out$prec_1)), col = c("red", "black"), lty = c(2, 1), lwd = 2)
#' legend("topright", legend = c(expression(W[1]^-1), expression(hat(W)[1]^-1)),
#'        col = c("red", "black"), lty = c(2, 1), bty = "n", lwd = 2)
#' }
#'
#' # Density estimate for the precision of evolution (\eqn{1/W_1})
#' \dontrun{
#' plot(density(out$prec_1), main = "Posterior Density Estimate of Evolution Precision",
#'      xlab = expression(W[1]^-1), ylab = "Density", lwd = 2)
#' abline(v = c(prec1_true, median(out$prec_1)), col = c("red", "black"),
#'        lty = c(2, 1), lwd = 2)
#' legend("topright", legend = c(expression(W[1]^-1), expression(hat(W)[1]^-1)),
#'        col = c("red", "black"), lty = c(2, 1), bty = "n", lwd = 2)
#' }
#'
#' # Traceplot for the precision of observation (\eqn{1/V})
#' range_prec_y <- range(out$prec_y)
#' r1_precy <- range_prec_y[1]; r2_precy <- range_prec_y[2] + 0.2*diff(range_prec_y)
#' \dontrun{
#' plot.ts(out$prec_y, ylab = expression(1/V), main = "Trace Plot of Observation Precision",
#'         xlab = "Iterations", col = "gray", ylim = c(r1_precy, r2_precy))
#' abline(h = c(prec_y_true, median(out$prec_y)), col = c("red", "black"), lty = c(2, 1), lwd = 2)
#' legend("topright", legend = c(expression(V^-1), expression(hat(V)^-1)),
#'        col = c("red", "black"), lty = c(2, 1), bty = "n", lwd = 2)
#' }
#'
#' # Density estimate for the precision of observation (\eqn{1/V})
#' \dontrun{
#' plot(density(out$prec_y), main = "Posterior Density Estimate of Observation Precision",
#'      xlab = expression(V^-1), ylab = "Density", lwd = 2)
#' abline(v = c(prec_y_true, median(out$prec_y)), col = c("red", "black"),
#'        lty = c(2, 1), lwd = 2)
#' legend("topright", legend = c(expression(V^-1), expression(hat(V)^-1)),
#'        col = c("red", "black"), lty = c(2, 1), bty = "n", lwd = 2)
#' }
#'
#' @seealso \link[base]{.Call}, \link[base]{set.seed}, \link[stats]{rnorm}
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
                            prior_prec_y_rate) {
  if (!is.numeric(y)) {
    stop("`y` must be a numeric vector")
  }
  if (!is.numeric(burnin) || length(burnin) != 1 || burnin < 0) {
    stop("`burnin` must be a single integer ≥ 0")
  }
  if (!is.numeric(thinning) || length(thinning) != 1 || thinning < 1) {
    stop("`thinning` must be a single integer ≥ 1")
  }
  if (!is.numeric(n_chain) || length(n_chain) != 1 || n_chain < 1) {
    stop("`n_chain` must be a single integer ≥ 1")
  }

  .Call(
    "_pdm_mcmc_locallevel",
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
