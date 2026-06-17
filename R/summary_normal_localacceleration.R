#' Summary method for normal_localacceleration objects
#'
#' @description Produces comprehensive posterior statistics including means,
#'   standard deviations, medians, and credible intervals.
#'
#' @param object An object of class \code{normal_localacceleration}, typically the
#'   result of calling \code{\link{mcmc_normal_localacceleration}}.
#' @param probs Numeric vector of probabilities for credible intervals.
#'   Default is \code{c(0.025, 0.975)} for 95\% credible intervals.
#' @param ... Additional arguments (currently unused).
#'
#' @return An object of class \code{summary.normal_localacceleration}, which is a list
#'   containing:
#'   \describe{
#'     \item{\code{model_type}}{Character string indicating the model type}
#'     \item{\code{n_obs}}{Number of observations}
#'     \item{\code{n_chain}}{Number of MCMC samples}
#'     \item{\code{burnin}}{Number of burn-in iterations}
#'     \item{\code{thinning}}{Thinning interval}
#'     \item{\code{probs}}{Probabilities used for credible intervals}
#'     \item{\code{scalar_params}}{Data frame with summary statistics for
#'       scalar parameters (theta_01, theta_02, theta_03, W_1^{-1}, W_2^{-1}, W_3^{-1}, V^{-1})}
#'     \item{\code{theta1_summary}}{Summary statistics for the latent level
#'       \eqn{\theta_{t,1}} aggregated across time}
#'     \item{\code{theta2_summary}}{Summary statistics for the latent trend
#'       \eqn{\theta_{t,2}} aggregated across time}
#'     \item{\code{theta3_summary}}{Summary statistics for the latent acceleration
#'       \eqn{\theta_{t,3}} aggregated across time}
#'   }
#'
#' @details
#' This method provides complete posterior inference with multiple statistics:
#' \describe{
#'   \item{\strong{Mean}}{Expected value under the posterior (minimizes squared error loss)}
#'   \item{\strong{Median}}{Typical value (minimizes absolute error loss, shown in \code{print()})}
#'   \item{\strong{SD}}{Posterior standard deviation (measure of uncertainty)}
#'   \item{\strong{CI}}{Credible intervals at specified probabilities}
#' }
#'
#' For a quick overview showing only medians, use \code{print(x)}.
#'
#' For time-varying parameters like \eqn{\theta_{t,1}}, \eqn{\theta_{t,2}}, and
#' \eqn{\theta_{t,3}}, only summary statistics across time are reported. Use
#' \code{plot()} to visualize the full trajectories.
#'
#' @examples
#' \donttest{
#' ## Simulation of data
#' n <- 1000 # Number of observations to simulate
#'
#' # True parameters for simulation:
#' theta01_true <- 10         # Initial level (theta[0,1])
#' theta02_true <- 0.5        # Initial trend (theta[0,2])
#' theta03_true <- 0.01       # Initial acceleration (theta[0,3])
#' prec1_true   <- 1 / 0.100  # Level innovation precision (1/W[1])
#' prec2_true   <- 1 / 0.010  # Trend innovation precision (1/W[2])
#' prec3_true   <- 1 / 0.001  # Acceleration innovation precision (1/W[3])
#' prec_y_true  <- 1 / 1.000  # Observation precision (1/V)
#'
#' # Use a fixed seed for data simulation
#' set.seed(123)
#'
#' # Generate noise terms:
#' u1      <- rnorm(n, sd = sqrt(1 / prec1_true))  # Level noise
#' u2      <- rnorm(n, sd = sqrt(1 / prec2_true))  # Trend noise
#' u3      <- rnorm(n, sd = sqrt(1 / prec3_true))  # Acceleration noise
#' epsilon <- rnorm(n, sd = sqrt(1 / prec_y_true)) # Observation noise
#'
#' # Simulate latent states and observations:
#' theta1_true    <- numeric(n)
#' theta2_true    <- numeric(n)
#' theta3_true    <- numeric(n)
#' theta3_true[1] <- theta03_true + u3[1]
#' theta2_true[1] <- theta02_true + theta03_true + u2[1]
#' theta1_true[1] <- theta01_true + theta02_true + u1[1]
#' for (t in 2:n) {
#'   theta3_true[t] <- theta3_true[t-1] + u3[t]
#'   theta2_true[t] <- theta2_true[t-1] + theta3_true[t-1] + u2[t]
#'   theta1_true[t] <- theta1_true[t-1] + theta2_true[t-1] + u1[t]
#' }
#' y <- theta1_true + epsilon # Observed data
#'
#' ## Running the Gibbs sampler
#' out <- mcmc_normal_localacceleration(
#'   y,
#'   burnin             = 2000,
#'   thinning           = 100,
#'   n_chain            = 1000,
#'   prior_theta01_mean = y[1],
#'   prior_theta01_prec = 1 / var(y),
#'   prior_theta02_mean = y[1] / 2,
#'   prior_theta02_prec = 1 / var(y),
#'   prior_theta03_mean = 0,
#'   prior_theta03_prec = 1e-3,
#'   prior_prec1_shape  = 1e-1,
#'   prior_prec1_rate   = 1e-1,
#'   prior_prec2_shape  = 1e-2,
#'   prior_prec2_rate   = 1e-2,
#'   prior_prec3_shape  = 1e-1,
#'   prior_prec3_rate   = 1e-2,
#'   prior_prec_y_shape = 1e-1,
#'   prior_prec_y_rate  = 1e-1,
#'   verbose            = TRUE,
#'   bar_width          = 60,
#'   seed               = 456
#' )
#'
#' # Quick overview (medians only)
#' print(out)
#'
#' # Full statistics
#' summary(out)
#'
#' # Custom credible intervals
#' summary(out, probs = c(0.05, 0.95))  # 90% CI
#' summary(out, probs = c(0.10, 0.90))  # 80% CI
#' }
#'
#' @seealso \code{\link{mcmc_normal_localacceleration}},
#'   \code{\link{print.normal_localacceleration}}
#'
#' @export
summary.normal_localacceleration <- function(object,
                                             probs = c(0.025, 0.975),
                                             ...) {

  # Validate input
  validate_summary_input(object, probs, "normal_localacceleration")

  # Scalar parameters (initial states and precisions)
  scalar_params <- data.frame(
    Parameter = c("theta_01", "theta_02", "theta_03",
                  "W_1^-1", "W_2^-1", "W_3^-1", "V^-1"),
    rbind(
      compute_summary_stats(object$theta_01, probs),
      compute_summary_stats(object$theta_02, probs),
      compute_summary_stats(object$theta_03, probs),
      compute_summary_stats(object$prec_theta1, probs),
      compute_summary_stats(object$prec_theta2, probs),
      compute_summary_stats(object$prec_theta3, probs),
      compute_summary_stats(object$prec_y, probs)
    )
  )

  # Latent level summary aggregated across time (detailed version)
  theta1_summary <- format_timevarying_summary(object$theta_1, probs, "detailed")

  # Latent trend summary aggregated across time (detailed version)
  theta2_summary <- format_timevarying_summary(object$theta_2, probs, "detailed")

  # Latent acceleration summary aggregated across time (detailed version)
  theta3_summary <- format_timevarying_summary(object$theta_3, probs, "detailed")

  # Create summary object
  result <- list(
    model_type = attr(object, "model_type"),
    n_obs = attr(object, "n_obs"),
    n_chain = attr(object, "n_chain"),
    burnin = attr(object, "burnin"),
    thinning = attr(object, "thinning"),
    probs = probs,
    scalar_params = scalar_params,
    theta1_summary = theta1_summary,
    theta2_summary = theta2_summary,
    theta3_summary = theta3_summary
  )

  class(result) <- "summary.normal_localacceleration"
  return(result)
}


#' Print method for summary.normal_localacceleration objects
#'
#' @description Prints comprehensive posterior statistics in a readable format.
#'
#' @param x An object of class \code{summary.normal_localacceleration}, typically the
#'   result of calling \code{summary()} on a \code{normal_localacceleration} object.
#' @param digits Integer, number of significant digits to display. Default is 3.
#' @param ... Additional arguments (currently unused).
#'
#' @return Invisibly returns the input object \code{x}.
#'
#' @examples
#' \donttest{
#' ## Simulation of data
#' n <- 1000
#' theta01_true <- 10
#' theta02_true <- 0.5
#' theta03_true <- 0.01
#' prec1_true   <- 1 / 0.100
#' prec2_true   <- 1 / 0.010
#' prec3_true   <- 1 / 0.001
#' prec_y_true  <- 1 / 1.000
#'
#' set.seed(123)
#' u1      <- rnorm(n, sd = sqrt(1 / prec1_true))
#' u2      <- rnorm(n, sd = sqrt(1 / prec2_true))
#' u3      <- rnorm(n, sd = sqrt(1 / prec3_true))
#' epsilon <- rnorm(n, sd = sqrt(1 / prec_y_true))
#'
#' theta1_true    <- numeric(n)
#' theta2_true    <- numeric(n)
#' theta3_true    <- numeric(n)
#' theta3_true[1] <- theta03_true + u3[1]
#' theta2_true[1] <- theta02_true + theta03_true + u2[1]
#' theta1_true[1] <- theta01_true + theta02_true + u1[1]
#' for (t in 2:n) {
#'   theta3_true[t] <- theta3_true[t-1] + u3[t]
#'   theta2_true[t] <- theta2_true[t-1] + theta3_true[t-1] + u2[t]
#'   theta1_true[t] <- theta1_true[t-1] + theta2_true[t-1] + u1[t]
#' }
#' y <- theta1_true + epsilon
#'
#' out <- mcmc_normal_localacceleration(
#'   y,
#'   burnin             = 2000,
#'   thinning           = 10,
#'   n_chain            = 1000,
#'   prior_theta01_mean = y[1],
#'   prior_theta01_prec = 1 / var(y),
#'   prior_theta02_mean = y[1] / 2,
#'   prior_theta02_prec = 1 / var(y),
#'   prior_theta03_mean = 0,
#'   prior_theta03_prec = 1e-3,
#'   prior_prec1_shape  = 1e-1,
#'   prior_prec1_rate   = 1e-1,
#'   prior_prec2_shape  = 1e-2,
#'   prior_prec2_rate   = 1e-2,
#'   prior_prec3_shape  = 1e-1,
#'   prior_prec3_rate   = 1e-2,
#'   prior_prec_y_shape = 1e-1,
#'   prior_prec_y_rate  = 1e-1,
#'   verbose            = TRUE,
#'   bar_width          = 60,
#'   seed               = 456
#' )
#'
#' s <- summary(out)
#' print(s)
#' # or simply:
#' summary(out)
#' }
#'
#' @seealso \code{\link{summary.normal_localacceleration}},
#'   \code{\link{print.normal_localacceleration}}
#'
#' @export
print.summary.normal_localacceleration <- function(x, digits = 3, ...) {

  cat("\n")
  cat("Summary: Gaussian Local-Acceleration Model\n")
  cat(strrep("=", 75), "\n\n", sep = "")

  # Model information
  cat("Model Information:\n")
  cat("  Type:              ", x$model_type, "\n", sep = "")
  cat("  Observations:      ", x$n_obs, "\n", sep = "")
  cat("  MCMC samples:      ", x$n_chain, "\n", sep = "")
  cat("  Burn-in:           ", x$burnin, "\n", sep = "")
  cat("  Thinning:          ", x$thinning, "\n\n", sep = "")

  # Credible interval level
  ci_level <- (x$probs[2] - x$probs[1]) * 100
  cat("Credible Intervals: ", sprintf("%.1f", ci_level), "%\n\n", sep = "")

  # Explanation of statistics
  cat("Statistics Legend:\n")
  cat("  Mean   = Posterior mean (minimizes squared error)\n")
  cat("  Median = Posterior median (minimizes absolute error, shown in print())\n")
  cat("  SD     = Posterior standard deviation\n")
  cat("  CI     = Credible interval at specified level\n\n")

  # Scalar parameters
  print_formatted_table(x$scalar_params, digits, "Scalar Parameters:")

  # Latent level summary
  print_formatted_table(x$theta1_summary, digits, "Latent Level (theta_{t,1}) Summary:")

  # Latent trend summary
  print_formatted_table(x$theta2_summary, digits, "Latent Trend (theta_{t,2}) Summary:")

  # Latent acceleration summary
  print_formatted_table(x$theta3_summary, digits, "Latent Acceleration (theta_{t,3}) Summary:")

  # Print footer
  print_summary_footer()

  invisible(x)
}
