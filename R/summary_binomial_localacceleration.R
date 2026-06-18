#' Summary method for binomial_localacceleration objects
#'
#' @description Produces comprehensive posterior statistics including means,
#'   standard deviations, medians, and credible intervals.
#'
#' @param object An object of class \code{binomial_localacceleration}, typically
#'   the result of calling \code{\link{mcmc_binomial_localacceleration}}.
#' @param probs Numeric vector of probabilities for credible intervals.
#'   Default is \code{c(0.025, 0.975)} for 95\% credible intervals.
#' @param ... Additional arguments (currently unused).
#'
#' @return An object of class \code{summary.binomial_localacceleration}, which is
#'   a list containing:
#'   \describe{
#'     \item{\code{model_type}}{Character string indicating the model type}
#'     \item{\code{n_obs}}{Number of observations}
#'     \item{\code{n_chain}}{Number of MCMC samples}
#'     \item{\code{burnin}}{Number of burn-in iterations}
#'     \item{\code{thinning}}{Thinning interval}
#'     \item{\code{probs}}{Probabilities used for credible intervals}
#'     \item{\code{scalar_params}}{Data frame with summary statistics for
#'       scalar parameters (theta_01, theta_02, theta_03, W_1^-1, W_2^-1, W_3^-1)}
#'     \item{\code{theta1_summary}}{Summary statistics for the latent level
#'       \eqn{\theta_{t,1}} aggregated across time}
#'     \item{\code{theta2_summary}}{Summary statistics for the latent trend
#'       \eqn{\theta_{t,2}} aggregated across time}
#'     \item{\code{theta3_summary}}{Summary statistics for the latent acceleration
#'       \eqn{\theta_{t,3}} aggregated across time}
#'     \item{\code{alpha_summary}}{Summary statistics for the binomial success
#'       probabilities \eqn{\alpha_t} aggregated across time (min, median, max)}
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
#' For time-varying parameters like \eqn{\theta_{t,1}}, \eqn{\theta_{t,2}},
#' \eqn{\theta_{t,3}}, and \eqn{\alpha_t}, only summary statistics across time
#' are reported. Use \code{plot()} to visualize the full trajectories.
#'
#' @examples
#' \donttest{
#' ## Simulation of data
#' n <- 500        # Number of observations to simulate
#' n_trials <- 20  # Number of binomial trials
#'
#' # True parameters for simulation:
#' theta01_true <- 0.5     # Initial level state (theta[0,1]) on logit scale
#' theta02_true <- 0.01    # Initial trend state (theta[0,2]) on logit scale
#' theta03_true <- 0.001   # Initial acceleration state (theta[0,3]) on logit scale
#' prec1_true <- 100       # Level innovation precision (1/W[1])
#' prec2_true <- 400       # Trend innovation precision (1/W[2])
#' prec3_true <- 1600      # Acceleration innovation precision (1/W[3])
#'
#' # Use a fixed seed for data simulation
#' set.seed(123)
#'
#' # Generate noise terms:
#' u1 <- rnorm(n, sd = sqrt(1/prec1_true))  # Level evolution noise (u1[t])
#' u2 <- rnorm(n, sd = sqrt(1/prec2_true))  # Trend evolution noise (u2[t])
#' u3 <- rnorm(n, sd = sqrt(1/prec3_true))  # Acceleration evolution noise (u3[t])
#'
#' # Simulate latent states and observations:
#' theta1_true <- numeric(n)
#' theta2_true <- numeric(n)
#' theta3_true <- numeric(n)
#' theta3_true[1] <- theta03_true + u3[1]
#' theta2_true[1] <- theta02_true + theta03_true + u2[1]
#' theta1_true[1] <- theta01_true + theta02_true + u1[1]
#' for (t in 2:n) {
#'   theta3_true[t] <- theta3_true[t-1] + u3[t]
#'   theta2_true[t] <- theta2_true[t-1] + theta3_true[t-1] + u2[t]
#'   theta1_true[t] <- theta1_true[t-1] + theta2_true[t-1] + u1[t]
#' }
#' alpha_true <- plogis(theta1_true)  # Success probabilities
#' y <- rbinom(n, size = n_trials, prob = alpha_true)  # Observed binomial counts
#'
#' ## Running the Gibbs sampler
#' # Run the Gibbs sampler with specified priors and a seed
#' out <- mcmc_binomial_localacceleration(
#'   y,
#'   n_trials                = n_trials,
#'   burnin                  = 1000,
#'   thinning                = 50,
#'   n_chain                 = 1000,
#'   prior_theta01_mean      = 0,
#'   prior_theta01_prec      = 1,
#'   prior_theta02_mean      = 0,
#'   prior_theta02_prec      = 1,
#'   prior_theta03_mean      = 0,
#'   prior_theta03_prec      = 1,
#'   prior_prec1_shape       = 100,
#'   prior_prec1_rate        = 1,
#'   prior_prec2_shape       = 400,
#'   prior_prec2_rate        = 1,
#'   prior_prec3_shape       = 1600,
#'   prior_prec3_rate        = 1,
#'   lag_update              = 50,
#'   max_step_size           = 0.1,
#'   base_adaptation_rate    = 1,
#'   decay_exponent          = 0.6,
#'   target_acceptance       = 0.44,
#'   min_deviation_threshold = NULL,  # Uses practical default: 1.0/50 = 0.02
#'   return_log_sigma        = FALSE,
#'   return_accept_prop      = TRUE,
#'   verbose                 = TRUE,  # Enable progress bar
#'   bar_width               = 60,    # Progress bar width
#'   seed                    = 456
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
#' @seealso \code{\link{mcmc_binomial_localacceleration}},
#'   \code{\link{print.binomial_localacceleration}}
#'
#' @export
summary.binomial_localacceleration <- function(object,
                                               probs = c(0.025, 0.975),
                                               ...) {

  # Validate input
  validate_summary_input(object, probs, "binomial_localacceleration")

  # Scalar parameters (initial states and precisions)
  scalar_params <- data.frame(
    Parameter = c("theta_01", "theta_02", "theta_03",
                  "W_1^-1", "W_2^-1", "W_3^-1"),
    rbind(
      compute_summary_stats(object$theta_01, probs),
      compute_summary_stats(object$theta_02, probs),
      compute_summary_stats(object$theta_03, probs),
      compute_summary_stats(object$prec_theta1, probs),
      compute_summary_stats(object$prec_theta2, probs),
      compute_summary_stats(object$prec_theta3, probs)
    )
  )

  # Latent level summary aggregated across time (detailed version)
  theta1_summary <- format_timevarying_summary(object$theta_1, probs, "detailed")

  # Latent trend summary aggregated across time (detailed version)
  theta2_summary <- format_timevarying_summary(object$theta_2, probs, "detailed")

  # Latent acceleration summary aggregated across time (detailed version)
  theta3_summary <- format_timevarying_summary(object$theta_3, probs, "detailed")

  # Alpha (binomial success probabilities) summary aggregated across time (simple version)
  # Note: Using "simple" for consistency with mixture models
  alpha_summary <- format_timevarying_summary(object$alpha, probs, "simple")

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
    theta3_summary = theta3_summary,
    alpha_summary = alpha_summary
  )

  class(result) <- "summary.binomial_localacceleration"
  return(result)
}


#' Print method for summary.binomial_localacceleration objects
#'
#' @description Prints comprehensive posterior statistics in a readable format.
#'
#' @param x An object of class \code{summary.binomial_localacceleration}, typically
#'   the result of calling \code{summary()} on a \code{binomial_localacceleration}
#'   object.
#' @param digits Integer, number of significant digits to display. Default is 3.
#' @param ... Additional arguments (currently unused).
#'
#' @return Invisibly returns the input object \code{x}.
#'
#' @examples
#' \donttest{
#' ## Simulation of data
#' n <- 500
#' n_trials <- 20
#' theta01_true <- 0.5
#' theta02_true <- 0.01
#' theta03_true <- 0.001
#' prec1_true <- 100
#' prec2_true <- 400
#' prec3_true <- 1600
#'
#' set.seed(123)
#' u1 <- rnorm(n, sd = sqrt(1/prec1_true))
#' u2 <- rnorm(n, sd = sqrt(1/prec2_true))
#' u3 <- rnorm(n, sd = sqrt(1/prec3_true))
#'
#' theta1_true <- numeric(n)
#' theta2_true <- numeric(n)
#' theta3_true <- numeric(n)
#' theta3_true[1] <- theta03_true + u3[1]
#' theta2_true[1] <- theta02_true + theta03_true + u2[1]
#' theta1_true[1] <- theta01_true + theta02_true + u1[1]
#' for (t in 2:n) {
#'   theta3_true[t] <- theta3_true[t-1] + u3[t]
#'   theta2_true[t] <- theta2_true[t-1] + theta3_true[t-1] + u2[t]
#'   theta1_true[t] <- theta1_true[t-1] + theta2_true[t-1] + u1[t]
#' }
#' alpha_true <- plogis(theta1_true)
#' y <- rbinom(n, size = n_trials, prob = alpha_true)
#'
#' out <- mcmc_binomial_localacceleration(
#'   y,
#'   n_trials                = n_trials,
#'   burnin                  = 1000,
#'   thinning                = 50,
#'   n_chain                 = 1000,
#'   prior_theta01_mean      = 0,
#'   prior_theta01_prec      = 1,
#'   prior_theta02_mean      = 0,
#'   prior_theta02_prec      = 1,
#'   prior_theta03_mean      = 0,
#'   prior_theta03_prec      = 1,
#'   prior_prec1_shape       = 100,
#'   prior_prec1_rate        = 1,
#'   prior_prec2_shape       = 400,
#'   prior_prec2_rate        = 1,
#'   prior_prec3_shape       = 1600,
#'   prior_prec3_rate        = 1,
#'   lag_update              = 50,
#'   max_step_size           = 0.1,
#'   base_adaptation_rate    = 1,
#'   decay_exponent          = 0.6,
#'   target_acceptance       = 0.44,
#'   min_deviation_threshold = NULL,
#'   return_log_sigma        = FALSE,
#'   return_accept_prop      = TRUE,
#'   verbose                 = TRUE,
#'   bar_width               = 60,
#'   seed                    = 456
#' )
#'
#' s <- summary(out)
#' print(s)
#' # or simply:
#' summary(out)
#' }
#'
#' @seealso \code{\link{summary.binomial_localacceleration}},
#'   \code{\link{print.binomial_localacceleration}}
#'
#' @export
print.summary.binomial_localacceleration <- function(x, digits = 3, ...) {

  cat("\n")
  cat("Summary: Logit-Binomial Local-Acceleration Model\n")
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
  print_formatted_table(x$theta1_summary, digits,
                        "Latent Level (theta_{t,1}) Summary (logit scale):")

  # Latent trend summary
  print_formatted_table(x$theta2_summary, digits,
                        "Latent Trend (theta_{t,2}) Summary (logit scale):")

  # Latent acceleration summary
  print_formatted_table(x$theta3_summary, digits,
                        "Latent Acceleration (theta_{t,3}) Summary (logit scale):")

  # Binomial success probabilities summary
  print_formatted_table(x$alpha_summary, digits,
                        "Success Probabilities (alpha_t) Summary:")

  # Print footer
  print_summary_footer()

  invisible(x)
}
