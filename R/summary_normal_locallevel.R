#' Summary method for normal_locallevel objects
#'
#' @description Produces comprehensive posterior statistics including means,
#'   standard deviations, medians, and credible intervals.
#'
#' @param object An object of class \code{normal_locallevel}, typically the
#'   result of calling \code{\link{mcmc_normal_locallevel}}.
#' @param probs Numeric vector of probabilities for credible intervals.
#'   Default is \code{c(0.025, 0.975)} for 95\% credible intervals.
#' @param ... Additional arguments (currently unused).
#'
#' @return An object of class \code{summary.normal_locallevel}, which is a list
#'   containing:
#'   \describe{
#'     \item{\code{model_type}}{Character string indicating the model type}
#'     \item{\code{n_obs}}{Number of observations}
#'     \item{\code{n_chain}}{Number of MCMC samples}
#'     \item{\code{burnin}}{Number of burn-in iterations}
#'     \item{\code{thinning}}{Thinning interval}
#'     \item{\code{probs}}{Probabilities used for credible intervals}
#'     \item{\code{scalar_params}}{Data frame with summary statistics for
#'       scalar parameters (theta_01, W_1^{-1}, V^{-1})}
#'     \item{\code{theta_summary}}{Summary statistics for the latent level
#'       \eqn{\theta_{t,1}} aggregated across time (min, median, max, and final
#'       time point)}
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
#' For time-varying parameters like \eqn{\theta_{t,1}}, only summary statistics
#' across time are reported. Use \code{plot()} to visualize the full
#' trajectories.
#'
#' @examples
#' \dontrun{
#' ## Description
#' # This example demonstrates how to:
#' # 1. Simulate data from a local-level dynamic model
#' # 2. Use `mcmc_normal_locallevel` to estimate parameters and latent states
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
#' u1 <- rnorm(n, sd = sqrt(1/prec1_true))  # Evolution noise (u1[t])
#' e  <- rnorm(n, sd = sqrt(1/prec_y_true)) # Observation noise (e[t])
#'
#' # Simulate latent states and observations:
#' theta1_true <- cumsum(c(theta0_true, u1))[-1]  # theta[t1] series
#' y <- theta1_true + e                           # Observed data (y[t])
#'
#' ## Running the Gibbs sampler
#' out <- mcmc_normal_locallevel(
#'   y,
#'   burnin   = 1000,
#'   thinning = 10,
#'   n_chain  = 1000,
#'   prior_theta01_mean = y[1],
#'   prior_theta01_prec = 1 / var(y),
#'   prior_prec1_shape  = 1e-2,
#'   prior_prec1_rate   = 1e-2,
#'   prior_prec_y_shape = 1e-2,
#'   prior_prec_y_rate  = 1e-2,
#'   seed = 456
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
#' @seealso \code{\link{mcmc_normal_locallevel}},
#'   \code{\link{print.normal_locallevel}}
#'
#' @export
summary.normal_locallevel <- function(object,
                                      probs = c(0.025, 0.975),
                                      ...) {

  if (!inherits(object, "normal_locallevel")) {
    stop("Object must be of class 'normal_locallevel'")
  }

  if (!is.numeric(probs) || any(probs < 0) || any(probs > 1)) {
    stop("`probs` must be numeric values between 0 and 1")
  }

  if (length(probs) != 2) {
    stop("`probs` must have exactly 2 elements for lower and upper bounds")
  }

  if (probs[1] >= probs[2]) {
    stop("`probs[1]` must be less than `probs[2]`")
  }

  # Helper function to compute summary statistics
  compute_stats <- function(x, probs) {
    data.frame(
      Mean = mean(x),
      SD = sd(x),
      Median = median(x),
      CI_Lower = quantile(x, probs[1]),
      CI_Upper = quantile(x, probs[2]),
      row.names = NULL
    )
  }

  # Scalar parameters (initial state and precisions)
  scalar_params <- data.frame(
    Parameter = c("theta_01", "W_1^-1", "V^-1"),
    rbind(
      compute_stats(object$theta_01, probs),
      compute_stats(object$prec_1, probs),
      compute_stats(object$prec_y, probs)
    )
  )

  # Latent level summary aggregated across time
  theta_mean_time <- apply(object$theta_1, 2, mean)
  theta_sd_time <- apply(object$theta_1, 2, sd)
  theta_median_time <- apply(object$theta_1, 2, median)
  theta_ci_lower_time <- apply(object$theta_1, 2, quantile, probs = probs[1])
  theta_ci_upper_time <- apply(object$theta_1, 2, quantile, probs = probs[2])

  theta_summary <- data.frame(
    Statistic = c(
      "Min across time",
      "Median across time",
      "Max across time",
      "Final time point"
    ),
    Mean = c(
      min(theta_mean_time),
      median(theta_mean_time),
      max(theta_mean_time),
      tail(theta_mean_time, 1L)
    ),
    SD = c(
      min(theta_sd_time),
      median(theta_sd_time),
      max(theta_sd_time),
      tail(theta_sd_time, 1L)
    ),
    Median = c(
      min(theta_median_time),
      median(theta_median_time),
      max(theta_median_time),
      tail(theta_median_time, 1L)
    ),
    CI_Lower = c(
      min(theta_ci_lower_time),
      median(theta_ci_lower_time),
      max(theta_ci_lower_time),
      tail(theta_ci_lower_time, 1L)
    ),
    CI_Upper = c(
      min(theta_ci_upper_time),
      median(theta_ci_upper_time),
      max(theta_ci_upper_time),
      tail(theta_ci_upper_time, 1L)
    )
  )

  # Create summary object
  result <- list(
    model_type = attr(object, "model_type"),
    n_obs = attr(object, "n_obs"),
    n_chain = attr(object, "n_chain"),
    burnin = attr(object, "burnin"),
    thinning = attr(object, "thinning"),
    probs = probs,
    scalar_params = scalar_params,
    theta_summary = theta_summary
  )

  class(result) <- "summary.normal_locallevel"
  return(result)
}


#' Print method for summary.normal_locallevel objects
#'
#' @description Prints comprehensive posterior statistics in a readable format.
#'
#' @param x An object of class \code{summary.normal_locallevel}, typically the
#'   result of calling \code{summary()} on a \code{normal_locallevel} object.
#' @param digits Integer, number of significant digits to display. Default is 3.
#' @param ... Additional arguments (currently unused).
#'
#' @return Invisibly returns the input object \code{x}.
#'
#' @examples
#' \dontrun{
#' ## Simulation of data (see ?mcmc_normal_locallevel for full details)
#' n <- 1000
#' theta0_true <- 10
#' prec1_true <- 1
#' prec_y_true <- 5
#' set.seed(123)
#' u1 <- rnorm(n, sd = sqrt(1 / prec1_true))
#' e  <- rnorm(n, sd = sqrt(1 / prec_y_true))
#' theta1_true <- cumsum(c(theta0_true, u1))[-1]
#' y <- theta1_true + e
#'
#' out <- mcmc_normal_locallevel(
#'   y,
#'   burnin   = 1000,
#'   thinning = 10,
#'   n_chain  = 1000,
#'   prior_theta01_mean = y[1],
#'   prior_theta01_prec = 1 / var(y),
#'   prior_prec1_shape  = 1e-2,
#'   prior_prec1_rate   = 1e-2,
#'   prior_prec_y_shape = 1e-2,
#'   prior_prec_y_rate  = 1e-2,
#'   seed = 456
#' )
#'
#' s <- summary(out)
#' print(s)
#' # or simply:
#' summary(out)
#' }
#'
#' @seealso \code{\link{summary.normal_locallevel}},
#'   \code{\link{print.normal_locallevel}}
#'
#' @export
print.summary.normal_locallevel <- function(x, digits = 3, ...) {

  cat("\n")
  cat("Summary: Gaussian Local-Level Model\n")
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
  cat("Scalar Parameters:\n")
  cat(strrep("-", 75), "\n", sep = "")

  scalar_print <- x$scalar_params
  scalar_print[, -1] <- lapply(scalar_print[, -1], function(col) {
    sprintf(paste0("%.", digits, "f"), col)
  })

  print(scalar_print, row.names = FALSE, right = TRUE)
  cat("\n")

  # Latent level summary
  cat("Latent Level (theta_t) Summary:\n")
  cat(strrep("-", 75), "\n", sep = "")

  theta_print <- x$theta_summary
  theta_print[, -1] <- lapply(theta_print[, -1], function(col) {
    sprintf(paste0("%.", digits, "f"), col)
  })

  print(theta_print, row.names = FALSE, right = TRUE)

  cat("\n")
  cat(strrep("-", 75), "\n", sep = "")
  cat("Note: For time-varying parameters, use plot() to visualize trajectories.\n")
  cat("      For quick overview with medians only, use print().\n")
  cat(strrep("-", 75), "\n\n", sep = "")

  invisible(x)
}

