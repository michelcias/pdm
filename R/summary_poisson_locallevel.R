#' Summary method for poisson_locallevel objects
#'
#' @description Produces comprehensive posterior statistics including means,
#'   standard deviations, medians, and credible intervals.
#'
#' @param object An object of class \code{poisson_locallevel}, typically
#'   the result of calling \code{\link{mcmc_poisson_locallevel}}.
#' @param ci_level Credible interval level; a single numeric value strictly
#'   between 0 and 1. Defaults to \code{0.95}. The reported interval is the
#'   Highest Posterior Density Interval (HPDI), i.e. the shortest contiguous
#'   interval containing that probability mass of the posterior.
#' @param ... Additional arguments (currently unused).
#'
#' @return An object of class \code{summary.poisson_locallevel}, which is
#'   a list containing:
#'   \describe{
#'     \item{\code{model_type}}{Character string indicating the model type}
#'     \item{\code{n_obs}}{Number of observations}
#'     \item{\code{n_chain}}{Number of MCMC samples}
#'     \item{\code{burnin}}{Number of burn-in iterations}
#'     \item{\code{thinning}}{Thinning interval}
#'     \item{\code{ci_level}}{Credible interval level used (HPDI)}
#'     \item{\code{scalar_params}}{Data frame with summary statistics for
#'       scalar parameters (theta_01, W_1^-1)}
#'     \item{\code{theta1_summary}}{Summary statistics for the latent level
#'       \eqn{\theta_{t,1}} aggregated across time}
#'     \item{\code{alpha_summary}}{Summary statistics for the Poisson rates
#'       \eqn{\alpha_t} aggregated across time (min, median, max)}
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
#' For time-varying parameters like \eqn{\theta_{t,1}} and \eqn{\alpha_t},
#' only summary statistics across time are reported. Use \code{plot()} to
#' visualize the full trajectories.
#'
#' @examples
#' \donttest{
#' ## Simulation of data
#' n <- 500        # Number of observations to simulate
#'
#' # True parameters for simulation:
#' theta0_true <- 0.5     # Initial state (theta[01]) on log scale
#' prec1_true <- 100      # Innovation precision (1/W[1])
#'
#' # Use a fixed seed for data simulation
#' set.seed(123)
#'
#' # Generate noise terms:
#' u1 <- rnorm(n, sd = sqrt(1/prec1_true))  # Evolution noise (u1[t])
#'
#' # Simulate latent states and observations:
#' theta1_true <- cumsum(c(theta0_true, u1))[-1]  # theta[t1] series on log scale
#' alpha_true <- exp(pmin(theta1_true, 10))  # cap to prevent Inf

#' y <- rpois(n, lambda = alpha_true)             # Observed Poisson counts
#'
#' ## Running the Gibbs sampler
#' out <- mcmc_poisson_locallevel(
#'   y,
#'   burnin                  = 1000,
#'   thinning                = 50,
#'   n_chain                 = 1000,
#'   prior_theta01_mean      = 0,
#'   prior_theta01_prec      = 1,
#'   prior_prec1_shape       = 100,
#'   prior_prec1_rate        = 1,
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
#' # Quick overview (medians only)
#' print(out)
#'
#' # Full statistics
#' summary(out)
#'
#' # Custom credible intervals
#' summary(out, ci_level = 0.90)  # 90% HPD interval
#' summary(out, ci_level = 0.80)  # 80% HPD interval
#' }
#'
#' @seealso \code{\link{mcmc_poisson_locallevel}},
#'   \code{\link{print.poisson_locallevel}}
#'
#' @export
summary.poisson_locallevel <- function(object,
                                       ci_level = 0.95,
                                       ...) {

  # Validate input
  validate_summary_input(object, ci_level, "poisson_locallevel")

  # Scalar parameters (initial state and precision)
  scalar_params <- data.frame(
    Parameter = c("theta_01", "W_1^-1"),
    rbind(
      compute_summary_stats(object$theta_01, ci_level),
      compute_summary_stats(object$prec_theta1, ci_level)
    )
  )

  # Latent level summary aggregated across time (detailed version)
  theta1_summary <- format_timevarying_summary(object$theta_1, ci_level, "detailed")

  # Alpha (Poisson rates) summary aggregated across time (simple version)
  alpha_summary <- format_timevarying_summary(object$alpha, ci_level, "simple")

  # Create summary object
  result <- list(
    model_type = attr(object, "model_type"),
    n_obs = attr(object, "n_obs"),
    n_chain = attr(object, "n_chain"),
    burnin = attr(object, "burnin"),
    thinning = attr(object, "thinning"),
    ci_level = ci_level,
    scalar_params = scalar_params,
    theta1_summary = theta1_summary,
    alpha_summary = alpha_summary
  )

  class(result) <- "summary.poisson_locallevel"
  return(result)
}


#' Print method for summary.poisson_locallevel objects
#'
#' @description Prints comprehensive posterior statistics in a readable format.
#'
#' @param x An object of class \code{summary.poisson_locallevel}, typically
#'   the result of calling \code{summary()} on a \code{poisson_locallevel}
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
#' theta0_true <- 0.5
#' prec1_true <- 100
#'
#' set.seed(123)
#' u1 <- rnorm(n, sd = sqrt(1/prec1_true))
#'
#' theta1_true <- cumsum(c(theta0_true, u1))[-1]
#' alpha_true <- exp(pmin(theta1_true, 10))  # cap to prevent Inf
#' y <- rpois(n, lambda = alpha_true)
#'
#' out <- mcmc_poisson_locallevel(
#'   y,
#'   burnin                  = 1000,
#'   thinning                = 50,
#'   n_chain                 = 1000,
#'   prior_theta01_mean      = 0,
#'   prior_theta01_prec      = 1,
#'   prior_prec1_shape       = 100,
#'   prior_prec1_rate        = 1,
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
#' @seealso \code{\link{summary.poisson_locallevel}},
#'   \code{\link{print.poisson_locallevel}}
#'
#' @export
print.summary.poisson_locallevel <- function(x, digits = 3, ...) {

  cat("\n")
  cat("Summary: Log-Poisson Local-Level Model\n")
  cat(strrep("=", 75), "\n\n", sep = "")

  # Model information
  cat("Model Information:\n")
  cat("  Type:              ", x$model_type, "\n", sep = "")
  cat("  Observations:      ", x$n_obs, "\n", sep = "")
  cat("  MCMC samples:      ", x$n_chain, "\n", sep = "")
  cat("  Burn-in:           ", x$burnin, "\n", sep = "")
  cat("  Thinning:          ", x$thinning, "\n\n", sep = "")

  # Credible interval level
  ci_level_pct <- x$ci_level * 100
  cat("Credible Intervals (HPDI): ", sprintf("%.1f", ci_level_pct), "%\n\n", sep = "")

  # Explanation of statistics
  cat("Statistics Legend:\n")
  cat("  Mean   = Posterior mean (minimizes squared error)\n")
  cat("  Median = Posterior median (minimizes absolute error, shown in print())\n")
  cat("  SD     = Posterior standard deviation\n")
  cat("  CI     = HPD interval (shortest interval at the specified level)\n\n")

  # Scalar parameters
  print_formatted_table(x$scalar_params, digits, "Scalar Parameters:")

  # Latent level summary
  print_formatted_table(x$theta1_summary, digits,
                        "Latent Level (theta_{t,1}) Summary (log scale):")

  # Poisson rates summary
  print_formatted_table(x$alpha_summary, digits,
                        "Poisson Rates (alpha_t) Summary:")

  # Print footer
  print_summary_footer()

  invisible(x)
}
