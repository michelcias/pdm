#' Summary method for probit_bernoulli_locallevel objects
#'
#' @description Produces comprehensive posterior statistics including means,
#'   standard deviations, medians, and credible intervals.
#'
#' @param object An object of class `probit_bernoulli_locallevel`, typically
#'   the result of calling \code{\link{mcmc_probit_bernoulli_locallevel}}.
#' @param ci_level Credible interval level; a single numeric value strictly
#'   between 0 and 1. Defaults to `0.95`. The reported interval is the
#'   Highest Posterior Density Interval (HPDI), i.e. the shortest contiguous
#'   interval containing that probability mass of the posterior.
#' @param ... Additional arguments (currently unused).
#'
#' @return An object of class `summary.probit_bernoulli_locallevel`, which is
#'   a list containing:
#'   \describe{
#'     \item{\code{model_type}}{Character string indicating the model type}
#'     \item{\code{n_obs}}{Number of observations}
#'     \item{\code{n_draws}}{Number of MCMC samples}
#'     \item{\code{burnin}}{Number of burn-in iterations}
#'     \item{\code{thinning}}{Thinning interval}
#'     \item{\code{ci_level}}{Credible interval level used (HPDI)}
#'     \item{\code{scalar_params}}{Data frame with summary statistics for
#'       scalar parameters (theta_01, W_1^-1)}
#'     \item{\code{theta1_summary}}{Summary statistics for the latent level
#'       \eqn{\theta_{t,1}} aggregated across time}
#'     \item{\code{alpha_summary}}{Summary statistics for the Bernoulli
#'       probabilities \eqn{\alpha_t} aggregated across time (min, median, max)}
#'   }
#'
#' @details
#' This method provides complete posterior inference with multiple statistics:
#' \describe{
#'   \item{\strong{Mean}}{Expected value under the posterior (minimizes squared error loss)}
#'   \item{\strong{Median}}{Typical value (minimizes absolute error loss, shown in `print()`)}
#'   \item{\strong{SD}}{Posterior standard deviation (measure of uncertainty)}
#'   \item{\strong{CI}}{Credible intervals at specified probabilities}
#' }
#'
#' For a quick overview showing only medians, use `print(x)`.
#'
#' For time-varying parameters like \eqn{\theta_{t,1}} and \eqn{\alpha_t},
#' only summary statistics across time are reported. Use `plot()` to
#' visualize the full trajectories.
#'
#' @examples
#' \donttest{
#' ## Simulation of data
#' n <- 200  # Number of observations to simulate
#'
#' set.seed(123)
#'
#' # Generate true success probabilities following a sinusoidal pattern
#' alpha_true <- (sin(2 * pi * seq_len(n) / n) + 2) / 4
#'
#' # Generate Bernoulli observations
#' y <- rbinom(n, size = 1, prob = alpha_true)
#'
#' ## Running the Gibbs sampler
#' out <- mcmc_probit_bernoulli_locallevel(
#'   y,
#'   burnin             = 1000,
#'   thinning           = 20,
#'   n_draws            = 500,
#'   prior_theta01_mean = 0,
#'   prior_theta01_prec = 1,
#'   prior_prec1_shape  = 100,
#'   prior_prec1_rate   = 1,
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
#' summary(out, ci_level = 0.90)  # 90% HPD interval
#' summary(out, ci_level = 0.80)  # 80% HPD interval
#' }
#'
#' @seealso
#'   \code{\link{mcmc_probit_bernoulli_locallevel}} (model generator),
#'   \code{\link{plot.probit_bernoulli_locallevel}}, \code{\link{print.probit_bernoulli_locallevel}}.
#'
#' @export
summary.probit_bernoulli_locallevel <- function(object,
                                                ci_level = 0.95,
                                                ...) {

  # Validate input
  validate_summary_input(object, ci_level, "probit_bernoulli_locallevel")

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

  # Alpha (Bernoulli probabilities) summary aggregated across time (simple version)
  alpha_summary <- format_timevarying_summary(object$alpha, ci_level, "simple")

  # Create summary object
  result <- list(
    model_type = attr(object, "model_type"),
    n_obs = attr(object, "n_obs"),
    n_draws = attr(object, "n_draws"),
    burnin = attr(object, "burnin"),
    thinning = attr(object, "thinning"),
    ci_level = ci_level,
    scalar_params = scalar_params,
    theta1_summary = theta1_summary,
    alpha_summary = alpha_summary
  )

  class(result) <- "summary.probit_bernoulli_locallevel"
  return(result)
}


#' Print method for summary.probit_bernoulli_locallevel objects
#'
#' @description Prints comprehensive posterior statistics in a readable format.
#'
#' @param x An object of class `summary.probit_bernoulli_locallevel`, typically
#'   the result of calling `summary()` on a `probit_bernoulli_locallevel`
#'   object.
#' @param digits Integer, number of significant digits to display. Default is 3.
#' @param ... Additional arguments (currently unused).
#'
#' @return Invisibly returns the input object `x`.
#'
#' @examples
#' \donttest{
#' ## Simulation of data
#' n <- 200
#' set.seed(123)
#' alpha_true <- (sin(2 * pi * seq_len(n) / n) + 2) / 4
#' y <- rbinom(n, size = 1, prob = alpha_true)
#'
#' out <- mcmc_probit_bernoulli_locallevel(
#'   y,
#'   burnin             = 1000,
#'   thinning           = 20,
#'   n_draws            = 500,
#'   prior_theta01_mean = 0,
#'   prior_theta01_prec = 1,
#'   prior_prec1_shape  = 100,
#'   prior_prec1_rate   = 1,
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
#' @seealso \code{\link{summary.probit_bernoulli_locallevel}},
#'   \code{\link{print.probit_bernoulli_locallevel}}
#'
#' @export
print.summary.probit_bernoulli_locallevel <- function(x, digits = 3, ...) {

  cat("\n")
  cat("Summary: Probit-Bernoulli Local-Level Model\n")
  cat(strrep("=", 75), "\n\n", sep = "")

  # Model information
  cat("Model Information:\n")
  cat("  Type:              ", x$model_type, "\n", sep = "")
  cat("  Observations:      ", x$n_obs, "\n", sep = "")
  cat("  MCMC draws:        ", x$n_draws, "\n", sep = "")
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
                        "Latent Level (theta_{t,1}) Summary (probit scale):")

  # Bernoulli probabilities summary
  print_formatted_table(x$alpha_summary, digits,
                        "Bernoulli Probabilities (alpha_t) Summary:")

  # Print footer
  print_summary_footer()

  invisible(x)
}
