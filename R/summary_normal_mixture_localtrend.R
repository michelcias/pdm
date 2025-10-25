#' Summary method for normal_mixture_localtrend objects
#'
#' @description Produces comprehensive posterior statistics including means,
#'   standard deviations, medians, and credible intervals.
#'
#' @param object An object of class \code{normal_mixture_localtrend}, typically
#'   the result of calling \code{\link{mcmc_normal_mixture_localtrend}}.
#' @param probs Numeric vector of probabilities for credible intervals.
#'   Default is \code{c(0.025, 0.975)} for 95\% credible intervals.
#' @param ... Additional arguments (currently unused).
#'
#' @return An object of class \code{summary.normal_mixture_localtrend}, which is
#'   a list containing:
#'   \describe{
#'     \item{\code{link}}{Character string indicating the link function used}
#'     \item{\code{model_type}}{Character string indicating the model type}
#'     \item{\code{n_obs}}{Number of observations}
#'     \item{\code{n_chain}}{Number of MCMC samples}
#'     \item{\code{burnin}}{Number of burn-in iterations}
#'     \item{\code{thinning}}{Thinning interval}
#'     \item{\code{probs}}{Probabilities used for credible intervals}
#'     \item{\code{mixture_params}}{Data frame with summary statistics for
#'       mixture component parameters (mu_1, mu_2, phi_1, phi_2)}
#'     \item{\code{state_params}}{Data frame with summary statistics for
#'       dynamic state parameters (theta_01, theta_02, W_1^{-1}, W_2^{-1})}
#'     \item{\code{alpha_summary}}{Summary statistics for the mixture weights
#'       alpha_t (min, median, max across time)}
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
#' For time-varying parameters like alpha_t, only summary statistics across
#' time are reported. Use \code{plot()} to visualize the full trajectories.
#'
#' @examples
#' \dontrun{
#' # Run MCMC
#' out <- mcmc_normal_mixture_localtrend(y, link = "logit", ...)
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
#' @seealso \code{\link{mcmc_normal_mixture_localtrend}},
#'   \code{\link{print.normal_mixture_localtrend}},
#'   \code{\link{plot.normal_mixture_localtrend}}
#'
#' @export
summary.normal_mixture_localtrend <- function(object,
                                              probs = c(0.025, 0.975),
                                              ...) {

  if (!inherits(object, "normal_mixture_localtrend")) {
    stop("Object must be of class 'normal_mixture_localtrend'")
  }

  # Validate input
  if (!inherits(object, "normal_mixture_localtrend")) {
    stop("Object must be of class 'normal_mixture_localtrend'")
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

  # Mixture component parameters
  mixture_params <- data.frame(
    Parameter = c("mu_1", "mu_2", "phi_1", "phi_2"),
    rbind(
      compute_stats(object$mu_1, probs),
      compute_stats(object$mu_2, probs),
      compute_stats(object$prec_1, probs),
      compute_stats(object$prec_2, probs)
    )
  )

  # Dynamic state parameters
  state_params <- data.frame(
    Parameter = c("theta_01", "theta_02", "W_1^-1", "W_2^-1"),
    rbind(
      compute_stats(object$theta_01, probs),
      compute_stats(object$theta_02, probs),
      compute_stats(object$prec_theta1, probs),
      compute_stats(object$prec_theta2, probs)
    )
  )

  # Alpha summary (across time)
  alpha_median_time <- apply(object$alpha, 2, median)
  alpha_summary <- data.frame(
    Statistic = c("Min across time", "Median across time", "Max across time"),
    Value = c(
      min(alpha_median_time),
      median(alpha_median_time),
      max(alpha_median_time)
    )
  )

  # Create summary object
  result <- list(
    link = attr(object, "link"),
    model_type = attr(object, "model_type"),
    n_obs = attr(object, "n_obs"),
    n_chain = attr(object, "n_chain"),
    burnin = attr(object, "burnin"),
    thinning = attr(object, "thinning"),
    probs = probs,
    mixture_params = mixture_params,
    state_params = state_params,
    alpha_summary = alpha_summary
  )

  class(result) <- "summary.normal_mixture_localtrend"
  return(result)
}


#' Print method for summary.normal_mixture_localtrend objects
#'
#' @description Prints comprehensive posterior statistics in a readable format.
#'
#' @param x An object of class \code{summary.normal_mixture_localtrend}, typically
#'   the result of calling \code{summary()} on a \code{normal_mixture_localtrend}
#'   object.
#' @param digits Integer, number of significant digits to display. Default is 3.
#' @param ... Additional arguments (currently unused).
#'
#' @return Invisibly returns the input object \code{x}.
#'
#' @examples
#' \dontrun{
#' out <- mcmc_normal_mixture_localtrend(y, link = "logit", ...)
#' s <- summary(out)
#' print(s)
#' # or simply:
#' summary(out)
#' }
#'
#' @export
print.summary.normal_mixture_localtrend <- function(x, digits = 3, ...) {

  cat("\n")
  cat("Summary: Gaussian Mixture Model with Dynamic Mixture Weights\n")
  cat(strrep("=", 75), "\n\n", sep = "")

  # Model information
  cat("Model Information:\n")
  cat("  Type:              ", x$model_type, "\n", sep = "")
  cat("  Link function:     ", x$link, "\n", sep = "")
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

  # Mixture component parameters
  cat("Mixture Component Parameters:\n")
  cat(strrep("-", 75), "\n", sep = "")

  # Format the table
  mixture_print <- x$mixture_params
  mixture_print[, -1] <- lapply(mixture_print[, -1], function(col) {
    sprintf(paste0("%.", digits, "f"), col)
  })

  print(mixture_print, row.names = FALSE, right = TRUE)
  cat("\n")

  # Dynamic state parameters
  cat("Dynamic State Parameters:\n")
  cat(strrep("-", 75), "\n", sep = "")

  state_print <- x$state_params
  state_print[, -1] <- lapply(state_print[, -1], function(col) {
    sprintf(paste0("%.", digits, "f"), col)
  })

  print(state_print, row.names = FALSE, right = TRUE)
  cat("\n")

  # Alpha summary
  cat("Mixture Weights (alpha_t) Summary:\n")
  cat(strrep("-", 75), "\n", sep = "")

  alpha_print <- x$alpha_summary
  alpha_print$Value <- sprintf(paste0("%.", digits, "f"), alpha_print$Value)

  print(alpha_print, row.names = FALSE, right = TRUE)

  cat("\n")
  cat(strrep("-", 75), "\n", sep = "")
  cat("Note: For time-varying parameters, use plot() to visualize trajectories.\n")
  cat("      For quick overview with medians only, use print().\n")
  cat(strrep("-", 75), "\n\n", sep = "")

  invisible(x)
}
