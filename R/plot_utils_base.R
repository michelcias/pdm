#' Internal plotting utilities for base R graphics
#'
#' @description This file contains shared plotting functions used by all
#'   plot methods in the pdm package. These functions implement base R
#'   graphics for MCMC diagnostic plots, mixture parameter visualizations,
#'   and credible interval displays.
#'
#' @details These functions are not exported and are intended for internal
#'   use only by the plot.* methods. They provide a consistent visual
#'   style across all model types (locallevel, localtrend, localacceleration).
#'
#' @keywords internal
#' @noRd
NULL


#' Plot 4-panel MCMC diagnostics for a single parameter (base graphics)
#'
#' @description Creates a comprehensive diagnostic page with trace plot,
#'   autocorrelation function, posterior density, and running mean convergence
#'   diagnostic for a single MCMC parameter.
#'
#' @param param_samples Numeric vector of MCMC samples for the parameter.
#' @param param_name Expression or quoted expression for the parameter name
#'   (used in plot title), e.g., \code{quote(mu[1])}.
#' @param param_label Expression for axis labels, e.g., \code{expression(mu[1])}.
#' @param true_value Numeric or NULL. If provided, adds a reference line at
#'   the true parameter value (useful for simulation studies).
#' @param ... Additional arguments (currently unused, for future extensibility).
#'
#' @return NULL (invisibly). Function is called for its side effects (plotting).
#'
#' @details The four diagnostic panels are:
#'   \enumerate{
#'     \item Trace plot with median reference line
#'     \item Autocorrelation function (ACF)
#'     \item Posterior density with mean and median lines
#'     \item Running mean to assess convergence
#'   }
#'
#' @keywords internal
#' @noRd
plot_param_diagnostics_base <- function(param_samples,
                                        param_name,
                                        param_label,
                                        true_value = NULL,
                                        ...) {

  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  par(mfrow = c(2, 2), mar = c(4, 4, 3, 1), oma = c(0, 0, 3, 0),
      mgp = c(2.5, 1, 0))

  # 1. Trace Plot
  range_param <- range(param_samples)
  range_param[2] <- range_param[2] + 0.25 * diff(range_param)

  plot(param_samples, type = "l", col = "gray40", lwd = 0.8,
       xlab = "Iteration", ylab = param_label, ylim = range_param,
       main = "Trace Plot")
  segments(x0 = 1, y0 = median(param_samples),
           x1 = length(param_samples), y1 = median(param_samples),
           col = "red", lwd = 2, lty = 2)
  if (!is.null(true_value)) {
    segments(x0 = 1, y0 = true_value,
             x1 = length(param_samples), y1 = true_value,
             col = "blue", lwd = 2, lty = 3)
    legend("topright",
           legend = c("Trace", "Median", "True Value"),
           col = c("gray40", "red", "blue"),
           lwd = c(0.8, 2, 2),
           horiz = TRUE,
           lty = c(1, 2, 3),
           bty = "n", cex = 0.8)
  } else {
    legend("topright",
           legend = c("Trace", "Median"),
           col = c("gray40", "red"),
           lwd = c(0.8, 2), horiz = TRUE,
           lty = c(1, 2), bty = "n", cex = 0.8)
  }
  grid()

  # 2. Autocorrelation Function
  acf(param_samples, main = "Autocorrelation",
      col = "steelblue", lwd = 2)

  # 3. Posterior Density
  dens <- density(param_samples)
  plot(dens, main = "Posterior Density",
       xlab = param_label, lwd = 2, col = "darkgreen")
  polygon(dens, col = grDevices::rgb(0, 0.5, 0, 0.2), border = NA)
  segments(x0 = median(param_samples), y0 = 0,
           x1 = median(param_samples), y1 = max(dens$y),
           col = "red", lwd = 2, lty = 2)
  segments(x0 = mean(param_samples), y0 = 0,
           x1 = mean(param_samples), y1 = max(dens$y),
           col = "blue", lwd = 2, lty = 3)
  legend("topright",
         legend = c("Median", "Mean"),
         col = c("red", "blue"),
         lty = c(2, 3), lwd = 2, bty = "n", cex = 0.8)

  # 4. Running Mean (Convergence Check)
  running_mean <- cumsum(param_samples) / seq_along(param_samples)
  range_runnint <- range(running_mean, median(param_samples))
  range_runnint[2] <- range_runnint[2] + 0.25 * diff(range_runnint)

  plot(running_mean, type = "l", col = "steelblue", lwd = 2,
       xlab = "Iteration", ylab = param_label, ylim = range_runnint,
       main = "Running Mean")
  segments(x0 = 1, y0 = median(param_samples),
           x1 = length(param_samples), y1 = median(param_samples),
           col = "red", lwd = 2, lty = 2)
  if (!is.null(true_value)) {
    segments(x0 = 1, y0 = true_value,
             x1 = length(param_samples), y1 = true_value,
             col = "blue", lwd = 2, lty = 3)
    legend("topright",
           legend = c("Running Mean", "Median", "True Value"),
           col = c("steelblue", "red", "blue"),
           lwd = c(2, 2, 2),
           horiz = TRUE,
           lty = c(1, 2, 3),
           bty = "n", cex = 0.8)
  } else {
    legend("topright",
           legend = c("Running Mean", "Median"),
           col = c("steelblue", "red"),
           lwd = c(2, 2),
           horiz = TRUE,
           lty = c(1, 2),
           bty = "n", cex = 0.8)
  }
  grid()

  # Overall title
  mtext(substitute(paste("MCMC Diagnostics: ", x), list(x = param_name)),
        outer = TRUE, cex = 1.3, font = 2)

  invisible(NULL)
}


#' Plot mixture component bivariate relationships (base graphics)
#'
#' @description Creates a 2x2 grid of scatterplots showing relationships
#'   between mixture component parameters (means and precisions).
#'
#' @param mu_1 Numeric vector of MCMC samples for component 1 mean.
#' @param mu_2 Numeric vector of MCMC samples for component 2 mean.
#' @param prec_1 Numeric vector of MCMC samples for component 1 precision.
#' @param prec_2 Numeric vector of MCMC samples for component 2 precision.
#' @param which Integer vector specifying which subplots to display (1:4).
#'   If NULL, all four plots are shown.
#' @param ... Additional arguments (currently unused).
#'
#' @return NULL (invisibly). Function is called for its side effects (plotting).
#'
#' @details The four panels show:
#'   \enumerate{
#'     \item mu_1 vs mu_2 (component separation)
#'     \item mu_1 vs phi_1 (mean-precision relationship for component 1)
#'     \item mu_2 vs phi_2 (mean-precision relationship for component 2)
#'     \item phi_1 vs phi_2 (precision comparison)
#'   }
#'
#' @keywords internal
#' @noRd
plot_mixture_params_base <- function(mu_1, mu_2, prec_1, prec_2,
                                     which = NULL, ...) {

  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  if (is.null(which)) which <- 1:4

  par(mfrow = c(2, 2), mar = c(4, 4, 2, 1), oma = c(0, 0, 2, 0),
      mgp = c(2.5, 1, 0))

  # Plot 1: mu_1 vs mu_2
  if (1 %in% which) {
    plot(mu_1, mu_2,
         xlab = expression(mu[1]),
         ylab = expression(mu[2]),
         main = expression(paste(mu[1], " vs ", mu[2])),
         pch = 16, col = grDevices::rgb(0.2, 0.5, 0.8, 0.15))
    grid()
  }

  # Plot 2: mu_1 vs phi_1
  if (2 %in% which) {
    plot(mu_1, prec_1,
         xlab = expression(mu[1]),
         ylab = expression(phi[1]),
         main = expression(paste(mu[1], " vs ", phi[1])),
         pch = 16, col = grDevices::rgb(0.2, 0.5, 0.8, 0.15))
    grid()
  }

  # Plot 3: mu_2 vs phi_2
  if (3 %in% which) {
    plot(mu_2, prec_2,
         xlab = expression(mu[2]),
         ylab = expression(phi[2]),
         main = expression(paste(mu[2], " vs ", phi[2])),
         pch = 16, col = grDevices::rgb(0, 0.5, 0, 0.15))
    grid()
  }

  # Plot 4: phi_1 vs phi_2
  if (4 %in% which) {
    plot(prec_1, prec_2,
         xlab = expression(phi[1]),
         ylab = expression(phi[2]),
         main = expression(paste(phi[1], " vs ", phi[2])),
         pch = 16, col = grDevices::rgb(0.5, 0, 0.5, 0.15))
    grid()
  }

  mtext("Mixture Component Parameters (Bivariate Relationships)",
        outer = TRUE, cex = 1.3, font = 2)

  invisible(NULL)
}


#' Plot mixture weight trajectory with credible intervals (base graphics)
#'
#' @description Creates a two-page visualization of mixture weights:
#'   Page 1 shows alpha_t trajectory with credible bands,
#'   Page 2 shows posterior probabilities P(z_t = 1 | data).
#'
#' @param alpha Matrix of MCMC samples for mixture weights (n_chain x n_obs).
#' @param z Matrix of MCMC samples for component indicators (n_chain x n_obs).
#' @param ci Logical; whether to display credible intervals.
#' @param ci_level Numeric between 0 and 1; credible interval level.
#' @param ... Additional arguments (currently unused).
#'
#' @return NULL (invisibly). Function is called for its side effects (plotting).
#'
#' @details
#'   Page 1 displays the median trajectory of alpha_t with optional credible
#'   bands. Page 2 shows the posterior mean of z_t, colored by the decision
#'   threshold at 0.5.
#'
#' @keywords internal
#' @noRd
plot_mixture_weights_base <- function(alpha, z, ci = TRUE,
                                      ci_level = 0.95, ...) {

  if (ci && (!is.numeric(ci_level) || length(ci_level) != 1 ||
             ci_level <= 0 || ci_level >= 1)) {
    stop("`ci_level` must be a single numeric value between 0 and 1")
  }

  n_obs <- ncol(alpha)
  time_grid <- seq_len(n_obs)

  # Compute credible bands for alpha
  alpha_median <- apply(alpha, 2, stats::median)
  if (ci) {
    ci_lower_prob <- (1 - ci_level) / 2
    ci_upper_prob <- 1 - ci_lower_prob
    alpha_lower <- apply(alpha, 2, stats::quantile, probs = ci_lower_prob)
    alpha_upper <- apply(alpha, 2, stats::quantile, probs = ci_upper_prob)
    ci_label <- paste0(round(ci_level * 100), "% CI")
  }

  # =========================================================================
  # Page 1: alpha_t trajectory
  # =========================================================================

  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar), add = TRUE)

  par(mfrow = c(1, 1), mar = c(4, 4, 2, 1), oma = c(0, 0, 2, 0),
      mgp = c(2.5, 1, 0))

  plot(time_grid, alpha_median, type = "l", lwd = 2.5, col = "blue",
       xlab = "Time", ylab = expression(alpha[t]),
       ylim = c(0, 1.1), axes = FALSE)

  axis(side = 1)
  axis(side = 2, at = seq(0, 1, by = 0.2))

  # Credible band
  if (ci) {
    polygon(c(time_grid, rev(time_grid)),
            c(alpha_lower, rev(alpha_upper)),
            col = grDevices::rgb(0.2, 0.5, 0.8, 0.3), border = NA)
  }

  # Re-draw median on top
  lines(time_grid, alpha_median, lwd = 2.5, col = "blue")

  grid()

  # Legend
  if (ci) {
    legend("topright", horiz = TRUE,
           legend = c(expression(hat(alpha)[t]), ci_label),
           col = c("blue", grDevices::rgb(0.2, 0.5, 0.8, 0.3)),
           lty = c(1, 1), lwd = c(2.5, 10),
           bty = "n")
  } else {
    legend("topright", horiz = TRUE,
           legend = expression(hat(alpha)[t]),
           col = "blue",
           lty = 1, lwd = 2.5,
           bty = "n")
  }

  mtext(expression(paste("Time-Varying Mixture Weight: ", alpha[t])),
        outer = TRUE, cex = 1.3, font = 2)

  # =========================================================================
  # Page 2: z_t posterior probabilities
  # =========================================================================

  par(mfrow = c(1, 1), mar = c(4, 4, 2, 1), oma = c(0, 0, 2, 0))

  z_prob <- apply(z, 2, mean)

  plot(z_prob,
       type = "h",
       lwd = 2,
       col = ifelse(z_prob > 0.5, "purple", "blue"),
       xlab = "Time",
       ylab = expression(paste("P(", z[t], " = 1 | data)")),
       ylim = c(0, 1.1),
       axes = FALSE)

  axis(side = 1)
  axis(side = 2, at = c(0, 0.5, 1))

  # Add threshold line
  segments(x0 = 1, y0 = 0.5, x1 = n_obs, y1 = 0.5,
           col = "red", lwd = 2, lty = 2)

  legend("topright",
         horiz = TRUE,
         legend = c(expression(paste("P(", z[t], " = 1 | data)") > 0.5),
                    expression(paste("P(", z[t], " = 1 | data)") <= 0.5),
                    "Threshold"),
         col = c("purple", "blue", "red"),
         lty = c(1, 1, 2), lwd = 2,
         bty = "n")

  grid(nx = NA, ny = NULL)

  mtext(expression(paste("Posterior Probability: P(", z[t], " = 1 | data)")),
        outer = TRUE, cex = 1.3, font = 2)

  invisible(NULL)
}


#' Validate credible interval level
#'
#' @description Internal helper to validate ci_level parameter.
#'
#' @param ci_level Numeric value to validate.
#'
#' @return NULL if valid, stops with error message if invalid.
#'
#' @keywords internal
#' @noRd
validate_ci_level <- function(ci_level) {
  if (!is.numeric(ci_level) || length(ci_level) != 1 ||
      ci_level <= 0 || ci_level >= 1) {
    stop("`ci_level` must be a single numeric value between 0 and 1")
  }
  invisible(NULL)
}
