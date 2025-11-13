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
#' @param color Character; color for the main diagnostic lines. Default is
#'   "steelblue". This should match the parameter type:
#'   \itemize{
#'     \item "steelblue" for level-related parameters (theta_1, theta_01, W_1^{-1})
#'     \item "firebrick" for trend-related parameters (theta_2, theta_02, W_2^{-1})
#'     \item "darkgreen" for acceleration-related parameters (theta_3, theta_03, W_3^{-1})
#'     \item "gray30" or other neutral colors for observation/mixture parameters
#'   }
#' @param ... Additional arguments (currently unused, for future extensibility).
#'
#' @return NULL (invisibly). Function is called for its side effects (plotting).
#'
#' @details The four diagnostic panels are:
#'   \enumerate{
#'     \item Trace plot with median reference line (solid, parameter color)
#'     \item Autocorrelation function (ACF) using parameter color
#'     \item Posterior density with median line (solid, parameter color)
#'     \item Running mean to assess convergence (solid, parameter color)
#'   }
#'
#'   Estimated values use solid lines in the parameter's color.
#'   True values (when provided) use dashed lines in black or dark gray.
#'
#' @keywords internal
#' @noRd
plot_param_diagnostics_base <- function(param_samples,
                                        param_name,
                                        param_label,
                                        true_value = NULL,
                                        color = "steelblue",
                                        ...) {

  # Setup plotting environment
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  par(mfrow = c(2, 2),
      mar = c(4, 4, 3, 1),
      oma = c(0, 0, 3, 0),
      mgp = c(2.5, 1, 0))

  # =========================================================================
  # Panel 1: Trace Plot
  # =========================================================================

  # Compute y-axis range with buffer
  range_param <- range(param_samples)
  range_param[2] <- range_param[2] + 0.25 * diff(range_param)

  plot(param_samples,
       type = "l",
       col = "gray40",
       lwd = 0.8,
       xlab = "Iteration",
       ylab = param_label,
       ylim = range_param,
       main = "Trace Plot")

  # Add median reference line
  segments(x0 = 1,
           y0 = median(param_samples),
           x1 = length(param_samples),
           y1 = median(param_samples),
           col = color,
           lwd = 2,
           lty = 1)

  # Add true value reference line (if provided)
  if (!is.null(true_value)) {
    segments(x0 = 1,
             y0 = true_value,
             x1 = length(param_samples),
             y1 = true_value,
             col = "black",
             lwd = 2,
             lty = 2)
    legend("topright",
           legend = c("Trace", "Median", "True Value"),
           col = c("gray40", color, "black"),
           lwd = c(0.8, 2, 2),
           lty = c(1, 1, 2),
           horiz = TRUE,
           bty = "n",
           cex = 0.8)
  } else {
    legend("topright",
           legend = c("Trace", "Median"),
           col = c("gray40", color),
           lwd = c(0.8, 2),
           lty = c(1, 1),
           horiz = TRUE,
           bty = "n",
           cex = 0.8)
  }
  grid()

  # =========================================================================
  # Panel 2: Autocorrelation Function
  # =========================================================================

  acf(param_samples,
      main = "",
      col = color,
      lwd = 2)
  title(main = "Autocorrelation")
  grid()

  # =========================================================================
  # Panel 3: Posterior Density
  # =========================================================================

  dens <- density(param_samples)

  plot(dens,
       main = "Posterior Density",
       xlab = param_label,
       lwd = 2,
       col = color,
       ylim = c(0, max(dens$y) * 1.25))

  # Add shaded density area
  polygon(dens,
          col = grDevices::adjustcolor(color, alpha.f = 0.2),
          border = NA)

  # Add median reference line
  segments(x0 = median(param_samples),
           y0 = 0,
           x1 = median(param_samples),
           y1 = max(dens$y),
           col = color,
           lwd = 2,
           lty = 1)

  # Add true value reference line (if provided)
  if (!is.null(true_value)) {
    segments(x0 = true_value,
             y0 = 0,
             x1 = true_value,
             y1 = max(dens$y),
             col = "black",
             lwd = 2,
             lty = 2)
    legend("topright",
           legend = c("Median", "True Value"),
           col = c(color, "black"),
           lwd = c(2, 2),
           lty = c(1, 2),
           horiz = TRUE,
           bty = "n",
           cex = 0.8)
  } else {
    legend("topright",
           legend = "Median",
           col = color,
           lwd = 2,
           lty = 1,
           horiz = TRUE,
           bty = "n",
           cex = 0.8)
  }
  grid()

  # =========================================================================
  # Panel 4: Running Mean (Convergence Diagnostic)
  # =========================================================================

  running_mean <- cumsum(param_samples) / seq_along(param_samples)

  # Compute y-axis range with buffer
  range_running <- range(running_mean, mean(param_samples))
  range_running[2] <- range_running[2] + 0.25 * diff(range_running)

  plot(running_mean,
       type = "l",
       col = grDevices::adjustcolor(color, alpha.f = 0.4),
       lwd = 2,
       lty = 1,
       xlab = "Iteration",
       ylab = param_label,
       ylim = range_running,
       main = "Running Mean")

  # Add mean reference line
  segments(x0 = 1,
           y0 = mean(param_samples),
           x1 = length(param_samples),
           y1 = mean(param_samples),
           col = color,
           lwd = 2,
           lty = 3)

  # Add true value reference line (if provided)
  if (!is.null(true_value)) {
    segments(x0 = 1,
             y0 = true_value,
             x1 = length(param_samples),
             y1 = true_value,
             col = "black",
             lwd = 2,
             lty = 2)
    legend("topright",
           legend = c("Running Mean", "Mean", "True Value"),
           col = c(color, color, "black"),
           lwd = c(2, 2, 2),
           lty = c(1, 3, 2),
           horiz = TRUE,
           bty = "n",
           cex = 0.8)
  } else {
    legend("topright",
           legend = c("Running Mean", "Mean"),
           col = c(color, color),
           lwd = c(2, 2),
           lty = c(1, 3),
           horiz = TRUE,
           bty = "n",
           cex = 0.8)
  }
  grid()

  # =========================================================================
  # Overall Title
  # =========================================================================

  mtext(substitute(paste("MCMC Diagnostics: ", x), list(x = param_name)),
        outer = TRUE,
        cex = 1.3,
        font = 2)

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
#'   Note: Mixture parameters do not correspond to dynamic states (level, trend,
#'   acceleration), so they use neutral gray colors to avoid confusion with the
#'   state-specific color scheme (blue for level, red for trend, green for acceleration).
#'
#' @keywords internal
#' @noRd
plot_mixture_params_base <- function(mu_1,
                                     mu_2,
                                     prec_1,
                                     prec_2,
                                     which = NULL,
                                     ...) {

  # Setup plotting environment
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  if (is.null(which)) {
    which <- 1:4
  }

  par(mfrow = c(2, 2),
      mar = c(4, 4, 2, 1),
      oma = c(0, 0, 2, 0),
      mgp = c(2.5, 1, 0))

  # =========================================================================
  # Panel 1: mu_1 vs mu_2 (Component Separation)
  # =========================================================================

  if (1 %in% which) {
    plot(mu_1,
         mu_2,
         xlab = expression(mu[1]),
         ylab = expression(mu[2]),
         main = expression(paste(mu[1], " vs ", mu[2])),
         pch = 16,
         col = grDevices::rgb(0.3, 0.3, 0.3, 0.3))
    grid()
  }

  # =========================================================================
  # Panel 2: mu_1 vs phi_1 (Component 1 Mean-Precision)
  # =========================================================================

  if (2 %in% which) {
    plot(mu_1,
         prec_1,
         xlab = expression(mu[1]),
         ylab = expression(phi[1]),
         main = expression(paste(mu[1], " vs ", phi[1])),
         pch = 16,
         col = grDevices::rgb(0.3, 0.3, 0.3, 0.3))
    grid()
  }

  # =========================================================================
  # Panel 3: mu_2 vs phi_2 (Component 2 Mean-Precision)
  # =========================================================================

  if (3 %in% which) {
    plot(mu_2,
         prec_2,
         xlab = expression(mu[2]),
         ylab = expression(phi[2]),
         main = expression(paste(mu[2], " vs ", phi[2])),
         pch = 16,
         col = grDevices::rgb(0.3, 0.3, 0.3, 0.3))
    grid()
  }

  # =========================================================================
  # Panel 4: phi_1 vs phi_2 (Precision Comparison)
  # =========================================================================

  if (4 %in% which) {
    plot(prec_1,
         prec_2,
         xlab = expression(phi[1]),
         ylab = expression(phi[2]),
         main = expression(paste(phi[1], " vs ", phi[2])),
         pch = 16,
         col = grDevices::rgb(0.3, 0.3, 0.3, 0.3))
    grid()
  }

  # =========================================================================
  # Overall Title
  # =========================================================================

  mtext("Mixture Component Parameters (Bivariate Relationships)",
        outer = TRUE,
        cex = 1.3,
        font = 2)

  invisible(NULL)
}


# =============================================================================
# Generic Alpha Plotting Functions
# =============================================================================

#' Plot alpha trajectory with credible intervals (base graphics)
#'
#' @description Generic function to plot time-varying alpha_t trajectory
#'   with optional credible bands. Can overlay observed data and true values
#'   for simulation studies.
#'
#' @param alpha Matrix of MCMC samples for alpha (n_chain x n_obs).
#' @param ci Logical; whether to display credible intervals.
#' @param ci_level Numeric between 0 and 1; credible interval level.
#' @param title Character or expression; main title for the plot.
#' @param obs_data Numeric vector of observed data (proportions or binary).
#'   If NULL, no observations are plotted.
#' @param obs_label Character; legend label for observed data.
#' @param show_obs Logical; whether to display observed data points.
#'   Default is TRUE. Ignored if obs_data is NULL.
#' @param obs_color Character; color for observed data points.
#' @param obs_pch Integer; point character for observed data.
#' @param obs_cex Numeric; point size for observed data.
#' @param true_alpha Numeric vector; true alpha values for simulation studies.
#'   If provided, overlays the true trajectory.
#' @param ... Additional arguments (currently unused).
#'
#' @return NULL (invisibly). Function is called for its side effects (plotting).
#'
#' @details This is a generic plotting function used by both mixture models
#'   (for mixture weights) and binomial/Bernoulli models (for success
#'   probabilities). The function creates a single-page plot with:
#'   \itemize{
#'     \item Median trajectory of alpha_t (blue line)
#'     \item Optional credible interval band (gray)
#'     \item Optional observed data overlay (points)
#'     \item Optional true values (for simulation validation)
#'   }
#'
#' @keywords internal
#' @noRd
plot_alpha_trajectory_base <- function(alpha,
                                       ci = TRUE,
                                       ci_level = 0.95,
                                       title = NULL,
                                       obs_data = NULL,
                                       obs_label = "Observed",
                                       show_obs = TRUE,
                                       obs_color = "lightgreen",
                                       obs_pch = 16,
                                       obs_cex = 0.6,
                                       true_alpha = NULL,
                                       ...) {

  # =========================================================================
  # Validate Parameters
  # =========================================================================

  if (ci && (!is.numeric(ci_level) ||
             length(ci_level) != 1 ||
             ci_level <= 0 ||
             ci_level >= 1)) {
    stop("`ci_level` must be a single numeric value between 0 and 1")
  }

  if (!is.logical(show_obs) || length(show_obs) != 1) {
    stop("`show_obs` must be a single logical value")
  }

  # =========================================================================
  # Prepare Data
  # =========================================================================

  n_obs <- ncol(alpha)
  time_grid <- seq_len(n_obs)

  # Compute summary statistics
  alpha_median <- apply(alpha, 2, stats::median)

  if (ci) {
    ci_lower_prob <- (1 - ci_level) / 2
    ci_upper_prob <- 1 - ci_lower_prob
    alpha_lower <- apply(alpha, 2, stats::quantile, probs = ci_lower_prob)
    alpha_upper <- apply(alpha, 2, stats::quantile, probs = ci_upper_prob)
    ci_label <- paste0(round(ci_level * 100), "% CI")
  }

  # =========================================================================
  # Setup Plotting Environment
  # =========================================================================

  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar), add = TRUE)

  par(mfrow = c(1, 1),
      mar = c(4, 4, 2, 1),
      oma = c(0, 0, 2, 0),
      mgp = c(2.5, 1, 0))

  # =========================================================================
  # Create Base Plot
  # =========================================================================

  plot(time_grid,
       alpha_median,
       type = "l",
       lwd = 2.5,
       col = "steelblue",
       xlab = "Time",
       ylab = expression(alpha[t]),
       ylim = c(0, 1.1),
       axes = FALSE,
       main = "")

  axis(side = 1)
  axis(side = 2, at = seq(0, 1, by = 0.2))

  # =========================================================================
  # Add Observed Data (if provided and requested)
  # =========================================================================

  if (!is.null(obs_data) && show_obs) {
    points(time_grid,
           obs_data,
           pch = obs_pch,
           cex = obs_cex,
           col = obs_color)
  }

  # =========================================================================
  # Add True Alpha (if provided)
  # =========================================================================

  if (!is.null(true_alpha)) {
    lines(time_grid,
          true_alpha,
          lwd = 2.5,
          col = "black",
          lty = 2)
  }

  # =========================================================================
  # Add Credible Interval Band
  # =========================================================================

  if (ci) {
    polygon(c(time_grid, rev(time_grid)),
            c(alpha_lower, rev(alpha_upper)),
            col = grDevices::adjustcolor("steelblue", alpha.f = 0.2),
            border = NA)
    # Redraw median line on top
    lines(time_grid,
          alpha_median,
          lwd = 2.5,
          col = "steelblue")
  }

  # =========================================================================
  # Add Grid Lines
  # =========================================================================

  grid(nx = NA, ny = NULL)
  segments(x0 = axTicks(1),
           y0 = -0.04,
           x1 = axTicks(1),
           y1 = 1.03,
           col = "lightgray",
           lwd = par("lwd"),
           lty = "dotted")

  # =========================================================================
  # Build Legend
  # =========================================================================

  legend_items <- c(expression(hat(alpha)[t]))
  legend_cols <- c("steelblue")
  legend_lty <- c(1)
  legend_lwd <- c(2.5)
  legend_pch <- c(NA)

  # Add credible interval to legend
  if (ci) {
    legend_items <- c(legend_items, ci_label)
    legend_cols <- c(legend_cols,
                     grDevices::adjustcolor("steelblue", alpha.f = 0.3))
    legend_lty <- c(legend_lty, 1)
    legend_lwd <- c(legend_lwd, 10)
    legend_pch <- c(legend_pch, NA)
  }

  # Add true alpha to legend
  if (!is.null(true_alpha)) {
    legend_items <- c(expression(alpha[t]), legend_items)
    legend_cols <- c("black", legend_cols)
    legend_lty <- c(2, legend_lty)
    legend_lwd <- c(2.5, legend_lwd)
    legend_pch <- c(NA, legend_pch)
  }

  # Add observed data to legend
  if (!is.null(obs_data) && show_obs) {
    legend_items <- c(legend_items, obs_label)
    legend_cols <- c(legend_cols, obs_color)
    legend_lty <- c(legend_lty, NA)
    legend_lwd <- c(legend_lwd, NA)
    legend_pch <- c(legend_pch, obs_pch)
  }

  # Display legend
  legend("topright",
         legend = legend_items,
         col = legend_cols,
         lty = legend_lty,
         lwd = legend_lwd,
         pch = legend_pch,
         horiz = TRUE,
         bty = "n")

  # =========================================================================
  # Add Overall Title
  # =========================================================================

  if (!is.null(title)) {
    mtext(title,
          outer = TRUE,
          cex = 1.3,
          font = 2)
  }

  invisible(NULL)
}


#' Plot component indicator probabilities (base graphics)
#'
#' @description Plots posterior probabilities P(z_t = 1 | data) for mixture
#'   models, showing which component is more likely at each time point.
#'
#' @param z Matrix of MCMC samples for component indicators (n_chain x n_obs).
#' @param threshold Numeric; decision threshold for coloring (default 0.5).
#' @param color_above Character; color when P(z_t = 1) > threshold.
#' @param color_below Character; color when P(z_t = 1) <= threshold.
#' @param ... Additional arguments (currently unused).
#'
#' @return NULL (invisibly). Function is called for its side effects (plotting).
#'
#' @details Creates a bar plot showing the posterior probability that each
#'   observation belongs to component 1. Bars are colored based on whether
#'   the probability exceeds the threshold (default 0.5), making it easy to
#'   identify the most likely component assignment at each time point.
#'
#' @keywords internal
#' @noRd
plot_component_probabilities_base <- function(z,
                                              threshold = 0.5,
                                              color_above = "purple",
                                              color_below = "steelblue",
                                              ...) {

  n_obs <- ncol(z)
  time_grid <- seq_len(n_obs)

  # Compute posterior probabilities P(z_t = 1 | data)
  z_prob <- apply(z, 2, mean)

  # Setup plotting area
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar), add = TRUE)

  par(mfrow = c(1, 1),
      mar = c(4, 4, 2, 1),
      oma = c(0, 0, 2, 0))

  # Create bar plot with threshold-based coloring
  plot(z_prob,
       type = "h",
       lwd = 2,
       col = ifelse(z_prob > threshold, color_above, color_below),
       xlab = "Time",
       ylab = expression(paste("P(", z[t], " = 1 | data)")),
       ylim = c(0, 1.1),
       axes = FALSE)

  axis(side = 1)
  axis(side = 2, at = c(0, 0.5, 1))

  # Add threshold reference line
  segments(x0 = 1,
           y0 = threshold,
           x1 = n_obs,
           y1 = threshold,
           col = "black",
           lwd = 2,
           lty = 2)

  # Add legend
  legend("topright",
         horiz = TRUE,
         legend = c(paste0("P(z_t = 1) > ", threshold),
                    paste0("P(z_t = 1) ≤ ", threshold),
                    "Threshold"),
         col = c(color_above, color_below, "black"),
         lty = c(1, 1, 2),
         lwd = 2,
         bty = "n")

  grid(nx = NA, ny = NULL)

  # Overall title
  mtext(expression(paste("Posterior Probability: P(", z[t], " = 1 | data)")),
        outer = TRUE,
        cex = 1.3,
        font = 2)

  invisible(NULL)
}


# =============================================================================
# Model Family-Specific Wrappers
# =============================================================================

#' Plot binomial success probabilities (base graphics)
#'
#' @description Wrapper function for plotting binomial model success
#'   probabilities (alpha_t). Used by all binomial model types
#'   (locallevel, localtrend, localacceleration).
#'
#' @param x An object inheriting from a binomial model class.
#' @param ci Logical; whether to display credible intervals.
#' @param ci_level Numeric; credible interval level.
#' @param show_obs Logical; whether to display observed proportions.
#'   Default is TRUE.
#' @param true_alpha Numeric vector; true alpha values for simulation studies.
#'   If provided, overlays the true trajectory.
#' @param ... Additional arguments passed to plot_alpha_trajectory_base.
#'
#' @return NULL (invisibly). Function is called for side effects (plotting).
#'
#' @details This function extracts alpha samples and observed data from
#'   binomial model objects and delegates to the generic
#'   \code{plot_alpha_trajectory_base()} function. It automatically computes
#'   observed proportions from y/n_trials.
#'
#'   Used by:
#'   \itemize{
#'     \item \code{plot.binomial_locallevel}
#'     \item \code{plot.binomial_localtrend}
#'     \item \code{plot.binomial_localacceleration}
#'   }
#'
#' @keywords internal
#' @noRd
plot_binomial_alpha_base <- function(x,
                                     ci = TRUE,
                                     ci_level = 0.95,
                                     show_obs = TRUE,
                                     true_alpha = NULL,
                                     ...) {

  # Validate parameters
  if (ci && (!is.numeric(ci_level) || length(ci_level) != 1 ||
             ci_level <= 0 || ci_level >= 1)) {
    stop("`ci_level` must be a single numeric value between 0 and 1")
  }

  if (!is.logical(show_obs) || length(show_obs) != 1) {
    stop("`show_obs` must be a single logical value")
  }

  # Extract observed data
  y <- attr(x, "y")
  n_trials <- attr(x, "n_trials")

  # Compute observed proportions if available and requested
  obs_data <- NULL
  if (show_obs && !is.null(y) && !is.null(n_trials)) {
    obs_data <- y / n_trials
  }

  # Delegate to generic alpha trajectory plotting function
  plot_alpha_trajectory_base(
    alpha = x$alpha,
    ci = ci,
    ci_level = ci_level,
    title = "Binomial Success Probabilities",
    obs_data = obs_data,
    obs_label = expression(y[t] / n["trials"]),
    show_obs = show_obs,
    obs_color = grDevices::rgb(0.75, 0.3, 0.0, 0.5),
    obs_pch = 16,
    obs_cex = 0.8,
    true_alpha = true_alpha,
    ...
  )

  invisible(NULL)
}


#' Plot acceptance proportions (base graphics)
#'
#' @description Plots Metropolis-Hastings acceptance proportions over time points,
#'   showing median acceptance with min-max range and target reference line.
#'
#' @param accept_prop Matrix of acceptance proportions (n_chain x n_obs).
#' @param target_acceptance Numeric; target acceptance proportion for reference line.
#'   Default is 0.44 (theoretically optimal for univariate random-walk proposals).
#' @param ... Additional arguments (currently unused).
#'
#' @return NULL (invisibly). Function is called for its side effects (plotting).
#'
#' @keywords internal
#' @noRd
plot_acceptance_proportions_base <- function(accept_prop,
                                             target_acceptance = 0.44,
                                             ...) {

  # Compute summary statistics
  min_acc <- apply(accept_prop, 2, min)
  max_acc <- apply(accept_prop, 2, max)
  med_acc <- apply(accept_prop, 2, median)

  # Compute y-axis range with buffer
  range_acc <- range(min_acc, max_acc)
  r1_acc <- range_acc[1] - 0.05
  r2_acc <- range_acc[2] + 0.25 * diff(range_acc)

  # Setup plotting area
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  par(mfrow = c(1, 1),
      mar = c(4, 4, 2, 1),
      oma = c(0, 0, 2, 0))

  # Plot median trajectory
  plot(med_acc,
       type = "l",
       col = "black",
       lwd = 2,
       xlab = "Time",
       ylab = "Acceptance Proportion",
       ylim = c(r1_acc, r2_acc),
       main = "")

  # Add min-max range band
  polygon(c(1:length(med_acc), rev(1:length(med_acc))),
          c(min_acc, rev(max_acc)),
          col = grDevices::rgb(0.7, 0.7, 0.7, alpha = 0.3),
          border = NA)

  # Add target acceptance reference line
  abline(h = target_acceptance,
         col = "red",
         lty = 3,
         lwd = 2)

  # Add legend
  legend("topright",
         legend = c("Median",
                    "Range",
                    sprintf("Target (%.2f)", target_acceptance)),
         col = c("black", "gray", "red"),
         lty = c(1, 1, 3),
         lwd = c(2, 8, 2),
         horiz = TRUE,
         bty = "n")

  grid()

  # Overall title
  mtext("Metropolis-Hastings Acceptance Proportions",
        outer = TRUE,
        cex = 1.3,
        font = 2)

  invisible(NULL)
}


#' Plot Bernoulli probabilities (base graphics)
#'
#' @description Wrapper function for plotting Bernoulli model probabilities
#'   (alpha_t). Used by all probit Bernoulli model types.
#'
#' @param x An object inheriting from a probit_bernoulli model class.
#' @param ci Logical; whether to display credible intervals.
#' @param ci_level Numeric; credible interval level.
#' @param show_obs Logical; whether to display observed binary outcomes.
#'   Default is TRUE.
#' @param true_alpha Numeric vector; true alpha values for simulation studies.
#'   If provided, overlays the true trajectory.
#' @param ... Additional arguments (currently unused).
#'
#' @return NULL (invisibly). Function is called for side effects (plotting).
#'
#' @details This function creates a specialized plot for Bernoulli models where
#'   observed outcomes are binary (0/1). Unlike binomial models that show
#'   proportions, this plots y=1 at height 1.0 and y=0 at height 0.0.
#'
#'   Used by:
#'   \itemize{
#'     \item \code{plot.probit_bernoulli_locallevel}
#'     \item \code{plot.probit_bernoulli_localtrend}
#'     \item \code{plot.probit_bernoulli_localacceleration}
#'   }
#'
#' @keywords internal
#' @noRd
plot_bernoulli_alpha_base <- function(x,
                                      ci = TRUE,
                                      ci_level = 0.95,
                                      show_obs = TRUE,
                                      true_alpha = NULL,
                                      ...) {

  # Validate parameters
  if (ci && (!is.numeric(ci_level) || length(ci_level) != 1 ||
             ci_level <= 0 || ci_level >= 1)) {
    stop("`ci_level` must be a single numeric value between 0 and 1")
  }

  if (!is.logical(show_obs) || length(show_obs) != 1) {
    stop("`show_obs` must be a single logical value")
  }

  # Extract observed binary outcomes
  y <- attr(x, "y")

  # Prepare observed data: map binary values directly to 0.0 and 1.0
  obs_data <- NULL
  if (show_obs && !is.null(y)) {
    obs_data <- as.numeric(y)
  }

  # Delegate to generic alpha trajectory plotting function
  plot_alpha_trajectory_base(
    alpha = x$alpha,
    ci = ci,
    ci_level = ci_level,
    title = "Bernoulli Probabilities",
    obs_data = obs_data,
    obs_label = expression(y[t]),
    show_obs = show_obs,
    obs_color = grDevices::rgb(0.75, 0.3, 0.0, 0.5),
    obs_pch = 16,
    obs_cex = 0.8,
    true_alpha = true_alpha,
    ...
  )

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
#'   This function delegates to two specialized functions:
#'   \itemize{
#'     \item \code{plot_alpha_trajectory_base()}: Page 1 (alpha_t trajectory)
#'     \item \code{plot_component_probabilities_base()}: Page 2 (z_t probabilities)
#'   }
#'
#' @keywords internal
#' @noRd
plot_mixture_weights_base <- function(alpha,
                                      z,
                                      ci = TRUE,
                                      ci_level = 0.95,
                                      ...) {

  # Page 1: Alpha trajectory with credible bands
  plot_alpha_trajectory_base(
    alpha = alpha,
    ci = ci,
    ci_level = ci_level,
    title = expression(paste("Time-Varying Mixture Weight: ", alpha[t])),
    obs_data = NULL,
    show_obs = FALSE,
    ...
  )

  # Page 2: Component probabilities
  plot_component_probabilities_base(z, ...)

  invisible(NULL)
}


#' Validate credible interval level
#'
#' @description Internal helper to validate ci_level parameter.
#'
#' @param ci_level Numeric value to validate.
#'
#' @return NULL (invisibly) if valid, stops with error message if invalid.
#'
#' @keywords internal
#' @noRd
validate_ci_level <- function(ci_level) {
  if (!is.numeric(ci_level) ||
      length(ci_level) != 1 ||
      ci_level <= 0 ||
      ci_level >= 1) {
    stop("`ci_level` must be a single numeric value between 0 and 1")
  }
  invisible(NULL)
}

# =============================================================================
# Generic Dashboard Functions
# =============================================================================

#' Generic complete dashboard for mixture models (base graphics)
#'
#' @description Creates a comprehensive multi-page diagnostic dashboard
#'   for any mixture model type, automatically adapting to the model's
#'   polynomial order.
#'
#' @param x An object inheriting from "pdm_mcmc".
#' @param ask Logical; if TRUE, prompts user before each new page.
#' @param ci Logical; whether to display credible intervals.
#' @param ci_level Numeric between 0 and 1; credible interval level.
#' @param true_values Named list containing true values (or NULL).
#'   Expected elements depend on model type and order.
#' @param ... Additional arguments passed to plotting functions.
#'
#' @return NULL (invisibly). Function is called for side effects (plotting).
#'
#' @details Creates a complete dashboard with the following pages:
#'   \enumerate{
#'     \item MCMC diagnostics for each parameter
#'     \item Mixture parameters bivariate relationships (mixture models only)
#'     \item Dynamic state trajectories
#'     \item Dynamic state diagnostics
#'     \item Phase space trajectories (order 2-3 only)
#'     \item Mixture weights and component probabilities (mixture models only)
#'   }
#'
#' @keywords internal
#' @noRd
plot_all_mixture_generic_base <- function(x,
                                          ask = TRUE,
                                          ci = TRUE,
                                          ci_level = 0.95,
                                          true_values = NULL,
                                          ...) {

  # Setup plotting environment
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  if (ask) {
    oldask <- par(ask = TRUE)
    on.exit(par(oldask), add = TRUE)
  }

  # Detect model type and order
  model_info <- detect_model_type(x)
  model_order <- model_info$model_order

  # Get parameter configuration
  param_config <- get_param_config(x,
                                   model_info$model_class,
                                   model_order)
  n_params <- length(param_config)

  # Section 1: MCMC diagnostics for each parameter
  plot_mcmc_diagnostics_generic(x,
                                which = seq_len(n_params),
                                param_config = param_config,
                                true_values = true_values,
                                ...)

  # Section 2: Mixture parameters (mixture models only)
  if (model_info$has_mixture) {
    plot_mixture_params_base(x$mu_1,
                             x$mu_2,
                             x$prec_1,
                             x$prec_2,
                             ...)
  }

  # Section 3: Dynamic states (trajectories, diagnostics, phase space)
  plot_dynamic_states_generic_base(x,
                                   model_order = model_order,
                                   ci = ci,
                                   ci_level = ci_level,
                                   true_values = true_values,
                                   ...)

  # Section 4: Mixture weights (mixture models only)
  if (model_info$has_mixture) {
    plot_mixture_weights_base(x$alpha,
                              x$z,
                              ci = ci,
                              ci_level = ci_level,
                              ...)
  }

  invisible(NULL)
}


#' Generic dynamic states plotting dispatcher (base graphics)
#'
#' @description Orchestrates dynamic state plotting for any model order,
#'   creating appropriate pages for trajectories, innovations, and diagnostics.
#'
#' @param x An object inheriting from "pdm_mcmc".
#' @param which Integer vector specifying which state plot pages to display.
#'   If NULL, all pages are shown.
#' @param model_order Integer: 1, 2, or 3. If NULL, auto-detected.
#' @param ci Logical; whether to display credible intervals.
#' @param ci_level Numeric between 0 and 1; credible interval level.
#' @param true_values Named list with true values (or NULL).
#'   Expected elements: theta_1, theta_2 (if order >= 2), theta_3 (if order == 3).
#'   Each element should be a numeric vector of length n_obs.
#' @param ... Additional arguments (currently unused).
#'
#' @return NULL (invisibly). Function is called for side effects (plotting).
#'
#' @details Number of pages generated:
#'   \itemize{
#'     \item Order 1: 2 pages (trajectory + 1x3 diagnostics)
#'     \item Order 2: 3 pages (trajectory + 2x3 diagnostics + 1x1 phase space)
#'     \item Order 3: 3 pages (trajectory + 3x3 diagnostics + 1x2 phase space)
#'   }
#'
#' @keywords internal
#' @noRd
plot_dynamic_states_generic_base <- function(x,
                                             which = NULL,
                                             model_order = NULL,
                                             ci = TRUE,
                                             ci_level = 0.95,
                                             true_values = NULL,
                                             ...) {

  # Auto-detect model order if needed
  if (is.null(model_order)) {
    model_info <- detect_model_type(x)
    model_order <- model_info$model_order
  }

  validate_ci_level(ci_level)

  # Determine number of pages based on model order
  # Order 1: 2 pages (trajectory + diagnostics)
  # Orders 2-3: 3 pages (trajectory + diagnostics + phase space)
  n_pages <- if (model_order == 1) 2L else 3L

  if (is.null(which)) {
    which <- seq_len(n_pages)
  }

  # Validate which parameter
  if (any(which < 1) || any(which > n_pages)) {
    stop("`which` must be between 1 and ", n_pages)
  }

  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  n_obs <- attr(x, "n_obs")
  time_grid <- seq_len(n_obs)

  # Prepare credible interval parameters
  if (ci) {
    ci_lower_prob <- (1 - ci_level) / 2
    ci_upper_prob <- 1 - ci_lower_prob
    ci_label <- paste0(round(ci_level * 100), "% CI")
  }

  # Get state summaries and labels
  state_summaries <- summarise_all_states(x, model_order, ci, ci_level)
  state_labels <- get_state_labels(model_order)

  # Pre-compute innovations if diagnostic page is requested (Page 2)
  innovations <- NULL
  innov_summaries <- NULL
  innov_labels <- NULL

  if (2 %in% which) {
    innovations <- compute_innovations(x, model_order)
    innov_summaries <- lapply(innovations, summarise_state, ci, ci_level)
    innov_labels <- get_innovation_labels(model_order)
  }

  # =========================================================================
  # Page 1: State Trajectories (All Orders)
  # =========================================================================

  if (1 %in% which) {
    par(mfrow = c(model_order, 1),
        mar = c(4, 4, 3, 1),
        oma = c(0, 0, 2, 0),
        mgp = c(2.5, 1, 0))

    for (i in seq_len(model_order)) {
      state_name <- state_labels$state_names[i]
      summary_i <- state_summaries[[state_name]]

      # Extract true state if provided
      true_state_i <- NULL
      if (!is.null(true_values)) {
        true_state_i <- true_values[[state_name]]
      }

      plot_state_trajectory_base(
        time_grid = time_grid,
        median = summary_i$median,
        lower = summary_i$lower,
        upper = summary_i$upper,
        ylab = state_labels$state_labels[[i]],
        main = state_labels$state_titles[i],
        col = state_labels$state_colors[i],
        ci = ci,
        ci_label = ci_label,
        true_state = true_state_i
      )
    }

    mtext("Dynamic State Trajectories",
          outer = TRUE,
          cex = 1.3,
          font = 2)
  }

  # =========================================================================
  # Page 2: Consolidated Diagnostics (All Orders)
  # =========================================================================

  if (2 %in% which) {

    # -----------------------------------------------------------------------
    # ORDER 1: Consolidated 1x3 Diagnostics
    # -----------------------------------------------------------------------
    if (model_order == 1) {
      par(mfrow = c(1, 3),
          mar = c(4, 4, 2.5, 1),
          oma = c(0, 0, 3, 0),
          mgp = c(2.5, 1, 0))

      # --- Prepare Data ---
      theta_1_median <- state_summaries$theta_1$median
      innov_1_median <- innov_summaries$innov_1$median
      col_level <- innov_labels$innov_colors[1]

      theta_lag <- theta_1_median[-n_obs]
      theta_current <- theta_1_median[-1]
      innov_1_lagged <- innov_1_median[-1]

      # Panel 1: Level Transition
      plot(theta_lag,
           theta_current,
           xlab = expression(theta["t-1,1"]),
           ylab = expression(theta["t,1"]),
           main = "Level Transition",
           pch = 16,
           cex = 0.8,
           col = col_level)
      abline(a = 0, b = 1, col = "black", lty = 2, lwd = 1.5)
      grid()

      # Panel 2: Level Diagnostic
      plot(theta_lag,
           innov_1_lagged,
           xlab = expression(theta["t-1,1"]),
           ylab = expression(u["t,1"]),
           main = "Level Diagnostic",
           pch = 16,
           cex = 0.8,
           col = col_level)
      abline(h = 0, col = "black", lty = 2, lwd = 1.5)
      grid()

      # Panel 3: Level Innovation
      plot_innovation_base(
        time_grid = time_grid,
        median = innov_summaries$innov_1$median,
        lower = innov_summaries$innov_1$lower,
        upper = innov_summaries$innov_1$upper,
        ylab = innov_labels$innov_labels[[1]],
        main = innov_labels$innov_titles[1],
        col_bar = col_level,
        ci = ci,
        ci_label = ci_label
      )

      # Overall title
      mtext("Dynamic State Diagnostics",
            outer = TRUE,
            cex = 1.3,
            font = 2)
    }

    # -----------------------------------------------------------------------
    # ORDER 2: Consolidated 2x3 Diagnostics
    # -----------------------------------------------------------------------
    if (model_order == 2) {
      par(mfrow = c(2, 3),
          mar = c(4, 4, 2.5, 1),
          oma = c(0, 0, 3, 0),
          mgp = c(2.5, 1, 0))

      # --- Prepare Data ---
      theta_1_median <- state_summaries$theta_1$median
      theta_2_median <- state_summaries$theta_2$median

      innov_1_median <- innov_summaries$innov_1$median
      innov_2_median <- innov_summaries$innov_2$median

      col_level <- innov_labels$innov_colors[1]
      col_trend <- innov_labels$innov_colors[2]

      delta_theta_1 <- theta_1_median[-1] - theta_1_median[-n_obs]
      theta_2_lag <- theta_2_median[-n_obs]
      theta_2_current <- theta_2_median[-1]

      innov_1_lagged <- innov_1_median[-1]
      innov_2_lagged <- innov_2_median[-1]

      # --- ROW 1: LEVEL ---

      # Panel (1,1): Level Transition
      plot(theta_2_lag,
           delta_theta_1,
           xlab = expression(theta["t-1,2"]),
           ylab = expression(Delta*theta["t,1"]),
           main = "Level Transition",
           pch = 16,
           cex = 0.8,
           col = col_level)
      abline(a = 0, b = 1, col = "black", lty = 2, lwd = 1.5)
      grid()

      # Panel (1,2): Level Diagnostic
      plot(theta_2_lag,
           innov_1_lagged,
           xlab = expression(theta["t-1,2"]),
           ylab = expression(u["t,1"]),
           main = "Level Diagnostic",
           pch = 16,
           cex = 0.8,
           col = col_level)
      abline(h = 0, col = "black", lty = 2, lwd = 1.5)
      grid()

      # Panel (1,3): Level Innovation
      plot_innovation_base(
        time_grid = time_grid,
        median = innov_summaries$innov_1$median,
        lower = innov_summaries$innov_1$lower,
        upper = innov_summaries$innov_1$upper,
        ylab = innov_labels$innov_labels[[1]],
        main = innov_labels$innov_titles[1],
        col_bar = col_level,
        ci = ci,
        ci_label = ci_label
      )

      # --- ROW 2: TREND ---

      # Panel (2,1): Trend Transition
      plot(theta_2_lag,
           theta_2_current,
           xlab = expression(theta["t-1,2"]),
           ylab = expression(theta["t,2"]),
           main = "Trend Transition",
           pch = 16,
           cex = 0.8,
           col = col_trend)
      abline(a = 0, b = 1, col = "black", lty = 2, lwd = 1.5)
      grid()

      # Panel (2,2): Trend Diagnostic
      plot(theta_2_lag,
           innov_2_lagged,
           xlab = expression(theta["t-1,2"]),
           ylab = expression(u["t,2"]),
           main = "Trend Diagnostic",
           pch = 16,
           cex = 0.8,
           col = col_trend)
      abline(h = 0, col = "black", lty = 2, lwd = 1.5)
      grid()

      # Panel (2,3): Trend Innovation
      plot_innovation_base(
        time_grid = time_grid,
        median = innov_summaries$innov_2$median,
        lower = innov_summaries$innov_2$lower,
        upper = innov_summaries$innov_2$upper,
        ylab = innov_labels$innov_labels[[2]],
        main = innov_labels$innov_titles[2],
        col_bar = col_trend,
        ci = ci,
        ci_label = ci_label
      )

      # Overall title
      mtext("Dynamic State Diagnostics",
            outer = TRUE,
            cex = 1.3,
            font = 2)
    }

    # -----------------------------------------------------------------------
    # ORDER 3: Consolidated 3x3 Diagnostics
    # -----------------------------------------------------------------------
    if (model_order == 3) {
      par(mfrow = c(3, 3),
          mar = c(4, 4, 2.5, 1),
          oma = c(0, 0, 3, 0),
          mgp = c(2.5, 1, 0))

      # --- Prepare Data ---
      theta_1_median <- state_summaries$theta_1$median
      theta_2_median <- state_summaries$theta_2$median
      theta_3_median <- state_summaries$theta_3$median

      innov_1_median <- innov_summaries$innov_1$median
      innov_2_median <- innov_summaries$innov_2$median
      innov_3_median <- innov_summaries$innov_3$median

      col_level <- innov_labels$innov_colors[1]
      col_trend <- innov_labels$innov_colors[2]
      col_accel <- innov_labels$innov_colors[3]

      delta_theta_1 <- theta_1_median[-1] - theta_1_median[-n_obs]
      delta_theta_2 <- theta_2_median[-1] - theta_2_median[-n_obs]

      theta_2_lag <- theta_2_median[-n_obs]
      theta_3_lag <- theta_3_median[-n_obs]
      theta_3_current <- theta_3_median[-1]

      innov_1_lagged <- innov_1_median[-1]
      innov_2_lagged <- innov_2_median[-1]
      innov_3_lagged <- innov_3_median[-1]

      # --- ROW 1: LEVEL ---

      # Panel (1,1): Level Transition
      plot(theta_2_lag,
           delta_theta_1,
           xlab = expression(theta["t-1,2"]),
           ylab = expression(Delta*theta["t,1"]),
           main = "Level Transition",
           pch = 16,
           cex = 0.8,
           col = col_level)
      abline(a = 0, b = 1, col = "black", lty = 2, lwd = 1.5)
      grid()

      # Panel (1,2): Level Diagnostic
      plot(theta_2_lag,
           innov_1_lagged,
           xlab = expression(theta["t-1,2"]),
           ylab = expression(u["t,1"]),
           main = "Level Diagnostic",
           pch = 16,
           cex = 0.8,
           col = col_level)
      abline(h = 0, col = "black", lty = 2, lwd = 1.5)
      grid()

      # Panel (1,3): Level Innovation
      plot_innovation_base(
        time_grid = time_grid,
        median = innov_summaries$innov_1$median,
        lower = innov_summaries$innov_1$lower,
        upper = innov_summaries$innov_1$upper,
        ylab = innov_labels$innov_labels[[1]],
        main = innov_labels$innov_titles[1],
        col_bar = col_level,
        ci = ci,
        ci_label = ci_label
      )

      # --- ROW 2: TREND ---

      # Panel (2,1): Trend Transition
      plot(theta_3_lag,
           delta_theta_2,
           xlab = expression(theta["t-1,3"]),
           ylab = expression(Delta*theta["t,2"]),
           main = "Trend Transition",
           pch = 16,
           cex = 0.8,
           col = col_trend)
      abline(a = 0, b = 1, col = "black", lty = 2, lwd = 1.5)
      grid()

      # Panel (2,2): Trend Diagnostic
      plot(theta_3_lag,
           innov_2_lagged,
           xlab = expression(theta["t-1,3"]),
           ylab = expression(u["t,2"]),
           main = "Trend Diagnostic",
           pch = 16,
           cex = 0.8,
           col = col_trend)
      abline(h = 0, col = "black", lty = 2, lwd = 1.5)
      grid()

      # Panel (2,3): Trend Innovation
      plot_innovation_base(
        time_grid = time_grid,
        median = innov_summaries$innov_2$median,
        lower = innov_summaries$innov_2$lower,
        upper = innov_summaries$innov_2$upper,
        ylab = innov_labels$innov_labels[[2]],
        main = innov_labels$innov_titles[2],
        col_bar = col_trend,
        ci = ci,
        ci_label = ci_label
      )

      # --- ROW 3: ACCELERATION ---

      # Panel (3,1): Acceleration Transition
      plot(theta_3_lag,
           theta_3_current,
           xlab = expression(theta["t-1,3"]),
           ylab = expression(theta["t,3"]),
           main = "Acceleration Transition",
           pch = 16,
           cex = 0.8,
           col = col_accel)
      abline(a = 0, b = 1, col = "black", lty = 2, lwd = 1.5)
      grid()

      # Panel (3,2): Acceleration Diagnostic
      plot(theta_3_lag,
           innov_3_lagged,
           xlab = expression(theta["t-1,3"]),
           ylab = expression(u["t,3"]),
           main = "Acceleration Diagnostic",
           pch = 16,
           cex = 0.8,
           col = col_accel)
      abline(h = 0, col = "black", lty = 2, lwd = 1.5)
      grid()

      # Panel (3,3): Acceleration Innovation
      plot_innovation_base(
        time_grid = time_grid,
        median = innov_summaries$innov_3$median,
        lower = innov_summaries$innov_3$lower,
        upper = innov_summaries$innov_3$upper,
        ylab = innov_labels$innov_labels[[3]],
        main = innov_labels$innov_titles[3],
        col_bar = col_accel,
        ci = ci,
        ci_label = ci_label
      )

      # Overall title
      mtext("Dynamic State Diagnostics",
            outer = TRUE,
            cex = 1.3,
            font = 2)
    }
  }

  # =========================================================================
  # Page 3: Phase Space Trajectories (Order 2 and 3 Only)
  # =========================================================================

  if (3 %in% which && model_order > 1) {

    theta_1_median <- state_summaries$theta_1$median
    theta_2_median <- state_summaries$theta_2$median

    # -----------------------------------------------------------------------
    # ORDER 2: Single Phase Space Plot (1x1)
    # -----------------------------------------------------------------------
    if (model_order == 2) {
      par(mfrow = c(1, 1),
          mar = c(4, 4, 3, 1),
          oma = c(0, 0, 3, 0),
          mgp = c(2.5, 1, 0))

      # Compute y-axis range with buffer
      range_theta_2 <- range(theta_2_median)
      if (diff(range_theta_2) == 0) {
        range_theta_2 <- range_theta_2 + c(-0.5, 0.5)
      }
      range_theta_2[2] <- range_theta_2[2] + 0.25 * diff(range_theta_2)

      # Plot: Level-Trend Phase Space
      plot(theta_1_median,
           theta_2_median,
           type = "l",
           col = "gray40",
           lwd = 1.5,
           xlab = expression(theta["t,1"]),
           ylab = expression(theta["t,2"]),
           ylim = range_theta_2,
           main = "Level-Trend Trajectory")
      grid()

      # Mark start and end points
      points(theta_1_median[1],
             theta_2_median[1],
             pch = 21,
             bg = "green",
             col = "black",
             cex = 1.5)
      points(theta_1_median[n_obs],
             theta_2_median[n_obs],
             pch = 21,
             bg = "red",
             col = "black",
             cex = 1.5)

      # Add legend
      legend("topright",
             legend = c("Start", "End"),
             horiz = TRUE,
             pch = 16,
             col = c("green", "red"),
             bty = "n",
             cex = 0.9)

      # Overall title
      mtext("State-Space Phase Trajectory",
            outer = TRUE,
            cex = 1.3,
            font = 2)
    }

    # -----------------------------------------------------------------------
    # ORDER 3: Two Phase Space Plots (1x2)
    # -----------------------------------------------------------------------
    if (model_order == 3) {
      par(mfrow = c(1, 2),
          mar = c(4, 4, 3, 1),
          oma = c(0, 0, 3, 0),
          mgp = c(2.5, 1, 0))

      theta_3_median <- state_summaries$theta_3$median

      # Compute y-axis ranges with buffer
      range_theta_2 <- range(theta_2_median)
      range_theta_3 <- range(theta_3_median)

      if (diff(range_theta_2) == 0) {
        range_theta_2 <- range_theta_2 + c(-0.5, 0.5)
      }
      if (diff(range_theta_3) == 0) {
        range_theta_3 <- range_theta_3 + c(-0.5, 0.5)
      }

      range_theta_2[2] <- range_theta_2[2] + 0.25 * diff(range_theta_2)
      range_theta_3[2] <- range_theta_3[2] + 0.25 * diff(range_theta_3)

      # Panel 1: Level-Trend Phase Space
      plot(theta_1_median,
           theta_2_median,
           type = "l",
           col = "gray40",
           lwd = 1.5,
           xlab = expression(theta["t,1"]),
           ylab = expression(theta["t,2"]),
           ylim = range_theta_2,
           main = "Level-Trend Trajectory")
      grid()

      # Mark start and end points
      points(theta_1_median[1],
             theta_2_median[1],
             pch = 21,
             bg = "green",
             col = "black",
             cex = 1.5)
      points(theta_1_median[n_obs],
             theta_2_median[n_obs],
             pch = 21,
             bg = "red",
             col = "black",
             cex = 1.5)

      # Add legend
      legend("topright",
             legend = c("Start", "End"),
             horiz = TRUE,
             pch = 16,
             col = c("green", "red"),
             bty = "n",
             cex = 0.9)

      # Panel 2: Trend-Acceleration Phase Space
      plot(theta_2_median,
           theta_3_median,
           type = "l",
           col = "gray40",
           lwd = 1.5,
           xlab = expression(theta["t,2"]),
           ylab = expression(theta["t,3"]),
           ylim = range_theta_3,
           main = "Trend-Acceleration Trajectory")
      grid()

      # Mark start and end points
      points(theta_2_median[1],
             theta_3_median[1],
             pch = 21,
             bg = "green",
             col = "black",
             cex = 1.5)
      points(theta_2_median[n_obs],
             theta_3_median[n_obs],
             pch = 21,
             bg = "red",
             col = "black",
             cex = 1.5)

      # Add legend
      legend("topright",
             legend = c("Start", "End"),
             horiz = TRUE,
             pch = 16,
             col = c("green", "red"),
             bty = "n",
             cex = 0.9)

      # Overall title
      mtext("State-Space Phase Trajectories",
            outer = TRUE,
            cex = 1.3,
            font = 2)
    }
  }

  invisible(NULL)
}
