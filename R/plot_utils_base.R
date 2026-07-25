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
#'   (used in plot title), e.g., `quote(mu[1])`.
#' @param param_label Expression for axis labels, e.g., `expression(mu[1])`.
#' @param true_value Numeric or NULL. If provided, adds a reference line at
#'   the true parameter value (useful for simulation studies).
#' @param color Character; color for the main diagnostic lines. Default is
#'   "steelblue". This should match the parameter type:
#'   \itemize{
#'     \item "steelblue" for level-related parameters (theta_1, theta_01, W_1^-1)
#'     \item "firebrick" for trend-related parameters (theta_2, theta_02, W_2^-1)
#'     \item "darkgreen" for acceleration-related parameters (theta_3, theta_03, W_3^-1)
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
#'   between mixture component parameters (means and precisions). Optionally
#'   overlays true parameter values for simulation validation studies.
#'
#' @param mu_1 Numeric vector of MCMC samples for component 1 mean.
#' @param mu_2 Numeric vector of MCMC samples for component 2 mean.
#' @param prec_1 Numeric vector of MCMC samples for component 1 precision.
#' @param prec_2 Numeric vector of MCMC samples for component 2 precision.
#' @param which Integer vector specifying which subplots to display (1:4).
#'   If NULL, all four plots are shown.
#' @param true_values Named list containing true parameter values for comparison.
#'   If NULL (default), no true values are displayed. Expected elements:
#'   \describe{
#'     \item{mu_1}{Numeric scalar; true mean of component 1}
#'     \item{mu_2}{Numeric scalar; true mean of component 2}
#'     \item{prec_1}{Numeric scalar; true precision of component 1 (phi_1)}
#'     \item{prec_2}{Numeric scalar; true precision of component 2 (phi_2)}
#'   }
#'   Any subset of these elements can be provided. Only panels with complete
#'   true values for both parameters will display the true value marker.
#' @param ... Additional arguments (currently unused, for future extensibility).
#'
#' @return NULL (invisibly). Function is called for its side effects (plotting).
#'
#' @details The four panels show:
#'   \enumerate{
#'     \item mu_1 vs mu_2 (component separation in mean space)
#'     \item mu_1 vs phi_1 (mean-precision relationship for component 1)
#'     \item mu_2 vs phi_2 (mean-precision relationship for component 2)
#'     \item phi_1 vs phi_2 (precision comparison between components)
#'   }
#'
#'   \strong{True Value Display:}
#'   When `true_values` is provided, true parameter values are marked
#'   with red "X" symbols (pch = 4) on each panel. This allows visual assessment
#'   of whether the posterior distribution correctly covers the true values,
#'   which is essential for validating model performance in simulation studies.
#'
#'   \strong{Color Scheme:}
#'   Note: Mixture parameters do not correspond to dynamic states (level, trend,
#'   acceleration), so they use neutral gray colors to avoid confusion with the
#'   state-specific color scheme (blue for level, red for trend, green for
#'   acceleration).
#'
#'   \strong{Interpretation:}
#'   \itemize{
#'     \item \strong{Panel 1 (mu_1 vs mu_2):} Shows component separation. Points
#'       far from the diagonal indicate well-separated components. True value
#'       should fall within the posterior cloud.
#'     \item \strong{Panels 2-3 (mu vs phi):} Show mean-precision relationships
#'       for each component. Can reveal posterior dependencies between location
#'       and scale parameters.
#'     \item \strong{Panel 4 (phi_1 vs phi_2):} Shows precision relationship.
#'       Points near diagonal suggest similar variances between components.
#'   }
#'
#' @examples
#' \donttest{
#' # Example 1: Basic usage without true values
#' plot_mixture_params_base(
#'   mu_1 = mcmc_output$mu_1,
#'   mu_2 = mcmc_output$mu_2,
#'   prec_1 = mcmc_output$prec_1,
#'   prec_2 = mcmc_output$prec_2
#' )
#'
#' # Example 2: Display only specific panels
#' plot_mixture_params_base(
#'   mu_1 = mcmc_output$mu_1,
#'   mu_2 = mcmc_output$mu_2,
#'   prec_1 = mcmc_output$prec_1,
#'   prec_2 = mcmc_output$prec_2,
#'   which = c(1, 4)  # Only mu_1 vs mu_2 and phi_1 vs phi_2
#' )
#'
#' # Example 3: Simulation validation with true values
#' true_params <- list(
#'   mu_1 = 0,
#'   mu_2 = 2,
#'   prec_1 = 4,
#'   prec_2 = 4
#' )
#'
#' plot_mixture_params_base(
#'   mu_1 = mcmc_output$mu_1,
#'   mu_2 = mcmc_output$mu_2,
#'   prec_1 = mcmc_output$prec_1,
#'   prec_2 = mcmc_output$prec_2,
#'   true_values = true_params
#' )
#'
#' # Example 4: Partial true values (only means)
#' plot_mixture_params_base(
#'   mu_1 = mcmc_output$mu_1,
#'   mu_2 = mcmc_output$mu_2,
#'   prec_1 = mcmc_output$prec_1,
#'   prec_2 = mcmc_output$prec_2,
#'   true_values = list(mu_1 = 0, mu_2 = 2)  # Only Panel 1 will show true values
#' )
#' }
#'
#' @seealso \code{\link{plot.normal_mixture_localacceleration}} for the main
#'   plot method that calls this function.
#'
#' @keywords internal
#' @noRd
plot_mixture_params_base <- function(mu_1,
                                     mu_2,
                                     prec_1,
                                     prec_2,
                                     which = NULL,
                                     true_values = NULL,
                                     ...) {

  # ===========================================================================
  # INPUT VALIDATION
  # ===========================================================================

  # Validate MCMC samples
  if (!is.numeric(mu_1) || !is.numeric(mu_2) ||
      !is.numeric(prec_1) || !is.numeric(prec_2)) {
    stop("All MCMC draw arguments must be numeric vectors")
  }

  n_samples <- length(mu_1)
  if (length(mu_2) != n_samples || length(prec_1) != n_samples ||
      length(prec_2) != n_samples) {
    stop("All MCMC draw vectors must have the same length")
  }

  # Validate which parameter
  if (is.null(which)) {
    which <- 1:4
  }

  if (!is.numeric(which) || any(which != floor(which))) {
    stop("`which` must be an integer vector")
  }

  if (any(which < 1) || any(which > 4)) {
    stop("`which` must contain values between 1 and 4")
  }

  # ===========================================================================
  # EXTRACT TRUE VALUES (if provided)
  # ===========================================================================

  true_mu_1 <- NULL
  true_mu_2 <- NULL
  true_prec_1 <- NULL
  true_prec_2 <- NULL

  if (!is.null(true_values)) {
    if (!is.list(true_values)) {
      warning("`true_values` must be a named list. Ignoring true values.")
      true_values <- NULL
    } else {
      # Extract true values if they exist
      if ("mu_1" %in% names(true_values)) {
        true_mu_1 <- true_values$mu_1
        if (!is.numeric(true_mu_1) || length(true_mu_1) != 1) {
          warning("`true_values$mu_1` must be a single numeric value. Ignoring.")
          true_mu_1 <- NULL
        }
      }

      if ("mu_2" %in% names(true_values)) {
        true_mu_2 <- true_values$mu_2
        if (!is.numeric(true_mu_2) || length(true_mu_2) != 1) {
          warning("`true_values$mu_2` must be a single numeric value. Ignoring.")
          true_mu_2 <- NULL
        }
      }

      if ("prec_1" %in% names(true_values)) {
        true_prec_1 <- true_values$prec_1
        if (!is.numeric(true_prec_1) || length(true_prec_1) != 1) {
          warning("`true_values$prec_1` must be a single numeric value. Ignoring.")
          true_prec_1 <- NULL
        }
      }

      if ("prec_2" %in% names(true_values)) {
        true_prec_2 <- true_values$prec_2
        if (!is.numeric(true_prec_2) || length(true_prec_2) != 1) {
          warning("`true_values$prec_2` must be a single numeric value. Ignoring.")
          true_prec_2 <- NULL
        }
      }
    }
  }

  # Define color scheme
  col_comp1 <- grDevices::rgb(1.0, 0.55, 0.0, 0.4)   # darkorange
  col_comp2 <- grDevices::rgb(0.58, 0.0, 0.83, 0.4)  # darkviolet
  col_between <- grDevices::rgb(0.79, 0.28, 0.41, 0.4)  # mediumpurple

  # ===========================================================================
  # SETUP PLOTTING ENVIRONMENT
  # ===========================================================================

  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  par(mfrow = c(2, 2),
      mar = c(4, 4, 2, 1),
      oma = c(0, 0, 2, 0),
      mgp = c(2.5, 1, 0))

  # ===========================================================================
  # PANEL 1: mu_1 vs mu_2 (Component Separation)
  # ===========================================================================

  if (1 %in% which) {
    # Create scatterplot of posterior samples
    plot(mu_1,
         mu_2,
         xlab = expression(mu[1]),
         ylab = expression(mu[2]),
         main = expression(paste(mu[1], " vs ", mu[2])),
         pch = 16,
         col = col_between)
    grid()

    # Add true value marker if both true means are available
    if (!is.null(true_mu_1) && !is.null(true_mu_2)) {
      points(true_mu_1,
             true_mu_2,
             pch = 4,
             cex = 2,
             lwd = 3,
             col = "black")

      # Add legend for true value
      legend("topright",
             legend = "True Value",
             pch = 4,
             col = "black",
             pt.lwd = 3,
             bty = "n",
             cex = 0.9)
    }
  }

  # ===========================================================================
  # PANEL 2: mu_1 vs phi_1 (Component 1 Mean-Precision)
  # ===========================================================================

  if (2 %in% which) {
    # Create scatterplot showing relationship between mean and precision
    plot(mu_1,
         prec_1,
         xlab = expression(mu[1]),
         ylab = expression(phi[1]),
         main = expression(paste(mu[1], " vs ", phi[1])),
         pch = 16,
         col = col_comp1)
    grid()

    # Add true value marker if both parameters are available
    if (!is.null(true_mu_1) && !is.null(true_prec_1)) {
      points(true_mu_1,
             true_prec_1,
             pch = 4,
             cex = 2,
             lwd = 3,
             col = "black")

      # Add legend for true value
      legend("topright",
             legend = "True Value",
             pch = 4,
             col = "black",
             pt.lwd = 3,
             bty = "n",
             cex = 0.9)
    }
  }

  # ===========================================================================
  # PANEL 3: mu_2 vs phi_2 (Component 2 Mean-Precision)
  # ===========================================================================

  if (3 %in% which) {
    # Create scatterplot showing relationship between mean and precision
    plot(mu_2,
         prec_2,
         xlab = expression(mu[2]),
         ylab = expression(phi[2]),
         main = expression(paste(mu[2], " vs ", phi[2])),
         pch = 16,
         col = col_comp2)
    grid()

    # Add true value marker if both parameters are available
    if (!is.null(true_mu_2) && !is.null(true_prec_2)) {
      points(true_mu_2,
             true_prec_2,
             pch = 4,
             cex = 2,
             lwd = 3,
             col = "black")

      # Add legend for true value
      legend("topright",
             legend = "True Value",
             pch = 4,
             col = "black",
             pt.lwd = 3,
             bty = "n",
             cex = 0.9)
    }
  }

  # ===========================================================================
  # PANEL 4: phi_1 vs phi_2 (Precision Comparison)
  # ===========================================================================

  if (4 %in% which) {
    # Create scatterplot comparing precisions between components
    plot(prec_1,
         prec_2,
         xlab = expression(phi[1]),
         ylab = expression(phi[2]),
         main = expression(paste(phi[1], " vs ", phi[2])),
         pch = 16,
         col = col_between)
    grid()

    # Add true value marker if both precisions are available
    if (!is.null(true_prec_1) && !is.null(true_prec_2)) {
      points(true_prec_1,
             true_prec_2,
             pch = 4,
             cex = 2,
             lwd = 3,
             col = "black")

      # Add legend for true value
      legend("topright",
             legend = "True Value",
             pch = 4,
             col = "black",
             pt.lwd = 3,
             bty = "n",
             cex = 0.9)
    }
  }

  # ===========================================================================
  # OVERALL TITLE
  # ===========================================================================

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
#' @param alpha Matrix of MCMC samples for alpha (n_draws x n_obs).
#' @param ci Logical; whether to display credible intervals.
#' @param ci_level Numeric between 0 and 1; credible interval level.
#' @param title Character or expression; main title for the plot.
#' @param obs_data Numeric vector of observed data (proportions, binary, or counts).
#'   If NULL, no observations are plotted.
#' @param obs_label Character; legend label for observed data.
#' @param show_obs Logical; whether to display observed data points.
#'   Default is TRUE. Ignored if obs_data is NULL.
#' @param obs_color Character; color for observed data points.
#' @param obs_pch Integer; point character for observed data.
#' @param obs_cex Numeric; point size for observed data.
#' @param true_alpha Numeric vector; true alpha values for simulation studies.
#'   If provided, overlays the true trajectory.
#' @param ylim_auto Logical; whether to automatically compute y-axis limits
#'   based on data. Default is FALSE. If FALSE, uses ylim = c(0, 1.1) for
#'   probability-scale plots (binomial/Bernoulli/mixture). If TRUE, computes
#'   ylim from data range (appropriate for Poisson rates which can exceed 1).
#' @param ... Additional arguments (currently unused).
#'
#' @return NULL (invisibly). Function is called for its side effects (plotting).
#'
#' @details This is a generic plotting function used by mixture models
#'   (for mixture weights), binomial/Bernoulli models (for success
#'   probabilities), and Poisson models (for event rates). The function
#'   creates a single-page plot with:
#'   \itemize{
#'     \item Median trajectory of alpha_t (blue line)
#'     \item Optional credible interval band (gray)
#'     \item Optional observed data overlay (points)
#'     \item Optional true values (for simulation validation)
#'   }
#'
#'   \strong{Y-axis scaling:}
#'   \itemize{
#'     \item For probability models (binomial/Bernoulli/mixture weights):
#'       Set `ylim_auto = FALSE` to use fixed \[0, 1.1\] range
#'     \item For rate models (Poisson): Set `ylim_auto = TRUE` to
#'       compute range dynamically from data
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
                                       ylim_auto = FALSE,
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

  if (! is.logical(show_obs) || length(show_obs) != 1) {
    stop("`show_obs` must be a single logical value")
  }

  if (!is.logical(ylim_auto) || length(ylim_auto) != 1) {
    stop("`ylim_auto` must be a single logical value")
  }

  # =========================================================================
  # Prepare Data
  # =========================================================================

  n_obs <- ncol(alpha)
  time_grid <- seq_len(n_obs)

  # Compute summary statistics
  alpha_median <- apply(alpha, 2, stats::median)

  if (ci) {
    ci_mat <- hpdi(alpha, ci_level)
    alpha_lower <- ci_mat[, "lower"]
    alpha_upper <- ci_mat[, "upper"]
    ci_label <- paste0(round(ci_level * 100), "% CI")
  }

  # =========================================================================
  # Compute Y-axis Limits
  # =========================================================================

  if (ylim_auto) {
    # Automatic range computation for rate models (e.g., Poisson)
    y_min <- 0  # Always start at zero for rates/probabilities

    # Collect all relevant values for range calculation
    range_vals <- alpha_median
    if (ci) {
      range_vals <- c(range_vals, alpha_lower, alpha_upper)
    }
    if (! is.null(obs_data) && show_obs) {
      range_vals <- c(range_vals, obs_data)
    }
    if (!is.null(true_alpha)) {
      range_vals <- c(range_vals, true_alpha)
    }

    y_max <- max(range_vals, na.rm = TRUE)

    # Add 10% buffer to top
    y_max <- y_max * 1.1

    ylim_val <- c(y_min, y_max)
  } else {
    # Fixed range for probability models (binomial/Bernoulli/mixture weights)
    ylim_val <- c(0, 1.1)
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
       ylim = ylim_val,
       axes = FALSE,
       main = "")

  # Determine y-axis tick locations
  if (ylim_auto) {
    # Automatic ticks for rate models
    axis(side = 2)
  } else {
    # Fixed ticks for probability models
    axis(side = 2, at = seq(0, 1, by = 0.2))
  }

  axis(side = 1)

  # =========================================================================
  # Add Observed Data (if provided and requested)
  # =========================================================================

  if (! is.null(obs_data) && show_obs) {
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

  # Vertical grid lines
  y_top <- ylim_val[2]
  y_bottom <- ylim_val[1] - 0.04 * diff(ylim_val)

  segments(x0 = axTicks(1),
           y0 = y_bottom,
           x1 = axTicks(1),
           y1 = y_top * 1.03,
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
  if (! is.null(obs_data) && show_obs) {
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
#'   Uses mixture component colors for visual consistency.
#'
#' @param z Matrix of MCMC samples for component indicators (n_draws x n_obs).
#' @param threshold Numeric; decision threshold for coloring (default 0.5).
#' @param color_above Character; color when P(z_t = 1) > threshold.
#'   Default is "darkviolet" (Component 2).
#' @param color_below Character; color when P(z_t = 1) <= threshold.
#'   Default is "darkorange" (Component 1).
#' @param true_z Numeric vector; true component indicators for simulation studies.
#'   If provided, overlays the true values as red markers.
#' @param ... Additional arguments (currently unused).
#'
#' @return NULL (invisibly). Function is called for its side effects (plotting).
#'
#' @details Creates a bar plot showing the posterior probability that each
#'   observation belongs to component 2 (z_t = 1). Bars are colored based on
#'   whether the probability exceeds the threshold (default 0.5), making it
#'   easy to identify the most likely component assignment at each time point.
#'
#'   \strong{Interpretation:}
#'   \itemize{
#'     \item \strong{Orange bars} (P(z_t = 1) <= 0.5): Observation more likely
#'       from Component 1 (lower mean component in the constraint mu_1 < mu_2)
#'     \item \strong{Violet bars} (P(z_t = 1) > 0.5): Observation more likely
#'       from Component 2 (higher mean component)
#'     \item \strong{Red X markers} (if true_z provided): True component membership
#'   }
#'
#'   The color scheme matches the mixture parameter diagnostics, where:
#'   \itemize{
#'     \item darkorange represents Component 1 (mu_1, phi_1)
#'     \item darkviolet represents Component 2 (mu_2, phi_2)
#'   }
#'
#' @keywords internal
#' @noRd
plot_component_probabilities_base <- function(z,
                                              threshold = 0.5,
                                              color_above = "darkviolet",
                                              color_below = "darkorange",
                                              true_z = NULL,
                                              ...) {

  # ===========================================================================
  # INPUT VALIDATION
  # ===========================================================================

  if (!is.matrix(z)) {
    stop("`z` must be a matrix")
  }

  if (!is.numeric(threshold) || length(threshold) != 1 ||
      threshold <= 0 || threshold >= 1) {
    stop("`threshold` must be a single numeric value between 0 and 1")
  }

  # Validate true_z if provided
  if (!is.null(true_z)) {
    if (!is.numeric(true_z)) {
      warning("`true_z` must be numeric. Ignoring true values.")
      true_z <- NULL
    } else if (length(true_z) != ncol(z)) {
      warning("`true_z` length must match number of time points. Ignoring true values.")
      true_z <- NULL
    }
  }

  # ===========================================================================
  # COMPUTE POSTERIOR PROBABILITIES
  # ===========================================================================

  n_obs <- ncol(z)
  time_grid <- seq_len(n_obs)

  # Compute posterior probabilities P(z_t = 1 | data)
  z_prob <- apply(z, 2, mean)

  # ===========================================================================
  # SETUP PLOTTING ENVIRONMENT
  # ===========================================================================

  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar), add = TRUE)

  par(mfrow = c(1, 1),
      mar = c(4, 4, 2, 1),
      oma = c(0, 0, 2, 0),
      mgp = c(2.5, 1, 0))

  # ===========================================================================
  # CREATE BAR PLOT WITH COMPONENT COLORS
  # ===========================================================================

  # Create bar plot with threshold-based coloring using component colors
  # color_below (darkorange) = Component 1 (z_t = 0, or P(z_t = 1) <= 0.5)
  # color_above (darkviolet) = Component 2 (z_t = 1, or P(z_t = 1) > 0.5)
  plot(z_prob,
       type = "h",
       lwd = 2,
       col = ifelse(z_prob > threshold, color_above, color_below),
       xlab = "Time",
       ylab = expression(paste("p(", z[t], " = 1 | data)")),
       ylim = c(0, 1.1),
       axes = FALSE)

  axis(side = 1)
  axis(side = 2, at = c(0, 0.5, 1))

  # ===========================================================================
  # ADD REFERENCE LINES AND TRUE VALUES
  # ===========================================================================

  # Add threshold reference line
  segments(x0 = 1,
           y0 = threshold,
           x1 = n_obs,
           y1 = threshold,
           col = "darkgray",
           lwd = 2,
           lty = 2)

  # Overlay true z values if provided
  if (!is.null(true_z)) {
    points(time_grid,
           true_z,
           pch = 4,
           cex = 0.5,
           lwd = 2.5,
           col = "black")
  }

  # ===========================================================================
  # ADD GRID
  # ===========================================================================

  grid(nx = NA, ny = NULL)

  # ===========================================================================
  # BUILD AND DISPLAY LEGEND
  # ===========================================================================

  # Build legend with component-based interpretation
  legend_items <- c(
    paste0("p(z = 1 | data) > ", threshold, "   "),
    paste0("p(z = 1 | data) \u2264 ", threshold, "   "),
    "Threshold"
  )
  legend_cols <- c(color_above, color_below, "darkgray")
  legend_lty <- c(1, 1, 2)
  legend_lwd <- c(2, 2, 2)
  legend_pch <- c(NA, NA, NA)

  # Add true_z to legend if provided
  if (!is.null(true_z)) {
    legend_items <- c(legend_items, expression(paste("True ", z[t])))
    legend_cols <- c(legend_cols, "black")
    legend_lty <- c(legend_lty, NA)
    legend_lwd <- c(legend_lwd, 2.5)
    legend_pch <- c(legend_pch, 4)
  }

  legend("topright",
         horiz = TRUE,
         legend = legend_items,
         col = legend_cols,
         lty = legend_lty,
         lwd = legend_lwd,
         pch = legend_pch,
         bty = "n",
         cex = 0.9)

  # ===========================================================================
  # OVERALL TITLE
  # ===========================================================================

  mtext(expression(paste("Posterior Probability: p(", z[t], " = 1 | data)")),
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
#'   `plot_alpha_trajectory_base()` function. It automatically computes
#'   observed proportions from y/n_trials.
#'
#'   Used by:
#'   \itemize{
#'     \item `plot.binomial_locallevel`
#'     \item `plot.binomial_localtrend`
#'     \item `plot.binomial_localacceleration`
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
    ylim_auto = FALSE,
    ...
  )

  invisible(NULL)
}


#' Plot acceptance proportions (base graphics)
#'
#' @description Plots Metropolis-Hastings acceptance proportions over time points,
#'   showing median acceptance with min-max range and target reference line.
#'
#' @param accept_prop Matrix of acceptance proportions (n_draws x n_obs).
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
#'     \item `plot.probit_bernoulli_locallevel`
#'     \item `plot.probit_bernoulli_localtrend`
#'     \item `plot.probit_bernoulli_localacceleration`
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

#' Plot Poisson rates (base graphics)
#'
#' @description Wrapper function for plotting Poisson model rates
#'   (alpha_t). Used by all Poisson model types (locallevel, localtrend,
#'   localacceleration).
#'
#' @param x An object inheriting from a poisson model class.
#' @param ci Logical; whether to display credible intervals.
#' @param ci_level Numeric; credible interval level.
#' @param show_obs Logical; whether to display observed counts.
#'   Default is TRUE.
#' @param true_alpha Numeric vector; true alpha values for simulation studies.
#'   If provided, overlays the true trajectory.
#' @param ... Additional arguments passed to plot_alpha_trajectory_base.
#'
#' @return NULL (invisibly). Function is called for side effects (plotting).
#'
#' @details This function extracts alpha samples and observed data from
#'   Poisson model objects and delegates to the generic
#'   `plot_alpha_trajectory_base()` function. Unlike binomial models,
#'   observed counts are plotted directly without proportion calculations,
#'   and the y-axis is automatically scaled to accommodate count data.
#'
#'   Used by:
#'   \itemize{
#'     \item `plot.poisson_locallevel`
#'     \item `plot.poisson_localtrend`
#'     \item `plot.poisson_localacceleration`
#'   }
#'
#' @keywords internal
#' @noRd
plot_poisson_alpha_base <- function(x,
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

  # Prepare observed counts if available and requested
  obs_data <- NULL
  if (show_obs && !is.null(y)) {
    obs_data <- y  # Direct counts, no proportion calculation
  }

  # Delegate to generic alpha trajectory plotting function with automatic ylim
  plot_alpha_trajectory_base(
    alpha = x$alpha,
    ci = ci,
    ci_level = ci_level,
    title = "Poisson Rates",
    obs_data = obs_data,
    obs_label = expression(y[t]),
    show_obs = show_obs,
    obs_color = grDevices::rgb(0.75, 0.3, 0.0, 0.5),
    obs_pch = 16,
    obs_cex = 0.8,
    true_alpha = true_alpha,
    ylim_auto = TRUE,  # KEY:  Enable automatic y-axis scaling for Poisson rates
    ...
  )

  invisible(NULL)
}


#' Plot mixture weight trajectory with credible intervals (base graphics)
#'
#' @description Creates a two-page visualization of mixture weights:
#'   Page 1 shows alpha_t trajectory with credible bands and optional data overlay,
#'   Page 2 shows posterior probabilities P(z_t = 1 | data) with component colors.
#'
#' @param alpha Matrix of MCMC samples for mixture weights (n_draws x n_obs).
#' @param z Matrix of MCMC samples for component indicators (n_draws x n_obs).
#' @param ci Logical; whether to display credible intervals.
#' @param ci_level Numeric between 0 and 1; credible interval level.
#' @param overlay_data Logical; whether to overlay observed data on alpha plot.
#'   Default is TRUE. If TRUE and observed data is available, the data is
#'   rescaled to \[0, 1\] and plotted on Page 1 for visual context.
#' @param obs_data Numeric vector of observed data values (length n_obs).
#'   If NULL, attempts to extract from attributes. If not available and
#'   overlay_data = TRUE, a warning is issued. Data is automatically rescaled
#'   to the unit interval for visualization.
#' @param true_alpha Numeric vector; true alpha values for simulation studies.
#'   If provided, overlays the true trajectory on Page 1.
#' @param true_z Numeric vector; true component indicators for simulation studies.
#'   If provided, overlays the true values on Page 2.
#' @param ... Additional arguments passed to plotting functions.
#'
#' @return NULL (invisibly). Function is called for its side effects (plotting).
#'
#' @details
#'   This function delegates to two specialized functions:
#'   \itemize{
#'     \item `plot_alpha_trajectory_base()`: Page 1 (alpha_t trajectory)
#'     \item `plot_component_probabilities_base()`: Page 2 (z_t probabilities)
#'   }
#'
#'   \strong{Data Overlay and Rescaling (Page 1):}
#'   When `overlay_data = TRUE` and observed data is available, the original
#'   data is rescaled to the unit interval \[0, 1\] using min-max normalization:
#'   \deqn{y_{scaled} = \frac{y - \min(y)}{\max(y) - \min(y)}}
#'
#'   This rescaling improves visibility by mapping the data to the same scale
#'   as the mixture weights (alpha_t in \[0, 1\]). The rescaled data helps identify
#'   temporal patterns and potential relationships between observed values and
#'   component membership probabilities.
#'
#'   \strong{Important:} The rescaling is purely for visualization purposes and
#'   does not affect the model estimation. The legend clearly indicates that
#'   the displayed data is rescaled.
#'
#'   \strong{Color Scheme for Component Probabilities (Page 2):}
#'   Page 2 uses the mixture component colors to visualize membership probabilities:
#'   \itemize{
#'     \item \strong{darkorange} (Component 1): P(z_t = 1) <= 0.5 (more likely Component 1)
#'     \item \strong{darkviolet} (Component 2): P(z_t = 1) > 0.5 (more likely Component 2)
#'   }
#'
#'   This color scheme matches the mixture component parameter colors used in
#'   MCMC diagnostics, providing visual consistency across all plots.
#'
#' @keywords internal
#' @noRd
plot_mixture_weights_base <- function(alpha,
                                      z,
                                      ci = TRUE,
                                      ci_level = 0.95,
                                      overlay_data = TRUE,
                                      obs_data = NULL,
                                      true_alpha = NULL,
                                      true_z = NULL,
                                      ...) {

  # ===========================================================================
  # INPUT VALIDATION
  # ===========================================================================

  if (!is.matrix(alpha) || !is.matrix(z)) {
    stop("`alpha` and `z` must be matrices")
  }

  if (ncol(alpha) != ncol(z)) {
    stop("`alpha` and `z` must have the same number of columns (time points)")
  }

  # Validate ci_level
  if (ci && (!is.numeric(ci_level) || length(ci_level) != 1 ||
             ci_level <= 0 || ci_level >= 1)) {
    stop("`ci_level` must be a single numeric value between 0 and 1")
  }

  # Validate overlay_data
  if (!is.logical(overlay_data) || length(overlay_data) != 1) {
    stop("`overlay_data` must be a single logical value")
  }

  # ===========================================================================
  # PREPARE OBSERVED DATA FOR OVERLAY WITH RESCALING
  # ===========================================================================

  # Determine if we should show observed data
  show_obs <- overlay_data && !is.null(obs_data)
  obs_data_rescaled <- NULL
  obs_label <- expression(y[t])

  # Validate and rescale obs_data if overlay is requested
  if (overlay_data && !is.null(obs_data)) {
    if (!is.numeric(obs_data)) {
      warning("`obs_data` must be numeric. Data overlay disabled.")
      show_obs <- FALSE
    } else if (length(obs_data) != ncol(alpha)) {
      warning("`obs_data` length must match number of time points. Data overlay disabled.")
      show_obs <- FALSE
    } else {
      # Rescale data to [0, 1] using min-max normalization
      min_y <- min(obs_data, na.rm = TRUE)
      max_y <- max(obs_data, na.rm = TRUE)

      # Check if all values are the same (avoid division by zero)
      if (max_y - min_y < .Machine$double.eps) {
        warning("Observed data has no variation (all values equal).",
                "Data overlay disabled.")
        show_obs <- FALSE
      } else {
        obs_data_rescaled <- (obs_data - min_y) / (max_y - min_y)

        # Update label to indicate rescaling
        obs_label <- expression(paste(y[t], " (rescaled)"))

        # Optional: Print rescaling info for user reference
        message("Note: Observed data rescaled to [0, 1] for visualization.")
        message(sprintf("  Original range: [%.3f, %.3f]", min_y, max_y))
      }
    }
  }

  # Issue message if overlay requested but no data available
  if (overlay_data && is.null(obs_data)) {
    message("Note: overlay_data = TRUE but no observed data available.",
            "Plotting without data overlay.")
  }

  # ===========================================================================
  # PAGE 1: Alpha trajectory with credible bands and optional rescaled data
  # ===========================================================================

  plot_alpha_trajectory_base(
    alpha = alpha,
    ci = ci,
    ci_level = ci_level,
    title = expression(paste("Time-Varying Mixture Weight: ", alpha[t])),
    obs_data = obs_data_rescaled,
    obs_label = obs_label,
    show_obs = show_obs,
    obs_color = grDevices::rgb(0.75, 0.3, 0.0, 0.5),
    obs_pch = 16,
    obs_cex = 0.8,
    true_alpha = true_alpha,
    ...
  )

  # ===========================================================================
  # PAGE 2: Component probabilities with mixture colors
  # ===========================================================================

  plot_component_probabilities_base(
    z = z,
    threshold = 0.5,
    color_above = "darkviolet",   # Component 2 color (z_t = 1)
    color_below = "darkorange",   # Component 1 color (z_t = 0)
    true_z = true_z,
    ...
  )

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
#' @param overlay_data Logical; whether to overlay observed data on mixture
#'   weight plots. Default is TRUE. If TRUE and data is available, observed
#'   values are rescaled to \[0, 1\] and plotted for visual context.
#' @param true_values Named list containing true values (or NULL).
#'   Expected elements depend on model type and order. For mixture models:
#'   \itemize{
#'     \item Mixture parameters: mu_1, mu_2, prec_1, prec_2
#'     \item Initial states: theta_01, theta_02, theta_03
#'     \item Innovation precisions: prec_theta1, prec_theta2, prec_theta3
#'     \item State trajectories: theta_1, theta_2, theta_3 (vectors of length n_obs)
#'     \item Mixture weights: alpha (vector of length n_obs)
#'     \item Component indicators: z (vector of length n_obs)
#'   }
#' @param ... Additional arguments passed to plotting functions.
#'
#' @return NULL (invisibly). Function is called for side effects (plotting).
#'
#' @details Creates a complete dashboard with the following pages:
#'   \enumerate{
#'     \item MCMC diagnostics for each parameter (4-panel plots)
#'     \item Mixture parameters bivariate relationships (mixture models only)
#'     \item Dynamic state trajectories with credible bands
#'     \item Dynamic state diagnostics (transitions and innovations)
#'     \item Phase space trajectories (order 2-3 only)
#'     \item Mixture weights and component probabilities (mixture models only)
#'   }
#'
#'   The total number of pages varies by model:
#'   \itemize{
#'     \item \strong{Mixture + Order 1:} 6 + 2 + 2 = 10 pages
#'     \item \strong{Mixture + Order 2:} 8 + 2 + 3 + 2 = 15 pages
#'     \item \strong{Mixture + Order 3:} 10 + 2 + 3 + 2 = 17 pages
#'   }
#'
#' @keywords internal
#' @noRd
plot_all_mixture_generic_base <- function(x,
                                          ask = TRUE,
                                          ci = TRUE,
                                          ci_level = 0.95,
                                          overlay_data = TRUE,
                                          true_values = NULL,
                                          ...) {

  # ===========================================================================
  # SETUP PLOTTING ENVIRONMENT
  # ===========================================================================

  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  if (ask) {
    oldask <- par(ask = TRUE)
    on.exit(par(oldask), add = TRUE)
  }

  # ===========================================================================
  # DETECT MODEL CHARACTERISTICS
  # ===========================================================================

  model_info <- detect_model_type(x)
  model_order <- model_info$model_order

  # Get parameter configuration
  param_config <- get_param_config(x,
                                   model_info$model_class,
                                   model_order)
  n_params <- length(param_config)

  # ===========================================================================
  # SECTION 1: MCMC diagnostics for each parameter
  # ===========================================================================

  plot_mcmc_diagnostics_generic(x,
                                which = seq_len(n_params),
                                param_config = param_config,
                                true_values = true_values,
                                ...)

  # ===========================================================================
  # SECTION 2: Mixture parameters (mixture models only)
  # ===========================================================================

  if (model_info$has_mixture) {
    plot_mixture_params_base(x$mu_1,
                             x$mu_2,
                             x$prec_1,
                             x$prec_2,
                             true_values = true_values,
                             ...)
  }

  # ===========================================================================
  # SECTION 3: Dynamic states (trajectories, diagnostics, phase space)
  # ===========================================================================

  plot_dynamic_states_generic_base(x,
                                   model_order = model_order,
                                   ci = ci,
                                   ci_level = ci_level,
                                   true_values = true_values,
                                   ...)

  # ===========================================================================
  # SECTION 4: Mixture weights (mixture models only)
  # ===========================================================================

  if (model_info$has_mixture) {
    plot_mixture_weights_base(x$alpha,
                              x$z,
                              ci = ci,
                              ci_level = ci_level,
                              overlay_data = overlay_data,
                              obs_data = attr(x, "y"),
                              true_alpha = true_values$alpha,
                              true_z = true_values$z)
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
