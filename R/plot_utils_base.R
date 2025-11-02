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
         pch = 16, col = grDevices::rgb(0.1, 0.3, 0.6, 0.3))
    grid()
  }

  # Plot 2: mu_1 vs phi_1
  if (2 %in% which) {
    plot(mu_1, prec_1,
         xlab = expression(mu[1]),
         ylab = expression(phi[1]),
         main = expression(paste(mu[1], " vs ", phi[1])),
         pch = 16, col = grDevices::rgb(0.1, 0.5, 0.2, 0.3))
    grid()
  }

  # Plot 3: mu_2 vs phi_2
  if (3 %in% which) {
    plot(mu_2, prec_2,
         xlab = expression(mu[2]),
         ylab = expression(phi[2]),
         main = expression(paste(mu[2], " vs ", phi[2])),
         pch = 16, col = grDevices::rgb(0.8, 0.4, 0.1, 0.3))
    grid()
  }

  # Plot 4: phi_1 vs phi_2
  if (4 %in% which) {
    plot(prec_1, prec_2,
         xlab = expression(phi[1]),
         ylab = expression(phi[2]),
         main = expression(paste(phi[1], " vs ", phi[2])),
         pch = 16, col = grDevices::rgb(0.5, 0.1, 0.5, 0.3))
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

# =============================================================================
# Generic Dashboard Functions (Base Graphics)
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
#' @param ... Additional arguments passed to plotting functions.
#'
#' @return NULL (invisibly). Function is called for side effects (plotting).
#'
#' @details Generates a complete dashboard with the following pages:
#'   \itemize{
#'     \item Pages 1-n: MCMC diagnostics for each parameter (4 panels each)
#'     \item Page n+1: Mixture component bivariate relationships
#'     \item Pages n+2 onwards: Dynamic state trajectories and diagnostics
#'     \item Final pages: Mixture weight alpha_t and component indicators z_t
#'   }
#'
#'   The number of pages adapts automatically based on model order:
#'   \itemize{
#'     \item Order 1 (locallevel): 10 total pages
#'     \item Order 2 (localtrend): 13 total pages
#'     \item Order 3 (localacceleration): 16 total pages
#'   }
#'
#' @keywords internal
#' @noRd
plot_all_mixture_generic_base <- function(x, ask = TRUE, ci = TRUE,
                                          ci_level = 0.95, ...) {

  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  if (ask) {
    oldask <- par(ask = TRUE)
    on.exit(par(oldask), add = TRUE)
  }

  # 1. Detect model type
  model_info <- detect_model_type(x)
  model_order <- model_info$model_order

  # 2. Get parameter configuration
  param_config <- get_param_config(x, model_info$model_class, model_order)
  n_params <- length(param_config)

  # 3. Pages 1-n: MCMC diagnostics for each parameter
  plot_mcmc_diagnostics_generic(x, which = seq_len(n_params),
                                param_config = param_config,
                                engine = "base", ...)

  # 4. Page n+1: Mixture parameters (only for mixture models)
  if (model_info$has_mixture) {
    plot_mixture_params_base(x$mu_1, x$mu_2, x$prec_1, x$prec_2, ...)
  }

  # 5. Pages n+2 onwards: Dynamic states
  plot_dynamic_states_generic_base(x, model_order = model_order,
                                   ci = ci, ci_level = ci_level, ...)

  # 6. Final pages: Mixture weights (only for mixture models)
  if (model_info$has_mixture) {
    plot_mixture_weights_base(x$alpha, x$z, ci = ci, ci_level = ci_level, ...)
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
#' @param ... Additional arguments (currently unused).
#'
#' @return NULL (invisibly). Function is called for side effects (plotting).
#'
#' @details Number of pages generated:
#'   \itemize{
#'     \item Order 1: 2 pages (trajectory + diagnostics)
#'     \item Order 2: 2 pages (trajectories + diagnostics with scatter plot)
#'     \item Order 3: 3 pages (trajectories + diagnostics + state space)
#'   }
#'
#' @keywords internal
#' @noRd
plot_dynamic_states_generic_base <- function(x, which = NULL,
                                             model_order = NULL,
                                             ci = TRUE, ci_level = 0.95, ...) {

  # Auto-detect model order if needed
  if (is.null(model_order)) {
    model_info <- detect_model_type(x)
    model_order <- model_info$model_order
  }

  validate_ci_level(ci_level)

  # Determine number of pages (order 1 now has 2 pages)
  n_pages <- if (model_order == 1) 2L else model_order

  if (is.null(which)) {
    which <- seq_len(n_pages)
  }

  # Validate which
  if (any(which < 1) || any(which > n_pages)) {
    stop("`which` must be between 1 and ", n_pages)
  }

  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  n_obs <- attr(x, "n_obs")
  time_grid <- seq_len(n_obs)

  # Prepare CI parameters
  if (ci) {
    ci_lower_prob <- (1 - ci_level) / 2
    ci_upper_prob <- 1 - ci_lower_prob
    ci_label <- paste0(round(ci_level * 100), "% CI")
  }

  # Get state summaries and labels
  state_summaries <- summarise_all_states(x, model_order, ci, ci_level)
  state_labels <- get_state_labels(model_order)

  # =========================================================================
  # Page 1: State Trajectories
  # =========================================================================

  if (1 %in% which) {
    par(mfrow = c(model_order, 1), mar = c(4, 4, 3, 1),
        oma = c(0, 0, 2, 0), mgp = c(2.5, 1, 0))

    for (i in seq_len(model_order)) {
      state_name <- state_labels$state_names[i]
      summary_i <- state_summaries[[state_name]]

      plot_state_trajectory_base(
        time_grid = time_grid,
        median = summary_i$median,
        lower = summary_i$lower,
        upper = summary_i$upper,
        ylab = state_labels$state_labels[[i]],
        main = state_labels$state_titles[i],
        col = state_labels$state_colors[i],
        ci = ci,
        ci_label = ci_label
      )
    }

    mtext("Dynamic State Trajectories", outer = TRUE, cex = 1.3, font = 2)
  }

  # =========================================================================
  # Page 2: Diagnostics - Different layouts for each order
  # =========================================================================

  if (2 %in% which) {
    # Compute innovations
    innovations <- compute_innovations(x, model_order)
    innov_summaries <- lapply(innovations, summarise_state, ci, ci_level)
    innov_labels <- get_innovation_labels(model_order)

    # -----------------------------------------------------------------------
    # ORDER 1: [Trajectory, Innovation] side by side
    # -----------------------------------------------------------------------
    if (model_order == 1) {
      par(mfrow = c(1, 1), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0),
          mgp = c(2.5, 1, 0))

      plot_innovation_base(
        time_grid = time_grid,
        median = innov_summaries$innov_1$median,
        lower = innov_summaries$innov_1$lower,
        upper = innov_summaries$innov_1$upper,
        ylab = innov_labels$innov_labels[[1]],
        main = innov_labels$innov_titles[1],
        col_bar = innov_labels$innov_colors[1],
        ci = ci,
        ci_label = ci_label
      )

      mtext("Dynamic State Diagnostics", outer = TRUE, cex = 1.3, font = 2)
    }

    # -----------------------------------------------------------------------
    # ORDER 2: [Scatter, Innov1, Joint, Innov2]
    # -----------------------------------------------------------------------
    if (model_order == 2) {
      par(mfrow = c(2, 2), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))

      # Plot 2.1: State Space (theta_1 vs theta_2)
      theta_df <- data.frame(
        theta1 = as.vector(x$theta_1),
        theta2 = as.vector(x$theta_2)
      )

      max_points <- 5000L
      if (nrow(theta_df) > max_points) {
        set.seed(123)
        theta_df <- theta_df[sample.int(nrow(theta_df), max_points), ]
      }

      plot(theta_df$theta1, theta_df$theta2,
           xlab = expression(theta["t,1"]),
           ylab = expression(theta["t,2"]),
           main = "State Space",
           pch = 16, cex = 0.9, col = grDevices::rgb(1, 0.5, 0, 0.2))
      grid()

      # Plot 2.2: Level Innovations
      plot_innovation_base(
        time_grid = time_grid,
        median = innov_summaries$innov_1$median,
        lower = innov_summaries$innov_1$lower,
        upper = innov_summaries$innov_1$upper,
        ylab = innov_labels$innov_labels[[1]],
        main = innov_labels$innov_titles[1],
        col_bar = innov_labels$innov_colors[1],
        ci = ci,
        ci_label = ci_label
      )

      # Plot 2.3: Joint Trajectories
      ylim_range <- range(c(state_summaries$theta_1$median,
                            state_summaries$theta_2$median))
      if (diff(ylim_range) == 0) {
        ylim_range <- ylim_range + c(-0.75, 0.75)
      }
      ylim_range[2] <- ylim_range[2] + 0.3 * diff(ylim_range)

      plot(time_grid, state_summaries$theta_1$median, type = "l", lwd = 2,
           col = state_labels$state_colors[1],
           xlab = "Time", ylab = "State Value",
           main = "Joint Trajectories", ylim = ylim_range)
      lines(time_grid, state_summaries$theta_2$median,
            col = state_labels$state_colors[2], lwd = 2)
      grid()
      legend("topright", horiz = TRUE,
             legend = state_labels$state_labels,
             col = state_labels$state_colors,
             lty = 1, lwd = 2, bty = "n")

      # Plot 2.4: Trend Innovations
      plot_innovation_base(
        time_grid = time_grid,
        median = innov_summaries$innov_2$median,
        lower = innov_summaries$innov_2$lower,
        upper = innov_summaries$innov_2$upper,
        ylab = innov_labels$innov_labels[[2]],
        main = innov_labels$innov_titles[2],
        col_bar = innov_labels$innov_colors[2],
        ci = ci,
        ci_label = ci_label
      )

      mtext("Dynamic State Diagnostics", outer = TRUE, cex = 1.3, font = 2)
    }

    # -----------------------------------------------------------------------
    # ORDER 3: [Joint, Innov1, Innov2, Innov3]
    # -----------------------------------------------------------------------
    if (model_order == 3) {
      par(mfrow = c(2, 2), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))

      # Plot 2.1: Joint Trajectories
      ylim_range <- range(c(state_summaries$theta_1$median,
                            state_summaries$theta_2$median,
                            state_summaries$theta_3$median))
      if (diff(ylim_range) == 0) {
        ylim_range <- ylim_range + c(-0.75, 0.75)
      }
      ylim_range[2] <- ylim_range[2] + 0.3 * diff(ylim_range)

      plot(time_grid, state_summaries$theta_1$median, type = "l", lwd = 2,
           col = state_labels$state_colors[1],
           xlab = "Time", ylab = "State Value",
           main = "Joint Trajectories", ylim = ylim_range)
      lines(time_grid, state_summaries$theta_2$median,
            col = state_labels$state_colors[2], lwd = 2)
      lines(time_grid, state_summaries$theta_3$median,
            col = state_labels$state_colors[3], lwd = 2)
      grid()
      legend("topright", horiz = TRUE,
             legend = state_labels$state_labels,
             col = state_labels$state_colors,
             lty = 1, lwd = 2, bty = "n")

      # Plot 2.2: Level Innovations
      plot_innovation_base(
        time_grid = time_grid,
        median = innov_summaries$innov_1$median,
        lower = innov_summaries$innov_1$lower,
        upper = innov_summaries$innov_1$upper,
        ylab = innov_labels$innov_labels[[1]],
        main = innov_labels$innov_titles[1],
        col_bar = innov_labels$innov_colors[1],
        ci = ci,
        ci_label = ci_label
      )

      # Plot 2.3: Trend Innovations
      plot_innovation_base(
        time_grid = time_grid,
        median = innov_summaries$innov_2$median,
        lower = innov_summaries$innov_2$lower,
        upper = innov_summaries$innov_2$upper,
        ylab = innov_labels$innov_labels[[2]],
        main = innov_labels$innov_titles[2],
        col_bar = innov_labels$innov_colors[2],
        ci = ci,
        ci_label = ci_label
      )

      # Plot 2.4: Acceleration Innovations
      plot_innovation_base(
        time_grid = time_grid,
        median = innov_summaries$innov_3$median,
        lower = innov_summaries$innov_3$lower,
        upper = innov_summaries$innov_3$upper,
        ylab = innov_labels$innov_labels[[3]],
        main = innov_labels$innov_titles[3],
        col_bar = innov_labels$innov_colors[3],
        ci = ci,
        ci_label = ci_label
      )

      mtext("Dynamic State Diagnostics", outer = TRUE, cex = 1.3, font = 2)
    }
  }

  # =========================================================================
  # Page 3: State Space Relationships (order 3 only)
  # =========================================================================

  if (3 %in% which && model_order == 3) {
    par(mfrow = c(1, 1), mar = c(4, 4, 3, 1), oma = c(0, 0, 0, 0))

    # Subsample for performance
    max_points <- 5000L
    theta_df <- data.frame(
      theta1 = as.vector(x$theta_1),
      theta2 = as.vector(x$theta_2),
      theta3 = as.vector(x$theta_3)
    )

    if (nrow(theta_df) > max_points) {
      set.seed(123)
      theta_df <- theta_df[sample.int(nrow(theta_df), max_points), ]
    }

    pairs(theta_df,
          pch = 16, cex = 0.6,
          col = grDevices::rgb(0.2, 0.5, 0.8, 0.2),
          main = "Pairwise State Relationships",
          labels = c(expression(theta["t,1"]),
                     expression(theta["t,2"]),
                     expression(theta["t,3"])))
  }

  invisible(NULL)
}
