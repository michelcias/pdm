#' Internal plotting utilities for ggplot2 graphics
#'
#' @description This file contains shared plotting functions using ggplot2
#'   for MCMC diagnostic visualizations. These functions provide modern,
#'   publication-ready graphics as an alternative to base R plots.
#'
#' @details These functions are not exported and are intended for internal
#'   use only by the plot.* methods. They require the ggplot2 package
#'   and optionally use patchwork for layout composition.
#'
#' @keywords internal
#' @noRd
NULL


#' Plot 4-panel MCMC diagnostics for a single parameter (ggplot2)
#'
#' @description Creates a comprehensive diagnostic page with trace plot,
#'   autocorrelation function, posterior density, and running mean convergence
#'   diagnostic using ggplot2.
#'
#' @param param_samples Numeric vector of MCMC samples for the parameter.
#' @param param_name Character string for the parameter name (e.g., "mu_1").
#' @param param_label_text Character string for axis labels (e.g., "mu_1").
#'   This will be converted to proper expressions internally.
#' @param ... Additional arguments (currently unused).
#'
#' @return If patchwork is available, returns a combined ggplot object.
#'   Otherwise, prints plots sequentially and returns NULL invisibly.
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
plot_param_diagnostics_ggplot <- function(param_samples,
                                          param_name,
                                          param_label_text,
                                          ...) {

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for ggplot2 engine")
  }

  # Prepare data
  df <- data.frame(
    iteration = seq_along(param_samples),
    value = param_samples,
    running_mean = cumsum(param_samples) / seq_along(param_samples)
  )

  # 1. Trace plot
  p1 <- ggplot2::ggplot(df, ggplot2::aes(x = .data$iteration, y = .data$value)) +
    ggplot2::geom_line(ggplot2::aes(color = "Trace"), linewidth = 0.5) +
    ggplot2::geom_hline(ggplot2::aes(yintercept = median(param_samples),
                                     color = "Median"),
                        linetype = "dashed", linewidth = 1) +
    ggplot2::scale_color_manual(
      values = c("Trace" = "gray40", "Median" = "red"),
      breaks = c("Trace", "Median")
    ) +
    ggplot2::labs(title = "Trace Plot", x = "Iteration") +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      legend.position = "top",
      legend.title = ggplot2::element_blank(),
      legend.direction = "horizontal"
    )

  # 2. ACF plot
  acf_data <- stats::acf(param_samples, plot = FALSE)
  df_acf <- data.frame(
    lag = acf_data$lag,
    acf = acf_data$acf
  )

  ci_line <- 1.96 / sqrt(length(param_samples))

  p2 <- ggplot2::ggplot(df_acf, ggplot2::aes(x = .data$lag, y = .data$acf)) +
    ggplot2::geom_hline(yintercept = 0, color = "black") +
    ggplot2::geom_segment(ggplot2::aes(xend = .data$lag, yend = 0),
                          color = "steelblue", linewidth = 1) +
    ggplot2::geom_hline(yintercept = c(-ci_line, ci_line),
                        linetype = "dashed", color = "blue") +
    ggplot2::labs(title = "Autocorrelation", x = "Lag", y = "ACF") +
    ggplot2::theme_minimal()

  # 3. Density plot
  p3 <- ggplot2::ggplot(df, ggplot2::aes(x = .data$value)) +
    ggplot2::geom_density(fill = "darkgreen", alpha = 0.3, linewidth = 1) +
    ggplot2::geom_vline(ggplot2::aes(xintercept = median(param_samples),
                                     color = "Median"),
                        linetype = "dashed", linewidth = 1) +
    ggplot2::geom_vline(ggplot2::aes(xintercept = mean(param_samples),
                                     color = "Mean"),
                        linetype = "dotted", linewidth = 1) +
    ggplot2::scale_color_manual(
      values = c("Median" = "red", "Mean" = "blue"),
      breaks = c("Median", "Mean")
    ) +
    ggplot2::guides(
      color = ggplot2::guide_legend(
        order = 1,
        override.aes = list(linetype = c("dashed", "dotted"),
                            linewidth = 1)
      )
    ) +
    ggplot2::labs(title = "Posterior Density", y = "Density") +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      legend.position = "top",
      legend.title = ggplot2::element_blank(),
      legend.direction = "horizontal"
    )

  # 4. Running mean
  p4 <- ggplot2::ggplot(df, ggplot2::aes(x = .data$iteration, y = .data$running_mean)) +
    ggplot2::geom_line(ggplot2::aes(color = "Running Mean"), linewidth = 1) +
    ggplot2::geom_hline(ggplot2::aes(yintercept = median(param_samples),
                                     color = "Median"),
                        linetype = "dashed", linewidth = 1) +
    ggplot2::scale_color_manual(
      values = c("Running Mean" = "steelblue", "Median" = "red"),
      breaks = c("Running Mean", "Median")
    ) +
    ggplot2::labs(title = "Running Mean", x = "Iteration") +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      legend.position = "top",
      legend.title = ggplot2::element_blank(),
      legend.direction = "horizontal"
    )

  # Convert parameter names to expressions for axis labels and title
  param_expr <- switch(param_label_text,
                       "mu_1" = expression(mu[1]),
                       "mu_2" = expression(mu[2]),
                       "phi_1" = expression(phi[1]),
                       "phi_2" = expression(phi[2]),
                       "V^{-1}" = expression(V^{-1}),
                       "theta_01" = expression(theta["0,1"]),
                       "theta_02" = expression(theta["0,2"]),
                       "theta_03" = expression(theta["0,3"]),
                       "W_1^{-1}" = expression(W[1]^{-1}),
                       "W_2^{-1}" = expression(W[2]^{-1}),
                       "W_3^{-1}" = expression(W[3]^{-1}),
                       param_label_text  # fallback to original text
  )

  # Create title expression
  title_expr <- switch(param_label_text,
                       "mu_1" = expression(paste("MCMC Diagnostics: ", mu[1])),
                       "mu_2" = expression(paste("MCMC Diagnostics: ", mu[2])),
                       "phi_1" = expression(paste("MCMC Diagnostics: ", phi[1])),
                       "phi_2" = expression(paste("MCMC Diagnostics: ", phi[2])),
                       "V^{-1}" = expression(paste("MCMC Diagnostics: ", V^{-1})),
                       "theta_01" = expression(paste("MCMC Diagnostics: ", theta["0,1"])),
                       "theta_02" = expression(paste("MCMC Diagnostics: ", theta["0,2"])),
                       "theta_03" = expression(paste("MCMC Diagnostics: ", theta["0,3"])),
                       "W_1^{-1}" = expression(paste("MCMC Diagnostics: ", W[1]^{-1})),
                       "W_2^{-1}" = expression(paste("MCMC Diagnostics: ", W[2]^{-1})),
                       "W_3^{-1}" = expression(paste("MCMC Diagnostics: ", W[3]^{-1})),
                       paste("MCMC Diagnostics:", param_name)  # fallback
  )

  # Add y-axis labels with expressions
  p1 <- p1 + ggplot2::ylab(param_expr)
  p3 <- p3 + ggplot2::xlab(param_expr)
  p4 <- p4 + ggplot2::ylab(param_expr)

  # Combine with patchwork if available
  if (requireNamespace("patchwork", quietly = TRUE)) {
    combined <- (p1 + p2) / (p3 + p4) +
      patchwork::plot_annotation(
        title = title_expr,
        theme = ggplot2::theme(plot.title = ggplot2::element_text(size = 16,
                                                                  face = "bold"))
      )
    return(combined)
  } else {
    # Print sequentially
    print(p1)
    print(p2)
    print(p3)
    print(p4)
    return(invisible(NULL))
  }
}


#' Plot mixture component bivariate relationships (ggplot2)
#'
#' @description Creates a 2x2 grid of scatterplots showing relationships
#'   between mixture component parameters using ggplot2.
#'
#' @param mu_1 Numeric vector of MCMC samples for component 1 mean.
#' @param mu_2 Numeric vector of MCMC samples for component 2 mean.
#' @param prec_1 Numeric vector of MCMC samples for component 1 precision.
#' @param prec_2 Numeric vector of MCMC samples for component 2 precision.
#' @param which Integer vector specifying which subplots to display (1:4).
#'   If NULL, all four plots are shown.
#' @param ... Additional arguments (currently unused).
#'
#' @return NULL (invisibly). Prints plots as side effects.
#'
#' @keywords internal
#' @noRd
plot_mixture_params_ggplot <- function(mu_1, mu_2, prec_1, prec_2,
                                       which = NULL, ...) {

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for ggplot2 engine")
  }

  if (is.null(which)) {
    which <- 1:4
  }

  if (any(!which %in% 1:4)) {
    stop("`which` must be between 1 and 4")
  }

  col_mu_mu <- grDevices::rgb(0.1, 0.3, 0.6, alpha = 0.3)
  col_mu_phi1 <- grDevices::rgb(0.1, 0.5, 0.2, alpha = 0.3)
  col_mu_phi2 <- grDevices::rgb(0.8, 0.4, 0.1, alpha = 0.3)
  col_phi_phi <- grDevices::rgb(0.5, 0.1, 0.5, alpha = 0.3)

  base_theme <- ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(
      panel.grid.major = ggplot2::element_line(color = "grey85"),
      panel.grid.minor = ggplot2::element_line(color = "grey92"),
      plot.title = ggplot2::element_text(face = "bold", size = 13),
      legend.position = "none"
    )

  plots <- list()

  add_plot <- function(p) {
    plots[[length(plots) + 1]] <<- p + base_theme
  }

  if (1 %in% which) {
    df1 <- data.frame(mu_1 = mu_1, mu_2 = mu_2)
    p1 <- ggplot2::ggplot(df1, ggplot2::aes(x = .data$mu_1, y = .data$mu_2)) +
      ggplot2::geom_point(color = col_mu_mu, shape = 16, size = 1.5) +
      ggplot2::labs(
        title = expression(paste(mu[1], " vs ", mu[2])),
        x = expression(mu[1]),
        y = expression(mu[2])
      )
    add_plot(p1)
  }

  if (2 %in% which) {
    df2 <- data.frame(mu_1 = mu_1, phi_1 = prec_1)
    p2 <- ggplot2::ggplot(df2, ggplot2::aes(x = .data$mu_1, y = .data$phi_1)) +
      ggplot2::geom_point(color = col_mu_phi1, shape = 16, size = 1.5) +
      ggplot2::labs(
        title = expression(paste(mu[1], " vs ", phi[1])),
        x = expression(mu[1]),
        y = expression(phi[1])
      )
    add_plot(p2)
  }

  if (3 %in% which) {
    df3 <- data.frame(mu_2 = mu_2, phi_2 = prec_2)
    p3 <- ggplot2::ggplot(df3, ggplot2::aes(x = .data$mu_2, y = .data$phi_2)) +
      ggplot2::geom_point(color = col_mu_phi2, shape = 16, size = 1.5) +
      ggplot2::labs(
        title = expression(paste(mu[2], " vs ", phi[2])),
        x = expression(mu[2]),
        y = expression(phi[2])
      )
    add_plot(p3)
  }

  if (4 %in% which) {
    df4 <- data.frame(phi_1 = prec_1, phi_2 = prec_2)
    p4 <- ggplot2::ggplot(df4, ggplot2::aes(x = .data$phi_1, y = .data$phi_2)) +
      ggplot2::geom_point(color = col_phi_phi, shape = 16, size = 1.5) +
      ggplot2::labs(
        title = expression(paste(phi[1], " vs ", phi[2])),
        x = expression(phi[1]),
        y = expression(phi[2])
      )
    add_plot(p4)
  }

  if (!length(plots)) {
    return(invisible(NULL))
  }

  if (requireNamespace("patchwork", quietly = TRUE) && length(plots) > 1) {
    combined <- patchwork::wrap_plots(plots, ncol = 2) +
      patchwork::plot_annotation(
        title = "Mixture Component Parameters (Bivariate Relationships)",
        theme = ggplot2::theme(
          plot.title = ggplot2::element_text(size = 14, face = "bold",
                                             hjust = 0.5)
        )
      )
    print(combined)
  } else {
    if (length(plots) == 1) {
      plots[[1]] <- plots[[1]] +
        ggplot2::labs(subtitle = "Mixture Component Parameters (Bivariate Relationships)") +
        ggplot2::theme(
          plot.subtitle = ggplot2::element_text(size = 14, face = "bold",
                                                hjust = 0.5)
        )
    }
    for (p in plots) {
      print(p)
    }
  }

  invisible(NULL)
}


#' Plot mixture weight trajectory with credible intervals (ggplot2)
#'
#' @description Creates a two-page visualization of mixture weights using ggplot2.
#'
#' @param alpha Matrix of MCMC samples for mixture weights (n_chain x n_obs).
#' @param z Matrix of MCMC samples for component indicators (n_chain x n_obs).
#' @param ci Logical; whether to display credible intervals.
#' @param ci_level Numeric between 0 and 1; credible interval level.
#' @param ... Additional arguments (currently unused).
#'
#' @return NULL (invisibly). Prints plots as side effects.
#'
#' @keywords internal
#' @noRd
plot_mixture_weights_ggplot <- function(alpha, z, ci = TRUE,
                                        ci_level = 0.95, ...) {

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for ggplot2 engine")
  }

  # Validate ci_level
  if (ci && (!is.numeric(ci_level) || length(ci_level) != 1 ||
             ci_level <= 0 || ci_level >= 1)) {
    stop("`ci_level` must be a single numeric value between 0 and 1")
  }

  n_obs <- ncol(alpha)
  time_grid <- seq_len(n_obs)

  # Calculate quantile probabilities
  ci_lower_prob <- (1 - ci_level) / 2
  ci_upper_prob <- 1 - ci_lower_prob
  ci_pct <- round(ci_level * 100)

  # Page 1: alpha_t
  df_alpha <- data.frame(
    time = time_grid,
    median = apply(alpha, 2, stats::median)
  )

  if (ci) {
    df_alpha$lower <- apply(alpha, 2, stats::quantile, probs = ci_lower_prob)
    df_alpha$upper <- apply(alpha, 2, stats::quantile, probs = ci_upper_prob)
  }

  p1 <- ggplot2::ggplot(df_alpha, ggplot2::aes(x = .data$time))

  if (ci) {
    p1 <- p1 +
      ggplot2::geom_ribbon(
        ggplot2::aes(ymin = .data$lower, ymax = .data$upper, fill = "CI"),
        alpha = 0.3
      ) +
      ggplot2::scale_fill_manual(
        values = c("CI" = "steelblue"),
        breaks = "CI",
        labels = paste0(ci_pct, "% CI")
      )
  }

  p1 <- p1 +
    ggplot2::geom_line(
      ggplot2::aes(y = .data$median, colour = "Median"),
      linewidth = 1.2
    ) +
    ggplot2::scale_color_manual(
      values = c("Median" = "blue"),
      breaks = "Median",
      labels = expression(hat(alpha)[t])
    ) +
    ggplot2::scale_y_continuous(
      limits = c(0, 1.0),
      breaks = seq(0, 1, by = 0.2)
    ) +
    ggplot2::labs(
      title = expression(paste("Time-Varying Mixture Weight: ", alpha[t])),
      x = "Time",
      y = expression(alpha[t])
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      panel.grid.major = ggplot2::element_line(color = "grey85"),
      panel.grid.minor = ggplot2::element_line(color = "grey92"),
      plot.title = ggplot2::element_text(face = "bold", size = 13),
      legend.position = "top",
      legend.title = ggplot2::element_blank(),
      legend.direction = "horizontal"
    )

  if (ci) {
    p1 <- p1 +
      ggplot2::guides(
        colour = ggplot2::guide_legend(order = 1),
        fill = ggplot2::guide_legend(order = 2)
      )
  } else {
    p1 <- p1 +
      ggplot2::guides(colour = ggplot2::guide_legend(order = 1))
  }

  print(p1)

  # Page 2: z_t
  z_prob <- apply(z, 2, mean)
  df_z <- data.frame(time = time_grid, prob = z_prob)

  p2 <- ggplot2::ggplot(df_z, ggplot2::aes(x = .data$time, y = .data$prob)) +
    ggplot2::geom_segment(
      ggplot2::aes(xend = .data$time, yend = 0, colour = .data$prob > 0.5),
      linewidth = 1.5
    ) +
    ggplot2::scale_colour_manual(
      values = c("TRUE" = "purple", "FALSE" = "blue"),
      breaks = c("FALSE", "TRUE"),
      labels = c(
        expression(paste("P(", z[t], " = 1 | data) ≤ 0.5")),
        expression(paste("P(", z[t], " = 1 | data) > 0.5"))
      )
    ) +
    ggplot2::geom_hline(
      ggplot2::aes(yintercept = 0.5, linetype = "Threshold"),
      color = "red",
      linewidth = 1
    ) +
    ggplot2::scale_linetype_manual(
      values = c("Threshold" = "dashed")
    ) +
    ggplot2::scale_y_continuous(
      limits = c(0, 1.0),
      breaks = c(0, 0.5, 1)
    ) +
    ggplot2::labs(
      title = expression(paste("Posterior Probability: P(", z[t], " = 1 | data)")),
      x = "Time",
      y = expression(paste("P(", z[t], " = 1 | data)")),
      colour = NULL,
      linetype = NULL
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      panel.grid.major = ggplot2::element_line(color = "grey85"),
      panel.grid.minor = ggplot2::element_line(color = "grey92"),
      panel.grid.major.x = ggplot2::element_blank(),
      plot.title = ggplot2::element_text(face = "bold", size = 13),
      legend.position = "top",
      legend.direction = "horizontal"
    ) +
    ggplot2::guides(
      colour = ggplot2::guide_legend(order = 1),
      linetype = ggplot2::guide_legend(order = 2)
    )

  print(p2)

  return(invisible(NULL))
}


# =============================================================================
# Generic Dashboard Functions (ggplot2 Graphics)
# =============================================================================

#' Generic complete dashboard for mixture models (ggplot2)
#'
#' @description Creates a comprehensive multi-page diagnostic dashboard
#'   for any mixture model type using ggplot2, automatically adapting to
#'   the model's polynomial order.
#'
#' @param x An object inheriting from "pdm_mcmc".
#' @param ask Logical; if TRUE, prompts user before each new page.
#' @param ci Logical; whether to display credible intervals.
#' @param ci_level Numeric between 0 and 1; credible interval level.
#' @param ... Additional arguments passed to plotting functions.
#'
#' @return NULL (invisibly). Function is called for side effects (plotting).
#'
#' @details Generates a complete dashboard with ggplot2 graphics.
#'   The number of pages adapts automatically based on model order.
#'
#'   Requires ggplot2 package. Optionally uses patchwork for layouts.
#'
#' @keywords internal
#' @noRd
plot_all_mixture_generic_ggplot <- function(x, ask = TRUE, ci = TRUE,
                                            ci_level = 0.95, ...) {

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for ggplot2 engine")
  }

  if (ask) {
    message("Press [Enter] to see next plot...")
  }

  # 1. Detect model type
  model_info <- detect_model_type(x)
  model_order <- model_info$model_order

  # 2. Get parameter configuration
  param_config <- get_param_config(x, model_info$model_class, model_order)
  n_params <- length(param_config)

  # 3. Pages 1-n: MCMC diagnostics
  plot_mcmc_diagnostics_generic(x, which = seq_len(n_params),
                                param_config = param_config,
                                engine = "ggplot2", ...)
  if (ask) readline()

  # 4. Page n+1: Mixture parameters (only for mixture models)
  if (model_info$has_mixture) {
    plot_mixture_params_ggplot(x$mu_1, x$mu_2, x$prec_1, x$prec_2, ...)
    if (ask) readline()
  }

  # 5. Pages n+2 onwards: Dynamic states
  plot_dynamic_states_generic_ggplot(x, model_order = model_order,
                                     ci = ci, ci_level = ci_level, ...)
  if (ask) readline()

  # 6. Final pages: Mixture weights (only for mixture models)
  if (model_info$has_mixture) {
    plot_mixture_weights_ggplot(x$alpha, x$z, ci = ci, ci_level = ci_level, ...)
  }

  invisible(NULL)
}


#' Generic dynamic states plotting dispatcher (ggplot2)
#'
#' @description Orchestrates dynamic state plotting for any model order
#'   using ggplot2, creating appropriate pages for trajectories, innovations,
#'   and diagnostics.
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
#'     \item Order 1: 2 pages (trajectory + diagnostics with innovations only)
#'     \item Order 2: 2 pages (trajectories + diagnostics with scatter plot)
#'     \item Order 3: 3 pages (trajectories + diagnostics + state space)
#'   }
#'
#' @keywords internal
#' @noRd
plot_dynamic_states_generic_ggplot <- function(x, which = NULL,
                                               model_order = NULL,
                                               ci = TRUE, ci_level = 0.95, ...) {

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for ggplot2 engine")
  }

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

  n_obs <- attr(x, "n_obs")
  time_grid <- seq_len(n_obs)
  patchwork_available <- requireNamespace("patchwork", quietly = TRUE)

  # Prepare CI parameters
  if (ci) {
    ci_lower_prob <- (1 - ci_level) / 2
    ci_upper_prob <- 1 - ci_lower_prob
    ci_pct <- round(ci_level * 100)
  }

  # Get state summaries and labels
  state_summaries <- summarise_all_states(x, model_order, ci, ci_level)
  state_labels <- get_state_labels(model_order)

  # Base theme
  base_theme <- ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(
      panel.grid.major = ggplot2::element_line(color = "grey85"),
      panel.grid.minor = ggplot2::element_line(color = "grey92"),
      plot.title = ggplot2::element_text(face = "bold", size = 13)
    )

  legend_outside <- ggplot2::theme(
    legend.position = "top",
    legend.title = ggplot2::element_blank(),
    legend.direction = "horizontal"
  )

  # =========================================================================
  # Page 1: State Trajectories
  # =========================================================================

  if (1 %in% which) {
    plots <- list()

    for (i in seq_len(model_order)) {
      state_name <- state_labels$state_names[i]
      summary_i <- state_summaries[[state_name]]

      df_state <- data.frame(
        time = time_grid,
        median = summary_i$median
      )

      if (ci) {
        df_state$lower <- summary_i$lower
        df_state$upper <- summary_i$upper
      }

      p <- ggplot2::ggplot(df_state, ggplot2::aes(x = .data$time))

      if (ci) {
        p <- p +
          ggplot2::geom_ribbon(
            ggplot2::aes(ymin = .data$lower, ymax = .data$upper, fill = "CI"),
            alpha = 0.5, colour = NA
          )
      }

      p <- p +
        ggplot2::geom_line(
          ggplot2::aes(y = .data$median, colour = "Median"),
          linewidth = 1.2
        ) +
        ggplot2::scale_color_manual(
          values = c("Median" = state_labels$state_colors[i]),
          breaks = "Median",
          labels = state_labels$state_titles[[i]]
        )

      if (ci) {
        p <- p +
          ggplot2::scale_fill_manual(
            values = c("CI" = "gray70"),
            breaks = "CI",
            labels = paste0(ci_pct, "% CI")
          ) +
          ggplot2::guides(
            colour = ggplot2::guide_legend(order = 1),
            fill = ggplot2::guide_legend(order = 2)
          )
      } else {
        p <- p +
          ggplot2::guides(colour = ggplot2::guide_legend(order = 1))
      }

      p <- p +
        ggplot2::labs(
          title = state_labels$state_titles[i],
          x = "Time",
          y = "State Value"
        ) +
        base_theme +
        legend_outside

      plots[[i]] <- p
    }

    if (patchwork_available) {
      combined <- patchwork::wrap_plots(plots, ncol = 1) +
        patchwork::plot_annotation(
          title = "Dynamic State Trajectories",
          theme = ggplot2::theme(
            plot.title = ggplot2::element_text(size = 14, face = "bold",
                                               hjust = 0.5)
          )
        )
      print(combined)
    } else {
      plots[[1]] <- plots[[1]] +
        ggplot2::labs(subtitle = "Dynamic State Trajectories") +
        ggplot2::theme(
          plot.subtitle = ggplot2::element_text(size = 14, face = "bold",
                                                hjust = 0.5)
        )
      for (p in plots) {
        print(p)
      }
    }
  }

  # =========================================================================
  # Page 2: Diagnostics - Different layouts for each order
  # =========================================================================

  if (2 %in% which) {
    # Compute innovations
    innovations <- compute_innovations(x, model_order)
    innov_summaries <- lapply(innovations, summarise_state, ci, ci_level)
    innov_labels <- get_innovation_labels(model_order)

    plots <- list()

    # Helper function to create innovation plot
    create_innov_plot <- function(innov_summary, innov_idx) {
      df_innov <- data.frame(
        time = time_grid,
        median = innov_summary$median
      )

      if (ci) {
        df_innov$lower <- innov_summary$lower
        df_innov$upper <- innov_summary$upper
      }

      p <- ggplot2::ggplot(df_innov, ggplot2::aes(x = .data$time))

      if (ci && "lower" %in% names(df_innov)) {
        p <- p +
          ggplot2::geom_ribbon(
            ggplot2::aes(ymin = .data$lower, ymax = .data$upper, fill = "CI"),
            alpha = 0.4, colour = NA
          )
      }

      p <- p +
        ggplot2::geom_segment(
          ggplot2::aes(xend = .data$time, y = 0, yend = .data$median,
                       colour = "Median"),
          linewidth = 1.05
        ) +
        ggplot2::geom_hline(yintercept = 0, colour = "red",
                            linetype = "dashed", linewidth = 0.8) +
        ggplot2::scale_color_manual(
          values = c("Median" = innov_labels$innov_colors[innov_idx]),
          breaks = "Median",
          labels = "Median"
        )

      if (ci && "lower" %in% names(df_innov)) {
        p <- p +
          ggplot2::scale_fill_manual(
            values = c("CI" = "gray70"),
            breaks = "CI",
            labels = paste0(ci_pct, "% CI")
          ) +
          ggplot2::guides(
            colour = ggplot2::guide_legend(order = 1),
            fill = ggplot2::guide_legend(order = 2)
          )
      } else {
        p <- p +
          ggplot2::guides(colour = ggplot2::guide_legend(order = 1))
      }

      p <- p +
        ggplot2::labs(
          title = innov_labels$innov_titles[innov_idx],
          x = "Time",
          y = innov_labels$innov_labels[[innov_idx]]
        ) +
        base_theme +
        legend_outside

      return(p)
    }

    # -----------------------------------------------------------------------
    # ORDER 1: [Innovation only] - single panel
    # -----------------------------------------------------------------------
    if (model_order == 1) {
      plots[[1]] <- create_innov_plot(innov_summaries$innov_1, 1)

      # Print with single-column layout
      if (patchwork_available) {
        combined <- patchwork::wrap_plots(plots, ncol = 1) +
          patchwork::plot_annotation(
            title = "Dynamic State Diagnostics",
            theme = ggplot2::theme(
              plot.title = ggplot2::element_text(size = 14, face = "bold",
                                                 hjust = 0.5)
            )
          )
        print(combined)
      } else {
        plots[[1]] <- plots[[1]] +
          ggplot2::labs(subtitle = "Dynamic State Diagnostics") +
          ggplot2::theme(
            plot.subtitle = ggplot2::element_text(size = 14, face = "bold",
                                                  hjust = 0.5)
          )
        print(plots[[1]])
      }

      # Return early to skip the general layout logic below
      return(invisible(NULL))
    }

    # -----------------------------------------------------------------------
    # ORDER 2: [Scatter, Innov1, Joint, Innov2]
    # -----------------------------------------------------------------------
    if (model_order == 2) {

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

      p_scatter <- ggplot2::ggplot(theta_df,
                                   ggplot2::aes(x = .data$theta1, y = .data$theta2)) +
        ggplot2::geom_point(colour = grDevices::rgb(1, 0.5, 0, 0.2),
                            size = 1.2) +
        ggplot2::labs(
          title = "State Space",
          x = expression(theta["t,1"]),
          y = expression(theta["t,2"])
        ) +
        base_theme +
        ggplot2::theme(legend.position = "none")

      plots[[1]] <- p_scatter
      plots[[2]] <- create_innov_plot(innov_summaries$innov_1, 1)

      # Plot 2.3: Joint Trajectories
      df_joint <- data.frame(
        time = rep(time_grid, 2),
        median = c(state_summaries$theta_1$median,
                   state_summaries$theta_2$median),
        state = factor(rep(c("theta_1", "theta_2"), each = n_obs),
                       levels = c("theta_1", "theta_2"))
      )

      p_joint <- ggplot2::ggplot(df_joint, ggplot2::aes(x = .data$time,
                                                        y = .data$median,
                                                        colour = .data$state)) +
        ggplot2::geom_line(linewidth = 1.2) +
        ggplot2::scale_colour_manual(
          values = c("theta_1" = state_labels$state_colors[1],
                     "theta_2" = state_labels$state_colors[2]),
          breaks = c("theta_1", "theta_2"),
          labels = state_labels$state_labels[1:2]
        ) +
        ggplot2::guides(colour = ggplot2::guide_legend(order = 1)) +
        ggplot2::labs(
          title = "Joint Trajectories",
          x = "Time",
          y = "State Value"
        ) +
        base_theme +
        legend_outside

      plots[[3]] <- p_joint
      plots[[4]] <- create_innov_plot(innov_summaries$innov_2, 2)
    }

    # -----------------------------------------------------------------------
    # ORDER 3: [Joint, Innov1, Innov2, Innov3]
    # -----------------------------------------------------------------------
    if (model_order == 3) {

      # Plot 2.1: Joint Trajectories
      df_joint <- data.frame(
        time = rep(time_grid, 3),
        median = c(state_summaries$theta_1$median,
                   state_summaries$theta_2$median,
                   state_summaries$theta_3$median),
        state = factor(rep(c("theta_1", "theta_2", "theta_3"), each = n_obs),
                       levels = c("theta_1", "theta_2", "theta_3"))
      )

      p_joint <- ggplot2::ggplot(df_joint, ggplot2::aes(x = .data$time,
                                                        y = .data$median,
                                                        colour = .data$state)) +
        ggplot2::geom_line(linewidth = 1.2) +
        ggplot2::scale_colour_manual(
          values = setNames(state_labels$state_colors,
                            c("theta_1", "theta_2", "theta_3")),
          breaks = c("theta_1", "theta_2", "theta_3"),
          labels = state_labels$state_labels
        ) +
        ggplot2::guides(colour = ggplot2::guide_legend(order = 1)) +
        ggplot2::labs(
          title = "Joint Trajectories",
          x = "Time",
          y = "State Value"
        ) +
        base_theme +
        legend_outside

      plots[[1]] <- p_joint
      plots[[2]] <- create_innov_plot(innov_summaries$innov_1, 1)
      plots[[3]] <- create_innov_plot(innov_summaries$innov_2, 2)
      plots[[4]] <- create_innov_plot(innov_summaries$innov_3, 3)
    }

    # Layout and print (for order 2 and 3 only)
    if (model_order >= 2) {
      if (patchwork_available) {
        combined <- patchwork::wrap_plots(plots, ncol = 2, byrow = TRUE) +
          patchwork::plot_annotation(
            title = "Dynamic State Diagnostics",
            theme = ggplot2::theme(
              plot.title = ggplot2::element_text(size = 14, face = "bold",
                                                 hjust = 0.5)
            )
          )
        print(combined)
      } else {
        plots[[1]] <- plots[[1]] +
          ggplot2::labs(subtitle = "Dynamic State Diagnostics") +
          ggplot2::theme(
            plot.subtitle = ggplot2::element_text(size = 14, face = "bold",
                                                  hjust = 0.5)
          )
        for (p in plots) {
          print(p)
        }
      }
    }
  }

  # =========================================================================
  # Page 3: State Space Relationships (order 3 only)
  # =========================================================================

  if (3 %in% which && model_order == 3) {
    theta_df <- data.frame(
      theta1 = as.vector(x$theta_1),
      theta2 = as.vector(x$theta_2),
      theta3 = as.vector(x$theta_3)
    )

    max_points <- 5000L
    if (nrow(theta_df) > max_points) {
      set.seed(123)
      theta_df <- theta_df[sample.int(nrow(theta_df), max_points), ]
    }

    p12 <- ggplot2::ggplot(theta_df,
                           ggplot2::aes(x = .data$theta1, y = .data$theta2)) +
      ggplot2::geom_point(colour = grDevices::rgb(0.2, 0.5, 0.8, 0.2),
                          size = 1.2) +
      ggplot2::labs(
        x = expression(theta["t,1"]),
        y = expression(theta["t,2"])
      ) +
      base_theme +
      ggplot2::theme(legend.position = "none")

    p13 <- ggplot2::ggplot(theta_df,
                           ggplot2::aes(x = .data$theta1, y = .data$theta3)) +
      ggplot2::geom_point(colour = grDevices::rgb(0.8, 0.3, 0.3, 0.2),
                          size = 1.2) +
      ggplot2::labs(
        x = expression(theta["t,1"]),
        y = expression(theta["t,3"])
      ) +
      base_theme +
      ggplot2::theme(legend.position = "none")

    p23 <- ggplot2::ggplot(theta_df,
                           ggplot2::aes(x = .data$theta2, y = .data$theta3)) +
      ggplot2::geom_point(colour = grDevices::rgb(0.3, 0.8, 0.4, 0.2),
                          size = 1.2) +
      ggplot2::labs(
        x = expression(theta["t,2"]),
        y = expression(theta["t,3"])
      ) +
      base_theme +
      ggplot2::theme(legend.position = "none")

    plots <- list(p12, p13, p23)

    if (patchwork_available) {
      combined <- patchwork::wrap_plots(plots, ncol = 2) +
        patchwork::plot_annotation(
          title = "Pairwise State Relationships",
          theme = ggplot2::theme(
            plot.title = ggplot2::element_text(size = 14, face = "bold",
                                               hjust = 0.5)
          )
        )
      print(combined)
    } else {
      plots[[1]] <- plots[[1]] +
        ggplot2::labs(subtitle = "Pairwise State Relationships") +
        ggplot2::theme(
          plot.subtitle = ggplot2::element_text(size = 14, face = "bold",
                                                hjust = 0.5)
        )
      for (p in plots) {
        print(p)
      }
    }
  }

  invisible(NULL)
}
