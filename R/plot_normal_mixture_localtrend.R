#' Plot method for normal_mixture_localtrend objects
#'
#' @description Produces diagnostic plots for MCMC output from Gaussian mixture
#'   models with dynamic mixture weights.
#'
#' @param x An object of class \code{normal_mixture_localtrend}.
#' @param type Character string specifying the type of plot. One of:
#'   \describe{
#'     \item{\code{"all"}}{Complete dashboard with all diagnostic plots (default)}
#'     \item{\code{"mcmc"}}{MCMC convergence diagnostics (trace plots, ACF, running means)}
#'     \item{\code{"params"}}{Mixture component parameters (mu, phi)}
#'     \item{\code{"states"}}{Dynamic states (theta_1, theta_2 trajectories)}
#'     \item{\code{"alpha"}}{Mixture weights over time (alpha_t and z_t)}
#'   }
#' @param which Integer vector specifying which diagnostic plots to display.
#'   For \code{type = "mcmc"}:
#'   \describe{
#'     \item{1}{mu_1 (component 1 mean)}
#'     \item{2}{mu_2 (component 2 mean)}
#'     \item{3}{phi_1 (component 1 precision)}
#'     \item{4}{phi_2 (component 2 precision)}
#'     \item{5}{theta_01 (initial level)}
#'     \item{6}{theta_02 (initial trend)}
#'     \item{7}{W_1^{-1} (level innovation precision)}
#'     \item{8}{W_2^{-1} (trend innovation precision)}
#'   }
#'   For \code{type = "params"}, \code{type = "states"}: indices of subplots.
#'   For \code{type = "alpha"}: not used (both alpha_t and z_t are shown).
#'   If \code{NULL} (default), all available plots are shown.
#' @param engine Character string specifying the graphics engine. One of:
#'   \describe{
#'     \item{\code{"base"}}{Base R graphics (default, no dependencies)}
#'     \item{\code{"ggplot2"}}{ggplot2 graphics (requires \pkg{ggplot2})}
#'   }
#' @param ask Logical; if \code{TRUE}, the user is asked before each plot when
#'   \code{type = "all"}. Default is \code{interactive()} when \code{type = "all"},
#'   \code{FALSE} otherwise.
#' @param overlay_data Logical; for \code{type = "alpha"}, whether to overlay
#'   the original data (if available). Default is \code{TRUE}.
#' @param ... Additional arguments passed to plotting functions.
#'
#' @return Invisibly returns the input object \code{x}.
#'
#' @details
#' This function provides comprehensive visual diagnostics for Bayesian MCMC output:
#'
#' \strong{MCMC Diagnostics (\code{type = "mcmc"}):}
#'
#' Each parameter gets a dedicated page with 4 panels:
#' \itemize{
#'   \item \strong{Trace Plot:} Shows MCMC chain trajectory to assess mixing
#'   \item \strong{Autocorrelation:} ACF plot to detect serial correlation
#'   \item \strong{Posterior Density:} Marginal posterior distribution
#'   \item \strong{Running Mean:} Cumulative mean to assess convergence
#' }
#'
#' Available parameters: mu_1, mu_2, phi_1, phi_2, theta_01, theta_02, W_1^{-1}, W_2^{-1}
#'
#' \strong{Mixture Parameters (\code{type = "params"}):}
#' \itemize{
#'   \item Joint posterior of component means (mu_1 vs mu_2)
#'   \item Marginal posteriors for means and precisions
#'   \item Component separation diagnostics
#' }
#'
#' \strong{Dynamic States (\code{type = "states"}):}
#' \itemize{
#'   \item Time-varying state trajectories with credible bands
#'   \item Innovation sequences
#'   \item State space representations
#' }
#'
#' \strong{Mixture Weights (\code{type = "alpha"}):}
#' \itemize{
#'   \item Page 1: alpha_t trajectory with credible bands and data overlay
#'   \item Page 2: Posterior probabilities of component membership (z_t)
#' }
#'
#' \strong{Complete Dashboard (\code{type = "all"}):}
#'
#' Generates 12 pages in total:
#' \itemize{
#'   \item Pages 1-8: Individual parameter diagnostics (4 panels each)
#'   \item Page 9: Mixture parameters overview
#'   \item Page 10: Dynamic states overview
#'   \item Page 11: Mixture weight alpha_t
#'   \item Page 12: Component membership z_t
#' }
#'
#' The \code{engine} argument allows choosing between base R graphics (lightweight,
#' no dependencies) and ggplot2 (modern, publication-ready). If ggplot2 is not
#' installed and \code{engine = "ggplot2"}, the function falls back to base graphics
#' with a warning.
#'
#' @section Dependencies:
#'
#' The ggplot2 engine has optional dependencies for enhanced visualizations:
#' \itemize{
#'   \item \pkg{hexbin}: For hexagonal binning in joint posterior plots
#'   \item \pkg{patchwork}: For combining multiple plots into layouts
#' }
#'
#' If these packages are not installed, the function will use fallback methods
#' (e.g., scatterplots instead of hexbins). Install with:
#' \code{install.packages(c("hexbin", "patchwork"))}
#'
#' @examples
#' \dontrun{
#' # Fit model
#' out <- mcmc_normal_mixture_localtrend(y, link = "logit", ...)
#'
#' # Complete dashboard (12 pages)
#' plot(out, type = "all", engine = "base")
#'
#' # Diagnostics for specific parameters
#' plot(out, type = "mcmc", which = 1:2)  # Only mu_1 and mu_2
#' plot(out, type = "mcmc", which = 3:4)  # Only phi_1 and phi_2
#'
#' # Mixture weights (2 pages: alpha_t and z_t)
#' plot(out, type = "alpha")
#'
#' # Save to multi-page PDF
#' pdf("diagnostics.pdf", width = 10, height = 8)
#' plot(out, type = "all", ask = FALSE)
#' dev.off()
#'
#' # Use ggplot2 engine
#' plot(out, type = "mcmc", which = 1, engine = "ggplot2")
#' }
#'
#' @seealso \code{\link{mcmc_normal_mixture_localtrend}},
#'   \code{\link{summary.normal_mixture_localtrend}}
#'
#' @export
plot.normal_mixture_localtrend <- function(x,
                                           type = c("all", "mcmc", "params",
                                                    "states", "alpha"),
                                           which = NULL,
                                           engine = c("base", "ggplot2"),
                                           ask = NULL,
                                           overlay_data = TRUE,
                                           ...) {

  # Validate inputs
  type <- match.arg(type)
  engine <- match.arg(engine)

  # Check ggplot2 availability
  if (engine == "ggplot2" && !requireNamespace("ggplot2", quietly = TRUE)) {
    warning("Package 'ggplot2' is not installed. Falling back to base graphics.")
    engine <- "base"
  }

  # Set default for ask
  if (is.null(ask)) {
    ask <- interactive() && type == "all"
  }

  # Dispatch to appropriate plotting function
  if (engine == "base") {
    switch(type,
           all = plot_all_base(x, ask = ask, overlay_data = overlay_data, ...),
           mcmc = plot_mcmc_diagnostics_base(x, which = which, ...),
           params = plot_mixture_params_base(x, which = which, ...),
           states = plot_dynamic_states_base(x, which = which, ...),
           alpha = plot_mixture_weights_base(x, overlay_data = overlay_data, ...)
    )
  } else {
    switch(type,
           all = plot_all_ggplot(x, ask = ask, overlay_data = overlay_data, ...),
           mcmc = plot_mcmc_diagnostics_ggplot(x, which = which, ...),
           params = plot_mixture_params_ggplot(x, which = which, ...),
           states = plot_dynamic_states_ggplot(x, which = which, ...),
           alpha = plot_mixture_weights_ggplot(x, overlay_data = overlay_data, ...)
    )
  }

  invisible(x)
}


# ============================================================================
# Base R Graphics Functions
# ============================================================================

#' Plot 4-panel diagnostics for a single parameter
#' @keywords internal
#' @noRd
plot_param_diagnostics <- function(param_samples,
                                   param_name,
                                   param_label,
                                   true_value = NULL,
                                   ...) {

  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  par(mfrow = c(2, 2), mar = c(4, 4, 3, 1), oma = c(0, 0, 3, 0))

  # 1. Trace Plot
  plot(param_samples, type = "l", col = "gray40", lwd = 0.8,
       xlab = "Iteration", ylab = param_label,
       main = "Trace Plot")
  abline(h = median(param_samples), col = "red", lwd = 2, lty = 2)
  if (!is.null(true_value)) {
    abline(h = true_value, col = "blue", lwd = 2, lty = 3)
  }
  grid()

  # 2. Autocorrelation Function
  acf(param_samples, main = "Autocorrelation",
      col = "steelblue", lwd = 2)

  # 3. Posterior Density
  dens <- density(param_samples)
  plot(dens, main = "Posterior Density",
       xlab = param_label, lwd = 2, col = "darkgreen")
  polygon(dens, col = rgb(0, 0.5, 0, 0.2), border = NA)
  abline(v = median(param_samples), col = "red", lwd = 2, lty = 2)
  abline(v = mean(param_samples), col = "blue", lwd = 2, lty = 3)
  legend("topright",
         legend = c("Median", "Mean"),
         col = c("red", "blue"),
         lty = c(2, 3), lwd = 2, bty = "n", cex = 0.8)

  # 4. Running Mean (Convergence Check)
  running_mean <- cumsum(param_samples) / seq_along(param_samples)
  plot(running_mean, type = "l", col = "steelblue", lwd = 2,
       xlab = "Iteration", ylab = param_label,
       main = "Running Mean")
  abline(h = median(param_samples), col = "red", lwd = 2, lty = 2)
  if (!is.null(true_value)) {
    abline(h = true_value, col = "blue", lwd = 2, lty = 3)
  }
  grid()

  # Overall title
  mtext(paste("MCMC Diagnostics:", param_name),
        outer = TRUE, cex = 1.3, font = 2)
}


#' Complete dashboard with base R graphics
#' @keywords internal
#' @noRd
plot_all_base <- function(x, ask = TRUE, overlay_data = TRUE, ...) {
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  if (ask) {
    oldask <- par(ask = TRUE)
    on.exit(par(oldask), add = TRUE)
  }

  # Pages 1-8: Individual parameter diagnostics
  plot_mcmc_diagnostics_base(x, which = 1:8, ...)

  # Page 9: Mixture parameters overview
  plot_mixture_params_base(x, ...)

  # Page 10: Dynamic states
  plot_dynamic_states_base(x, ...)

  # Pages 11-12: Mixture weights (alpha_t and z_t)
  plot_mixture_weights_base(x, overlay_data = overlay_data, ...)
}


#' MCMC diagnostics with base R graphics
#' @keywords internal
#' @noRd
plot_mcmc_diagnostics_base <- function(x, which = NULL, ...) {

  # Define available parameters
  all_params <- list(
    mu_1 = list(samples = x$mu_1,
                name = "mu_1",
                label = expression(mu[1])),
    mu_2 = list(samples = x$mu_2,
                name = "mu_2",
                label = expression(mu[2])),
    phi_1 = list(samples = x$prec_1,
                 name = "phi_1",
                 label = expression(phi[1])),
    phi_2 = list(samples = x$prec_2,
                 name = "phi_2",
                 label = expression(phi[2])),
    theta_01 = list(samples = x$theta_01,
                    name = "theta_01",
                    label = expression(theta["0,1"])),
    theta_02 = list(samples = x$theta_02,
                    name = "theta_02",
                    label = expression(theta["0,2"])),
    W1_inv = list(samples = x$prec_theta1,
                  name = "W_1^{-1}",
                  label = expression(W[1]^{-1})),
    W2_inv = list(samples = x$prec_theta2,
                  name = "W_2^{-1}",
                  label = expression(W[2]^{-1}))
  )

  # Default: all parameters
  if (is.null(which)) {
    which <- seq_along(all_params)
  }

  # Validate which
  if (any(which < 1) || any(which > length(all_params))) {
    stop(sprintf("`which` must be between 1 and %d", length(all_params)))
  }

  # Plot each selected parameter on its own page
  for (i in which) {
    param_info <- all_params[[i]]
    plot_param_diagnostics(
      param_samples = param_info$samples,
      param_name = param_info$name,
      param_label = param_info$label,
      ...
    )
  }
}


#' Mixture parameters with base R graphics
#' @keywords internal
#' @noRd
plot_mixture_params_base <- function(x, which = NULL, ...) {

  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  if (is.null(which)) which <- 1:4

  par(mfrow = c(2, 2), mar = c(4, 4, 2, 1), oma = c(0, 0, 2, 0))

  # Plot 1: Joint posterior mu_1 vs mu_2
  if (1 %in% which) {
    # Scatterplot with transparency (no dependencies)
    plot(x$mu_1, x$mu_2,
         xlab = expression(mu[1]),
         ylab = expression(mu[2]),
         main = "Joint Posterior",
         pch = 16, cex = 0.5, col = rgb(0.2, 0.5, 0.8, 0.15))

    # Add reference line y = x
    abline(a = 0, b = 1, col = "red", lwd = 2, lty = 2)
    grid()
  }

  # Plot 2: Marginal density mu_1
  if (2 %in% which) {
    dens <- density(x$mu_1)
    plot(dens, main = expression(paste("Posterior: ", mu[1])),
         xlab = expression(mu[1]), lwd = 2, col = "steelblue")
    polygon(dens, col = rgb(0.2, 0.5, 0.8, 0.3), border = NA)
    abline(v = median(x$mu_1), col = "red", lwd = 2, lty = 2)
    abline(v = mean(x$mu_1), col = "blue", lwd = 2, lty = 3)
    legend("topright", legend = c("Median", "Mean"),
           col = c("red", "blue"), lty = c(2, 3), lwd = 2, bty = "n")
  }

  # Plot 3: Marginal density mu_2
  if (3 %in% which) {
    dens <- density(x$mu_2)
    plot(dens, main = expression(paste("Posterior: ", mu[2])),
         xlab = expression(mu[2]), lwd = 2, col = "darkgreen")
    polygon(dens, col = rgb(0, 0.5, 0, 0.3), border = NA)
    abline(v = median(x$mu_2), col = "red", lwd = 2, lty = 2)
    abline(v = mean(x$mu_2), col = "blue", lwd = 2, lty = 3)
  }

  # Plot 4: Precision comparison
  if (4 %in% which) {
    xlim_range <- range(c(x$prec_1, x$prec_2))
    dens1 <- density(x$prec_1)
    dens2 <- density(x$prec_2)

    plot(dens1, main = "Component Precisions",
         xlab = expression(phi), lwd = 2, col = "steelblue",
         xlim = xlim_range,
         ylim = range(c(dens1$y, dens2$y)))
    polygon(dens1, col = rgb(0.2, 0.5, 0.8, 0.3), border = NA)
    lines(dens2, lwd = 2, col = "darkgreen")
    polygon(dens2, col = rgb(0, 0.5, 0, 0.3), border = NA)
    legend("topright",
           legend = c(expression(phi[1]), expression(phi[2])),
           col = c("steelblue", "darkgreen"),
           lwd = 2, bty = "n")
  }

  mtext("Mixture Component Parameters", outer = TRUE, cex = 1.3, font = 2)
}


#' Dynamic states with base R graphics
#' @keywords internal
#' @noRd
plot_dynamic_states_base <- function(x, which = NULL, ...) {

  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  if (is.null(which)) which <- 1:4

  par(mfrow = c(2, 2), mar = c(4, 4, 2, 1), oma = c(0, 0, 2, 0))

  n_obs <- attr(x, "n_obs")
  time_grid <- seq_len(n_obs)

  # Compute credible bands
  theta1_median <- apply(x$theta_1, 2, median)
  theta1_q025 <- apply(x$theta_1, 2, quantile, probs = 0.025)
  theta1_q975 <- apply(x$theta_1, 2, quantile, probs = 0.975)

  theta2_median <- apply(x$theta_2, 2, median)
  theta2_q025 <- apply(x$theta_2, 2, quantile, probs = 0.025)
  theta2_q975 <- apply(x$theta_2, 2, quantile, probs = 0.975)

  # Plot 1: theta_1 trajectory
  if (1 %in% which) {
    plot(time_grid, theta1_median, type = "l", lwd = 2,
         xlab = "Time", ylab = expression(theta["t,1"]),
         main = expression(paste("Level State: ", theta["t,1"])),
         ylim = range(c(theta1_q025, theta1_q975)))

    polygon(c(time_grid, rev(time_grid)),
            c(theta1_q025, rev(theta1_q975)),
            col = rgb(0.7, 0.7, 0.7, 0.5), border = NA)

    lines(time_grid, theta1_median, lwd = 2, col = "black")
    grid()
  }

  # Plot 2: theta_2 trajectory
  if (2 %in% which) {
    plot(time_grid, theta2_median, type = "l", lwd = 2,
         xlab = "Time", ylab = expression(theta["t,2"]),
         main = expression(paste("Trend State: ", theta["t,2"])),
         ylim = range(c(theta2_q025, theta2_q975)))

    polygon(c(time_grid, rev(time_grid)),
            c(theta2_q025, rev(theta2_q975)),
            col = rgb(0.7, 0.7, 0.7, 0.5), border = NA)

    lines(time_grid, theta2_median, lwd = 2, col = "black")
    grid()
  }

  # Plot 3: Innovations (theta_1 differences)
  if (3 %in% which) {
    innovations <- t(apply(x$theta_1, 1, diff))
    innov_median <- apply(innovations, 2, median)
    innov_q025 <- apply(innovations, 2, quantile, probs = 0.025)
    innov_q975 <- apply(innovations, 2, quantile, probs = 0.975)

    plot(time_grid[-1], innov_median, type = "h", lwd = 2, col = "steelblue",
         xlab = "Time", ylab = expression(Delta*theta["t,1"]),
         main = "Level Innovations",
         ylim = range(c(innov_q025, innov_q975)))
    abline(h = 0, col = "red", lty = 2, lwd = 2)
    grid()
  }

  # Plot 4: State space
  if (4 %in% which) {
    theta1_vec <- as.vector(x$theta_1)
    theta2_vec <- as.vector(x$theta_2)

    # Sample if too many points (for performance)
    max_points <- 5000
    if (length(theta1_vec) > max_points) {
      idx <- sample(length(theta1_vec), max_points)
      theta1_vec <- theta1_vec[idx]
      theta2_vec <- theta2_vec[idx]
    }

    plot(theta1_vec, theta2_vec,
         xlab = expression(theta["t,1"]),
         ylab = expression(theta["t,2"]),
         main = "State Space",
         pch = 16, cex = 0.4, col = rgb(1, 0.5, 0, 0.2))
    grid()
  }

  mtext("Dynamic State Trajectories", outer = TRUE, cex = 1.3, font = 2)
}


#' Mixture weights with base R graphics (2 pages)
#' @keywords internal
#' @noRd
plot_mixture_weights_base <- function(x, overlay_data = TRUE, ...) {

  n_obs <- attr(x, "n_obs")
  time_grid <- seq_len(n_obs)

  # Compute credible bands for alpha
  alpha_median <- apply(x$alpha, 2, median)
  alpha_q025 <- apply(x$alpha, 2, quantile, probs = 0.025)
  alpha_q975 <- apply(x$alpha, 2, quantile, probs = 0.975)

  # =========================================================================
  # Page 1: alpha_t trajectory
  # =========================================================================

  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar), add = TRUE)

  par(mfrow = c(1, 1), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))

  plot(time_grid, alpha_median, type = "n", lwd = 2,
       xlab = "Time", ylab = expression(alpha[t]),
       main = "Mixture Weights Over Time",
       ylim = c(0, 1))

  # Credible band
  polygon(c(time_grid, rev(time_grid)),
          c(alpha_q025, rev(alpha_q975)),
          col = rgb(0.2, 0.5, 0.8, 0.3), border = NA)

  # Median line
  lines(time_grid, alpha_median, lwd = 2.5, col = "blue")

  # Overlay data if available
  if (overlay_data && !is.null(attr(x, "y"))) {
    y <- attr(x, "y")
    y_scaled <- (y - min(y)) / (max(y) - min(y))
    lines(time_grid, y_scaled, col = "red", lwd = 1.5, lty = 2)
    legend("topright",
           legend = c(expression(alpha[t]), "Data (scaled)", "95% CI"),
           col = c("blue", "red", rgb(0.2, 0.5, 0.8, 0.3)),
           lty = c(1, 2, 1), lwd = c(2.5, 1.5, 10),
           bty = "n")
  } else {
    legend("topright",
           legend = c(expression(alpha[t]), "95% CI"),
           col = c("blue", rgb(0.2, 0.5, 0.8, 0.3)),
           lty = c(1, 1), lwd = c(2.5, 10),
           bty = "n")
  }

  grid()
  mtext(expression(paste("Time-Varying Mixture Weight: ", alpha[t])),
        outer = TRUE, cex = 1.3, font = 2)

  # =========================================================================
  # Page 2: z_t posterior probabilities
  # =========================================================================

  par(mfrow = c(1, 1), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))

  z_prob <- apply(x$z, 2, mean)

  # Create barplot
  barplot(z_prob,
          col = ifelse(z_prob > 0.5, "purple", "gray70"),
          border = NA,
          ylim = c(0, 1),
          xlab = "Time",
          ylab = expression(P(z[t] == 1)),
          main = "Posterior Probability of Component 2",
          space = 0)

  # Add threshold line
  abline(h = 0.5, col = "red", lwd = 2, lty = 2)

  legend("topright",
         legend = c("P(z = 1) > 0.5", "P(z = 1) ≤ 0.5", "Threshold"),
         fill = c("purple", "gray70", NA),
         border = c(NA, NA, NA),
         lty = c(NA, NA, 2),
         lwd = c(NA, NA, 2),
         col = c(NA, NA, "red"),
         bty = "n")

  grid(nx = NA, ny = NULL)  # Only horizontal lines

  mtext(expression(paste("Component Membership: ", z[t])),
        outer = TRUE, cex = 1.3, font = 2)
}


# ============================================================================
# ggplot2 Graphics Functions
# ============================================================================

#' Plot 4-panel diagnostics for a single parameter (ggplot2)
#' @keywords internal
#' @noRd
plot_param_diagnostics_ggplot <- function(param_samples,
                                          param_name,
                                          param_label_text,
                                          ...) {

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required")
  }

  # Prepare data
  df <- data.frame(
    iteration = seq_along(param_samples),
    value = param_samples,
    running_mean = cumsum(param_samples) / seq_along(param_samples)
  )

  # 1. Trace plot
  p1 <- ggplot2::ggplot(df, ggplot2::aes(x = iteration, y = value)) +
    ggplot2::geom_line(color = "gray40", linewidth = 0.5) +
    ggplot2::geom_hline(yintercept = median(param_samples),
                        color = "red", linetype = "dashed", linewidth = 1) +
    ggplot2::labs(title = "Trace Plot", x = "Iteration", y = param_label_text) +
    ggplot2::theme_minimal()

  # 2. ACF plot
  acf_data <- acf(param_samples, plot = FALSE)
  df_acf <- data.frame(
    lag = acf_data$lag,
    acf = acf_data$acf
  )

  ci_line <- 1.96 / sqrt(length(param_samples))

  p2 <- ggplot2::ggplot(df_acf, ggplot2::aes(x = lag, y = acf)) +
    ggplot2::geom_hline(yintercept = 0, color = "black") +
    ggplot2::geom_segment(ggplot2::aes(xend = lag, yend = 0),
                          color = "steelblue", linewidth = 1) +
    ggplot2::geom_hline(yintercept = c(-ci_line, ci_line),
                        linetype = "dashed", color = "blue") +
    ggplot2::labs(title = "Autocorrelation", x = "Lag", y = "ACF") +
    ggplot2::theme_minimal()

  # 3. Density plot
  p3 <- ggplot2::ggplot(df, ggplot2::aes(x = value)) +
    ggplot2::geom_density(fill = "darkgreen", alpha = 0.3, linewidth = 1) +
    ggplot2::geom_vline(xintercept = median(param_samples),
                        color = "red", linetype = "dashed", linewidth = 1) +
    ggplot2::geom_vline(xintercept = mean(param_samples),
                        color = "blue", linetype = "dotted", linewidth = 1) +
    ggplot2::labs(title = "Posterior Density", x = param_label_text, y = "Density") +
    ggplot2::theme_minimal()

  # 4. Running mean
  p4 <- ggplot2::ggplot(df, ggplot2::aes(x = iteration, y = running_mean)) +
    ggplot2::geom_line(color = "steelblue", linewidth = 1) +
    ggplot2::geom_hline(yintercept = median(param_samples),
                        color = "red", linetype = "dashed", linewidth = 1) +
    ggplot2::labs(title = "Running Mean", x = "Iteration", y = param_label_text) +
    ggplot2::theme_minimal()

  # Combine with patchwork if available
  if (requireNamespace("patchwork", quietly = TRUE)) {
    combined <- (p1 + p2) / (p3 + p4) +
      patchwork::plot_annotation(
        title = paste("MCMC Diagnostics:", param_name),
        theme = ggplot2::theme(plot.title = ggplot2::element_text(size = 16, face = "bold"))
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


#' Complete dashboard with ggplot2 graphics
#' @keywords internal
#' @noRd
plot_all_ggplot <- function(x, ask = TRUE, overlay_data = TRUE, ...) {

  if (ask) {
    message("Press [Enter] to see next plot...")
  }

  # Pages 1-8: Individual parameter diagnostics
  p_mcmc <- plot_mcmc_diagnostics_ggplot(x, which = 1:8, ...)
  if (!is.null(p_mcmc)) print(p_mcmc)
  if (ask) readline()

  # Page 9: Mixture parameters
  p_params <- plot_mixture_params_ggplot(x, ...)
  if (!is.null(p_params)) print(p_params)
  if (ask) readline()

  # Page 10: Dynamic states
  p_states <- plot_dynamic_states_ggplot(x, ...)
  if (!is.null(p_states)) print(p_states)
  if (ask) readline()

  # Pages 11-12: Mixture weights
  p_alpha <- plot_mixture_weights_ggplot(x, overlay_data = overlay_data, ...)
  if (!is.null(p_alpha)) print(p_alpha)
}


#' MCMC diagnostics with ggplot2
#' @keywords internal
#' @noRd
plot_mcmc_diagnostics_ggplot <- function(x, which = NULL, ...) {

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required")
  }

  # Define available parameters
  all_params <- list(
    mu_1 = list(samples = x$mu_1, name = "mu_1", label = "mu_1"),
    mu_2 = list(samples = x$mu_2, name = "mu_2", label = "mu_2"),
    phi_1 = list(samples = x$prec_1, name = "phi_1", label = "phi_1"),
    phi_2 = list(samples = x$prec_2, name = "phi_2", label = "phi_2"),
    theta_01 = list(samples = x$theta_01, name = "theta_01", label = "theta_01"),
    theta_02 = list(samples = x$theta_02, name = "theta_02", label = "theta_02"),
    W1_inv = list(samples = x$prec_theta1, name = "W_1^{-1}", label = "W_1^{-1}"),
    W2_inv = list(samples = x$prec_theta2, name = "W_2^{-1}", label = "W_2^{-1}")
  )

  if (is.null(which)) {
    which <- seq_along(all_params)
  }

  if (any(which < 1) || any(which > length(all_params))) {
    stop(sprintf("`which` must be between 1 and %d", length(all_params)))
  }

  # Plot each selected parameter
  for (i in which) {
    param_info <- all_params[[i]]
    p <- plot_param_diagnostics_ggplot(
      param_samples = param_info$samples,
      param_name = param_info$name,
      param_label_text = param_info$label,
      ...
    )
    if (!is.null(p)) print(p)
  }

  return(invisible(NULL))
}


#' Mixture parameters with ggplot2
#' @keywords internal
#' @noRd
plot_mixture_params_ggplot <- function(x, which = NULL, ...) {

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required")
  }

  if (!requireNamespace("hexbin", quietly = TRUE) && 1 %in% which) {
    message("Note: Install 'hexbin' for improved joint posterior visualization.\n",
            "  Run: install.packages('hexbin')")
  }

  if (is.null(which)) which <- 1:4

  plots <- list()

  # Plot 1: Joint posterior (with hexbin fallback)
  if (1 %in% which) {
    df1 <- data.frame(mu_1 = x$mu_1, mu_2 = x$mu_2)

    # Try hexbin first, fallback to geom_point
    if (requireNamespace("hexbin", quietly = TRUE)) {
      plots[[1]] <- ggplot2::ggplot(df1, ggplot2::aes(x = mu_1, y = mu_2)) +
        ggplot2::geom_hex(bins = 50) +
        ggplot2::scale_fill_gradient(low = "lightblue", high = "darkblue") +
        ggplot2::geom_abline(slope = 1, intercept = 0,
                             color = "red", linetype = "dashed", linewidth = 1) +
        ggplot2::labs(title = "Joint Posterior: Component Means",
                      x = expression(mu[1]), y = expression(mu[2])) +
        ggplot2::theme_minimal() +
        ggplot2::theme(legend.position = "right")
    } else {
      # Fallback: scatterplot with transparency
      plots[[1]] <- ggplot2::ggplot(df1, ggplot2::aes(x = mu_1, y = mu_2)) +
        ggplot2::geom_point(alpha = 0.15, color = "steelblue", size = 1) +
        ggplot2::geom_abline(slope = 1, intercept = 0,
                             color = "red", linetype = "dashed", linewidth = 1) +
        ggplot2::labs(title = "Joint Posterior: Component Means",
                      subtitle = "(Install 'hexbin' for hexagonal binning)",
                      x = expression(mu[1]), y = expression(mu[2])) +
        ggplot2::theme_minimal()
    }
  }

  # Plot 2: Marginal mu_1
  if (2 %in% which) {
    df2 <- data.frame(value = x$mu_1)

    plots[[2]] <- ggplot2::ggplot(df2, ggplot2::aes(x = value)) +
      ggplot2::geom_density(fill = "steelblue", alpha = 0.4, linewidth = 1.2) +
      ggplot2::geom_vline(xintercept = median(x$mu_1),
                          color = "red", linetype = "dashed", linewidth = 1) +
      ggplot2::labs(title = expression(paste("Posterior: ", mu[1])),
                    x = expression(mu[1]), y = "Density") +
      ggplot2::theme_minimal()
  }

  # Plot 3: Marginal mu_2
  if (3 %in% which) {
    df3 <- data.frame(value = x$mu_2)

    plots[[3]] <- ggplot2::ggplot(df3, ggplot2::aes(x = value)) +
      ggplot2::geom_density(fill = "darkgreen", alpha = 0.4, linewidth = 1.2) +
      ggplot2::geom_vline(xintercept = median(x$mu_2),
                          color = "red", linetype = "dashed", linewidth = 1) +
      ggplot2::labs(title = expression(paste("Posterior: ", mu[2])),
                    x = expression(mu[2]), y = "Density") +
      ggplot2::theme_minimal()
  }

  # Plot 4: Precisions comparison
  if (4 %in% which) {
    df4 <- data.frame(
      value = c(x$prec_1, x$prec_2),
      component = rep(c("phi_1", "phi_2"), each = length(x$prec_1))
    )

    plots[[4]] <- ggplot2::ggplot(df4, ggplot2::aes(x = value, fill = component)) +
      ggplot2::geom_density(alpha = 0.5, linewidth = 1) +
      ggplot2::scale_fill_manual(values = c("steelblue", "darkgreen"),
                                 labels = c(expression(phi[1]), expression(phi[2]))) +
      ggplot2::labs(title = "Component Precisions",
                    x = expression(phi), y = "Density") +
      ggplot2::theme_minimal() +
      ggplot2::theme(legend.position = "top")
  }

  # Combine plots
  if (requireNamespace("patchwork", quietly = TRUE) && length(plots) > 1) {
    return(patchwork::wrap_plots(plots, ncol = 2))
  } else if (length(plots) == 1) {
    return(plots[[1]])
  } else {
    # Fallback: print sequentially if patchwork not available
    for (p in plots) print(p)
    return(invisible(NULL))
  }
}


#' Dynamic states with ggplot2
#' @keywords internal
#' @noRd
plot_dynamic_states_ggplot <- function(x, which = NULL, ...) {

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required")
  }

  if (is.null(which)) which <- 1:2

  n_obs <- attr(x, "n_obs")
  time_grid <- seq_len(n_obs)

  plots <- list()

  # Plot 1: theta_1 trajectory
  if (1 %in% which) {
    df1 <- data.frame(
      time = time_grid,
      median = apply(x$theta_1, 2, median),
      lower = apply(x$theta_1, 2, quantile, probs = 0.025),
      upper = apply(x$theta_1, 2, quantile, probs = 0.975)
    )

    plots[[1]] <- ggplot2::ggplot(df1, ggplot2::aes(x = time)) +
      ggplot2::geom_ribbon(ggplot2::aes(ymin = lower, ymax = upper),
                           fill = "gray70", alpha = 0.5) +
      ggplot2::geom_line(ggplot2::aes(y = median), linewidth = 1.2) +
      ggplot2::labs(title = expression(paste("Level State: ", theta["t,1"])),
                    x = "Time", y = expression(theta["t,1"])) +
      ggplot2::theme_minimal()
  }

  # Plot 2: theta_2 trajectory
  if (2 %in% which) {
    df2 <- data.frame(
      time = time_grid,
      median = apply(x$theta_2, 2, median),
      lower = apply(x$theta_2, 2, quantile, probs = 0.025),
      upper = apply(x$theta_2, 2, quantile, probs = 0.975)
    )

    plots[[2]] <- ggplot2::ggplot(df2, ggplot2::aes(x = time)) +
      ggplot2::geom_ribbon(ggplot2::aes(ymin = lower, ymax = upper),
                           fill = "gray70", alpha = 0.5) +
      ggplot2::geom_line(ggplot2::aes(y = median), linewidth = 1.2) +
      ggplot2::labs(title = expression(paste("Trend State: ", theta["t,2"])),
                    x = "Time", y = expression(theta["t,2"])) +
      ggplot2::theme_minimal()
  }

  # Combine plots
  if (requireNamespace("patchwork", quietly = TRUE) && length(plots) > 1) {
    return(patchwork::wrap_plots(plots, ncol = 1))
  } else if (length(plots) == 1) {
    return(plots[[1]])
  } else {
    for (p in plots) print(p)
    return(invisible(NULL))
  }
}


#' Mixture weights with ggplot2 (2 pages)
#' @keywords internal
#' @noRd
plot_mixture_weights_ggplot <- function(x, overlay_data = TRUE, ...) {

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required")
  }

  n_obs <- attr(x, "n_obs")
  time_grid <- seq_len(n_obs)

  # Page 1: alpha_t
  df_alpha <- data.frame(
    time = time_grid,
    median = apply(x$alpha, 2, median),
    lower = apply(x$alpha, 2, quantile, probs = 0.025),
    upper = apply(x$alpha, 2, quantile, probs = 0.975)
  )

  p1 <- ggplot2::ggplot(df_alpha, ggplot2::aes(x = time)) +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = lower, ymax = upper),
                         fill = "steelblue", alpha = 0.3) +
    ggplot2::geom_line(ggplot2::aes(y = median), color = "blue", linewidth = 1.2) +
    ggplot2::labs(title = expression(paste("Mixture Weights: ", alpha[t])),
                  x = "Time", y = expression(alpha[t])) +
    ggplot2::ylim(0, 1) +
    ggplot2::theme_minimal()

  if (overlay_data && !is.null(attr(x, "y"))) {
    y <- attr(x, "y")
    y_scaled <- (y - min(y)) / (max(y) - min(y))
    df_data <- data.frame(time = time_grid, y_scaled = y_scaled)

    p1 <- p1 + ggplot2::geom_line(data = df_data,
                                  ggplot2::aes(y = y_scaled),
                                  color = "red", linetype = "dashed", linewidth = 1)
  }

  print(p1)

  # Page 2: z_t
  z_prob <- apply(x$z, 2, mean)
  df_z <- data.frame(time = time_grid, prob = z_prob)

  p2 <- ggplot2::ggplot(df_z, ggplot2::aes(x = time, y = prob)) +
    ggplot2::geom_col(ggplot2::aes(fill = prob > 0.5), width = 1) +
    ggplot2::scale_fill_manual(values = c("gray70", "purple"),
                               labels = c("P(z = 1) ≤ 0.5", "P(z = 1) > 0.5")) +
    ggplot2::geom_hline(yintercept = 0.5, color = "red", linetype = "dashed", linewidth = 1) +
    ggplot2::labs(title = expression(paste("Component Membership: ", z[t])),
                  x = "Time", y = expression(P(z[t] == 1)),
                  fill = NULL) +
    ggplot2::ylim(0, 1) +
    ggplot2::theme_minimal() +
    ggplot2::theme(legend.position = "top")

  print(p2)

  return(invisible(NULL))
}
