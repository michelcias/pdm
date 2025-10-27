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
#' @param ci Logical; whether to display credible intervals in plots that
#'   support them. Default is \code{TRUE}.
#' @param ci_level Numeric; Bayesian confidence level for credible intervals
#'   (between 0 and 1). Default is \code{0.95}.
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
#' Generates 13 pages in total:
#' \itemize{
#'   \item Pages 1-8: Individual parameter diagnostics (4 panels each)
#'   \item Page 9: Mixture parameters (bivariate relationships)
#'   \item Pages 10-11: Dynamic state trajectories and diagnostics
#'   \item Page 12: Mixture weight alpha_t
#'   \item Page 13: Component membership P(z_t = 1 | data)
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
                                           ci = TRUE,
                                           ci_level = 0.95,
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
           all = plot_all_base(x, ask = ask, overlay_data = overlay_data,
                               ci = ci, ci_level = ci_level, ...),
           mcmc = plot_mcmc_diagnostics_base(x, which = which, ...),
           params = plot_mixture_params_base(x, which = which, ...),
           states = plot_dynamic_states_base(x, which = which,
                                             ci = ci, ci_level = ci_level, ...),
           alpha = plot_mixture_weights_base(x, overlay_data = overlay_data,
                                             ci = ci, ci_level = ci_level, ...),
    )
  } else {
    switch(type,
           all = plot_all_ggplot(x, ask = ask, overlay_data = overlay_data,
                                 ci = ci, ci_level = ci_level, ...),
           mcmc = plot_mcmc_diagnostics_ggplot(x, which = which, ...),
           params = plot_mixture_params_ggplot(x, which = which, ...),
           states = plot_dynamic_states_ggplot(x, which = which,
                                               ci = ci, ci_level = ci_level, ...),
           alpha = plot_mixture_weights_ggplot(x, overlay_data = overlay_data,
                                               ci = ci, ci_level = ci_level, ...)
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

  par(mfrow = c(2, 2), mar = c(4, 4, 3, 1), oma = c(0, 0, 3, 0),
      mgp = c(2.5, 1, 0))

  # 1. Trace Plot
  plot(param_samples, type = "l", col = "gray40", lwd = 0.8,
       xlab = "Iteration", ylab = param_label,
       main = "Trace Plot")
  segments(x0 = 1, y0 = median(param_samples),
           x1 = length(param_samples), y1 = median(param_samples),
           col = "red", lwd = 2, lty = 2)
  if (!is.null(true_value)) {
    segments(x0 = 1, y0 = true_value,
             x1 = length(param_samples), y1 = true_value,
             col = "blue", lwd = 2, lty = 3)
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
  plot(running_mean, type = "l", col = "steelblue", lwd = 2,
       xlab = "Iteration", ylab = param_label,
       main = "Running Mean")
  segments(x0 = 1, y0 = median(param_samples),
           x1 = length(param_samples), y1 = median(param_samples),
           col = "red", lwd = 2, lty = 2)
  if (!is.null(true_value)) {
    segments(x0 = 1, y0 = true_value,
             x1 = length(param_samples), y1 = true_value,
             col = "blue", lwd = 2, lty = 3)
  }
  grid()

  # Overall title
  mtext(substitute(paste("MCMC Diagnostics: ", x), list(x = param_name)),
        outer = TRUE, cex = 1.3, font = 2)
}


#' Complete dashboard with base R graphics
#' @keywords internal
#' @noRd
plot_all_base <- function(x, ask = TRUE, overlay_data = FALSE,
                          ci = TRUE, ci_level = 0.95, ...) {
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  if (ask) {
    oldask <- par(ask = TRUE)
    on.exit(par(oldask), add = TRUE)
  }

  # Pages 1-8: Individual parameter diagnostics (4 panels each)
  plot_mcmc_diagnostics_base(x, which = 1:8, ...)

  # Page 9: Mixture parameters (bivariate relationships)
  plot_mixture_params_base(x, ...)

  # Pages 10-11: Dynamic states (trajectories and diagnostics)
  plot_dynamic_states_base(x, which = 1:2, ci = ci, ci_level = ci_level, ...)

  # Pages 12-13: Mixture weights (alpha_t e z_t)
  plot_mixture_weights_base(x, overlay_data = overlay_data,
                            ci = ci, ci_level = ci_level, ...)
}


#' MCMC diagnostics with base R graphics
#' @keywords internal
#' @noRd
plot_mcmc_diagnostics_base <- function(x, which = NULL, ...) {

  # Define available parameters
  all_params <- list(
    mu_1 = list(samples = x$mu_1,
                name = quote(mu[1]),
                label = expression(mu[1])),
    mu_2 = list(samples = x$mu_2,
                name = quote(mu[2]),
                label = expression(mu[2])),
    phi_1 = list(samples = x$prec_1,
                 name = quote(phi[1]),
                 label = expression(phi[1])),
    phi_2 = list(samples = x$prec_2,
                 name = quote(phi[2]),
                 label = expression(phi[2])),
    theta_01 = list(samples = x$theta_01,
                    name = quote(theta["0,1"]),
                    label = expression(theta["0,1"])),
    theta_02 = list(samples = x$theta_02,
                    name = quote(theta["0,2"]),
                    label = expression(theta["0,2"])),
    W1_inv = list(samples = x$prec_theta1,
                  name = quote(W[1]^{-1}),
                  label = expression(1/W[1])),
    W2_inv = list(samples = x$prec_theta2,
                  name = quote(W[2]^{-1}),
                  label = expression(1/W[2]))
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

  par(mfrow = c(2, 2), mar = c(4, 4, 2, 1), oma = c(0, 0, 2, 0),
      mgp = c(2.5, 1, 0))

  # Plot 1: mu_1 vs mu_2
  if (1 %in% which) {
    plot(x$mu_1, x$mu_2,
         xlab = expression(mu[1]),
         ylab = expression(mu[2]),
         main = expression(paste(mu[1], " vs ", mu[2])),
         pch = 16, col = rgb(0.2, 0.5, 0.8, 0.15))
    grid()
  }

  # Plot 2: mu_1 vs phi_1
  if (2 %in% which) {
    plot(x$mu_1, x$prec_1,
         xlab = expression(mu[1]),
         ylab = expression(phi[1]),
         main = expression(paste(mu[1], " vs ", phi[1])),
         pch = 16, col = rgb(0.2, 0.5, 0.8, 0.15))
    grid()
  }

  # Plot 3: mu_2 vs phi_2
  if (3 %in% which) {
    plot(x$mu_2, x$prec_2,
         xlab = expression(mu[2]),
         ylab = expression(phi[2]),
         main = expression(paste(mu[2], " vs ", phi[2])),
         pch = 16, col = rgb(0, 0.5, 0, 0.15))
    grid()
  }

  # Plot 4: phi_1 vs phi_2
  if (4 %in% which) {
    plot(x$prec_1, x$prec_2,
         xlab = expression(phi[1]),
         ylab = expression(phi[2]),
         main = expression(paste(phi[1], " vs ", phi[2])),
         pch = 16, col = rgb(0.5, 0, 0.5, 0.15))
    grid()
  }

  mtext("Mixture Component Parameters (Bivariate Relationships)",
        outer = TRUE, cex = 1.3, font = 2)
}


#' Dynamic states with base R graphics (2 pages)
#' @keywords internal
#' @noRd
plot_dynamic_states_base <- function(x, which = NULL, ci = TRUE,
                                     ci_level = 0.95, ...) {

  if (ci) {
    if (!is.numeric(ci_level) || ci_level <= 0 || ci_level >= 1) {
      stop("`ci_level` must be a numeric value between 0 and 1")
    }
    ci_lower_prob <- (1 - ci_level) / 2
    ci_upper_prob <- 1 - ci_lower_prob
    ci_label <- paste0(round(ci_level * 100), "% CI")
  }

  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  n_obs <- attr(x, "n_obs")
  time_grid <- seq_len(n_obs)

  # Default: both pages
  if (is.null(which)) which <- 1:2

  # =========================================================================
  # Page 1: State Trajectories (theta_1 and theta_2)
  # =========================================================================

  if (1 %in% which) {
    par(mfrow = c(2, 1), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0),
        mgp = c(2.5, 1, 0))

    # Compute credible bands for theta_1
    theta1_median <- apply(x$theta_1, 2, median)
    if (ci) {
      theta1_lower <- apply(x$theta_1, 2, quantile, probs = ci_lower_prob)
      theta1_upper <- apply(x$theta_1, 2, quantile, probs = ci_upper_prob)
      range_theta1 <- range(c(theta1_lower, theta1_upper))
    } else {
      range_theta1 <- range(theta1_median)
    }
    if (diff(range_theta1) == 0) {
      range_theta1 <- range_theta1 + c(-0.5, 0.5)
    }
    range_theta1[2] <- range_theta1[2] + 0.25 * diff(range_theta1)

    # Plot 1.1: theta_1 trajectory
    plot(time_grid, theta1_median, type = "l", lwd = 2,
         xlab = "Time", ylab = expression(theta["t,1"]),
         main = "Level State",
         ylim = range_theta1)

    if (ci) {
      polygon(c(time_grid, rev(time_grid)),
              c(theta1_lower, rev(theta1_upper)),
              col = rgb(0.7, 0.7, 0.7, 0.5), border = NA)
    }

    lines(time_grid, theta1_median, lwd = 2, col = "black")
    grid()
    if (ci) {
      legend("topright", legend = c(expression(hat(theta)["t,1"]), ci_label),
             col = c("black", rgb(0.7, 0.7, 0.7, 0.5)), horiz = TRUE,
             lty = c(1, 1), lwd = c(2, 8), bty = "n")
    } else {
      legend("topright", legend = expression(hat(theta)["t,1"]),
             col = "black", horiz = TRUE, lty = 1, lwd = 2, bty = "n")
    }

    # Compute credible bands for theta_2
    theta2_median <- apply(x$theta_2, 2, median)
    if (ci) {
      theta2_lower <- apply(x$theta_2, 2, quantile, probs = ci_lower_prob)
      theta2_upper <- apply(x$theta_2, 2, quantile, probs = ci_upper_prob)
      range_theta2 <- range(c(theta2_lower, theta2_upper))
    } else {
      range_theta2 <- range(theta2_median)
    }
    if (diff(range_theta2) == 0) {
      range_theta2 <- range_theta2 + c(-0.5, 0.5)
    }
    range_theta2[2] <- range_theta2[2] + 0.25 * diff(range_theta2)

    # Plot 1.2: theta_2 trajectory
    plot(time_grid, theta2_median, type = "l", lwd = 2,
         xlab = "Time", ylab = expression(hat(theta)["t,2"]),
         main = "Trend State",
         ylim = range_theta2)

    if (ci) {
      polygon(c(time_grid, rev(time_grid)),
              c(theta2_lower, rev(theta2_upper)),
              col = rgb(0.7, 0.7, 0.7, 0.5), border = NA)
    }

    lines(time_grid, theta2_median, lwd = 2, col = "black")
    grid()
    if (ci) {
      legend("topright", legend = c(expression(theta["t,2"]), ci_label),
             col = c("black", rgb(0.7, 0.7, 0.7, 0.5)), horiz = TRUE,
             lty = c(1, 1), lwd = c(2, 8), bty = "n")
    } else {
      legend("topright", legend = expression(theta["t,2"]),
             col = "black", horiz = TRUE, lty = 1, lwd = 2, bty = "n")
    }

    mtext("Dynamic State Trajectories", outer = TRUE, cex = 1.3, font = 2)
  }

  # =========================================================================
  # Page 2: Diagnostics (State space, innovations, etc.)
  # =========================================================================

  if (2 %in% which) {
    par(mfrow = c(2, 2), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))

    # Recompute if not already done
    if (!exists("theta1_median")) {
      theta1_median <- apply(x$theta_1, 2, median)
      theta2_median <- apply(x$theta_2, 2, median)
    }

    # Plot 2.1: State space (theta_1 vs theta_2)
    theta1_vec <- as.vector(x$theta_1)
    theta2_vec <- as.vector(x$theta_2)

    # Sample if too many points
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
         pch = 16, cex = 0.9, col = rgb(1, 0.5, 0, 0.2))
    grid()

    # Plot 2.2: Level innovations (u_{t,1})
    # u_{t,1} = theta_{t,1} - theta_{t-1,1} - theta_{t-1,2}
    # Uses matrix algebra for efficiency
    innovations_1 <- x$theta_1[, -1] - x$theta_1[, -ncol(x$theta_1)] - x$theta_2[, -ncol(x$theta_2)]

    innov1_median <- apply(innovations_1, 2, median)
    if (ci) {
      innov1_lower <- apply(innovations_1, 2, quantile, probs = ci_lower_prob)
      innov1_upper <- apply(innovations_1, 2, quantile, probs = ci_upper_prob)
      range_innov1 <- range(c(innov1_lower, innov1_upper))
    } else {
      range_innov1 <- range(innov1_median)
    }
    if (diff(range_innov1) == 0) {
      range_innov1 <- range_innov1 + c(-0.5, 0.5)
    }
    range_innov1[2] <- range_innov1[2] + 0.25 * diff(range_innov1)

    plot(time_grid[-1], innov1_median, type = "h", lwd = 2, col = "steelblue",
         xlab = "Time", ylab = expression(u["t,1"]),
         main = "Level Innovations",
         ylim = range_innov1)

    if (ci) {
      polygon(c(time_grid[-1], rev(time_grid[-1])),
              c(innov1_lower, rev(innov1_upper)),
              col = rgb(0.7, 0.7, 0.7, 0.4), border = NA)
    }

    segments(x0 = 1, y0 = 0, x1 = length(innov1_median), y1 = 0,
             col = "red", lty = 2, lwd = 2)

    grid()
    if (ci) {
      legend("topright", legend = c("Median", ci_label),
             col = c("steelblue", rgb(0.7, 0.7, 0.7, 0.4)), horiz = TRUE,
             lty = c(1, 1), lwd = c(2, 8), bty = "n")
    } else {
      legend("topright", legend = "Median", col = "steelblue", horiz = TRUE,
             lty = 1, lwd = 2, bty = "n")
    }

    # Plot 2.3: Trend innovations (u_{t,2})
    # u_{t,2} = theta_{t,2} - theta_{t-1,2} - theta_{t-1,3}
    innovations_2 <- t(apply(x$theta_2, 1, diff))

    innov2_median <- apply(innovations_2, 2, median)
    if (ci) {
      innov2_lower <- apply(innovations_2, 2, quantile, probs = ci_lower_prob)
      innov2_upper <- apply(innovations_2, 2, quantile, probs = ci_upper_prob)
      range_innov2 <- range(c(innov2_lower, innov2_upper))
    } else {
      range_innov2 <- range(innov2_median)
    }
    if (diff(range_innov2) == 0) {
      range_innov2 <- range_innov2 + c(-0.5, 0.5)
    }
    range_innov2[2] <- range_innov2[2] + 0.25 * diff(range_innov2)

    plot(time_grid[-1], innov2_median, type = "h", lwd = 2, col = "darkgreen",
         xlab = "Time", ylab = expression(u["t,2"]),
         main = "Trend Innovations",
         ylim = range_innov2)

    if (ci) {
      polygon(c(time_grid[-1], rev(time_grid[-1])),
              c(innov2_lower, rev(innov2_upper)),
              col = rgb(0.7, 0.7, 0.7, 0.4), border = NA)
    }

    segments(x0 = 1, y0 = 0, x1 = length(innov2_median), y1 = 0,
             col = "red", lty = 2, lwd = 2)
    grid()
    if (ci) {
      legend("topright", legend = c("Median", ci_label),
             col = c("darkgreen", rgb(0.7, 0.7, 0.7, 0.4)), horiz = TRUE,
             lty = c(1, 1), lwd = c(2, 8), bty = "n")
    } else {
      legend("topright", legend = "Median", col = "darkgreen", horiz = TRUE,
             lty = 1, lwd = 2, bty = "n")
    }

    # Plot 2.4: Trajectory plot (theta_1 and theta_2 together)
    ylim_range <- range(c(theta1_median, theta2_median))
    ylim_range[2] <- ylim_range[2] + 0.25 * diff(ylim_range)

    plot(time_grid, theta1_median, type = "l", lwd = 2, col = "steelblue",
         xlab = "Time", ylab = "State Value",
         main = "Joint Trajectories",
         ylim = ylim_range)
    lines(time_grid, theta2_median, lwd = 2, col = "darkgreen")
    grid()
    legend("topright", horiz = TRUE,
           legend = c(expression(theta["t,1"]), expression(theta["t,2"])),
           col = c("steelblue", "darkgreen"),
           lty = 1, lwd = 2, bty = "n")

    mtext("Dynamic State Diagnostics", outer = TRUE, cex = 1.3, font = 2)
  }
}


#' Mixture weights with base R graphics (2 pages)
#' @keywords internal
#' @noRd
plot_mixture_weights_base <- function(x, overlay_data = FALSE, ci = TRUE,
                                      ci_level = 0.95, ...) {

  if (ci) {
    if (!is.numeric(ci_level) || ci_level <= 0 || ci_level >= 1) {
      stop("`ci_level` must be a numeric value between 0 and 1")
    }
    ci_lower_prob <- (1 - ci_level) / 2
    ci_upper_prob <- 1 - ci_lower_prob
    ci_label <- paste0(round(ci_level * 100), "% CI")
  }

  n_obs <- attr(x, "n_obs")
  time_grid <- seq_len(n_obs)

  # Compute credible bands for alpha
  alpha_median <- apply(x$alpha, 2, median)
  if (ci) {
    alpha_lower <- apply(x$alpha, 2, quantile, probs = ci_lower_prob)
    alpha_upper <- apply(x$alpha, 2, quantile, probs = ci_upper_prob)
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
            col = rgb(0.2, 0.5, 0.8, 0.3), border = NA)
  }

  # Re-draw median on top
  lines(time_grid, alpha_median, lwd = 2.5, col = "blue")

  grid()

  # Simplified legend (moved to bottom-right to avoid overlap)
  if (ci) {
    legend("topright", horiz = TRUE,
           legend = c(expression(hat(alpha)[t]), ci_label),
           col = c("blue", rgb(0.2, 0.5, 0.8, 0.3)),
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

  z_prob <- apply(x$z, 2, mean)

  plot(z_prob,
       type = "h",
       lwd = 2,
       col = ifelse(z_prob > 0.5, "purple", "blue"),
       xlab = "Time",
       ylab = expression(paste("P(", z[t], " = 1 | data)")),
       ylim = c(0, 1.1),
       # main = "Posterior Probabilities of Component Membership",
       axes = FALSE)

  axis(side = 1)
  axis(side = 2, at = c(0, 0.5, 1))

  # # Create barplot
  # barplot(z_prob,
  #         col = ifelse(z_prob > 0.5, "purple", "blue"),
  #         border = NA,
  #         ylim = c(0, 1),
  #         xlab = "Time",
  #         ylab = expression(paste("P(", z[t], " = 1 | data)")),
  #         space = 0)

  # Add threshold line
  segments(x0 = 1, y0 = 0.5, x1 = n_obs, y1 = 0.5,
           col = "red", lwd = 2, lty = 2)

  legend("topright",
         horiz = TRUE,
         legend = c("P > 0.5", "P <= 0.5", "Threshold"),
         fill = c("purple", "blue", NA),
         border = c(NA, NA, NA),
         lty = c(NA, NA, 2),
         lwd = c(NA, NA, 2),
         col = c(NA, NA, "red"),
         bty = "n")

  grid(nx = NA, ny = NULL)

  mtext(expression(paste("Posterior Probability: P(", z[t], " = 1 | data)")),
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
    ggplot2::labs(title = "Trace Plot", x = "Iteration") +
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
    ggplot2::labs(title = "Posterior Density", y = "Density") +
    ggplot2::theme_minimal()

  # 4. Running mean
  p4 <- ggplot2::ggplot(df, ggplot2::aes(x = iteration, y = running_mean)) +
    ggplot2::geom_line(color = "steelblue", linewidth = 1) +
    ggplot2::geom_hline(yintercept = median(param_samples),
                        color = "red", linetype = "dashed", linewidth = 1) +
    ggplot2::labs(title = "Running Mean", x = "Iteration") +
    ggplot2::theme_minimal()

  # Convert parameter names to expressions for both axis labels and title
  param_expr <- switch(param_label_text,
                       "mu_1" = expression(mu[1]),
                       "mu_2" = expression(mu[2]),
                       "phi_1" = expression(phi[1]),
                       "phi_2" = expression(phi[2]),
                       "theta_01" = expression(theta["0,1"]),
                       "theta_02" = expression(theta["0,2"]),
                       "W_1^{-1}" = expression(W[1]^{-1}),
                       "W_2^{-1}" = expression(W[2]^{-1}),
                       param_label_text  # fallback to original text
  )

  # Create title expression
  title_expr <- switch(param_label_text,
                       "mu_1" = expression(paste("MCMC Diagnostics: ", mu[1])),
                       "mu_2" = expression(paste("MCMC Diagnostics: ", mu[2])),
                       "phi_1" = expression(paste("MCMC Diagnostics: ", phi[1])),
                       "phi_2" = expression(paste("MCMC Diagnostics: ", phi[2])),
                       "theta_01" = expression(paste("MCMC Diagnostics: ", theta["0,1"])),
                       "theta_02" = expression(paste("MCMC Diagnostics: ", theta["0,2"])),
                       "W_1^{-1}" = expression(paste("MCMC Diagnostics: ", W[1]^{-1})),
                       "W_2^{-1}" = expression(paste("MCMC Diagnostics: ", W[2]^{-1})),
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
plot_all_ggplot <- function(x, ask = TRUE, overlay_data = TRUE,
                            ci = TRUE, ci_level = 0.95, ...) {

  if (ask) {
    message("Press [Enter] to see next plot...")
  }

  # Pages 1-8: Individual parameter diagnostics
  plot_mcmc_diagnostics_ggplot(x, which = 1:8, ...)
  if (ask) readline()

  # Page 9: Mixture parameters
  plot_mixture_params_ggplot(x, ...)
  if (ask) readline()

  # Page 10: Dynamic state trajectories
  plot_dynamic_states_ggplot(x, which = 1, ci = ci, ci_level = ci_level, ...)
  if (ask) readline()

  # Page 11: Dynamic state diagnostics
  plot_dynamic_states_ggplot(x, which = 2, ci = ci, ci_level = ci_level, ...)
  if (ask) readline()

  # Pages 12-13: Mixture weights
  plot_mixture_weights_ggplot(x, overlay_data = overlay_data,
                              ci = ci, ci_level = ci_level, ...)
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

  if (is.null(which)) {
    which <- 1:4
  }

  if (any(!which %in% 1:4)) {
    stop("`which` must be between 1 and 4")
  }

  col_mu_mu <- grDevices::rgb(0.2, 0.5, 0.8, alpha = 0.15)
  col_mu_phi1 <- grDevices::rgb(0.2, 0.5, 0.8, alpha = 0.15)
  col_mu_phi2 <- grDevices::rgb(0, 0.5, 0, alpha = 0.15)
  col_phi_phi <- grDevices::rgb(0.5, 0, 0.5, alpha = 0.15)

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
    df1 <- data.frame(mu_1 = x$mu_1, mu_2 = x$mu_2)
    p1 <- ggplot2::ggplot(df1, ggplot2::aes(x = mu_1, y = mu_2)) +
      ggplot2::geom_point(color = col_mu_mu, shape = 16, size = 1.5) +
      ggplot2::labs(
        title = expression(paste(mu[1], " vs ", mu[2])),
        x = expression(mu[1]),
        y = expression(mu[2])
      )
    add_plot(p1)
  }

  if (2 %in% which) {
    df2 <- data.frame(mu_1 = x$mu_1, phi_1 = x$prec_1)
    p2 <- ggplot2::ggplot(df2, ggplot2::aes(x = mu_1, y = phi_1)) +
      ggplot2::geom_point(color = col_mu_phi1, shape = 16, size = 1.5) +
      ggplot2::labs(
        title = expression(paste(mu[1], " vs ", phi[1])),
        x = expression(mu[1]),
        y = expression(phi[1])
      )
    add_plot(p2)
  }

  if (3 %in% which) {
    df3 <- data.frame(mu_2 = x$mu_2, phi_2 = x$prec_2)
    p3 <- ggplot2::ggplot(df3, ggplot2::aes(x = mu_2, y = phi_2)) +
      ggplot2::geom_point(color = col_mu_phi2, shape = 16, size = 1.5) +
      ggplot2::labs(
        title = expression(paste(mu[2], " vs ", phi[2])),
        x = expression(mu[2]),
        y = expression(phi[2])
      )
    add_plot(p3)
  }

  if (4 %in% which) {
    df4 <- data.frame(phi_1 = x$prec_1, phi_2 = x$prec_2)
    p4 <- ggplot2::ggplot(df4, ggplot2::aes(x = phi_1, y = phi_2)) +
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
          plot.title = ggplot2::element_text(size = 14, face = "bold", hjust = 0.5)
        )
      )
    print(combined)
  } else {
    if (length(plots) == 1) {
      plots[[1]] <- plots[[1]] +
        ggplot2::labs(subtitle = "Mixture Component Parameters (Bivariate Relationships)") +
        ggplot2::theme(
          plot.subtitle = ggplot2::element_text(size = 14, face = "bold", hjust = 0.5)
        )
    }
    for (p in plots) {
      print(p)
    }
  }

  invisible(NULL)
}


#' Dynamic states with ggplot2
#' @keywords internal
#' @noRd
plot_dynamic_states_ggplot <- function(x, which = NULL, ci = TRUE, ci_level = 0.95, ...) {

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required")
  }

  # Validate ci_level
  if (!is.numeric(ci_level) || ci_level <= 0 || ci_level >= 1) {
    stop("`ci_level` must be a numeric value between 0 and 1")
  }

  if (is.null(which)) {
    which <- 1:2
  }

  if (any(!which %in% 1:2)) {
    stop("`which` must be 1 or 2")
  }

  n_obs <- attr(x, "n_obs")
  time_grid <- seq_len(n_obs)
  patchwork_available <- requireNamespace("patchwork", quietly = TRUE)

  # Calculate quantile probabilities
  ci_lower_prob <- (1 - ci_level) / 2
  ci_upper_prob <- 1 - ci_lower_prob

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

  theta1_median <- apply(x$theta_1, 2, median)
  theta2_median <- apply(x$theta_2, 2, median)

  if (ci) {
    theta1_lower <- apply(x$theta_1, 2, stats::quantile, probs = ci_lower_prob)
    theta1_upper <- apply(x$theta_1, 2, stats::quantile, probs = ci_upper_prob)
    theta2_lower <- apply(x$theta_2, 2, stats::quantile, probs = ci_lower_prob)
    theta2_upper <- apply(x$theta_2, 2, stats::quantile, probs = ci_upper_prob)
  }

  if (1 %in% which) {
    df_theta1 <- data.frame(
      time = time_grid,
      median = theta1_median
    )

    if (ci) {
      df_theta1$lower <- theta1_lower
      df_theta1$upper <- theta1_upper
    }

    df_theta2 <- data.frame(
      time = time_grid,
      median = theta2_median
    )

    if (ci) {
      df_theta2$lower <- theta2_lower
      df_theta2$upper <- theta2_upper
    }

    ci_pct <- round(ci_level * 100)

    p_level <- ggplot2::ggplot(df_theta1, ggplot2::aes(x = time))

    if (ci) {
      p_level <- p_level +
        ggplot2::geom_ribbon(
          ggplot2::aes(ymin = lower, ymax = upper, fill = "CI"),
          alpha = 0.5, colour = NA
        )
    }

    p_level <- p_level +
      ggplot2::geom_line(
        ggplot2::aes(y = median, colour = "Median"),
        linewidth = 1.2
      ) +
      ggplot2::scale_color_manual(
        values = c("Median" = "black"),
        breaks = "Median",
        labels = expression(hat(theta)["t,1"])
      )

    if (ci) {
      p_level <- p_level +
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
      p_level <- p_level +
        ggplot2::guides(colour = ggplot2::guide_legend(order = 1))
    }

    p_level <- p_level +
      ggplot2::labs(
        title = "Level State",
        x = "Time",
        y = expression(theta["t,1"])
      ) +
      base_theme +
      legend_outside

    p_trend <- ggplot2::ggplot(df_theta2, ggplot2::aes(x = time))

    if (ci) {
      p_trend <- p_trend +
        ggplot2::geom_ribbon(
          ggplot2::aes(ymin = lower, ymax = upper, fill = "CI"),
          alpha = 0.5, colour = NA
        )
    }

    p_trend <- p_trend +
      ggplot2::geom_line(
        ggplot2::aes(y = median, colour = "Median"),
        linewidth = 1.2
      ) +
      ggplot2::scale_color_manual(
        values = c("Median" = "black"),
        breaks = "Median",
        labels = expression(hat(theta)["t,2"])
      )

    if (ci) {
      p_trend <- p_trend +
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
      p_trend <- p_trend +
        ggplot2::guides(colour = ggplot2::guide_legend(order = 1))
    }

    p_trend <- p_trend +
      ggplot2::labs(
        title = "Trend State",
        x = "Time",
        y = expression(theta["t,2"])
      ) +
      base_theme +
      legend_outside

    page1 <- list(p_level, p_trend)

    if (patchwork_available) {
      combined1 <- patchwork::wrap_plots(page1, ncol = 1) +
        patchwork::plot_annotation(
          title = "Dynamic State Trajectories",
          theme = ggplot2::theme(
            plot.title = ggplot2::element_text(size = 14, face = "bold", hjust = 0.5)
          )
        )
      print(combined1)
    } else {
      page1[[1]] <- page1[[1]] +
        ggplot2::labs(subtitle = "Dynamic State Trajectories") +
        ggplot2::theme(
          plot.subtitle = ggplot2::element_text(size = 14, face = "bold", hjust = 0.5)
        )
      for (p in page1) {
        print(p)
      }
    }
  }

  if (2 %in% which) {
    theta1_vec <- as.vector(x$theta_1)
    theta2_vec <- as.vector(x$theta_2)

    max_points <- 5000L
    if (length(theta1_vec) > max_points) {
      idx <- sample.int(length(theta1_vec), max_points)
      theta1_vec <- theta1_vec[idx]
      theta2_vec <- theta2_vec[idx]
    }

    df_state <- data.frame(theta1 = theta1_vec, theta2 = theta2_vec)

    innovations_1 <- x$theta_1[, -1, drop = FALSE] - x$theta_1[, -ncol(x$theta_1), drop = FALSE] -
      x$theta_2[, -ncol(x$theta_2), drop = FALSE]
    innovations_2 <- t(apply(x$theta_2, 1, diff))

    time_innov <- time_grid[-1]

    innov1_median <- apply(innovations_1, 2, median)
    innov2_median <- apply(innovations_2, 2, median)

    df_innov1 <- data.frame(
      time = time_innov,
      median = innov1_median
    )

    if (ci) {
      df_innov1$lower <- apply(innovations_1, 2, stats::quantile, probs = ci_lower_prob)
      df_innov1$upper <- apply(innovations_1, 2, stats::quantile, probs = ci_upper_prob)
    }

    df_innov2 <- data.frame(
      time = time_innov,
      median = innov2_median
    )

    if (ci) {
      df_innov2$lower <- apply(innovations_2, 2, stats::quantile, probs = ci_lower_prob)
      df_innov2$upper <- apply(innovations_2, 2, stats::quantile, probs = ci_upper_prob)
    }

    df_joint <- data.frame(
      time = rep(time_grid, 2),
      value = c(theta1_median, theta2_median),
      state = factor(rep(c("theta1", "theta2"), each = n_obs),
                     levels = c("theta1", "theta2"))
    )

    ci_pct <- round(ci_level * 100)

    p_state <- ggplot2::ggplot(df_state, ggplot2::aes(x = theta1, y = theta2)) +
      ggplot2::geom_point(
        colour = grDevices::rgb(1, 0.5, 0, alpha = 0.2),
        shape = 16, size = 1.2
      ) +
      ggplot2::labs(
        title = "State Space",
        x = expression(theta["t,1"]),
        y = expression(theta["t,2"])
      ) +
      base_theme +
      ggplot2::theme(legend.position = "none")

    p_innov1 <- ggplot2::ggplot(df_innov1, ggplot2::aes(x = time))

    if (ci) {
      p_innov1 <- p_innov1 +
        ggplot2::geom_ribbon(
          ggplot2::aes(ymin = lower, ymax = upper, fill = "CI"),
          alpha = 0.4, colour = NA
        )
    }

    p_innov1 <- p_innov1 +
      ggplot2::geom_segment(
        ggplot2::aes(xend = time, y = 0, yend = median, colour = "Median"),
        linewidth = 1.1
      ) +
      ggplot2::geom_hline(yintercept = 0, colour = "red", linetype = "dashed", linewidth = 0.8) +
      ggplot2::scale_color_manual(
        values = c("Median" = "steelblue"),
        breaks = "Median",
        labels = "Median"
      )

    if (ci) {
      p_innov1 <- p_innov1 +
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
      p_innov1 <- p_innov1 +
        ggplot2::guides(colour = ggplot2::guide_legend(order = 1))
    }

    p_innov1 <- p_innov1 +
      ggplot2::labs(
        title = "Level Innovations",
        x = "Time",
        y = expression(u["t,1"])
      ) +
      base_theme +
      legend_outside

    p_innov2 <- ggplot2::ggplot(df_innov2, ggplot2::aes(x = time))

    if (ci) {
      p_innov2 <- p_innov2 +
        ggplot2::geom_ribbon(
          ggplot2::aes(ymin = lower, ymax = upper, fill = "CI"),
          alpha = 0.4, colour = NA
        )
    }

    p_innov2 <- p_innov2 +
      ggplot2::geom_segment(
        ggplot2::aes(xend = time, y = 0, yend = median, colour = "Median"),
        linewidth = 1.1
      ) +
      ggplot2::geom_hline(yintercept = 0, colour = "red", linetype = "dashed", linewidth = 0.8) +
      ggplot2::scale_color_manual(
        values = c("Median" = "darkgreen"),
        breaks = "Median",
        labels = "Median"
      )

    if (ci) {
      p_innov2 <- p_innov2 +
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
      p_innov2 <- p_innov2 +
        ggplot2::guides(colour = ggplot2::guide_legend(order = 1))
    }

    p_innov2 <- p_innov2 +
      ggplot2::labs(
        title = "Trend Innovations",
        x = "Time",
        y = expression(u["t,2"])
      ) +
      base_theme +
      legend_outside

    p_joint <- ggplot2::ggplot(df_joint, ggplot2::aes(x = time, y = value, colour = state)) +
      ggplot2::geom_line(linewidth = 1.2) +
      ggplot2::scale_color_manual(
        values = c("theta1" = "steelblue", "theta2" = "darkgreen"),
        breaks = c("theta1", "theta2"),
        labels = c(expression(theta["t,1"]), expression(theta["t,2"]))
      ) +
      ggplot2::guides(colour = ggplot2::guide_legend(order = 1)) +
      ggplot2::labs(
        title = "Joint Trajectories",
        x = "Time",
        y = "State Value"
      ) +
      base_theme +
      legend_outside

    page2 <- list(p_state, p_innov1, p_innov2, p_joint)

    if (patchwork_available) {
      combined2 <- patchwork::wrap_plots(page2, ncol = 2) +
        patchwork::plot_annotation(
          title = "Dynamic State Diagnostics",
          theme = ggplot2::theme(
            plot.title = ggplot2::element_text(size = 14, face = "bold", hjust = 0.5)
          )
        )
      print(combined2)
    } else {
      page2[[1]] <- page2[[1]] +
        ggplot2::labs(subtitle = "Dynamic State Diagnostics") +
        ggplot2::theme(
          plot.subtitle = ggplot2::element_text(size = 14, face = "bold", hjust = 0.5)
        )
      for (p in page2) {
        print(p)
      }
    }
  }

  invisible(NULL)
}


#' Mixture weights with ggplot2 (2 pages)
#' @keywords internal
#' @noRd
plot_mixture_weights_ggplot <- function(x, overlay_data = TRUE, ci = TRUE, ci_level = 0.95, ...) {

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required")
  }

  # Validate ci_level
  if (!is.numeric(ci_level) || ci_level <= 0 || ci_level >= 1) {
    stop("`ci_level` must be a numeric value between 0 and 1")
  }

  n_obs <- attr(x, "n_obs")
  time_grid <- seq_len(n_obs)

  # Calculate quantile probabilities
  ci_lower_prob <- (1 - ci_level) / 2
  ci_upper_prob <- 1 - ci_lower_prob
  ci_pct <- round(ci_level * 100)

  # Page 1: alpha_t
  df_alpha <- data.frame(
    time = time_grid,
    median = apply(x$alpha, 2, median)
  )

  if (ci) {
    df_alpha$lower <- apply(x$alpha, 2, quantile, probs = ci_lower_prob)
    df_alpha$upper <- apply(x$alpha, 2, quantile, probs = ci_upper_prob)
  }

  p1 <- ggplot2::ggplot(df_alpha, ggplot2::aes(x = time))

  if (ci) {
    p1 <- p1 +
      ggplot2::geom_ribbon(
        ggplot2::aes(ymin = lower, ymax = upper, fill = "CI"),
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
      ggplot2::aes(y = median, colour = "Median"),
      linewidth = 1.2
    ) +
    ggplot2::scale_color_manual(
      values = c("Median" = "blue"),
      breaks = "Median",
      labels = expression(hat(alpha)[t])
    ) +
    ggplot2::scale_y_continuous(
      limits = c(0, 1.1),
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
  z_prob <- apply(x$z, 2, mean)
  df_z <- data.frame(time = time_grid, prob = z_prob)

  p2 <- ggplot2::ggplot(df_z, ggplot2::aes(x = time, y = prob)) +
    ggplot2::geom_col(
      ggplot2::aes(fill = prob > 0.5),
      width = 1
    ) +
    ggplot2::scale_fill_manual(
      values = c("TRUE" = "purple", "FALSE" = "blue"),
      breaks = c("FALSE", "TRUE"),
      labels = c("P(z = 1 | data) <= 0.5", "P(z = 1 | data) > 0.5")
    ) +
    ggplot2::geom_hline(
      yintercept = 0.5,
      color = "red",
      linetype = "dashed",
      linewidth = 1
    ) +
    ggplot2::scale_y_continuous(
      limits = c(0, 1.1),
      breaks = c(0, 0.5, 1)
    ) +
    ggplot2::labs(
      title = expression(paste("Posterior Probability: P(", z[t], " = 1 | data)")),
      x = "Time",
      y = expression(paste("P(", z[t], " = 1 | data)")),
      fill = NULL
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      panel.grid.major = ggplot2::element_line(color = "grey85"),
      panel.grid.minor = ggplot2::element_line(color = "grey92"),
      panel.grid.major.x = ggplot2::element_blank(),
      plot.title = ggplot2::element_text(face = "bold", size = 13),
      legend.position = "top",
      legend.direction = "horizontal"
    )

  print(p2)

  return(invisible(NULL))
}
