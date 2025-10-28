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
#' ## Simulation of data
#' n <- 400  # Number of observations to simulate
#'
#' # Use a fixed seed for data simulation
#' set.seed(123)
#'
#' # Generate true mixture weights following a sinusoidal pattern
#' grid_vals <- seq_len(n) / n
#' alpha_true <- (sin(2 * pi * grid_vals) + sin(4 * pi * grid_vals) + 2) / 4
#'
#' # Generate latent component indicators
#' z_true <- rbinom(n, size = 1, prob = alpha_true)
#'
#' # Generate observations from mixture of N(0, 1/4) and N(2, 1/4)
#' mu_1_true <- 0
#' mu_2_true <- 2
#' sigma_1_true <- 0.5  # sqrt(1/4)
#' sigma_2_true <- 0.5  # sqrt(1/4)
#'
#' mu_y <- (1 - z_true) * mu_1_true + z_true * mu_2_true
#' sigma_y <- (1 - z_true) * sigma_1_true + z_true * sigma_2_true
#' y <- rnorm(n, mean = mu_y, sd = sigma_y)
#'
#' ## Running the Gibbs sampler with logit link
#' out_logit <- mcmc_normal_mixture_localtrend(
#'   y,
#'   link               = "logit",
#'   burnin             = 2000,
#'   thinning           = 10,
#'   n_chain            = 1000,
#'   prior_mu01_mean    = NULL,  # Use default (25th percentile)
#'   prior_mu01_prec    = 0.01,
#'   prior_prec01_shape = 0.01,
#'   prior_prec01_rate  = 0.01,
#'   prior_mu02_mean    = NULL,  # Use default (75th percentile)
#'   prior_mu02_prec    = 0.01,
#'   prior_prec02_shape = 0.01,
#'   prior_prec02_rate  = 0.01,
#'   prior_theta01_mean = 0,
#'   prior_theta01_prec = 1,
#'   prior_theta02_mean = 0,
#'   prior_theta02_prec = 1,
#'   prior_prec1_shape  = 100,
#'   prior_prec1_rate   = 1,
#'   prior_prec2_shape  = 400,
#'   prior_prec2_rate   = 1,
#'   lag_update         = 50,
#'   max_step_size      = 1.0,
#'   base_adaptation_rate = 0.01,
#'   decay_exponent     = 0.6,
#'   target_acceptance  = 0.44,
#'   min_deviation_threshold = NULL,  # Use default (1/lag_update)
#'   return_log_sigma   = FALSE,
#'   return_accept_prop = FALSE,
#'   verbose            = TRUE,
#'   bar_width          = 60,
#'   seed               = 456
#' )
#'
#' # Complete dashboard (13 pages)
#' plot(out_logit, type = "all", engine = "base")
#'
#' # Diagnostics for specific parameters
#' plot(out_logit, type = "mcmc", which = 1:2)  # Only mu_1 and mu_2
#' plot(out_logit, type = "mcmc", which = 3:4)  # Only phi_1 and phi_2
#'
#' # Mixture weights (2 pages: alpha_t and z_t)
#' plot(out_logit, type = "alpha")
#'
#' # Save to multi-page PDF
#' pdf("diagnostics.pdf", width = 10, height = 8)
#' plot(out_logit, type = "all", ask = FALSE)
#' dev.off()
#'
#' # Use ggplot2 engine
#' plot(out_logit, type = "mcmc", which = 1, engine = "ggplot2")
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
           all = plot_all_base_lt(x, ask = ask, overlay_data = overlay_data,
                                  ci = ci, ci_level = ci_level, ...),
           mcmc = plot_mcmc_diagnostics_base_lt(x, which = which, ...),
           params = plot_mixture_params_base(x$mu_1, x$mu_2, x$prec_1, x$prec_2,
                                             which = which, ...),
           states = plot_dynamic_states_base_lt(x, which = which,
                                                ci = ci, ci_level = ci_level, ...),
           alpha = plot_mixture_weights_base(x$alpha, x$z,
                                             ci = ci, ci_level = ci_level, ...),
    )
  } else {
    switch(type,
           all = plot_all_ggplot_lt(x, ask = ask, overlay_data = overlay_data,
                                    ci = ci, ci_level = ci_level, ...),
           mcmc = plot_mcmc_diagnostics_ggplot_lt(x, which = which, ...),
           params = plot_mixture_params_ggplot(x$mu_1, x$mu_2, x$prec_1, x$prec_2,
                                               which = which, ...),
           states = plot_dynamic_states_ggplot_lt(x, which = which,
                                                  ci = ci, ci_level = ci_level, ...),
           alpha = plot_mixture_weights_ggplot(x$alpha, x$z,
                                               ci = ci, ci_level = ci_level, ...)
    )
  }

  invisible(x)
}


# ============================================================================
# Local-Trend Specific Functions (Base R Graphics)
# ============================================================================

#' Complete dashboard with base R graphics (local-trend)
#' @keywords internal
#' @noRd
plot_all_base_lt <- function(x, ask = TRUE, overlay_data = FALSE,
                             ci = TRUE, ci_level = 0.95, ...) {
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  if (ask) {
    oldask <- par(ask = TRUE)
    on.exit(par(oldask), add = TRUE)
  }

  # Pages 1-8: Individual parameter diagnostics (4 panels each)
  plot_mcmc_diagnostics_base_lt(x, which = 1:8, ...)

  # Page 9: Mixture parameters (bivariate relationships)
  plot_mixture_params_base(x$mu_1, x$mu_2, x$prec_1, x$prec_2, ...)

  # Pages 10-11: Dynamic states (trajectories and diagnostics)
  plot_dynamic_states_base_lt(x, which = 1:2, ci = ci, ci_level = ci_level, ...)

  # Pages 12-13: Mixture weights (alpha_t and z_t)
  plot_mixture_weights_base(x$alpha, x$z, ci = ci, ci_level = ci_level, ...)
}


#' MCMC diagnostics with base R graphics (local-trend)
#' @keywords internal
#' @noRd
plot_mcmc_diagnostics_base_lt <- function(x, which = NULL, ...) {

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
    plot_param_diagnostics_base(
      param_samples = param_info$samples,
      param_name = param_info$name,
      param_label = param_info$label,
      ...
    )
  }
}


#' Dynamic states with base R graphics (local-trend, 2 pages)
#' @keywords internal
#' @noRd
plot_dynamic_states_base_lt <- function(x, which = NULL, ci = TRUE,
                                        ci_level = 0.95, ...) {

  validate_ci_level(ci_level)

  if (ci) {
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
    theta1_median <- apply(x$theta_1, 2, stats::median)
    if (ci) {
      theta1_lower <- apply(x$theta_1, 2, stats::quantile, probs = ci_lower_prob)
      theta1_upper <- apply(x$theta_1, 2, stats::quantile, probs = ci_upper_prob)
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
         xlab = "Time", ylab = "State Value",
         main = "Level State",
         ylim = range_theta1)

    if (ci) {
      polygon(c(time_grid, rev(time_grid)),
              c(theta1_lower, rev(theta1_upper)),
              col = grDevices::rgb(0.7, 0.7, 0.7, 0.5), border = NA)
    }

    lines(time_grid, theta1_median, lwd = 2, col = "black")
    grid()
    if (ci) {
      legend("topright", legend = c(expression(hat(theta)["t,1"]), ci_label),
             col = c("black", grDevices::rgb(0.7, 0.7, 0.7, 0.5)), horiz = TRUE,
             lty = c(1, 1), lwd = c(2, 8), bty = "n")
    } else {
      legend("topright", legend = expression(hat(theta)["t,1"]),
             col = "black", horiz = TRUE, lty = 1, lwd = 2, bty = "n")
    }

    # Compute credible bands for theta_2
    theta2_median <- apply(x$theta_2, 2, stats::median)
    if (ci) {
      theta2_lower <- apply(x$theta_2, 2, stats::quantile, probs = ci_lower_prob)
      theta2_upper <- apply(x$theta_2, 2, stats::quantile, probs = ci_upper_prob)
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
         xlab = "Time", ylab = "State Value",
         main = "Trend State",
         ylim = range_theta2)

    if (ci) {
      polygon(c(time_grid, rev(time_grid)),
              c(theta2_lower, rev(theta2_upper)),
              col = grDevices::rgb(0.7, 0.7, 0.7, 0.5), border = NA)
    }

    lines(time_grid, theta2_median, lwd = 2, col = "black")
    grid()
    if (ci) {
      legend("topright", legend = c(expression(hat(theta)["t,2"]), ci_label),
             col = c("black", grDevices::rgb(0.7, 0.7, 0.7, 0.5)), horiz = TRUE,
             lty = c(1, 1), lwd = c(2, 8), bty = "n")
    } else {
      legend("topright", legend = expression(hat(theta)["t,2"]),
             col = "black", horiz = TRUE, lty = 1, lwd = 2, bty = "n")
    }

    mtext("Dynamic State Trajectories", outer = TRUE, cex = 1.3, font = 2)
  }

  # =========================================================================
  # Page 2: Diagnostics (State space, innovations, joint trajectory)
  # =========================================================================

  if (2 %in% which) {
    par(mfrow = c(2, 2), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))

    # Recompute if not already done
    if (!exists("theta1_median")) {
      theta1_median <- apply(x$theta_1, 2, stats::median)
      theta2_median <- apply(x$theta_2, 2, stats::median)
    }

    # Plot 2.1: State space (theta_1 vs theta_2)
    theta1_vec <- as.vector(x$theta_1)
    theta2_vec <- as.vector(x$theta_2)

    # Sample if too many points
    max_points <- 5000L
    if (length(theta1_vec) > max_points) {
      set.seed(123)
      idx <- sample.int(length(theta1_vec), max_points)
      theta1_vec <- theta1_vec[idx]
      theta2_vec <- theta2_vec[idx]
    }

    plot(theta1_vec, theta2_vec,
         xlab = expression(theta["t,1"]),
         ylab = expression(theta["t,2"]),
         main = "State Space",
         pch = 16, cex = 0.9, col = grDevices::rgb(1, 0.5, 0, 0.2))
    grid()

    # Plot 2.2: Level innovations (u_{t,1})
    if (n_obs >= 2) {
      innovations_1 <- x$theta_1[, -1, drop = FALSE] -
        x$theta_1[, -ncol(x$theta_1), drop = FALSE] -
        x$theta_2[, -ncol(x$theta_2), drop = FALSE]

      innov1_median <- apply(innovations_1, 2, stats::median)
      if (ci) {
        innov1_lower <- apply(innovations_1, 2, stats::quantile, probs = ci_lower_prob)
        innov1_upper <- apply(innovations_1, 2, stats::quantile, probs = ci_upper_prob)
        range_innov1 <- range(c(innov1_lower, innov1_upper))
      } else {
        range_innov1 <- range(innov1_median)
      }
      if (diff(range_innov1) == 0) {
        range_innov1 <- range_innov1 + c(-0.5, 0.5)
      }
      range_innov1[2] <- range_innov1[2] + 0.25 * diff(range_innov1)

      plot(time_grid[-1], innov1_median, type = "l", lwd = 2, col = "steelblue",
           xlab = "Time", ylab = expression(u["t,1"]),
           main = "Level Innovations",
           ylim = range_innov1)

      if (ci) {
        polygon(c(time_grid[-1], rev(time_grid[-1])),
                c(innov1_lower, rev(innov1_upper)),
                col = grDevices::rgb(0.7, 0.7, 0.7, 0.4), border = NA)
      }

      segments(x0 = 1, y0 = 0, x1 = length(innov1_median), y1 = 0,
               col = "red", lty = 2, lwd = 2)

      grid()
      if (ci) {
        legend("topright", legend = c("Median", ci_label),
               col = c("steelblue", grDevices::rgb(0.7, 0.7, 0.7, 0.4)), horiz = TRUE,
               lty = c(1, 1), lwd = c(2, 8), bty = "n")
      } else {
        legend("topright", legend = "Median", col = "steelblue", horiz = TRUE,
               lty = 1, lwd = 2, bty = "n")
      }

      # Plot 2.3: Trend innovations (u_{t,2})
      innovations_2 <- t(apply(x$theta_2, 1, diff))

      innov2_median <- apply(innovations_2, 2, stats::median)
      if (ci) {
        innov2_lower <- apply(innovations_2, 2, stats::quantile, probs = ci_lower_prob)
        innov2_upper <- apply(innovations_2, 2, stats::quantile, probs = ci_upper_prob)
        range_innov2 <- range(c(innov2_lower, innov2_upper))
      } else {
        range_innov2 <- range(innov2_median)
      }
      if (diff(range_innov2) == 0) {
        range_innov2 <- range_innov2 + c(-0.5, 0.5)
      }
      range_innov2[2] <- range_innov2[2] + 0.25 * diff(range_innov2)

      plot(time_grid[-1], innov2_median, type = "l", lwd = 2, col = "darkgreen",
           xlab = "Time", ylab = expression(u["t,2"]),
           main = "Trend Innovations",
           ylim = range_innov2)

      if (ci) {
        polygon(c(time_grid[-1], rev(time_grid[-1])),
                c(innov2_lower, rev(innov2_upper)),
                col = grDevices::rgb(0.7, 0.7, 0.7, 0.4), border = NA)
      }

      segments(x0 = 1, y0 = 0, x1 = length(innov2_median), y1 = 0,
               col = "red", lty = 2, lwd = 2)
      grid()
      if (ci) {
        legend("topright", legend = c("Median", ci_label),
               col = c("darkgreen", grDevices::rgb(0.7, 0.7, 0.7, 0.4)), horiz = TRUE,
               lty = c(1, 1), lwd = c(2, 8), bty = "n")
      } else {
        legend("topright", legend = "Median", col = "darkgreen", horiz = TRUE,
               lty = 1, lwd = 2, bty = "n")
      }
    } else {
      # Not enough observations
      plot.new()
      title("Level Innovations")
      text(x = 0.5, y = 0.5,
           labels = "Innovations require at least 2 observations",
           cex = 1.1)

      plot.new()
      title("Trend Innovations")
      text(x = 0.5, y = 0.5,
           labels = "Innovations require at least 2 observations",
           cex = 1.1)
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
           legend = c(expression(hat(theta)["t,1"]), expression(hat(theta)["t,2"])),
           col = c("steelblue", "darkgreen"),
           lty = 1, lwd = 2, bty = "n")

    mtext("Dynamic State Diagnostics", outer = TRUE, cex = 1.3, font = 2)
  }
}


# ============================================================================
# Local-Trend Specific Functions (ggplot2 Graphics)
# ============================================================================

#' Complete dashboard with ggplot2 graphics (local-trend)
#' @keywords internal
#' @noRd
plot_all_ggplot_lt <- function(x, ask = TRUE, overlay_data = TRUE,
                               ci = TRUE, ci_level = 0.95, ...) {

  if (ask) {
    message("Press [Enter] to see next plot...")
  }

  # Pages 1-8: Individual parameter diagnostics
  plot_mcmc_diagnostics_ggplot_lt(x, which = 1:8, ...)
  if (ask) readline()

  # Page 9: Mixture parameters
  plot_mixture_params_ggplot(x$mu_1, x$mu_2, x$prec_1, x$prec_2, ...)
  if (ask) readline()

  # Page 10: Dynamic state trajectories
  plot_dynamic_states_ggplot_lt(x, which = 1, ci = ci, ci_level = ci_level, ...)
  if (ask) readline()

  # Page 11: Dynamic state diagnostics
  plot_dynamic_states_ggplot_lt(x, which = 2, ci = ci, ci_level = ci_level, ...)
  if (ask) readline()

  # Pages 12-13: Mixture weights
  plot_mixture_weights_ggplot(x$alpha, x$z, ci = ci, ci_level = ci_level, ...)
}


#' MCMC diagnostics with ggplot2 (local-trend)
#' @keywords internal
#' @noRd
plot_mcmc_diagnostics_ggplot_lt <- function(x, which = NULL, ...) {

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


#' Dynamic states with ggplot2 (local-trend)
#' @keywords internal
#' @noRd
plot_dynamic_states_ggplot_lt <- function(x, which = NULL,
                                          ci = TRUE, ci_level = 0.95, ...) {

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required")
  }

  validate_ci_level(ci_level)

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
  if (ci) {
    ci_lower_prob <- (1 - ci_level) / 2
    ci_upper_prob <- 1 - ci_lower_prob
    ci_pct <- round(ci_level * 100)
  }

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

  theta1_median <- apply(x$theta_1, 2, stats::median)
  theta2_median <- apply(x$theta_2, 2, stats::median)

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

    p_level <- ggplot2::ggplot(df_theta1, ggplot2::aes(x = .data$time))

    if (ci) {
      p_level <- p_level +
        ggplot2::geom_ribbon(
          ggplot2::aes(ymin = .data$lower, ymax = .data$upper, fill = "CI"),
          alpha = 0.5, colour = NA
        )
    }

    p_level <- p_level +
      ggplot2::geom_line(
        ggplot2::aes(y = .data$median, colour = "Median"),
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
        y = "State Value"
      ) +
      base_theme +
      legend_outside

    p_trend <- ggplot2::ggplot(df_theta2, ggplot2::aes(x = .data$time))

    if (ci) {
      p_trend <- p_trend +
        ggplot2::geom_ribbon(
          ggplot2::aes(ymin = .data$lower, ymax = .data$upper, fill = "CI"),
          alpha = 0.5, colour = NA
        )
    }

    p_trend <- p_trend +
      ggplot2::geom_line(
        ggplot2::aes(y = .data$median, colour = "Median"),
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
        y = "State Value"
      ) +
      base_theme +
      legend_outside

    page1 <- list(p_level, p_trend)

    if (patchwork_available) {
      combined1 <- patchwork::wrap_plots(page1, ncol = 1) +
        patchwork::plot_annotation(
          title = "Dynamic State Trajectories",
          theme = ggplot2::theme(
            plot.title = ggplot2::element_text(size = 14, face = "bold",
                                               hjust = 0.5)
          )
        )
      print(combined1)
    } else {
      page1[[1]] <- page1[[1]] +
        ggplot2::labs(subtitle = "Dynamic State Trajectories") +
        ggplot2::theme(
          plot.subtitle = ggplot2::element_text(size = 14, face = "bold",
                                                hjust = 0.5)
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
      set.seed(123)
      idx <- sample.int(length(theta1_vec), max_points)
      theta1_vec <- theta1_vec[idx]
      theta2_vec <- theta2_vec[idx]
    }

    df_state <- data.frame(theta1 = theta1_vec, theta2 = theta2_vec)

    if (n_obs >= 2) {
      innovations_1 <- x$theta_1[, -1, drop = FALSE] -
        x$theta_1[, -ncol(x$theta_1), drop = FALSE] -
        x$theta_2[, -ncol(x$theta_2), drop = FALSE]
      innovations_2 <- t(apply(x$theta_2, 1, diff))

      time_innov <- time_grid[-1]

      innov1_median <- apply(innovations_1, 2, stats::median)
      innov2_median <- apply(innovations_2, 2, stats::median)

      df_innov1 <- data.frame(
        time = time_innov,
        median = innov1_median
      )

      if (ci) {
        df_innov1$lower <- apply(innovations_1, 2, stats::quantile,
                                 probs = ci_lower_prob)
        df_innov1$upper <- apply(innovations_1, 2, stats::quantile,
                                 probs = ci_upper_prob)
      }

      df_innov2 <- data.frame(
        time = time_innov,
        median = innov2_median
      )

      if (ci) {
        df_innov2$lower <- apply(innovations_2, 2, stats::quantile,
                                 probs = ci_lower_prob)
        df_innov2$upper <- apply(innovations_2, 2, stats::quantile,
                                 probs = ci_upper_prob)
      }
    }

    df_joint <- data.frame(
      time = rep(time_grid, 2),
      value = c(theta1_median, theta2_median),
      state = factor(rep(c("theta1", "theta2"), each = n_obs),
                     levels = c("theta1", "theta2"))
    )

    p_state <- ggplot2::ggplot(df_state, ggplot2::aes(x = .data$theta1, y = .data$theta2)) +
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

    if (n_obs >= 2) {
      p_innov1 <- ggplot2::ggplot(df_innov1, ggplot2::aes(x = .data$time))

      if (ci) {
        p_innov1 <- p_innov1 +
          ggplot2::geom_ribbon(
            ggplot2::aes(ymin = .data$lower, ymax = .data$upper, fill = "CI"),
            alpha = 0.4, colour = NA
          )
      }

      p_innov1 <- p_innov1 +
        ggplot2::geom_line(
          ggplot2::aes(y = .data$median, colour = "Median"),
          linewidth = 1.1
        ) +
        ggplot2::geom_hline(yintercept = 0, colour = "red", linetype = "dashed",
                            linewidth = 0.8) +
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

      p_innov2 <- ggplot2::ggplot(df_innov2, ggplot2::aes(x = .data$time))

      if (ci) {
        p_innov2 <- p_innov2 +
          ggplot2::geom_ribbon(
            ggplot2::aes(ymin = .data$lower, ymax = .data$upper, fill = "CI"),
            alpha = 0.4, colour = NA
          )
      }

      p_innov2 <- p_innov2 +
        ggplot2::geom_line(
          ggplot2::aes(y = .data$median, colour = "Median"),
          linewidth = 1.1
        ) +
        ggplot2::geom_hline(yintercept = 0, colour = "red", linetype = "dashed",
                            linewidth = 0.8) +
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
    } else {
      # Not enough observations
      p_innov1 <- ggplot2::ggplot() +
        ggplot2::annotate("text", x = 0, y = 0,
                          label = "Innovations require at least 2 observations",
                          size = 4) +
        ggplot2::theme_void() +
        ggplot2::labs(title = "Level Innovations")

      p_innov2 <- ggplot2::ggplot() +
        ggplot2::annotate("text", x = 0, y = 0,
                          label = "Innovations require at least 2 observations",
                          size = 4) +
        ggplot2::theme_void() +
        ggplot2::labs(title = "Trend Innovations")
    }

    p_joint <- ggplot2::ggplot(df_joint, ggplot2::aes(x = .data$time, y = .data$value,
                                                      colour = .data$state)) +
      ggplot2::geom_line(linewidth = 1.2) +
      ggplot2::scale_color_manual(
        values = c("theta1" = "steelblue", "theta2" = "darkgreen"),
        breaks = c("theta1", "theta2"),
        labels = c(expression(hat(theta)["t,1"]), expression(hat(theta)["t,2"]))
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
            plot.title = ggplot2::element_text(size = 14, face = "bold",
                                               hjust = 0.5)
          )
        )
      print(combined2)
    } else {
      page2[[1]] <- page2[[1]] +
        ggplot2::labs(subtitle = "Dynamic State Diagnostics") +
        ggplot2::theme(
          plot.subtitle = ggplot2::element_text(size = 14, face = "bold",
                                                hjust = 0.5)
        )
      for (p in page2) {
        print(p)
      }
    }
  }

  invisible(NULL)
}
