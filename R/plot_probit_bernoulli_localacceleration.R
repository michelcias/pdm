#' Plot method for probit_bernoulli_localacceleration objects
#'
#' @description Produces diagnostic plots for MCMC output from Bernoulli
#'   local acceleration models with probit link.
#'
#' @param x An object of class \code{probit_bernoulli_localacceleration}.
#' @param type Character string specifying the type of plot. One of:
#'   \describe{
#'     \item{\code{"all"}}{Complete dashboard with all diagnostic plots (default)}
#'     \item{\code{"mcmc"}}{MCMC convergence diagnostics (trace plots, ACF, running means)}
#'     \item{\code{"states"}}{Dynamic states (theta_1, theta_2, theta_3 trajectories)}
#'     \item{\code{"alpha"}}{Bernoulli probabilities over time (alpha_t)}
#'   }
#' @param which Integer vector specifying which diagnostic plots to display.
#'   For \code{type = "mcmc"}:
#'   \describe{
#'     \item{1}{theta_01 (initial level)}
#'     \item{2}{theta_02 (initial trend)}
#'     \item{3}{theta_03 (initial acceleration)}
#'     \item{4}{W_1^{-1} (level innovation precision)}
#'     \item{5}{W_2^{-1} (trend innovation precision)}
#'     \item{6}{W_3^{-1} (acceleration innovation precision)}
#'   }
#'   For \code{type = "states"}: indices of subplots.
#'   For \code{type = "alpha"}: not used (alpha_t is shown).
#'   If \code{NULL} (default), all available plots are shown.
#' @param engine Character string specifying the graphics engine. One of:
#'   \describe{
#'     \item{\code{"base"}}{Base R graphics (default, no dependencies)}
#'     \item{\code{"ggplot2"}}{ggplot2 graphics (requires \pkg{ggplot2})}
#'   }
#' @param ask Logical; if \code{TRUE}, the user is asked before each plot when
#'   \code{type = "all"}. Default is \code{interactive()} when \code{type = "all"},
#'   \code{FALSE} otherwise.
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
#' Available parameters: theta_01, theta_02, theta_03, W_1^{-1}, W_2^{-1}, W_3^{-1}
#'
#' \strong{Dynamic States (\code{type = "states"}):}
#' \itemize{
#'   \item Time-varying state trajectories with credible bands (on probit scale)
#'   \item Innovation sequences
#'   \item State space relationships
#' }
#'
#' \strong{Bernoulli Probabilities (\code{type = "alpha"}):}
#' \itemize{
#'   \item alpha_t trajectory with credible bands
#'   \item Binary observations overlay (if available)
#' }
#'
#' \strong{Complete Dashboard (\code{type = "all"}):}
#'
#' Generates 10 pages in total:
#' \itemize{
#'   \item Pages 1-6: Individual parameter diagnostics (4 panels each)
#'   \item Pages 7-9: Dynamic state trajectories and diagnostics
#'   \item Page 10: Bernoulli probabilities alpha_t
#' }
#'
#' The \code{engine} argument allows choosing between base R graphics (lightweight,
#' no dependencies) and ggplot2 (modern, publication-ready). If ggplot2 is not
#' installed and \code{engine = "ggplot2"}, the function falls back to base graphics
#' with a warning.
#'
#' @section Dependencies:
#'
#' The ggplot2 engine has an optional dependency for enhanced visualizations:
#' \itemize{
#'   \item \pkg{patchwork}: For combining multiple plots into layouts
#' }
#'
#' If this package is not installed, the function will display plots sequentially.
#' Install with: \code{install.packages("patchwork")}
#'
#' @examples
#' \dontrun{
#' ## Simulation of data
#' set.seed(123)
#' n <- 500
#'
#' # Generate true probabilities
#' grid_vals <- seq_len(n) / n
#' alpha_true <- (sin(4 * pi * grid_vals) + sin(8 * pi * grid_vals) + 2) / 4
#'
#' # Generate Bernoulli observations
#' y <- rbinom(n, size = 1, prob = alpha_true)
#'
#' ## Running the Gibbs sampler
#' out <- mcmc_probit_bernoulli_localacceleration(
#'   y,
#'   burnin = 1000,
#'   thinning = 50,
#'   n_chain = 1000,
#'   prior_theta01_mean = 0,
#'   prior_theta01_prec = 1,
#'   prior_theta02_mean = 0,
#'   prior_theta02_prec = 1,
#'   prior_theta03_mean = 0,
#'   prior_theta03_prec = 1,
#'   prior_prec1_shape = 100,
#'   prior_prec1_rate = 1,
#'   prior_prec2_shape = 400,
#'   prior_prec2_rate = 1,
#'   prior_prec3_shape = 1600,
#'   prior_prec3_rate = 1,
#'   verbose = TRUE,
#'   seed = 456
#' )
#'
#' # Complete dashboard (10 pages)
#' plot(out, type = "all", engine = "base")
#'
#' # Diagnostics for specific parameters
#' plot(out, type = "mcmc", which = 1:3)  # Initial states
#' plot(out, type = "mcmc", which = 4:6)  # Innovation precisions
#'
#' # Bernoulli probabilities only
#' plot(out, type = "alpha")
#'
#' # Dynamic states only
#' plot(out, type = "states")
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
#' @seealso \code{\link{mcmc_probit_bernoulli_localacceleration}},
#'   \code{\link{summary.probit_bernoulli_localacceleration}}
#'
#' @export
plot.probit_bernoulli_localacceleration <- function(x,
                                                    type = c("all", "mcmc", "states", "alpha"),
                                                    which = NULL,
                                                    engine = c("base", "ggplot2"),
                                                    ask = NULL,
                                                    ci = TRUE,
                                                    ci_level = 0.95,
                                                    ...) {

  type <- match.arg(type)
  engine <- match.arg(engine)

  if (engine == "ggplot2" && !requireNamespace("ggplot2", quietly = TRUE)) {
    warning("Package 'ggplot2' is not installed. Falling back to base graphics.")
    engine <- "base"
  }

  if (is.null(ask)) {
    ask <- interactive() && type == "all"
  }

  if (engine == "base") {
    switch(type,
           all = {
             oldpar <- par(no.readonly = TRUE)
             on.exit(par(oldpar))
             if (ask) {
               oldask <- par(ask = TRUE)
               on.exit(par(oldask), add = TRUE)
             }
             plot_mcmc_diagnostics_generic(x, which = NULL, engine = "base", ...)
             plot_dynamic_states_generic_base(x, which = NULL, ci = ci,
                                              ci_level = ci_level, ...)
             plot_bernoulli_alpha_base(x, ci = ci, ci_level = ci_level, ...)
           },
           mcmc = plot_mcmc_diagnostics_generic(x, which = which,
                                                engine = "base", ...),
           states = plot_dynamic_states_generic_base(x, which = which,
                                                     ci = ci, ci_level = ci_level, ...),
           alpha = plot_bernoulli_alpha_base(x, ci = ci, ci_level = ci_level, ...)
    )
  } else {
    switch(type,
           all = {
             if (ask) {
               message("Press [Enter] to see next plot...")
             }
             plot_mcmc_diagnostics_generic(x, which = NULL, engine = "ggplot2", ...)
             if (ask) readline()
             plot_dynamic_states_generic_ggplot(x, which = NULL, ci = ci,
                                                ci_level = ci_level, ...)
             if (ask) readline()
             plot_bernoulli_alpha_ggplot(x, ci = ci, ci_level = ci_level, ...)
           },
           mcmc = plot_mcmc_diagnostics_generic(x, which = which,
                                                engine = "ggplot2", ...),
           states = plot_dynamic_states_generic_ggplot(x, which = which,
                                                       ci = ci, ci_level = ci_level, ...),
           alpha = plot_bernoulli_alpha_ggplot(x, ci = ci, ci_level = ci_level, ...)
    )
  }

  invisible(x)
}


#' Plot Bernoulli probabilities (base graphics)
#'
#' @keywords internal
#' @noRd
plot_bernoulli_alpha_base <- function(x, ci = TRUE, ci_level = 0.95, ...) {

  if (ci && (!is.numeric(ci_level) || length(ci_level) != 1 ||
             ci_level <= 0 || ci_level >= 1)) {
    stop("`ci_level` must be a single numeric value between 0 and 1")
  }

  n_obs <- attr(x, "n_obs")
  time_grid <- seq_len(n_obs)

  # Compute summary statistics for alpha
  alpha_median <- apply(x$alpha, 2, stats::median)
  if (ci) {
    ci_lower_prob <- (1 - ci_level) / 2
    ci_upper_prob <- 1 - ci_lower_prob
    alpha_lower <- apply(x$alpha, 2, stats::quantile, probs = ci_lower_prob)
    alpha_upper <- apply(x$alpha, 2, stats::quantile, probs = ci_upper_prob)
    ci_label <- paste0(round(ci_level * 100), "% CI")
  }

  # Get observed binary outcomes
  y <- attr(x, "y")

  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar), add = TRUE)

  par(mfrow = c(1, 1), mar = c(4, 4, 2, 1), oma = c(0, 0, 2, 0),
      mgp = c(2.5, 1, 0))

  # Determine y-axis range
  if (ci) {
    range_vals <- range(alpha_lower, alpha_upper, na.rm = TRUE)
  } else {
    range_vals <- range(alpha_median, na.rm = TRUE)
  }
  range_vals[1] <- max(0, range_vals[1] - 0.05)
  range_vals[2] <- min(1, range_vals[2] + 0.25 * diff(range_vals))

  # Base plot
  plot(time_grid, alpha_median, type = "l", lwd = 2.5, col = "blue",
       xlab = "Time", ylab = expression(alpha[t]),
       ylim = range_vals, axes = FALSE, main = "")

  axis(side = 1)
  axis(side = 2, at = seq(0, 1, by = 0.2))

  # Add credible band
  if (ci) {
    polygon(c(time_grid, rev(time_grid)),
            c(alpha_lower, rev(alpha_upper)),
            col = grDevices::rgb(0.2, 0.5, 0.8, 0.3), border = NA)
    lines(time_grid, alpha_median, lwd = 2.5, col = "blue")
  }

  # Add observed binary outcomes
  if (!is.null(y)) {
    # Plot y=1 and y=0 with different colors
    points(time_grid[y == 1], rep(1.0, sum(y == 1)),
           pch = 16, cex = 0.6, col = "darkgreen")
    points(time_grid[y == 0], rep(0.0, sum(y == 0)),
           pch = 16, cex = 0.6, col = "red")
  }

  grid()

  # Legend
  legend_items <- c(expression(hat(alpha)[t]))
  legend_cols <- c("blue")
  legend_lty <- c(1)
  legend_lwd <- c(2.5)
  legend_pch <- c(NA)

  if (ci) {
    legend_items <- c(legend_items, ci_label)
    legend_cols <- c(legend_cols, grDevices::rgb(0.2, 0.5, 0.8, 0.3))
    legend_lty <- c(legend_lty, 1)
    legend_lwd <- c(legend_lwd, 10)
    legend_pch <- c(legend_pch, NA)
  }

  if (!is.null(y)) {
    legend_items <- c(legend_items, "y = 1", "y = 0")
    legend_cols <- c(legend_cols, "darkgreen", "red")
    legend_lty <- c(legend_lty, NA, NA)
    legend_lwd <- c(legend_lwd, NA, NA)
    legend_pch <- c(legend_pch, 16, 16)
  }

  legend("topright",
         legend = legend_items,
         col = legend_cols,
         lty = legend_lty,
         lwd = legend_lwd,
         pch = legend_pch,
         horiz = FALSE,
         bty = "n")

  mtext("Bernoulli Probabilities", outer = TRUE, cex = 1.3, font = 2)

  invisible(NULL)
}


#' Plot Bernoulli probabilities (ggplot2)
#'
#' @keywords internal
#' @noRd
plot_bernoulli_alpha_ggplot <- function(x, ci = TRUE, ci_level = 0.95, ...) {

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for ggplot2 engine")
  }

  if (ci && (!is.numeric(ci_level) || length(ci_level) != 1 ||
             ci_level <= 0 || ci_level >= 1)) {
    stop("`ci_level` must be a single numeric value between 0 and 1")
  }

  n_obs <- attr(x, "n_obs")
  time_grid <- seq_len(n_obs)

  # Calculate quantile probabilities
  ci_lower_prob <- (1 - ci_level) / 2
  ci_upper_prob <- 1 - ci_lower_prob
  ci_pct <- round(ci_level * 100)

  # Prepare data frame for alpha
  df_alpha <- data.frame(
    time = time_grid,
    median = apply(x$alpha, 2, stats::median)
  )

  if (ci) {
    df_alpha$lower <- apply(x$alpha, 2, stats::quantile, probs = ci_lower_prob)
    df_alpha$upper <- apply(x$alpha, 2, stats::quantile, probs = ci_upper_prob)
  }

  # Get observed binary outcomes
  y <- attr(x, "y")

  # Create plot
  p <- ggplot2::ggplot(df_alpha, ggplot2::aes(x = .data$time))

  if (ci) {
    p <- p +
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

  p <- p +
    ggplot2::geom_line(
      ggplot2::aes(y = .data$median, colour = "Median"),
      linewidth = 1.2
    ) +
    ggplot2::scale_color_manual(
      values = c("Median" = "blue", "y=1" = "darkgreen", "y=0" = "red"),
      breaks = c("Median", "y=1", "y=0"),
      labels = c(expression(hat(alpha)[t]), "y = 1", "y = 0")
    )

  # Add observed binary outcomes if available
  if (!is.null(y)) {
    df_obs <- data.frame(
      time = time_grid,
      y = y,
      y_pos = ifelse(y == 1, 1.0, 0.0),
      y_label = factor(ifelse(y == 1, "y=1", "y=0"),
                       levels = c("y=1", "y=0"))
    )

    p <- p +
      ggplot2::geom_point(
        data = df_obs,
        ggplot2::aes(y = .data$y_pos, colour = .data$y_label),
        size = 1.5
      )
  }

  p <- p +
    ggplot2::scale_y_continuous(
      limits = c(0, 1.0),
      breaks = seq(0, 1, by = 0.2)
    ) +
    ggplot2::labs(
      title = "Bernoulli Probabilities",
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
    p <- p +
      ggplot2::guides(
        colour = ggplot2::guide_legend(order = 1),
        fill = ggplot2::guide_legend(order = 2)
      )
  }

  print(p)

  invisible(NULL)
}
