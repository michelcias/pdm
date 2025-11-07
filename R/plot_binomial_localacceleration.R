#' Plot method for binomial_localacceleration objects
#'
#' @description Produces diagnostic plots for MCMC output from binomial
#'   local acceleration models with logit link.
#'
#' @param x An object of class \code{binomial_localacceleration}.
#' @param type Character string specifying the type of plot. One of:
#'   \describe{
#'     \item{\code{"all"}}{Complete dashboard with all diagnostic plots (default)}
#'     \item{\code{"mcmc"}}{MCMC convergence diagnostics (trace plots, ACF, running means)}
#'     \item{\code{"states"}}{Dynamic states (theta_1, theta_2, theta_3 trajectories)}
#'     \item{\code{"alpha"}}{Success probabilities over time (alpha_t)}
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
#' @param show_obs Logical; whether to display observed proportions on the
#'   success probabilities plot (\code{type = "alpha"}). When \code{TRUE}
#'   (default), observed proportions \eqn{y_t / n_{trials}} are overlaid as
#'   red points on the alpha_t trajectory. Set to \code{FALSE} to show only
#'   the estimated trajectory without observations. This parameter only
#'   affects \code{type = "alpha"} and \code{type = "all"}.
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
#'   \item Time-varying state trajectories with credible bands (on logit scale)
#'   \item Innovation sequences
#'   \item State space relationships
#' }
#'
#' \strong{Success Probabilities (\code{type = "alpha"}):}
#' \itemize{
#'   \item alpha_t trajectory with credible bands
#'   \item Optional observed proportions overlay (controlled by \code{show_obs})
#' }
#'
#' \strong{Complete Dashboard (\code{type = "all"}):}
#'
#' Generates 11 pages in total:
#' \itemize{
#'   \item Pages 1-6: Individual parameter diagnostics (4 panels each)
#'   \item Pages 7-9: Dynamic state trajectories and diagnostics
#'   \item Page 10: Success probabilities alpha_t
#'   \item Page 11: Acceptance rates (if available)
#' }
#'
#' The \code{engine} argument allows choosing between base R graphics (lightweight,
#' no dependencies) and ggplot2 (modern, publication-ready). If ggplot2 is not
#' installed and \code{engine = "ggplot2"}, the function falls back to base graphics
#' with a warning.
#'
#' @section Controlling Observed Data Display:
#'
#' The \code{show_obs} parameter provides control over the display of observed
#' proportions in the success probabilities plot:
#'
#' \itemize{
#'   \item When \code{show_obs = TRUE} (default): Observed proportions are shown
#'     as red points overlaid on the estimated alpha_t trajectory. This is useful
#'     for model validation and assessing goodness-of-fit.
#'   \item When \code{show_obs = FALSE}: Only the estimated trajectory is shown,
#'     which can be clearer for presentations or when focusing on the temporal
#'     pattern of the success probabilities.
#' }
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
#' n_trials <- 20
#'
#' # True parameters
#' theta01_true <- 0.5
#' theta02_true <- 0.01
#' theta03_true <- 0.001
#' prec1_true <- 100
#' prec2_true <- 400
#' prec3_true <- 1600
#'
#' # Generate noise
#' u1 <- rnorm(n, sd = sqrt(1/prec1_true))
#' u2 <- rnorm(n, sd = sqrt(1/prec2_true))
#' u3 <- rnorm(n, sd = sqrt(1/prec3_true))
#'
#' # Simulate states
#' theta1_true <- numeric(n)
#' theta2_true <- numeric(n)
#' theta3_true <- numeric(n)
#' theta3_true[1] <- theta03_true + u3[1]
#' theta2_true[1] <- theta02_true + theta03_true + u2[1]
#' theta1_true[1] <- theta01_true + theta02_true + u1[1]
#' for (t in 2:n) {
#'   theta3_true[t] <- theta3_true[t-1] + u3[t]
#'   theta2_true[t] <- theta2_true[t-1] + theta3_true[t-1] + u2[t]
#'   theta1_true[t] <- theta1_true[t-1] + theta2_true[t-1] + u1[t]
#' }
#' alpha_true <- plogis(theta1_true)
#' y <- rbinom(n, size = n_trials, prob = alpha_true)
#'
#' ## Running the Gibbs sampler
#' out <- mcmc_binomial_localacceleration(
#'   y,
#'   n_trials = n_trials,
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
#' # Complete dashboard (11 pages)
#' plot(out, type = "all", engine = "base")
#'
#' # Diagnostics for specific parameters
#' plot(out, type = "mcmc", which = 1:3)  # Initial states
#' plot(out, type = "mcmc", which = 4:6)  # Innovation precisions
#'
#' # Success probabilities with observed proportions (default)
#' plot(out, type = "alpha")
#'
#' # Success probabilities WITHOUT observed proportions
#' plot(out, type = "alpha", show_obs = FALSE)
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
#'
#' # Compare with and without observations using ggplot2
#' plot(out, type = "alpha", engine = "ggplot2", show_obs = TRUE)
#' plot(out, type = "alpha", engine = "ggplot2", show_obs = FALSE)
#' }
#'
#' @seealso \code{\link{mcmc_binomial_localacceleration}},
#'   \code{\link{summary.binomial_localacceleration}}
#'
#' @export
plot.binomial_localacceleration <- function(x,
                                            type = c("all", "mcmc", "states", "alpha"),
                                            which = NULL,
                                            engine = c("base", "ggplot2"),
                                            ask = NULL,
                                            ci = TRUE,
                                            ci_level = 0.95,
                                            show_obs = TRUE,
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
             plot_binomial_alpha_base(x, ci = ci, ci_level = ci_level,
                                      show_obs = show_obs, ...)
             # Plot acceptance rates if available
             if (!is.null(x$accept_prop)) {
               plot_acceptance_rates_base(x$accept_prop, ...)
             }
           },
           mcmc = plot_mcmc_diagnostics_generic(x, which = which,
                                                engine = "base", ...),
           states = plot_dynamic_states_generic_base(x, which = which,
                                                     ci = ci, ci_level = ci_level, ...),
           alpha = plot_binomial_alpha_base(x, ci = ci, ci_level = ci_level,
                                            show_obs = show_obs, ...)
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
             plot_binomial_alpha_ggplot(x, ci = ci, ci_level = ci_level,
                                        show_obs = show_obs, ...)
             # Plot acceptance rates if available
             if (!is.null(x$accept_prop)) {
               if (ask) readline()
               plot_acceptance_rates_ggplot(x$accept_prop, ...)
             }
           },
           mcmc = plot_mcmc_diagnostics_generic(x, which = which,
                                                engine = "ggplot2", ...),
           states = plot_dynamic_states_generic_ggplot(x, which = which,
                                                       ci = ci, ci_level = ci_level, ...),
           alpha = plot_binomial_alpha_ggplot(x, ci = ci, ci_level = ci_level,
                                              show_obs = show_obs, ...)
    )
  }

  invisible(x)
}


#' Plot acceptance rates (base graphics)
#'
#' @keywords internal
#' @noRd
plot_acceptance_rates_base <- function(accept_prop, ...) {

  min_acc <- apply(accept_prop, 2, min)
  max_acc <- apply(accept_prop, 2, max)
  med_acc <- apply(accept_prop, 2, median)

  range_acc <- range(min_acc, max_acc)
  r1_acc <- range_acc[1] - 0.05
  r2_acc <- range_acc[2] + 0.25 * diff(range_acc)

  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  par(mfrow = c(1, 1), mar = c(4, 4, 2, 1), oma = c(0, 0, 2, 0))

  plot(
    med_acc,
    type = "l",
    col = "black",
    lwd = 2,
    xlab = "Time (t)",
    ylab = "Acceptance Rate",
    ylim = c(r1_acc, r2_acc),
    main = ""
  )

  polygon(
    c(1:length(med_acc), rev(1:length(med_acc))),
    c(min_acc, rev(max_acc)),
    col = grDevices::rgb(0.7, 0.7, 0.7, alpha = 0.3),
    border = NA
  )

  abline(h = 0.44, col = "red", lty = 2, lwd = 2)

  legend(
    "topright",
    legend = c("Median acceptance", "Min-Max range", "Target (0.44)"),
    col = c("black", "gray", "red"),
    lty = c(1, 1, 2),
    lwd = c(2, 8, 2),
    horiz = TRUE,
    bty = "n"
  )

  grid()

  mtext("Metropolis-Hastings Acceptance Rates", outer = TRUE, cex = 1.3, font = 2)

  invisible(NULL)
}


#' Plot acceptance rates (ggplot2)
#'
#' @keywords internal
#' @noRd
plot_acceptance_rates_ggplot <- function(accept_prop, ...) {

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for ggplot2 engine")
  }

  df_acc <- data.frame(
    time = seq_len(ncol(accept_prop)),
    median = apply(accept_prop, 2, median),
    min = apply(accept_prop, 2, min),
    max = apply(accept_prop, 2, max)
  )

  p <- ggplot2::ggplot(df_acc, ggplot2::aes(x = .data$time)) +
    ggplot2::geom_ribbon(
      ggplot2::aes(ymin = .data$min, ymax = .data$max, fill = "Range"),
      alpha = 0.3
    ) +
    ggplot2::geom_line(
      ggplot2::aes(y = .data$median, colour = "Median"),
      linewidth = 1.2
    ) +
    ggplot2::geom_hline(
      ggplot2::aes(yintercept = 0.44, linetype = "Target"),
      color = "red",
      linewidth = 1
    ) +
    ggplot2::scale_colour_manual(
      values = c("Median" = "black"),
      breaks = "Median"
    ) +
    ggplot2::scale_fill_manual(
      values = c("Range" = "gray"),
      breaks = "Range",
      labels = "Min-Max range"
    ) +
    ggplot2::scale_linetype_manual(
      values = c("Target" = "dashed"),
      labels = "Target (0.44)"
    ) +
    ggplot2::labs(
      title = "Metropolis-Hastings Acceptance Rates",
      x = "Time (t)",
      y = "Acceptance Rate",
      colour = NULL,
      fill = NULL,
      linetype = NULL
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      panel.grid.major = ggplot2::element_line(color = "grey85"),
      panel.grid.minor = ggplot2::element_line(color = "grey92"),
      plot.title = ggplot2::element_text(face = "bold", size = 13),
      legend.position = "top",
      legend.direction = "horizontal"
    ) +
    ggplot2::guides(
      colour = ggplot2::guide_legend(order = 1),
      fill = ggplot2::guide_legend(order = 2),
      linetype = ggplot2::guide_legend(order = 3)
    )

  print(p)

  invisible(NULL)
}
