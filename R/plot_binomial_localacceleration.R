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
#'     \item{\code{"acceptance"}}{Metropolis-Hastings acceptance proportions (if available)}
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
#'   For \code{type = "alpha"} or \code{type = "acceptance"}: not used.
#'   If \code{NULL} (default), all available plots are shown.
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
#' @param true_values Named list containing true scalar parameter values for comparison
#'   in MCMC diagnostic plots (\code{type = "mcmc"}) and alpha plots (\code{type = "alpha"}).
#'   Expected elements:
#'   \describe{
#'     \item{\code{theta_01}}{Scalar: True initial level}
#'     \item{\code{theta_02}}{Scalar: True initial trend}
#'     \item{\code{theta_03}}{Scalar: True initial acceleration}
#'     \item{\code{prec_theta1}}{Scalar: True level innovation precision (W_1^{-1})}
#'     \item{\code{prec_theta2}}{Scalar: True trend innovation precision (W_2^{-1})}
#'     \item{\code{prec_theta3}}{Scalar: True acceleration innovation precision (W_3^{-1})}
#'     \item{\code{alpha}}{Vector: True alpha_t probabilities over time (for \code{type = "alpha"})}
#'   }
#'   If \code{NULL} (default), no true values are displayed.
#' @param true_states Named list containing true dynamic state trajectories for comparison
#'   in state plots (\code{type = "states"}). Expected elements:
#'   \describe{
#'     \item{\code{theta_1}}{Vector: True theta_1 state values over time (length n_obs)}
#'     \item{\code{theta_2}}{Vector: True theta_2 state values over time (length n_obs)}
#'     \item{\code{theta_3}}{Vector: True theta_3 state values over time (length n_obs)}
#'   }
#'   If \code{NULL} (default), no true state trajectories are displayed.
#'
#'   \strong{Note:} If you only have the true \code{alpha}, you can obtain \code{theta_1}
#'   using \code{qlogis(alpha)}. However, \code{theta_2} (trend) and \code{theta_3}
#'   (acceleration) cannot be recovered from \code{alpha} alone. They must come from
#'   your simulation data.
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
#' \strong{Acceptance Proportions (\code{type = "acceptance"}):}
#' \itemize{
#'   \item Metropolis-Hastings acceptance proportions over time
#'   \item Min-Max range across MCMC iterations
#'   \item Target acceptance proportion reference line (uses the \code{target_acceptance}
#'     value specified in \code{mcmc_binomial_localacceleration})
#'   \item Only available if the model was run with \code{return_accept_prop = TRUE}
#' }
#'
#' \strong{Complete Dashboard (\code{type = "all"}):}
#'
#' Generates up to 11 pages in total:
#' \itemize{
#'   \item Pages 1-6: Individual parameter diagnostics (4 panels each)
#'   \item Pages 7-9: Dynamic state trajectories and diagnostics
#'   \item Page 10: Success probabilities alpha_t
#'   \item Page 11: Acceptance proportions (only if available)
#' }
#'
#' @section Target Acceptance Proportion:
#'
#' The acceptance proportion plot displays a reference line showing the target acceptance
#' proportion that was specified when running \code{mcmc_binomial_localacceleration}.
#' This allows visual assessment of whether the adaptive Metropolis-Hastings algorithm
#' successfully achieved the desired acceptance proportion. The target value is automatically
#' extracted from the model object and displayed in the plot legend.
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
#' @examples
#' \dontrun{
#' ## Simulation of data
#' set.seed(123)
#' n <- 500
#' n_trials <- 20
#'
#' # Generate true probabilities
#' grid_vals <- seq_len(n) / n
#' alpha_true <- (sin(4 * pi * grid_vals) + sin(8 * pi * grid_vals) + 2) / 4
#'
#' # Generate binomial observations
#' y <- rbinom(n, size = n_trials, prob = alpha_true)
#'
#' ## Running the Gibbs sampler with acceptance proportion tracking
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
#'   target_acceptance = 0.44,      # Specify target acceptance proportion
#'   return_accept_prop = TRUE,     # Enable acceptance proportion tracking
#'   verbose = TRUE,
#'   seed = 456
#' )
#'
#' # Complete dashboard (11 pages, includes acceptance proportions if available)
#' plot(out, type = "all")
#'
#' # Only acceptance proportions (will show target line at 0.44)
#' plot(out, type = "acceptance")
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
#' # Example with different target acceptance proportion
#' out2 <- mcmc_binomial_localacceleration(
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
#'   target_acceptance = 0.30,      # Different target
#'   return_accept_prop = TRUE,
#'   verbose = TRUE,
#'   seed = 789
#' )
#'
#' # Acceptance plot will now show target line at 0.30
#' plot(out2, type = "acceptance")
#'
#' # Save to multi-page PDF
#' pdf("diagnostics.pdf", width = 10, height = 8)
#' plot(out, type = "all", ask = FALSE)
#' dev.off()
#'
#' }
#'
#' @seealso \code{\link{mcmc_binomial_localacceleration}},
#'   \code{\link{summary.binomial_localacceleration}}
#'
#' @export
plot.binomial_localacceleration <- function(x,
                                            type = c("all", "mcmc", "states", "alpha", "acceptance"),
                                            which = NULL,
                                            ask = NULL,
                                            ci = TRUE,
                                            ci_level = 0.95,
                                            show_obs = TRUE,
                                            true_values = NULL,
                                            ...) {

  type <- match.arg(type)

  # Check if acceptance proportions are available only when specifically requested
  if (type == "acceptance" && is.null(x$accept_prop)) {
    stop("Acceptance proportions are not available. ",
         "Re-run mcmc_binomial_localacceleration() with return_accept_prop = TRUE.")
  }

  if (is.null(ask)) {
    ask <- interactive() && type == "all"
  }

  # Extract target_acceptance with fallback for backward compatibility
  target_acc <- attr(x, "target_acceptance")
  if (is.null(target_acc)) {
    target_acc <- 0.44  # Default fallback for objects created before this feature
  }

  switch(type,
         all = {
           oldpar <- par(no.readonly = TRUE)
           on.exit(par(oldpar))
           if (ask) {
             oldask <- par(ask = TRUE)
             on.exit(par(oldask), add = TRUE)
           }
           plot_mcmc_diagnostics_generic(x,
                                         which = NULL,
                                         true_values = true_values,
                                         ...)
           plot_dynamic_states_generic_base(x,
                                            which = NULL,
                                            ci = ci,
                                            ci_level = ci_level,
                                            true_values = true_values,
                                            ...)
           plot_binomial_alpha_base(x,
                                    ci = ci,
                                    ci_level = ci_level,
                                    show_obs = show_obs,
                                    true_alpha = true_values$alpha,
                                    ...)
           # Plot acceptance proportions only if available
           if (!is.null(x$accept_prop)) {
             plot_acceptance_proportions_base(x$accept_prop,
                                              target_acceptance = target_acc,
                                              ...)
           }
         },
         mcmc = plot_mcmc_diagnostics_generic(x, which = which,
                                              true_values = true_values,
                                              ...),
         states = plot_dynamic_states_generic_base(x,
                                                   which = which,
                                                   ci = ci,
                                                   ci_level = ci_level,
                                                   true_values = true_values,
                                                   ...),
         alpha = plot_binomial_alpha_base(x,
                                          ci = ci,
                                          ci_level = ci_level,
                                          show_obs = show_obs,
                                          true_alpha = true_values$alpha,
                                          ...),
         acceptance = {
           if (!is.null(x$accept_prop)) {
             plot_acceptance_proportions_base(x$accept_prop,
                                              target_acceptance = target_acc,
                                              ...)
           }
         }
  )

  invisible(x)
}

