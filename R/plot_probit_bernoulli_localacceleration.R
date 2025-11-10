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
#' @param ask Logical; if \code{TRUE}, the user is asked before each plot when
#'   \code{type = "all"}. Default is \code{interactive()} when \code{type = "all"},
#'   \code{FALSE} otherwise.
#' @param ci Logical; whether to display credible intervals in plots that
#'   support them. Default is \code{TRUE}.
#' @param ci_level Numeric; Bayesian confidence level for credible intervals
#'   (between 0 and 1). Default is \code{0.95}.
#' @param true_values Named list containing true parameter values for comparison.
#'   Expected elements:
#'   \describe{
#'     \item{\code{theta_01}}{True initial level}
#'     \item{\code{theta_02}}{True initial trend}
#'     \item{\code{theta_03}}{True initial acceleration}
#'     \item{\code{prec_theta1}}{True level innovation precision (W_1^{-1})}
#'     \item{\code{prec_theta2}}{True trend innovation precision (W_2^{-1})}
#'     \item{\code{prec_theta3}}{True acceleration innovation precision (W_3^{-1})}
#'     \item{\code{theta_1}}{Numeric vector of true theta_1 states over time}
#'     \item{\code{theta_2}}{Numeric vector of true theta_2 states over time}
#'     \item{\code{theta_3}}{Numeric vector of true theta_3 states over time}
#'     \item{\code{alpha}}{Numeric vector of true alpha_t probabilities over time}
#'   }
#'   If \code{NULL} (default), no true values are displayed.
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
#' plot(out, type = "all")
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
#' }
#'
#' @seealso \code{\link{mcmc_probit_bernoulli_localacceleration}},
#'   \code{\link{summary.probit_bernoulli_localacceleration}}
#'
#' @export
plot.probit_bernoulli_localacceleration <- function(x,
                                                    type = c("all", "mcmc", "states", "alpha"),
                                                    which = NULL,
                                                    ask = NULL,
                                                    ci = TRUE,
                                                    ci_level = 0.95,
                                                    true_values = NULL,
                                                    ...) {

  type <- match.arg(type)

  if (is.null(ask)) {
    ask <- interactive() && type == "all"
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
           plot_bernoulli_alpha_base(x,
                                     ci = ci,
                                     ci_level = ci_level,
                                     true_values = true_values$alpha,
                                     ...)
         },
         mcmc = plot_mcmc_diagnostics_generic(x,
                                              which = which,
                                              true_values = true_values,
                                              ...),
         states = plot_dynamic_states_generic_base(x,
                                                   which = which,
                                                   ci = ci,
                                                   ci_level = ci_level,
                                                   true_values = true_values,
                                                   ...),
         alpha = plot_bernoulli_alpha_base(x,
                                           ci = ci,
                                           ci_level = ci_level,
                                           true_alpha = true_values$alpha,
                                           ...)
  )

  invisible(x)
}

