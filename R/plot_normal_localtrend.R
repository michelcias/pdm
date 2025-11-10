#' Plot method for normal_localtrend objects
#'
#' @description Produces diagnostic plots for MCMC output from Gaussian
#'   local trend models.
#'
#' @param x An object of class \code{normal_localtrend}.
#' @param type Character string specifying the type of plot. One of:
#'   \describe{
#'     \item{\code{"all"}}{Complete dashboard with all diagnostic plots (default)}
#'     \item{\code{"mcmc"}}{MCMC convergence diagnostics (trace plots, ACF, running means)}
#'     \item{\code{"states"}}{Dynamic states (theta_1, theta_2 trajectories)}
#'   }
#' @param which Integer vector specifying which diagnostic plots to display.
#'   For \code{type = "mcmc"}:
#'   \describe{
#'     \item{1}{V^{-1} (observation precision)}
#'     \item{2}{theta_01 (initial level)}
#'     \item{3}{theta_02 (initial trend)}
#'     \item{4}{W_1^{-1} (level innovation precision)}
#'     \item{5}{W_2^{-1} (trend innovation precision)}
#'   }
#'   For \code{type = "states"}: indices of subplots.
#'   If \code{NULL} (default), all available plots are shown.
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
#' Available parameters: V^{-1}, theta_01, theta_02, W_1^{-1}, W_2^{-1}
#'
#' \strong{Dynamic States (\code{type = "states"}):}
#' \itemize{
#'   \item Time-varying state trajectories with credible bands
#'   \item Innovation sequences
#'   \item State space relationships
#' }
#'
#' \strong{Complete Dashboard (\code{type = "all"}):}
#'
#' Generates 7 pages in total:
#' \itemize{
#'   \item Pages 1-5: Individual parameter diagnostics (4 panels each)
#'   \item Pages 6-7: Dynamic state trajectories and diagnostics
#' }
#' @examples
#' \dontrun{
#' ## Simulation of data
#' set.seed(123)
#' n <- 1000
#'
#' # True parameters
#' theta01_true <- 10
#' theta02_true <- 0.5
#' prec1_true <- 1 / 0.10
#' prec2_true <- 1 / 0.01
#' prec_y_true <- 1 / 1.00
#'
#' # Generate noise
#' u1 <- rnorm(n, sd = sqrt(1 / prec1_true))
#' u2 <- rnorm(n, sd = sqrt(1 / prec2_true))
#' epsilon <- rnorm(n, sd = sqrt(1 / prec_y_true))
#'
#' # Simulate states
#' theta1_true <- numeric(n)
#' theta2_true <- numeric(n)
#' theta2_true[1] <- theta02_true + u2[1]
#' theta1_true[1] <- theta01_true + theta02_true + u1[1]
#' for (t in 2:n) {
#'   theta2_true[t] <- theta2_true[t-1] + u2[t]
#'   theta1_true[t] <- theta1_true[t-1] + theta2_true[t-1] + u1[t]
#' }
#' y <- theta1_true + epsilon
#'
#' ## Running the Gibbs sampler
#' out <- mcmc_normal_localtrend(
#'   y,
#'   burnin = 2000,
#'   thinning = 100,
#'   n_chain = 1000,
#'   prior_theta01_mean = y[1] / 2,
#'   prior_theta01_prec = 1 / var(y),
#'   prior_theta02_mean = y[1] / 2,
#'   prior_theta02_prec = 1 / var(y),
#'   prior_prec1_shape = 1e-1,
#'   prior_prec1_rate = 1e-1,
#'   prior_prec2_shape = 1e-2,
#'   prior_prec2_rate = 1e-2,
#'   prior_prec_y_shape = 1e-1,
#'   prior_prec_y_rate = 1e-1,
#'   verbose = TRUE,
#'   seed = 456
#' )
#'
#' # Complete dashboard (7 pages)
#' plot(out, type = "all")
#'
#' # Diagnostics for specific parameters
#' plot(out, type = "mcmc", which = 1)  # Only V^{-1}
#' plot(out, type = "mcmc", which = 2:3)  # Only theta_01 and theta_02
#'
#' # Dynamic states only
#' plot(out, type = "states")
#'
#' # Save to multi-page PDF
#' pdf("diagnostics.pdf", width = 10, height = 8)
#' plot(out, type = "all", ask = FALSE)
#' dev.off()
#' }
#'
#' @seealso \code{\link{mcmc_normal_localtrend}},
#'   \code{\link{summary.normal_localtrend}}
#'
#' @export
plot.normal_localtrend <- function(x,
                                   type = c("all", "mcmc", "states"),
                                   which = NULL,
                                   ask = NULL,
                                   ci = TRUE,
                                   ci_level = 0.95,
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
           plot_mcmc_diagnostics_generic(x, which = NULL, ...)
           plot_dynamic_states_generic_base(x, which = NULL, ci = ci,
                                            ci_level = ci_level, ...)
         },
         mcmc = plot_mcmc_diagnostics_generic(x, which = which, ...),
         states = plot_dynamic_states_generic_base(x, which = which,
                                                   ci = ci, ci_level = ci_level, ...)
  )

  invisible(x)
}
