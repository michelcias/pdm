#' Plot method for normal_locallevel objects
#'
#' @description Produces diagnostic plots for MCMC output from Gaussian
#'   local level models.
#'
#' @param x An object of class \code{normal_locallevel}.
#' @param type Character string specifying the type of plot. One of:
#'   \describe{
#'     \item{\code{"all"}}{Complete dashboard with all diagnostic plots (default)}
#'     \item{\code{"mcmc"}}{MCMC convergence diagnostics (trace plots, ACF, running means)}
#'     \item{\code{"states"}}{Dynamic states (theta_1 trajectory)}
#'   }
#' @param which Integer vector specifying which diagnostic plots to display.
#'   For \code{type = "mcmc"}:
#'   \describe{
#'     \item{1}{phi_y (observation precision)}
#'     \item{2}{theta_01 (initial level)}
#'     \item{3}{W_1^{-1} (level innovation precision)}
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
#' Available parameters: phi_y, theta_01, W_1^{-1}
#'
#' \strong{Dynamic States (\code{type = "states"}):}
#' \itemize{
#'   \item Time-varying state trajectory with credible bands
#'   \item Innovation sequence
#' }
#'
#' \strong{Complete Dashboard (\code{type = "all"}):}
#'
#' Generates 5 pages in total:
#' \itemize{
#'   \item Pages 1-3: Individual parameter diagnostics (4 panels each)
#'   \item Pages 4-5: Dynamic state trajectory and diagnostics
#' }
#' @examples
#' \dontrun{
#' ## Simulation of data
#' set.seed(123)
#' n <- 100
#'
#' theta_true <- cumsum(rnorm(n, mean = 0, sd = 0.1))
#' y <- rnorm(n, mean = theta_true, sd = 0.5)
#'
#' ## Running the Gibbs sampler
#' out <- mcmc_normal_locallevel(
#'   y,
#'   burnin             = 1000,
#'   thinning           = 5,
#'   n_chain            = 1000,
#'   prior_theta01_mean = 0,
#'   prior_theta01_prec = 1,
#'   prior_prec_y_shape = 0.01,
#'   prior_prec_y_rate  = 0.01,
#'   prior_prec1_shape  = 0.01,
#'   prior_prec1_rate   = 0.01,
#'   verbose            = TRUE,
#'   seed               = 456
#' )
#'
#' # Complete dashboard (5 pages)
#' plot(out, type = "all")
#'
#' # Diagnostics for specific parameters
#' plot(out, type = "mcmc", which = 1)  # Only phi_y
#' plot(out, type = "mcmc", which = 2)  # Only theta_01
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
#' @seealso \code{\link{mcmc_normal_locallevel}},
#'   \code{\link{summary.normal_locallevel}}
#'
#' @export
plot.normal_locallevel <- function(x,
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
