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
#' plot(out_logit, type = "all")
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
                                           ask = NULL,
                                           overlay_data = TRUE,
                                           ci = TRUE,
                                           ci_level = 0.95,
                                           ...) {

  type <- match.arg(type)

  if (is.null(ask)) {
    ask <- interactive() && type == "all"
  }

  switch(type,
         all = plot_all_mixture_generic_base(x, ask = ask, ci = ci,
                                             ci_level = ci_level, ...),
         mcmc = plot_mcmc_diagnostics_generic(x, which = which, ...),
         params = plot_mixture_params_base(x$mu_1, x$mu_2, x$prec_1, x$prec_2,
                                           which = which, ...),
         states = plot_dynamic_states_generic_base(x, which = which,
                                                   ci = ci, ci_level = ci_level, ...),
         alpha = plot_mixture_weights_base(x$alpha, x$z,
                                           ci = ci, ci_level = ci_level, ...)
  )

  invisible(x)
}
