#' Plot method for poisson_mixture_localacceleration objects
#'
#' @description Produces diagnostic plots for MCMC output from Poisson mixture
#'   models with dynamic mixture weights.
#'
#' @param x An object of class `poisson_mixture_localacceleration`.
#' @param type Character string specifying the type of plot. One of:
#'   \describe{
#'     \item{\code{"all"}}{Complete dashboard with all diagnostic plots (default)}
#'     \item{\code{"mcmc"}}{MCMC convergence diagnostics (trace plots, ACF, running means)}
#'     \item{\code{"params"}}{Mixture component rates (lambda_1, lambda_2)}
#'     \item{\code{"states"}}{Dynamic states (theta_1, theta_2 and theta_3 trajectories)}
#'     \item{\code{"alpha"}}{Mixture weights over time (alpha_t and z_t)}
#'     \item{\code{"acceptance"}}{Metropolis-Hastings acceptance proportions (if available)}
#'   }
#' @param which Integer vector specifying which diagnostic plots to display.
#'   For `type = "mcmc"`:
#'   \describe{
#'     \item{1}{lambda_1 (component 1 rate)}
#'     \item{2}{lambda_2 (component 2 rate)}
#'     \item{3}{theta_01 (initial level)}
#'     \item{4}{theta_02 (initial trend)}
#'     \item{5}{theta_03 (initial acceleration)}
#'     \item{6}{W_1^-1 (level innovation precision)}
#'     \item{7}{W_2^-1 (trend innovation precision)}
#'     \item{8}{W_3^-1 (acceleration innovation precision)}
#'   }
#'   For `type = "params"`, `type = "states"`: indices of subplots.
#'   For `type = "alpha"` or `type = "acceptance"`: not used.
#'   If `NULL` (default), all available plots are shown.
#' @param ask Logical; if `TRUE`, the user is asked before each plot when
#'   `type = "all"`. Default is `interactive()` when `type = "all"`,
#'   `FALSE` otherwise.
#' @param overlay_data Logical; for `type = "alpha"`, whether to overlay
#'   the original data (if available). Default is `TRUE`.
#' @param ci Logical; whether to display credible intervals in plots that
#'   support them. Default is `TRUE`.
#' @param ci_level Numeric; Bayesian confidence level for credible intervals
#'   (between 0 and 1). Default is `0.95`.
#' @param true_values Named list containing true parameter values and/or state
#'   trajectories for comparison with MCMC estimates. Valid names for this model
#'   are `lambda_1`, `lambda_2`, `theta_01`, `theta_02`, `theta_03`,
#'   `prec_theta1`, `prec_theta2`, `prec_theta3`, `theta_1`, `theta_2`,
#'   `theta_3`, `alpha` and `z`. If `NULL` (default), no true values are
#'   displayed.
#' @param ... Additional arguments passed to the underlying plotting functions.
#'
#' @return Invisibly returns the input object `x`.
#'
#' @details The `"acceptance"` diagnostic exists only under the logit link,
#'   which samples the latent states with an adaptive Metropolis-Hastings step,
#'   and only when the fit was produced with `return_accept_prop = TRUE`. Under
#'   the probit link the sampler is pure Gibbs (Albert-Chib augmentation), so
#'   there is nothing to accept or reject; `type = "all"` skips the panel
#'   silently in both cases, while asking for it directly is an error that says
#'   which of the two applies.
#'
#' @examples
#' \donttest{
#' ## Simulation of data
#' n <- 200  # Number of observations to simulate
#'
#' # Use a fixed seed for data simulation
#' set.seed(123)
#'
#' # Generate true mixture weights with curvature (rise then fall)
#' grid_vals <- seq_len(n) / n
#' alpha_true <- plogis(-3 + 12 * grid_vals - 12 * grid_vals^2)
#'
#' # Generate latent component indicators
#' z_true <- rbinom(n, size = 1, prob = alpha_true)
#'
#' # Generate counts from a mixture of Poisson(2) and Poisson(10)
#' lambda_1_true <- 2
#' lambda_2_true <- 10
#' lambda_y <- (1 - z_true) * lambda_1_true + z_true * lambda_2_true
#' y <- rpois(n, lambda = lambda_y)
#'
#' ## Running the Gibbs sampler with logit link
#' out_logit <- mcmc_poisson_mixture_localacceleration(
#'   y,
#'   link               = "logit",
#'   burnin             = 1000,
#'   thinning           = 10,
#'   n_draws            = 500,
#'   prior_theta01_prec = 1,
#'   prior_theta02_prec = 1,
#'   prior_theta03_prec = 1,
#'   prior_prec1_shape  = 100,
#'   prior_prec1_rate   = 1,
#'   prior_prec2_shape  = 100,
#'   prior_prec2_rate   = 1,
#'   prior_prec3_shape  = 100,
#'   prior_prec3_rate   = 1,
#'   return_accept_prop = TRUE,
#'   verbose            = FALSE,
#'   seed               = 456
#' )
#'
#' ## Posterior analysis and visualization
#' # Use the plot method for comprehensive diagnostics
#' plot(out_logit, type = "all")                  # Complete dashboard
#' plot(out_logit, type = "mcmc", which = 1:8)    # Scalar-parameter diagnostics
#' plot(out_logit, type = "params")               # Component rates
#' plot(out_logit, type = "states")               # Dynamic states
#' plot(out_logit, type = "alpha")                # Mixture weights
#' plot(out_logit, type = "acceptance")           # Only if return_accept_prop = TRUE
#'
#' plot(out_logit, type = "params",
#'      true_values = list(lambda_1 = lambda_1_true, lambda_2 = lambda_2_true))
#' plot(out_logit, type = "alpha",
#'      true_values = list(alpha = alpha_true, z = z_true))
#' }
#'
#' @seealso
#'   \code{\link{mcmc_poisson_mixture_localacceleration}} (model generator),
#'   \code{\link{print.poisson_mixture_localacceleration}},
#'   \code{\link{summary.poisson_mixture_localacceleration}}.
#'
#' @export
plot.poisson_mixture_localacceleration <- function(x,
                                                   type = c("all", "mcmc", "params",
                                                            "states", "alpha", "acceptance"),
                                                   which = NULL,
                                                   ask = NULL,
                                                   overlay_data = TRUE,
                                                   ci = TRUE,
                                                   ci_level = 0.95,
                                                   true_values = NULL,
                                                   ...) {

  type <- match.arg(type)

  plot_poisson_mixture_dispatch(
    x, type, which, ask, overlay_data, ci, ci_level, true_values,
    generator = "mcmc_poisson_mixture_localacceleration", ...
  )
}
