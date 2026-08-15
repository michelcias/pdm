#' Summary method for poisson_mixture_localtrend objects
#'
#' @description Produces comprehensive posterior statistics including means,
#'   standard deviations, medians, and credible intervals.
#'
#' @param object An object of class `poisson_mixture_localtrend`, typically
#'   the result of calling \code{\link{mcmc_poisson_mixture_localtrend}}.
#' @param ci_level Credible interval level; a single numeric value strictly
#'   between 0 and 1. Defaults to `0.95`. The reported interval is the
#'   Highest Posterior Density Interval (HPDI), i.e. the shortest contiguous
#'   interval containing that probability mass of the posterior.
#' @param ... Additional arguments (currently unused).
#'
#' @return An object of class `summary.poisson_mixture_localtrend`, which is
#'   a list containing:
#'   \describe{
#'     \item{\code{link}}{Character string indicating the link function used}
#'     \item{\code{title}}{Character string naming the model family, used by the
#'       printed header}
#'     \item{\code{model_type}}{Character string indicating the model type}
#'     \item{\code{n_obs}}{Number of observations}
#'     \item{\code{n_draws}}{Number of MCMC samples}
#'     \item{\code{burnin}}{Number of burn-in iterations}
#'     \item{\code{thinning}}{Thinning interval}
#'     \item{\code{ci_level}}{Credible interval level used (HPDI)}
#'     \item{\code{mixture_params}}{Data frame with summary statistics for the
#'       component rates (lambda_1, lambda_2) and their difference}
#'     \item{\code{state_params}}{Data frame with summary statistics for
#'       dynamic state parameters (theta_01, theta_02, W_1^-1, W_2^-1)}
#'     \item{\code{alpha_summary}}{Summary statistics for the mixture weights
#'       alpha_t (min, median, max across time)}
#'   }
#'
#' @details
#' This method provides complete posterior inference with multiple statistics:
#' \describe{
#'   \item{\strong{Mean}}{Expected value under the posterior (minimizes squared error loss)}
#'   \item{\strong{Median}}{Typical value (minimizes absolute error loss, shown in `print()`)}
#'   \item{\strong{SD}}{Posterior standard deviation (measure of uncertainty)}
#'   \item{\strong{CI}}{Credible intervals at specified probabilities}
#' }
#'
#' For a quick overview showing only medians, use `print(x)`.
#'
#' For time-varying parameters like alpha_t, only summary statistics across
#' time are reported. Use `plot()` to visualize the full trajectories.
#'
#' @examples
#' \donttest{
#' ## Simulation of data
#' n <- 200  # Number of observations to simulate
#'
#' # Use a fixed seed for data simulation
#' set.seed(123)
#'
#' # Generate true mixture weights with a slow drift from low to high
#' grid_vals <- seq_len(n) / n
#' alpha_true <- plogis(-3 + 6 * grid_vals)
#'
#' # Generate latent component indicators
#' z_true <- rbinom(n, size = 1, prob = alpha_true)
#'
#' # Generate counts from a mixture of Poisson(2) and Poisson(10)
#' lambda_y <- (1 - z_true) * 2 + z_true * 10
#' y <- rpois(n, lambda = lambda_y)
#'
#' ## Running the Gibbs sampler with logit link
#' out_logit <- mcmc_poisson_mixture_localtrend(
#'   y,
#'   link               = "logit",
#'   burnin             = 1000,
#'   thinning           = 10,
#'   n_draws            = 500,
#'   prior_theta01_prec = 1,
#'   prior_theta02_prec = 1,
#'   prior_prec1_shape  = 100,
#'   prior_prec1_rate   = 1,
#'   prior_prec2_shape  = 100,
#'   prior_prec2_rate   = 1,
#'   verbose            = FALSE,
#'   seed               = 456
#' )
#'
#' # Quick overview (medians only)
#' print(out_logit)
#'
#' # Full statistics
#' summary(out_logit)
#'
#' # Custom credible intervals
#' summary(out_logit, ci_level = 0.90)  # 90% HPD interval
#' summary(out_logit, ci_level = 0.80)  # 80% HPD interval
#' }
#'
#' @seealso
#'   \code{\link{mcmc_poisson_mixture_localtrend}} (model generator),
#'   \code{\link{plot.poisson_mixture_localtrend}}, \code{\link{print.poisson_mixture_localtrend}}.
#'
#' @export
summary.poisson_mixture_localtrend <- function(object,
                                               ci_level = 0.95,
                                               ...) {

  summary_poisson_mixture(object, ci_level,
                          cls   = "poisson_mixture_localtrend",
                          order = "trend")
}


#' Print method for summary.poisson_mixture_localtrend objects
#'
#' @description Prints comprehensive posterior statistics in a readable format.
#'
#' @param x An object of class `summary.poisson_mixture_localtrend`, typically
#'   the result of calling `summary()` on a `poisson_mixture_localtrend`
#'   object.
#' @param digits Integer, number of significant digits to display. Default is 3.
#' @param ... Additional arguments (currently unused).
#'
#' @return Invisibly returns the input object `x`.
#'
#' @examples
#' \donttest{
#' ## Simulation of data
#' n <- 200  # Number of observations to simulate
#'
#' # Use a fixed seed for data simulation
#' set.seed(123)
#'
#' # Generate true mixture weights with a slow drift from low to high
#' grid_vals <- seq_len(n) / n
#' alpha_true <- plogis(-3 + 6 * grid_vals)
#'
#' # Generate latent component indicators
#' z_true <- rbinom(n, size = 1, prob = alpha_true)
#'
#' # Generate counts from a mixture of Poisson(2) and Poisson(10)
#' lambda_y <- (1 - z_true) * 2 + z_true * 10
#' y <- rpois(n, lambda = lambda_y)
#'
#' ## Running the Gibbs sampler with logit link
#' out_logit <- mcmc_poisson_mixture_localtrend(
#'   y,
#'   link               = "logit",
#'   burnin             = 1000,
#'   thinning           = 10,
#'   n_draws            = 500,
#'   prior_theta01_prec = 1,
#'   prior_theta02_prec = 1,
#'   prior_prec1_shape  = 100,
#'   prior_prec1_rate   = 1,
#'   prior_prec2_shape  = 100,
#'   prior_prec2_rate   = 1,
#'   verbose            = FALSE,
#'   seed               = 456
#' )
#'
#' s <- summary(out_logit)
#' print(s)
#' # or simply:
#' summary(out_logit)
#' }
#'
#' @export
print.summary.poisson_mixture_localtrend <- function(x, digits = 3, ...) {
  print_summary_poisson_mixture(x, digits)
}
