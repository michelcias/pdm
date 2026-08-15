#' Summary method for poisson_mixture_locallevel objects
#'
#' @description Produces comprehensive posterior statistics including means,
#'   standard deviations, medians, and credible intervals.
#'
#' @param object An object of class `poisson_mixture_locallevel`, typically
#'   the result of calling \code{\link{mcmc_poisson_mixture_locallevel}}.
#' @param ci_level Credible interval level; a single numeric value strictly
#'   between 0 and 1. Defaults to `0.95`. The reported interval is the
#'   Highest Posterior Density Interval (HPDI), i.e. the shortest contiguous
#'   interval containing that probability mass of the posterior.
#' @param ... Additional arguments (currently unused).
#'
#' @return An object of class `summary.poisson_mixture_locallevel`, which is
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
#'       dynamic state parameters (theta_01, W_1^-1)}
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
#' # Generate true mixture weights following a sinusoidal pattern
#' grid_vals <- seq_len(n) / n
#' alpha_true <- (sin(2 * pi * grid_vals) + 2) / 4
#'
#' # Generate latent component indicators
#' z_true <- rbinom(n, size = 1, prob = alpha_true)
#'
#' # Generate counts from a mixture of Poisson(2) and Poisson(10)
#' lambda_y <- (1 - z_true) * 2 + z_true * 10
#' y <- rpois(n, lambda = lambda_y)
#'
#' ## Running the Gibbs sampler with logit link
#' out_logit <- mcmc_poisson_mixture_locallevel(
#'   y,
#'   link               = "logit",
#'   burnin             = 1000,
#'   thinning           = 10,
#'   n_draws            = 500,
#'   prior_theta01_prec = 1,
#'   prior_prec1_shape  = 100,
#'   prior_prec1_rate   = 1,
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
#'   \code{\link{mcmc_poisson_mixture_locallevel}} (model generator),
#'   \code{\link{plot.poisson_mixture_locallevel}}, \code{\link{print.poisson_mixture_locallevel}}.
#'
#' @export
summary.poisson_mixture_locallevel <- function(object,
                                               ci_level = 0.95,
                                               ...) {

  summary_poisson_mixture(object, ci_level,
                          cls   = "poisson_mixture_locallevel",
                          order = "level")
}


#' Print method for summary.poisson_mixture_locallevel objects
#'
#' @description Prints comprehensive posterior statistics in a readable format.
#'
#' @param x An object of class `summary.poisson_mixture_locallevel`, typically
#'   the result of calling `summary()` on a `poisson_mixture_locallevel`
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
#' # Generate true mixture weights following a sinusoidal pattern
#' grid_vals <- seq_len(n) / n
#' alpha_true <- (sin(2 * pi * grid_vals) + 2) / 4
#'
#' # Generate latent component indicators
#' z_true <- rbinom(n, size = 1, prob = alpha_true)
#'
#' # Generate counts from a mixture of Poisson(2) and Poisson(10)
#' lambda_y <- (1 - z_true) * 2 + z_true * 10
#' y <- rpois(n, lambda = lambda_y)
#'
#' ## Running the Gibbs sampler with logit link
#' out_logit <- mcmc_poisson_mixture_locallevel(
#'   y,
#'   link               = "logit",
#'   burnin             = 1000,
#'   thinning           = 10,
#'   n_draws            = 500,
#'   prior_theta01_prec = 1,
#'   prior_prec1_shape  = 100,
#'   prior_prec1_rate   = 1,
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
print.summary.poisson_mixture_locallevel <- function(x, digits = 3, ...) {
  print_summary_poisson_mixture(x, digits)
}


#' Build the summary object for any Poisson mixture fit
#'
#' The three orders differ only in which state parameters they carry, and
#' `format_state_params()` already takes that as an argument. Sharing the body
#' keeps the three `summary()` methods from drifting apart in the table they
#' assemble.
#'
#' @param object The fitted object.
#' @param ci_level Credible level.
#' @param cls Expected class, for the validation message.
#' @param order One of `"level"`, `"trend"`, `"acceleration"`.
#'
#' @return A `summary.<cls>` object.
#'
#' @keywords internal
#' @noRd
summary_poisson_mixture <- function(object, ci_level, cls, order) {

  # Validate input
  validate_summary_input(object, ci_level, cls)

  result <- list(
    link = attr(object, "link"),
    title = "Poisson Mixture Model with Dynamic Mixture Weights",
    model_type = attr(object, "model_type"),
    n_obs = attr(object, "n_obs"),
    n_draws = attr(object, "n_draws"),
    burnin = attr(object, "burnin"),
    thinning = attr(object, "thinning"),
    ci_level = ci_level,
    mixture_params = format_poisson_mixture_params(object, ci_level),
    state_params = format_state_params(object, ci_level, order = order),
    alpha_summary = format_timevarying_summary(object$alpha, ci_level, "simple")
  )

  class(result) <- paste0("summary.", cls)
  result
}


#' Print a Poisson mixture summary object
#'
#' @param x The summary object.
#' @param digits Integer, significant digits.
#'
#' @return `x`, invisibly.
#'
#' @keywords internal
#' @noRd
print_summary_poisson_mixture <- function(x, digits) {

  # Print header
  print_summary_header(x)

  # Mixture component rates
  print_formatted_table(x$mixture_params, digits, "Mixture Component Rates:")

  # Dynamic state parameters
  print_formatted_table(x$state_params, digits, "Dynamic State Parameters:")

  # Alpha summary
  print_formatted_table(x$alpha_summary, digits, "Mixture Weights (alpha_t) Summary:")

  # Print footer
  print_summary_footer()

  invisible(x)
}
