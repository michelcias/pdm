#' Summary method for normal_mixture_localtrend objects
#'
#' @description Produces comprehensive posterior statistics including means,
#'   standard deviations, medians, and credible intervals.
#'
#' @param object An object of class `normal_mixture_localtrend`, typically
#'   the result of calling \code{\link{mcmc_normal_mixture_localtrend}}.
#' @param ci_level Credible interval level; a single numeric value strictly
#'   between 0 and 1. Defaults to `0.95`. The reported interval is the
#'   Highest Posterior Density Interval (HPDI), i.e. the shortest contiguous
#'   interval containing that probability mass of the posterior.
#' @param ... Additional arguments (currently unused).
#'
#' @return An object of class `summary.normal_mixture_localtrend`, which is
#'   a list containing:
#'   \describe{
#'     \item{\code{link}}{Character string indicating the link function used}
#'     \item{\code{model_type}}{Character string indicating the model type}
#'     \item{\code{n_obs}}{Number of observations}
#'     \item{\code{n_chain}}{Number of MCMC samples}
#'     \item{\code{burnin}}{Number of burn-in iterations}
#'     \item{\code{thinning}}{Thinning interval}
#'     \item{\code{ci_level}}{Credible interval level used (HPDI)}
#'     \item{\code{mixture_params}}{Data frame with summary statistics for
#'       mixture component parameters (mu_1, mu_2, phi_1, phi_2)}
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
#'   max_step_size      = 0.1,
#'   base_adaptation_rate = 1.0,
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
#' @seealso \code{\link{mcmc_normal_mixture_localtrend}},
#'   \code{\link{print.normal_mixture_localtrend}},
#'   \code{\link{plot.normal_mixture_localtrend}}
#'
#' @export
summary.normal_mixture_localtrend <- function(object,
                                              ci_level = 0.95,
                                              ...) {

  # Validate input
  validate_summary_input(object, ci_level, "normal_mixture_localtrend")

  # Mixture component parameters
  mixture_params <- format_mixture_params(object, ci_level)

  # Dynamic state parameters (local trend)
  state_params <- format_state_params(object, ci_level, order = "trend")

  # Alpha summary (across time)
  alpha_summary <- format_timevarying_summary(object$alpha, ci_level, "simple")

  # Create summary object
  result <- list(
    link = attr(object, "link"),
    model_type = attr(object, "model_type"),
    n_obs = attr(object, "n_obs"),
    n_chain = attr(object, "n_chain"),
    burnin = attr(object, "burnin"),
    thinning = attr(object, "thinning"),
    ci_level = ci_level,
    mixture_params = mixture_params,
    state_params = state_params,
    alpha_summary = alpha_summary
  )

  class(result) <- "summary.normal_mixture_localtrend"
  return(result)
}


#' Print method for summary.normal_mixture_localtrend objects
#'
#' @description Prints comprehensive posterior statistics in a readable format.
#'
#' @param x An object of class `summary.normal_mixture_localtrend`, typically
#'   the result of calling `summary()` on a `normal_mixture_localtrend`
#'   object.
#' @param digits Integer, number of significant digits to display. Default is 3.
#' @param ... Additional arguments (currently unused).
#'
#' @return Invisibly returns the input object `x`.
#'
#' @examples
#' \donttest{
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
#'   link                    = "logit",
#'   burnin                  = 2000,
#'   thinning                = 10,
#'   n_chain                 = 1000,
#'   prior_mu01_mean         = NULL,  # Use default (25th percentile)
#'   prior_mu01_prec         = 0.01,
#'   prior_prec01_shape      = 0.01,
#'   prior_prec01_rate       = 0.01,
#'   prior_mu02_mean         = NULL,  # Use default (75th percentile)
#'   prior_mu02_prec         = 0.01,
#'   prior_prec02_shape      = 0.01,
#'   prior_prec02_rate       = 0.01,
#'   prior_theta01_mean      = 0,
#'   prior_theta01_prec      = 1,
#'   prior_theta02_mean      = 0,
#'   prior_theta02_prec      = 1,
#'   prior_prec1_shape       = 100,
#'   prior_prec1_rate        = 1,
#'   prior_prec2_shape       = 400,
#'   prior_prec2_rate        = 1,
#'   lag_update              = 50,
#'   max_step_size           = 0.1,
#'   base_adaptation_rate    = 1.0,
#'   decay_exponent          = 0.6,
#'   target_acceptance       = 0.44,
#'   min_deviation_threshold = NULL,  # Use default (1/lag_update)
#'   return_log_sigma        = FALSE,
#'   return_accept_prop      = FALSE,
#'   verbose                 = TRUE,
#'   bar_width               = 60,
#'   seed                    = 456
#' )
#'
#' s <- summary(out_logit)
#' print(s)
#' # or simply:
#' summary(out_logit)
#' }
#'
#' @export
print.summary.normal_mixture_localtrend <- function(x, digits = 3, ...) {

  # Print header
  print_summary_header(x)

  # Mixture component parameters
  print_formatted_table(x$mixture_params, digits, "Mixture Component Parameters:")

  # Dynamic state parameters
  print_formatted_table(x$state_params, digits, "Dynamic State Parameters:")

  # Alpha summary
  print_formatted_table(x$alpha_summary, digits, "Mixture Weights (alpha_t) Summary:")

  # Print footer
  print_summary_footer()

  invisible(x)
}
