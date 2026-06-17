#' Summary method for normal_locallevel objects
#'
#' @description Produces comprehensive posterior statistics including means,
#'   standard deviations, medians, and credible intervals.
#'
#' @param object An object of class \code{normal_locallevel}, typically the
#'   result of calling \code{\link{mcmc_normal_locallevel}}.
#' @param probs Numeric vector of probabilities for credible intervals.
#'   Default is \code{c(0.025, 0.975)} for 95\% credible intervals.
#' @param ... Additional arguments (currently unused).
#'
#' @return An object of class \code{summary.normal_locallevel}, which is a list
#'   containing:
#'   \describe{
#'     \item{\code{model_type}}{Character string indicating the model type}
#'     \item{\code{n_obs}}{Number of observations}
#'     \item{\code{n_chain}}{Number of MCMC samples}
#'     \item{\code{burnin}}{Number of burn-in iterations}
#'     \item{\code{thinning}}{Thinning interval}
#'     \item{\code{probs}}{Probabilities used for credible intervals}
#'     \item{\code{scalar_params}}{Data frame with summary statistics for
#'       scalar parameters (theta_01, W_1^{-1}, V^{-1})}
#'     \item{\code{theta_summary}}{Summary statistics for the latent level
#'       \eqn{\theta_{t,1}} aggregated across time (min, median, max, and final
#'       time point)}
#'   }
#'
#' @details
#' This method provides complete posterior inference with multiple statistics:
#' \describe{
#'   \item{\strong{Mean}}{Expected value under the posterior (minimizes squared error loss)}
#'   \item{\strong{Median}}{Typical value (minimizes absolute error loss, shown in \code{print()})}
#'   \item{\strong{SD}}{Posterior standard deviation (measure of uncertainty)}
#'   \item{\strong{CI}}{Credible intervals at specified probabilities}
#' }
#'
#' For a quick overview showing only medians, use \code{print(x)}.
#'
#' For time-varying parameters like \eqn{\theta_{t,1}}, only summary statistics
#' across time are reported. Use \code{plot()} to visualize the full
#' trajectories.
#'
#' @examples
#' \donttest{
#' ## Description
#' # This example demonstrates how to:
#' # 1. Simulate data from a local-level dynamic model
#' # 2. Use `mcmc_normal_locallevel` to estimate parameters and latent states
#' # 3. Perform a detailed posterior analysis with visualizations
#' # 4. Set a seed for reproducibility
#'
#' ## Simulation of data
#' n <- 1000  # Number of observations to simulate
#'
#' # True parameters for simulation:
#' theta0_true <- 10  # Initial state (theta[01])
#' prec1_true <- 1    # Innovation precision (1/W[1])
#' prec_y_true <- 5   # Observation precision (1/V)
#'
#' # Use a fixed seed for data simulation
#' set.seed(123)
#'
#' # Generate noise terms:
#' u1 <- rnorm(n, sd = sqrt(1/prec1_true))  # Evolution noise (u1[t])
#' e  <- rnorm(n, sd = sqrt(1/prec_y_true)) # Observation noise (e[t])
#'
#' # Simulate latent states and observations:
#' theta1_true <- cumsum(c(theta0_true, u1))[-1]  # theta[t1] series
#' y <- theta1_true + e                           # Observed data (y[t])
#'
#' ## Running the Gibbs sampler
#' out <- mcmc_normal_locallevel(
#'   y,
#'   burnin   = 1000,
#'   thinning = 10,
#'   n_chain  = 1000,
#'   prior_theta01_mean = y[1],
#'   prior_theta01_prec = 1 / var(y),
#'   prior_prec1_shape  = 1e-2,
#'   prior_prec1_rate   = 1e-2,
#'   prior_prec_y_shape = 1e-2,
#'   prior_prec_y_rate  = 1e-2,
#'   verbose            = TRUE,
#'   bar_width          = 60,
#'   seed = 456
#' )
#'
#' # Quick overview (medians only)
#' print(out)
#'
#' # Full statistics
#' summary(out)
#'
#' # Custom credible intervals
#' summary(out, probs = c(0.05, 0.95))  # 90% CI
#' summary(out, probs = c(0.10, 0.90))  # 80% CI
#' }
#'
#' @seealso \code{\link{mcmc_normal_locallevel}},
#'   \code{\link{print.normal_locallevel}}
#'
#' @export
summary.normal_locallevel <- function(object,
                                      probs = c(0.025, 0.975),
                                      ...) {

  # Validate input
  validate_summary_input(object, probs, "normal_locallevel")

  # Scalar parameters (initial state and precisions)
  scalar_params <- format_scalar_params_locallevel(object, probs)

  # Latent level summary aggregated across time (detailed version)
  theta_summary <- format_timevarying_summary(object$theta_1, probs, "detailed")

  # Create summary object
  result <- list(
    model_type = attr(object, "model_type"),
    n_obs = attr(object, "n_obs"),
    n_chain = attr(object, "n_chain"),
    burnin = attr(object, "burnin"),
    thinning = attr(object, "thinning"),
    probs = probs,
    scalar_params = scalar_params,
    theta_summary = theta_summary
  )

  class(result) <- "summary.normal_locallevel"
  return(result)
}


#' Print method for summary.normal_locallevel objects
#'
#' @description Prints comprehensive posterior statistics in a readable format.
#'
#' @param x An object of class \code{summary.normal_locallevel}, typically the
#'   result of calling \code{summary()} on a \code{normal_locallevel} object.
#' @param digits Integer, number of significant digits to display. Default is 3.
#' @param ... Additional arguments (currently unused).
#'
#' @return Invisibly returns the input object \code{x}.
#'
#' @examples
#' \donttest{
#' ## Simulation of data (see ?mcmc_normal_locallevel for full details)
#' n <- 1000
#' theta0_true <- 10
#' prec1_true <- 1
#' prec_y_true <- 5
#' set.seed(123)
#' u1 <- rnorm(n, sd = sqrt(1 / prec1_true))
#' e  <- rnorm(n, sd = sqrt(1 / prec_y_true))
#' theta1_true <- cumsum(c(theta0_true, u1))[-1]
#' y <- theta1_true + e
#'
#' out <- mcmc_normal_locallevel(
#'   y,
#'   burnin             = 1000,
#'   thinning           = 10,
#'   n_chain            = 1000,
#'   prior_theta01_mean = y[1],
#'   prior_theta01_prec = 1 / var(y),
#'   prior_prec1_shape  = 1e-2,
#'   prior_prec1_rate   = 1e-2,
#'   prior_prec_y_shape = 1e-2,
#'   prior_prec_y_rate  = 1e-2,
#'   verbose            = TRUE,
#'   bar_width          = 60,
#'   seed               = 456
#' )
#'
#' s <- summary(out)
#' print(s)
#' # or simply:
#' summary(out)
#' }
#'
#' @seealso \code{\link{summary.normal_locallevel}},
#'   \code{\link{print.normal_locallevel}}
#'
#' @export
print.summary.normal_locallevel <- function(x, digits = 3, ...) {

  # Print header
  print_summary_header(x)

  # Scalar parameters
  print_formatted_table(x$scalar_params, digits, "Scalar Parameters:")

  # Latent level summary
  print_formatted_table(x$theta_summary, digits, "Latent Level (theta_{t,1}) Summary:")

  # Print footer
  print_summary_footer()

  invisible(x)
}
