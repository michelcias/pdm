#' Constructor for poisson_mixture_localacceleration class
#'
#' @description Internal constructor function for creating objects of class
#'   `poisson_mixture_localacceleration`. This function is called by
#'   \code{\link{mcmc_poisson_mixture_localacceleration}} and should not be
#'   called directly by users.
#'
#' @param result List containing MCMC results returned by the C function.
#' @param link Character string specifying the link function used ("logit" or "probit").
#' @param n_obs Integer, number of observations in the original data.
#' @param n_draws Integer, number of MCMC samples retained after burn-in and thinning.
#' @param burnin Integer, number of burn-in iterations.
#' @param thinning Integer, thinning interval.
#' @param y Numeric vector of original observed counts.
#'
#' @return An object of class
#'   `c("poisson_mixture_localacceleration", "pdm_mcmc", "list")`
#'   with the following structure:
#'   \describe{
#'     \item{Data components}{All elements from `result` (lambda_1, lambda_2,
#'       theta_1, theta_2, theta_3, theta_01, theta_02, theta_03, prec_theta1,
#'       prec_theta2, prec_theta3, alpha, z, and optionally log_sigma and
#'       accept_prop)}
#'     \item{Attributes}{
#'       \itemize{
#'         \item `link`: Link function used
#'         \item `n_obs`: Number of observations
#'         \item `n_draws`: Number of MCMC samples
#'         \item `burnin`: Burn-in iterations
#'         \item `thinning`: Thinning interval
#'         \item `model_type`: "localacceleration" (polynomial order 3)
#'         \item `y`: Original observed data
#'       }
#'     }
#'   }
#'
#' @details This constructor adds class attributes and metadata to the raw MCMC
#'   output, enabling the use of S3 methods like `summary()`, `plot()`,
#'   and `print()`.
#'
#'   The class hierarchy is:
#'   \itemize{
#'     \item `poisson_mixture_localacceleration`: Specific model class
#'     \item `pdm_mcmc`: General MCMC class for the pdm package
#'     \item `list`: Base R list class
#'   }
#'
#' @keywords internal
#' @noRd
new_poisson_mixture_localacceleration <- function(result,
                                                  link,
                                                  n_obs,
                                                  n_draws,
                                                  burnin,
                                                  thinning,
                                                  y) {

  # Validate that result is a non-empty list
  if (!is.list(result) || length(result) == 0) {
    stop("Internal error: result must be a non-empty list")
  }

  # Validate link function
  if (!is.character(link) || length(link) != 1 || !link %in% c("logit", "probit")) {
    stop("Internal error: link must be either 'logit' or 'probit'")
  }

  # Validate metadata types before adding as attributes
  if (!is.numeric(n_obs) || length(n_obs) != 1 || n_obs <= 0) {
    stop("Internal error: n_obs must be a positive scalar")
  }
  if (!is.numeric(n_draws) || length(n_draws) != 1 || n_draws <= 0) {
    stop("Internal error: n_draws must be a positive scalar")
  }
  if (!is.numeric(burnin) || length(burnin) != 1 || burnin < 0) {
    stop("Internal error: burnin must be a non-negative scalar")
  }
  if (!is.numeric(thinning) || length(thinning) != 1 || thinning < 1) {
    stop("Internal error: thinning must be a positive scalar >= 1")
  }
  if (!is.numeric(y) || length(y) != n_obs) {
    stop("Internal error: y must be a numeric vector of length n_obs")
  }

  # Add class hierarchy
  class(result) <- c("poisson_mixture_localacceleration", "pdm_mcmc", "list")

  # Add metadata as attributes
  attr(result, "link") <- link
  attr(result, "n_obs") <- as.integer(n_obs)
  attr(result, "n_draws") <- as.integer(n_draws)
  attr(result, "burnin") <- as.integer(burnin)
  attr(result, "thinning") <- as.integer(thinning)
  attr(result, "model_type") <- "localacceleration"  # Polynomial order 3
  attr(result, "y") <- y  # Store original data for plotting

  return(result)
}


#' Validator for poisson_mixture_localacceleration class
#'
#' @description Internal function to validate objects of class
#'   `poisson_mixture_localacceleration`. Checks that all required components
#'   are present and have correct dimensions.
#'
#' @param x An object to validate.
#'
#' @return The input object `x` if validation succeeds.
#' @keywords internal
#' @noRd
validate_poisson_mixture_localacceleration <- function(x) {
  validate_poisson_mixture(x, "poisson_mixture_localacceleration", order = 3L)
}


#' Check if object is of class poisson_mixture_localacceleration
#'
#' @description Test whether an object is of class
#'   `poisson_mixture_localacceleration`.
#'
#' @param x An object to test.
#'
#' @return Logical value: `TRUE` if `x` inherits from
#'   `poisson_mixture_localacceleration`, `FALSE` otherwise.
#'
#' @examples
#' ## A minimal fit is all this test needs; see
#' ## ?mcmc_poisson_mixture_localacceleration for a realistic analysis.
#' set.seed(123)
#' n <- 40
#' z <- rbinom(n, 1, plogis(cumsum(rnorm(n, sd = 0.2))))
#' y <- rpois(n, ifelse(z == 1, 8, 1))
#'
#' out <- mcmc_poisson_mixture_localacceleration(
#'   y,
#'   link     = "logit",
#'   burnin   = 50,
#'   thinning = 1,
#'   n_draws  = 50,
#'   verbose  = FALSE,
#'   seed     = 456
#' )
#'
#' is.poisson_mixture_localacceleration(out)     # TRUE
#' is.poisson_mixture_localacceleration(list())  # FALSE
#'
#' @export
is.poisson_mixture_localacceleration <- function(x) {
  inherits(x, "poisson_mixture_localacceleration")
}


#' Print method for poisson_mixture_localacceleration objects
#'
#' @description Prints a concise summary showing posterior medians.
#'   Use `summary()` for means, standard deviations, and credible intervals.
#'
#' @param x An object of class `poisson_mixture_localacceleration`.
#' @param digits Integer, number of decimal places to display. Default is 3.
#' @param ... Additional arguments (currently unused).
#'
#' @return Invisibly returns the input object `x`.
#'
#' @details This method provides a quick overview using posterior medians,
#'   which are robust to outliers and skewness in the posterior distribution.
#'
#'   For comprehensive statistics including means, standard deviations, and
#'   credible intervals, use `summary(x)`.
#'
#'   \strong{Why medians?}
#'   \itemize{
#'     \item Robust to outliers and long tails
#'     \item More representative for skewed posteriors (e.g., rate parameters)
#'     \item Minimizes absolute error loss
#'     \item Less sensitive to incomplete MCMC convergence
#'   }
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
#' lambda_y <- (1 - z_true) * 2 + z_true * 10
#' y <- rpois(n, lambda = lambda_y)
#'
#' ## Running the Gibbs sampler with logit link
#' out_logit <- mcmc_poisson_mixture_localacceleration(
#'   y,
#'   link                    = "logit",
#'   burnin                  = 1000,
#'   thinning                = 10,
#'   n_draws                 = 500,
#'   prior_theta01_prec      = 1,
#'   prior_theta02_prec      = 1,
#'   prior_theta03_prec      = 1,
#'   prior_prec1_shape       = 100,
#'   prior_prec1_rate        = 1,
#'   prior_prec2_shape       = 100,
#'   prior_prec2_rate        = 1,
#'   prior_prec3_shape       = 100,
#'   prior_prec3_rate        = 1,
#'   verbose                 = TRUE,
#'   bar_width               = 60,
#'   seed                    = 456
#' )
#'
#' # Quick overview (medians only)
#' print(out_logit)
#' # or simply:
#' out_logit
#'
#' # For full statistics with means and credible intervals:
#' summary(out_logit)
#' }
#'
#' @seealso
#'   \code{\link{mcmc_poisson_mixture_localacceleration}} (model generator),
#'   \code{\link{plot.poisson_mixture_localacceleration}},
#'   \code{\link{summary.poisson_mixture_localacceleration}}.
#'
#' @export
print.poisson_mixture_localacceleration <- function(x, digits = 3, ...) {
  print_poisson_mixture(x, "poisson_mixture_localacceleration", digits, order = 3L)
}
