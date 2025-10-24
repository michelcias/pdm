#' Constructor for normal_mixture_localtrend class
#'
#' @description Internal constructor function for creating objects of class
#'   \code{normal_mixture_localtrend}. This function is called by
#'   \code{\link{mcmc_normal_mixture_localtrend}} and should not be called
#'   directly by users.
#'
#' @param result List containing MCMC results returned by the C function.
#' @param link Character string specifying the link function used ("logit" or "probit").
#' @param n_obs Integer, number of observations in the original data.
#' @param n_chain Integer, number of MCMC samples retained after burn-in and thinning.
#' @param burnin Integer, number of burn-in iterations.
#' @param thinning Integer, thinning interval.
#' @param y Numeric vector of original observed data.
#'
#' @return An object of class \code{c("normal_mixture_localtrend", "pdm_mcmc", "list")}
#'   with the following structure:
#'   \describe{
#'     \item{Data components}{All elements from \code{result} (mu_1, mu_2, prec_1,
#'       prec_2, theta_1, theta_2, theta_01, theta_02, prec_theta1, prec_theta2,
#'       alpha, z, and optionally log_sigma and accept_prop)}
#'     \item{Attributes}{
#'       \itemize{
#'         \item \code{link}: Link function used
#'         \item \code{n_obs}: Number of observations
#'         \item \code{n_chain}: Number of MCMC samples
#'         \item \code{burnin}: Burn-in iterations
#'         \item \code{thinning}: Thinning interval
#'         \item \code{model_type}: "localtrend" (polynomial order 2)
#'         \item \code{y}: Original observed data
#'       }
#'     }
#'   }
#'
#' @details This constructor adds class attributes and metadata to the raw MCMC
#'   output, enabling the use of S3 methods like \code{summary()}, \code{plot()},
#'   and \code{print()}.
#'
#'   The class hierarchy is:
#'   \itemize{
#'     \item \code{normal_mixture_localtrend}: Specific model class
#'     \item \code{pdm_mcmc}: General MCMC class for the pdm package
#'     \item \code{list}: Base R list class
#'   }
#'
#' @keywords internal
#' @noRd
new_normal_mixture_localtrend <- function(result,
                                          link,
                                          n_obs,
                                          n_chain,
                                          burnin,
                                          thinning,
                                          y) {

  # Validate that result is a list
  if (!is.list(result)) {
    stop("Internal error: result must be a list")
  }

  # Add class hierarchy
  class(result) <- c("normal_mixture_localtrend", "pdm_mcmc", "list")

  # Add metadata as attributes
  attr(result, "link") <- link
  attr(result, "n_obs") <- n_obs
  attr(result, "n_chain") <- n_chain
  attr(result, "burnin") <- burnin
  attr(result, "thinning") <- thinning
  attr(result, "model_type") <- "localtrend"  # Polynomial order 2
  attr(result, "y") <- y  # Store original data for plotting

  return(result)
}


#' Validator for normal_mixture_localtrend class
#'
#' @description Internal function to validate objects of class
#'   \code{normal_mixture_localtrend}. Checks that all required components
#'   are present and have correct dimensions.
#'
#' @param x An object to validate.
#'
#' @return The input object \code{x} if validation succeeds.
#' @keywords internal
#' @noRd
validate_normal_mixture_localtrend <- function(x) {

  # Check class
  if (!inherits(x, "normal_mixture_localtrend")) {
    stop("Object must inherit from class 'normal_mixture_localtrend'")
  }

  # Required components
  required_components <- c("mu_1", "mu_2", "prec_1", "prec_2",
                           "theta_1", "theta_2", "theta_01", "theta_02",
                           "prec_theta1", "prec_theta2", "alpha", "z")

  missing <- setdiff(required_components, names(x))
  if (length(missing) > 0) {
    stop("Missing required components: ", paste(missing, collapse = ", "))
  }

  # Check dimensions
  n_chain <- as.integer(attr(x, "n_chain"))
  n_obs <- as.integer(attr(x, "n_obs"))

  # Scalar parameters should have length n_chain
  scalar_params <- c("mu_1", "mu_2", "prec_1", "prec_2",
                     "theta_01", "theta_02", "prec_theta1", "prec_theta2")

  for (param in scalar_params) {
    if (length(x[[param]]) != n_chain) {
      stop(sprintf("Component '%s' should have length %d but has length %d",
                   param, n_chain, length(x[[param]])))
    }
  }

  # Matrix parameters should be n_chain x n_obs
  matrix_params <- c("theta_1", "theta_2", "alpha", "z")

  for (param in matrix_params) {
    if (!is.matrix(x[[param]])) {
      stop(sprintf("Component '%s' must be a matrix", param))
    }

    dims <- dim(x[[param]])

    # Check if dimensions match (comparing values, not types)
    # Use == instead of identical() to avoid integer vs numeric issues
    is_correct_order <- (dims[1] == n_chain && dims[2] == n_obs)
    is_transposed <- (dims[1] == n_obs && dims[2] == n_chain)

    if (!is_correct_order && !is_transposed) {
      stop(sprintf("Component '%s' has dimensions [%d x %d], expected [%d x %d]",
                   param, dims[1], dims[2], n_chain, n_obs))
    }

    # If dimensions are transposed, transpose back to [n_chain x n_obs]
    if (is_transposed) {
      x[[param]] <- t(x[[param]])
    }
  }

  # Check link attribute
  link <- attr(x, "link")
  if (!link %in% c("logit", "probit")) {
    stop("Attribute 'link' must be either 'logit' or 'probit'")
  }

  # If logit, optional components may be present
  if (link == "logit") {
    if (!is.null(x$log_sigma)) {
      if (!is.matrix(x$log_sigma)) {
        stop("Component 'log_sigma' must be a matrix")
      }
      dims <- dim(x$log_sigma)
      is_correct_order <- (dims[1] == n_chain && dims[2] == n_obs)
      is_transposed <- (dims[1] == n_obs && dims[2] == n_chain)

      if (!is_correct_order && !is_transposed) {
        stop(sprintf("Component 'log_sigma' has dimensions [%d x %d], expected [%d x %d]",
                     dims[1], dims[2], n_chain, n_obs))
      }

      if (is_transposed) {
        x$log_sigma <- t(x$log_sigma)
      }
    }

    if (!is.null(x$accept_prop)) {
      if (!is.matrix(x$accept_prop)) {
        stop("Component 'accept_prop' must be a matrix")
      }
      dims <- dim(x$accept_prop)
      is_correct_order <- (dims[1] == n_chain && dims[2] == n_obs)
      is_transposed <- (dims[1] == n_obs && dims[2] == n_chain)

      if (!is_correct_order && !is_transposed) {
        stop(sprintf("Component 'accept_prop' has dimensions [%d x %d], expected [%d x %d]",
                     dims[1], dims[2], n_chain, n_obs))
      }

      if (is_transposed) {
        x$accept_prop <- t(x$accept_prop)
      }
    }
  }

  return(x)
}


#' Check if object is of class normal_mixture_localtrend
#'
#' @description Test whether an object is of class \code{normal_mixture_localtrend}.
#'
#' @param x An object to test.
#'
#' @return Logical value: \code{TRUE} if \code{x} inherits from
#'   \code{normal_mixture_localtrend}, \code{FALSE} otherwise.
#'
#' @examples
#' \dontrun{
#' # After running mcmc_normal_mixture_localtrend
#' is.normal_mixture_localtrend(out_logit)  # TRUE
#' is.normal_mixture_localtrend(list())     # FALSE
#' }
#'
#' @export
is.normal_mixture_localtrend <- function(x) {
  inherits(x, "normal_mixture_localtrend")
}


#' Print method for normal_mixture_localtrend objects
#'
#' @description Prints a concise summary of a \code{normal_mixture_localtrend} object.
#'
#' @param x An object of class \code{normal_mixture_localtrend}.
#' @param ... Additional arguments (currently unused).
#'
#' @return Invisibly returns the input object \code{x}.
#'
#' @examples
#' \dontrun{
#' out_logit <- mcmc_normal_mixture_localtrend(y, link = "logit", ...)
#' print(out_logit)
#' # or simply:
#' out_logit
#' }
#'
#' @export
print.normal_mixture_localtrend <- function(x, ...) {

  cat("\n")
  cat("Gaussian Mixture Model with Dynamic Mixture Weights\n")
  cat(strrep("=", 55), "\n\n", sep = "")

  cat("Model type:        Local Trend (2nd order polynomial)\n")
  cat("Link function:     ", attr(x, "link"), "\n", sep = "")
  cat("Observations:      ", attr(x, "n_obs"), "\n", sep = "")
  cat("MCMC samples:      ", attr(x, "n_chain"), "\n", sep = "")
  cat("Burn-in:           ", attr(x, "burnin"), "\n", sep = "")
  cat("Thinning:          ", attr(x, "thinning"), "\n\n", sep = "")

  cat("Posterior medians:\n")
  cat("  mu_1:   ", sprintf("%.3f", median(x$mu_1)), "\n", sep = "")
  cat("  mu_2:   ", sprintf("%.3f", median(x$mu_2)), "\n", sep = "")
  cat("  phi_1:  ", sprintf("%.3f", median(x$prec_1)), "\n", sep = "")
  cat("  phi_2:  ", sprintf("%.3f", median(x$prec_2)), "\n\n", sep = "")

  cat("Use summary() for detailed statistics\n")
  cat("Use plot() for diagnostic plots\n\n")

  invisible(x)
}
