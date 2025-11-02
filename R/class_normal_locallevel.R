#' Constructor for normal_locallevel class
#'
#' @description Internal constructor function for creating objects of class
#'   \code{normal_locallevel}. This function is called by
#'   \code{\link{mcmc_normal_locallevel}} and should not be called directly by
#'   users.
#'
#' @param result List containing MCMC results returned by the C function.
#' @param n_obs Integer, number of observations in the original data.
#' @param n_chain Integer, number of MCMC samples retained after burn-in and thinning.
#' @param burnin Integer, number of burn-in iterations.
#' @param thinning Integer, thinning interval.
#' @param y Numeric vector of original observed data.
#'
#' @return An object of class \code{c("normal_locallevel", "pdm_mcmc", "list")}
#'   with the following structure:
#'   \describe{
#'     \item{Data components}{All elements from \code{result} (theta_1, theta_01,
#'       prec_theta1, prec_y)}
#'     \item{Attributes}{
#'       \itemize{
#'         \item \code{n_obs}: Number of observations
#'         \item \code{n_chain}: Number of MCMC samples
#'         \item \code{burnin}: Burn-in iterations
#'         \item \code{thinning}: Thinning interval
#'         \item \code{model_type}: \code{"locallevel"} (polynomial order 1)
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
#'     \item \code{normal_locallevel}: Specific model class
#'     \item \code{pdm_mcmc}: General MCMC class for the pdm package
#'     \item \code{list}: Base R list class
#'   }
#'
#' @keywords internal
#' @noRd
new_normal_locallevel <- function(result,
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
  class(result) <- c("normal_locallevel", "pdm_mcmc", "list")

  # Add metadata as attributes
  attr(result, "n_obs") <- n_obs
  attr(result, "n_chain") <- n_chain
  attr(result, "burnin") <- burnin
  attr(result, "thinning") <- thinning
  attr(result, "model_type") <- "locallevel"  # Polynomial order 1
  attr(result, "y") <- y  # Store original data for plotting

  return(result)
}


#' Validator for normal_locallevel class
#'
#' @description Internal function to validate objects of class
#'   \code{normal_locallevel}. Checks that all required components are present
#'   and have correct dimensions.
#'
#' @param x An object to validate.
#'
#' @return The input object \code{x} if validation succeeds.
#' @keywords internal
#' @noRd
validate_normal_locallevel <- function(x) {

  # Check class
  if (!inherits(x, "normal_locallevel")) {
    stop("Object must inherit from class 'normal_locallevel'")
  }

  # Required components
  required_components <- c("theta_1", "theta_01", "prec_theta1", "prec_y")

  missing <- setdiff(required_components, names(x))
  if (length(missing) > 0) {
    stop("Missing required components: ", paste(missing, collapse = ", "))
  }

  # Extract metadata
  n_chain <- as.integer(attr(x, "n_chain"))
  n_obs <- as.integer(attr(x, "n_obs"))

  if (is.na(n_chain) || n_chain <= 0) {
    stop("Attribute 'n_chain' must be a positive integer")
  }
  if (is.na(n_obs) || n_obs <= 0) {
    stop("Attribute 'n_obs' must be a positive integer")
  }

  # Scalar parameters should have length n_chain
  scalar_params <- c("theta_01", "prec_theta1", "prec_y")

  for (param in scalar_params) {
    if (length(x[[param]]) != n_chain) {
      stop(sprintf("Component '%s' should have length %d but has length %d",
                   param, n_chain, length(x[[param]])))
    }
  }

  # theta_1 should be n_chain x n_obs
  if (!is.matrix(x$theta_1)) {
    stop("Component 'theta_1' must be a matrix")
  }

  dims <- dim(x$theta_1)
  is_correct_order <- (dims[1] == n_chain && dims[2] == n_obs)
  is_transposed <- (dims[1] == n_obs && dims[2] == n_chain)

  if (!is_correct_order && !is_transposed) {
    stop(sprintf("Component 'theta_1' has dimensions [%d x %d], expected [%d x %d]",
                 dims[1], dims[2], n_chain, n_obs))
  }

  if (is_transposed) {
    x$theta_1 <- t(x$theta_1)
  }

  # Validate stored data attribute
  y <- attr(x, "y")
  if (!is.null(y)) {
    if (!is.numeric(y) || length(y) != n_obs) {
      stop("Attribute 'y' must be a numeric vector with length equal to 'n_obs'")
    }
  }

  # Model type attribute should be locallevel
  model_type <- attr(x, "model_type")
  if (!identical(model_type, "locallevel")) {
    stop("Attribute 'model_type' must be 'locallevel'")
  }

  return(x)
}


#' Check if object is of class normal_locallevel
#'
#' @description Test whether an object is of class \code{normal_locallevel}.
#'
#' @param x An object to test.
#'
#' @return Logical value: \code{TRUE} if \code{x} inherits from
#'   \code{normal_locallevel}, \code{FALSE} otherwise.
#'
#' @examples
#' \dontrun{
#' ## Simulate data (same setup as ?mcmc_normal_locallevel)
#' n <- 1000
#'
#' # True parameters for simulation
#' theta0_true <- 10
#' prec1_true <- 1
#' prec_y_true <- 5
#'
#' set.seed(123)
#' u1 <- rnorm(n, sd = sqrt(1 / prec1_true))
#' e  <- rnorm(n, sd = sqrt(1 / prec_y_true))
#' theta1_true <- cumsum(c(theta0_true, u1))[-1]
#' y <- theta1_true + e
#'
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
#'   seed = 456
#' )
#'
#' is.normal_locallevel(out)  # TRUE
#' is.normal_locallevel(list())  # FALSE
#' }
#'
#' @export
is.normal_locallevel <- function(x) {
  inherits(x, "normal_locallevel")
}


#' Print method for normal_locallevel objects
#'
#' @description Prints a concise summary showing posterior medians.
#'   Use \code{summary()} for comprehensive statistics when available.
#'
#' @param x An object of class \code{normal_locallevel}.
#' @param digits Integer, number of decimal places to display. Default is 3.
#' @param ... Additional arguments (currently unused).
#'
#' @return Invisibly returns the input object \code{x}.
#'
#' @details This method provides a quick overview using posterior medians,
#'   which are robust to outliers and skewness in the posterior distribution.
#'
#' @examples
#' \dontrun{
#' ## Simulate data (same setup as ?mcmc_normal_locallevel)
#' n <- 1000
#'
#' theta0_true <- 10
#' prec1_true <- 1
#' prec_y_true <- 5
#'
#' set.seed(123)
#' u1 <- rnorm(n, sd = sqrt(1 / prec1_true))
#' e  <- rnorm(n, sd = sqrt(1 / prec_y_true))
#' theta1_true <- cumsum(c(theta0_true, u1))[-1]
#' y <- theta1_true + e
#'
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
#'   seed = 456
#' )
#'
#' print(out)
#' }
#'
#' @seealso \code{\link{mcmc_normal_locallevel}}
#' @export
print.normal_locallevel <- function(x, digits = 3, ...) {

  cat("\n")
  cat("Gaussian Local-Level Model\n")
  cat(strrep("=", 70), "\n\n", sep = "")

  # Model metadata
  cat("Model:\n")
  cat("  Type:              ", attr(x, "model_type"),
      " (1st order polynomial)\n", sep = "")
  cat("\n")

  # MCMC metadata
  cat("MCMC:\n")
  cat("  Observations:      ", attr(x, "n_obs"), "\n", sep = "")
  cat("  Samples retained:  ", attr(x, "n_chain"), "\n", sep = "")
  cat("  Burn-in:           ", attr(x, "burnin"), "\n", sep = "")
  cat("  Thinning:          ", attr(x, "thinning"), "\n\n", sep = "")

  # Calculate medians
  med_theta01 <- median(x$theta_01)
  med_prec1 <- median(x$prec_theta1)
  med_precy <- median(x$prec_y)

  # Determine field width for alignment (width = digits + 4 for sign, decimal, padding)
  field_width <- digits + 4

  # Posterior medians for scalar parameters
  cat("Posterior Medians (Scalars):\n")
  cat("  theta_01:  ", sprintf(paste0("%", field_width, ".", digits, "f"), med_theta01),
      "  (initial level)\n", sep = "")
  cat("  W_1^-1:    ", sprintf(paste0("%", field_width, ".", digits, "f"), med_prec1),
      "  (level innovation precision)\n", sep = "")
  cat("  V^-1:      ", sprintf(paste0("%", field_width, ".", digits, "f"), med_precy),
      "  (observation precision)\n\n", sep = "")

  # Median trajectory summary for theta_1
  theta_1_median <- apply(x$theta_1, 2, median)
  theta_1_min <- min(theta_1_median)
  theta_1_max <- max(theta_1_median)
  theta_1_final <- theta_1_median[length(theta_1_median)]

  cat("Latent Level (theta_t) Median Summary:\n")
  cat("  Range:  [",
      sprintf(paste0("%", field_width, ".", digits, "f"), theta_1_min),
      ", ",
      sprintf(paste0("%", field_width, ".", digits, "f"), theta_1_max),
      "]\n", sep = "")
  cat("  Final:  ",
      sprintf(paste0("%", field_width, ".", digits, "f"), theta_1_final),
      "  (median level at last time point)\n\n", sep = "")

  # User guidance
  cat(strrep("-", 70), "\n", sep = "")
  cat("Note: Showing posterior medians (robust central tendency).\n")
  cat("      For additional summaries: summary(x) when available.\n")
  cat("      For visual diagnostics: plot(x) if implemented.\n")
  cat(strrep("-", 70), "\n\n", sep = "")

  invisible(x)
}
