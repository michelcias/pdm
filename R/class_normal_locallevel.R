#' Constructor for normal_locallevel class
#'
#' @description Internal constructor function for creating objects of class
#'   `normal_locallevel`. This function is called by
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
#' @return An object of class `c("normal_locallevel", "pdm_mcmc", "list")`
#'   with the following structure:
#'   \describe{
#'     \item{Data components}{All elements from `result` (theta_1, theta_01,
#'       prec_theta1, prec_y)}
#'     \item{Attributes}{
#'       \itemize{
#'         \item `n_obs`: Number of observations
#'         \item `n_chain`: Number of MCMC samples
#'         \item `burnin`: Burn-in iterations
#'         \item `thinning`: Thinning interval
#'         \item `model_type`: `"locallevel"` (polynomial order 1)
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
#'     \item `normal_locallevel`: Specific model class
#'     \item `pdm_mcmc`: General MCMC class for the pdm package
#'     \item `list`: Base R list class
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

  # Validate that result is a non-empty list
  if (!is.list(result) || length(result) == 0) {
    stop("Internal error: result must be a non-empty list")
  }

  # Validate metadata types before adding as attributes
  if (!is.numeric(n_obs) || length(n_obs) != 1 || n_obs <= 0) {
    stop("Internal error: n_obs must be a positive scalar")
  }
  if (!is.numeric(n_chain) || length(n_chain) != 1 || n_chain <= 0) {
    stop("Internal error: n_chain must be a positive scalar")
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
  class(result) <- c("normal_locallevel", "pdm_mcmc", "list")

  # Add metadata as attributes
  attr(result, "n_obs") <- as.integer(n_obs)
  attr(result, "n_chain") <- as.integer(n_chain)
  attr(result, "burnin") <- as.integer(burnin)
  attr(result, "thinning") <- as.integer(thinning)
  attr(result, "model_type") <- "locallevel"  # Polynomial order 1
  attr(result, "y") <- y  # Store original data for plotting

  return(result)
}


#' Validator for normal_locallevel class
#'
#' @description Internal function to validate objects of class
#'   `normal_locallevel`. Checks that all required components are present
#'   and have correct dimensions.
#'
#' @param x An object to validate.
#'
#' @return The input object `x` if validation succeeds.
#' @keywords internal
#' @noRd
validate_normal_locallevel <- function(x) {

  # 1. Check class
  if (!inherits(x, "normal_locallevel")) {
    stop("Object must inherit from class 'normal_locallevel'")
  }

  # 2. Check required components exist
  required_components <- c("theta_1", "theta_01", "prec_theta1", "prec_y")
  missing <- setdiff(required_components, names(x))
  if (length(missing) > 0) {
    stop("Missing required components: ", paste(missing, collapse = ", "))
  }

  # 3. Extract and validate metadata attributes
  n_chain_raw <- attr(x, "n_chain")
  n_obs_raw <- attr(x, "n_obs")

  if (is.null(n_chain_raw)) {
    stop("Missing required attribute 'n_chain'")
  }
  if (is.null(n_obs_raw)) {
    stop("Missing required attribute 'n_obs'")
  }

  n_chain <- as.integer(n_chain_raw)
  n_obs <- as.integer(n_obs_raw)

  if (length(n_chain) == 0 || is.na(n_chain) || n_chain <= 0) {
    stop("Attribute 'n_chain' must be a positive integer")
  }
  if (length(n_obs) == 0 || is.na(n_obs) || n_obs <= 0) {
    stop("Attribute 'n_obs' must be a positive integer")
  }

  # 4. Validate scalar parameters (type, length, finiteness, positivity)
  scalar_params <- c("theta_01", "prec_theta1", "prec_y")
  precision_params <- c("prec_theta1", "prec_y")

  for (param in scalar_params) {
    # Check type
    if (!is.numeric(x[[param]])) {
      stop(sprintf("Component '%s' must be numeric", param))
    }

    # Check length
    if (length(x[[param]]) != n_chain) {
      stop(sprintf(
        "Component '%s' should have length %d but has length %d",
        param, n_chain, length(x[[param]])
      ))
    }

    # Check finiteness
    if (any(!is.finite(x[[param]]))) {
      n_bad <- sum(!is.finite(x[[param]]))
      stop(sprintf(
        "Component '%s' contains %d non-finite values (NA, NaN, or Inf)",
        param, n_bad
      ))
    }

    # Check positivity for precision parameters
    if (param %in% precision_params) {
      if (any(x[[param]] <= 0)) {
        n_nonpositive <- sum(x[[param]] <= 0)
        stop(sprintf(
          "Component '%s' (precision) must be positive, but %d/%d values are non-positive",
          param, n_nonpositive, n_chain
        ))
      }
    }
  }

  # 5. Validate theta_1 matrix
  if (!is.matrix(x$theta_1)) {
    stop("Component 'theta_1' must be a matrix")
  }

  if (!is.numeric(x$theta_1)) {
    stop("Component 'theta_1' must be numeric")
  }

  dims <- dim(x$theta_1)

  if (dims[1] != n_chain || dims[2] != n_obs) {
    stop(sprintf(
      "Component 'theta_1' has incorrect dimensions [%d x %d], expected [%d x %d].\n  Each row should be one MCMC sample, each column one time point.",
      dims[1], dims[2], n_chain, n_obs
    ))
  }

  # Check finiteness
  if (any(!is.finite(x$theta_1))) {
    n_bad <- sum(!is.finite(x$theta_1))
    stop(sprintf(
      "Component 'theta_1' contains %d non-finite values (NA, NaN, or Inf)",
      n_bad
    ))
  }

  # 6. Validate stored data attribute (if present)
  y <- attr(x, "y")
  if (!is.null(y)) {
    if (!is.numeric(y)) {
      stop("Attribute 'y' must be numeric")
    }
    if (length(y) != n_obs) {
      stop(sprintf(
        "Attribute 'y' has length %d but should have length %d (n_obs)",
        length(y), n_obs
      ))
    }
    if (any(!is.finite(y))) {
      stop("Attribute 'y' contains non-finite values")
    }
  }

  # 7. Validate model type
  model_type <- attr(x, "model_type")
  if (is.null(model_type)) {
    stop("Missing required attribute 'model_type'")
  }
  if (!identical(model_type, "locallevel")) {
    stop(sprintf(
      "Attribute 'model_type' must be 'locallevel', got '%s'",
      as.character(model_type)
    ))
  }

  # 8. Validate optional metadata (if you use them elsewhere)
  burnin <- attr(x, "burnin")
  if (!is.null(burnin)) {
    if (!is.numeric(burnin) || length(burnin) != 1 || burnin < 0) {
      stop("Attribute 'burnin' must be a non-negative scalar")
    }
  }

  thinning <- attr(x, "thinning")
  if (!is.null(thinning)) {
    if (!is.numeric(thinning) || length(thinning) != 1 || thinning < 1) {
      stop("Attribute 'thinning' must be a positive scalar >= 1")
    }
  }

  return(x)
}


#' Check if object is of class normal_locallevel
#'
#' @description Test whether an object is of class `normal_locallevel`.
#'
#' @param x An object to test.
#'
#' @return Logical value: `TRUE` if `x` inherits from
#'   `normal_locallevel`, `FALSE` otherwise.
#'
#' @examples
#' ## A minimal fit is all this test needs; see
#' ## ?mcmc_normal_locallevel for a realistic analysis.
#' set.seed(123)
#' n <- 40
#' y <- cumsum(c(10, rnorm(n)))[-1] + rnorm(n, sd = 0.5)
#'
#' out <- mcmc_normal_locallevel(
#'   y,
#'   burnin             = 50,
#'   thinning           = 1,
#'   n_chain            = 50,
#'   prior_theta01_mean = y[1],
#'   prior_theta01_prec = 1 / var(y),
#'   prior_prec1_shape  = 1e-2,
#'   prior_prec1_rate   = 1e-2,
#'   prior_prec_y_shape = 1e-2,
#'   prior_prec_y_rate  = 1e-2,
#'   verbose            = FALSE,
#'   seed               = 456
#' )
#'
#' is.normal_locallevel(out)     # TRUE
#' is.normal_locallevel(list())  # FALSE
#'
#' @export
is.normal_locallevel <- function(x) {
  inherits(x, "normal_locallevel")
}


#' Print method for normal_locallevel objects
#'
#' @description Prints a concise summary showing posterior medians.
#'   Use `summary()` for comprehensive statistics when available.
#'
#' @param x An object of class `normal_locallevel`.
#' @param digits Integer, number of decimal places to display. Default is 3.
#' @param ... Additional arguments (currently unused).
#'
#' @return Invisibly returns the input object `x`.
#'
#' @details This method provides a quick overview using posterior medians,
#'   which are robust to outliers and skewness in the posterior distribution.
#'
#' @examples
#' \donttest{
#' ## Simulate data (same setup as ?mcmc_normal_locallevel)
#' n <- 200
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
#'   burnin             = 1000,
#'   thinning           = 10,
#'   n_chain            = 500,
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
#' print(out)
#' }
#'
#' @seealso
#'   \code{\link{mcmc_normal_locallevel}} (model generator),
#'   \code{\link{plot.normal_locallevel}}, \code{\link{summary.normal_locallevel}}.
#'
#' @export
print.normal_locallevel <- function(x, digits = 3, ...) {

  # Validate input
  if (!inherits(x, "normal_locallevel")) {
    stop("Object must be of class 'normal_locallevel'")
  }

  # Validate and coerce digits parameter
  if (!is.numeric(digits) || length(digits) != 1 || digits < 0) {
    warning("'digits' must be a non-negative scalar; using default digits = 3")
    digits <- 3
  }
  digits <- as.integer(digits)

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
  cat("  Thinning:          ", attr(x, "thinning"), "\n", sep = "")
  # Which version and seed produced this fit. Several defaults have moved
  # across releases, so a saved object needs to say where it came from.
  cat("  pdm version:       ", attr(x, "pdm_version"), "\n", sep = "")
  cat("  Seed:              ",
      if (is.null(attr(x, "seed"))) "not set" else attr(x, "seed"),
      "\n\n", sep = "")

  # Calculate medians with error handling
  tryCatch({
    med_theta01 <- median(x$theta_01, na.rm = FALSE)
    med_prec1 <- median(x$prec_theta1, na.rm = FALSE)
    med_precy <- median(x$prec_y, na.rm = FALSE)
  }, error = function(e) {
    stop("Error calculating posterior medians: ", e$message, call. = FALSE)
  })

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

  # Median trajectory summary for theta_1 with error handling
  tryCatch({
    theta_1_median <- apply(x$theta_1, 2, median, na.rm = FALSE)
    theta_1_min <- min(theta_1_median)
    theta_1_max <- max(theta_1_median)
    theta_1_final <- theta_1_median[length(theta_1_median)]

    cat("Latent Level (theta_{t,1}) Median Summary:\n")
    cat("  Range:  [",
        sprintf(paste0("%", field_width, ".", digits, "f"), theta_1_min),
        ", ",
        sprintf(paste0("%", field_width, ".", digits, "f"), theta_1_max),
        "]\n", sep = "")
    cat("  Final:  ",
        sprintf(paste0("%", field_width, ".", digits, "f"), theta_1_final),
        "  (median level at last time point)\n\n", sep = "")
  }, error = function(e) {
    cat("Latent Level (theta_{t,1}) Median Summary:\n")
    cat("  [Error computing trajectory summary: ", e$message, "]\n\n", sep = "")
  })

  # User guidance
  cat(strrep("-", 70), "\n", sep = "")
  cat("Note: Showing posterior medians (robust central tendency).\n")
  cat("      For additional summaries: summary(x) when available.\n")
  cat("      For visual diagnostics: plot(x) if implemented.\n")
  cat(strrep("-", 70), "\n\n", sep = "")

  invisible(x)
}
