#' Constructor for normal_localtrend class
#'
#' @description Internal constructor function for creating objects of class
#'   `normal_localtrend`. This function is called by
#'   \code{\link{mcmc_normal_localtrend}} and should not be called directly by
#'   users.
#'
#' @param result List containing MCMC results returned by the C function.
#' @param n_obs Integer, number of observations in the original data.
#' @param n_chain Integer, number of MCMC samples retained after burn-in and thinning.
#' @param burnin Integer, number of burn-in iterations.
#' @param thinning Integer, thinning interval.
#' @param y Numeric vector of original observed data.
#'
#' @return An object of class `c("normal_localtrend", "pdm_mcmc", "list")`
#'   with the following structure:
#'   \describe{
#'     \item{Data components}{All elements from `result` (theta_1, theta_2,
#'       theta_01, theta_02, prec_theta1, prec_theta2, prec_y)}
#'     \item{Attributes}{
#'       \itemize{
#'         \item `n_obs`: Number of observations
#'         \item `n_chain`: Number of MCMC samples
#'         \item `burnin`: Burn-in iterations
#'         \item `thinning`: Thinning interval
#'         \item `model_type`: `"localtrend"` (polynomial order 2)
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
#'     \item `normal_localtrend`: Specific model class
#'     \item `pdm_mcmc`: General MCMC class for the pdm package
#'     \item `list`: Base R list class
#'   }
#'
#' @keywords internal
#' @noRd
new_normal_localtrend <- function(result,
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
  class(result) <- c("normal_localtrend", "pdm_mcmc", "list")

  # Add metadata as attributes
  attr(result, "n_obs") <- as.integer(n_obs)
  attr(result, "n_chain") <- as.integer(n_chain)
  attr(result, "burnin") <- as.integer(burnin)
  attr(result, "thinning") <- as.integer(thinning)
  attr(result, "model_type") <- "localtrend"  # Polynomial order 2
  attr(result, "y") <- y  # Store original data for plotting

  return(result)
}


#' Validator for normal_localtrend class
#'
#' @description Internal function to validate objects of class
#'   `normal_localtrend`. Checks that all required components are present
#'   and have correct dimensions.
#'
#' @param x An object to validate.
#'
#' @return The input object `x` if validation succeeds.
#' @keywords internal
#' @noRd
validate_normal_localtrend <- function(x) {

  # 1. Check class
  if (!inherits(x, "normal_localtrend")) {
    stop("Object must inherit from class 'normal_localtrend'")
  }

  # 2. Check required components exist
  required_components <- c("theta_1", "theta_2", "theta_01", "theta_02",
                           "prec_theta1", "prec_theta2", "prec_y")
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
  scalar_params <- c("theta_01", "theta_02", "prec_theta1", "prec_theta2", "prec_y")
  precision_params <- c("prec_theta1", "prec_theta2", "prec_y")

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

  # 5. Validate theta_1 and theta_2 matrices
  matrix_params <- c("theta_1", "theta_2")

  for (param in matrix_params) {
    if (!is.matrix(x[[param]])) {
      stop(sprintf("Component '%s' must be a matrix", param))
    }

    if (!is.numeric(x[[param]])) {
      stop(sprintf("Component '%s' must be numeric", param))
    }

    dims <- dim(x[[param]])

    if (dims[1] != n_chain || dims[2] != n_obs) {
      stop(sprintf(
        "Component '%s' has incorrect dimensions [%d x %d], expected [%d x %d].\n  Each row should be one MCMC sample, each column one time point.",
        param, dims[1], dims[2], n_chain, n_obs
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
  if (!identical(model_type, "localtrend")) {
    stop(sprintf(
      "Attribute 'model_type' must be 'localtrend', got '%s'",
      as.character(model_type)
    ))
  }

  # 8. Validate optional metadata
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


#' Check if object is of class normal_localtrend
#'
#' @description Test whether an object is of class `normal_localtrend`.
#'
#' @param x An object to test.
#'
#' @return Logical value: `TRUE` if `x` inherits from
#'   `normal_localtrend`, `FALSE` otherwise.
#'
#' @examples
#' ## A minimal fit is all this test needs; see
#' ## ?mcmc_normal_localtrend for a realistic analysis.
#' set.seed(123)
#' n <- 40
#' y <- cumsum(c(10, rnorm(n)))[-1] + rnorm(n, sd = 0.5)
#'
#' out <- mcmc_normal_localtrend(
#'   y,
#'   burnin             = 50,
#'   thinning           = 1,
#'   n_chain            = 50,
#'   prior_theta01_mean = y[1],
#'   prior_theta01_prec = 1 / var(y),
#'   prior_theta02_mean = 0,
#'   prior_theta02_prec = 1 / var(y),
#'   prior_prec1_shape  = 1e-2,
#'   prior_prec1_rate   = 1e-2,
#'   prior_prec2_shape  = 1e-2,
#'   prior_prec2_rate   = 1e-2,
#'   prior_prec_y_shape = 1e-2,
#'   prior_prec_y_rate  = 1e-2,
#'   verbose            = FALSE,
#'   seed               = 456
#' )
#'
#' is.normal_localtrend(out)     # TRUE
#' is.normal_localtrend(list())  # FALSE
#'
#' @export
is.normal_localtrend <- function(x) {
  inherits(x, "normal_localtrend")
}


#' Print method for normal_localtrend objects
#'
#' @description Prints a concise summary showing posterior medians.
#'   Use `summary()` for comprehensive statistics when available.
#'
#' @param x An object of class `normal_localtrend`.
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
#' ## Simulate data (same setup as ?mcmc_normal_localtrend)
#' n <- 200
#'
#' theta01_true <- 10
#' theta02_true <- 0.5
#' prec1_true   <- 1 / 0.10
#' prec2_true   <- 1 / 0.01
#' prec_y_true  <- 1 / 1.00
#'
#' set.seed(123)
#' u1      <- rnorm(n, sd = sqrt(1 / prec1_true))
#' u2      <- rnorm(n, sd = sqrt(1 / prec2_true))
#' epsilon <- rnorm(n, sd = sqrt(1 / prec_y_true))
#'
#' theta1_true    <- numeric(n)
#' theta2_true    <- numeric(n)
#' theta2_true[1] <- theta02_true + u2[1]
#' theta1_true[1] <- theta01_true + theta02_true + u1[1]
#' for (t in 2:n) {
#'   theta2_true[t] <- theta2_true[t-1] + u2[t]
#'   theta1_true[t] <- theta1_true[t-1] + theta2_true[t-1] + u1[t]
#' }
#' y <- theta1_true + epsilon
#'
#' out <- mcmc_normal_localtrend(
#'   y,
#'   burnin             = 1000,
#'   thinning           = 20,
#'   n_chain            = 500,
#'   prior_theta01_mean = y[1] / 2,
#'   prior_theta01_prec = 1 / var(y),
#'   prior_theta02_mean = y[1] / 2,
#'   prior_theta02_prec = 1 / var(y),
#'   prior_prec1_shape  = 1e-1,
#'   prior_prec1_rate   = 1e-1,
#'   prior_prec2_shape  = 1e-2,
#'   prior_prec2_rate   = 1e-2,
#'   prior_prec_y_shape = 1e-1,
#'   prior_prec_y_rate  = 1e-1,
#'   verbose            = TRUE,
#'   bar_width          = 60,
#'   seed               = 456
#' )
#'
#' print(out)
#' }
#'
#' @seealso
#'   \code{\link{mcmc_normal_localtrend}} (model generator),
#'   \code{\link{plot.normal_localtrend}}, \code{\link{summary.normal_localtrend}}.
#'
#' @export
print.normal_localtrend <- function(x, digits = 3, ...) {

  # Validate input
  if (!inherits(x, "normal_localtrend")) {
    stop("Object must be of class 'normal_localtrend'")
  }

  # Validate and coerce digits parameter
  if (!is.numeric(digits) || length(digits) != 1 || digits < 0) {
    warning("'digits' must be a non-negative scalar; using default digits = 3")
    digits <- 3
  }
  digits <- as.integer(digits)

  cat("\n")
  cat("Gaussian Local-Trend Model\n")
  cat(strrep("=", 70), "\n\n", sep = "")

  # Model metadata
  cat("Model:\n")
  cat("  Type:              ", attr(x, "model_type"),
      " (2nd order polynomial)\n", sep = "")
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
    med_theta02 <- median(x$theta_02, na.rm = FALSE)
    med_prec1 <- median(x$prec_theta1, na.rm = FALSE)
    med_prec2 <- median(x$prec_theta2, na.rm = FALSE)
    med_precy <- median(x$prec_y, na.rm = FALSE)
  }, error = function(e) {
    stop("Error calculating posterior medians: ", e$message, call. = FALSE)
  })

  # Determine field width for alignment
  field_width <- digits + 4

  # Posterior medians for scalar parameters
  cat("Posterior Medians (Scalars):\n")
  cat("  theta_01:  ", sprintf(paste0("%", field_width, ".", digits, "f"), med_theta01),
      "  (initial level)\n", sep = "")
  cat("  theta_02:  ", sprintf(paste0("%", field_width, ".", digits, "f"), med_theta02),
      "  (initial trend)\n", sep = "")
  cat("  W_1^-1:    ", sprintf(paste0("%", field_width, ".", digits, "f"), med_prec1),
      "  (level innovation precision)\n", sep = "")
  cat("  W_2^-1:    ", sprintf(paste0("%", field_width, ".", digits, "f"), med_prec2),
      "  (trend innovation precision)\n", sep = "")
  cat("  V^-1:      ", sprintf(paste0("%", field_width, ".", digits, "f"), med_precy),
      "  (observation precision)\n\n", sep = "")

  # Median trajectory summary for theta_1
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

  # Median trajectory summary for theta_2
  tryCatch({
    theta_2_median <- apply(x$theta_2, 2, median, na.rm = FALSE)
    theta_2_min <- min(theta_2_median)
    theta_2_max <- max(theta_2_median)
    theta_2_final <- theta_2_median[length(theta_2_median)]

    cat("Latent Trend (theta_{t,2}) Median Summary:\n")
    cat("  Range:  [",
        sprintf(paste0("%", field_width, ".", digits, "f"), theta_2_min),
        ", ",
        sprintf(paste0("%", field_width, ".", digits, "f"), theta_2_max),
        "]\n", sep = "")
    cat("  Final:  ",
        sprintf(paste0("%", field_width, ".", digits, "f"), theta_2_final),
        "  (median trend at last time point)\n\n", sep = "")
  }, error = function(e) {
    cat("Latent Trend (theta_{t,2}) Median Summary:\n")
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
