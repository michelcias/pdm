#' Constructor for binomial_locallevel class
#'
#' @description Internal constructor function for creating objects of class
#'   `binomial_locallevel`. This function is called by
#'   \code{\link{mcmc_binomial_locallevel}} and should not be called directly by
#'   users.
#'
#' @param result List containing MCMC results returned by the C function.
#' @param n_obs Integer, number of observations in the original data.
#' @param n_chain Integer, number of MCMC samples retained after burn-in and thinning.
#' @param burnin Integer, number of burn-in iterations.
#' @param thinning Integer, thinning interval.
#' @param y Numeric vector of original observed data.
#' @param n_trials Numeric scalar, number of trials for each binomial observation.
#' @param target_acceptance Numeric, target acceptance proportion for Metropolis-Hastings.
#'
#' @return An object of class `c("binomial_locallevel", "pdm_mcmc", "list")`
#'   with the following structure:
#'   \describe{
#'     \item{Data components}{All elements from `result` (theta_1,
#'       theta_01, prec_theta1, alpha, and optionally log_sigma and accept_prop)}
#'     \item{Attributes}{
#'       \itemize{
#'         \item `n_obs`: Number of observations
#'         \item `n_chain`: Number of MCMC samples
#'         \item `burnin`: Burn-in iterations
#'         \item `thinning`: Thinning interval
#'         \item `model_type`: `"locallevel"` (polynomial order 1)
#'         \item `link`: `"logit"` (link function)
#'         \item `y`: Original observed data
#'         \item `n_trials`: Number of trials
#'         \item `target_acceptance`: Target acceptance proportion
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
#'     \item `binomial_locallevel`: Specific model class
#'     \item `pdm_mcmc`: General MCMC class for the pdm package
#'     \item `list`: Base R list class
#'   }
#'
#' @keywords internal
#' @noRd
new_binomial_locallevel <- function(result,
                                    n_obs,
                                    n_chain,
                                    burnin,
                                    thinning,
                                    y,
                                    n_trials,
                                    target_acceptance) {

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
  if (!is.numeric(n_trials) || length(n_trials) != 1 || n_trials <= 0) {
    stop("Internal error: n_trials must be a positive scalar")
  }
  if (!is.numeric(target_acceptance) || length(target_acceptance) != 1 ||
      target_acceptance <= 0 || target_acceptance >= 1) {
    stop("Internal error: target_acceptance must be a scalar in (0,1)")
  }

  # Add class hierarchy
  class(result) <- c("binomial_locallevel", "pdm_mcmc", "list")

  # Add metadata as attributes
  attr(result, "n_obs") <- as.integer(n_obs)
  attr(result, "n_chain") <- as.integer(n_chain)
  attr(result, "burnin") <- as.integer(burnin)
  attr(result, "thinning") <- as.integer(thinning)
  attr(result, "model_type") <- "locallevel"  # Polynomial order 1
  attr(result, "link") <- "logit"  # Binomial model uses logit link
  attr(result, "y") <- y  # Store original data for plotting
  attr(result, "n_trials") <- as.numeric(n_trials)  # Store number of trials
  attr(result, "target_acceptance") <- as.numeric(target_acceptance)  # Store target acceptance

  return(result)
}


#' Validator for binomial_locallevel class
#'
#' @description Internal function to validate objects of class
#'   `binomial_locallevel`. Checks that all required components are present
#'   and have correct dimensions.
#'
#' @param x An object to validate.
#'
#' @return The input object `x` if validation succeeds.
#' @keywords internal
#' @noRd
validate_binomial_locallevel <- function(x) {

  # 1. Check class
  if (!inherits(x, "binomial_locallevel")) {
    stop("Object must inherit from class 'binomial_locallevel'")
  }

  # 2. Check required components exist
  required_components <- c("theta_1", "theta_01", "prec_theta1", "alpha")
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
  scalar_params <- c("theta_01", "prec_theta1")
  precision_params <- c("prec_theta1")

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

  # 5. Validate theta_1 and alpha matrices
  matrix_params <- c("theta_1", "alpha")

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

  # 6. Validate alpha values are probabilities
  if (any(x$alpha < 0 | x$alpha > 1)) {
    n_bad <- sum(x$alpha < 0 | x$alpha > 1)
    stop(sprintf(
      "Component 'alpha' contains %d values outside [0,1] (not valid probabilities)",
      n_bad
    ))
  }

  # 7. Validate optional components (log_sigma and accept_prop)
  if (!is.null(x$log_sigma)) {
    if (!is.matrix(x$log_sigma)) {
      stop("Component 'log_sigma' must be a matrix")
    }
    if (!is.numeric(x$log_sigma)) {
      stop("Component 'log_sigma' must be numeric")
    }

    dims <- dim(x$log_sigma)
    if (dims[1] != n_chain || dims[2] != n_obs) {
      stop(sprintf(
        "Component 'log_sigma' has incorrect dimensions [%d x %d], expected [%d x %d]",
        dims[1], dims[2], n_chain, n_obs
      ))
    }

    if (any(!is.finite(x$log_sigma))) {
      n_bad <- sum(!is.finite(x$log_sigma))
      stop(sprintf(
        "Component 'log_sigma' contains %d non-finite values",
        n_bad
      ))
    }
  }

  if (!is.null(x$accept_prop)) {
    if (!is.matrix(x$accept_prop)) {
      stop("Component 'accept_prop' must be a matrix")
    }
    if (!is.numeric(x$accept_prop)) {
      stop("Component 'accept_prop' must be numeric")
    }

    dims <- dim(x$accept_prop)
    if (dims[1] != n_chain || dims[2] != n_obs) {
      stop(sprintf(
        "Component 'accept_prop' has incorrect dimensions [%d x %d], expected [%d x %d]",
        dims[1], dims[2], n_chain, n_obs
      ))
    }

    if (any(!is.finite(x$accept_prop))) {
      n_bad <- sum(!is.finite(x$accept_prop))
      stop(sprintf(
        "Component 'accept_prop' contains %d non-finite values",
        n_bad
      ))
    }

    # Acceptance proportions should be between 0 and 1
    if (any(x$accept_prop < 0 | x$accept_prop > 1)) {
      n_bad <- sum(x$accept_prop < 0 | x$accept_prop > 1)
      stop(sprintf(
        "Component 'accept_prop' contains %d values outside [0,1]",
        n_bad
      ))
    }
  }

  # 8. Validate stored data attribute (if present)
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

  # 9. Validate n_trials attribute
  n_trials <- attr(x, "n_trials")
  if (is.null(n_trials)) {
    stop("Missing required attribute 'n_trials'")
  }
  if (!is.numeric(n_trials) || length(n_trials) != 1 || n_trials <= 0) {
    stop("Attribute 'n_trials' must be a positive scalar")
  }

  # 10. Validate model type
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

  # 11. Validate link function
  link <- attr(x, "link")
  if (is.null(link)) {
    stop("Missing required attribute 'link'")
  }
  if (!identical(link, "logit")) {
    stop(sprintf(
      "Attribute 'link' must be 'logit', got '%s'",
      as.character(link)
    ))
  }

  # 12. Validate optional metadata
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

  # 13. Validate target_acceptance attribute
  target_acceptance <- attr(x, "target_acceptance")
  if (!is.null(target_acceptance)) {
    if (!is.numeric(target_acceptance) || length(target_acceptance) != 1 ||
        target_acceptance <= 0 || target_acceptance >= 1) {
      stop("Attribute 'target_acceptance' must be a scalar in (0,1)")
    }
  }

  return(x)
}


#' Check if object is of class binomial_locallevel
#'
#' @description Test whether an object is of class `binomial_locallevel`.
#'
#' @param x An object to test.
#'
#' @return Logical value: `TRUE` if `x` inherits from
#'   `binomial_locallevel`, `FALSE` otherwise.
#'
#' @examples
#' \donttest{
#' ## Simulate data (same setup as ?mcmc_binomial_locallevel)
#' set.seed(123)
#' n <- 500
#' n_trials <- 20
#'
#' # Generate true probabilities
#' alpha_true <-  (sin(2 * pi * seq_len(n) / n) + 2) / 4
#'
#' # Generate binomial observations
#' y <- rbinom(n, size = n_trials, prob = alpha_true)
#'
#' out <- mcmc_binomial_locallevel(
#'   y,
#'   n_trials                = n_trials,
#'   burnin                  = 1000,
#'   thinning                = 50,
#'   n_chain                 = 1000,
#'   prior_theta01_mean      = 0,
#'   prior_theta01_prec      = 1,
#'   prior_prec1_shape       = 100,
#'   prior_prec1_rate        = 1,
#'   lag_update              = 50,
#'   max_step_size           = 0.1,
#'   base_adaptation_rate    = 1,
#'   decay_exponent          = 0.6,
#'   target_acceptance       = 0.44,
#'   min_deviation_threshold = NULL,
#'   return_log_sigma        = FALSE,
#'   return_accept_prop      = TRUE,
#'   verbose                 = TRUE,
#'   bar_width               = 60,
#'   seed                    = 456
#' )
#'
#' is.binomial_locallevel(out)  # TRUE
#' is.binomial_locallevel(list())  # FALSE
#' }
#'
#' @export
is.binomial_locallevel <- function(x) {
  inherits(x, "binomial_locallevel")
}


#' Print method for binomial_locallevel objects
#'
#' @description Prints a concise summary showing posterior medians.
#'   Use `summary()` for comprehensive statistics when available.
#'
#' @param x An object of class `binomial_locallevel`.
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
#' ## Simulate data (same setup as ?mcmc_binomial_locallevel)
#' set.seed(123)
#' n <- 500
#' n_trials <- 20
#'
#' # Generate true probabilities
#' alpha_true <-  (sin(2 * pi * seq_len(n) / n) + 2) / 4
#'
#' # Generate binomial observations
#' y <- rbinom(n, size = n_trials, prob = alpha_true)
#'
#' out <- mcmc_binomial_locallevel(
#'   y,
#'   n_trials                = n_trials,
#'   burnin                  = 1000,
#'   thinning                = 50,
#'   n_chain                 = 1000,
#'   prior_theta01_mean      = 0,
#'   prior_theta01_prec      = 1,
#'   prior_prec1_shape       = 100,
#'   prior_prec1_rate        = 1,
#'   lag_update              = 50,
#'   max_step_size           = 0.1,
#'   base_adaptation_rate    = 1,
#'   decay_exponent          = 0.6,
#'   target_acceptance       = 0.44,
#'   min_deviation_threshold = NULL,
#'   return_log_sigma        = FALSE,
#'   return_accept_prop      = TRUE,
#'   verbose                 = TRUE,
#'   bar_width               = 60,
#'   seed                    = 456
#' )
#'
#' print(out)
#' }
#'
#' @seealso
#'   \code{\link{mcmc_binomial_locallevel}} (model generator),
#'   \code{\link{plot.binomial_locallevel}}, \code{\link{summary.binomial_locallevel}}.
#'
#' @export
print.binomial_locallevel <- function(x, digits = 3, ...) {

  # Validate input
  if (!inherits(x, "binomial_locallevel")) {
    stop("Object must be of class 'binomial_locallevel'")
  }

  # Validate and coerce digits parameter
  if (!is.numeric(digits) || length(digits) != 1 || digits < 0) {
    warning("'digits' must be a non-negative scalar; using default digits = 3")
    digits <- 3
  }
  digits <- as.integer(digits)

  cat("\n")
  cat("Binomial Local-Level Model (Logit Link)\n")
  cat(strrep("=", 70), "\n\n", sep = "")

  # Model metadata
  cat("Model:\n")
  cat("  Type:              ", attr(x, "model_type"),
      " (1st order polynomial)\n", sep = "")
  cat("  Link function:     ", attr(x, "link"), "\n", sep = "")
  cat("  Number of trials:  ", attr(x, "n_trials"), "\n", sep = "")
  cat("\n")

  # MCMC metadata
  cat("MCMC:\n")
  cat("  Observations:      ", attr(x, "n_obs"), "\n", sep = "")
  cat("  Samples retained:  ", attr(x, "n_chain"), "\n", sep = "")
  cat("  Burn-in:           ", attr(x, "burnin"), "\n", sep = "")
  cat("  Thinning:          ", attr(x, "thinning"), "\n\n", sep = "")

  # Calculate medians with error handling
  tryCatch({
    med_theta01 <- median(x$theta_01, na.rm = FALSE)
    med_prec1 <- median(x$prec_theta1, na.rm = FALSE)
  }, error = function(e) {
    stop("Error calculating posterior medians: ", e$message, call. = FALSE)
  })

  # Determine field width for alignment
  field_width <- digits + 4

  # Posterior medians for scalar parameters
  cat("Posterior Medians (Scalars):\n")
  cat("  theta_01:  ", sprintf(paste0("%", field_width, ".", digits, "f"), med_theta01),
      "  (initial level)\n", sep = "")
  cat("  W_1^-1:    ", sprintf(paste0("%", field_width, ".", digits, "f"), med_prec1),
      "  (level innovation precision)\n\n", sep = "")

  # Median trajectory summary for theta_1
  tryCatch({
    theta_1_median <- apply(x$theta_1, 2, median, na.rm = FALSE)
    theta_1_min <- min(theta_1_median)
    theta_1_max <- max(theta_1_median)
    theta_1_final <- theta_1_median[length(theta_1_median)]

    cat("Latent Level (theta_{t,1}) Median Summary (logit scale):\n")
    cat("  Range:  [",
        sprintf(paste0("%", field_width-2, ".", digits, "f"), theta_1_min),
        ", ",
        sprintf(paste0("%", field_width-2, ".", digits, "f"), theta_1_max),
        "]\n", sep = "")
    cat("  Final:  ",
        sprintf(paste0("%", field_width-2, ".", digits, "f"), theta_1_final),
        "  (median level at last time point)\n\n", sep = "")
  }, error = function(e) {
    cat("Latent Level (theta_{t,1}) Median Summary:\n")
    cat("  [Error computing trajectory summary: ", e$message, "]\n\n", sep = "")
  })

  # Summary of success probabilities
  tryCatch({
    alpha_median_time <- apply(x$alpha, 2, median, na.rm = FALSE)
    alpha_min <- min(alpha_median_time)
    alpha_max <- max(alpha_median_time)
    alpha_med <- median(alpha_median_time)

    cat("Success Probabilities (alpha_t):\n")
    cat("  Range:  [",
        sprintf(paste0("%", field_width-2, ".", digits, "f"), alpha_min),
        ", ",
        sprintf(paste0("%", field_width-2, ".", digits, "f"), alpha_max),
        "]\n", sep = "")
    cat("  Median: ",
        sprintf(paste0("%", field_width-2, ".", digits, "f"), alpha_med), "\n\n", sep = "")
  }, error = function(e) {
    cat("Success Probabilities (alpha_t):\n")
    cat("  [Error computing summary: ", e$message, "]\n\n", sep = "")
  })

  # User guidance
  cat(strrep("-", 70), "\n", sep = "")
  cat("Note: Showing posterior medians (robust central tendency).\n")
  cat("      For additional summaries: summary(x) when available.\n")
  cat("      For visual diagnostics: plot(x) if implemented.\n")
  cat(strrep("-", 70), "\n\n", sep = "")

  invisible(x)
}
