#' Constructor for normal_mixture_localacceleration class
#'
#' @description Internal constructor function for creating objects of class
#'   \code{normal_mixture_localacceleration}. This function is called by
#'   \code{\link{mcmc_normal_mixture_localacceleration}} and should not be called
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
#' @return An object of class \code{c("normal_mixture_localacceleration", "pdm_mcmc", "list")}
#'   with the following structure:
#'   \describe{
#'     \item{Data components}{All elements from \code{result} (mu_1, mu_2, prec_1,
#'       prec_2, theta_1, theta_2, theta_3, theta_01, theta_02, theta_03,
#'       prec_theta1, prec_theta2, prec_theta3, alpha, z, and optionally log_sigma
#'       and accept_prop)}
#'     \item{Attributes}{
#'       \itemize{
#'         \item \code{link}: Link function used
#'         \item \code{n_obs}: Number of observations
#'         \item \code{n_chain}: Number of MCMC samples
#'         \item \code{burnin}: Burn-in iterations
#'         \item \code{thinning}: Thinning interval
#'         \item \code{model_type}: "localacceleration" (3rd order polynomial)
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
#'     \item \code{normal_mixture_localacceleration}: Specific model class
#'     \item \code{pdm_mcmc}: General MCMC class for the pdm package
#'     \item \code{list}: Base R list class
#'   }
#'
#' @keywords internal
#' @noRd
new_normal_mixture_localacceleration <- function(result,
                                                 link,
                                                 n_obs,
                                                 n_chain,
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
  class(result) <- c("normal_mixture_localacceleration", "pdm_mcmc", "list")

  # Add metadata as attributes
  attr(result, "link") <- link
  attr(result, "n_obs") <- as.integer(n_obs)
  attr(result, "n_chain") <- as.integer(n_chain)
  attr(result, "burnin") <- as.integer(burnin)
  attr(result, "thinning") <- as.integer(thinning)
  attr(result, "model_type") <- "localacceleration"  # Polynomial order 3
  attr(result, "y") <- y  # Store original data for plotting

  return(result)
}


#' Validator for normal_mixture_localacceleration class
#'
#' @description Internal function to validate objects of class
#'   \code{normal_mixture_localacceleration}. Checks that all required components
#'   are present and have correct dimensions.
#'
#' @param x An object to validate.
#'
#' @return The input object \code{x} if validation succeeds.
#' @keywords internal
#' @noRd
validate_normal_mixture_localacceleration <- function(x) {

  # 1. Check class
  if (!inherits(x, "normal_mixture_localacceleration")) {
    stop("Object must inherit from class 'normal_mixture_localacceleration'")
  }

  # 2. Check required components exist
  required_components <- c(
    "mu_1", "mu_2", "prec_1", "prec_2",
    "theta_1", "theta_2", "theta_3",
    "theta_01", "theta_02", "theta_03",
    "prec_theta1", "prec_theta2", "prec_theta3",
    "alpha", "z"
  )

  missing <- setdiff(required_components, names(x))
  if (length(missing) > 0) {
    stop("Missing required components: ", paste(missing, collapse = ", "))
  }

  # 3. Extract and validate metadata attributes
  n_chain_raw <- attr(x, "n_chain")
  n_obs_raw <- attr(x, "n_obs")
  link <- attr(x, "link")

  if (is.null(n_chain_raw)) {
    stop("Missing required attribute 'n_chain'")
  }
  if (is.null(n_obs_raw)) {
    stop("Missing required attribute 'n_obs'")
  }
  if (is.null(link)) {
    stop("Missing required attribute 'link'")
  }

  n_chain <- as.integer(n_chain_raw)
  n_obs <- as.integer(n_obs_raw)

  if (length(n_chain) == 0 || is.na(n_chain) || n_chain <= 0) {
    stop("Attribute 'n_chain' must be a positive integer")
  }
  if (length(n_obs) == 0 || is.na(n_obs) || n_obs <= 0) {
    stop("Attribute 'n_obs' must be a positive integer")
  }

  # 4. Validate link function
  if (!is.character(link) || length(link) != 1 || !link %in% c("logit", "probit")) {
    stop("Attribute 'link' must be either 'logit' or 'probit'")
  }

  # 5. Validate scalar parameters (type, length, finiteness, positivity)
  scalar_params <- c(
    "mu_1", "mu_2", "prec_1", "prec_2",
    "theta_01", "theta_02", "theta_03",
    "prec_theta1", "prec_theta2", "prec_theta3"
  )
  precision_params <- c("prec_1", "prec_2", "prec_theta1", "prec_theta2", "prec_theta3")

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

  # 6. Validate matrix parameters
  matrix_params <- c("theta_1", "theta_2", "theta_3", "alpha", "z")

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

  # 7. Validate alpha values are probabilities
  if (any(x$alpha < 0 | x$alpha > 1)) {
    n_bad <- sum(x$alpha < 0 | x$alpha > 1)
    stop(sprintf(
      "Component 'alpha' contains %d values outside [0,1] (not valid probabilities)",
      n_bad
    ))
  }

  # 8. Validate z values are binary
  if (!all(x$z %in% c(0, 1))) {
    stop("Component 'z' must contain only binary values (0 or 1)")
  }

  # 9. Validate optional components for logit link
  if (link == "logit") {
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
  }

  # 10. Validate stored data attribute (if present)
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

  # 11. Validate model type
  model_type <- attr(x, "model_type")
  if (is.null(model_type)) {
    stop("Missing required attribute 'model_type'")
  }
  if (!identical(model_type, "localacceleration")) {
    stop(sprintf(
      "Attribute 'model_type' must be 'localacceleration', got '%s'",
      as.character(model_type)
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

  return(x)
}


#' Check if object is of class normal_mixture_localacceleration
#'
#' @description Test whether an object is of class \code{normal_mixture_localacceleration}.
#'
#' @param x An object to test.
#'
#' @return Logical value: \code{TRUE} if \code{x} inherits from
#'   \code{normal_mixture_localacceleration}, \code{FALSE} otherwise.
#'
#' @examples
#' \dontrun{
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
#' out_logit <- mcmc_normal_mixture_localacceleration(
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
#'   prior_theta03_mean      = 0,
#'   prior_theta03_prec      = 1,
#'   prior_prec1_shape       = 100,
#'   prior_prec1_rate        = 1,
#'   prior_prec2_shape       = 400,
#'   prior_prec2_rate        = 1,
#'   prior_prec3_shape       = 900,
#'   prior_prec3_rate        = 1,
#'   lag_update              = 50,
#'   max_step_size           = 1.0,
#'   base_adaptation_rate    = 0.01,
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
#' is.normal_mixture_localacceleration(out_logit)  # TRUE
#' is.normal_mixture_localacceleration(list())     # FALSE
#' }
#'
#' @export
is.normal_mixture_localacceleration <- function(x) {
  inherits(x, "normal_mixture_localacceleration")
}


#' Print method for normal_mixture_localacceleration objects
#'
#' @description Prints a concise summary showing posterior medians.
#'   Use \code{summary()} for means, standard deviations, and credible intervals.
#'
#' @param x An object of class \code{normal_mixture_localacceleration}.
#' @param digits Integer, number of decimal places to display. Default is 3.
#' @param ... Additional arguments (currently unused).
#'
#' @return Invisibly returns the input object \code{x}.
#'
#' @details This method provides a quick overview using posterior medians,
#'   which are robust to outliers and skewness in the posterior distribution.
#'
#'   For comprehensive statistics including means, standard deviations, and
#'   credible intervals, use \code{summary(x)}.
#'
#'   \strong{Why medians?}
#'   \itemize{
#'     \item Robust to outliers and long tails
#'     \item More representative for skewed posteriors (e.g., precision parameters)
#'     \item Minimizes absolute error loss
#'     \item Less sensitive to incomplete MCMC convergence
#'   }
#'
#' @examples
#' \dontrun{
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
#' out_logit <- mcmc_normal_mixture_localacceleration(
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
#'   prior_theta03_mean      = 0,
#'   prior_theta03_prec      = 1,
#'   prior_prec1_shape       = 100,
#'   prior_prec1_rate        = 1,
#'   prior_prec2_shape       = 400,
#'   prior_prec2_rate        = 1,
#'   prior_prec3_shape       = 900,
#'   prior_prec3_rate        = 1,
#'   lag_update              = 50,
#'   max_step_size           = 1.0,
#'   base_adaptation_rate    = 0.01,
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
#' # Quick overview (medians only)
#' print(out_logit)
#' # or simply:
#' out_logit
#'
#' # For full statistics with means and credible intervals:
#' summary(out_logit)
#' }
#'
#' @seealso \code{\link{summary.normal_mixture_localacceleration}}
#' @export
print.normal_mixture_localacceleration <- function(x, digits = 3, ...) {

  # Validate input
  if (!inherits(x, "normal_mixture_localacceleration")) {
    stop("Object must be of class 'normal_mixture_localacceleration'")
  }

  # Validate and coerce digits parameter
  if (!is.numeric(digits) || length(digits) != 1 || digits < 0) {
    warning("'digits' must be a non-negative scalar; using default digits = 3")
    digits <- 3
  }
  digits <- as.integer(digits)

  cat("\n")
  cat("Gaussian Mixture Model with Dynamic Mixture Weights\n")
  cat(strrep("=", 70), "\n\n", sep = "")

  # Model metadata
  cat("Model:\n")
  cat("  Type:              ", attr(x, "model_type"),
      " (3rd order polynomial)\n", sep = "")
  cat("  Link function:     ", attr(x, "link"), "\n", sep = "")
  cat("\n")

  # MCMC metadata
  cat("MCMC:\n")
  cat("  Observations:      ", attr(x, "n_obs"), "\n", sep = "")
  cat("  Samples retained:  ", attr(x, "n_chain"), "\n", sep = "")
  cat("  Burn-in:           ", attr(x, "burnin"), "\n", sep = "")
  cat("  Thinning:          ", attr(x, "thinning"), "\n\n", sep = "")

  # Calculate medians with error handling
  tryCatch({
    med_mu1 <- median(x$mu_1, na.rm = FALSE)
    med_mu2 <- median(x$mu_2, na.rm = FALSE)
    med_phi1 <- median(x$prec_1, na.rm = FALSE)
    med_phi2 <- median(x$prec_2, na.rm = FALSE)
    med_theta01 <- median(x$theta_01, na.rm = FALSE)
    med_theta02 <- median(x$theta_02, na.rm = FALSE)
    med_theta03 <- median(x$theta_03, na.rm = FALSE)
    med_prec1 <- median(x$prec_theta1, na.rm = FALSE)
    med_prec2 <- median(x$prec_theta2, na.rm = FALSE)
    med_prec3 <- median(x$prec_theta3, na.rm = FALSE)
  }, error = function(e) {
    stop("Error calculating posterior medians: ", e$message, call. = FALSE)
  })

  # Determine field width for alignment
  field_width <- digits + 4

  # Posterior medians (mixture components)
  cat("Posterior Medians (Mixture Components):\n")
  cat("  mu_1:   ", sprintf(paste0("%", field_width, ".", digits, "f"), med_mu1),
      "  (component 1 mean)\n", sep = "")
  cat("  mu_2:   ", sprintf(paste0("%", field_width, ".", digits, "f"), med_mu2),
      "  (component 2 mean)\n", sep = "")
  cat("  phi_1:  ", sprintf(paste0("%", field_width, ".", digits, "f"), med_phi1),
      "  (component 1 precision)\n", sep = "")
  cat("  phi_2:  ", sprintf(paste0("%", field_width, ".", digits, "f"), med_phi2),
      "  (component 2 precision)\n\n", sep = "")

  # Posterior medians (dynamic states)
  cat("Posterior Medians (Dynamic States):\n")
  cat("  theta_01:   ", sprintf(paste0("%", field_width, ".", digits, "f"), med_theta01),
      "  (initial level)\n", sep = "")
  cat("  theta_02:   ", sprintf(paste0("%", field_width, ".", digits, "f"), med_theta02),
      "  (initial trend)\n", sep = "")
  cat("  theta_03:   ", sprintf(paste0("%", field_width, ".", digits, "f"), med_theta03),
      "  (initial acceleration)\n", sep = "")
  cat("  W_1^-1:     ", sprintf(paste0("%", field_width, ".", digits, "f"), med_prec1),
      "  (level precision)\n", sep = "")
  cat("  W_2^-1:     ", sprintf(paste0("%", field_width, ".", digits, "f"), med_prec2),
      "  (trend precision)\n", sep = "")
  cat("  W_3^-1:     ", sprintf(paste0("%", field_width, ".", digits, "f"), med_prec3),
      "  (acceleration precision)\n\n", sep = "")

  # Summary of time-varying alpha with error handling
  tryCatch({
    alpha_median_time <- apply(x$alpha, 2, median, na.rm = FALSE)
    alpha_min <- min(alpha_median_time)
    alpha_max <- max(alpha_median_time)
    alpha_med <- median(alpha_median_time)

    cat("Mixture Weights (alpha_t):\n")
    cat("  Range:  [",
        sprintf(paste0("%", field_width, ".", digits, "f"), alpha_min),
        ", ",
        sprintf(paste0("%", field_width, ".", digits, "f"), alpha_max),
        "]\n", sep = "")
    cat("  Median: ",
        sprintf(paste0("%", field_width, ".", digits, "f"), alpha_med), "\n\n", sep = "")
  }, error = function(e) {
    cat("Mixture Weights (alpha_t):\n")
    cat("  [Error computing summary: ", e$message, "]\n\n", sep = "")
  })

  # User guidance
  cat(strrep("-", 70), "\n", sep = "")
  cat("Note: Showing posterior medians (robust central tendency).\n")
  cat("      For means, SDs, and credible intervals: summary(x)\n")
  cat("      For visual diagnostics: plot(x)\n")
  cat(strrep("-", 70), "\n\n", sep = "")

  invisible(x)
}
