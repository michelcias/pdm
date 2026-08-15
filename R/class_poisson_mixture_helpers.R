#' Shared validator and print method for the Poisson mixture family
#'
#' The three `poisson_mixture_*` classes differ only in how many state
#' components they carry: one apiece for the local level, two for the local
#' trend, three for the local acceleration. Everything else -- the component
#' rates, the weights, the indicators, the metadata attributes and the whole
#' shape of the printed block -- is identical across the three.
#'
#' The Gaussian mixture family writes that logic out once per order, which is
#' three copies of roughly two hundred lines whose only real difference is a
#' vector of names. This file holds one copy parameterised by the order instead,
#' so a fix to the validation reaches all three orders by construction rather
#' than by remembering.
#'
#' @name poisson_mixture_helpers
#' @keywords internal
#' @noRd
NULL


#' Component names of a Poisson mixture fit at a given order
#'
#' @param order Integer 1, 2 or 3 (level, trend, acceleration).
#'
#' @return A list with `scalars` (the per-draw vectors), `matrices` (the
#'   per-draw-by-time matrices), and `positive` (the subset of `scalars`
#'   constrained to be positive).
#'
#' @keywords internal
#' @noRd
poisson_mixture_components <- function(order) {
  states <- paste0("theta_0", seq_len(order))
  precs  <- paste0("prec_theta", seq_len(order))
  rates  <- c("lambda_1", "lambda_2")

  list(
    scalars  = c(rates, states, precs),
    matrices = c(paste0("theta_", seq_len(order)), "alpha", "z"),
    positive = c(rates, precs)
  )
}


#' Validate a fitted Poisson mixture object
#'
#' @param x The object to validate.
#' @param cls The expected class name, e.g. `"poisson_mixture_localtrend"`.
#' @param order Integer 1, 2 or 3.
#'
#' @return `x`, invisibly unchanged, if every check passes.
#'
#' @keywords internal
#' @noRd
validate_poisson_mixture <- function(x, cls, order) {

  # 1. Check class
  if (!inherits(x, cls)) {
    stop(sprintf("Object must inherit from class '%s'", cls))
  }

  comps <- poisson_mixture_components(order)

  # 2. Check required components exist
  required_components <- c(comps$scalars, comps$matrices)

  missing <- setdiff(required_components, names(x))
  if (length(missing) > 0) {
    stop("Missing required components: ", paste(missing, collapse = ", "))
  }

  # 3. Extract and validate metadata attributes
  n_draws_raw <- attr(x, "n_draws")
  n_obs_raw <- attr(x, "n_obs")
  link <- attr(x, "link")

  if (is.null(n_draws_raw)) {
    stop("Missing required attribute 'n_draws'")
  }
  if (is.null(n_obs_raw)) {
    stop("Missing required attribute 'n_obs'")
  }
  if (is.null(link)) {
    stop("Missing required attribute 'link'")
  }

  n_draws <- as.integer(n_draws_raw)
  n_obs <- as.integer(n_obs_raw)

  if (length(n_draws) == 0 || is.na(n_draws) || n_draws <= 0) {
    stop("Attribute 'n_draws' must be a positive integer")
  }
  if (length(n_obs) == 0 || is.na(n_obs) || n_obs <= 0) {
    stop("Attribute 'n_obs' must be a positive integer")
  }

  # 4. Validate link function
  if (!is.character(link) || length(link) != 1 || !link %in% c("logit", "probit")) {
    stop("Attribute 'link' must be either 'logit' or 'probit'")
  }

  # 5. Validate scalar parameters (type, length, finiteness, positivity)
  for (param in comps$scalars) {
    # Check type
    if (!is.numeric(x[[param]])) {
      stop(sprintf("Component '%s' must be numeric", param))
    }

    # Check length
    if (length(x[[param]]) != n_draws) {
      stop(sprintf(
        "Component '%s' should have length %d but has length %d",
        param, n_draws, length(x[[param]])
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

    # Check positivity for the rates and the innovation precisions
    if (param %in% comps$positive) {
      if (any(x[[param]] <= 0)) {
        n_nonpositive <- sum(x[[param]] <= 0)
        stop(sprintf(
          "Component '%s' must be positive, but %d/%d values are non-positive",
          param, n_nonpositive, n_draws
        ))
      }
    }
  }

  # 6. Validate that the component rates respect the ordering constraint. This
  # is the identifiability constraint the sampler enforces by relabelling, so a
  # violation means a draw escaped the swap rather than that the data are odd.
  if (any(x$lambda_1 > x$lambda_2)) {
    n_bad <- sum(x$lambda_1 > x$lambda_2)
    stop(sprintf(
      "The label constraint lambda_1 <= lambda_2 is violated in %d/%d draws",
      n_bad, n_draws
    ))
  }

  # 7. Validate matrix parameters
  for (param in comps$matrices) {
    if (!is.matrix(x[[param]])) {
      stop(sprintf("Component '%s' must be a matrix", param))
    }

    if (!is.numeric(x[[param]])) {
      stop(sprintf("Component '%s' must be numeric", param))
    }

    dims <- dim(x[[param]])

    if (dims[1] != n_draws || dims[2] != n_obs) {
      stop(sprintf(
        "Component '%s' has incorrect dimensions [%d x %d], expected [%d x %d].\n  Each row should be one MCMC draw, each column one time point.",
        param, dims[1], dims[2], n_draws, n_obs
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

  # 8. Validate alpha values are probabilities
  if (any(x$alpha < 0 | x$alpha > 1)) {
    n_bad <- sum(x$alpha < 0 | x$alpha > 1)
    stop(sprintf(
      "Component 'alpha' contains %d values outside [0,1] (not valid probabilities)",
      n_bad
    ))
  }

  # 9. Validate z values are binary
  if (!all(x$z %in% c(0, 1))) {
    stop("Component 'z' must contain only binary values (0 or 1)")
  }

  # 10. Validate optional components for logit link
  if (link == "logit") {
    for (param in c("log_sigma", "accept_prop")) {
      if (is.null(x[[param]])) next

      if (!is.matrix(x[[param]])) {
        stop(sprintf("Component '%s' must be a matrix", param))
      }
      if (!is.numeric(x[[param]])) {
        stop(sprintf("Component '%s' must be numeric", param))
      }

      dims <- dim(x[[param]])
      if (dims[1] != n_draws || dims[2] != n_obs) {
        stop(sprintf(
          "Component '%s' has incorrect dimensions [%d x %d], expected [%d x %d]",
          param, dims[1], dims[2], n_draws, n_obs
        ))
      }

      if (any(!is.finite(x[[param]]))) {
        n_bad <- sum(!is.finite(x[[param]]))
        stop(sprintf("Component '%s' contains %d non-finite values", param, n_bad))
      }
    }

    # Acceptance proportions should be between 0 and 1
    if (!is.null(x$accept_prop) && any(x$accept_prop < 0 | x$accept_prop > 1)) {
      n_bad <- sum(x$accept_prop < 0 | x$accept_prop > 1)
      stop(sprintf(
        "Component 'accept_prop' contains %d values outside [0,1]", n_bad
      ))
    }
  }

  # 11. Validate stored data attribute (if present)
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
    if (any(y < 0 | y != floor(y))) {
      stop("Attribute 'y' must contain non-negative integer counts")
    }
  }

  # 12. Validate model type
  expected_type <- c("locallevel", "localtrend", "localacceleration")[order]
  model_type <- attr(x, "model_type")
  if (is.null(model_type)) {
    stop("Missing required attribute 'model_type'")
  }
  if (!identical(model_type, expected_type)) {
    stop(sprintf(
      "Attribute 'model_type' must be '%s', got '%s'",
      expected_type, as.character(model_type)
    ))
  }

  # 13. Validate optional metadata
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


#' Print a fitted Poisson mixture object
#'
#' @param x The fitted object.
#' @param cls The expected class name.
#' @param digits Integer, decimal places.
#' @param order Integer 1, 2 or 3.
#'
#' @return `x`, invisibly.
#'
#' @keywords internal
#' @noRd
print_poisson_mixture <- function(x, cls, digits, order) {

  # Validate input
  if (!inherits(x, cls)) {
    stop(sprintf("Object must be of class '%s'", cls))
  }

  # Validate and coerce digits parameter
  if (!is.numeric(digits) || length(digits) != 1 || digits < 0) {
    warning("'digits' must be a non-negative scalar; using default digits = 3")
    digits <- 3
  }
  digits <- as.integer(digits)

  cat("\n")
  cat("Poisson Mixture Model with Dynamic Mixture Weights\n")
  cat(strrep("=", 70), "\n\n", sep = "")

  # Model metadata
  cat("Model:\n")
  cat("  Type:              ", attr(x, "model_type"),
      " (", order, ordinal_suffix(order), " order polynomial)\n", sep = "")
  cat("  Link function:     ", attr(x, "link"), "\n", sep = "")
  cat("\n")

  # MCMC metadata
  cat("MCMC:\n")
  cat("  Observations:      ", attr(x, "n_obs"), "\n", sep = "")
  cat("  Draws retained:    ", attr(x, "n_draws"), "\n", sep = "")
  cat("  Burn-in:           ", attr(x, "burnin"), "\n", sep = "")
  cat("  Thinning:          ", attr(x, "thinning"), "\n", sep = "")
  # Which version and seed produced this fit. Several defaults have moved
  # across releases, so a saved object needs to say where it came from.
  cat("  pdm version:       ", attr(x, "pdm_version"), "\n", sep = "")
  cat("  Seed:              ",
      if (is.null(attr(x, "seed"))) "not set" else attr(x, "seed"),
      "\n\n", sep = "")

  # Determine field width for alignment
  field_width <- digits + 4
  fmt <- paste0("%", field_width, ".", digits, "f")

  # Calculate medians with error handling
  meds <- tryCatch({
    vapply(poisson_mixture_components(order)$scalars,
           function(nm) median(x[[nm]], na.rm = FALSE),
           numeric(1L))
  }, error = function(e) {
    stop("Error calculating posterior medians: ", e$message, call. = FALSE)
  })

  # Posterior medians (mixture components)
  cat("Posterior Medians (Mixture Components):\n")
  cat("  lambda_1:  ", sprintf(fmt, meds[["lambda_1"]]),
      "  (component 1 rate)\n", sep = "")
  cat("  lambda_2:  ", sprintf(fmt, meds[["lambda_2"]]),
      "  (component 2 rate)\n\n", sep = "")

  # Posterior medians (dynamic state)
  state_labels <- c("initial level", "initial trend", "initial acceleration")
  prec_labels  <- c("level innovation precision",
                    "trend innovation precision",
                    "acceleration innovation precision")

  cat("Posterior Medians (Dynamic State):\n")
  for (j in seq_len(order)) {
    cat("  theta_0", j, ":   ", sprintf(fmt, meds[[paste0("theta_0", j)]]),
        "  (", state_labels[j], ")\n", sep = "")
  }
  for (j in seq_len(order)) {
    cat("  W_", j, "^-1:     ", sprintf(fmt, meds[[paste0("prec_theta", j)]]),
        "  (", prec_labels[j], ")\n", sep = "")
  }
  cat("\n")

  # Summary of time-varying alpha with error handling
  tryCatch({
    alpha_median_time <- apply(x$alpha, 2, median, na.rm = FALSE)

    cat("Mixture Weights (alpha_t):\n")
    cat("  Range:  [", sprintf(fmt, min(alpha_median_time)), ", ",
        sprintf(fmt, max(alpha_median_time)), "]\n", sep = "")
    cat("  Median: ", sprintf(fmt, median(alpha_median_time)), "\n\n", sep = "")
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


#' Ordinal suffix for a small positive integer
#'
#' Used only to render "1st"/"2nd"/"3rd" in the printed model header, matching
#' the wording the Gaussian mixture print methods spell out by hand.
#'
#' @param k Integer 1, 2 or 3.
#'
#' @return "st", "nd" or "rd".
#'
#' @keywords internal
#' @noRd
ordinal_suffix <- function(k) {
  c("st", "nd", "rd")[k]
}
