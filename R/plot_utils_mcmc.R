#' Generic MCMC Diagnostic Utilities
#'
#' @description Internal utility functions for creating MCMC diagnostic plots
#'   across different model types in the pdm package. These functions provide
#'   a unified interface for plotting trace plots, autocorrelation functions,
#'   posterior densities, and convergence diagnostics.
#'
#' @details This file contains shared logic for:
#'   \itemize{
#'     \item Parameter configuration based on model type and order
#'     \item Generic MCMC diagnostic plot dispatching
#'     \item Coordination between base and ggplot2 graphics engines
#'   }
#'
#'   These functions are not exported and are intended for internal use only
#'   by the plot.* methods.
#'
#' @keywords internal
#' @noRd
NULL


# =============================================================================
# Model Type Detection
# =============================================================================

#' Detect model type from MCMC object
#'
#' @description Extracts model characteristics (observation family and
#'   polynomial order) from a pdm_mcmc object by examining its class names
#'   and attributes.
#'
#' @param x An object inheriting from "pdm_mcmc".
#'
#' @return A list with three components:
#'   \describe{
#'     \item{model_class}{Character: "mixture", "binomial", or "normal"}
#'     \item{model_order}{Integer: 1, 2, or 3 (polynomial order)}
#'     \item{has_mixture}{Logical: TRUE if model has mixture components}
#'   }
#'
#' @details The function determines model_class by examining the class hierarchy:
#'   \itemize{
#'     \item \code{"mixture"}: Classes containing "mixture" (Gaussian mixture models)
#'     \item \code{"binomial"}: Classes containing "binomial" or "bernoulli"
#'       (binomial/Bernoulli models with logit or probit link)
#'     \item \code{"normal"}: Classes containing "normal" (Gaussian observation models)
#'   }
#'
#'   The model_order is extracted from the "model_type" attribute, which must
#'   be one of: "locallevel" (1), "localtrend" (2), or "localacceleration" (3).
#'
#'   Key differences between model classes:
#'   \itemize{
#'     \item Normal models have observation precision parameter (\code{prec_y})
#'     \item Binomial models do NOT have \code{prec_y} (use link functions)
#'     \item Mixture models have 4 mixture parameters instead of \code{prec_y}
#'   }
#'
#' @keywords internal
#' @noRd
detect_model_type <- function(x) {

  # 1. Validate input
  if (!inherits(x, "pdm_mcmc")) {
    stop("Input must inherit from 'pdm_mcmc' class")
  }

  # 2. Detect model class from class names
  classes <- class(x)

  if (any(grepl("mixture", classes, fixed = TRUE))) {
    model_class <- "mixture"
    has_mixture <- TRUE
  } else if (any(grepl("binomial", classes, fixed = TRUE)) ||
             any(grepl("bernoulli", classes, fixed = TRUE))) {
    model_class <- "binomial"
    has_mixture <- FALSE
  } else if (any(grepl("normal", classes, fixed = TRUE))) {
    model_class <- "normal"
    has_mixture <- FALSE
  } else {
    stop("Unknown model class in object. Expected 'mixture', 'binomial', ",
         "'bernoulli', or 'normal' in class hierarchy.")
  }

  # 3. Extract polynomial order from model_type attribute
  model_type <- attr(x, "model_type")
  if (is.null(model_type)) {
    stop("Object must have 'model_type' attribute")
  }

  model_order <- switch(
    model_type,
    "locallevel" = 1L,
    "localtrend" = 2L,
    "localacceleration" = 3L,
    stop("Unknown model_type: ", model_type)
  )

  # 4. Return structured result
  list(
    model_class = model_class,
    model_order = model_order,
    has_mixture = has_mixture
  )
}


#' Get expected number of parameters for model type
#'
#' @description Calculates the total number of scalar parameters for a given
#'   model configuration based on its observation family and polynomial order.
#'
#' @param model_class Character: "mixture", "binomial", or "normal".
#' @param model_order Integer: 1, 2, or 3.
#'
#' @return Integer count of expected parameters.
#'
#' @details Parameter counts by model class:
#'   \itemize{
#'     \item \strong{Mixture models}: 4 mixture params (mu_1, mu_2, phi_1, phi_2) +
#'       model_order initial states + model_order innovation precisions
#'     \item \strong{Normal models}: 1 observation precision (V^{-1}) +
#'       model_order initial states + model_order innovation precisions
#'     \item \strong{Binomial models}: 0 observation precision (uses link function) +
#'       model_order initial states + model_order innovation precisions
#'   }
#'
#'   Examples:
#'   \itemize{
#'     \item mixture order 1 = 4 + 1 + 1 = 6
#'     \item normal order 2 = 1 + 2 + 2 = 5
#'     \item binomial order 3 = 0 + 3 + 3 = 6
#'   }
#'
#' @keywords internal
#' @noRd
get_n_params <- function(model_class, model_order) {

  # Validate inputs
  if (!model_class %in% c("mixture", "binomial", "normal")) {
    stop("`model_class` must be 'mixture', 'binomial', or 'normal'")
  }
  if (!model_order %in% c(1L, 2L, 3L)) {
    stop("`model_order` must be 1, 2, or 3")
  }

  # Calculate based on model type
  if (model_class == "mixture") {
    # 4 mixture components + initial states + innovation precisions
    4L + model_order + model_order
  } else if (model_class == "normal") {
    # 1 observation precision + initial states + innovation precisions
    1L + model_order + model_order
  } else {  # binomial
    # No observation precision + initial states + innovation precisions
    0L + model_order + model_order
  }
}


# =============================================================================
# Parameter Configuration
# =============================================================================

#' Generate parameter configuration for plotting
#'
#' @description Creates a standardized list of parameter metadata for all
#'   scalar parameters in a model, including MCMC samples and plotting
#'   specifications for both base and ggplot2 graphics engines.
#'
#' @param x An object inheriting from "pdm_mcmc".
#' @param model_class Character: "mixture", "binomial", or "normal". If NULL,
#'   will be auto-detected from x.
#' @param model_order Integer: 1, 2, or 3. If NULL, will be auto-detected
#'   from x.
#'
#' @return A named list where each element contains:
#'   \describe{
#'     \item{samples}{Numeric vector of MCMC samples (length n_chain)}
#'     \item{name}{Quoted expression for base graphics titles}
#'     \item{label}{Expression for base graphics axis labels}
#'     \item{name_str}{Character string for ggplot2 internal use}
#'     \item{label_str}{Character string for ggplot2 label mapping}
#'   }
#'
#' @details Parameters are ordered as:
#'   \enumerate{
#'     \item Mixture components (if \code{model_class = "mixture"}): mu_1, mu_2, phi_1, phi_2
#'     \item Observation precision (if \code{model_class = "normal"}): V^{-1}
#'     \item Initial states (all models): theta_01, theta_02, theta_03
#'     \item Innovation precisions (all models): W_1^{-1}, W_2^{-1}, W_3^{-1}
#'   }
#'
#'   This ordering matches the convention in existing plot files.
#'
#'   The W*_inv naming convention maps to prec_theta* in the object:
#'   W1_inv corresponds to x$prec_theta1, etc.
#'
#'   Note: Binomial models (binomial/Bernoulli with logit/probit link) do NOT
#'   have observation precision parameter.
#'
#' @keywords internal
#' @noRd
get_param_config <- function(x,
                             model_class = NULL,
                             model_order = NULL) {

  # 1. Auto-detect model characteristics if not provided
  if (is.null(model_class) || is.null(model_order)) {
    detected <- detect_model_type(x)
    if (is.null(model_class)) model_class <- detected$model_class
    if (is.null(model_order)) model_order <- detected$model_order
  }

  # 2. Validate inputs
  if (!model_class %in% c("mixture", "binomial", "normal")) {
    stop("`model_class` must be 'mixture', 'binomial', or 'normal'")
  }
  if (!model_order %in% c(1L, 2L, 3L)) {
    stop("`model_order` must be 1, 2, or 3")
  }

  # 3. Initialize configuration list
  config <- list()

  # 4. Add mixture components (if applicable)
  if (model_class == "mixture") {
    config$mu_1 <- list(
      samples = x$mu_1,
      name = quote(mu[1]),
      label = expression(mu[1]),
      name_str = "mu_1",
      label_str = "mu_1"
    )

    config$mu_2 <- list(
      samples = x$mu_2,
      name = quote(mu[2]),
      label = expression(mu[2]),
      name_str = "mu_2",
      label_str = "mu_2"
    )

    config$phi_1 <- list(
      samples = x$prec_1,
      name = quote(phi[1]),
      label = expression(phi[1]),
      name_str = "phi_1",
      label_str = "phi_1"
    )

    config$phi_2 <- list(
      samples = x$prec_2,
      name = quote(phi[2]),
      label = expression(phi[2]),
      name_str = "phi_2",
      label_str = "phi_2"
    )
  }

  # 5. Add observation precision (normal models only)
  if (model_class == "normal") {
    config$V_inv <- list(
      samples = x$prec_y,
      name = quote(V^{-1}),
      label = expression(V^{-1}),
      name_str = "V^{-1}",
      label_str = "V^{-1}"
    )
  }

  # 6. Add initial states based on polynomial order
  config$theta_01 <- list(
    samples = x$theta_01,
    name = quote(theta["0,1"]),
    label = expression(theta["0,1"]),
    name_str = "theta_01",
    label_str = "theta_01"
  )

  if (model_order >= 2L) {
    config$theta_02 <- list(
      samples = x$theta_02,
      name = quote(theta["0,2"]),
      label = expression(theta["0,2"]),
      name_str = "theta_02",
      label_str = "theta_02"
    )
  }

  if (model_order >= 3L) {
    config$theta_03 <- list(
      samples = x$theta_03,
      name = quote(theta["0,3"]),
      label = expression(theta["0,3"]),
      name_str = "theta_03",
      label_str = "theta_03"
    )
  }

  # 7. Add innovation precisions
  config$W1_inv <- list(
    samples = x$prec_theta1,
    name = quote(W[1]^{-1}),
    label = expression(W[1]^{-1}),
    name_str = "W_1^{-1}",
    label_str = "W_1^{-1}"
  )

  if (model_order >= 2L) {
    config$W2_inv <- list(
      samples = x$prec_theta2,
      name = quote(W[2]^{-1}),
      label = expression(W[2]^{-1}),
      name_str = "W_2^{-1}",
      label_str = "W_2^{-1}"
    )
  }

  if (model_order >= 3L) {
    config$W3_inv <- list(
      samples = x$prec_theta3,
      name = quote(W[3]^{-1}),
      label = expression(W[3]^{-1}),
      name_str = "W_3^{-1}",
      label_str = "W_3^{-1}"
    )
  }

  # 8. Validate configuration
  validate_param_config(config)

  return(config)
}


#' Validate parameter configuration structure
#'
#' @description Checks that a parameter configuration list has the correct
#'   structure with all required fields for each parameter.
#'
#' @param config A list of parameter specifications.
#'
#' @return TRUE if valid (invisible). Stops with error if invalid.
#'
#' @keywords internal
#' @noRd
validate_param_config <- function(config) {

  # Check that config is a list
  if (!is.list(config)) {
    stop("Parameter configuration must be a list")
  }

  if (length(config) == 0) {
    stop("Parameter configuration cannot be empty")
  }

  # Required fields for each parameter
  required_fields <- c("samples", "name", "label", "name_str", "label_str")

  # Check each parameter
  for (i in seq_along(config)) {
    param_name <- names(config)[i]
    param_info <- config[[i]]

    # Check that parameter info is a list
    if (!is.list(param_info)) {
      stop("Parameter '", param_name, "' must be a list")
    }

    # Check for required fields
    missing_fields <- setdiff(required_fields, names(param_info))
    if (length(missing_fields) > 0) {
      stop("Parameter '", param_name, "' missing fields: ",
           paste(missing_fields, collapse = ", "))
    }

    # Validate samples
    if (!is.numeric(param_info$samples)) {
      stop("Parameter '", param_name, "': samples must be numeric")
    }

    if (any(!is.finite(param_info$samples))) {
      stop("Parameter '", param_name, "': samples contain non-finite values")
    }
  }

  invisible(TRUE)
}


# =============================================================================
# Generic Plotting Dispatchers
# =============================================================================

#' Generic MCMC diagnostics plot dispatcher
#'
#' @description Dispatches MCMC diagnostic plots to the appropriate graphics
#'   engine (base or ggplot2) for one or more parameters.
#'
#' @param x An object inheriting from "pdm_mcmc".
#' @param which Integer vector specifying which parameters to plot. If NULL,
#'   all parameters are plotted.
#' @param param_config Pre-computed parameter configuration list. If NULL,
#'   will be auto-generated from x using \code{get_param_config()}.
#' @param engine Character: "base" or "ggplot2".
#' @param ... Additional arguments passed to plotting functions.
#'
#' @return NULL (invisibly). Function is called for side effects (plotting).
#'
#' @details For each selected parameter, creates a 4-panel diagnostic plot:
#'   trace plot, autocorrelation function, posterior density, and running mean.
#'   The actual plotting is delegated to \code{plot_param_diagnostics_base()}
#'   or \code{plot_param_diagnostics_ggplot()} from plot_utils_base.R and
#'   plot_utils_ggplot.R respectively.
#'
#' @keywords internal
#' @noRd
plot_mcmc_diagnostics_generic <- function(x,
                                          which = NULL,
                                          param_config = NULL,
                                          engine = c("base", "ggplot2"),
                                          true_values = NULL,
                                          ...) {

  # 1. Validate and match engine argument
  engine <- match.arg(engine)

  # 2. Check ggplot2 availability if needed
  if (engine == "ggplot2" && !requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for ggplot2 engine but is not installed")
  }

  # 3. Generate parameter configuration if not provided
  if (is.null(param_config)) {
    param_config <- get_param_config(x)
  }

  # 4. Validate parameter configuration
  validate_param_config(param_config)

  # 5. Set default for which (all parameters)
  if (is.null(which)) {
    which <- seq_along(param_config)
  }

  # 6. Validate which
  if (!is.numeric(which) || any(which != floor(which))) {
    stop("`which` must be an integer vector")
  }

  if (any(which < 1) || any(which > length(param_config))) {
    stop("`which` must be between 1 and ", length(param_config))
  }

  # 7. Plot each selected parameter
  for (i in which) {
    param_info <- param_config[[i]]

    true_value <- NULL

    if (!is.null(true_values) && engine == "base") {
      param_name <- param_info$name_str
      if (param_name == "theta_01") true_value <- true_values$theta_01
      if (param_name == "theta_02") true_value <- true_values$theta_02
      if (param_name == "theta_03") true_value <- true_values$theta_03
      if (param_name == "W_1^{-1}") true_value <- true_values$prec_theta1
      if (param_name == "W_2^{-1}") true_value <- true_values$prec_theta2
      if (param_name == "W_3^{-1}") true_value <- true_values$prec_theta3
    }

    if (engine == "base") {
      plot_param_diagnostics_base(
        param_samples = param_info$samples,
        param_name = param_info$name,
        param_label = param_info$label,
        true_value = true_value,
        ...
      )
    } else {
      # Delegate to ggplot2 function
      p <- plot_param_diagnostics_ggplot(
        param_samples = param_info$samples,
        param_name = param_info$name_str,
        param_label_text = param_info$label_str,
        ...
      )

      # Print plot if not NULL
      if (!is.null(p)) {
        print(p)
      }
    }
  }

  invisible(NULL)
}
