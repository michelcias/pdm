#' Helper: Compute summary statistics
#'
#' @description Computes mean, SD, median, and credible intervals for a numeric vector.
#'
#' @param x Numeric vector
#' @param ci_level Credible interval level; a single numeric value strictly
#'   between 0 and 1.
#'
#' @return Data frame with summary statistics
#' @keywords internal
#' @noRd
compute_summary_stats <- function(x, ci_level) {
  h <- hpdi(x, ci_level)
  data.frame(
    Mean = mean(x),
    SD = sd(x),
    Median = median(x),
    CI_Lower = h[["lower"]],
    CI_Upper = h[["upper"]],
    row.names = NULL
  )
}


#' Helper: Validate summary input
#'
#' @description Validates object class and ci_level argument for summary methods.
#'
#' @param object An object to validate
#' @param ci_level Credible interval level; a single numeric value strictly
#'   between 0 and 1.
#' @param expected_class Character string with expected class name
#'
#' @return Invisible NULL if validation passes, otherwise stops with error
#' @keywords internal
#' @noRd
validate_summary_input <- function(object, ci_level, expected_class) {
  if (!inherits(object, expected_class)) {
    stop(sprintf("Object must be of class '%s'", expected_class))
  }

  if (!is.numeric(ci_level) || length(ci_level) != 1L ||
      ci_level <= 0 || ci_level >= 1) {
    stop("`ci_level` must be a single numeric value between 0 and 1")
  }

  invisible(NULL)
}


#' Helper: Format mixture parameters summary
#'
#' @description Creates summary statistics for mixture component parameters
#'   (mu_1, mu_2, phi_1, phi_2).
#'
#' @param object An object containing mixture parameters
#' @param ci_level Credible interval level; a single numeric value strictly
#'   between 0 and 1.
#'
#' @return Data frame with mixture parameter summaries
#' @keywords internal
#' @noRd
format_mixture_params <- function(object, ci_level) {
  data.frame(
    Parameter = c("mu_1", "mu_2", "phi_1", "phi_2"),
    rbind(
      compute_summary_stats(object$mu_1, ci_level),
      compute_summary_stats(object$mu_2, ci_level),
      compute_summary_stats(object$prec_1, ci_level),
      compute_summary_stats(object$prec_2, ci_level)
    )
  )
}


#' Helper: Format dynamic state parameters
#'
#' @description Creates summary statistics for dynamic state parameters.
#'   Flexible to handle different polynomial orders (level, trend, acceleration).
#'
#' @param object An object containing state parameters
#' @param ci_level Credible interval level; a single numeric value strictly
#'   between 0 and 1.
#' @param order Character string: "level", "trend", or "acceleration"
#'
#' @return Data frame with state parameter summaries
#' @keywords internal
#' @noRd
format_state_params <- function(object, ci_level, order = c("level", "trend", "acceleration")) {
  order <- match.arg(order)

  # Start with level parameters (always present)
  params_list <- list(
    theta_01 = compute_summary_stats(object$theta_01, ci_level),
    prec_theta1 = compute_summary_stats(object$prec_theta1, ci_level)
  )
  param_names <- c("theta_01", "W_1^-1")

  # Add trend parameters if needed
  if (order %in% c("trend", "acceleration")) {
    params_list <- c(
      list(theta_01 = params_list$theta_01),
      list(theta_02 = compute_summary_stats(object$theta_02, ci_level)),
      list(prec_theta1 = params_list$prec_theta1),
      list(prec_theta2 = compute_summary_stats(object$prec_theta2, ci_level))
    )
    param_names <- c("theta_01", "theta_02", "W_1^-1", "W_2^-1")
  }

  # Add acceleration parameters if needed
  if (order == "acceleration") {
    params_list <- c(
      list(theta_01 = params_list$theta_01),
      list(theta_02 = params_list$theta_02),
      list(theta_03 = compute_summary_stats(object$theta_03, ci_level)),
      list(prec_theta1 = params_list$prec_theta1),
      list(prec_theta2 = params_list$prec_theta2),
      list(prec_theta3 = compute_summary_stats(object$prec_theta3, ci_level))
    )
    param_names <- c("theta_01", "theta_02", "theta_03", "W_1^-1", "W_2^-1", "W_3^-1")
  }

  data.frame(
    Parameter = param_names,
    do.call(rbind, params_list)
  )
}


#' Helper: Format scalar parameters for normal_locallevel
#'
#' @description Creates summary statistics for scalar parameters
#'   (theta_01, W_1^-1, V^-1) in normal_locallevel model.
#'
#' @param object An object containing scalar parameters
#' @param ci_level Credible interval level; a single numeric value strictly
#'   between 0 and 1.
#'
#' @return Data frame with scalar parameter summaries
#' @keywords internal
#' @noRd
format_scalar_params_locallevel <- function(object, ci_level) {
  data.frame(
    Parameter = c("theta_01", "W_1^-1", "V^-1"),
    rbind(
      compute_summary_stats(object$theta_01, ci_level),
      compute_summary_stats(object$prec_theta1, ci_level),
      compute_summary_stats(object$prec_y, ci_level)
    )
  )
}


#' Helper: Format time-varying parameter summary
#'
#' @description Creates summary statistics for time-varying parameters.
#'   Can produce either simple (min/median/max) or detailed (with mean/SD/CI) summaries.
#'
#' @param matrix_param Matrix of parameter samples (n_draws × n_obs)
#' @param ci_level Credible interval level; a single numeric value strictly
#'   between 0 and 1.
#' @param summary_type Character string: "simple" or "detailed"
#'
#' @return Data frame with time-varying parameter summaries
#' @keywords internal
#' @noRd
format_timevarying_summary <- function(matrix_param, ci_level, summary_type = c("simple", "detailed")) {
  summary_type <- match.arg(summary_type)

  # Compute statistics across time
  mean_time <- apply(matrix_param, 2, mean)
  sd_time <- apply(matrix_param, 2, sd)
  median_time <- apply(matrix_param, 2, median)
  ci_mat <- hpdi(matrix_param, ci_level)
  ci_lower_time <- ci_mat[, "lower"]
  ci_upper_time <- ci_mat[, "upper"]

  if (summary_type == "simple") {
    # Simple version (for alpha in mixture models)
    return(data.frame(
      Statistic = c("Min across time", "Median across time", "Max across time"),
      Value = c(
        min(median_time),
        median(median_time),
        max(median_time)
      )
    ))
  }

  # Detailed version (for theta in normal_locallevel)
  data.frame(
    Statistic = c(
      "Min across time",
      "Median across time",
      "Max across time",
      "Final time point"
    ),
    Mean = c(
      min(mean_time),
      median(mean_time),
      max(mean_time),
      tail(mean_time, 1L)
    ),
    SD = c(
      min(sd_time),
      median(sd_time),
      max(sd_time),
      tail(sd_time, 1L)
    ),
    Median = c(
      min(median_time),
      median(median_time),
      max(median_time),
      tail(median_time, 1L)
    ),
    CI_Lower = c(
      min(ci_lower_time),
      median(ci_lower_time),
      max(ci_lower_time),
      tail(ci_lower_time, 1L)
    ),
    CI_Upper = c(
      min(ci_upper_time),
      median(ci_upper_time),
      max(ci_upper_time),
      tail(ci_upper_time, 1L)
    )
  )
}


#' Helper: Print summary header
#'
#' @description Prints model information header for summary output.
#'
#' @param x Summary object
#'
#' @return Invisible NULL (prints to console)
#' @keywords internal
#' @noRd
print_summary_header <- function(x) {
  cat("\n")

  # Title based on model type - only used for mixture models now
  # Non-mixture models print their own titles
  if (!is.null(x$link)) {
    cat("Summary: Gaussian Mixture Model with Dynamic Mixture Weights\n")
    cat(strrep("=", 75), "\n\n", sep = "")
  }

  # Model information
  cat("Model Information:\n")
  cat("  Type:              ", x$model_type, "\n", sep = "")
  if (!is.null(x$link)) {
    cat("  Link function:     ", x$link, "\n", sep = "")
  }
  cat("  Observations:      ", x$n_obs, "\n", sep = "")
  cat("  MCMC samples:      ", x$n_draws, "\n", sep = "")
  cat("  Burn-in:           ", x$burnin, "\n", sep = "")
  cat("  Thinning:          ", x$thinning, "\n\n", sep = "")

  # Credible interval level
  ci_level_pct <- x$ci_level * 100
  cat("Credible Intervals (HPDI): ", sprintf("%.1f", ci_level_pct), "%\n\n", sep = "")

  # Explanation of statistics
  cat("Statistics Legend:\n")
  cat("  Mean   = Posterior mean (minimizes squared error)\n")
  cat("  Median = Posterior median (minimizes absolute error, shown in print())\n")
  cat("  SD     = Posterior standard deviation\n")
  cat("  CI     = HPD interval (shortest interval containing the specified\n")
  cat("           probability mass of the posterior)\n\n")
}


#' Helper: Print formatted table
#'
#' @description Prints a data frame with formatted numeric columns.
#'
#' @param df Data frame to print
#' @param digits Integer, number of significant digits to display
#' @param header Optional character string for section header
#'
#' @return Invisible NULL (prints to console)
#' @keywords internal
#' @noRd
print_formatted_table <- function(df, digits = 3, header = NULL) {
  if (!is.null(header)) {
    cat(header, "\n")
    cat(strrep("-", 75), "\n", sep = "")
  }

  df_print <- df
  numeric_cols <- sapply(df, is.numeric)
  df_print[numeric_cols] <- lapply(df[numeric_cols], function(col) {
    sprintf(paste0("%.", digits, "f"), col)
  })

  print(df_print, row.names = FALSE, right = TRUE)
  cat("\n")
}


#' Helper: Print summary footer
#'
#' @description Prints footer with usage notes for summary output.
#'
#' @return Invisible NULL (prints to console)
#' @keywords internal
#' @noRd
print_summary_footer <- function() {
  cat(strrep("-", 75), "\n", sep = "")
  cat("Note: For time-varying parameters, use plot() to visualize trajectories.\n")
  cat("      For quick overview with medians only, use print().\n")
  cat(strrep("-", 75), "\n\n", sep = "")
}
