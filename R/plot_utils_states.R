#' Generic Dynamic State Plotting Utilities
#'
#' @description Internal utility functions for plotting dynamic state
#'   trajectories, innovations, and state-space relationships across
#'   different polynomial orders in the pdm package.
#'
#' @details This file contains shared logic for:
#'   \itemize{
#'     \item Extracting state matrices based on model order
#'     \item Computing state summaries (medians, credible intervals)
#'     \item Generic state trajectory plotting
#'     \item Innovation sequence visualization
#'     \item State-space relationship plots
#'   }
#'
#'   These functions are not exported and are intended for internal use only
#'   by the plot.* methods.
#'
#' @keywords internal
#' @noRd
NULL


# =============================================================================
# State Matrix Extraction
# =============================================================================

#' Extract state matrices based on model order
#'
#' @description Retrieves all dynamic state matrices (theta_*) from a MCMC
#'   object based on its polynomial order.
#'
#' @param x An object inheriting from "pdm_mcmc".
#' @param model_order Integer: 1, 2, or 3. If NULL, will be auto-detected
#'   from x.
#'
#' @return A named list of state matrices:
#'   \describe{
#'     \item{theta_1}{Matrix (n_chain x n_obs) for level/first state}
#'     \item{theta_2}{Matrix for trend (order >= 2 only)}
#'     \item{theta_3}{Matrix for acceleration (order 3 only)}
#'   }
#'
#' @keywords internal
#' @noRd
get_state_matrices <- function(x, model_order = NULL) {

  # Auto-detect model order if not provided
  if (is.null(model_order)) {
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
  }

  # Validate model_order
  if (!model_order %in% c(1L, 2L, 3L)) {
    stop("`model_order` must be 1, 2, or 3")
  }

  # Extract state matrices
  states <- list(theta_1 = x$theta_1)

  if (model_order >= 2L) {
    states$theta_2 <- x$theta_2
  }

  if (model_order >= 3L) {
    states$theta_3 <- x$theta_3
  }

  return(states)
}


#' Get number of states for a given model order
#'
#' @description Returns the number of dynamic states based on polynomial order.
#'
#' @param model_order Integer: 1, 2, or 3.
#'
#' @return Integer: number of states (same as model_order).
#'
#' @keywords internal
#' @noRd
get_n_states <- function(model_order) {
  if (!model_order %in% c(1L, 2L, 3L)) {
    stop("`model_order` must be 1, 2, or 3")
  }
  return(model_order)
}


# =============================================================================
# State Summary Functions
# =============================================================================

#' Compute summary statistics for a state matrix
#'
#' @description Calculates median and credible intervals for each time point
#'   of a dynamic state matrix.
#'
#' @param state_matrix Matrix (n_chain x n_obs) of MCMC samples.
#' @param ci Logical; whether to compute credible intervals.
#' @param ci_level Numeric between 0 and 1; credible interval level.
#'
#' @return A list with components:
#'   \describe{
#'     \item{median}{Numeric vector of length n_obs}
#'     \item{lower}{Numeric vector (if ci = TRUE)}
#'     \item{upper}{Numeric vector (if ci = TRUE)}
#'   }
#'
#' @keywords internal
#' @noRd
summarise_state <- function(state_matrix, ci = TRUE, ci_level = 0.95) {

  # Validate ci_level
  if (ci && (!is.numeric(ci_level) || length(ci_level) != 1 ||
             ci_level <= 0 || ci_level >= 1)) {
    stop("`ci_level` must be a single numeric value between 0 and 1")
  }

  # Compute median
  med <- apply(state_matrix, 2, stats::median)

  # Compute credible intervals if requested (Highest Posterior Density)
  if (ci) {
    ci_mat <- hpdi(state_matrix, ci_level)
    lower <- ci_mat[, "lower"]
    upper <- ci_mat[, "upper"]
  } else {
    lower <- NULL
    upper <- NULL
  }

  list(median = med, lower = lower, upper = upper)
}


#' Summarise all states in a model
#'
#' @description Computes summaries for all dynamic states based on model order.
#'
#' @param x An object inheriting from "pdm_mcmc".
#' @param model_order Integer: 1, 2, or 3. If NULL, auto-detected.
#' @param ci Logical; whether to compute credible intervals.
#' @param ci_level Numeric between 0 and 1; credible interval level.
#'
#' @return A named list where each element is the output of
#'   `summarise_state()` for the corresponding state.
#'
#' @keywords internal
#' @noRd
summarise_all_states <- function(x, model_order = NULL, ci = TRUE,
                                 ci_level = 0.95) {

  # Get state matrices
  states <- get_state_matrices(x, model_order)

  # Summarise each state
  summaries <- lapply(states, summarise_state, ci = ci, ci_level = ci_level)

  return(summaries)
}


# =============================================================================
# Innovation Computation
# =============================================================================

#' Compute innovations for all states
#'
#' @description Calculates innovation sequences (u_t) for all dynamic states
#'   based on the state-space model structure.
#'
#' @param x An object inheriting from "pdm_mcmc".
#' @param model_order Integer: 1, 2, or 3. If NULL, auto-detected.
#'
#' @return A named list of innovation matrices (n_chain x n_obs):
#'   \describe{
#'     \item{innov_1}{Level innovations}
#'     \item{innov_2}{Trend innovations (order >= 2)}
#'     \item{innov_3}{Acceleration innovations (order 3)}
#'   }
#'
#' @details For a local-level model (order 1):
#'   \deqn{u_{t,1} = \theta_{t,1} - \theta_{t-1,1}}
#'
#'   For local-trend (order 2):
#'   \deqn{u_{t,1} = \theta_{t,1} - \theta_{t-1,1} - \theta_{t-1,2}}
#'   \deqn{u_{t,2} = \theta_{t,2} - \theta_{t-1,2}}
#'
#'   For local-acceleration (order 3):
#'   \deqn{u_{t,1} = \theta_{t,1} - \theta_{t-1,1} - \theta_{t-1,2}}
#'   \deqn{u_{t,2} = \theta_{t,2} - \theta_{t-1,2} - \theta_{t-1,3}}
#'   \deqn{u_{t,3} = \theta_{t,3} - \theta_{t-1,3}}
#'
#' @keywords internal
#' @noRd
compute_innovations <- function(x, model_order = NULL) {

  # Auto-detect model order
  if (is.null(model_order)) {
    model_type <- attr(x, "model_type")
    model_order <- switch(
      model_type,
      "locallevel" = 1L,
      "localtrend" = 2L,
      "localacceleration" = 3L,
      stop("Unknown model_type: ", model_type)
    )
  }

  n_obs <- attr(x, "n_obs")

  innovations <- list()

  # Order 1: Simple random walk
  if (model_order == 1L) {
    # u_{t,1} = theta_{t,1} - theta_{t-1,1}
    innovations$innov_1 <- x$theta_1
    innovations$innov_1[, 1] <- x$theta_1[, 1] - x$theta_01

    if (n_obs > 1) {
      innovations$innov_1[, 2:n_obs] <-
        x$theta_1[, 2:n_obs, drop = FALSE] -
        x$theta_1[, 1:(n_obs - 1), drop = FALSE]
    }
  }

  # Order 2: Local-trend
  if (model_order == 2L) {
    # Level innovations: u_{t,1} = theta_{t,1} - theta_{t-1,1} - theta_{t-1,2}
    innovations$innov_1 <- x$theta_1
    innovations$innov_1[, 1] <- x$theta_1[, 1] - (x$theta_01 + x$theta_02)

    if (n_obs > 1) {
      innovations$innov_1[, 2:n_obs] <-
        x$theta_1[, 2:n_obs, drop = FALSE] -
        x$theta_1[, 1:(n_obs - 1), drop = FALSE] -
        x$theta_2[, 1:(n_obs - 1), drop = FALSE]
    }

    # Trend innovations: u_{t,2} = theta_{t,2} - theta_{t-1,2}
    innovations$innov_2 <- x$theta_2
    innovations$innov_2[, 1] <- x$theta_2[, 1] - x$theta_02

    if (n_obs > 1) {
      innovations$innov_2[, 2:n_obs] <-
        x$theta_2[, 2:n_obs, drop = FALSE] -
        x$theta_2[, 1:(n_obs - 1), drop = FALSE]
    }
  }

  # Order 3: Local-acceleration
  if (model_order == 3L) {
    # Level innovations
    innovations$innov_1 <- x$theta_1
    innovations$innov_1[, 1] <- x$theta_1[, 1] - (x$theta_01 + x$theta_02)

    if (n_obs > 1) {
      innovations$innov_1[, 2:n_obs] <-
        x$theta_1[, 2:n_obs, drop = FALSE] -
        x$theta_1[, 1:(n_obs - 1), drop = FALSE] -
        x$theta_2[, 1:(n_obs - 1), drop = FALSE]
    }

    # Trend innovations
    innovations$innov_2 <- x$theta_2
    innovations$innov_2[, 1] <- x$theta_2[, 1] - (x$theta_02 + x$theta_03)

    if (n_obs > 1) {
      innovations$innov_2[, 2:n_obs] <-
        x$theta_2[, 2:n_obs, drop = FALSE] -
        x$theta_2[, 1:(n_obs - 1), drop = FALSE] -
        x$theta_3[, 1:(n_obs - 1), drop = FALSE]
    }

    # Acceleration innovations
    innovations$innov_3 <- x$theta_3
    innovations$innov_3[, 1] <- x$theta_3[, 1] - x$theta_03

    if (n_obs > 1) {
      innovations$innov_3[, 2:n_obs] <-
        x$theta_3[, 2:n_obs, drop = FALSE] -
        x$theta_3[, 1:(n_obs - 1), drop = FALSE]
    }
  }

  return(innovations)
}


# =============================================================================
# Base Graphics Plotting Functions
# =============================================================================

#' Plot single state trajectory with base graphics
#'
#' @description Creates a trajectory plot for a single dynamic state with
#'   optional credible bands and true values.
#'
#' @param time_grid Numeric vector of time indices.
#' @param median Numeric vector of median values.
#' @param lower Numeric vector of lower CI bounds (or NULL).
#' @param upper Numeric vector of upper CI bounds (or NULL).
#' @param ylab Expression or character for y-axis label.
#' @param main Character string for plot title.
#' @param col Color for median line.
#' @param ci Logical; whether to display credible intervals.
#' @param ci_label Character string for CI legend label.
#' @param true_state Numeric vector of true state values (or NULL).
#'   If provided, overlays the true trajectory for simulation validation.
#'
#' @return NULL (invisibly). Function is called for side effects (plotting).
#'
#' @keywords internal
#' @noRd
plot_state_trajectory_base <- function(time_grid, median, lower = NULL,
                                       upper = NULL, ylab, main, col,
                                       ci = TRUE, ci_label = NULL,
                                       true_state = NULL) {

  # Determine y-axis range (include true_state if provided)
  if (ci && !is.null(lower)) {
    range_vals <- range(c(lower, upper))
  } else {
    range_vals <- range(median)
  }

  # Include true_state in range calculation
  if (!is.null(true_state)) {
    range_vals <- range(c(range_vals, true_state))
  }

  if (diff(range_vals) == 0) {
    range_vals <- range_vals + c(-0.5, 0.5)
  }
  range_vals[2] <- range_vals[2] + 0.25 * diff(range_vals)

  # Base plot
  plot(time_grid, median, type = "l", lwd = 2, col = col,
       xlab = "Time", ylab = ylab, main = main, ylim = range_vals)

  # Add credible band
  if (ci && !is.null(lower)) {
    polygon(c(time_grid, rev(time_grid)),
            c(lower, rev(upper)),
            col = grDevices::adjustcolor(col, alpha.f = 0.2), border = NA)
    lines(time_grid, median, lwd = 2, col = col)
  }

  # Add true state trajectory (if provided)
  if (!is.null(true_state)) {
    lines(time_grid, true_state, lwd = 2, col = "black", lty = 2)
  }

  grid()

  # Legend
  legend_items <- c("Median")
  legend_cols <- c(col)
  legend_lty <- c(1)
  legend_lwd <- c(2)

  if (ci && !is.null(ci_label)) {
    legend_items <- c(legend_items, ci_label)
    legend_cols <- c(legend_cols, grDevices::adjustcolor(col, alpha.f = 0.2))
    legend_lty <- c(legend_lty, 1)
    legend_lwd <- c(legend_lwd, 8)
  }

  if (!is.null(true_state)) {
    legend_items <- c(legend_items, "True State")
    legend_cols <- c(legend_cols, "black")
    legend_lty <- c(legend_lty, 2)
    legend_lwd <- c(legend_lwd, 2)
  }

  legend("topright",
         legend = legend_items,
         col = legend_cols,
         lty = legend_lty,
         lwd = legend_lwd,
         horiz = TRUE,
         bty = "n")

  invisible(NULL)
}


#' Plot innovation sequence with base graphics
#'
#' @description Creates a bar plot of innovation sequences with optional
#'   credible bands.
#'
#' @param time_grid Numeric vector of time indices.
#' @param median Numeric vector of median innovation values.
#' @param lower Numeric vector of lower CI bounds (or NULL).
#' @param upper Numeric vector of upper CI bounds (or NULL).
#' @param ylab Expression or character for y-axis label.
#' @param main Character string for plot title.
#' @param col_bar Color for innovation bars.
#' @param ci Logical; whether to display credible intervals.
#' @param ci_label Character string for CI legend label.
#'
#' @return NULL (invisibly). Function is called for side effects (plotting).
#'
#' @keywords internal
#' @noRd
plot_innovation_base <- function(time_grid, median, lower = NULL, upper = NULL,
                                 ylab, main, col_bar, ci = TRUE,
                                 ci_label = NULL) {

  # Determine y-axis range
  if (ci && !is.null(lower)) {
    range_vals <- range(c(lower, upper))
  } else {
    range_vals <- range(median)
  }

  if (diff(range_vals) == 0) {
    range_vals <- range_vals + c(-0.5, 0.5)
  }
  range_vals[2] <- range_vals[2] + 0.3 * diff(range_vals)

  # Base plot (using type = "h" for vertical bars)
  plot(time_grid, median, type = "h", lwd = 2, col = col_bar,
       xlab = "Time", ylab = ylab, main = main, ylim = range_vals)

  # Add credible band
  if (ci && !is.null(lower)) {
    polygon(c(time_grid, rev(time_grid)),
            c(lower, rev(upper)),
            col = grDevices::adjustcolor(col_bar, alpha.f = 0.2), border = NA)
  }

  # Reference line at zero
  abline(h = 0, col = "black", lty = 2, lwd = 2)

  grid()

  # Legend
  if (ci && !is.null(ci_label)) {
    legend("topright",
           legend = c("Median", ci_label),
           col = c(col_bar, grDevices::adjustcolor(col_bar, alpha.f = 0.2)),
           horiz = TRUE,
           lty = c(1, 1),
           lwd = c(2, 8),
           bty = "n")
  } else {
    legend("topright",
           legend = "Median",
           col = col_bar,
           horiz = TRUE,
           lty = 1,
           lwd = 2,
           bty = "n")
  }

  invisible(NULL)
}


# =============================================================================
# State Labels and Metadata
# =============================================================================

#' Get state labels for plotting
#'
#' @description Returns appropriate labels and titles for state plots based
#'   on model order.
#'
#' @param model_order Integer: 1, 2, or 3.
#'
#' @return A list with components:
#'   \describe{
#'     \item{state_names}{Character vector of state names}
#'     \item{state_labels}{List of expressions for axis labels}
#'     \item{state_titles}{Character vector of plot titles}
#'     \item{state_colors}{Character vector of colors}
#'   }
#'
#' @keywords internal
#' @noRd
get_state_labels <- function(model_order) {

  if (!model_order %in% c(1L, 2L, 3L)) {
    stop("`model_order` must be 1, 2, or 3")
  }

  labels <- list(
    state_names = character(0),
    state_labels = expression(theta["t,1"]),
    state_titles = character(0),
    state_colors = character(0)
  )

  # Order 1
  labels$state_names <- c("theta_1")
  # labels$state_labels <- list(labels$state_labels, expression(theta["t,1"]))
  labels$state_titles <- c("Level State")
  labels$state_colors <- c("steelblue")

  # Order 2
  if (model_order >= 2L) {
    labels$state_names <- c(labels$state_names, "theta_2")
    labels$state_labels <- c(labels$state_labels, expression(theta["t,2"]))
    labels$state_titles <- c(labels$state_titles, "Trend State")
    labels$state_colors <- c(labels$state_colors, "firebrick")
  }

  # Order 3
  if (model_order >= 3L) {
    labels$state_names <- c(labels$state_names, "theta_3")
    labels$state_labels <- c(labels$state_labels, expression(theta["t,3"]))
    labels$state_titles <- c(labels$state_titles, "Acceleration State")
    labels$state_colors <- c(labels$state_colors, "darkgreen")
  }

  return(labels)
}


#' Get innovation labels for plotting
#'
#' @description Returns appropriate labels for innovation plots based on
#'   model order.
#'
#' @param model_order Integer: 1, 2, or 3.
#'
#' @return A list with components:
#'   \describe{
#'     \item{innov_labels}{List of expressions for axis labels}
#'     \item{innov_titles}{Character vector of plot titles}
#'     \item{innov_colors}{Character vector of colors}
#'   }
#'
#' @keywords internal
#' @noRd
get_innovation_labels <- function(model_order) {

  if (!model_order %in% c(1L, 2L, 3L)) {
    stop("`model_order` must be 1, 2, or 3")
  }

  labels <- list(
    innov_labels = list(expression(u["t,1"])),
    innov_titles = c("Level Innovations"),
    innov_colors = c("steelblue")
  )

  if (model_order >= 2L) {
    labels$innov_labels <- c(labels$innov_labels, list(expression(u["t,2"])))
    labels$innov_titles <- c(labels$innov_titles, "Trend Innovations")
    labels$innov_colors <- c(labels$innov_colors, "firebrick")
  }

  if (model_order >= 3L) {
    labels$innov_labels <- c(labels$innov_labels, list(expression(u["t,3"])))
    labels$innov_titles <- c(labels$innov_titles, "Acceleration Innovations")
    labels$innov_colors <- c(labels$innov_colors, "darkgreen")
  }

  return(labels)
}
