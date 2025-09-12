#-------------------------------------------------------------------------------
# Visualization helpers: parameters and states (generalized for N components)
# - This module includes:
#   (a) Generic parameter/state plotting utilities (backward compatible)
#   (b) Per-level parameter triptychs using expression() via bquote:
#       For each level k, produce a 2x3 figure:
#        • Row 1: theta_0k  -> Histogram, Trace, ACF
#        • Row 2: prec_k    -> Histogram, Trace, ACF
#   (c) Improved state labels using expression for theta[t,k] with robust k-detection
#   (d) Figure titles for per-level panels positioned above all subplots (outer mtext)
#-------------------------------------------------------------------------------

#------------------------------#
# Utilities for label handling #
#------------------------------#

#' Internal: build an expression label from a conventional parameter name
#' Recognized patterns:
#'  - "theta_0k"  (e.g., "theta_01", "theta_02", ...)
#'  - "prec_k"    (e.g., "prec_1",  "prec_2", ...)
#'  - "prec_y"    -> 1/V
#' Returns an expression object, or NULL if not recognized.
.param_label_expr <- function(param_name) {
  # theta_0k
  if (grepl("^theta_0[0-9]+$", param_name)) {
    k <- as.integer(sub("^theta_0([0-9]+)$", "\\1", param_name))
    # Use single subscript "0k" (e.g., theta[01]) as requested
    return(bquote(theta[.(paste0("0", k))]))
  }
  # prec_k
  if (grepl("^prec_[0-9]+$", param_name)) {
    k <- as.integer(sub("^prec_([0-9]+)$", "\\1", param_name))
    return(bquote(1/W[.(k)]))
  }
  # prec_y
  if (identical(param_name, "prec_y")) {
    return(bquote(1/V))
  }
  # Fallback
  return(NULL)
}

#' Internal: try to extract state level k from a string containing "theta"
#' Examples matched (k = 2): "theta_2", "theta 2", "theta[2]", "theta_2 (trend)"
.extract_k_from_text <- function(txt) {
  if (is.null(txt) || !is.character(txt) || length(txt) == 0) return(NA_integer_)
  m <- regexpr("theta\\s*(?:\\[|_)?\\s*([0-9]+)", txt, perl = TRUE, ignore.case = TRUE)
  if (m[1] > 0) {
    k_str <- regmatches(txt, m)
    # Keep only the captured group (the digits)
    k <- as.integer(sub(".*?(\\d+).*", "\\1", k_str))
    if (is.finite(k)) return(k)
  }
  NA_integer_
}

#' Internal: build an expression label for a state given available hints
#' Tries, in order: explicit k (if provided), state_name, list key (nm).
#' Returns an expression(theta[t,k]) or NULL.
.state_label_expr <- function(nm, state_name = NULL, k_hint = NULL) {
  k <- NA_integer_
  if (!is.null(k_hint) && is.finite(k_hint)) k <- as.integer(k_hint)
  if (!is.finite(k)) k <- .extract_k_from_text(state_name)
  if (!is.finite(k)) k <- .extract_k_from_text(nm)
  if (is.finite(k)) {
    # Use substitute to robustly build theta[t, k] so k is never dropped
    return(as.expression(substitute(theta[t, kk], list(kk = k))))
  }
  NULL
}

#-----------------------------------------------#
# Generic parameter plots (posterior + trace)   #
#-----------------------------------------------#

#' Plot posterior + trace for a single parameter
#' @param chain numeric vector of MCMC samples for the parameter
#' @param true_value true parameter value (used as reference line)
#' @param param_name label for the parameter (string)
#' @param color histogram color
#' @param use_expressions if TRUE, use expression labels when possible
#' @param expression_label optional explicit expression label (overrides auto)
create_parameter_plots <- function(chain,
                                   true_value,
                                   param_name,
                                   color = "lightblue",
                                   use_expressions = FALSE,
                                   expression_label = NULL) {
  op <- par(no.readonly = TRUE); on.exit(par(op), add = TRUE)
  par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

  label_expr <- NULL
  if (use_expressions) {
    label_expr <- if (!is.null(expression_label)) expression_label else .param_label_expr(param_name)
  }

  # Posterior
  if (!is.null(label_expr)) {
    hist(chain,
         main = bquote("Posterior: " ~ .(label_expr)),
         xlab = label_expr, col = color, border = "white", probability = TRUE)
  } else {
    hist(chain,
         main = paste("Posterior:", param_name),
         xlab = param_name, col = color, border = "white", probability = TRUE)
  }
  if (is.finite(true_value)) abline(v = true_value, col = "red", lwd = 2, lty = 2)
  abline(v = median(chain), col = "blue", lwd = 2)
  legend("topright", legend = c("True", "Median"), col = c("red", "blue"),
         lty = c(2, 1), lwd = 2, bty = "n")

  # Trace
  if (!is.null(label_expr)) {
    plot(chain, type = 'l',
         main = bquote("Trace: " ~ .(label_expr)),
         col = "blue", xlab = "Iteration", ylab = label_expr)
  } else {
    plot(chain, type = 'l',
         main = paste("Trace:", param_name),
         col = "blue", xlab = "Iteration", ylab = param_name)
  }
  if (is.finite(true_value)) abline(h = true_value, col = "red", lty = 2, lwd = 2)
}

#-------------------------------------------#
# State trajectory with 95% CI (true vs est)#
#-------------------------------------------#

#' Plot state trajectory with 95% CI (true vs estimated)
#' @param state_name label for legend/axes (string)
#' @param true_states numeric vector
#' @param estimated_states numeric vector
#' @param ci_lower numeric vector
#' @param ci_upper numeric vector
#' @param n_plot number of initial points to plot
#' @param title optional custom title
#' @param ylab optional custom y-label (string; ignored if label_expr provided)
#' @param label_expr optional expression to use in y-axis and title
plot_state_with_ci <- function(state_name,
                               true_states,
                               estimated_states,
                               ci_lower,
                               ci_upper,
                               n_plot = 200,
                               title = NULL,
                               ylab = NULL,
                               label_expr = NULL) {
  n <- length(true_states)
  time_idx <- 1:min(n_plot, n)

  # Build title and y-label
  if (!is.null(label_expr)) {
    ttl <- if (is.null(title)) bquote("State: " ~ .(label_expr) ~ " (True vs. Estimated)") else title
    ylab_final <- label_expr
  } else {
    ttl <- if (is.null(title)) paste("State:", state_name, "(True vs. Estimated)") else title
    ylab_final <- if (is.null(ylab)) state_name else ylab
  }

  plot(time_idx, true_states[time_idx], type = 'l', col = "red", lty = 2, lwd = 2,
       main = ttl, xlab = "Time", ylab = ylab_final,
       ylim = range(c(true_states[time_idx], ci_lower[time_idx], ci_upper[time_idx])))
  lines(time_idx, estimated_states[time_idx], col = "blue", lwd = 1.5)
  polygon(c(time_idx, rev(time_idx)),
          c(ci_lower[time_idx], rev(ci_upper[time_idx])),
          col = rgb(0, 0, 1, 0.2), border = NA)
  legend("topleft", legend = c("True", "Est.", "95% CI"),
         col = c("red", "blue", rgb(0,0,1,0.2)),
         lty = c(2,1,1), lwd = c(2,1.5,8), bty = "n")
}

#------------------------------------------------#
# Multi-state plots (grid of state trajectories)  #
#------------------------------------------------#

#' Generate plots for an arbitrary number of states (one panel per state)
#' Distinguishes levels automatically and shows theta[t,k] in titles/ylabs.
#' @param states named list; each item contains true, estimate, ci_lower, ci_upper
#'        Optional per-item fields: state_name, k (numeric), label_expr (expression)
#' @param n_plot number of initial points per state
#' @param use_state_expressions if TRUE, improve labels using expression(theta[t,k])
generate_state_plots <- function(states, n_plot = 200, use_state_expressions = TRUE) {
  stopifnot(is.list(states), length(states) > 0)

  k <- length(states)
  nrow_layout <- ceiling(sqrt(k))
  ncol_layout <- ceiling(k / nrow_layout)

  op <- par(no.readonly = TRUE); on.exit(par(op), add = TRUE)
  par(mfrow = c(nrow_layout, ncol_layout), mar = c(4, 4, 3, 1))

  for (nm in names(states)) {
    s <- states[[nm]]

    # Derive an expression label, preferring:
    # 1) explicit label_expr
    # 2) explicit k (s$k), then state_name, then key nm
    label_expr <- NULL
    if (use_state_expressions) {
      if (!is.null(s$label_expr)) {
        label_expr <- s$label_expr
      } else {
        k_hint <- if (!is.null(s$k) && is.finite(s$k)) as.integer(s$k) else NULL
        label_expr <- .state_label_expr(nm = nm, state_name = s$state_name, k_hint = k_hint)
      }
    }

    plot_state_with_ci(
      state_name       = if (!is.null(s$state_name)) s$state_name else nm,
      true_states      = s$true,
      estimated_states = s$estimate,
      ci_lower         = s$ci_lower,
      ci_upper         = s$ci_upper,
      n_plot           = n_plot,
      label_expr       = label_expr
    )
  }
}

#-------------------------------------------------------------#
# Per-level parameter triptychs (theta_0k and prec_k panels)  #
#-------------------------------------------------------------#

#' Internal: draw histogram panel
.draw_hist <- function(chain, true_value, label_expr, col_fill = "lightblue") {
  hist(chain, probability = TRUE, col = col_fill, border = "white",
       main = bquote("Posterior: " ~ .(label_expr)),
       xlab = label_expr)
  abline(v = median(chain), col = "blue", lwd = 2)
  if (is.finite(true_value)) abline(v = true_value, col = "red", lwd = 2, lty = 2)
  legend("topright",
         legend = c("True", "Median"),
         col = c("red", "blue"), lty = c(2, 1), lwd = 2, bty = "n")
}

#' Internal: draw trace panel
.draw_trace <- function(chain, true_value, label_expr, col_line = "blue") {
  plot(chain, type = "l",
       main = bquote("Trace: " ~ .(label_expr)),
       xlab = "Iteration", ylab = label_expr, col = col_line)
  if (is.finite(true_value)) abline(h = true_value, col = "red", lwd = 2, lty = 2)
}

#' Internal: draw ACF panel
.draw_acf <- function(chain, label_expr, max_lag_acf = 60, col_stem = "steelblue") {
  acf_obj <- acf(chain, plot = FALSE, lag.max = max_lag_acf)
  lags <- as.numeric(acf_obj$lag)[-1]
  acfs <- as.numeric(acf_obj$acf)[-1]
  n    <- length(chain)
  ci   <- qnorm(0.975) / sqrt(n)

  plot(lags, acfs, type = "h", lwd = 2, col = col_stem,
       main = bquote("ACF: " ~ .(label_expr)),
       xlab = "Lag", ylab = "ACF", ylim = c(min(-ci, min(acfs, na.rm = TRUE)),
                                            max(ci,  max(acfs, na.rm = TRUE))))
  abline(h = 0, col = "gray40")
  abline(h = c(-ci, ci), col = "gray60", lty = 3)
}

#' Internal: draw the 3-plot set (hist, trace, acf) for a parameter
.param_triptych <- function(chain, true_value, label_expr,
                            col_fill = "lightblue", col_line = "blue",
                            max_lag_acf = 60) {
  .draw_hist(chain, true_value, label_expr, col_fill)
  .draw_trace(chain, true_value, label_expr, col_line)
  .draw_acf(chain, label_expr, max_lag_acf)
}

#' Generate one figure (2x3) per level k with theta_0k and prec_k triptychs
#' The figure title is drawn above all six subplots (using outer margins).
#' @param param_chains named list of numeric chains (theta_01, theta_02, ..., prec_1, prec_2, ...)
#' @param true_values named numeric vector of true values (optional)
#' @param max_lag_acf maximum lag for ACF
#' @param palette optional list with colors per param type: list(theta_fill, theta_line, prec_fill, prec_line)
generate_level_parameter_triptychs <- function(param_chains,
                                               true_values = NULL,
                                               max_lag_acf = 60,
                                               palette = NULL) {
  stopifnot(is.list(param_chains), length(param_chains) > 0)

  # Default colors
  if (is.null(palette)) {
    palette <- list(
      theta_fill = "lightblue",
      theta_line = "blue",
      prec_fill  = "lightcoral",
      prec_line  = "darkred"
    )
  }

  # Detect available levels by chain names
  theta_names <- grep("^theta_0[0-9]+$", names(param_chains), value = TRUE)
  prec_names  <- grep("^prec_[0-9]+$",     names(param_chains), value = TRUE)

  lev_theta <- suppressWarnings(as.integer(sub("^theta_0([0-9]+)$", "\\1", theta_names)))
  lev_prec  <- suppressWarnings(as.integer(sub("^prec_([0-9]+)$",    "\\1", prec_names)))
  levels_k  <- sort(intersect(lev_theta, lev_prec))

  if (length(levels_k) == 0) {
    cat("No matching pairs (theta_0k, prec_k) found in param_chains.\n")
    return(invisible(NULL))
  }

  for (k in levels_k) {
    theta_key <- sprintf("theta_0%d", k)
    prec_key  <- sprintf("prec_%d", k)

    if (!all(c(theta_key, prec_key) %in% names(param_chains))) next

    theta_chain <- param_chains[[theta_key]]
    prec_chain  <- param_chains[[prec_key]]

    theta_true <- if (!is.null(true_values) && theta_key %in% names(true_values))
      true_values[[theta_key]] else NA_real_
    prec_true  <- if (!is.null(true_values) && prec_key %in% names(true_values))
      true_values[[prec_key]] else NA_real_

    # Set per-figure layout with outer margins for a global title
    op <- par(no.readonly = TRUE); on.exit(par(op), add = TRUE)
    par(mfrow = c(2, 3), mar = c(4, 4, 3, 1), oma = c(0, 0, 4, 0))

    # Row 1: theta_0k (single subscript "0k")
    theta_expr <- bquote(theta[.(paste0("0", k))])
    .param_triptych(theta_chain, theta_true, theta_expr,
                    col_fill = palette$theta_fill,
                    col_line = palette$theta_line,
                    max_lag_acf = max_lag_acf)

    # Row 2: prec_k
    prec_expr <- bquote(1/W[.(k)])
    .param_triptych(prec_chain, prec_true, prec_expr,
                    col_fill = palette$prec_fill,
                    col_line = palette$prec_line,
                    max_lag_acf = max_lag_acf)

    # Common title above all subplots (expression-based)
    mtext(
      text = bquote("Level k = " ~ .(k) ~ " (" ~ theta[.(paste0("0", k))] ~ " and " ~ 1/W[.(k)] ~ ")"),
      side = 3, outer = TRUE, line = 1.2, cex = 1.1
    )
  }

  invisible(NULL)
}

#-----------------------------------------------------------#
# Organized visualization suite: parameters + optional states
#-----------------------------------------------------------#

#' Organized visualization suite:
#' - Optional individual parameter plots (disabled by default)
#' - Per-level triptychs (theta_0k & prec_k) with a single title above all subplots
#' - State plots with improved labels (expression(theta[t,k])) that distinguish k
#' @param param_chains named list of parameter chains
#' @param true_values named vector of true values
#' @param states named list (see generate_state_plots); optional
#' @param use_expressions if TRUE, use expression labels in parameter plots
#' @param per_level_panels if TRUE, produce per-level triptychs (2x3 by level)
#' @param show_individual_param_plots if TRUE, also show simple hist+trace by parameter (default FALSE)
#' @param max_lag_acf maximum lag for ACF in per-level panels
#' @param level_palette color palette for per-level triptychs
#' @param use_state_expressions if TRUE, use expression(theta[t,k]) in state plots
generate_organized_plots <- function(param_chains,
                                     true_values,
                                     states = NULL,
                                     use_expressions = TRUE,
                                     per_level_panels = TRUE,
                                     show_individual_param_plots = FALSE,
                                     max_lag_acf = 60,
                                     level_palette = NULL,
                                     use_state_expressions = TRUE) {
  cat("=== GENERATING ORGANIZED VISUALIZATIONS ===\n\n")

  # Parameters (generic hist + trace) — optional
  if (show_individual_param_plots && !is.null(param_chains) && length(param_chains) > 0) {
    cat("Generating plots: Model Parameters (individual)...\n")
    for (nm in names(param_chains)) {
      expr_label <- if (use_expressions) .param_label_expr(nm) else NULL
      create_parameter_plots(
        chain = param_chains[[nm]],
        true_value = if (nm %in% names(true_values)) true_values[[nm]] else NA_real_,
        param_name = nm,
        color = "lightblue",
        use_expressions = use_expressions,
        expression_label = expr_label
      )
    }
  }

  # Per-level triptychs (theta_0k and prec_k), each with a global title above all subplots
  if (per_level_panels && !is.null(param_chains) && length(param_chains) > 0) {
    cat("Generating per-level triptychs: theta_0k and prec_k (Histogram, Trace, ACF)...\n")
    generate_level_parameter_triptychs(
      param_chains  = param_chains,
      true_values   = true_values,
      max_lag_acf   = max_lag_acf,
      palette       = level_palette
    )
  }

  # States (with improved expression labels that distinguish the level k)
  if (!is.null(states) && length(states) > 0) {
    cat("Generating plots: State Estimates...\n")
    generate_state_plots(states, use_state_expressions = use_state_expressions)
  }

  cat("=== VISUALIZATION COMPLETED ===\n\n")
}
