#-------------------------------------------------------------------------------
# Visualization helpers: parameters and states (generalized for N components)
#-------------------------------------------------------------------------------

#' Plot posterior + trace for a single parameter
#' @param chain numeric vector of MCMC samples for the parameter
#' @param true_value true parameter value (used as reference line)
#' @param param_name label for the parameter
#' @param color histogram color
create_parameter_plots <- function(chain, true_value, param_name, color = "lightblue") {
  op <- par(no.readonly = TRUE); on.exit(par(op), add = TRUE)
  par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

  # Posterior
  hist(chain,
       main = paste("Posterior:", param_name),
       xlab = param_name, col = color, border = "white", probability = TRUE)
  if (is.finite(true_value)) abline(v = true_value, col = "red", lwd = 2, lty = 2)
  abline(v = median(chain), col = "blue", lwd = 2)
  legend("topright", legend = c("True", "Median"), col = c("red", "blue"),
         lty = c(2, 1), lwd = 2, bty = "n")

  # Trace
  plot(chain, type = 'l', main = paste("Trace:", param_name),
       col = "blue", xlab = "Iteration", ylab = param_name)
  if (is.finite(true_value)) abline(h = true_value, col = "red", lty = 2, lwd = 2)
}

#' Plot state trajectory with 95% CI (true vs estimated)
#' @param state_name label for legend/axes
#' @param true_states numeric vector
#' @param estimated_states numeric vector
#' @param ci_lower numeric vector
#' @param ci_upper numeric vector
#' @param n_plot number of initial points to plot
#' @param title optional custom title
#' @param ylab optional custom y-label
plot_state_with_ci <- function(state_name,
                               true_states,
                               estimated_states,
                               ci_lower,
                               ci_upper,
                               n_plot = 200,
                               title = NULL,
                               ylab = NULL) {
  n <- length(true_states)
  time_idx <- 1:min(n_plot, n)
  ttl <- if (is.null(title)) paste("State:", state_name, "(True vs. Estimated)") else title
  ylb <- if (is.null(ylab)) state_name else ylab

  plot(time_idx, true_states[time_idx], type = 'l', col = "red", lty = 2, lwd = 2,
       main = ttl, xlab = "Time", ylab = ylb,
       ylim = range(c(true_states[time_idx], ci_lower[time_idx], ci_upper[time_idx])))
  lines(time_idx, estimated_states[time_idx], col = "blue", lwd = 1.5)
  polygon(c(time_idx, rev(time_idx)),
          c(ci_lower[time_idx], rev(ci_upper[time_idx])),
          col = rgb(0, 0, 1, 0.2), border = NA)
  legend("topleft", legend = c("True", "Est.", "95% CI"),
         col = c("red", "blue", rgb(0,0,1,0.2)),
         lty = c(2,1,1), lwd = c(2,1.5,8), bty = "n")
}

#' Generate plots for an arbitrary number of states (one panel per state)
#' @param states named list; each item contains true, estimate, ci_lower, ci_upper
#' @param n_plot number of initial points per state
generate_state_plots <- function(states, n_plot = 200) {
  stopifnot(is.list(states), length(states) > 0)

  k <- length(states)
  nrow_layout <- ceiling(sqrt(k))
  ncol_layout <- ceiling(k / nrow_layout)

  op <- par(no.readonly = TRUE); on.exit(par(op), add = TRUE)
  par(mfrow = c(nrow_layout, ncol_layout), mar = c(4, 4, 3, 1))

  for (nm in names(states)) {
    s <- states[[nm]]
    plot_state_with_ci(
      state_name       = if (!is.null(s$state_name)) s$state_name else nm,
      true_states      = s$true,
      estimated_states = s$estimate,
      ci_lower         = s$ci_lower,
      ci_upper         = s$ci_upper,
      n_plot           = n_plot
    )
  }
}

#' Organized visualization suite: parameters + optional states
#' @param param_chains named list of parameter chains
#' @param true_values named vector of true values
#' @param states named list (see generate_state_plots); optional
generate_organized_plots <- function(param_chains, true_values, states = NULL) {
  cat("=== GENERATING ORGANIZED VISUALIZATIONS ===\n\n")

  # Parameters
  if (!is.null(param_chains) && length(param_chains) > 0) {
    cat("Generating plots: Model Parameters...\n")
    for (nm in names(param_chains)) {
      create_parameter_plots(
        chain = param_chains[[nm]],
        true_value = if (nm %in% names(true_values)) true_values[[nm]] else NA_real_,
        param_name = nm,
        color = "lightblue"
      )
    }
  }

  # States
  if (!is.null(states) && length(states) > 0) {
    cat("Generating plots: State Estimates...\n")
    generate_state_plots(states)
  }

  cat("=== VISUALIZATION COMPLETED ===\n\n")
}