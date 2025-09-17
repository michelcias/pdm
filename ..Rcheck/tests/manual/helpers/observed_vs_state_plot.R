#-------------------------------------------------------------------------------
# Visualization helper: Observed data vs State (true and estimated)
#-------------------------------------------------------------------------------

#' Plot observed vs state (true and estimated) for up to n_plot points
#' @param y observed vector
#' @param true_state state ground-truth vector
#' @param estimated_state state estimated vector (e.g., median)
#' @param title plot title
#' @param xlab x-axis label
#' @param ylab y-axis label
#' @param n_plot number of initial points to plot (default 200)
plot_observed_vs_state <- function(y,
                                   true_state,
                                   estimated_state,
                                   title = "Observed Data vs. State",
                                   xlab = "Time",
                                   ylab = "Value",
                                   n_plot = 200) {
  stopifnot(length(y) == length(true_state), length(y) == length(estimated_state))
  n <- length(y)
  time_idx <- 1:min(n_plot, n)

  plot(time_idx, y[time_idx], type = 'p', col = "gray50", pch = 16, cex = 0.5,
       main = title, xlab = xlab, ylab = ylab,
       ylim = range(c(y[time_idx], true_state[time_idx], estimated_state[time_idx])))
  lines(time_idx, true_state[time_idx], col = "red", lty = 2, lwd = 2)
  lines(time_idx, estimated_state[time_idx], col = "blue", lwd = 1.5)
  legend("topleft",
         legend = c("Observed", "True State", "Estimated State"),
         col = c("gray50", "red", "blue"),
         pch = c(16, NA, NA),
         lty = c(NA, 2, 1),
         lwd = c(NA, 2, 1.5),
         bty = "n")
}