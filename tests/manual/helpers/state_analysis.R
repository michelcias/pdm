#-------------------------------------------------------------------------------
# State series analysis helpers (generalized for N components)
# Extracted from the Local Level script and generalized:
# - Segment analysis
# - Global quality metrics
# - Outliers / problematic points
# - Temporal autocorrelation by lag
# - ESS at selected time points
#-------------------------------------------------------------------------------

#' Detailed analysis for a single state series
#' @param true_values numeric vector with ground-truth values (length n)
#' @param estimates numeric vector with estimates (e.g., medians) (length n)
#' @param ci_lower numeric vector with lower CI (length n)
#' @param ci_upper numeric vector with upper CI (length n)
#' @param chain MCMC matrix [n_draws x n] with samples per time point
#' @param state_name label for the state (e.g., "theta_1" or "level")
#' @param n_segments number of temporal segments (default 5)
#' @param max_lag maximum lag for temporal ACF (default 10)
#' @param n_draws chain length (defaults to nrow(chain))
#' @param n_timepoints how many time points to sample for ESS (default 10)
analyze_state_series <- function(true_values,
                                 estimates,
                                 ci_lower,
                                 ci_upper,
                                 chain,
                                 state_name = "state",
                                 n_segments = 5,
                                 max_lag = 10,
                                 n_draws = nrow(chain),
                                 n_timepoints = 10) {
  n <- length(true_values)
  if (!is.matrix(chain) || ncol(chain) != n) {
    stop("chain must be a matrix [n_draws x n] consistent with the series length.")
  }

  cat(sprintf("=== %s ANALYSIS ===\n\n", toupper(state_name)))

  # 1) Segment analysis
  cat("1. Analysis by Temporal Segments:\n")
  cat("==============================================================================\n")
  segment_size <- n %/% n_segments
  segment_stats <- data.frame(
    Segment = 1:n_segments,
    Time_Range = paste0("[", (0:(n_segments-1))*segment_size + 1, "-",
                        (1:n_segments)*segment_size, "]"),
    True_Mean = NA_real_,
    Est_Mean = NA_real_,
    Est_SD = NA_real_,
    Coverage_95 = NA_real_,
    RMSE = NA_real_
  )

  for (i in 1:n_segments) {
    start_idx <- (i-1)*segment_size + 1
    end_idx <- i*segment_size
    segment_stats$True_Mean[i] <- mean(true_values[start_idx:end_idx])
    segment_stats$Est_Mean[i]  <- mean(estimates[start_idx:end_idx])
    segment_stats$Est_SD[i]    <- mean(apply(chain[, start_idx:end_idx, drop = FALSE], 2, sd))
    segment_stats$Coverage_95[i] <- mean((true_values[start_idx:end_idx] >= ci_lower[start_idx:end_idx]) &
                                         (true_values[start_idx:end_idx] <= ci_upper[start_idx:end_idx]))
    segment_stats$RMSE[i] <- sqrt(mean((estimates[start_idx:end_idx] - true_values[start_idx:end_idx])^2))
  }

  cat("Segment |   Time Range | True Mean | Est Mean |  Est SD |  Coverage |    RMSE\n")
  cat("==============================================================================\n")
  for (i in 1:n_segments) {
    cat(sprintf("%7d | %12s | %9.4f | %8.4f | %7.4f | %8.2f%% | %7.4f\n",
                segment_stats$Segment[i], segment_stats$Time_Range[i],
                segment_stats$True_Mean[i], segment_stats$Est_Mean[i],
                segment_stats$Est_SD[i], 100*segment_stats$Coverage_95[i],
                segment_stats$RMSE[i]))
  }
  cat("==============================================================================\n\n")

  # 2) Global quality metrics
  cat("2. Global Quality Metrics:\n")
  cat("==============================\n")
  abs_errors <- abs(estimates - true_values)
  rel_denom  <- pmax(1e-12, abs(true_values))
  rel_errors <- abs_errors / rel_denom
  coverage   <- (true_values >= ci_lower) & (true_values <= ci_upper)

  global_metrics <- data.frame(
    Metric = c("RMSE", "MAE", "MAPE (%)", "Coverage_95 (%)", "Correlation", "R^2"),
    Value = c(
      sqrt(mean((estimates - true_values)^2)),
      mean(abs_errors),
      100 * mean(abs((estimates - true_values) / rel_denom)),
      100 * mean(coverage),
      suppressWarnings(cor(estimates, true_values)),
      suppressWarnings(cor(estimates, true_values)^2)
    )
  )
  cat("Metric          |      Value\n")
  cat("==============================\n")
  for (i in 1:nrow(global_metrics)) {
    cat(sprintf("%-15s | %10.6f\n", global_metrics$Metric[i], global_metrics$Value[i]))
  }
  cat("============================\n\n")

  # 3) Outliers and problematic points
  cat("3. Outliers and Problematic Points:\n")
  cat("==================================================================================\n")
  top_k <- min(10, n)
  worst_abs_idx <- order(abs_errors, decreasing = TRUE)[1:top_k]
  worst_rel_idx <- order(rel_errors, decreasing = TRUE)[1:top_k]
  coverage_failures <- which(!coverage)

  outlier_analysis <- data.frame(
    Category = c("Worst 10 Abs Errors", "Worst 10 Rel Errors", "Coverage Failures"),
    Count = c(length(worst_abs_idx), length(worst_rel_idx), length(coverage_failures)),
    Mean_Error = c(
      mean(abs_errors[worst_abs_idx]),
      mean(rel_errors[worst_rel_idx]),
      ifelse(length(coverage_failures) > 0, mean(abs_errors[coverage_failures]), NA)
    ),
    Max_Error = c(
      max(abs_errors[worst_abs_idx]),
      max(rel_errors[worst_rel_idx]),
      ifelse(length(coverage_failures) > 0, max(abs_errors[coverage_failures]), NA)
    ),
    First_5_Points = c(
      paste(head(worst_abs_idx, 5), collapse = ", "),
      paste(head(worst_rel_idx, 5), collapse = ", "),
      ifelse(length(coverage_failures) >= 5,
             paste(head(coverage_failures, 5), collapse = ", "),
             ifelse(length(coverage_failures) > 0, paste(coverage_failures, collapse = ", "), "None"))
    )
  )
  cat("Category             | Count | Mean Error |  Max Error |          First 5 Points\n")
  cat("==================================================================================\n")
  for (i in 1:nrow(outlier_analysis)) {
    cat(sprintf("%-20s | %5d | %10.6f | %10.6f | %s\n",
                outlier_analysis$Category[i], outlier_analysis$Count[i],
                outlier_analysis$Mean_Error[i], outlier_analysis$Max_Error[i],
                outlier_analysis$First_5_Points[i]))
  }
  cat("=================================================================================\n\n")

  # 4) Temporal autocorrelation (true, estimated, error)
  cat("4. Temporal Autocorrelation Analysis:\n")
  cat("======================================\n")
  lag_analysis <- data.frame(
    Lag = 1:max_lag,
    True_ACF = sapply(1:max_lag, function(k) if (k < n) cor(true_values[1:(n-k)], true_values[(1+k):n]) else NA),
    Est_ACF  = sapply(1:max_lag, function(k) if (k < n) cor(estimates[1:(n-k)],  estimates[(1+k):n])  else NA),
    Error_ACF= sapply(1:max_lag, function(k) if (k < n) cor(abs_errors[1:(n-k)], abs_errors[(1+k):n]) else NA)
  )
  cat("Lag | True ACF |  Est ACF | Error ACF\n")
  cat("======================================\n")
  for (i in 1:nrow(lag_analysis)) {
    cat(sprintf("%3d | %8.4f | %8.4f | %9.4f\n",
                lag_analysis$Lag[i], lag_analysis$True_ACF[i],
                lag_analysis$Est_ACF[i], lag_analysis$Error_ACF[i]))
  }
  cat("======================================\n\n")

  # 5) ESS at selected time points
  cat("5. Effective Sample Size by Time Points:\n")
  cat("====================================\n")
  time_points <- round(seq(1, n, length.out = n_timepoints))
  ess_by_time <- sapply(time_points, function(t_idx) {
    chain_t <- chain[, t_idx]
    acf_vals <- acf(chain_t, plot = FALSE, lag.max = min(100, length(chain_t) / 4))$acf[-1]
    max(1, length(chain_t) / (1 + 2 * sum(acf_vals[acf_vals > 0])))
  })

  ess_time_analysis <- data.frame(
    Time_Point = time_points,
    ESS = round(ess_by_time),
    Efficiency = round(100 * ess_by_time / n_draws, 1)
  )

  cat("Time Point |   ESS | Efficiency (%)\n")
  cat("====================================\n")
  for (i in 1:nrow(ess_time_analysis)) {
    cat(sprintf("%10d | %5d | %9.1f\n",
                ess_time_analysis$Time_Point[i], ess_time_analysis$ESS[i],
                ess_time_analysis$Efficiency[i]))
  }
  cat("====================================\n")
  ess_min  <- round(min(ess_by_time[is.finite(ess_by_time)]))
  ess_max  <- round(max(ess_by_time[is.finite(ess_by_time)]))
  ess_mean <- round(mean(ess_by_time[is.finite(ess_by_time)]), 1)
  ess_eff  <- round(100 * ess_mean / n_draws, 1)
  cat(sprintf("Overall ESS range: [%d, %d]\n", ess_min, ess_max))
  cat(sprintf("Mean ESS: %.1f (%.1f%% efficiency)\n\n", ess_mean, ess_eff))
}

#' Analyze multiple state series in sequence
#' @param states named list; each item is a list with:
#'        true, estimate, ci_lower, ci_upper, chain, optionally state_name
#' @param n_segments, max_lag, n_draws, n_timepoints see analyze_state_series
analyze_multiple_states <- function(states,
                                    n_segments = 5,
                                    max_lag = 10,
                                    n_draws = NULL,
                                    n_timepoints = 10) {
  stopifnot(is.list(states), length(states) > 0)
  for (nm in names(states)) {
    s <- states[[nm]]
    nm_eff <- if (!is.null(s$state_name)) s$state_name else nm
    analyze_state_series(
      true_values = s$true,
      estimates   = s$estimate,
      ci_lower    = s$ci_lower,
      ci_upper    = s$ci_upper,
      chain       = s$chain,
      state_name  = nm_eff,
      n_segments  = n_segments,
      max_lag     = max_lag,
      n_draws     = if (is.null(n_draws)) nrow(s$chain) else n_draws,
      n_timepoints= n_timepoints
    )
  }
}