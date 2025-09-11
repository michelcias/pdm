#-------------------------------------------------------------------------------
# Adaptation diagnostics for CWMH: acceptance rates and log_sigma
#-------------------------------------------------------------------------------
# Inputs:
# - theta_1_updated: matrix [n_iter x n], 0/1 indicators per iteration and component
# - log_sigma_hist:  matrix [n_iter x n], log_sigma only at adaptation iterations (NA elsewhere)
# - lag_update, target_acceptance, max_step_size, base_adaptation_rate, decay_exponent
# - n_iter, burnin, thinning
# Outputs:
# - Plots and console summaries
#-------------------------------------------------------------------------------

run_adaptation_diagnostics <- function(theta_1_updated,
                                       log_sigma_hist,
                                       lag_update,
                                       target_acceptance,
                                       max_step_size,
                                       base_adaptation_rate,
                                       decay_exponent,
                                       n_iter,
                                       burnin,
                                       thinning,
                                       band_tolerance = 0.05) {
  if (!is.matrix(theta_1_updated) || !is.matrix(log_sigma_hist)) {
    stop("theta_1_updated and log_sigma_hist must be matrices")
  }
  n <- ncol(theta_1_updated)

  # Overall acceptance per iteration
  acc_overall <- rowMeans(theta_1_updated, na.rm = TRUE)

  # Identify adaptation iterations (rows with any finite log_sigma recorded)
  adapt_idx <- which(rowSums(is.finite(log_sigma_hist)) > 0)
  if (length(adapt_idx) == 0) {
    cat("No adaptation iterations found in log_sigma_hist. Skipping adaptation diagnostics.\n")
    return(invisible(NULL))
  }

  # Rolling window acceptance (last lag_update iterations) at adaptation times
  roll_accrate_mean   <- numeric(length(adapt_idx))
  roll_accrate_median <- numeric(length(adapt_idx))
  roll_accrate_q25    <- numeric(length(adapt_idx))
  roll_accrate_q75    <- numeric(length(adapt_idx))
  roll_frac_in_band   <- numeric(length(adapt_idx))

  # log_sigma summaries at adaptation times
  ls_mean <- numeric(length(adapt_idx))
  ls_q25  <- numeric(length(adapt_idx))
  ls_q75  <- numeric(length(adapt_idx))

  # Mean absolute change in log_sigma between adaptation points
  ls_mac  <- rep(NA_real_, length(adapt_idx))  # NA for the first point

  # Theoretical diminishing step size at adaptation points
  step_size_theoretical <- pmin(max_step_size, base_adaptation_rate / (adapt_idx ^ decay_exponent))

  for (k in seq_along(adapt_idx)) {
    t <- adapt_idx[k]
    w_start <- max(2, t - lag_update + 1) # start from 2 to skip uninitialized row 1
    window  <- w_start:t
    by_comp <- colMeans(theta_1_updated[window, , drop = FALSE])

    roll_accrate_mean[k]   <- mean(by_comp)
    roll_accrate_median[k] <- median(by_comp)
    roll_accrate_q25[k]    <- quantile(by_comp, 0.25)
    roll_accrate_q75[k]    <- quantile(by_comp, 0.75)

    roll_frac_in_band[k]   <- mean(abs(by_comp - target_acceptance) <= band_tolerance)

    ls_row <- log_sigma_hist[t, ]
    ls_row <- ls_row[is.finite(ls_row)]
    if (length(ls_row)) {
      ls_mean[k] <- mean(ls_row)
      ls_q25[k]  <- quantile(ls_row, 0.25)
      ls_q75[k]  <- quantile(ls_row, 0.75)
    } else {
      ls_mean[k] <- ls_q25[k] <- ls_q75[k] <- NA_real_
    }

    if (k > 1) {
      prev_t <- adapt_idx[k - 1]
      prev_row <- log_sigma_hist[prev_t, ]
      cur_row  <- log_sigma_hist[t, ]
      sel <- is.finite(prev_row) & is.finite(cur_row)
      if (any(sel)) {
        ls_mac[k] <- mean(abs(cur_row[sel] - prev_row[sel]))
      }
    }
  }

  # --- Plots ---
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par), add = TRUE)
  par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

  # 1) Rolling acceptance mean vs target with band
  y_min <- min(roll_accrate_q25, target_acceptance - band_tolerance, na.rm = TRUE)
  y_max <- max(roll_accrate_q75, target_acceptance + band_tolerance, na.rm = TRUE)
  plot(adapt_idx, roll_accrate_mean, type = "l", col = "black", lwd = 2,
       xlab = "Iteration", ylab = "Rolling acceptance (mean across components)",
       main = "Acceptance vs target (window = lag_update)",
       ylim = c(y_min, y_max))
  abline(h = target_acceptance, col = "red", lwd = 2, lty = 2)
  rect(xleft = min(adapt_idx), ybottom = target_acceptance - band_tolerance,
       xright = max(adapt_idx), ytop = target_acceptance + band_tolerance,
       col = rgb(1, 0, 0, 0.08), border = NA)
  lines(adapt_idx, roll_accrate_q25, col = "steelblue", lwd = 1, lty = 3)
  lines(adapt_idx, roll_accrate_q75, col = "steelblue", lwd = 1, lty = 3)
  legend("bottomright",
         legend = c("Mean acc.", "IQR (25–75%)", "Target", sprintf("±%.02f band", band_tolerance)),
         col = c("black", "steelblue", "red", rgb(1, 0, 0, 0.2)), lty = c(1, 3, 2, NA),
         lwd = c(2, 1, 2, NA), pch = c(NA, NA, NA, 15), pt.cex = 2, bty = "n")

  # 2) Fraction of components within band
  plot(adapt_idx, 100 * roll_frac_in_band, type = "o", pch = 16, col = "darkgreen",
       xlab = "Iteration", ylab = "Components within band (%)",
       main = "Proportion within target band")
  abline(h = 80, col = "gray60", lty = 3) # 80% reference
  abline(h = 95, col = "gray60", lty = 3) # 95% reference

  # 3) log_sigma mean and IQR across components
  yls <- range(c(ls_q25, ls_q75), na.rm = TRUE)
  plot(adapt_idx, ls_mean, type = "l", col = "purple", lwd = 2,
       xlab = "Iteration", ylab = "log_sigma (mean across components)",
       main = "log_sigma summary at adaptation", ylim = yls)
  lines(adapt_idx, ls_q25, col = "purple", lty = 3)
  lines(adapt_idx, ls_q75, col = "purple", lty = 3)
  legend("topright", legend = c("Mean", "IQR"), col = c("purple", "purple"),
         lty = c(1, 3), lwd = c(2, 1), bty = "n")

  # 4) Mean absolute change in log_sigma vs theoretical step size
  mac <- ls_mac
  plot(adapt_idx, mac, type = "o", pch = 16, col = "orange",
       xlab = "Iteration", ylab = "Mean |Δ log_sigma|",
       main = "Change in log_sigma vs theoretical step size")
  lines(adapt_idx, step_size_theoretical, col = "red", lwd = 2, lty = 2)
  legend("topright", legend = c("Mean |Δ log_sigma|", "Theoretical step size"),
         col = c("orange", "red"), lty = c(1, 2), lwd = c(1, 2), pch = c(16, NA), bty = "n")

  # --- Textual summary & checks ---
  cat("=== CWMH Adaptation Diagnostics ===\n")
  cat(sprintf("- lag_update = %d | target_acceptance = %.3f | band = ±%.3f\n",
              lag_update, target_acceptance, band_tolerance))
  last_k <- length(adapt_idx)
  if (last_k >= 1) {
    cat("\nLatest adaptation window summary:\n")
    cat(sprintf("  Iteration: %d\n", adapt_idx[last_k]))
    cat(sprintf("  Rolling acceptance mean/median: %.3f / %.3f\n",
                roll_accrate_mean[last_k], roll_accrate_median[last_k]))
    cat(sprintf("  Rolling acceptance IQR: [%.3f, %.3f]\n",
                roll_accrate_q25[last_k], roll_accrate_q75[last_k]))
    cat(sprintf("  Fraction within band: %.1f%%\n", 100 * roll_frac_in_band[last_k]))
    cat(sprintf("  log_sigma mean and IQR: mean=%.3f, IQR=[%.3f, %.3f]\n",
                ls_mean[last_k], ls_q25[last_k], ls_q75[last_k]))
    if (!is.na(ls_mac[last_k])) {
      cat(sprintf("  Mean |Δ log_sigma| (last step): %.4f | theoretical step: %.4f\n",
                  ls_mac[last_k], step_size_theoretical[last_k]))
    }
  }

  # Warnings and guidance
  band_ok <- roll_frac_in_band[last_k] >= 0.8
  if (!band_ok) {
    cat("\n[Warning] Less than 80% of components are within the target band.\n")
    cat("Consider:\n")
    cat("- Increasing lag_update to stabilize acceptance estimates.\n")
    cat("- Decreasing base_adaptation_rate for gentler changes.\n")
    cat("- Tweaking decay_exponent toward 0.6–0.8 for stronger diminishing adaptation.\n")
  }

  # Check extreme log_sigma (heuristic bounds)
  ls_all <- as.numeric(log_sigma_hist[adapt_idx, ])
  ls_all <- ls_all[is.finite(ls_all)]
  if (length(ls_all)) {
    if (any(ls_all < log(1e-4))) {
      cat("\n[Warning] Very small proposal scales detected (log_sigma < log(1e-4)).\n")
    }
    if (any(ls_all > log(5))) {
      cat("\n[Warning] Very large proposal scales detected (log_sigma > log(5)).\n")
    }
  }

  cat("\nNotes:\n")
  cat("- Theoretical step size uses: min(max_step_size, base_rate / iter^decay).\n")
  cat("- log_sigma_hist is tracked for diagnostics; the CWMH wrapper used in the loop\n")
  cat("  uses its internal log_sigma (not injected).\n")
  invisible(list(
    adapt_idx = adapt_idx,
    roll = list(mean = roll_accrate_mean, median = roll_accrate_median,
                q25 = roll_accrate_q25, q75 = roll_accrate_q75,
                frac_in_band = roll_frac_in_band),
    log_sigma = list(mean = ls_mean, q25 = ls_q25, q75 = ls_q75, mac = ls_mac),
    step_size_theoretical = step_size_theoretical,
    acc_overall = acc_overall
  ))
}