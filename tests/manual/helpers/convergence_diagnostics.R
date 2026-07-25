#-------------------------------------------------------------------------------
# Helper functions for MCMC convergence diagnostics
#-------------------------------------------------------------------------------

#' Compute effective sample sizes for multiple chains
#' @param param_list Named list of parameter chains (numeric vectors)
#' @return Named vector of ESS values
compute_effective_sample_sizes <- function(param_list) {
  sapply(param_list, function(x) {
    acf_vals <- acf(x, plot = FALSE, lag.max = min(100, length(x) / 4))$acf[-1]
    max(1, length(x) / (1 + 2 * sum(acf_vals[acf_vals > 0])))
  })
}

#' Print ESS analysis table
#' @param param_list Named list of parameter chains
#' @param n_draws Number of MCMC samples
print_ess_table <- function(param_list, n_draws) {
  ess_values <- compute_effective_sample_sizes(param_list)

  cat("Effective Sample Size Analysis:\n")
  cat("==========================================================================\n")
  cat("Parameter   |   Chain Size |        ESS |  Efficiency (%) |       Status\n")
  cat("==========================================================================\n")

  for (param_name in names(ess_values)) {
    efficiency <- 100 * ess_values[param_name] / n_draws
    status <- ifelse(efficiency > 50, "EXCELLENT",
                     ifelse(efficiency > 25, "GOOD",
                            ifelse(efficiency > 10, "ACCEPTABLE", "POOR")))
    cat(sprintf("%-11s | %12d | %10.1f | %15.1f | %12s\n",
                param_name, n_draws, ess_values[param_name], efficiency, status))
  }
  cat("==========================================================================\n")
  cat(sprintf("Overall Assessment: Min ESS = %.1f, Mean ESS = %.1f\n\n",
              min(ess_values), mean(ess_values)))
}

#' Run comprehensive convergence diagnostics using coda package
#' @param param_list Named list of parameter chains
run_coda_diagnostics <- function(param_list) {
  if (!requireNamespace("coda", quietly = TRUE)) {
    cat("Note: Install 'coda' package for enhanced convergence diagnostics.\n\n")
    return(invisible(NULL))
  }

  # Convert to mcmc objects
  mcmc_objects <- lapply(param_list, coda::mcmc)

  # 1) Geweke diagnostic
  print_geweke_table(mcmc_objects)

  # 2) Heidelberger–Welch diagnostic
  print_heidelberg_table(mcmc_objects)

  # 3) Raftery–Lewis diagnostic (robust/fallback)
  n_draws <- length(param_list[[1]])
  print_raftery_lewis_table(mcmc_objects, n_draws)

  # 4) Overall convergence assessment
  print_overall_convergence_table(mcmc_objects, param_list)
}

#' Print Geweke diagnostic table
#' @param mcmc_objects Named list of coda::mcmc objects
print_geweke_table <- function(mcmc_objects) {
  cat("Geweke Convergence Diagnostic:\n")
  cat("=====================================================================\n")
  cat("Parameter   |    Z-Score |  |Z| < 2.0 |      Status | Interpretation\n")
  cat("=====================================================================\n")

  geweke_results <- lapply(mcmc_objects, coda::geweke.diag)
  for (param in names(geweke_results)) {
    z_score <- geweke_results[[param]]$z
    abs_z <- abs(z_score)
    status <- ifelse(abs_z < 1.96, "PASS", "FAIL")
    interp <- ifelse(abs_z < 1.96, "Converged",
                     ifelse(abs_z < 2.58, "Marginal", "Not Converged"))
    cat(sprintf("%-11s | %10.4f | %10s | %11s | %14s\n",
                param, z_score, ifelse(abs_z < 2.0, "YES", "NO"), status, interp))
  }
  cat("=====================================================================\n\n")
}

#' Print Heidelberger–Welch stationarity/halfwidth table
#' @param mcmc_objects Named list of coda::mcmc objects
print_heidelberg_table <- function(mcmc_objects) {
  cat("Heidelberger-Welch Stationarity and Halfwidth Tests:\n")
  cat("=================================================================================\n")
  cat("Parameter   | Stationarity | Start Iter |  Halfwidth |      Mean |     Status\n")
  cat("=================================================================================\n")

  for (param in names(mcmc_objects)) {
    res <- try(coda::heidel.diag(mcmc_objects[[param]]), silent = TRUE)
    if (inherits(res, "try-error")) {
      cat(sprintf("%-11s | %12s | %10s | %10s | %8s | %10s\n",
                  param, "ERROR", "-", "-", "-", "CHECK"))
      next
    }
    stationarity_status <- ifelse(res[1, 1], "PASS", "FAIL")
    halfwidth_status    <- ifelse(res[1, 3], "PASS", "FAIL")
    overall_status      <- ifelse(res[1, 1] && res[1, 3], "CONVERGED", "CHECK")
    cat(sprintf("%-11s | %12s | %10d | %10s | %9.4f | %10s\n",
                param,
                stationarity_status,
                res[1, 2],
                halfwidth_status,
                res[1, 4],
                overall_status))
  }
  cat("=================================================================================\n\n")
}

#' Print Raftery–Lewis diagnostic table with robust fallbacks
#' @param mcmc_objects Named list of coda::mcmc objects
#' @param n_draws Length of each chain (number of saved samples)
print_raftery_lewis_table <- function(mcmc_objects, n_draws) {
  cat("Raftery-Lewis Diagnostic (Burn-in and Sample Size Requirements):\n")
  cat("=============================================================================================================\n")
  cat("Parameter   | Quantile | Accuracy | Probability | Burn-in (M) | Total (N) | Lower (Nmin) |     Dependence\n")
  cat("=============================================================================================================\n")

  # Combine into a single multivariate mcmc with named columns
  combined <- coda::mcmc(do.call(cbind, lapply(mcmc_objects, as.matrix)))
  colnames(combined) <- names(mcmc_objects)

  # Try a grid of (q, r) until one works
  accuracy_levels <- c(0.005, 0.01, 0.02, 0.05)
  quantile_levels <- c(0.025, 0.05, 0.1)
  raftery_success <- FALSE
  raftery_result  <- NULL
  q_used <- NA_real_
  r_used <- NA_real_

  for (q in quantile_levels) {
    for (r in accuracy_levels) {
      attempt <- try(coda::raftery.diag(combined, q = q, r = r, s = 0.95), silent = TRUE)
      if (!inherits(attempt, "try-error") &&
          is.matrix(attempt$resmatrix) && nrow(attempt$resmatrix) > 0) {
        raftery_success <- TRUE
        raftery_result  <- attempt
        q_used <- q; r_used <- r
        cat(sprintf("Note: Using quantile=%.3f, accuracy=%.3f for analysis\n", q_used, r_used))
        break
      }
    }
    if (raftery_success) break
  }

  if (raftery_success) {
    rm <- raftery_result$resmatrix
    for (i in seq_len(nrow(rm))) {
      param_name <- rownames(rm)[i]
      M <- rm[i, "M"]; N <- rm[i, "N"]; Nmin <- rm[i, "Nmin"]; I <- rm[i, "I"]
      dep_status <- ifelse(I < 5, "Low", ifelse(I < 10, "Moderate", "High"))
      chain_ok   <- n_draws >= N
      star <- ifelse(chain_ok, "", " *")
      cat(sprintf("%-11s |    %4.1f%% |     %.1f%% |         95%% | %11d | %9d | %12d | %8.2f (%s)%s\n",
                  param_name, 100*q_used, 100*r_used, M, N, Nmin, I, dep_status, star))
    }

    max_required <- max(rm[, "N"])
    if (n_draws < max_required) {
      cat("=============================================================================================================\n")
      cat(sprintf("WARNING: Current chain size (%d) is smaller than recommended (%d)\n", n_draws, max_required))
      cat("* Marked parameters may need longer chains for reliable estimates\n")
      cat(sprintf("Recommendation: Increase chain size to at least %d samples\n", max_required))
    }
  } else {
    cat("Unable to compute Raftery-Lewis diagnostic.\n")
    cat("This typically indicates insufficient chain size for the desired precision or strong autocorrelation.\n")
    cat("===============================================================================================================\n")
  }

  cat("\n")
}

#' Print overall convergence assessment, summarizing available tests
#' @param mcmc_objects Named list of coda::mcmc objects
#' @param param_list Named list of numeric chains (same ordering as mcmc_objects)
print_overall_convergence_table <- function(mcmc_objects, param_list) {
  # Prepare ESS
  n_draws <- length(param_list[[1]])
  ess <- compute_effective_sample_sizes(param_list)
  params <- names(mcmc_objects)

  # Geweke and Heidel results
  geweke <- lapply(mcmc_objects, coda::geweke.diag)
  heidel <- lapply(mcmc_objects, function(z) try(coda::heidel.diag(z), silent = TRUE))

  # Raftery on combined (best-effort)
  raftery_success <- FALSE
  raftery_result  <- NULL
  if (requireNamespace("coda", quietly = TRUE)) {
    combined <- coda::mcmc(do.call(cbind, lapply(mcmc_objects, as.matrix)))
    colnames(combined) <- params
    accuracy_levels <- c(0.005, 0.01, 0.02, 0.05)
    quantile_levels <- c(0.025, 0.05, 0.1)
    for (q in quantile_levels) {
      for (r in accuracy_levels) {
        attempt <- try(coda::raftery.diag(combined, q = q, r = r, s = 0.95), silent = TRUE)
        if (!inherits(attempt, "try-error") &&
            is.matrix(attempt$resmatrix) && nrow(attempt$resmatrix) > 0) {
          raftery_success <- TRUE
          raftery_result  <- attempt
          break
        }
      }
      if (raftery_success) break
    }
  }

  cat("Overall Convergence Assessment:\n")
  cat("=========================================================================================\n")
  cat("Parameter   |     ESS |   Efficiency | Geweke | Heidelberg | Raftery-Lewis |     Overall\n")
  cat("=========================================================================================\n")

  for (i in seq_along(params)) {
    p <- params[i]
    ess_val <- ess[p]
    efficiency <- 100 * ess_val / n_draws

    # Geweke pass
    gz <- try(geweke[[p]]$z, silent = TRUE)
    geweke_pass <- !inherits(gz, "try-error") && is.finite(gz) && abs(gz) < 1.96

    # Heidel pass
    hz <- heidel[[p]]
    heidel_pass <- !inherits(hz, "try-error") && is.matrix(hz) && nrow(hz) >= 1 && hz[1, 1] && hz[1, 3]

    # Raftery pass (if available)
    if (raftery_success) {
      rm <- raftery_result$resmatrix
      if (p %in% rownames(rm)) {
        I <- rm[p, "I"]; N <- rm[p, "N"]
        raftery_pass <- (I < 5) && (n_draws >= N)
      } else {
        raftery_pass <- FALSE
      }
      raftery_status <- ifelse(raftery_pass, "PASS", "FAIL")
    } else {
      raftery_pass <- FALSE
      raftery_status <- "N/A"
    }

    available_tests <- c(geweke_pass, heidel_pass)
    if (raftery_success) available_tests <- c(available_tests, raftery_pass)
    tests_passed <- sum(available_tests)
    total_tests  <- length(available_tests)
    overall_status <- ifelse(tests_passed == total_tests, "EXCELLENT",
                             ifelse(tests_passed >= total_tests * 0.75, "GOOD",
                                    ifelse(tests_passed >= total_tests * 0.5, "ACCEPTABLE", "POOR")))

    cat(sprintf("%-11s | %7.1f | %11.1f%% | %6s | %10s | %13s | %11s\n",
                p, ess_val, efficiency,
                ifelse(geweke_pass, "PASS", "FAIL"),
                ifelse(heidel_pass, "PASS", "FAIL"),
                raftery_status,
                overall_status))
  }
  cat("=========================================================================================\n\n")
}
