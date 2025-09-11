#-------------------------------------------------------------------------------
# Helper functions for MCMC convergence diagnostics
#-------------------------------------------------------------------------------

#' Compute effective sample sizes for multiple chains
#' @param param_list Named list of parameter chains
#' @return Named vector of ESS values
compute_effective_sample_sizes <- function(param_list) {
  sapply(param_list, function(x) {
    acf_vals <- acf(x, plot=FALSE, lag.max=min(100, length(x)/4))$acf[-1]
    max(1, length(x) / (1 + 2 * sum(acf_vals[acf_vals > 0])))
  })
}

#' Print ESS analysis table
#' @param param_list Named list of parameter chains
#' @param n_chain Number of MCMC samples
print_ess_table <- function(param_list, n_chain) {
  ess_values <- compute_effective_sample_sizes(param_list)
  
  cat("Effective Sample Size Analysis:\n")
  cat("==========================================================================\n")
  cat("Parameter   |   Chain Size |        ESS |  Efficiency (%) |       Status\n")
  cat("==========================================================================\n")
  
  for(param_name in names(ess_values)) {
    efficiency <- 100 * ess_values[param_name] / n_chain
    status <- ifelse(efficiency > 50, "EXCELLENT",
                     ifelse(efficiency > 25, "GOOD",
                            ifelse(efficiency > 10, "ACCEPTABLE", "POOR")))
    cat(sprintf("%-11s | %12d | %10.1f | %15.1f | %12s\n",
                param_name, n_chain, ess_values[param_name], efficiency, status))
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
  
  # Geweke diagnostic
  print_geweke_table(mcmc_objects)
  
  # Heidelberger-Welch diagnostic
  print_heidelberg_table(mcmc_objects)
  
  # Raftery-Lewis diagnostic with robust handling
  print_raftery_lewis_table(mcmc_objects, length(param_list[[1]]))
  
  # Overall assessment
  print_overall_convergence_table(mcmc_objects, param_list)
}

#' Print Geweke diagnostic table
print_geweke_table <- function(mcmc_objects) {
  cat("Geweke Convergence Diagnostic:\n")
  cat("====================================================================\n")
  cat("Parameter   |    Z-Score |  |Z| < 2.0 |      Status | Interpretation\n")
  cat("====================================================================\n")
  
  geweke_results <- lapply(mcmc_objects, coda::geweke.diag)
  for(param in names(geweke_results)) {
    z_score <- geweke_results[[param]]$z
    abs_z <- abs(z_score)
    status <- ifelse(abs_z < 1.96, "PASS", "FAIL")
    interp <- ifelse(abs_z < 1.96, "Converged",
                     ifelse(abs_z < 2.58, "Marginal", "Not Converged"))
    cat(sprintf("%-11s | %10.4f | %10s | %11s | %14s\n",
                param, z_score, ifelse(abs_z < 2.0, "YES", "NO"), status, interp))
  }
  cat("====================================================================\n\n")
}

# Continuar com print_heidelberg_table, print_raftery_lewis_table, etc...