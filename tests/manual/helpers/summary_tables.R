#-------------------------------------------------------------------------------
# Helper functions for generating summary tables in MCMC validation scripts
#-------------------------------------------------------------------------------

#' Compute comprehensive summary statistics for a parameter chain
#' @param chain Numeric vector of MCMC samples
#' @param true_value True parameter value for comparison
#' @return List with summary statistics
compute_summary_stats <- function(chain, true_value) {
  list(
    median = median(chain),
    mean = mean(chain),
    sd = sd(chain),
    ci_lower = quantile(chain, 0.025),
    ci_upper = quantile(chain, 0.975),
    coverage = (true_value >= quantile(chain, 0.025) & true_value <= quantile(chain, 0.975)),
    rel_error = 100 * abs(median(chain) - true_value) / max(1e-12, abs(true_value)),
    abs_error = abs(median(chain) - true_value)
  )
}

#' Generate formatted posterior estimates table
#' @param param_list Named list of parameter chains
#' @param true_values Named vector of true parameter values
#' @param n_chain Number of MCMC samples
print_posterior_estimates_table <- function(param_list, true_values, n_chain) {
  cat("Enhanced Summary of Posterior Estimates:\n")
  cat("=====================================================================================================================\n")
  cat("Parameter | True Value | Median Est |   Post SD | Monte Carlo SE |                   95% CI | Coverage | Rel Err | Abs Err\n")
  cat("=====================================================================================================================\n")

  for (param_name in names(param_list)) {
    if (param_name %in% names(true_values)) {
      stats <- compute_summary_stats(param_list[[param_name]], true_values[param_name])
      cat(sprintf("%-9s | %10.3f | %10.3f | %9.3f | %14.3f | [%10.3f, %10.3f] | %6s | %6.2f%% | %6.4f\n",
                  param_name, true_values[param_name], stats$median, stats$sd,
                  stats$sd / sqrt(n_chain), stats$ci_lower, stats$ci_upper,
                  ifelse(stats$coverage, "YES", "NO"), stats$rel_error, stats$abs_error))
    }
  }
  cat("=====================================================================================================================\n")
  cat("Note: Monte Carlo SE measures the precision of posterior mean estimates\n\n")
}

#' Generate quantiles table
#' @param param_list Named list of parameter chains
print_quantiles_table <- function(param_list) {
  cat("Detailed Posterior Quantiles:\n")
  cat("===================================================================\n")
  cat("Parameter   |     2.5% |      25% |      50% |      75% |    97.5%\n")
  cat("===================================================================\n")

  for (param_name in names(param_list)) {
    quantiles <- quantile(param_list[[param_name]], c(0.025, 0.25, 0.5, 0.75, 0.975))
    cat(sprintf("%-11s | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f\n",
                param_name, quantiles[1], quantiles[2], quantiles[3], quantiles[4], quantiles[5]))
  }
  cat("===================================================================\n\n")
}
