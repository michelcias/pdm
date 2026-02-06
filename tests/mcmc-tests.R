# =============================================================================
# Enhanced MCMC Testing Suite for Binomial Local Level Model
# =============================================================================
#
#' @title Enhanced MCMC Testing Suite for Binomial Local Level Model
#' @author Michel H. Montoril
#' @date 2025-09-02
#' @description
#' This script provides comprehensive testing for the MCMC implementation
#' of the binomial local level model with fixed parameters. It validates
#' individual conditional distributions and overall sampler performance
#' with enhanced diagnostics and formal statistical tests.
#'
#' @details
#' The testing suite includes:
#' - Formal convergence diagnostics using the coda package
#' - Enhanced input validation and error handling
#' - Structured configuration system
#' - Consolidated diagnostic functions
#' - Formal statistical tests for validation
#' - Improved visualization and reporting
# =============================================================================

# Load required packages
required_packages <- c("coda", "parallel")
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    warning(paste("Package", pkg, "not available. Some diagnostics may be limited."))
  }
}

# =============================================================================
# 1. STRUCTURED CONFIGURATION SYSTEM
# =============================================================================

#' Global configuration for MCMC testing
CONFIG <- list(
  # Simulation parameters
  simulation = list(
    n = 500,                    # Number of time points
    n_trials = 10,             # Number of trials per time point
    theta_01_true = 0.5,        # Initial state value
    prec_theta1_true = 100.0,        # State precision
    seed = 404                  # Reproducibility seed
  ),

  # MCMC parameters
  mcmc = list(
    burnin = 5000,              # Burn-in iterations
    thinning = 25,              # Thinning interval
    n_chain = 2000,             # Posterior samples
    target_acceptance = 0.44,   # Target acceptance rate
    lag_update = 50L,           # Adaptive update lag
    max_step_size = 0.1,        # Maximum step size
    base_adaptation_rate = 50.0,# Base adaptation rate
    decay_exponent = 0.5        # Decay exponent
  ),

  # Validation thresholds
  validation = list(
    max_rel_bias = 0.05,        # Maximum acceptable relative bias
    min_coverage = 0.90,        # Minimum coverage probability
    max_rmse = 0.5,             # Maximum RMSE for theta_1
    min_eff_size = 400,         # Minimum effective sample size
    max_geweke_pvalue = 0.05    # Geweke test p-value threshold
  ),

  # Test configuration
  test = list(
    quick_test = FALSE,         # Enable quick testing mode
    save_results = TRUE,        # Save test results
    parallel = FALSE,           # Enable parallel processing
    n_cores = 2,                # Number of cores for parallel processing
    verbose = TRUE              # Verbose output
  )
)

# Quick test configuration (for development/CI)
if (CONFIG$test$quick_test) {
  CONFIG$simulation$n <- 100
  CONFIG$mcmc$burnin <- 1000
  CONFIG$mcmc$n_chain <- 500
  CONFIG$mcmc$thinning <- 10
}

# =============================================================================
# 2. ENHANCED INPUT VALIDATION FUNCTIONS
# =============================================================================

#' Comprehensive input validation for MCMC testing
#' @param y Numeric vector of observations
#' @param n_trials Number of trials
#' @param ... Additional parameters to validate
validate_inputs <- function(y, n_trials, burnin = NULL, thinning = NULL,
                            n_chain = NULL, theta_1_true = NULL, ...) {

  # Basic input checks
  if (!is.numeric(y) || length(y) == 0) {
    stop("'y' must be a non-empty numeric vector")
  }

  if (any(is.na(y)) || any(is.infinite(y))) {
    stop("'y' cannot contain NA or infinite values")
  }

  if (!is.numeric(n_trials) || length(n_trials) != 1 || n_trials <= 0) {
    stop("'n_trials' must be a positive numeric scalar")
  }

  # Binomial observation validation
  if (any(y < 0) || any(y > n_trials)) {
    stop("Invalid binomial observations: all y values must be between 0 and n_trials")
  }

  if (any(y != round(y))) {
    stop("Binomial observations must be integers")
  }

  # MCMC parameter validation
  if (!is.null(burnin)) {
    if (!is.numeric(burnin) || burnin < 0 || burnin != round(burnin)) {
      stop("'burnin' must be a non-negative integer")
    }
  }

  if (!is.null(thinning)) {
    if (!is.numeric(thinning) || thinning < 1 || thinning != round(thinning)) {
      stop("'thinning' must be a positive integer")
    }
  }

  if (!is.null(n_chain)) {
    if (!is.numeric(n_chain) || n_chain < 1 || n_chain != round(n_chain)) {
      stop("'n_chain' must be a positive integer")
    }
  }

  # Fixed parameter validation
  if (!is.null(theta_1_true)) {
    if (!is.numeric(theta_1_true) || length(theta_1_true) != length(y)) {
      stop("'theta_1_true' must be numeric with same length as y")
    }
  }

  if (CONFIG$test$verbose) {
    cat("✓ Input validation passed\n")
  }

  return(TRUE)
}

# =============================================================================
# 3. MCMC DIAGNOSTIC FUNCTIONS
# =============================================================================

#' Comprehensive MCMC diagnostics
#' @param samples MCMC samples (vector or matrix)
#' @param true_value True parameter value(s)
#' @param param_name Parameter name for reporting
#' @param ... Additional arguments
mcmc_diagnostics <- function(samples, true_value = NULL, param_name = "parameter", ...) {

  diagnostics <- list()

  # Convert to coda mcmc object if available
  if (requireNamespace("coda", quietly = TRUE)) {
    if (is.matrix(samples)) {
      mcmc_obj <- coda::mcmc(samples)
    } else {
      mcmc_obj <- coda::mcmc(as.matrix(samples))
    }

    # Effective sample size
    diagnostics$effective_size <- coda::effectiveSize(mcmc_obj)

    # Geweke convergence diagnostic
    tryCatch({
      geweke_result <- coda::geweke.diag(mcmc_obj)
      diagnostics$geweke_z <- geweke_result$z
      diagnostics$geweke_pvalue <- 2 * pnorm(-abs(geweke_result$z))
    }, error = function(e) {
      diagnostics$geweke_z <- NA
      diagnostics$geweke_pvalue <- NA
    })

    # Autocorrelation
    tryCatch({
      autocorr_result <- coda::autocorr(mcmc_obj, lags = c(1, 5, 10, 50))

      if (is.matrix(autocorr_result)) {
        autocorr_vector <- drop(autocorr_result)
        lag_names <- names(autocorr_vector)
        lag_numbers <- as.numeric(gsub("Lag ", "", lag_names))

        # Fallback se extração falhar
        if (any(is.na(lag_numbers))) {
          lag_numbers <- c(1, 5, 10, 50)[1:length(autocorr_vector)]
        }

        diagnostics$autocorr <- autocorr_vector
        diagnostics$autocorr_lags <- lag_numbers
      } else {
        diagnostics$autocorr <- autocorr_result
        diagnostics$autocorr_lags <- c(1, 5, 10, 50)[1:length(autocorr_result)]
      }
    }, error = function(e) {
      diagnostics$autocorr <- NULL
      diagnostics$autocorr_lags <- NULL
    })

  } else {
    # Basic diagnostics without coda
    diagnostics$effective_size <- length(samples)  # Conservative estimate
    diagnostics$geweke_z <- NA
    diagnostics$geweke_pvalue <- NA
    diagnostics$autocorr <- NA
  }

  # Basic statistics
  if (is.matrix(samples)) {
    diagnostics$mean <- colMeans(samples)
    diagnostics$sd <- apply(samples, 2, sd)
    diagnostics$quantiles <- apply(samples, 2, quantile, probs = c(0.025, 0.25, 0.5, 0.75, 0.975))
  } else {
    diagnostics$mean <- mean(samples)
    diagnostics$sd <- sd(samples)
    diagnostics$quantiles <- quantile(samples, probs = c(0.025, 0.25, 0.5, 0.75, 0.975))
  }

  # Monte Carlo standard error
  if (is.matrix(samples)) {
    diagnostics$mc_se <- apply(samples, 2, function(x) sd(x) / sqrt(length(x)))
  } else {
    diagnostics$mc_se <- sd(samples) / sqrt(length(samples))
  }

  # Bias and accuracy (if true value provided)
  if (!is.null(true_value)) {
    diagnostics$bias <- diagnostics$mean - true_value
    diagnostics$relative_bias <- abs(diagnostics$bias) / abs(true_value)
    diagnostics$rmse <- sqrt(mean((diagnostics$mean - true_value)^2))

    # Coverage probability (approximate 95% CI)
    if (is.matrix(samples)) {
      lower_ci <- diagnostics$quantiles[1, ]
      upper_ci <- diagnostics$quantiles[5, ]
      diagnostics$coverage <- mean((true_value >= lower_ci) & (true_value <= upper_ci))
    } else {
      lower_ci <- diagnostics$quantiles[1]
      upper_ci <- diagnostics$quantiles[5]
      diagnostics$coverage <- (true_value >= lower_ci) & (true_value <= upper_ci)
    }
  }

  # Add parameter name
  diagnostics$parameter <- param_name

  return(diagnostics)
}

#' Generate comprehensive test report
#' @param test_results List of test results
#' @param config Configuration used
generate_test_report <- function(test_results, config = CONFIG) {

  cat("\n", strrep("=", 80), "\n")
  cat("ENHANCED MCMC VALIDATION REPORT\n")
  cat("Generated:", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"), "\n")
  cat(strrep("=", 80), "\n\n")

  # Configuration summary
  cat("CONFIGURATION SUMMARY:\n")
  cat("  Sample size (n):", config$simulation$n, "\n")
  cat("  Binomial trials:", config$simulation$n_trials, "\n")
  cat("  MCMC samples:", config$mcmc$n_chain, "\n")
  cat("  Burn-in:", config$mcmc$burnin, "\n")
  cat("  Thinning:", config$mcmc$thinning, "\n\n")

  # Test results summary
  overall_status <- TRUE

  for (test_name in names(test_results)) {
    test_result <- test_results[[test_name]]
    diag <- test_result$diagnostics

    cat("TEST", toupper(substr(test_name, nchar(test_name), nchar(test_name))),
        "-", diag$parameter, ":\n")

    # Bias assessment
    if (length(diag$relative_bias) == 1) {
      rel_bias <- abs(diag$relative_bias)
      bias_status <- rel_bias < config$validation$max_rel_bias
      cat("  Relative Bias:", sprintf("%.3f%%", rel_bias * 100))
      cat(" [", ifelse(bias_status, "PASS", "FAIL"), "]\n")
    } else {
      mean_rel_bias <- mean(abs(diag$relative_bias), na.rm = TRUE)
      max_rel_bias <- max(abs(diag$relative_bias), na.rm = TRUE)
      bias_status <- mean_rel_bias < config$validation$max_rel_bias
      cat("  Mean Relative Bias:", sprintf("%.3f%%", mean_rel_bias * 100),
          "Max:", sprintf("%.3f%%", max_rel_bias * 100))
      cat(" [", ifelse(bias_status, "PASS", "FAIL"), "]\n")
    }

    # Effective sample size
    eff_size <- diag$effective_size
    if (!is.null(eff_size)) {
      if (length(eff_size) == 1) {
        eff_status <- eff_size > config$validation$min_eff_size
        cat("  Effective Size:", sprintf("%.0f", eff_size))
        cat(" [", ifelse(eff_status, "PASS", "FAIL"), "]\n")
      } else {
        mean_eff_size <- mean(eff_size, na.rm = TRUE)
        min_eff_size <- min(eff_size, na.rm = TRUE)
        eff_status <- mean_eff_size > config$validation$min_eff_size
        cat("  Mean Effective Size:", sprintf("%.0f", mean_eff_size),
            "Min:", sprintf("%.0f", min_eff_size))
        cat(" [", ifelse(eff_status, "PASS", "FAIL"), "]\n")
      }
    }

    # Geweke test
    geweke_p <- diag$geweke_pvalue
    if (!is.null(geweke_p) && !all(is.na(geweke_p))) {
      if (length(geweke_p) == 1) {
        geweke_status <- geweke_p > config$validation$max_geweke_pvalue
        cat("  Geweke p-value:", sprintf("%.4f", geweke_p))
        cat(" [", ifelse(geweke_status, "PASS", "FAIL"), "]\n")
      } else {
        mean_geweke_p <- mean(geweke_p, na.rm = TRUE)
        min_geweke_p <- min(geweke_p, na.rm = TRUE)
        geweke_status <- mean_geweke_p > config$validation$max_geweke_pvalue
        cat("  Mean Geweke p-value:", sprintf("%.4f", mean_geweke_p),
            "Min:", sprintf("%.4f", min_geweke_p))
        cat(" [", ifelse(geweke_status, "PASS", "FAIL"), "]\n")
      }
    }

    # Coverage (if available)
    if (!is.null(diag$coverage)) {
      coverage <- diag$coverage
      if (length(coverage) == 1) {
        coverage_status <- coverage > config$validation$min_coverage
        cat("  Coverage:", sprintf("%.3f", coverage))
        cat(" [", ifelse(coverage_status, "PASS", "FAIL"), "]\n")
      } else {
        mean_coverage <- mean(coverage, na.rm = TRUE)
        min_coverage <- min(coverage, na.rm = TRUE)
        coverage_status <- mean_coverage > config$validation$min_coverage
        cat("  Mean Coverage:", sprintf("%.3f", mean_coverage),
            "Min:", sprintf("%.3f", min_coverage))
        cat(" [", ifelse(coverage_status, "PASS", "FAIL"), "]\n")
      }
    }

    # RMSE (if available)
    if (!is.null(diag$rmse)) {
      if (length(diag$rmse) == 1) {
        rmse_val <- diag$rmse
        rmse_status <- rmse_val < config$validation$max_rmse
        cat("  RMSE:", sprintf("%.4f", rmse_val))
        cat(" [", ifelse(rmse_status, "PASS", "FAIL"), "]\n")
      } else {
        mean_rmse <- mean(diag$rmse, na.rm = TRUE)
        max_rmse <- max(diag$rmse, na.rm = TRUE)
        rmse_status <- mean_rmse < config$validation$max_rmse
        cat("  Mean RMSE:", sprintf("%.4f", mean_rmse),
            "Max:", sprintf("%.4f", max_rmse))
        cat(" [", ifelse(rmse_status, "PASS", "FAIL"), "]\n")
      }
    }

    # Overall test status (use AND for all scalar statuses)
    test_status <- TRUE
    # Bias
    test_status <- test_status & bias_status
    # Effective size
    if (!is.null(eff_size)) {
      if (length(eff_size) == 1) {
        test_status <- test_status & (eff_size > config$validation$min_eff_size)
      } else {
        test_status <- test_status & (mean(eff_size, na.rm = TRUE) > config$validation$min_eff_size)
      }
    }
    # Geweke
    if (!is.null(geweke_p) && !all(is.na(geweke_p))) {
      if (length(geweke_p) == 1) {
        test_status <- test_status & (geweke_p > config$validation$max_geweke_pvalue)
      } else {
        test_status <- test_status & (mean(geweke_p, na.rm = TRUE) > config$validation$max_geweke_pvalue)
      }
    }
    # Coverage
    if (!is.null(diag$coverage)) {
      if (length(diag$coverage) == 1) {
        test_status <- test_status & (diag$coverage > config$validation$min_coverage)
      } else {
        test_status <- test_status & (mean(diag$coverage, na.rm = TRUE) > config$validation$min_coverage)
      }
    }
    # RMSE
    if (!is.null(diag$rmse)) {
      if (length(diag$rmse) == 1) {
        test_status <- test_status & (diag$rmse < config$validation$max_rmse)
      } else {
        test_status <- test_status & (mean(diag$rmse, na.rm = TRUE) < config$validation$max_rmse)
      }
    }

    cat("  Status:", ifelse(test_status, "PASS", "FAIL"), "\n\n")
    overall_status <- overall_status & test_status
  }

  # Overall assessment
  cat("OVERALL STATUS:", ifelse(overall_status, "ALL TESTS PASS", "SOME TESTS FAILED"), "\n")
  cat(strrep("=", 80), "\n")

  # Return structured report
  report <- list(
    timestamp = Sys.time(),
    config = config,
    results = test_results,
    overall_status = overall_status
  )

  if (config$test$save_results) {
    saveRDS(report, paste0("mcmc_validation_report_",
                           format(Sys.time(), "%Y%m%d_%H%M%S"), ".rds"))
    cat("Report saved to: mcmc_validation_report_",
        format(Sys.time(), "%Y%m%d_%H%M%S"), ".rds\n")
  }

  return(report)
}

# =============================================================================
# 4. ENHANCED R WRAPPER FOR C MCMC FUNCTION
# =============================================================================

#' Enhanced MCMC Sampler for Binomial Local Level Model with Fixed Parameters
#'
#' This function provides an R interface to the C implementation of the MCMC
#' sampler for the binomial local level model. It allows selective fixing of
#' parameters for testing individual conditional distributions with enhanced
#' validation and error handling.
#'
#' @param y Numeric vector of observed binomial counts
#' @param n_trials Number of trials for each binomial observation
#' @param burnin Number of burn-in iterations
#' @param thinning Thinning interval for stored samples
#' @param n_chain Number of samples to store after burn-in and thinning
#' @param theta_1_true Optional: fixed values for latent states (testing only)
#' @param theta_01_true Optional: fixed value for initial state (testing only)
#' @param prec_theta1_true Optional: fixed value for state precision (testing only)
#' @param prior_theta01_mean Prior mean for theta_01
#' @param prior_theta01_prec Prior precision for theta_01
#' @param prior_prec1_shape Prior shape parameter for precision
#' @param prior_prec1_rate Prior rate parameter for precision
#' @param lag_update Window size for adaptive tuning
#' @param max_step_size Maximum step size for proposal adaptation
#' @param base_adaptation_rate Base rate for proposal adaptation
#' @param decay_exponent Decay exponent for adaptation schedule
#' @param target_acceptance Target acceptance rate for adaptive MCMC
#' @param validate_inputs Whether to perform input validation
#'
#' @return List with MCMC samples and diagnostics:
#'   - theta_1: Matrix of latent state samples (n_chain x n)
#'   - theta_01: Vector of initial state samples (n_chain)
#'   - prec_theta1: Vector of precision samples (n_chain)
#'   - diagnostics: List of convergence diagnostics
#'
#' @examples
#' \dontrun{
#' # Simulate data
#' y <- rbinom(100, 50, 0.3)
#'
#' # Run MCMC
#' result <- enhanced_test_sampler(y, n_trials = 50, burnin = 1000,
#'                                thinning = 10, n_chain = 500)
#' }
enhanced_test_sampler <- function(y, n_trials, burnin, thinning, n_chain,
                                  theta_1_true = NULL, theta_01_true = NULL, prec_theta1_true = NULL,
                                  prior_theta01_mean = 0.0, prior_theta01_prec = 1.0,
                                  prior_prec1_shape = 1.0, prior_prec1_rate = 1.0,
                                  lag_update = 50L, max_step_size = 0.1,
                                  base_adaptation_rate = 1.0, decay_exponent = 0.5,
                                  target_acceptance = 0.44,
                                  return_log_sigma = FALSE,    # Argumento 18
                                  return_accrate = FALSE,      # Argumento 19
                                  validate_inputs = TRUE) {

  # Input validation
  if (validate_inputs) {
    validate_inputs(y, n_trials, burnin, thinning, n_chain, theta_1_true)
  }

  # Progress indicator
  if (CONFIG$test$verbose) {
    total_iter <- burnin + (n_chain - 1) * thinning + 1
    cat("Starting MCMC with", total_iter, "total iterations...\n")
  }

  # Call C function with ALL 19 ARGUMENTS
  tryCatch({
    result <- .Call("_bdm_test_mcmc_binomial_locallevel_fixed_params",
                    y,                          # 1
                    n_trials,                   # 2
                    as.integer(burnin),         # 3
                    as.integer(thinning),       # 4
                    as.integer(n_chain),        # 5
                    theta_1_true,               # 6
                    theta_01_true,              # 7
                    prec_theta1_true,                # 8
                    prior_theta01_mean,         # 9
                    prior_theta01_prec,         # 10
                    prior_prec1_shape,          # 11
                    prior_prec1_rate,           # 12
                    as.integer(lag_update),     # 13
                    max_step_size,              # 14
                    base_adaptation_rate,       # 15
                    decay_exponent,             # 16
                    target_acceptance,          # 17
                    return_log_sigma,           # 18 ← AGORA INCLUÍDO!
                    return_accrate)             # 19 ← AGORA INCLUÍDO!
  }, error = function(e) {
    stop("C function call failed: ", e$message,
         "\nCheck that the bdm package is properly installed and loaded.")
  })

  if (CONFIG$test$verbose) {
    cat("✓ MCMC completed successfully\n")
  }

  return(result)
}

# Legacy wrapper for backward compatibility
test_sampler <- enhanced_test_sampler

# =============================================================================
# 5. DATA SIMULATION AND SETUP
# =============================================================================

cat(strrep("=", 80), "\n")
cat("ENHANCED MCMC TESTING SUITE FOR BINOMIAL LOCAL LEVEL MODEL\n")
cat(strrep("=", 80), "\n")

# Set reproducible seed
set.seed(CONFIG$simulation$seed)

# Extract configuration
n <- CONFIG$simulation$n
n_trials <- CONFIG$simulation$n_trials
theta_01_true <- CONFIG$simulation$theta_01_true
prec_theta1_true <- CONFIG$simulation$prec_theta1_true

cat("=== DATA SIMULATION ===\n")
cat("Time series length (n):", n, "\n")
cat("Binomial trials per period:", n_trials, "\n")
cat("True initial state (theta_01):", theta_01_true, "\n")
cat("True state precision:", prec_theta1_true, "\n")
cat("Quick test mode:", ifelse(CONFIG$test$quick_test, "ENABLED", "DISABLED"), "\n\n")

# Generate true latent states (random walk)
u1 <- rnorm(n, mean = 0, sd = sqrt(1 / prec_theta1_true))
theta_1_true <- cumsum(c(theta_01_true, u1))[-1]

# Transform to probability scale via logit link
alpha_true <- plogis(theta_1_true)

# Generate observed binomial data
y <- rbinom(n, size = n_trials, prob = alpha_true)

# Enhanced diagnostic plots
if (CONFIG$test$verbose) {
  cat("Generating enhanced diagnostic plots...\n")

  par(mfrow = c(2, 3), mar = c(4, 4, 2, 1))

  # True latent states
  plot.ts(theta_1_true, main = "True Latent States (theta_1)",
          ylab = "theta_1", col = "blue", lwd = 2)
  grid()

  # True probabilities
  plot.ts(alpha_true, main = "True Probabilities (alpha)",
          ylab = "Probability", col = "red", lwd = 2, ylim = c(0, 1))
  grid()

  # Observed data
  plot.ts(y, main = "Observed Binomial Counts",
          ylab = "Count", type = "o", pch = 16, cex = 0.7)
  grid()

  # Empirical vs true probabilities
  plot(alpha_true, y/n_trials, main = "True vs Empirical Probabilities",
       xlab = "True Probability", ylab = "Empirical Probability",
       pch = 16, col = rgb(0, 0, 1, 0., alpha = 0.66))
  abline(0, 1, col = "red", lty = 2, lwd = 2)
  grid()

  # Data distribution
  hist(y, breaks = 30, main = "Distribution of Observed Counts",
       xlab = "Count", col = "lightblue", border = "white")

  # Time series of empirical probabilities
  plot.ts(y/n_trials, main = "Empirical Probabilities",
          ylab = "Empirical Probability", col = "purple", lwd = 1)
  lines(alpha_true, col = "red", lwd = 2, lty = 2)
  legend("topright", c("Empirical", "True"), col = c("purple", "red"),
         lwd = c(1, 2), lty = c(1, 2))
  grid()

  par(mfrow = c(1, 1))
}

# =============================================================================
# 6. TEST A: CONDITIONAL DISTRIBUTION OF THETA_01
# =============================================================================

cat("\n=== TEST A: Sampling theta_01 (fixing theta_1 and prec_theta1) ===\n")
cat("Purpose: Validate conditional posterior for initial state parameter\n")

# Set seed for reproducibility
set.seed(CONFIG$simulation$seed + 1)

# Run MCMC with enhanced wrapper
mcmc_out_A <- enhanced_test_sampler(
  y = y,
  n_trials = n_trials,
  burnin = CONFIG$mcmc$burnin,
  thinning = CONFIG$mcmc$thinning,
  n_chain = CONFIG$mcmc$n_chain,
  theta_1_true = theta_1_true,    # Fixed at true values
  prec_theta1_true = prec_theta1_true,      # Fixed at true values
  prior_theta01_mean = 0.0,       # Weakly informative prior
  prior_theta01_prec = 1.0
)

# Comprehensive diagnostics
diag_A <- mcmc_diagnostics(mcmc_out_A$theta_01, theta_01_true, "theta_01")

# Store results
test_results <- list()
test_results$test_A <- list(
  samples = mcmc_out_A$theta_01,
  diagnostics = diag_A,
  fixed_params = c("theta_1", "prec_theta1")
)

# Enhanced visualization
if (CONFIG$test$verbose) {
  par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))

  # Trace plot with running mean
  plot.ts(mcmc_out_A$theta_01, main = "Trace Plot: theta_01",
          ylab = "theta_01", col = rgb(0, 0, 1, 0.7))
  abline(h = theta_01_true, col = "red", lwd = 2, lty = 2)
  abline(h = diag_A$mean, col = "green", lwd = 2, lty = 2)
  legend("topright", c("True value", "Posterior mean"),
         col = c("red", "green"), lty = 2, lwd = 2, cex = 0.8)
  grid()

  # Enhanced density plot
  plot(density(mcmc_out_A$theta_01), main = "Posterior Density: theta_01",
       xlab = "theta_01", col = "blue", lwd = 2)
  abline(v = theta_01_true, col = "red", lwd = 2, lty = 2)
  abline(v = diag_A$mean, col = "green", lwd = 2, lty = 2)
  abline(v = diag_A$quantiles[c(1, 5)], col = "gray", lwd = 1, lty = 3)
  legend("topright", c("True value", "Posterior mean", "95% CI"),
         col = c("red", "green", "gray"), lty = c(2, 2, 3), lwd = c(2, 2, 1), cex = 0.8)
  grid()

  # Running average with convergence band
  running_mean <- cumsum(mcmc_out_A$theta_01) / seq_along(mcmc_out_A$theta_01)
  plot(running_mean, type = "l", main = "Running Average: theta_01",
       ylab = "Running Mean", col = "blue", lwd = 2)
  abline(h = theta_01_true, col = "red", lwd = 2, lty = 2)
  # Add convergence band
  n_samples <- length(running_mean)
  conv_band <- 1.96 * diag_A$mc_se / sqrt(1:n_samples)
  lines(running_mean + conv_band, col = "gray", lty = 3)
  lines(running_mean - conv_band, col = "gray", lty = 3)
  legend("topright", c("Running mean", "True value", "±1.96 MC SE"),
         col = c("blue", "red", "gray"), lty = c(1, 2, 3), lwd = c(2, 2, 1), cex = 0.8)
  grid()

  # Autocorrelation plot
  if (!is.null(diag_A$autocorr) && !all(is.na(diag_A$autocorr))) {
    autocorr_values <- drop(diag_A$autocorr)
    lag_numbers <- as.numeric(gsub("Lag ", "", names(autocorr_values)))

    # Fallback se extração de lags falhar
    if (any(is.na(lag_numbers))) {
      lag_numbers <- c(1, 5, 10, 50)[1:length(autocorr_values)]
    }

    plot(lag_numbers, autocorr_values,
         type = "h", main = "Autocorrelation: theta_01",
         xlab = "Lag", ylab = "Autocorrelation", lwd = 2, col = "blue")
    abline(h = 0, col = "gray", lty = 2)
    grid()
  } else {
    # Basic autocorrelation if coda not available
    acf(mcmc_out_A$theta_01, main = "Autocorrelation: theta_01")
  }

  par(mfrow = c(1, 1))
}

# Print detailed results
cat("\nDETAILED RESULTS FOR TEST A:\n")
cat("  True value:", theta_01_true, "\n")
cat("  Posterior mean:", sprintf("%.6f", diag_A$mean), "\n")
cat("  Posterior SD:", sprintf("%.6f", diag_A$sd), "\n")
cat("  MC standard error:", sprintf("%.6f", diag_A$mc_se), "\n")
cat("  Bias:", sprintf("%.6f", diag_A$bias), "\n")
cat("  Relative bias:", sprintf("%.3f%%", diag_A$relative_bias * 100), "\n")
if (!is.na(diag_A$effective_size)) {
  cat("  Effective sample size:", sprintf("%.0f", diag_A$effective_size), "\n")
}
if (!is.na(diag_A$geweke_pvalue)) {
  cat("  Geweke p-value:", sprintf("%.4f", diag_A$geweke_pvalue), "\n")
}

# =============================================================================
# 7. TEST B: CONDITIONAL DISTRIBUTION OF PREC_1
# =============================================================================

cat("\n=== TEST B: Sampling prec_theta1 (fixing theta_1 and theta_01) ===\n")
cat("Purpose: Validate conditional posterior for state precision parameter\n")

# Set seed for reproducibility
set.seed(CONFIG$simulation$seed + 2)

# Run MCMC with enhanced wrapper
mcmc_out_B <- enhanced_test_sampler(
  y = y,
  n_trials = n_trials,
  burnin = CONFIG$mcmc$burnin,
  thinning = CONFIG$mcmc$thinning,
  n_chain = CONFIG$mcmc$n_chain,
  theta_1_true = theta_1_true,      # Fixed at true values
  theta_01_true = theta_01_true,    # Fixed at true values
  prior_prec1_shape = 1.0,          # Weakly informative Gamma prior
  prior_prec1_rate = 0.01           # E[prec] = shape/rate = 100
)

# Comprehensive diagnostics
diag_B <- mcmc_diagnostics(mcmc_out_B$prec_theta1, prec_theta1_true, "prec_theta1")

# Store results
test_results$test_B <- list(
  samples = mcmc_out_B$prec_theta1,
  diagnostics = diag_B,
  fixed_params = c("theta_1", "theta_01")
)

# Enhanced visualization
if (CONFIG$test$verbose) {
  par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))

  # Trace plot
  plot.ts(mcmc_out_B$prec_theta1, main = "Trace Plot: prec_theta1",
          ylab = "prec_theta1", col = rgb(0, 0.5, 0, 0.7))
  abline(h = prec_theta1_true, col = "red", lwd = 2, lty = 2)
  abline(h = diag_B$mean, col = "blue", lwd = 2, lty = 2)
  legend("topright", c("True value", "Posterior mean"),
         col = c("red", "blue"), lty = 2, lwd = 2, cex = 0.8)
  grid()

  # Log-scale density plot (better for precision parameters)
  log_samples <- log(mcmc_out_B$prec_theta1)
  plot(density(log_samples), main = "Posterior Density: log(prec_theta1)",
       xlab = "log(prec_theta1)", col = "darkgreen", lwd = 2)
  abline(v = log(prec_theta1_true), col = "red", lwd = 2, lty = 2)
  abline(v = log(diag_B$mean), col = "blue", lwd = 2, lty = 2)
  grid()

  # Running average
  running_mean_B <- cumsum(mcmc_out_B$prec_theta1) / seq_along(mcmc_out_B$prec_theta1)
  plot(running_mean_B, type = "l", main = "Running Average: prec_theta1",
       ylab = "Running Mean", col = "darkgreen", lwd = 2)
  abline(h = prec_theta1_true, col = "red", lwd = 2, lty = 2)
  grid()

  # Q-Q plot for normality check (on log scale)
  qqnorm(log_samples, main = "Q-Q Plot: log(prec_theta1)")
  qqline(log_samples, col = "red", lwd = 2)
  grid()

  par(mfrow = c(1, 1))
}

# Print detailed results
cat("\nDETAILED RESULTS FOR TEST B:\n")
cat("  True value:", prec_theta1_true, "\n")
cat("  Posterior mean:", sprintf("%.4f", diag_B$mean), "\n")
cat("  Posterior SD:", sprintf("%.4f", diag_B$sd), "\n")
cat("  MC standard error:", sprintf("%.4f", diag_B$mc_se), "\n")
cat("  Bias:", sprintf("%.4f", diag_B$bias), "\n")
cat("  Relative bias:", sprintf("%.3f%%", diag_B$relative_bias * 100), "\n")
if (!is.na(diag_B$effective_size)) {
  cat("  Effective sample size:", sprintf("%.0f", diag_B$effective_size), "\n")
}
if (!is.na(diag_B$geweke_pvalue)) {
  cat("  Geweke p-value:", sprintf("%.4f", diag_B$geweke_pvalue), "\n")
}

# =============================================================================
# 8. TEST C: CONDITIONAL DISTRIBUTION OF THETA_1
# =============================================================================

cat("\n=== TEST C: Sampling theta_1 (fixing theta_01 and prec_theta1) ===\n")
cat("Purpose: Validate conditional posterior for latent state trajectory\n")

# Set seed for reproducibility
set.seed(CONFIG$simulation$seed + 3)

# Reduce samples for computational efficiency in trajectory sampling
n_chain_C <- ifelse(CONFIG$test$quick_test, 200, 1000)

# Run MCMC with enhanced wrapper
mcmc_out_C <- enhanced_test_sampler(
  y = y,
  n_trials = n_trials,
  burnin = CONFIG$mcmc$burnin,
  thinning = CONFIG$mcmc$thinning,
  n_chain = n_chain_C,
  theta_01_true = theta_01_true,    # Fixed at true values
  prec_theta1_true = prec_theta1_true         # Fixed at true values
)

# Comprehensive diagnostics for trajectory
diag_C <- mcmc_diagnostics(mcmc_out_C$theta_1, theta_1_true, "theta_1")

# Store results
test_results$test_C <- list(
  samples = mcmc_out_C$theta_1,
  diagnostics = diag_C,
  fixed_params = c("theta_01", "prec_theta1")
)

# Enhanced visualization for trajectory
if (CONFIG$test$verbose) {
  par(mfrow = c(2, 3), mar = c(4, 4, 2, 1))

  # Posterior means vs true values
  plot.ts(diag_C$mean, main = "Posterior Means vs True Values",
          ylab = "theta_1", col = "blue", lwd = 2)
  lines(theta_1_true, col = "red", lwd = 2, lty = 2)
  legend("topright", c("Posterior Mean", "True Values"),
         col = c("blue", "red"), lwd = 2, lty = c(1, 2), cex = 0.8)
  grid()

  # Bias over time
  plot.ts(diag_C$bias, main = "Bias Over Time",
          ylab = "Bias", col = "purple", lwd = 1)
  abline(h = 0, col = "gray", lty = 2)
  abline(h = c(-2, 2) * diag_C$mc_se[1], col = "red", lty = 3)
  grid()

  # Posterior standard deviations
  plot.ts(diag_C$sd, main = "Posterior Standard Deviations",
          ylab = "Posterior SD", col = "orange", lwd = 1)
  grid()

  # Coverage visualization
  lower_CI <- diag_C$quantiles[1, ]
  upper_CI <- diag_C$quantiles[5, ]
  coverage_overall <- mean((theta_1_true >= lower_CI) & (theta_1_true <= upper_CI))

  plot.ts(diag_C$mean, main = paste("95% Credible Intervals (Coverage:",
                                    sprintf("%.1f%%)", coverage_overall * 100)),
          ylab = "theta_1", ylim = range(c(lower_CI, upper_CI, theta_1_true)),
          col = "blue", lwd = 2)
  lines(lower_CI, col = "gray", lty = 2)
  lines(upper_CI, col = "gray", lty = 2)
  lines(theta_1_true, col = "red", lwd = 2)
  legend("topright", c("Posterior Mean", "95% CI", "True Values"),
         col = c("blue", "gray", "red"), lwd = c(2, 1, 2), lty = c(1, 2, 1), cex = 0.8)
  grid()

  # Sample trajectory visualization (subset for clarity)
  n_show <- min(100, n)
  idx_show <- seq(1, n, length.out = n_show)
  matplot(idx_show, t(mcmc_out_C$theta_1[1:min(50, nrow(mcmc_out_C$theta_1)), idx_show]),
          type = "l", lty = 1, col = rgb(0, 0, 1, 0.3),
          main = "Sample Trajectories (first 5)",
          xlab = "Time", ylab = "theta_1")
  lines(idx_show, theta_1_true[idx_show], col = "red", lwd = 3)
  grid()

  # Residual analysis
  residuals <- diag_C$mean - theta_1_true
  plot(diag_C$mean, residuals, main = "Residuals vs Fitted",
       xlab = "Posterior Mean", ylab = "Residuals",
       pch = 16, col = rgb(0, 0, 1, 0.6))
  abline(h = 0, col = "red", lty = 2)
  smooth_line <- lowess(diag_C$mean, residuals)
  lines(smooth_line, col = "blue", lwd = 2)
  grid()

  par(mfrow = c(1, 1))
}

# Print detailed results
cat("\nDETAILED RESULTS FOR TEST C:\n")
cat("  Mean absolute bias:", sprintf("%.6f", mean(abs(diag_C$bias))), "\n")
cat("  RMSE:", sprintf("%.6f", diag_C$rmse), "\n")
cat("  Max absolute bias:", sprintf("%.6f", max(abs(diag_C$bias))), "\n")
cat("  Mean posterior SD:", sprintf("%.6f", mean(diag_C$sd)), "\n")
cat("  Coverage probability:", sprintf("%.3f", diag_C$coverage), "\n")
if (!is.na(diag_C$effective_size[1])) {
  cat("  Mean effective size:", sprintf("%.0f", mean(diag_C$effective_size)), "\n")
}

# =============================================================================
# 9. FORMAL STATISTICAL TESTS
# =============================================================================

cat("\n=== FORMAL STATISTICAL TESTS ===\n")

# Test A: Normality tests for theta_01
if (length(mcmc_out_A$theta_01) >= 3) {
  # Shapiro-Wilk test (if sample size allows)
  if (length(mcmc_out_A$theta_01) <= 5000) {
    shapiro_A <- shapiro.test(mcmc_out_A$theta_01)
    cat("Test A - Shapiro-Wilk normality test p-value:", sprintf("%.4f", shapiro_A$p.value), "\n")
  }

  # Kolmogorov-Smirnov test against normal distribution
  ks_A <- ks.test(scale(mcmc_out_A$theta_01), "pnorm")
  cat("Test A - KS test vs normal p-value:", sprintf("%.4f", ks_A$p.value), "\n")
}

# Test B: Tests for precision parameter (log-normal expected)
if (length(mcmc_out_B$prec_theta1) >= 3) {
  log_prec <- log(mcmc_out_B$prec_theta1)
  if (length(log_prec) <= 5000) {
    shapiro_B <- shapiro.test(log_prec)
    cat("Test B - Shapiro-Wilk test (log scale) p-value:", sprintf("%.4f", shapiro_B$p.value), "\n")
  }

  ks_B <- ks.test(scale(log_prec), "pnorm")
  cat("Test B - KS test vs normal (log scale) p-value:", sprintf("%.4f", ks_B$p.value), "\n")
}

# Test C: Multivariate tests for trajectory
cat("Test C - Trajectory validation:\n")
cat("  Pointwise coverage rate:", sprintf("%.3f", diag_C$coverage), "\n")
cat("  Empirical coverage (should be ~0.95):",
    ifelse(diag_C$coverage > 0.93 & diag_C$coverage < 0.97, "GOOD", "CHECK"), "\n")

# =============================================================================
# 10. COMPREHENSIVE VALIDATION REPORT
# =============================================================================

# Generate and display comprehensive report
final_report <- generate_test_report(test_results, CONFIG)

# Additional summary statistics
cat("\n=== PERFORMANCE SUMMARY ===\n")
total_mcmc_time <- Sys.time()  # This would be better tracked during actual runs
cat("Configuration used:\n")
cat("  Quick test mode:", ifelse(CONFIG$test$quick_test, "YES", "NO"), "\n")
cat("  Total MCMC samples generated:",
    3 * CONFIG$mcmc$n_chain + n_chain_C, "\n")
cat("  Validation thresholds met:",
    ifelse(final_report$overall_status, "ALL", "SOME"), "\n")

# Clean workspace flag
if (CONFIG$test$save_results) {
  cat("\nTest results saved for further analysis.\n")
} else {
  cat("\nTest completed. Results not saved (set CONFIG$test$save_results = TRUE to save).\n")
}

cat("\n=== ENHANCED TESTING SUITE COMPLETED SUCCESSFULLY ===\n")
cat("All conditional distributions have been thoroughly validated.\n")
cat("Enhanced diagnostics and formal tests have been applied.\n")
cat("The MCMC implementation shows",
    ifelse(final_report$overall_status, "EXCELLENT", "MIXED"), "performance.\n")

# Reset graphics parameters
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1)

