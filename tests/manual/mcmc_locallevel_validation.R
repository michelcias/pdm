#-------------------------------------------------------------------------------
# Test Script for Local Level Model Auxiliary Functions
#
# Objective: Validate the auxiliary C functions (exposed in test_helpers.c)
#           by replicating the Gibbs sampler loop in R following mcmc_locallevel.c
#-------------------------------------------------------------------------------

# Clean environment
rm(list = ls())

# Ensure we're in the correct directory
if (!file.exists("DESCRIPTION")) {
  stop("This script must be run from the package root directory.")
}

# Load the package
suppressPackageStartupMessages({
  if (require(devtools, quietly = TRUE)) {
    devtools::load_all(".", quiet = TRUE)
  } else {
    # Fallback to installed package
    library(pdm)
  }
})

# --- 1. Data Simulation ---
n <- 1000  # Number of observations

# True parameters
theta0_true <- 0.1
prec1_true  <- 10
prec_y_true <- 1

# set.seed(123) # For reproducibility

# Generate noise
u1 <- rnorm(n, sd = sqrt(1 / prec1_true))
e  <- rnorm(n, sd = sqrt(1 / prec_y_true))

# Simulate latent states and observations
theta1_true <- cumsum(c(theta0_true, u1))[-1]
y <- theta1_true + e

# --- 2. MCMC Configuration ---
burnin   <- 10000
thinning <- 20
n_chain  <- 1000
# Use the same formula as the C implementation
n_iter   <- burnin + (n_chain - 1) * thinning + 1

# Prior hyperparameters (following C implementation nomenclature)
mean_theta01 <- 0                   # prior_theta01_mean
prec_theta01 <- .01                 # prior_theta01_prec
nu_01        <- 1e-2                # prior_prec1_shape
eta_01       <- 1e-2                # prior_prec1_rate
nu_y         <- 1e-2                # prior_prec_y_shape
eta_y        <- 1e-2                # prior_prec_y_rate

# --- 3. Chain Initialization ---
theta_1_chain  <- matrix(NA, nrow = n_chain, ncol = n)
theta_01_chain <- numeric(n_chain)
prec_1_chain   <- numeric(n_chain)
prec_y_chain   <- numeric(n_chain)

# --- 4. Initialization (iter = 0) following mcmc_locallevel.c ---
# set.seed(456)

# Arrays to store complete history (as in C implementation)
theta_01_post <- numeric(n_iter)
prec_1_post   <- numeric(n_iter)
prec_y_post   <- numeric(n_iter)
theta_1_post  <- matrix(NA, nrow = n_iter, ncol = n)

# Initial values exactly as in C implementation
theta_01_post[1] <- rnorm(1, mean_theta01, sqrt(1.0 / prec_theta01))
prec_1_post[1]   <- rgamma(1, nu_01, rate = eta_01)  # Note: R uses 'rate', C uses '1/eta'
prec_y_post[1]   <- rgamma(1, nu_y, rate = eta_y)
init_sd          <- sqrt(1.0 / prec_1_post[1])

# Initialize theta_1 as random walk
theta_1_post[1, 1] <- rnorm(1, theta_01_post[1], init_sd)
for (j in 2:n) {
  theta_1_post[1, j] <- rnorm(1, theta_1_post[1, j-1], init_sd)
}

# --- 5. MCMC Loop using Auxiliary Functions ---
cat("Starting MCMC loop...\n")
chain_idx <- 0  # Counter for saved samples

for (ii in 2:n_iter) {  # Start at 2 since iter=1 is already initialized

  # 1) Sample theta_1 (state vector)
  # Function: test_generate_theta_1_locallevel(data_, prec_data_, prec_theta_1_, theta_01_)
  # theta_1_post[ii, ] <- theta1_true
  theta_1_new <- .Call(
    "_pdm_test_generate_theta_1_locallevel",
    as.numeric(y),
    as.numeric(prec_y_post[ii-1]),      # use previous iteration
    as.numeric(prec_1_post[ii-1]),      # use previous iteration
    as.numeric(theta_01_post[ii-1])     # use previous iteration
  )

  if (length(theta_1_new) == n) {
    theta_1_post[ii, ] <- as.numeric(theta_1_new)
  } else {
    warning(paste("Iteration", ii, ": theta_1 was not updated correctly"))
    theta_1_post[ii, ] <- theta_1_post[ii-1, ]  # keep previous value
  }

  # 2) Sample prec_1 (innovation precision 1/W_1)
  # Function: test_generate_precision_theta_p(theta_0p_, theta_p_, nu_0p_, eta_0p_)
  # prec_1_post[ii] <- prec1_true
  prec_1_post[ii] <- .Call(
    "_pdm_test_generate_precision_theta_p",
    as.numeric(theta_01_post[ii-1]),    # theta_0p (use previous iteration)
    as.numeric(theta_1_post[ii, ]),     # theta_p (use current)
    as.numeric(nu_01),
    as.numeric(eta_01)
  )

  # 3) Sample theta_01 (initial state)
  # Function: test_generate_theta_01_locallevel(theta_1_, prec_theta_1_, mean_theta_01_, prec_theta_01_)
  # theta_01_post[ii] <- theta0_true
  theta_01_post[ii] <- .Call(
    "_pdm_test_generate_theta_01_locallevel",
    as.numeric(theta_1_post[ii, ]),     # use current
    as.numeric(prec_1_post[ii]),        # use current
    as.numeric(mean_theta01),
    as.numeric(prec_theta01)
  )

  # 4) Sample prec_y (data precision 1/V)
  # Function: test_generate_precision_data(y_, theta_1_, nu_y_, eta_y_)
  # prec_y_post[ii] <- prec1_true
  prec_y_post[ii] <- .Call(
    "_pdm_test_generate_precision_data",
    as.numeric(y),
    as.numeric(theta_1_post[ii, ]),     # use current
    as.numeric(nu_y),
    as.numeric(eta_y)
  )

  # Store samples after burn-in and applying thinning
  # Following exactly the logic of the C implementation
  if (ii >= (burnin + 1) && ((ii - burnin - 1) %% thinning == 0)) {
    chain_idx <- chain_idx + 1
    theta_1_chain[chain_idx, ]  <- theta_1_post[ii, ]
    theta_01_chain[chain_idx] <- theta_01_post[ii]
    prec_1_chain[chain_idx]   <- prec_1_post[ii]
    prec_y_chain[chain_idx]   <- prec_y_post[ii]
  }

  # Print progress
  if (ii %% 1000 == 0) {
    cat("  Iteration:", ii, "/", n_iter, "\n")
  }
}

cat("... MCMC loop completed.\n\n")

# --- 6. Enhanced Results Analysis ---
# Compute summary statistics for each parameter
compute_summary_stats <- function(chain, true_value) {
  list(
    median = median(chain),
    mean = mean(chain),
    sd = sd(chain),
    ci_lower = quantile(chain, 0.025),
    ci_upper = quantile(chain, 0.975),
    coverage = (true_value >= quantile(chain, 0.025) & true_value <= quantile(chain, 0.975)),
    rel_error = 100 * abs(median(chain) - true_value) / abs(true_value),
    abs_error = abs(median(chain) - true_value)
  )
}

# Compute statistics for each parameter
theta_01_stats <- compute_summary_stats(theta_01_chain, theta0_true)
prec_1_stats <- compute_summary_stats(prec_1_chain, prec1_true)
prec_y_stats <- compute_summary_stats(prec_y_chain, prec_y_true)

cat("Enhanced Summary of Posterior Estimates:\n")
cat("================================================================================\n")
cat("Parameter   | True Value | Median Est | Post SD  | 95% CI        | Coverage | Rel Err | Abs Err\n")
cat("================================================================================\n")
cat(sprintf("theta_01    | %10.4f | %10.4f | %8.4f | [%6.4f,%6.4f] | %8s | %6.2f%% | %7.4f\n",
            theta0_true, theta_01_stats$median, theta_01_stats$sd,
            theta_01_stats$ci_lower, theta_01_stats$ci_upper,
            ifelse(theta_01_stats$coverage, "YES", "NO"),
            theta_01_stats$rel_error, theta_01_stats$abs_error))
cat(sprintf("prec_1      | %10.4f | %10.4f | %8.4f | [%6.4f,%6.4f] | %8s | %6.2f%% | %7.4f\n",
            prec1_true, prec_1_stats$median, prec_1_stats$sd,
            prec_1_stats$ci_lower, prec_1_stats$ci_upper,
            ifelse(prec_1_stats$coverage, "YES", "NO"),
            prec_1_stats$rel_error, prec_1_stats$abs_error))
cat(sprintf("prec_y      | %10.4f | %10.4f | %8.4f | [%6.4f,%6.4f] | %8s | %6.2f%% | %7.4f\n",
            prec_y_true, prec_y_stats$median, prec_y_stats$sd,
            prec_y_stats$ci_lower, prec_y_stats$ci_upper,
            ifelse(prec_y_stats$coverage, "YES", "NO"),
            prec_y_stats$rel_error, prec_y_stats$abs_error))
cat("================================================================================\n\n")

# Additional quantile information
cat("Detailed Posterior Quantiles:\n")
cat("======================================================\n")
cat("Parameter   |   2.5%   |  25%     |  50%     |  75%     |  97.5%\n")
cat("======================================================\n")
theta_01_quantiles <- quantile(theta_01_chain, c(0.025, 0.25, 0.5, 0.75, 0.975))
prec_1_quantiles <- quantile(prec_1_chain, c(0.025, 0.25, 0.5, 0.75, 0.975))
prec_y_quantiles <- quantile(prec_y_chain, c(0.025, 0.25, 0.5, 0.75, 0.975))

cat(sprintf("theta_01    | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f\n",
            theta_01_quantiles[1], theta_01_quantiles[2], theta_01_quantiles[3],
            theta_01_quantiles[4], theta_01_quantiles[5]))
cat(sprintf("prec_1      | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f\n",
            prec_1_quantiles[1], prec_1_quantiles[2], prec_1_quantiles[3],
            prec_1_quantiles[4], prec_1_quantiles[5]))
cat(sprintf("prec_y      | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f\n",
            prec_y_quantiles[1], prec_y_quantiles[2], prec_y_quantiles[3],
            prec_y_quantiles[4], prec_y_quantiles[5]))
cat("======================================================\n\n")

# --- 7. Latent State (theta_1) Analysis ---

# Compute estimates and confidence intervals for theta_1
theta_1_estimate <- apply(theta_1_chain, 2, median)
theta_1_ci_lower <- apply(theta_1_chain, 2, quantile, 0.025)
theta_1_ci_upper <- apply(theta_1_chain, 2, quantile, 0.975)

# 1. Segmental Analysis
cat("=== LATENT STATE (theta_1) ANALYSIS ===\n\n")
cat("1. Analysis by Temporal Segments:\n")
cat("==========================================\n")
n_segments <- 5
segment_size <- n %/% n_segments
segment_stats <- data.frame(
  Segment = 1:n_segments,
  Time_Range = paste0("[", (0:(n_segments-1))*segment_size + 1, "-", 
                     (1:n_segments)*segment_size, "]"),
  True_Mean = NA,
  Est_Mean = NA,
  Est_SD = NA,
  Coverage_95 = NA,
  RMSE = NA
)

for (i in 1:n_segments) {
  start_idx <- (i-1)*segment_size + 1
  end_idx <- i*segment_size
  
  segment_stats$True_Mean[i] <- mean(theta1_true[start_idx:end_idx])
  segment_stats$Est_Mean[i] <- mean(theta_1_estimate[start_idx:end_idx])
  segment_stats$Est_SD[i] <- mean(apply(theta_1_chain[, start_idx:end_idx], 2, sd))
  segment_stats$Coverage_95[i] <- mean((theta1_true[start_idx:end_idx] >= theta_1_ci_lower[start_idx:end_idx]) & 
                                      (theta1_true[start_idx:end_idx] <= theta_1_ci_upper[start_idx:end_idx]))
  segment_stats$RMSE[i] <- sqrt(mean((theta_1_estimate[start_idx:end_idx] - theta1_true[start_idx:end_idx])^2))
}

cat("Segment |  Time Range  | True Mean | Est Mean | Est SD  | Coverage | RMSE\n")
cat("========================================================================\n")
for (i in 1:n_segments) {
  cat(sprintf("%7d | %12s | %9.4f | %8.4f | %7.4f | %8.2f%% | %6.4f\n",
              segment_stats$Segment[i], segment_stats$Time_Range[i], 
              segment_stats$True_Mean[i], segment_stats$Est_Mean[i], 
              segment_stats$Est_SD[i], 100*segment_stats$Coverage_95[i], 
              segment_stats$RMSE[i]))
}
cat("========================================================================\n\n")

# 2. Global Quality Metrics
cat("2. Global Quality Metrics for theta_1:\n")
cat("======================================\n")
abs_errors <- abs(theta_1_estimate - theta1_true)
rel_errors <- abs_errors / abs(theta1_true)
coverage <- (theta1_true >= theta_1_ci_lower) & (theta1_true <= theta_1_ci_upper)

theta_1_global_metrics <- data.frame(
  Metric = c("RMSE", "MAE", "MAPE (%)", "Coverage_95 (%)", "Correlation", "R²"),
  Value = c(
    sqrt(mean((theta_1_estimate - theta1_true)^2)),
    mean(abs(theta_1_estimate - theta1_true)),
    100 * mean(abs((theta_1_estimate - theta1_true) / theta1_true)),
    100 * mean(coverage),
    cor(theta_1_estimate, theta1_true),
    cor(theta_1_estimate, theta1_true)^2
  )
)

cat("Metric           | Value\n")
cat("=========================\n")
for (i in 1:nrow(theta_1_global_metrics)) {
  cat(sprintf("%-15s | %10.6f\n", theta_1_global_metrics$Metric[i], theta_1_global_metrics$Value[i]))
}
cat("=========================\n\n")

# 3. Outliers and Problematic Points
cat("3. Outliers and Problematic Points:\n")
cat("===================================\n")
worst_abs_idx <- order(abs_errors, decreasing=TRUE)[1:10]
worst_rel_idx <- order(rel_errors, decreasing=TRUE)[1:10]
coverage_failures <- which(!coverage)

outlier_analysis <- data.frame(
  Category = c("Worst 10 Abs Errors", "Worst 10 Rel Errors", "Coverage Failures"),
  Count = c(10, 10, length(coverage_failures)),
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
    paste(worst_abs_idx[1:5], collapse=", "),
    paste(worst_rel_idx[1:5], collapse=", "),
    ifelse(length(coverage_failures) >= 5, 
           paste(coverage_failures[1:5], collapse=", "), 
           ifelse(length(coverage_failures) > 0, paste(coverage_failures, collapse=", "), "None"))
  )
)

cat("Category            | Count | Mean Error | Max Error  | First 5 Points\n")
cat("=====================================================================\n")
for (i in 1:nrow(outlier_analysis)) {
  cat(sprintf("%-18s | %5d | %10.6f | %10.6f | %s\n",
              outlier_analysis$Category[i], outlier_analysis$Count[i], 
              outlier_analysis$Mean_Error[i], outlier_analysis$Max_Error[i], 
              outlier_analysis$First_5_Points[i]))
}
cat("=====================================================================\n\n")

# 4. Temporal Autocorrelation Analysis
cat("4. Temporal Autocorrelation Analysis:\n")
cat("====================================\n")
max_lag <- 10
lag_analysis <- data.frame(
  Lag = 1:max_lag,
  True_ACF = sapply(1:max_lag, function(k) {
    if(k < n) cor(theta1_true[1:(n-k)], theta1_true[(1+k):n]) else NA
  }),
  Est_ACF = sapply(1:max_lag, function(k) {
    if(k < n) cor(theta_1_estimate[1:(n-k)], theta_1_estimate[(1+k):n]) else NA
  }),
  Error_ACF = sapply(1:max_lag, function(k) {
    if(k < n) cor(abs_errors[1:(n-k)], abs_errors[(1+k):n]) else NA
  })
)

cat("Lag | True ACF | Est ACF  | Error ACF\n")
cat("===================================\n")
for (i in 1:nrow(lag_analysis)) {
  cat(sprintf("%3d | %8.4f | %8.4f | %9.4f\n",
              lag_analysis$Lag[i], lag_analysis$True_ACF[i], 
              lag_analysis$Est_ACF[i], lag_analysis$Error_ACF[i]))
}
cat("===================================\n\n")

# 5. Effective Sample Size by Time Points
cat("5. Effective Sample Size by Time Points:\n")
cat("========================================\n")
time_points <- seq(1, n, length.out = 10)
ess_by_time <- sapply(time_points, function(t) {
  t_idx <- round(t)
  chain_t <- theta_1_chain[, t_idx]
  acf_vals <- acf(chain_t, plot=FALSE, lag.max=min(100, length(chain_t)/4))$acf[-1]
  max(1, length(chain_t) / (1 + 2 * sum(acf_vals[acf_vals > 0])))
})

ess_time_analysis <- data.frame(
  Time_Point = round(time_points),
  ESS = round(ess_by_time),
  Efficiency = round(100 * ess_by_time / n_chain, 1)
)

cat("Time Point | ESS  | Efficiency (%)\n")
cat("===============================\n")
for (i in 1:nrow(ess_time_analysis)) {
  cat(sprintf("%10d | %4d | %12.1f\n",
              ess_time_analysis$Time_Point[i], ess_time_analysis$ESS[i], 
              ess_time_analysis$Efficiency[i]))
}
cat("===============================\n")

# Fix the problematic sprintf line - ensure all values are finite and numeric
ess_min <- round(min(ess_by_time[is.finite(ess_by_time)]))
ess_max <- round(max(ess_by_time[is.finite(ess_by_time)]))
ess_mean <- round(mean(ess_by_time[is.finite(ess_by_time)]), 1)
ess_efficiency <- round(100 * ess_mean / n_chain, 1)

cat(sprintf("Overall ESS range: [%d, %d]\n", ess_min, ess_max))
cat(sprintf("Mean ESS: %.1f (%.1f%% efficiency)\n\n", ess_mean, ess_efficiency))

# Convergence diagnostics for scalar parameters
cat("Convergence diagnostics (Scalar Parameters):\n")
cat("--------------------------------------------\n")
effective_sample_sizes <- sapply(list(theta_01_chain, prec_1_chain, prec_y_chain),
                                 function(x) {
                                   acf_vals <- acf(x, plot=FALSE, lag.max=min(100, length(x)/4))$acf[-1]
                                   length(x) / (1 + 2 * sum(acf_vals[acf_vals > 0]))
                                 })
names(effective_sample_sizes) <- c("theta_01", "prec_1", "prec_y")
print(round(effective_sample_sizes))

# --- 8. Visualization ---
op <- par(mfrow = c(2, 3))

# Posterior histograms
hist(theta_01_chain, main = expression(paste("Posterior of ", theta["0,1"])),
     xlab = "", col = "lightblue", border = "white")
abline(v = theta0_true, col = "red", lwd = 2, lty = 2)
abline(v = median(theta_01_chain), col = "blue", lwd = 2)

hist(prec_1_chain, main = expression(paste("Posterior of ", 1/W[1])),
     xlab = "", col = "lightgreen", border = "white")
abline(v = prec1_true, col = "red", lwd = 2, lty = 2)
abline(v = median(prec_1_chain), col = "blue", lwd = 2)

hist(prec_y_chain, main = expression(paste("Posterior of ", 1/V)),
     xlab = "", col = "lightcoral", border = "white")
abline(v = prec_y_true, col = "red", lwd = 2, lty = 2)
abline(v = median(prec_y_chain), col = "blue", lwd = 2)

# Trace plots for diagnostics
plot(theta_01_chain, type = 'l', main = expression(paste("Trace plot: ", theta["0,1"])),
     ylab = "", col = "blue")
abline(h = theta0_true, col = "red", lty = 2)

plot(prec_1_chain, type = 'l', main = expression(paste("Trace plot: ", 1/W[1])),
     ylab = "", col = "darkgreen")
abline(h = prec1_true, col = "red", lty = 2)

# Estimated vs true latent state
plot(theta1_true, type = 'l', col = "red", lty = 2, lwd = .5,
     main = "Latent State: True vs. Estimated", ylab = expression(theta["t,1"]))
lines(theta_1_estimate, col = "blue", lwd = 1)
polygon(c(1:n, rev(1:n)), c(theta_1_ci_lower, rev(theta_1_ci_upper)),
        col = rgb(0, 0, 1, 0.2), border = NA)
legend("topleft", legend = c("True", "Estimated", "95% CI"),
       col = c("red", "blue", rgb(0, 0, 1, 0.2)),
       lty = c(2, 1, 1), lwd = c(2, 2, 8), bty = "n")
par(op)