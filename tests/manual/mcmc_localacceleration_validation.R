#-------------------------------------------------------------------------------
# Test Script for Local Acceleration Model Auxiliary Functions
#
# Objective: Validate the auxiliary C functions (exposed in test_helpers.c)
#           by replicating the Gibbs sampler loop in R following mcmc_localacceleration.c
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

# Load required packages for enhanced diagnostics
suppressPackageStartupMessages({
  if (!require(coda, quietly = TRUE)) {
    cat("Warning: 'coda' package not available. Some convergence diagnostics will be skipped.\n")
  }
})

# --- 1. Data Simulation ---
n <- 1000  # Number of observations

# True parameters
theta01_true <- 0.1    # Initial level
theta02_true <- 0.05   # Initial trend
theta03_true <- 0.01   # Initial acceleration
prec1_true   <- 10     # Level precision
prec2_true   <- 20     # Trend precision
prec3_true   <- 30     # Acceleration precision
prec_y_true  <- 1      # Data precision

# set.seed(123) # For reproducibility

# Generate noise
u1 <- rnorm(n, sd = sqrt(1 / prec1_true))
u2 <- rnorm(n, sd = sqrt(1 / prec2_true))
u3 <- rnorm(n, sd = sqrt(1 / prec3_true))
e  <- rnorm(n, sd = sqrt(1 / prec_y_true))

# Simulate latent states and observations
theta1_true <- numeric(n)
theta2_true <- numeric(n)
theta3_true <- numeric(n)

# Initialize first values
theta1_true[1] <- theta01_true + theta02_true + u1[1]
theta2_true[1] <- theta02_true + theta03_true + u2[1]
theta3_true[1] <- theta03_true + u3[1]

# Generate remaining values
for (t in 2:n) {
  theta1_true[t] <- theta1_true[t-1] + theta2_true[t-1] + u1[t]
  theta2_true[t] <- theta2_true[t-1] + theta3_true[t-1] + u2[t]
  theta3_true[t] <- theta3_true[t-1] + u3[t]
}

# Observations
y <- theta1_true + e

# --- 2. MCMC Configuration ---
burnin   <- 1000
thinning <- 1
n_chain  <- 5000
# Use the same formula as the C implementation
n_iter   <- burnin + (n_chain - 1) * thinning + 1

# Prior hyperparameters (following C implementation nomenclature)
mean_theta01 <- 0                   # prior_theta01_mean
prec_theta01 <- .01                 # prior_theta01_prec
mean_theta02 <- 0                   # prior_theta02_mean
prec_theta02 <- .01                 # prior_theta02_prec
mean_theta03 <- 0                   # prior_theta03_mean
prec_theta03 <- .01                 # prior_theta03_prec
nu_01        <- 1e-1                # prior_prec1_shape
eta_01       <- 1e-1                # prior_prec1_rate
nu_02        <- 1e-1                # prior_prec2_shape
eta_02       <- 1e-1                # prior_prec2_rate
nu_03        <- 1e-1                # prior_prec3_shape
eta_03       <- 1e-1                # prior_prec3_rate
nu_y         <- 1e-1                # prior_prec_y_shape
eta_y        <- 1e-1                # prior_prec_y_rate

# --- 3. Chain Initialization ---
theta_1_chain  <- matrix(NA, nrow = n_chain, ncol = n)
theta_2_chain  <- matrix(NA, nrow = n_chain, ncol = n)
theta_3_chain  <- matrix(NA, nrow = n_chain, ncol = n)
theta_01_chain <- numeric(n_chain)
theta_02_chain <- numeric(n_chain)
theta_03_chain <- numeric(n_chain)
prec_1_chain   <- numeric(n_chain)
prec_2_chain   <- numeric(n_chain)
prec_3_chain   <- numeric(n_chain)
prec_y_chain   <- numeric(n_chain)

# --- 4. Initialization (iter = 0) following mcmc_localacceleration.c ---
# set.seed(456)

# Arrays to store complete history (as in C implementation)
theta_01_post <- numeric(n_iter)
theta_02_post <- numeric(n_iter)
theta_03_post <- numeric(n_iter)
prec_1_post   <- numeric(n_iter)
prec_2_post   <- numeric(n_iter)
prec_3_post   <- numeric(n_iter)
prec_y_post   <- numeric(n_iter)
theta_1_post  <- matrix(NA, nrow = n_iter, ncol = n)
theta_2_post  <- matrix(NA, nrow = n_iter, ncol = n)
theta_3_post  <- matrix(NA, nrow = n_iter, ncol = n)

# Initial values exactly as in C implementation
theta_01_post[1] <- rnorm(1, mean_theta01, sqrt(1.0 / prec_theta01))
theta_02_post[1] <- rnorm(1, mean_theta02, sqrt(1.0 / prec_theta02))
theta_03_post[1] <- rnorm(1, mean_theta03, sqrt(1.0 / prec_theta03))
prec_1_post[1]   <- rgamma(1, nu_01, rate = eta_01)
prec_2_post[1]   <- rgamma(1, nu_02, rate = eta_02)
prec_3_post[1]   <- rgamma(1, nu_03, rate = eta_03)
prec_y_post[1]   <- rgamma(1, nu_y, rate = eta_y)

# Initialize state vectors for t = 1..n
init_sd_1 <- sqrt(1.0 / prec_1_post[1])
init_sd_2 <- sqrt(1.0 / prec_2_post[1])
init_sd_3 <- sqrt(1.0 / prec_3_post[1])
theta_1_post[1, 1] <- rnorm(1, theta_01_post[1] + theta_02_post[1], init_sd_1)
theta_2_post[1, 1] <- rnorm(1, theta_02_post[1] + theta_03_post[1], init_sd_2)
theta_3_post[1, 1] <- rnorm(1, theta_03_post[1], init_sd_3)
for (j in 2:n) {
  theta_1_post[1, j] <- rnorm(1, theta_1_post[1, j-1] + theta_2_post[1, j-1], init_sd_1)
  theta_2_post[1, j] <- rnorm(1, theta_2_post[1, j-1] + theta_3_post[1, j-1], init_sd_2)
  theta_3_post[1, j] <- rnorm(1, theta_3_post[1, j-1], init_sd_3)
}

# --- 5. MCMC Loop using Auxiliary Functions ---
cat("Starting MCMC loop for Local Acceleration model...\n")

# Initialize progress bar
pb <- txtProgressBar(min = 2, max = n_iter, style = 3, width = 60, char = "\u27a4")
cat("Progress: ", rep(" ", 60), "\n")

chain_idx <- 0  # Counter for saved samples

for (ii in 2:n_iter) {  # Start at 2 since iter=1 is already initialized

  # 1) Sample acceleration state vector theta_3
  # Function: test_generate_theta_p(theta_pm1_, theta_p_, prec_theta_pm1_, prec_theta_p_, theta_0p_)
  theta_3_new <- .Call(
    "_pdm_test_generate_theta_p",
    as.numeric(theta_2_post[ii-1, ]),   # theta_pm1 (trend states, previous iter)
    as.numeric(prec_2_post[ii-1]),      # prec_theta_pm1 (trend precision, previous iter)
    as.numeric(prec_3_post[ii-1]),      # prec_theta_p (acceleration precision, previous iter)
    as.numeric(theta_03_post[ii-1])     # theta_0p (initial acceleration, previous iter)
  )

  if (length(theta_3_new) == n) {
    theta_3_post[ii, ] <- as.numeric(theta_3_new)
  } else {
    warning(paste("Iteration", ii, ": theta_3 was not updated correctly"))
    theta_3_post[ii, ] <- theta_3_post[ii-1, ]  # keep previous value
  }

  # 2) Sample innovation precision 1/W_3
  # Function: test_generate_precision_theta_p(theta_0p_, theta_p_, nu_0p_, eta_0p_)
  prec_3_post[ii] <- .Call(
    "_pdm_test_generate_precision_theta_p",
    as.numeric(theta_03_post[ii-1]),    # theta_0p (initial acceleration, previous iter)
    as.numeric(theta_3_post[ii, ]),     # theta_p (acceleration states, current)
    as.numeric(nu_03),                  # nu_0p (prior shape)
    as.numeric(eta_03)                  # eta_0p (prior rate)
  )

  # 3) Sample initial acceleration state theta_03
  # Function: test_generate_theta_0p(theta_0pm1_, theta_pm1_, theta_p_, prec_theta_pm1_, prec_theta_p_, mean_theta_0p_, prec_theta_0p_)
  theta_03_post[ii] <- .Call(
    "_pdm_test_generate_theta_0p",
    as.numeric(theta_2_post[ii-1, ]),   # theta_pm1 (trend states, current)
    as.numeric(theta_3_post[ii, ]),     # theta_p (acceleration states, current)
    as.numeric(theta_02_post[ii-1]),    # theta_0pm1 (initial trend, previous iter)
    as.numeric(prec_2_post[ii-1]),      # prec_theta_pm1 (trend precision, previous iter)
    as.numeric(prec_3_post[ii]),        # prec_theta_p (acceleration precision, current)
    as.numeric(mean_theta03),           # mean_theta_0p (prior mean)
    as.numeric(prec_theta03)            # prec_theta_0p (prior precision)
  )

  # 4) Sample trend state vector theta_2
  # Function: test_generate_theta_k(theta_km1_, theta_kp1_, theta_kp1_, prec_theta_km1_, prec_theta_k_, theta_0k_, theta_0kp1_)
  theta_2_new <- .Call(
    "_pdm_test_generate_theta_k",
    as.numeric(theta_1_post[ii-1, ]),   # theta_km1 (level states, previous iter)
    as.numeric(theta_3_post[ii, ]),     # theta_kp1 (acceleration states, current)
    as.numeric(prec_1_post[ii-1]),      # prec_theta_km1 (level precision, previous iter)
    as.numeric(prec_2_post[ii-1]),      # prec_theta_k (trend precision, previous iter)
    as.numeric(theta_02_post[ii-1]),    # theta_0k (initial trend, previous iter)
    as.numeric(theta_03_post[ii])       # theta_0kp1 (initial acceleration, current)
  )

  if (length(theta_2_new) == n) {
    theta_2_post[ii, ] <- as.numeric(theta_2_new)
  } else {
    warning(paste("Iteration", ii, ": theta_2 was not updated correctly"))
    theta_2_post[ii, ] <- theta_2_post[ii-1, ]  # keep previous value
  }

  # 5) Sample innovation precision 1/W_2
  # Function: test_generate_precision_theta_k(theta_0k_, theta_0kp1_, theta_k_, theta_kp1_, nu_0k_, eta_0k_)
  prec_2_post[ii] <- .Call(
    "_pdm_test_generate_precision_theta_k",
    as.numeric(theta_02_post[ii-1]),    # theta_0k (initial trend, previous iter)
    as.numeric(theta_03_post[ii]),      # theta_0kp1 (initial acceleration, current)
    as.numeric(theta_2_post[ii, ]),     # theta_k (trend states, current)
    as.numeric(theta_3_post[ii, ]),     # theta_kp1 (acceleration states, current)
    as.numeric(nu_02),                  # nu_0k (prior shape)
    as.numeric(eta_02)                  # eta_0k (prior rate)
  )

  # 6) Sample initial trend state theta_02
  # Function: test_generate_theta_0k(theta_0km1_, theta_0kp1_, theta_km1_, theta_k_, prec_theta_km1_, prec_theta_k_, mean_theta_0k_, prec_theta_0k_)
  theta_02_post[ii] <- .Call(
    "_pdm_test_generate_theta_0k",
    as.numeric(theta_1_post[ii-1, ]),   # theta_km1 (level states, previous iter)
    as.numeric(theta_2_post[ii, ]),     # theta_k (trend states, current)
    as.numeric(theta_01_post[ii-1]),    # theta_0km1 (initial level, previous iter)
    as.numeric(theta_03_post[ii]),      # theta_0kp1 (initial acceleration, current)
    as.numeric(prec_1_post[ii-1]),      # prec_theta_km1 (level precision, previous iter)
    as.numeric(prec_2_post[ii]),        # prec_theta_k (trend precision, current)
    as.numeric(mean_theta02),           # mean_theta_0k (prior mean)
    as.numeric(prec_theta02)            # prec_theta_0k (prior precision)
  )

  # 7) Sample level state vector theta_1
  # Function: test_generate_theta_1(data_, theta_2_, prec_data_, prec_theta_1_, theta_01_, theta_02_)
  theta_1_new <- .Call(
    "_pdm_test_generate_theta_1",
    as.numeric(y),                      # data (observed values)
    as.numeric(theta_2_post[ii, ]),     # theta_2 (trend states, current)
    as.numeric(prec_y_post[ii-1]),      # prec_data (data precision, previous iter)
    as.numeric(prec_1_post[ii-1]),      # prec_theta_1 (level precision, previous iter)
    as.numeric(theta_01_post[ii-1]),    # theta_01 (initial level, previous iter)
    as.numeric(theta_02_post[ii])       # theta_02 (initial trend, current)
  )

  if (length(theta_1_new) == n) {
    theta_1_post[ii, ] <- as.numeric(theta_1_new)
  } else {
    warning(paste("Iteration", ii, ": theta_1 was not updated correctly"))
    theta_1_post[ii, ] <- theta_1_post[ii-1, ]  # keep previous value
  }

  # 8) Sample innovation precision 1/W_1
  # Function: test_generate_precision_theta_k(theta_0k_, theta_0kp1_, theta_k_, theta_kp1_, nu_0k_, eta_0k_)
  prec_1_post[ii] <- .Call(
    "_pdm_test_generate_precision_theta_k",
    as.numeric(theta_01_post[ii-1]),    # theta_0k (initial level, previous iter)
    as.numeric(theta_02_post[ii]),      # theta_0kp1 (initial trend, current)
    as.numeric(theta_1_post[ii, ]),     # theta_k (level states, current)
    as.numeric(theta_2_post[ii, ]),     # theta_kp1 (trend states, current)
    as.numeric(nu_01),                  # nu_0k (prior shape)
    as.numeric(eta_01)                  # eta_0k (prior rate)
  )

  # 9) Sample initial level state theta_01
  # Function: test_generate_theta_01(theta_1_, theta_02_, prec_theta_1_, mean_theta_01_, prec_theta_01_)
  theta_01_post[ii] <- .Call(
    "_pdm_test_generate_theta_01",
    as.numeric(theta_1_post[ii, ]),     # theta_1 (level states, current)
    as.numeric(theta_02_post[ii]),      # theta_02 (initial trend, current)
    as.numeric(prec_1_post[ii]),        # prec_theta_1 (level precision, current)
    as.numeric(mean_theta01),           # mean_theta_01 (prior mean)
    as.numeric(prec_theta01)            # prec_theta_01 (prior precision)
  )

  # 10) Sample data precision 1/V
  # Function: test_generate_precision_data(y_, theta_1_, nu_y_, eta_y_)
  prec_y_post[ii] <- .Call(
    "_pdm_test_generate_precision_data",
    as.numeric(y),                      # y (observed data)
    as.numeric(theta_1_post[ii, ]),     # theta_1 (level states, current)
    as.numeric(nu_y),                   # nu_y (prior shape)
    as.numeric(eta_y)                   # eta_y (prior rate)
  )

  # Store samples after burn-in and applying thinning
  # Following exactly the logic of the C implementation
  if (ii >= (burnin + 1) && ((ii - burnin - 1) %% thinning == 0)) {
    chain_idx <- chain_idx + 1
    theta_1_chain[chain_idx, ]  <- theta_1_post[ii, ]
    theta_2_chain[chain_idx, ]  <- theta_2_post[ii, ]
    theta_3_chain[chain_idx, ]  <- theta_3_post[ii, ]
    theta_01_chain[chain_idx] <- theta_01_post[ii]
    theta_02_chain[chain_idx] <- theta_02_post[ii]
    theta_03_chain[chain_idx] <- theta_03_post[ii]
    prec_1_chain[chain_idx]   <- prec_1_post[ii]
    prec_2_chain[chain_idx]   <- prec_2_post[ii]
    prec_3_chain[chain_idx]   <- prec_3_post[ii]
    prec_y_chain[chain_idx]   <- prec_y_post[ii]
  }

  # Update progress bar every 100 iterations
  if (ii %% 100 == 0) {
    setTxtProgressBar(pb, ii)
  }
}

# Close progress bar
close(pb)
cat("\n... MCMC loop completed.\n\n")

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
theta_01_stats <- compute_summary_stats(theta_01_chain, theta01_true)
theta_02_stats <- compute_summary_stats(theta_02_chain, theta02_true)
theta_03_stats <- compute_summary_stats(theta_03_chain, theta03_true)
prec_1_stats   <- compute_summary_stats(prec_1_chain, prec1_true)
prec_2_stats   <- compute_summary_stats(prec_2_chain, prec2_true)
prec_3_stats   <- compute_summary_stats(prec_3_chain, prec3_true)
prec_y_stats   <- compute_summary_stats(prec_y_chain, prec_y_true)

cat("Enhanced Summary of Posterior Estimates:\n")
cat("=========================================================================================================================\n")
cat("Parameter   | True Value | Median Est |  Post SD | Monte Carlo SE |               95% CI | Coverage |  Rel Err | Abs Err\n")
cat("=========================================================================================================================\n")
cat(sprintf("theta_01    | %10.4f | %10.4f | %8.4f | %14.4f | [%8.4f, %8.4f] | %8s | %7.2f%% | %7.4f\n",
            theta01_true, theta_01_stats$median, theta_01_stats$sd, theta_01_stats$sd / sqrt(n_chain),
            theta_01_stats$ci_lower, theta_01_stats$ci_upper,
            ifelse(theta_01_stats$coverage, "YES", "NO"),
            theta_01_stats$rel_error, theta_01_stats$abs_error))
cat(sprintf("theta_02    | %10.4f | %10.4f | %8.4f | %14.4f | [%8.4f, %8.4f] | %8s | %7.2f%% | %7.4f\n",
            theta02_true, theta_02_stats$median, theta_02_stats$sd, theta_02_stats$sd / sqrt(n_chain),
            theta_02_stats$ci_lower, theta_02_stats$ci_upper,
            ifelse(theta_02_stats$coverage, "YES", "NO"),
            theta_02_stats$rel_error, theta_02_stats$abs_error))
cat(sprintf("theta_03    | %10.4f | %10.4f | %8.4f | %14.4f | [%8.4f, %8.4f] | %8s | %7.2f%% | %7.4f\n",
            theta03_true, theta_03_stats$median, theta_03_stats$sd, theta_03_stats$sd / sqrt(n_chain),
            theta_03_stats$ci_lower, theta_03_stats$ci_upper,
            ifelse(theta_03_stats$coverage, "YES", "NO"),
            theta_03_stats$rel_error, theta_03_stats$abs_error))
cat(sprintf("prec_1      | %10.4f | %10.4f | %8.4f | %14.4f | [%8.4f, %8.4f] | %8s | %7.2f%% | %7.4f\n",
            prec1_true, prec_1_stats$median, prec_1_stats$sd, prec_1_stats$sd / sqrt(n_chain),
            prec_1_stats$ci_lower, prec_1_stats$ci_upper,
            ifelse(prec_1_stats$coverage, "YES", "NO"),
            prec_1_stats$rel_error, prec_1_stats$abs_error))
cat(sprintf("prec_2      | %10.4f | %10.4f | %8.4f | %14.4f | [%8.4f, %8.4f] | %8s | %7.2f%% | %7.4f\n",
            prec2_true, prec_2_stats$median, prec_2_stats$sd, prec_2_stats$sd / sqrt(n_chain),
            prec_2_stats$ci_lower, prec_2_stats$ci_upper,
            ifelse(prec_2_stats$coverage, "YES", "NO"),
            prec_2_stats$rel_error, prec_2_stats$abs_error))
cat(sprintf("prec_3      | %10.4f | %10.4f | %8.4f | %14.4f | [%8.4f, %8.4f] | %8s | %7.2f%% | %7.4f\n",
            prec3_true, prec_3_stats$median, prec_3_stats$sd, prec_3_stats$sd / sqrt(n_chain),
            prec_3_stats$ci_lower, prec_3_stats$ci_upper,
            ifelse(prec_3_stats$coverage, "YES", "NO"),
            prec_3_stats$rel_error, prec_3_stats$abs_error))
cat(sprintf("prec_y      | %10.4f | %10.4f | %8.4f | %14.4f | [%8.4f, %8.4f] | %8s | %7.2f%% | %7.4f\n",
            prec_y_true, prec_y_stats$median, prec_y_stats$sd, prec_y_stats$sd / sqrt(n_chain),
            prec_y_stats$ci_lower, prec_y_stats$ci_upper,
            ifelse(prec_y_stats$coverage, "YES", "NO"),
            prec_y_stats$rel_error, prec_y_stats$abs_error))
cat("=========================================================================================================================\n")
cat("Note: Monte Carlo SE measures the precision of posterior mean estimates\n\n")

# Additional quantile information
cat("Detailed Posterior Quantiles:\n")
cat("===================================================================\n")
cat("Parameter   |     2.5% |      25% |      50% |      75% |    97.5%\n")
cat("===================================================================\n")
theta_01_quantiles <- quantile(theta_01_chain, c(0.025, 0.25, 0.5, 0.75, 0.975))
theta_02_quantiles <- quantile(theta_02_chain, c(0.025, 0.25, 0.5, 0.75, 0.975))
theta_03_quantiles <- quantile(theta_03_chain, c(0.025, 0.25, 0.5, 0.75, 0.975))
prec_1_quantiles <- quantile(prec_1_chain, c(0.025, 0.25, 0.5, 0.75, 0.975))
prec_2_quantiles <- quantile(prec_2_chain, c(0.025, 0.25, 0.5, 0.75, 0.975))
prec_3_quantiles <- quantile(prec_3_chain, c(0.025, 0.25, 0.5, 0.75, 0.975))
prec_y_quantiles <- quantile(prec_y_chain, c(0.025, 0.25, 0.5, 0.75, 0.975))

cat(sprintf("theta_01    | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f\n",
            theta_01_quantiles[1], theta_01_quantiles[2], theta_01_quantiles[3],
            theta_01_quantiles[4], theta_01_quantiles[5]))
cat(sprintf("theta_02    | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f\n",
            theta_02_quantiles[1], theta_02_quantiles[2], theta_02_quantiles[3],
            theta_02_quantiles[4], theta_02_quantiles[5]))
cat(sprintf("theta_03    | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f\n",
            theta_03_quantiles[1], theta_03_quantiles[2], theta_03_quantiles[3],
            theta_03_quantiles[4], theta_03_quantiles[5]))
cat(sprintf("prec_1      | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f\n",
            prec_1_quantiles[1], prec_1_quantiles[2], prec_1_quantiles[3],
            prec_1_quantiles[4], prec_1_quantiles[5]))
cat(sprintf("prec_2      | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f\n",
            prec_2_quantiles[1], prec_2_quantiles[2], prec_2_quantiles[3],
            prec_2_quantiles[4], prec_2_quantiles[5]))
cat(sprintf("prec_3      | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f\n",
            prec_3_quantiles[1], prec_3_quantiles[2], prec_3_quantiles[3],
            prec_3_quantiles[4], prec_3_quantiles[5]))
cat(sprintf("prec_y      | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f\n",
            prec_y_quantiles[1], prec_y_quantiles[2], prec_y_quantiles[3],
            prec_y_quantiles[4], prec_y_quantiles[5]))
cat("===================================================================\n\n")

# --- 7. Latent State Analysis for theta_1, theta_2 and theta_3 ---

# Compute estimates and confidence intervals for all states
theta_1_estimate <- apply(theta_1_chain, 2, median)
theta_1_ci_lower <- apply(theta_1_chain, 2, quantile, 0.025)
theta_1_ci_upper <- apply(theta_1_chain, 2, quantile, 0.975)

theta_2_estimate <- apply(theta_2_chain, 2, median)
theta_2_ci_lower <- apply(theta_2_chain, 2, quantile, 0.025)
theta_2_ci_upper <- apply(theta_2_chain, 2, quantile, 0.975)

theta_3_estimate <- apply(theta_3_chain, 2, median)
theta_3_ci_lower <- apply(theta_3_chain, 2, quantile, 0.025)
theta_3_ci_upper <- apply(theta_3_chain, 2, quantile, 0.975)

# Function to analyze latent state
analyze_latent_state <- function(true_values, estimates, ci_lower, ci_upper, chain, state_name) {
  cat(sprintf("=== %s ANALYSIS ===\n\n", toupper(state_name)))

  # 1. Segmental Analysis
  cat("1. Analysis by Temporal Segments:\n")
  cat("=========================================================================================\n")
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

    segment_stats$True_Mean[i] <- mean(true_values[start_idx:end_idx])
    segment_stats$Est_Mean[i] <- mean(estimates[start_idx:end_idx])
    segment_stats$Est_SD[i] <- mean(apply(chain[, start_idx:end_idx], 2, sd))
    segment_stats$Coverage_95[i] <- mean((true_values[start_idx:end_idx] >= ci_lower[start_idx:end_idx]) &
                                           (true_values[start_idx:end_idx] <= ci_upper[start_idx:end_idx]))
    segment_stats$RMSE[i] <- sqrt(mean((estimates[start_idx:end_idx] - true_values[start_idx:end_idx])^2))
  }

  cat("Segment |   Time Range |    True Mean |     Est Mean |     Est SD |  Coverage |    RMSE\n")
  cat("=========================================================================================\n")
  for (i in 1:n_segments) {
    cat(sprintf("%7d | %12s | %12.4f | %12.4f | %10.4f | %8.2f%% | %7.4f\n",
                segment_stats$Segment[i], segment_stats$Time_Range[i],
                segment_stats$True_Mean[i], segment_stats$Est_Mean[i],
                segment_stats$Est_SD[i], 100*segment_stats$Coverage_95[i],
                segment_stats$RMSE[i]))
  }
  cat("=========================================================================================\n\n")

  # 2. Global Quality Metrics
  cat("2. Global Quality Metrics:\n")
  cat("==============================\n")
  abs_errors <- abs(estimates - true_values)
  rel_errors <- abs_errors / abs(true_values)
  coverage <- (true_values >= ci_lower) & (true_values <= ci_upper)

  global_metrics <- data.frame(
    Metric = c("RMSE", "MAE", "MAPE (%)", "Coverage_95 (%)", "Correlation", "R^2"),
    Value = c(
      sqrt(mean((estimates - true_values)^2)),
      mean(abs(estimates - true_values)),
      100 * mean(abs((estimates - true_values) / true_values)),
      100 * mean(coverage),
      cor(estimates, true_values),
      cor(estimates, true_values)^2
    )
  )

  cat("Metric          |      Value\n")
  cat("==============================\n")
  for (i in 1:nrow(global_metrics)) {
    cat(sprintf("%-15s | %10.6f\n", global_metrics$Metric[i], global_metrics$Value[i]))
  }
  cat("==============================\n\n")

  # 3. Outliers and Problematic Points
  cat("3. Outliers and Problematic Points:\n")
  cat("==================================================================================\n")
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

  cat("Category             | Count | Mean Error |  Max Error |          First 5 Points\n")
  cat("==================================================================================\n")
  for (i in 1:nrow(outlier_analysis)) {
    cat(sprintf("%-20s | %5d | %10.6f | %10.6f | %s\n",
                outlier_analysis$Category[i], outlier_analysis$Count[i],
                outlier_analysis$Mean_Error[i], outlier_analysis$Max_Error[i],
                outlier_analysis$First_5_Points[i]))
  }
  cat("=================================================================================\n\n")

  # 4. Temporal Autocorrelation Analysis
  cat("4. Temporal Autocorrelation Analysis:\n")
  cat("======================================\n")
  max_lag <- 10
  lag_analysis <- data.frame(
    Lag = 1:max_lag,
    True_ACF = sapply(1:max_lag, function(k) {
      if(k < n) cor(true_values[1:(n-k)], true_values[(1+k):n]) else NA
    }),
    Est_ACF = sapply(1:max_lag, function(k) {
      if(k < n) cor(estimates[1:(n-k)], estimates[(1+k):n]) else NA
    }),
    Error_ACF = sapply(1:max_lag, function(k) {
      if(k < n) cor(abs_errors[1:(n-k)], abs_errors[(1+k):n]) else NA
    })
  )

  cat("Lag | True ACF |  Est ACF | Error ACF\n")
  cat("======================================\n")
  for (i in 1:nrow(lag_analysis)) {
    cat(sprintf("%3d | %8.4f | %8.4f | %9.4f\n",
                lag_analysis$Lag[i], lag_analysis$True_ACF[i],
                lag_analysis$Est_ACF[i], lag_analysis$Error_ACF[i]))
  }
  cat("======================================\n\n")

  # 5. Effective Sample Size by Time Points
  cat("5. Effective Sample Size by Time Points:\n")
  cat("====================================\n")
  time_points <- seq(1, n, length.out = 10)
  ess_by_time <- sapply(time_points, function(t) {
    t_idx <- round(t)
    chain_t <- chain[, t_idx]
    acf_vals <- acf(chain_t, plot=FALSE, lag.max=min(100, length(chain_t)/4))$acf[-1]
    max(1, length(chain_t) / (1 + 2 * sum(acf_vals[acf_vals > 0])))
  })

  ess_time_analysis <- data.frame(
    Time_Point = round(time_points),
    ESS = round(ess_by_time),
    Efficiency = round(100 * ess_by_time / n_chain, 1)
  )

  cat("Time Point |   ESS | Efficiency (%)\n")
  cat("====================================\n")
  for (i in 1:nrow(ess_time_analysis)) {
    cat(sprintf("%10d | %5d | %9.1f\n",
                ess_time_analysis$Time_Point[i], ess_time_analysis$ESS[i],
                ess_time_analysis$Efficiency[i]))
  }
  cat("====================================\n")

  ess_min <- round(min(ess_by_time[is.finite(ess_by_time)]))
  ess_max <- round(max(ess_by_time[is.finite(ess_by_time)]))
  ess_mean <- round(mean(ess_by_time[is.finite(ess_by_time)]), 1)
  ess_efficiency <- round(100 * ess_mean / n_chain, 1)

  cat(sprintf("Overall ESS range: [%d, %d]\n", ess_min, ess_max))
  cat(sprintf("Mean ESS: %.1f (%.1f%% efficiency)\n\n", ess_mean, ess_efficiency))
}

# Analyze all three latent states
analyze_latent_state(theta1_true, theta_1_estimate, theta_1_ci_lower, theta_1_ci_upper, theta_1_chain, "LEVEL STATE (theta_1)")
analyze_latent_state(theta2_true, theta_2_estimate, theta_2_ci_lower, theta_2_ci_upper, theta_2_chain, "TREND STATE (theta_2)")
analyze_latent_state(theta3_true, theta_3_estimate, theta_3_ci_lower, theta_3_ci_upper, theta_3_chain, "ACCELERATION STATE (theta_3)")

# --- 8. Enhanced Convergence Diagnostics ---
cat("=== ENHANCED CONVERGENCE DIAGNOSTICS ===\n\n")

# Calculate effective sample sizes OUTSIDE the functions (global scope)
effective_sample_sizes <- sapply(list(theta_01_chain, theta_02_chain, theta_03_chain, prec_1_chain, prec_2_chain, prec_3_chain, prec_y_chain),
                                 function(x) {
                                   acf_vals <- acf(x, plot=FALSE, lag.max=min(100, length(x)/4))$acf[-1]
                                   max(1, length(x) / (1 + 2 * sum(acf_vals[acf_vals > 0])))
                                 })
names(effective_sample_sizes) <- c("theta_01", "theta_02", "theta_03", "prec_1", "prec_2", "prec_3", "prec_y")

# 1. Basic Effective Sample Size - Improved Table
cat("1. Effective Sample Size Analysis:\n")
cat("==========================================================================\n")
cat("Parameter   |   Chain Size |        ESS |  Efficiency (%) |       Status\n")
cat("==========================================================================\n")

for(i in 1:length(effective_sample_sizes)) {
  efficiency <- 100 * effective_sample_sizes[i] / n_chain
  status <- ifelse(efficiency > 50, "EXCELLENT",
                   ifelse(efficiency > 25, "GOOD",
                          ifelse(efficiency > 10, "ACCEPTABLE", "POOR")))

  cat(sprintf("%-11s | %12d | %10.1f | %15.1f | %12s\n",
              names(effective_sample_sizes)[i],
              n_chain,
              effective_sample_sizes[i],
              efficiency,
              status))
}
cat("==========================================================================\n")
cat(sprintf("Overall Assessment: Min ESS = %.1f, Mean ESS = %.1f\n\n",
            min(effective_sample_sizes), mean(effective_sample_sizes)))

# Enhanced diagnostics if coda package is available
if (requireNamespace("coda", quietly = TRUE)) {

  # Convert to mcmc objects
  theta_01_mcmc <- coda::mcmc(theta_01_chain)
  theta_02_mcmc <- coda::mcmc(theta_02_chain)
  theta_03_mcmc <- coda::mcmc(theta_03_chain)
  prec_1_mcmc <- coda::mcmc(prec_1_chain)
  prec_2_mcmc <- coda::mcmc(prec_2_chain)
  prec_3_mcmc <- coda::mcmc(prec_3_chain)
  prec_y_mcmc <- coda::mcmc(prec_y_chain)

  # 2. Geweke Convergence Diagnostic - Improved Table
  cat("2. Geweke Convergence Diagnostic:\n")
  cat("=============================================================================\n")
  cat("Parameter   |    Z-Score |  |Z| < 2.0 |      Status |       Interpretation\n")
  cat("=============================================================================\n")

  geweke_results <- list(
    theta_01 = coda::geweke.diag(theta_01_mcmc),
    theta_02 = coda::geweke.diag(theta_02_mcmc),
    theta_03 = coda::geweke.diag(theta_03_mcmc),
    prec_1 = coda::geweke.diag(prec_1_mcmc),
    prec_2 = coda::geweke.diag(prec_2_mcmc),
    prec_3 = coda::geweke.diag(prec_3_mcmc),
    prec_y = coda::geweke.diag(prec_y_mcmc)
  )

  for(param in names(geweke_results)) {
    z_score <- geweke_results[[param]]$z
    abs_z <- abs(z_score)
    status <- ifelse(abs_z < 1.96, "PASS", "FAIL")
    interpretation <- ifelse(abs_z < 1.96, "Converged",
                             ifelse(abs_z < 2.58, "Marginal", "Not Converged"))

    cat(sprintf("%-11s | %10.4f | %10s | %11s | %17s\n",
                param, z_score,
                ifelse(abs_z < 2.0, "YES", "NO"),
                status, interpretation))
  }
  cat("=============================================================================\n")
  cat("Note: |Z-score| < 1.96 indicates convergence at 95% confidence level\n\n")

  # 3. Heidelberger-Welch Test - Improved Table
  cat("3. Heidelberger-Welch Stationarity and Halfwidth Tests:\n")
  cat("=====================================================================================\n")
  cat("Parameter   | Stationarity | Start Iter |  Halfwidth |     Mean |    Status Summary\n")
  cat("=====================================================================================\n")

  heidel_params <- list(theta_01_mcmc, theta_02_mcmc, theta_03_mcmc, prec_1_mcmc, prec_2_mcmc, prec_3_mcmc, prec_y_mcmc)
  heidel_names <- c("theta_01", "theta_02", "theta_03", "prec_1", "prec_2", "prec_3", "prec_y")

  for(i in 1:length(heidel_params)) {
    result <- coda::heidel.diag(heidel_params[[i]])
    stationarity_status <- ifelse(result[1,1], "PASS", "FAIL")
    halfwidth_status <- ifelse(result[1,3], "PASS", "FAIL")
    overall_status <- ifelse(result[1,1] && result[1,3], "CONVERGED", "CHECK NEEDED")

    cat(sprintf("%-11s | %12s | %10d | %10s | %8.4f | %15s\n",
                heidel_names[i],
                stationarity_status,
                result[1,2],
                halfwidth_status,
                result[1,4],
                overall_status))
  }
  cat("=====================================================================================\n")
  cat("Note: Both Stationarity and Halfwidth tests should PASS for reliable convergence\n\n")

  # 4. Raftery-Lewis Diagnostic - Organized Table with Error Handling
  cat("4. Raftery-Lewis Diagnostic (Burn-in and Sample Size Requirements):\n")
  cat("===============================================================================================================\n")
  cat("Parameter   | Quantile | Accuracy | Probability | Burn-in (M) | Total (N) | Lower (Nmin) |     Dependence\n")
  cat("===============================================================================================================\n")

  # Combine all parameters for Raftery-Lewis
  all_chains <- cbind("theta_01" = theta_01_mcmc,
                      "theta_02" = theta_02_mcmc,
                      "theta_03" = theta_03_mcmc,
                      "prec_1" = prec_1_mcmc,
                      "prec_2" = prec_2_mcmc,
                      "prec_3" = prec_3_mcmc,
                      "prec_y" = prec_y_mcmc)

  # Error handling for Raftery-Lewis diagnostic
  raftery_success <- FALSE
  raftery_result <- NULL

  # Try different accuracy levels if the original fails
  accuracy_levels <- c(0.005, 0.01, 0.02, 0.05)
  quantile_levels <- c(0.025, 0.05, 0.1)

  for(q_level in quantile_levels) {
    for(acc_level in accuracy_levels) {
      tryCatch({
        raftery_result <- coda::raftery.diag(all_chains, q=q_level, r=acc_level, s=0.95)
        raftery_success <- TRUE
        cat(sprintf("Note: Using quantile=%.3f, accuracy=%.3f for analysis\n", q_level, acc_level))
        break
      }, error = function(e) {
        # Continue to next accuracy level
      })
    }
    if(raftery_success) break
  }

  if(raftery_success && !is.null(raftery_result)) {
    for(i in 1:nrow(raftery_result$resmatrix)) {
      param_name <- rownames(raftery_result$resmatrix)[i]
      values <- raftery_result$resmatrix[i, ]
      dependence_factor <- values["I"]
      dependency_status <- ifelse(dependence_factor < 5, "Low",
                                  ifelse(dependence_factor < 10, "Moderate", "High"))

      # Check if current chain size meets requirements
      chain_adequate <- n_chain >= values["N"]
      status_marker <- ifelse(chain_adequate, "", " *")

      cat(sprintf("%-11s |    %.1f%% |     %.1f%% |         95%% | %11d | %9d | %12d | %8.2f (%s)%s\n",
                  param_name,
                  q_level*100, acc_level*100,
                  values["M"], values["N"], values["Nmin"],
                  dependence_factor, dependency_status, status_marker))
    }

    # Check if any parameter needs more samples
    max_required <- max(raftery_result$resmatrix[, "N"])
    if(n_chain < max_required) {
      cat("===============================================================================================================\n")
      cat(sprintf("WARNING: Current chain size (%d) is smaller than recommended (%d)\n",
                  n_chain, max_required))
      cat("* Marked parameters may need longer chains for reliable estimates\n")
      cat(sprintf("Recommendation: Increase chain size to at least %d samples\n", max_required))
    }

  } else {
    # If all attempts fail, provide alternative analysis
    cat("Unable to compute Raftery-Lewis diagnostic with current chain size.\n")
    cat("This typically indicates that the chain size is insufficient for the desired precision.\n")
    cat("==============================================================================================================\n")

    # Provide alternative chain size recommendations
    cat("\nAlternative Chain Size Assessment:\n")
    cat("=====================================\n")

    # Simple rule-of-thumb recommendations
    min_recommended <- 1000
    good_size <- 5000
    excellent_size <- 10000

    current_status <- ifelse(n_chain >= excellent_size, "EXCELLENT",
                             ifelse(n_chain >= good_size, "GOOD",
                                    ifelse(n_chain >= min_recommended, "ADEQUATE", "INSUFFICIENT")))

    cat(sprintf("Current chain size: %d (%s)\n", n_chain, current_status))
    cat(sprintf("Minimum recommended: %d\n", min_recommended))
    cat(sprintf("Good size: %d\n", good_size))
    cat(sprintf("Excellent size: %d\n", excellent_size))

    if(n_chain < min_recommended) {
      cat(sprintf("\nRECOMMENDATION: Increase chain size to at least %d\n", min_recommended))
    }
  }

  cat("==============================================================================================================\n")
  cat("Note: Dependence factor (I) < 5 indicates good mixing; I > 5 suggests high autocorrelation\n\n")

  # 5. Summary Assessment Table with Raftery-Lewis handling
  cat("5. Overall Convergence Assessment:\n")
  cat("=========================================================================================\n")
  cat("Parameter   |     ESS |   Efficiency | Geweke | Heidelberg | Raftery-Lewis |     Overall\n")
  cat("=========================================================================================\n")

  for(i in 1:length(heidel_names)) {
    param <- heidel_names[i]
    ess_val <- effective_sample_sizes[param]
    efficiency <- 100 * ess_val / n_chain

    # Get individual test results
    geweke_pass <- abs(geweke_results[[param]]$z) < 1.96
    heidel_result <- coda::heidel.diag(heidel_params[[i]])
    heidel_pass <- heidel_result[1,1] && heidel_result[1,3]

    # Raftery-Lewis assessment
    if(raftery_success && !is.null(raftery_result)) {
      dependence_factor <- raftery_result$resmatrix[i, "I"]
      raftery_pass <- dependence_factor < 5 && n_chain >= raftery_result$resmatrix[i, "N"]
      raftery_status <- ifelse(raftery_pass, "PASS", "FAIL")
    } else {
      raftery_pass <- FALSE
      raftery_status <- "N/A"
    }

    # Overall assessment (only count available tests)
    available_tests <- c(geweke_pass, heidel_pass)
    if(raftery_success) available_tests <- c(available_tests, raftery_pass)

    tests_passed <- sum(available_tests)
    total_tests <- length(available_tests)

    overall_status <- ifelse(tests_passed == total_tests, "EXCELLENT",
                             ifelse(tests_passed >= total_tests * 0.75, "GOOD",
                                    ifelse(tests_passed >= total_tests * 0.5, "ACCEPTABLE", "POOR")))

    cat(sprintf("%-11s | %7.1f | %11.1f%% | %6s | %10s | %13s | %11s\n",
                param, ess_val, efficiency,
                ifelse(geweke_pass, "PASS", "FAIL"),
                ifelse(heidel_pass, "PASS", "FAIL"),
                raftery_status,
                overall_status))
  }
  cat("=========================================================================================\n")

  # Additional guidance for chain size optimization
  cat("\n=== CHAIN SIZE OPTIMIZATION GUIDANCE ===\n")
  cat("Current Configuration:\n")
  cat(sprintf("- Burn-in: %d\n", burnin))
  cat(sprintf("- Thinning: %d\n", thinning))
  cat(sprintf("- Chain size: %d\n", n_chain))
  cat(sprintf("- Total iterations: %d\n", n_iter))

  if(raftery_success && !is.null(raftery_result)) {
    max_burnin <- max(raftery_result$resmatrix[, "M"])
    max_total <- max(raftery_result$resmatrix[, "N"])

    cat("\nRaftery-Lewis Recommendations:\n")
    cat(sprintf("- Recommended burn-in: %d\n", max_burnin))
    cat(sprintf("- Recommended total samples: %d\n", max_total))

    if(n_chain < max_total) {
      suggested_iterations <- max_burnin + (max_total - 1) * thinning + 1
      cat(sprintf("- Suggested total iterations: %d\n", suggested_iterations))
      cat(sprintf("- Increase factor: %.1fx current size\n", max_total / n_chain))
    }
  }

  cat("\nGeneral Recommendations:\n")
  cat("- ESS > 400 for reliable posterior estimates\n")
  cat("- ESS > 100 for basic convergence assessment\n")
  cat("- Efficiency > 10% indicates reasonable mixing\n")
  cat("- Consider increasing thinning if autocorrelation is high\n\n")
} else {
  cat("Note: Install 'coda' package for enhanced convergence diagnostics.\n\n")
}

# --- 9. Organized Visualization by Parameter Categories ---
cat("=== GENERATING ORGANIZED VISUALIZATIONS ===\n\n")

# 9.1 LOCAL LEVEL PARAMETERS
cat("Generating plots: Local Level Parameters (theta_01, prec_1)...\n")
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

# theta_01: posterior + trace
hist(theta_01_chain, main = expression(paste("Posterior: ", theta["0,1"])),
     xlab = expression(theta["0,1"]), col = "lightblue", border = "white", probability = TRUE)
abline(v = theta01_true, col = "red", lwd = 2, lty = 2)
abline(v = median(theta_01_chain), col = "blue", lwd = 2)
legend("topright", legend = c("True", "Median"), col = c("red", "blue"), lty = c(2, 1), lwd = 2, bty = "n")

plot(theta_01_chain, type = 'l', main = expression(paste("Trace: ", theta["0,1"])),
     col = "blue", xlab = "Iteration", ylab = expression(theta["0,1"]))
abline(h = theta01_true, col = "red", lty = 2, lwd = 2)

# prec_1: posterior + trace
hist(prec_1_chain, main = expression(paste("Posterior: ", 1/W[1])),
     xlab = expression(1/W[1]), col = "lightcoral", border = "white", probability = TRUE)
abline(v = prec1_true, col = "red", lwd = 2, lty = 2)
abline(v = median(prec_1_chain), col = "blue", lwd = 2)
legend("topright", legend = c("True", "Median"), col = c("red", "blue"), lty = c(2, 1), lwd = 2, bty = "n")

plot(prec_1_chain, type = 'l', main = expression(paste("Trace: ", 1/W[1])),
     col = "darkred", xlab = "Iteration", ylab = expression(1/W[1]))
abline(h = prec1_true, col = "red", lty = 2, lwd = 2)

# 9.2 LOCAL TREND PARAMETERS
cat("Generating plots: Local Trend Parameters (theta_02, prec_2)...\n")
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

# theta_02: posterior + trace
hist(theta_02_chain, main = expression(paste("Posterior: ", theta["0,2"])),
     xlab = expression(theta["0,2"]), col = "lightgreen", border = "white", probability = TRUE)
abline(v = theta02_true, col = "red", lwd = 2, lty = 2)
abline(v = median(theta_02_chain), col = "blue", lwd = 2)
legend("topright", legend = c("True", "Median"), col = c("red", "blue"), lty = c(2, 1), lwd = 2, bty = "n")

plot(theta_02_chain, type = 'l', main = expression(paste("Trace: ", theta["0,2"])),
     col = "darkgreen", xlab = "Iteration", ylab = expression(theta["0,2"]))
abline(h = theta02_true, col = "red", lty = 2, lwd = 2)

# prec_2: posterior + trace
hist(prec_2_chain, main = expression(paste("Posterior: ", 1/W[2])),
     xlab = expression(1/W[2]), col = "lightyellow", border = "white", probability = TRUE)
abline(v = prec2_true, col = "red", lwd = 2, lty = 2)
abline(v = median(prec_2_chain), col = "blue", lwd = 2)
legend("topright", legend = c("True", "Median"), col = c("red", "blue"), lty = c(2, 1), lwd = 2, bty = "n")

plot(prec_2_chain, type = 'l', main = expression(paste("Trace: ", 1/W[2])),
     col = "darkgoldenrod", xlab = "Iteration", ylab = expression(1/W[2]))
abline(h = prec2_true, col = "red", lty = 2, lwd = 2)

# 9.3 LOCAL ACCELERATION PARAMETERS
cat("Generating plots: Local Acceleration Parameters (theta_03, prec_3)...\n")
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

# theta_03: posterior + trace
hist(theta_03_chain, main = expression(paste("Posterior: ", theta["0,3"])),
     xlab = expression(theta["0,3"]), col = "lightcyan", border = "white", probability = TRUE)
abline(v = theta03_true, col = "red", lwd = 2, lty = 2)
abline(v = median(theta_03_chain), col = "blue", lwd = 2)
legend("topright", legend = c("True", "Median"), col = c("red", "blue"), lty = c(2, 1), lwd = 2, bty = "n")

plot(theta_03_chain, type = 'l', main = expression(paste("Trace: ", theta["0,3"])),
     col = "darkcyan", xlab = "Iteration", ylab = expression(theta["0,3"]))
abline(h = theta03_true, col = "red", lty = 2, lwd = 2)

# prec_3: posterior + trace
hist(prec_3_chain, main = expression(paste("Posterior: ", 1/W[3])),
     xlab = expression(1/W[3]), col = "lightpink", border = "white", probability = TRUE)
abline(v = prec3_true, col = "red", lwd = 2, lty = 2)
abline(v = median(prec_3_chain), col = "blue", lwd = 2)
legend("topright", legend = c("True", "Median"), col = c("red", "blue"), lty = c(2, 1), lwd = 2, bty = "n")

plot(prec_3_chain, type = 'l', main = expression(paste("Trace: ", 1/W[3])),
     col = "deeppink", xlab = "Iteration", ylab = expression(1/W[3]))
abline(h = prec3_true, col = "red", lty = 2, lwd = 2)

# 9.4 DATA PRECISION
cat("Generating plots: Data Precision (prec_y)...\n")
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

hist(prec_y_chain, main = expression(paste("Posterior: ", 1/V)),
     xlab = expression(1/V), col = "lavender", border = "white", probability = TRUE)
abline(v = prec_y_true, col = "red", lwd = 2, lty = 2)
abline(v = median(prec_y_chain), col = "blue", lwd = 2)
legend("topright", legend = c("True", "Median"), col = c("red", "blue"), lty = c(2, 1), lwd = 2, bty = "n")

plot(prec_y_chain, type = 'l', main = expression(paste("Trace: ", 1/V)),
     col = "mediumpurple", xlab = "Iteration", ylab = expression(1/V))
abline(h = prec_y_true, col = "red", lty = 2, lwd = 2)

# 9.5 STATE ESTIMATES
cat("Generating plots: State Estimates (theta_1, theta_2, theta_3)...\n")
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

n_plot <- min(200, n)
time_idx <- 1:n_plot

# Level state
plot(time_idx, theta1_true[time_idx], type = 'l', col = "red", lty = 2, lwd = 2,
     main = "Level State: True vs. Estimated", xlab = "Time", ylab = expression(theta["t,1"]),
     ylim = range(c(theta1_true[time_idx], theta_1_ci_lower[time_idx], theta_1_ci_upper[time_idx])))
lines(time_idx, theta_1_estimate[time_idx], col = "blue", lwd = 1.5)
polygon(c(time_idx, rev(time_idx)),
        c(theta_1_ci_lower[time_idx], rev(theta_1_ci_upper[time_idx])),
        col = rgb(0, 0, 1, 0.2), border = NA)
legend("topleft", legend = c("True", "Est.", "95% CI"),
       col = c("red", "blue", rgb(0,0,1,0.2)), lty = c(2,1,1), lwd = c(2,1.5,8), bty = "n")

# Trend state
plot(time_idx, theta2_true[time_idx], type = 'l', col = "red", lty = 2, lwd = 2,
     main = "Trend State: True vs. Estimated", xlab = "Time", ylab = expression(theta["t,2"]),
     ylim = range(c(theta2_true[time_idx], theta_2_ci_lower[time_idx], theta_2_ci_upper[time_idx])))
lines(time_idx, theta_2_estimate[time_idx], col = "darkgreen", lwd = 1.5)
polygon(c(time_idx, rev(time_idx)),
        c(theta_2_ci_lower[time_idx], rev(theta_2_ci_upper[time_idx])),
        col = rgb(0, 1, 0, 0.2), border = NA)
legend("topleft", legend = c("True", "Est.", "95% CI"),
       col = c("red", "darkgreen", rgb(0,1,0,0.2)), lty = c(2,1,1), lwd = c(2,1.5,8), bty = "n")

# Acceleration state
plot(time_idx, theta3_true[time_idx], type = 'l', col = "red", lty = 2, lwd = 2,
     main = "Acceleration State: True vs. Estimated", xlab = "Time", ylab = expression(theta["t,3"]),
     ylim = range(c(theta3_true[time_idx], theta_3_ci_lower[time_idx], theta_3_ci_upper[time_idx])))
lines(time_idx, theta_3_estimate[time_idx], col = "darkcyan", lwd = 1.5)
polygon(c(time_idx, rev(time_idx)),
        c(theta_3_ci_lower[time_idx], rev(theta_3_ci_upper[time_idx])),
        col = rgb(0, 1, 1, 0.2), border = NA)
legend("topleft", legend = c("True", "Est.", "95% CI"),
       col = c("red", "darkcyan", rgb(0,1,1,0.2)), lty = c(2,1,1), lwd = c(2,1.5,8), bty = "n")

# Observed vs Level
plot(time_idx, y[time_idx], type = 'p', col = "gray50", pch = 16, cex = 0.5,
     main = "Observed Data vs. Level State", xlab = "Time", ylab = "Value",
     ylim = range(c(y[time_idx], theta1_true[time_idx], theta_1_estimate[time_idx])))
lines(time_idx, theta1_true[time_idx], col = "red", lty = 2, lwd = 2)
lines(time_idx, theta_1_estimate[time_idx], col = "blue", lwd = 1.5)
legend("topleft", legend = c("Observed", "True Level", "Est. Level"),
       col = c("gray50", "red", "blue"), pch = c(16, NA, NA),
       lty = c(NA, 2, 1), lwd = c(NA, 2, 1.5), bty = "n")

# Reset plotting parameters
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1)

cat("=== VISUALIZATION COMPLETED ===\n\n")
cat("Summary: All visualizations have been organized into 5 logical categories:\n")
cat("1. Local Level Parameters (theta_01, prec_1)\n")
cat("2. Local Trend Parameters (theta_02, prec_2)\n")
cat("3. Local Acceleration Parameters (theta_03, prec_3)\n")
cat("4. Data Precision (prec_y)\n")
cat("5. State Estimates (theta_1, theta_2, theta_3, and observed data)\n\n")

cat("=== VALIDATION SCRIPT COMPLETED SUCCESSFULLY ===\n")
