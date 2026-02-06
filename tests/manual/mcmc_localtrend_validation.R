#-------------------------------------------------------------------------------
# Test Script for Local Trend Model (Validation with Helpers)
#
# Objective:
# - Validate and replicate the Gibbs sampler loop in R for the Gaussian
#   local trend model using the C helpers:
#     * _bdm_test_generate_theta_p                 (state updates for theta_2)
#     * _bdm_test_generate_precision_theta_p       (precision updates for W_p)
#     * _bdm_test_generate_theta_0p                (initial state update for theta_0p)
#     * _bdm_test_generate_theta_1                 (state updates for theta_1)
#     * _bdm_test_generate_precision_theta_k       (precision updates with neighbors)
#     * _bdm_test_generate_theta_01                (initial level)
#     * _bdm_test_generate_precision_data          (data precision)
#
# - Standardize outputs using helper functions from tests/manual/helpers:
#     * summary_tables.R
#     * convergence_diagnostics.R
#     * visualization_utils.R  (atualizado: expressões, painéis por nível, sem gráficos individuais)
#     * state_analysis.R
#     * observed_vs_state_plot.R
#-------------------------------------------------------------------------------

# Clean environment (optional)
# rm(list = ls())

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
    library(bdm)
  }
})

# Load required packages for enhanced diagnostics
suppressPackageStartupMessages({
  if (!require(coda, quietly = TRUE)) {
    cat("Warning: 'coda' package not available. Some convergence diagnostics will be skipped.\n")
  }
})

# Load helper functions
source("tests/manual/helpers/summary_tables.R")
source("tests/manual/helpers/convergence_diagnostics.R")
source("tests/manual/helpers/visualization_utils.R")  # atualizado
source("tests/manual/helpers/state_analysis.R")
source("tests/manual/helpers/observed_vs_state_plot.R")

#-------------------------------------------------------------------------------
# 1) Data Simulation
#-------------------------------------------------------------------------------
n <- 1000  # Number of observations

# True parameters
theta01_true <- 0.1   # Initial level
theta02_true <- 0.05  # Initial trend
prec1_true   <- 10    # Level precision
prec2_true   <- 20    # Trend precision
prec_y_true  <- 1     # Data precision

# set.seed(123) # For reproducibility

# Generate noise
u1 <- rnorm(n, sd = sqrt(1 / prec1_true))
u2 <- rnorm(n, sd = sqrt(1 / prec2_true))
e  <- rnorm(n, sd = sqrt(1 / prec_y_true))

# Simulate latent states and observations
theta1_true <- numeric(n)
theta2_true <- numeric(n)
theta1_true[1] <- theta01_true + theta02_true + u1[1]
theta2_true[1] <- theta02_true + u2[1]
for (t in 2:n) {
  theta1_true[t] <- theta1_true[t-1] + theta2_true[t-1] + u1[t]
  theta2_true[t] <- theta2_true[t-1] + u2[t]
}
y <- theta1_true + e

#-------------------------------------------------------------------------------
# 2) MCMC Configuration
#-------------------------------------------------------------------------------
burnin   <- 1000
thinning <- 1
n_chain  <- 1000
n_iter   <- burnin + (n_chain - 1) * thinning + 1

# Priors (C naming)
mean_theta01 <- 0     # prior_theta01_mean
prec_theta01 <- 0.01  # prior_theta01_prec
mean_theta02 <- 0     # prior_theta02_mean
prec_theta02 <- 0.01  # prior_theta02_prec
nu_01        <- 1e-1  # prior_prec1_shape
eta_01       <- 1e-1  # prior_prec1_rate
nu_02        <- 1e-1  # prior_prec2_shape
eta_02       <- 1e-1  # prior_prec2_rate
nu_y         <- 1e-1  # prior_prec_y_shape
eta_y        <- 1e-1  # prior_prec_y_rate

#-------------------------------------------------------------------------------
# 3) Chain Storage
#-------------------------------------------------------------------------------
theta_1_chain  <- matrix(NA_real_, nrow = n_chain, ncol = n)
theta_2_chain  <- matrix(NA_real_, nrow = n_chain, ncol = n)
theta_01_chain <- numeric(n_chain)
theta_02_chain <- numeric(n_chain)
prec_theta1_chain   <- numeric(n_chain)
prec_theta2_chain   <- numeric(n_chain)
prec_y_chain   <- numeric(n_chain)

# Full history arrays
theta_01_post <- numeric(n_iter)
theta_02_post <- numeric(n_iter)
prec_theta1_post   <- numeric(n_iter)
prec_theta2_post   <- numeric(n_iter)
prec_y_post   <- numeric(n_iter)
theta_1_post  <- matrix(NA_real_, nrow = n_iter, ncol = n)
theta_2_post  <- matrix(NA_real_, nrow = n_iter, ncol = n)

#-------------------------------------------------------------------------------
# 4) Initialization (iter = 1)
#-------------------------------------------------------------------------------
# set.seed(456)
theta_01_post[1] <- rnorm(1, mean_theta01, sqrt(1.0 / prec_theta01))
theta_02_post[1] <- rnorm(1, mean_theta02, sqrt(1.0 / prec_theta02))
prec_theta1_post[1]   <- rgamma(1, nu_01, rate = eta_01)
prec_theta2_post[1]   <- rgamma(1, nu_02, rate = eta_02)
prec_y_post[1]   <- rgamma(1, nu_y,  rate = eta_y)

init_sd_1 <- sqrt(1.0 / prec_theta1_post[1])
init_sd_2 <- sqrt(1.0 / prec_theta2_post[1])
theta_1_post[1, 1] <- rnorm(1, theta_01_post[1] + theta_02_post[1], init_sd_1)
theta_2_post[1, 1] <- rnorm(1, theta_02_post[1], init_sd_2)
for (j in 2:n) {
  theta_1_post[1, j] <- rnorm(1, theta_1_post[1, j-1] + theta_2_post[1, j-1], init_sd_1)
  theta_2_post[1, j] <- rnorm(1, theta_2_post[1, j-1], init_sd_2)
}

#-------------------------------------------------------------------------------
# 5) MCMC Loop (C helpers)
#-------------------------------------------------------------------------------
cat("Starting MCMC loop for Local Trend model...\n")
pb <- txtProgressBar(min = 2, max = n_iter, style = 3, width = 60, char = "\u27a4")
cat("Progress: ", rep(" ", 60), "\n")
chain_idx <- 0

for (ii in 2:n_iter) {

  # 1) theta_2 (trend states)
  theta_2_new <- .Call(
    "_bdm_test_generate_theta_p",
    as.numeric(theta_1_post[ii-1, ]),
    as.numeric(prec_theta1_post[ii-1]),
    as.numeric(prec_theta2_post[ii-1]),
    as.numeric(theta_02_post[ii-1])
  )
  if (length(theta_2_new) == n) {
    theta_2_post[ii, ] <- as.numeric(theta_2_new)
  } else {
    warning(paste("Iteration", ii, ": theta_2 was not updated correctly"))
    theta_2_post[ii, ] <- theta_2_post[ii-1, ]
  }

  # 2) precision 1/W_2
  prec_theta2_post[ii] <- .Call(
    "_bdm_test_generate_precision_theta_p",
    as.numeric(theta_02_post[ii-1]),
    as.numeric(theta_2_post[ii, ]),
    as.numeric(nu_02),
    as.numeric(eta_02)
  )

  # 3) initial trend theta_02
  theta_02_post[ii] <- .Call(
    "_bdm_test_generate_theta_0p",
    as.numeric(theta_1_post[ii-1, ]),
    as.numeric(theta_2_post[ii, ]),
    as.numeric(theta_01_post[ii-1]),
    as.numeric(prec_theta1_post[ii-1]),
    as.numeric(prec_theta2_post[ii]),
    as.numeric(mean_theta02),
    as.numeric(prec_theta02)
  )

  # 4) theta_1 (level states)
  theta_1_new <- .Call(
    "_bdm_test_generate_theta_1",
    as.numeric(y),
    as.numeric(theta_2_post[ii, ]),
    as.numeric(prec_y_post[ii-1]),
    as.numeric(prec_theta1_post[ii-1]),
    as.numeric(theta_01_post[ii-1]),
    as.numeric(theta_02_post[ii])
  )
  if (length(theta_1_new) == n) {
    theta_1_post[ii, ] <- as.numeric(theta_1_new)
  } else {
    warning(paste("Iteration", ii, ": theta_1 was not updated correctly"))
    theta_1_post[ii, ] <- theta_1_post[ii-1, ]
  }

  # 5) precision 1/W_1
  prec_theta1_post[ii] <- .Call(
    "_bdm_test_generate_precision_theta_k",
    as.numeric(theta_01_post[ii-1]),
    as.numeric(theta_02_post[ii]),
    as.numeric(theta_1_post[ii, ]),
    as.numeric(theta_2_post[ii, ]),
    as.numeric(nu_01),
    as.numeric(eta_01)
  )

  # 6) initial level theta_01
  theta_01_post[ii] <- .Call(
    "_bdm_test_generate_theta_01",
    as.numeric(theta_1_post[ii, ]),
    as.numeric(theta_02_post[ii]),
    as.numeric(prec_theta1_post[ii]),
    as.numeric(mean_theta01),
    as.numeric(prec_theta01)
  )

  # 7) data precision 1/V
  prec_y_post[ii] <- .Call(
    "_bdm_test_generate_precision_data",
    as.numeric(y),
    as.numeric(theta_1_post[ii, ]),
    as.numeric(nu_y),
    as.numeric(eta_y)
  )

  # Store samples after burn-in and thinning
  if (ii >= (burnin + 1) && ((ii - burnin - 1) %% thinning == 0)) {
    chain_idx <- chain_idx + 1
    theta_1_chain[chain_idx, ] <- theta_1_post[ii, ]
    theta_2_chain[chain_idx, ] <- theta_2_post[ii, ]
    theta_01_chain[chain_idx]  <- theta_01_post[ii]
    theta_02_chain[chain_idx]  <- theta_02_post[ii]
    prec_theta1_chain[chain_idx]    <- prec_theta1_post[ii]
    prec_theta2_chain[chain_idx]    <- prec_theta2_post[ii]
    prec_y_chain[chain_idx]    <- prec_y_post[ii]
  }

  if (ii %% 100 == 0) setTxtProgressBar(pb, ii)
}
close(pb)
cat("\n... MCMC loop completed.\n\n")

#-------------------------------------------------------------------------------
# 6) Posterior Summaries (parameters)
#-------------------------------------------------------------------------------
param_chains <- list(
  theta_01 = theta_01_chain,
  theta_02 = theta_02_chain,
  prec_theta1   = prec_theta1_chain,
  prec_theta2   = prec_theta2_chain,
  prec_y   = prec_y_chain
)
true_values <- c(
  theta_01 = theta01_true,
  theta_02 = theta02_true,
  prec_theta1   = prec1_true,
  prec_theta2   = prec2_true,
  prec_y   = prec_y_true
)

print_posterior_estimates_table(param_chains, true_values, n_chain)
print_quantiles_table(param_chains)

#-------------------------------------------------------------------------------
# 7) Latent State Summaries and Analysis
#-------------------------------------------------------------------------------
theta_1_estimate <- apply(theta_1_chain, 2, median)
theta_1_ci_lower <- apply(theta_1_chain, 2, quantile, 0.025)
theta_1_ci_upper <- apply(theta_1_chain, 2, quantile, 0.975)

theta_2_estimate <- apply(theta_2_chain, 2, median)
theta_2_ci_lower <- apply(theta_2_chain, 2, quantile, 0.025)
theta_2_ci_upper <- apply(theta_2_chain, 2, quantile, 0.975)

states_for_analysis <- list(
  theta_1 = list(
    state_name = "theta_1 (level)",
    true       = theta1_true,
    estimate   = theta_1_estimate,
    ci_lower   = theta_1_ci_lower,
    ci_upper   = theta_1_ci_upper,
    chain      = theta_1_chain
  ),
  theta_2 = list(
    state_name = "theta_2 (trend)",
    true       = theta2_true,
    estimate   = theta_2_estimate,
    ci_lower   = theta_2_ci_lower,
    ci_upper   = theta_2_ci_upper,
    chain      = theta_2_chain
  )
)
analyze_multiple_states(
  states       = states_for_analysis,
  n_segments   = 5,
  max_lag      = 10,
  n_chain      = n_chain,
  n_timepoints = 10
)

#-------------------------------------------------------------------------------
# 8) Convergence Diagnostics (parameters)
#-------------------------------------------------------------------------------
print_ess_table(param_chains, n_chain)
run_coda_diagnostics(param_chains)

#-------------------------------------------------------------------------------
# 9) Organized Visualizations (only per-level panels + improved state labels)
#-------------------------------------------------------------------------------
states_for_plot <- list(
  theta_1 = list(true = theta1_true, estimate = theta_1_estimate,
                 ci_lower = theta_1_ci_lower, ci_upper = theta_1_ci_upper),
  theta_2 = list(true = theta2_true, estimate = theta_2_estimate,
                 ci_lower = theta_2_ci_lower, ci_upper = theta_2_ci_upper)
)
generate_organized_plots(
  param_chains                 = param_chains,
  true_values                  = true_values,
  states                       = states_for_plot,
  use_expressions              = TRUE,
  per_level_panels             = TRUE,
  show_individual_param_plots  = FALSE,
  max_lag_acf                  = 60,
  use_state_expressions        = TRUE
)

# Observed vs level state (zoom em 200 primeiros pontos)
par(mfrow = c(1, 1), mar = c(4, 4, 3, 1))
plot_observed_vs_state(
  y               = y,
  true_state      = theta1_true,
  estimated_state = theta_1_estimate,
  title = "Observed Data vs. Level State",
  xlab  = "Time",
  ylab  = "Value",
  n_plot = 200
)

cat("=== LOCAL TREND VALIDATION COMPLETED ===\n")
