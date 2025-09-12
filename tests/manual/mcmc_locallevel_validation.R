#-------------------------------------------------------------------------------
# Test Script for Local Level Model (Validation with Helpers)
#
# Objective:
# - Validate and replicate the Gibbs sampler loop in R for the Gaussian
#   local level model using the C helpers:
#     * _pdm_test_generate_theta_1_locallevel      (state update)
#     * _pdm_test_generate_precision_theta_p       (precision update for W_1)
#     * _pdm_test_generate_theta_01_locallevel     (initial state update)
#     * _pdm_test_generate_precision_data          (precision update for V)
#
# - Standardize outputs using helper functions from tests/manual/helpers:
#     * summary_tables.R
#     * convergence_diagnostics.R
#     * visualization_utils.R  (atualizado: expressões, painéis por nível, sem gráficos individuais)
#     * state_analysis.R
#     * observed_vs_state_plot.R
#
# Notes:
# - This script assumes the package native symbols above are registered.
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
    library(pdm)
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
theta0_true <- 0.1   # Initial level (theta_{0,1})
prec1_true  <- 10    # Innovation precision 1/W_1
prec_y_true <- 1     # Data precision 1/V

# set.seed(123) # For reproducibility

# Generate noise
u1 <- rnorm(n, sd = sqrt(1 / prec1_true))
e  <- rnorm(n, sd = sqrt(1 / prec_y_true))

# Simulate latent state and observations
theta1_true <- cumsum(c(theta0_true, u1))[-1]
y <- theta1_true + e

#-------------------------------------------------------------------------------
# 2) MCMC Configuration
#-------------------------------------------------------------------------------
burnin   <- 1000
thinning <- 1
n_chain  <- 1000
# Total iterations: same formula as in C implementations
n_iter   <- burnin + (n_chain - 1) * thinning + 1

# Priors (matching naming from C implementations)
mean_theta01 <- 0       # prior_theta01_mean
prec_theta01 <- 0.01    # prior_theta01_prec
nu_01        <- 1e-1    # prior_prec1_shape
eta_01       <- 1e-1    # prior_prec1_rate
nu_y         <- 1e-1    # prior_prec_y_shape
eta_y        <- 1e-1    # prior_prec_y_rate

#-------------------------------------------------------------------------------
# 3) Chain Storage
#-------------------------------------------------------------------------------
theta_1_chain  <- matrix(NA_real_, nrow = n_chain, ncol = n)
theta_01_chain <- numeric(n_chain)
prec_1_chain   <- numeric(n_chain)
prec_y_chain   <- numeric(n_chain)

# Full history arrays (iteration-wise, including burn-in)
theta_01_post <- numeric(n_iter)
prec_1_post   <- numeric(n_iter)
prec_y_post   <- numeric(n_iter)
theta_1_post  <- matrix(NA_real_, nrow = n_iter, ncol = n)

#-------------------------------------------------------------------------------
# 4) Initialization (iter = 1) following reference organization
#-------------------------------------------------------------------------------
# set.seed(456)
theta_01_post[1] <- rnorm(1, mean_theta01, sqrt(1.0 / prec_theta01))
prec_1_post[1]   <- rgamma(1, nu_01, rate = eta_01)
prec_y_post[1]   <- rgamma(1, nu_y,  rate = eta_y)
init_sd          <- sqrt(1.0 / prec_1_post[1])

# Initialize theta_1 as random walk from theta_01
theta_1_post[1, 1] <- rnorm(1, theta_01_post[1], init_sd)
for (j in 2:n) {
  theta_1_post[1, j] <- rnorm(1, theta_1_post[1, j-1], init_sd)
}

#-------------------------------------------------------------------------------
# 5) MCMC Loop (C helpers)
#-------------------------------------------------------------------------------
cat("Starting MCMC loop for Local Level model...\n")

# Progress bar
pb <- txtProgressBar(min = 2, max = n_iter, style = 3, width = 60, char = "\u27a4")
cat("Progress: ", rep(" ", 60), "\n")

chain_idx <- 0  # Counter for saved samples

for (ii in 2:n_iter) {

  # 1) State vector theta_1
  theta_1_new <- .Call(
    "_pdm_test_generate_theta_1_locallevel",
    as.numeric(y),
    as.numeric(prec_y_post[ii-1]),
    as.numeric(prec_1_post[ii-1]),
    as.numeric(theta_01_post[ii-1])
  )
  if (length(theta_1_new) == n) {
    theta_1_post[ii, ] <- as.numeric(theta_1_new)
  } else {
    warning(paste("Iteration", ii, ": theta_1 was not updated correctly"))
    theta_1_post[ii, ] <- theta_1_post[ii-1, ]
  }

  # 2) Innovation precision 1/W_1
  prec_1_post[ii] <- .Call(
    "_pdm_test_generate_precision_theta_p",
    as.numeric(theta_01_post[ii-1]),
    as.numeric(theta_1_post[ii, ]),
    as.numeric(nu_01),
    as.numeric(eta_01)
  )

  # 3) Initial state theta_01
  theta_01_post[ii] <- .Call(
    "_pdm_test_generate_theta_01_locallevel",
    as.numeric(theta_1_post[ii, ]),
    as.numeric(prec_1_post[ii]),
    as.numeric(mean_theta01),
    as.numeric(prec_theta01)
  )

  # 4) Data precision 1/V
  prec_y_post[ii] <- .Call(
    "_pdm_test_generate_precision_data",
    as.numeric(y),
    as.numeric(theta_1_post[ii, ]),
    as.numeric(nu_y),
    as.numeric(eta_y)
  )

  # Store samples after burn-in and thinning
  if (ii >= (burnin + 1) && ((ii - burnin - 1) %% thinning == 0)) {
    chain_idx <- chain_idx + 1
    theta_1_chain[chain_idx, ] <- theta_1_post[ii, ]
    theta_01_chain[chain_idx]  <- theta_01_post[ii]
    prec_1_chain[chain_idx]    <- prec_1_post[ii]
    prec_y_chain[chain_idx]    <- prec_y_post[ii]
  }

  # Progress bar updates
  if (ii %% 100 == 0) setTxtProgressBar(pb, ii)
}

# Close progress bar
close(pb)
cat("\n... MCMC loop completed.\n\n")

#-------------------------------------------------------------------------------
# 6) Posterior Summaries (parameters)
#-------------------------------------------------------------------------------
param_chains <- list(
  theta_01 = theta_01_chain,
  prec_1   = prec_1_chain,
  prec_y   = prec_y_chain
)
true_values <- c(
  theta_01 = theta0_true,
  prec_1   = prec1_true,
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

states_for_analysis <- list(
  theta_1 = list(
    state_name = "theta_1",
    true       = theta1_true,
    estimate   = theta_1_estimate,
    ci_lower   = theta_1_ci_lower,
    ci_upper   = theta_1_ci_upper,
    chain      = theta_1_chain
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
# 8) Convergence Diagnostics (parameters) via helpers
#-------------------------------------------------------------------------------
print_ess_table(param_chains, n_chain)
run_coda_diagnostics(param_chains)  # requires 'coda', prints informative note if missing

#-------------------------------------------------------------------------------
# 9) Organized Visualizations (only per-level panels + improved state labels)
#-------------------------------------------------------------------------------
states_for_plot <- list(
  theta_1 = list(true = theta1_true, estimate = theta_1_estimate,
                 ci_lower = theta_1_ci_lower, ci_upper = theta_1_ci_upper)
)
generate_organized_plots(
  param_chains             = param_chains,
  true_values              = true_values,
  states                   = states_for_plot,
  use_expressions          = TRUE,
  per_level_panels         = TRUE,
  show_individual_param_plots = FALSE,  # não exibir gráficos individuais
  max_lag_acf              = 60,
  use_state_expressions    = TRUE       # melhora nomes dos gráficos de estado
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

cat("=== LOCAL LEVEL VALIDATION COMPLETED ===\n")
