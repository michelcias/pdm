#-------------------------------------------------------------------------------
# Test Script for Probit Bernoulli Local Level Model (Validation with Helpers)
#
# Objective:
# - Validate and replicate the Gibbs sampler loop in R for the
#   Bernoulli local level model with probit link using the C helpers:
#     * _pdm_test_generate_alpha_probit_bernoulli_locallevel (state update)
#     * _pdm_test_generate_precision_theta_p               (precision update)
#     * _pdm_test_generate_theta_01_locallevel             (initial state update)
#
# - Standardize outputs using helper functions from tests/manual/helpers:
#     * summary_tables.R
#     * convergence_diagnostics.R
#     * visualization_utils.R
#     * state_analysis.R
#     * observed_vs_state_plot.R
#     * adaptation_diagnostics.R (included for completeness; Gibbs needs no tuning)
#
# Notes:
# - This script assumes the package native symbols above are registered.
# - The probit sampler uses Albert-Chib augmentation and therefore has
#   acceptance probability 1 (no proposal tuning needed).
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
source("tests/manual/helpers/visualization_utils.R")
source("tests/manual/helpers/state_analysis.R")
source("tests/manual/helpers/observed_vs_state_plot.R")
source("tests/manual/helpers/adaptation_diagnostics.R")

#-------------------------------------------------------------------------------
# 1) Data Simulation
#-------------------------------------------------------------------------------
n <- 1000          # Number of observations

# True parameters (probit scale)
theta01_true <- 0.1   # Initial level theta_{0,1}
prec1_true   <- 1     # Innovation precision 1/W_1

# set.seed(123) # For reproducibility

# Generate latent state (random walk on probit scale)
u1 <- rnorm(n, sd = sqrt(1 / prec1_true))
theta1_true <- numeric(n)
theta1_true[1] <- theta01_true + u1[1]
for (t in 2:n) {
  theta1_true[t] <- theta1_true[t-1] + u1[t]
}

# Observations
alpha_true <- pnorm(theta1_true)
y <- rbinom(n, size = 1, prob = alpha_true)

#-------------------------------------------------------------------------------
# 2) MCMC Configuration
#-------------------------------------------------------------------------------
burnin   <- 10000
thinning <- 1
n_chain  <- 100000
# Total iterations: same formula as in C implementations
n_iter   <- burnin + (n_chain - 1) * thinning + 1

# Priors (matching naming from C implementations)
mean_theta01 <- 0       # prior_theta01_mean
prec_theta01 <- 0.01    # prior_theta01_prec
nu_01        <- 1e3     # prior_prec1_shape
eta_01       <- 1e3     # prior_prec1_rate

#-------------------------------------------------------------------------------
# 3) Chain Storage
#-------------------------------------------------------------------------------
theta_1_chain  <- matrix(NA_real_, nrow = n_chain, ncol = n)
theta_01_chain <- numeric(n_chain)
prec_1_chain   <- numeric(n_chain)
alpha_chain    <- matrix(NA_real_, nrow = n_chain, ncol = n)

# Full history arrays (iteration-wise, including burn-in)
theta_01_post <- numeric(n_iter)
prec_1_post   <- numeric(n_iter)
theta_1_post  <- matrix(NA_real_, nrow = n_iter, ncol = n)
alpha_post    <- matrix(NA_real_, nrow = n_iter, ncol = n)

# Placeholders for adaptation diagnostics (always 1 acceptance for Gibbs)
theta_1_updated <- matrix(1, nrow = n_iter, ncol = n)    # acceptance indicators
log_sigma_hist  <- matrix(NA_real_, nrow = n_iter, ncol = n)
accept_prop_hist <- matrix(1, nrow = n_iter, ncol = n)
log_sigma       <- rep(log(0.1), n)  # retained for diagnostic interface compatibility

#-------------------------------------------------------------------------------
# 4) Initialization (iter = 1) following reference organization
#-------------------------------------------------------------------------------
# set.seed(456)

theta_01_post[1] <- rnorm(1, mean_theta01, sqrt(1.0 / prec_theta01))
prec_1_post[1]   <- rgamma(1, nu_01, rate = eta_01)  # R uses 'rate'
init_sd          <- sqrt(1.0 / prec_1_post[1])

# Initialize theta_1 as random walk from theta_01
theta_1_post[1, 1] <- rnorm(1, theta_01_post[1], init_sd)
for (j in 2:n) {
  theta_1_post[1, j] <- rnorm(1, theta_1_post[1, j-1], init_sd)
}
alpha_post[1, ] <- pnorm(theta_1_post[1, ])

#-------------------------------------------------------------------------------
# 5) MCMC Loop (Gibbs state update)
#-------------------------------------------------------------------------------
cat("Starting MCMC loop for Probit Bernoulli Local Level model...\n")

# Progress bar
pb <- txtProgressBar(min = 2, max = n_iter, style = 3, width = 60, char = "\u27a4")
cat("Progress: ", rep(" ", 60), "\n")

chain_idx <- 0  # Counter for saved samples

for (ii in 2:n_iter) {

  # 1) Gibbs state update (probit link) + alpha
  #    _pdm_test_generate_alpha_probit_bernoulli_locallevel(theta_1_in, theta_01_in, prec_1_in, y)
  upd <- .Call(
    "_pdm_test_generate_alpha_probit_bernoulli_locallevel",
    as.numeric(theta_1_post[ii-1, ]),
    as.numeric(theta_01_post[ii-1]),
    as.numeric(prec_1_post[ii-1]),
    as.numeric(y)
  )
  theta_1_post[ii, ] <- as.numeric(upd$theta_1)
  alpha_post[ii, ]   <- as.numeric(upd$alpha)

  # 2) Innovation precision 1/W_1
  #    _pdm_test_generate_precision_theta_p(theta_0p_, theta_p_, nu_0p_, eta_0p_)
  prec_1_post[ii] <- .Call(
    "_pdm_test_generate_precision_theta_p",
    as.numeric(theta_01_post[ii-1]),    # theta_0p (previous iteration)
    as.numeric(theta_1_post[ii, ]),     # theta_p (current)
    as.numeric(nu_01),
    as.numeric(eta_01)
  )

  # 3) Initial state theta_01
  #    _pdm_test_generate_theta_01_locallevel(theta_1_, prec_theta_1_, mean_theta_01_, prec_theta_01_)
  theta_01_post[ii] <- .Call(
    "_pdm_test_generate_theta_01_locallevel",
    as.numeric(theta_1_post[ii, ]),     # current
    as.numeric(prec_1_post[ii]),        # current
    as.numeric(mean_theta01),
    as.numeric(prec_theta01)
  )

  # Store samples after burn-in and thinning (exactly as in reference logic)
  if (ii >= (burnin + 1) && ((ii - burnin - 1) %% thinning == 0)) {
    chain_idx <- chain_idx + 1
    theta_1_chain[chain_idx, ] <- theta_1_post[ii, ]
    theta_01_chain[chain_idx]  <- theta_01_post[ii]
    prec_1_chain[chain_idx]    <- prec_1_post[ii]
    alpha_chain[chain_idx, ]   <- alpha_post[ii, ]
  }

  # Progress bar update (coarser to reduce overhead)
  if (ii %% 1000 == 0) {
    setTxtProgressBar(pb, ii)
  }
}

# Close progress bar
close(pb)
cat("\n... MCMC loop completed.\n\n")

#-------------------------------------------------------------------------------
# 6) Posterior Summaries (parameters)
#-------------------------------------------------------------------------------
# Prepare inputs to helper tables
param_chains <- list(
  theta_01 = theta_01_chain,
  prec_1   = prec_1_chain
)
true_values <- c(
  theta_01 = theta01_true,
  prec_1   = prec1_true
)

print_posterior_estimates_table(param_chains, true_values, n_chain)
print_quantiles_table(param_chains)

#-------------------------------------------------------------------------------
# 7) Latent State (theta_1 on probit) and Probability (alpha) Summaries
#-------------------------------------------------------------------------------
# Compute median and 95% CI per time
theta_1_estimate <- apply(theta_1_chain, 2, median)
theta_1_ci_lower <- apply(theta_1_chain, 2, quantile, 0.025)
theta_1_ci_upper <- apply(theta_1_chain, 2, quantile, 0.975)

alpha_estimate <- pnorm(theta_1_estimate)
alpha_ci_lower <- pnorm(theta_1_ci_lower)
alpha_ci_upper <- pnorm(theta_1_ci_upper)

# State analysis: generalized helpers (N states)
states_for_analysis <- list(
  theta_1 = list(
    state_name = "theta_1 (probit)",
    true       = theta1_true,
    estimate   = theta_1_estimate,
    ci_lower   = theta_1_ci_lower,
    ci_upper   = theta_1_ci_upper,
    chain      = theta_1_chain
  ),
  alpha = list(
    state_name = "alpha (probability)",
    true       = alpha_true,
    estimate   = alpha_estimate,
    ci_lower   = alpha_ci_lower,
    ci_upper   = alpha_ci_upper,
    chain      = alpha_chain
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
# 9) Organized Visualizations (parameters + states)
#-------------------------------------------------------------------------------
# Plots for parameters and states (CI bands)
states_for_plot <- list(
  theta_1 = list(true = theta1_true, estimate = theta_1_estimate,
                 ci_lower = theta_1_ci_lower, ci_upper = theta_1_ci_upper),
  alpha   = list(true = alpha_true, estimate = alpha_estimate,
                 ci_lower = alpha_ci_lower, ci_upper = alpha_ci_upper)
)
generate_organized_plots(param_chains, true_values, states = states_for_plot)

# Observed rate vs. probability
par(mfrow = c(1, 1), mar = c(4, 4, 3, 1))
plot_observed_vs_state(
  y               = y,
  true_state      = alpha_true,
  estimated_state = alpha_estimate,
  title = "Observed Bernoulli outcome vs. probability (alpha)",
  xlab  = "Time",
  ylab  = "Probability",
  n_plot = 200
)

#-------------------------------------------------------------------------------
# 10) Adaptation Diagnostics (not applicable, but helper requires inputs)
#-------------------------------------------------------------------------------
run_adaptation_diagnostics(
  theta_1_updated      = theta_1_updated,
  log_sigma_hist       = log_sigma_hist,
  lag_update           = 0,                 # Set to zero to indicate no adaptation
  target_acceptance    = 1.0,
  max_step_size        = 0.0,
  base_adaptation_rate = 0.0,
  decay_exponent       = 0.0,
  n_iter               = n_iter,
  burnin               = burnin,
  thinning             = thinning,
  band_tolerance       = 0.05
)

cat("=== PROBIT BERNOULLI LOCAL LEVEL VALIDATION COMPLETED ===\n")
