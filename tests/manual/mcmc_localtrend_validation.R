#-------------------------------------------------------------------------------
# Test Script for Local Trend Model Auxiliary Functions
#
# Objective: Validate the auxiliary C functions (exposed in test_helpers.c)
#           by replicating the Gibbs sampler loop in R following mcmc_localtrend.c
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
theta01_true <- 0.1   # Initial level
theta02_true <- 0.05  # Initial trend
prec1_true   <- 10    # Level precision
prec2_true   <- 8     # Trend precision
prec_y_true  <- 1     # Data precision

# set.seed(123) # For reproducibility

# Generate noise
u1 <- rnorm(n, sd = sqrt(1 / prec1_true))
u2 <- rnorm(n, sd = sqrt(1 / prec2_true))
e  <- rnorm(n, sd = sqrt(1 / prec_y_true))

# Simulate latent states and observations
theta1_true <- numeric(n)
theta2_true <- numeric(n)

# Initialize first values
theta1_true[1] <- theta01_true + theta02_true + u1[1]
theta2_true[1] <- theta02_true + u2[1]

# Generate remaining values
for (t in 2:n) {
  theta1_true[t] <- theta1_true[t-1] + theta2_true[t-1] + u1[t]
  theta2_true[t] <- theta2_true[t-1] + u2[t]
}

# Observations
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
mean_theta02 <- 0                   # prior_theta02_mean
prec_theta02 <- .01                 # prior_theta02_prec
nu_01        <- 1e-2                # prior_prec1_shape
eta_01       <- 1e-2                # prior_prec1_rate
nu_02        <- 1e-2                # prior_prec2_shape
eta_02       <- 1e-2                # prior_prec2_rate
nu_y         <- 1e-2                # prior_prec_y_shape
eta_y        <- 1e-2                # prior_prec_y_rate

# --- 3. Chain Initialization ---
theta_1_chain  <- matrix(NA, nrow = n_chain, ncol = n)
theta_2_chain  <- matrix(NA, nrow = n_chain, ncol = n)
theta_01_chain <- numeric(n_chain)
theta_02_chain <- numeric(n_chain)
prec_1_chain   <- numeric(n_chain)
prec_2_chain   <- numeric(n_chain)
prec_y_chain   <- numeric(n_chain)

# --- 4. Initialization (iter = 0) following mcmc_localtrend.c ---
# set.seed(456)

# Arrays to store complete history (as in C implementation)
theta_01_post <- numeric(n_iter)
theta_02_post <- numeric(n_iter)
prec_1_post   <- numeric(n_iter)
prec_2_post   <- numeric(n_iter)
prec_y_post   <- numeric(n_iter)
theta_1_post  <- matrix(NA, nrow = n_iter, ncol = n)
theta_2_post  <- matrix(NA, nrow = n_iter, ncol = n)

# Initial values exactly as in C implementation
theta_01_post[1] <- rnorm(1, mean_theta01, sqrt(1.0 / prec_theta01))
theta_02_post[1] <- rnorm(1, mean_theta02, sqrt(1.0 / prec_theta02))
prec_1_post[1]   <- rgamma(1, nu_01, rate = eta_01)
prec_2_post[1]   <- rgamma(1, nu_02, rate = eta_02)
prec_y_post[1]   <- rgamma(1, nu_y, rate = eta_y)

# Initialize state vectors for t = 1..n
init_sd_1 <- sqrt(1.0 / prec_1_post[1])
init_sd_2 <- sqrt(1.0 / prec_2_post[1])
theta_1_post[1, 1] <- rnorm(1, theta_01_post[1] + theta_02_post[1], init_sd_1)
theta_2_post[1, 1] <- rnorm(1, theta_02_post[1], init_sd_2)
for (j in 2:n) {
  theta_1_post[1, j] <- rnorm(1, theta_1_post[1, j-1] + theta_2_post[1, j-1], init_sd_1)
  theta_2_post[1, j] <- rnorm(1, theta_2_post[1, j-1], init_sd_2)
}

# --- 5. MCMC Loop using Auxiliary Functions ---
cat("Starting MCMC loop for Local Trend model...\n")
chain_idx <- 0  # Counter for saved samples

for (ii in 2:n_iter) {  # Start at 2 since iter=1 is already initialized

  # 1) Sample trend state vector theta_2
  # Function: test_generate_theta_p(theta_pm1_, theta_p_, prec_theta_pm1_, prec_theta_p_, theta_0p_)
  theta_2_new <- .Call(
    "_pdm_test_generate_theta_p",
    as.numeric(theta_1_post[ii-1, ]),   # theta_pm1 (level states, previous iter)
    as.numeric(prec_1_post[ii-1]),      # prec_theta_pm1 (level precision, previous iter)
    as.numeric(prec_2_post[ii-1]),      # prec_theta_p (trend precision, previous iter)
    as.numeric(theta_02_post[ii-1])     # theta_0p (initial trend, previous iter)
  )
  
  if (length(theta_2_new) == n) {
    theta_2_post[ii, ] <- as.numeric(theta_2_new)
  } else {
    warning(paste("Iteration", ii, ": theta_2 was not updated correctly"))
    theta_2_post[ii, ] <- theta_2_post[ii-1, ]  # keep previous value
  }

  # 2) Sample innovation precision 1/W_2
  # Function: test_generate_precision_theta_p(theta_0p_, theta_p_, nu_0p_, eta_0p_)
  prec_2_post[ii] <- .Call(
    "_pdm_test_generate_precision_theta_p",
    as.numeric(theta_02_post[ii-1]),    # theta_0p (initial trend, previous iter)
    as.numeric(theta_2_post[ii, ]),     # theta_p (trend states, current)
    as.numeric(nu_02),                  # nu_0p (prior shape)
    as.numeric(eta_02)                  # eta_0p (prior rate)
  )

  # 3) Sample initial trend state theta_02
  # Function: test_generate_theta_0p(theta_0pm1_, theta_pm1_, theta_p_, prec_theta_pm1_, prec_theta_p_, mean_theta_0p_, prec_theta_0p_)
  theta_02_post[ii] <- .Call(
    "_pdm_test_generate_theta_0p",
    as.numeric(theta_01_post[ii-1]),    # theta_0pm1 (initial level, previous iter)
    as.numeric(theta_1_post[ii, ]),     # theta_pm1 (level states, current)
    as.numeric(theta_2_post[ii, ]),     # theta_p (trend states, current)
    as.numeric(prec_1_post[ii-1]),      # prec_theta_pm1 (level precision, previous iter)
    as.numeric(prec_2_post[ii]),        # prec_theta_p (trend precision, current)
    as.numeric(mean_theta02),           # mean_theta_0p (prior mean)
    as.numeric(prec_theta02)            # prec_theta_0p (prior precision)
  )

  # 4) Sample level state vector theta_1
  # Function: test_generate_theta_1_localtrend(data_, theta_2_, prec_data_, prec_theta_1_, theta_01_, theta_02_)
  theta_1_new <- .Call(
    "_pdm_test_generate_theta_1_localtrend",
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

  # 5) Sample innovation precision 1/W_1
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

  # 6) Sample initial level state theta_01
  # Function: test_generate_theta_01_localtrend(theta_1_, theta_02_, prec_theta_1_, mean_theta_01_, prec_theta_01_)
  theta_01_post[ii] <- .Call(
    "_pdm_test_generate_theta_01_localtrend",
    as.numeric(theta_1_post[ii, ]),     # theta_1 (level states, current)
    as.numeric(theta_02_post[ii]),      # theta_02 (initial trend, current)
    as.numeric(prec_1_post[ii]),        # prec_theta_1 (level precision, current)
    as.numeric(mean_theta01),           # mean_theta_01 (prior mean)
    as.numeric(prec_theta01)            # prec_theta_01 (prior precision)
  )

  # 7) Sample data precision 1/V
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
    theta_01_chain[chain_idx] <- theta_01_post[ii]
    theta_02_chain[chain_idx] <- theta_02_post[ii]
    prec_1_chain[chain_idx]   <- prec_1_post[ii]
    prec_2_chain[chain_idx]   <- prec_2_post[ii]
    prec_y_chain[chain_idx]   <- prec_y_post[ii]
  }

  # Print progress
  if (ii %% 1000 == 0) {
    cat("  Iteration:", ii, "/", n_iter, "\n")
  }
}

cat("... MCMC loop completed.\n\n")

# --- 6. Enhanced Results Analysis ---
# [Rest of the code remains exactly the same as in the previous version with all 5 tables]
# ...