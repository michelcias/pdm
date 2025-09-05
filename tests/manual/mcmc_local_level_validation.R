#-------------------------------------------------------------------------------
# Test Script for Local Level Model Auxiliary Functions
#
# Objective: Validate the C auxiliary functions (exposed in test_helpers.c)
#           by replicating the Gibbs sampler loop in R following mcmc_locallevel.c
#-------------------------------------------------------------------------------
rm(list = ls())
# Load the package to access compiled C functions
devtools::load_all(".")
library(pdm)

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

# --- 6. Results Analysis ---
cat("Summary of posterior estimates (Median):\n")
cat("-----------------------------------------------\n")
cat("Parameter        | True Value | Estimated  | Relative Error\n")
cat("-----------------------------------------------\n")
cat(sprintf("theta_01         | %10.4f | %10.4f | %10.2f%%\n",
            theta0_true, median(theta_01_chain),
            100*abs(median(theta_01_chain) - theta0_true)/theta0_true))
cat(sprintf("prec_1           | %10.4f | %10.4f | %10.2f%%\n",
            prec1_true, median(prec_1_chain),
            100*abs(median(prec_1_chain) - prec1_true)/prec1_true))
cat(sprintf("prec_y           | %10.4f | %10.4f | %10.2f%%\n",
            prec_y_true, median(prec_y_chain),
            100*abs(median(prec_y_chain) - prec_y_true)/prec_y_true))
cat("-----------------------------------------------\n\n")

# Convergence diagnostics
cat("Convergence diagnostics:\n")
cat("-----------------------------\n")
effective_sample_sizes <- sapply(list(theta_01_chain, prec_1_chain, prec_y_chain),
                                 function(x) {
                                   acf_vals <- acf(x, plot=FALSE, lag.max=min(100, length(x)/4))$acf[-1]
                                   length(x) / (1 + 2 * sum(acf_vals[acf_vals > 0]))
                                 })
names(effective_sample_sizes) <- c("theta_01", "prec_1", "prec_y")
print(round(effective_sample_sizes))

# --- 7. Visualization ---
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
theta_1_estimate <- apply(theta_1_chain, 2, median)
theta_1_ci_lower <- apply(theta_1_chain, 2, quantile, 0.025)
theta_1_ci_upper <- apply(theta_1_chain, 2, quantile, 0.975)

plot(theta1_true, type = 'l', col = "red", lty = 2, lwd = .5,
     main = "Latent State: True vs. Estimated", ylab = expression(theta["t,1"]))
lines(theta_1_estimate, col = "blue", lwd = 1)
polygon(c(1:n, rev(1:n)), c(theta_1_ci_lower, rev(theta_1_ci_upper)),
        col = rgb(0, 0, 1, 0.2), border = NA)
legend("topleft", legend = c("True", "Estimated", "95% CI"),
       col = c("red", "blue", rgb(0, 0, 1, 0.2)),
       lty = c(2, 1, 1), lwd = c(2, 2, 8), bty = "n")
par(op)