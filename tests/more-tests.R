set.seed(404)
n <- 500
n_trials <- 100

# True parameters for simulation
theta_01_true <- 0.5
prec_1_true <- 100.0

# Simulate data
u1 <- rnorm(n, sd = sqrt(1 / prec_1_true))
theta_1_true <- cumsum(c(theta_01_true, u1))[-1]
plot.ts(theta_1_true)

alpha_true <- plogis(theta_1_true)
plot.ts(alpha_true)

y <- rbinom(n, size = n_trials, prob = alpha_true)
plot.ts(y, type = "o")

# --- 2. R Wrapper for the Test Sampler ---
test_sampler <- function(y, n_trials, burnin, thinning, n_chain,
                         theta_1_true = NULL, theta_01_true = NULL, prec_1_true = NULL,
                         prior_theta01_mean = 0.0, prior_theta01_prec = 1.0,
                         prior_prec1_shape = 1.0, prior_prec1_rate = 1.0,
                         lag_update = 50L, max_step_size = 0.1,
                         base_adaptation_rate = 1.0, decay_exponent = 0.5,
                         target_acceptance = 0.44) {

  .Call("_pdm_test_mcmc_binomial_locallevel_fixed_params",
        y, n_trials, burnin, thinning, n_chain,
        theta_1_true, theta_01_true, prec_1_true,
        prior_theta01_mean, prior_theta01_prec,
        prior_prec1_shape, prior_prec1_rate,
        lag_update, max_step_size, base_adaptation_rate,
        decay_exponent, target_acceptance)
}

# Test A: Sample theta_01, fixing theta_1 and prec_1
set.seed(405)
mcmc_out_A <- test_sampler(y, n_trials,
                           burnin = 5000, thinning = 25, n_chain = 2000,
                           theta_1_true = theta_1_true,
                           prec_1_true = prec_1_true,
                           prior_theta01_mean = 0, # Prior for theta_01
                           prior_theta01_prec = 1.0)


plot.ts(mcmc_out_A$theta_1[1,])
points(theta_1_true, col = "red", type = "l")
# Check if the posterior mean of theta_01 is close to the true value
posterior_mean_A <- mean(mcmc_out_A$theta_01)
plot.ts(mcmc_out_A$theta_01)
abline(h = theta_01_true, col = "red")
abline(h = posterior_mean_A, col = "blue")
posterior_mean_A - theta_01_true

mcmc_out_B <- test_sampler(y, n_trials,
                           burnin = 5000, thinning = 25, n_chain = 2000,
                           theta_1_true = theta_1_true,
                           theta_01_true = theta_01_true,
                           prior_prec1_shape = 100, # Prior for prec_1
                           prior_prec1_rate = 1)

# Check if the posterior mean of prec_1 is close to the true value
posterior_mean_B <- mean(mcmc_out_B$prec_1)
plot.ts(mcmc_out_B$prec_1)
abline(h = prec_1_true, col = "red")
abline(h = posterior_mean_B, col = "blue")
posterior_mean_B - prec_1_true
abs(posterior_mean_B - prec_1_true) / prec_1_true

# --- 3. Run Tests for Each Conditional ---

mcmc_out_C <- test_sampler(y, n_trials,
                           burnin = 5000, thinning = 25, n_chain = 1000,
                           theta_01_true = theta_01_true,
                           prec_1_true = prec_1_true)

# Check if the posterior mean of theta_1 is close to the true value
posterior_mean_C <- colMeans(mcmc_out_C$theta_1)
# Check the average absolute difference
mean_abs_diff <- mean(abs(posterior_mean_C - theta_1_true))
plot.ts(posterior_mean_C)
points(theta_1_true, col = "red", type = "l")
