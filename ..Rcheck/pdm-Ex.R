pkgname <- "pdm"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
library('pdm')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("mcmc_binomial_localacceleration")
### * mcmc_binomial_localacceleration

flush(stderr()); flush(stdout())

### Name: mcmc_binomial_localacceleration
### Title: Gibbs Sampler for a Local-Acceleration Binomial Dynamic Model
### Aliases: mcmc_binomial_localacceleration

### ** Examples

## Description
# This example demonstrates how to:
# 1. Simulate data from a local-acceleration binomial dynamic model
# 2. Use `mcmc_binomial_localacceleration` to estimate parameters and latent states
# 3. Perform a detailed posterior analysis with visualizations
# 4. Set a seed for reproducibility

## Simulation of data
n <- 500        # Number of observations to simulate
n_trials <- 20  # Number of binomial trials

# True parameters for simulation:
theta01_true <- 0.5     # Initial level state (theta[0,1]) on logit scale
theta02_true <- 0.01    # Initial trend state (theta[0,2]) on logit scale
theta03_true <- 0.001   # Initial acceleration state (theta[0,3]) on logit scale
prec1_true <- 100       # Level innovation precision (1/W[1])
prec2_true <- 400       # Trend innovation precision (1/W[2])
prec3_true <- 1600      # Acceleration innovation precision (1/W[3])

# Use a fixed seed for data simulation
set.seed(123)

# Generate noise terms:
u1 <- rnorm(n, sd = sqrt(1/prec1_true))  # Level evolution noise (u1[t])
u2 <- rnorm(n, sd = sqrt(1/prec2_true))  # Trend evolution noise (u2[t])
u3 <- rnorm(n, sd = sqrt(1/prec3_true))  # Acceleration evolution noise (u3[t])

# Simulate latent states and observations:
theta1_true <- numeric(n)
theta2_true <- numeric(n)
theta3_true <- numeric(n)
theta3_true[1] <- theta03_true + u3[1]
theta2_true[1] <- theta02_true + theta03_true + u2[1]
theta1_true[1] <- theta01_true + theta02_true + u1[1]
for (t in 2:n) {
  theta3_true[t] <- theta3_true[t-1] + u3[t]
  theta2_true[t] <- theta2_true[t-1] + theta3_true[t-1] + u2[t]
  theta1_true[t] <- theta1_true[t-1] + theta2_true[t-1] + u1[t]
}
alpha_true <- plogis(theta1_true)  # Success probabilities
y <- rbinom(n, size = n_trials, prob = alpha_true)  # Observed binomial counts

## Running the Gibbs sampler
# Run the Gibbs sampler with specified priors and a seed
out <- mcmc_binomial_localacceleration(
  y,
  n_trials             = n_trials,
  burnin               = 1000,
  thinning             = 50,
  n_chain              = 1000,
  prior_theta01_mean   = 0,
  prior_theta01_prec   = 1,
  prior_theta02_mean   = 0,
  prior_theta02_prec   = 1,
  prior_theta03_mean   = 0,
  prior_theta03_prec   = 1,
  prior_prec1_shape    = 100,
  prior_prec1_rate     = 1,
  prior_prec2_shape    = 400,
  prior_prec2_rate     = 1,
  prior_prec3_shape    = 1600,
  prior_prec3_rate     = 1,
  lag_update           = 50,
  max_step_size        = 0.1,
  base_adaptation_rate = 1,
  decay_exponent       = 0.6,
  target_acceptance    = 0.44,
  return_log_sigma     = FALSE,
  return_accrate       = TRUE,
  seed                 = 456
)

## Posterior analysis and visualization
# The following plots show how to analyze the posterior distributions.
# Point estimates are based on the median of posterior samples.
## Not run: 
##D   # --- 0. Plot the simulated data ---
##D   plot(
##D     y,
##D     main = "Simulated binomial counts",
##D     ylab = expression(y[t]),
##D     xlab = "t",
##D     type = "o",
##D     pch = 16,
##D     cex = 0.7
##D   )
##D 
##D   # --- 1. Metropolis-Hastings Acceptance Rate Diagnostics ---
##D   # Extract acceptance rate statistics for adaptive MCMC performance evaluation
##D   acc <- out$accrate
##D   min_acc <- apply(X = acc, MARGIN = 2, FUN = min)
##D   max_acc <- apply(X = acc, MARGIN = 2, FUN = max)
##D   med_acc <- apply(X = acc, MARGIN = 2, FUN = median)
##D 
##D   # Calculate y-axis range for optimal legend positioning
##D   range_acc <- range(min_acc, max_acc)
##D   r1_acc <- range_acc[1] - 0.05
##D   r2_acc <- range_acc[2] + 0.15 * diff(range_acc)
##D 
##D   # Plot acceptance rates with target reference line and confidence bands
##D   plot(
##D     med_acc,
##D     type = "l",
##D     col = "black",
##D     lwd = 2,
##D     xlab = "Time (t)",
##D     ylab = "Acceptance Rate",
##D     main = "Metropolis-Hastings Acceptance Rates",
##D     ylim = c(r1_acc, r2_acc)
##D   )
##D 
##D   # Add confidence bands showing min-max range across MCMC iterations
##D   polygon(
##D     c(1:length(med_acc), rev(1:length(med_acc))),
##D     c(min_acc, rev(max_acc)),
##D     col = rgb(0.7, 0.7, 0.7, alpha = 0.3),
##D     border = NA
##D   )
##D 
##D   # Overlay target acceptance rate
##D   abline(h = 0.44, col = "red", lty = 2, lwd = 2)
##D 
##D   # Add informative legend positioned in the expanded y-range
##D   legend(
##D     "topright",
##D     legend = c("Median acceptance rate", "Min-Max range", "Target rate (0.44)"),
##D     col = c("black", "gray", "red"),
##D     lty = c(1, 1, 2),
##D     lwd = c(2, 8, 2),
##D     bty = "n"
##D   )
##D 
##D   # --- 2. Latent Level Trajectories (theta[t,1]) on logit scale ---
##D   # Visualize uncertainty by plotting multiple posterior trajectory samples
##D   num_traj_to_plot <- 20
##D 
##D   # Calculate y-axis range for optimal legend positioning
##D   range_traj <- range(out$theta_1[1:num_traj_to_plot, ], theta1_true)
##D   r1_traj <- range_traj[1] - 0.1 * diff(range_traj)
##D   r2_traj <- range_traj[2] + 0.1 * diff(range_traj)
##D 
##D   matplot(
##D     t(out$theta_1[1:num_traj_to_plot, ]),
##D     type = "l",
##D     lty = 1,
##D     col = grDevices::rainbow(num_traj_to_plot, alpha = 0.3),
##D     xlab = "t",
##D     ylab = expression(theta["t,1"]),
##D     main = "Posterior trajectory samples for latent level (logit scale)",
##D     ylim = c(r1_traj, r2_traj)
##D   )
##D   # Overlay the true trajectory
##D   lines(theta1_true, col = "black", lwd = 3, lty = 2)
##D   legend(
##D     "topright",
##D     legend = expression(theta["t,1"]),
##D     col = "black",
##D     lty = 2,
##D     lwd = 3,
##D     bty = "n"
##D   )
##D 
##D   # --- 3. Latent Trend Trajectories (theta[t,2]) on logit scale ---
##D   # Visualize uncertainty by plotting multiple posterior trajectory samples
##D   range_traj2 <- range(out$theta_2[1:num_traj_to_plot, ], theta2_true)
##D   r1_traj2 <- range_traj2[1] - 0.1 * diff(range_traj2)
##D   r2_traj2 <- range_traj2[2] + 0.1 * diff(range_traj2)
##D 
##D   matplot(
##D     t(out$theta_2[1:num_traj_to_plot, ]),
##D     type = "l",
##D     lty = 1,
##D     col = grDevices::rainbow(num_traj_to_plot, alpha = 0.3),
##D     xlab = "t",
##D     ylab = expression(theta["t,2"]),
##D     main = "Posterior trajectory samples for latent trend (logit scale)",
##D     ylim = c(r1_traj2, r2_traj2)
##D   )
##D   # Overlay the true trajectory
##D   lines(theta2_true, col = "black", lwd = 3, lty = 2)
##D   legend(
##D     "topright",
##D     legend = expression(theta["t,2"]),
##D     col = "black",
##D     lty = 2,
##D     lwd = 3,
##D     bty = "n"
##D   )
##D 
##D   # --- 4. Latent Acceleration Trajectories (theta[t,3]) on logit scale ---
##D   # Visualize uncertainty by plotting multiple posterior trajectory samples
##D   range_traj3 <- range(out$theta_3[1:num_traj_to_plot, ], theta3_true)
##D   r1_traj3 <- range_traj3[1] - 0.1 * diff(range_traj3)
##D   r2_traj3 <- range_traj3[2] + 0.1 * diff(range_traj3)
##D 
##D   matplot(
##D     t(out$theta_3[1:num_traj_to_plot, ]),
##D     type = "l",
##D     lty = 1,
##D     col = grDevices::rainbow(num_traj_to_plot, alpha = 0.3),
##D     xlab = "t",
##D     ylab = expression(theta["t,3"]),
##D     main = "Posterior trajectory samples for latent acceleration (logit scale)",
##D     ylim = c(r1_traj3, r2_traj3)
##D   )
##D   # Overlay the true trajectory
##D   lines(theta3_true, col = "black", lwd = 3, lty = 2)
##D   legend(
##D     "topright",
##D     legend = expression(theta["t,3"]),
##D     col = "black",
##D     lty = 2,
##D     lwd = 3,
##D     bty = "n"
##D   )
##D 
##D   # --- 5. Latent Level Point Estimates (theta[t,1]) ---
##D   # Plot true and estimated (median) latent level with credible intervals
##D   theta_1_estimate <- apply(X = out$theta_1, MARGIN = 2, FUN = median)
##D   theta_1_q025 <- apply(X = out$theta_1, MARGIN = 2, FUN = quantile, probs = 0.025)
##D   theta_1_q975 <- apply(X = out$theta_1, MARGIN = 2, FUN = quantile, probs = 0.975)
##D 
##D   range_theta_1 <- range(theta_1_estimate, theta1_true, theta_1_q025, theta_1_q975)
##D   r1_theta1 <- range_theta_1[1] - 0.1 * diff(range_theta_1)
##D   r2_theta1 <- range_theta_1[2] + 0.3 * diff(range_theta_1)
##D 
##D   plot(
##D     theta1_true,
##D     col = "red",
##D     type = "l",
##D     lwd = 3,
##D     xlab = "t",
##D     ylim = c(r1_theta1, r2_theta1),
##D     lty = 2,
##D     ylab = expression(theta["t,1"]),
##D     main = "Latent level estimation (logit scale)"
##D   )
##D 
##D   # Add 95% credible intervals
##D   polygon(
##D     c(1:length(theta_1_estimate), rev(1:length(theta_1_estimate))),
##D     c(theta_1_q025, rev(theta_1_q975)),
##D     col = rgb(0.7, 0.7, 0.7, alpha = 0.3),
##D     border = NA
##D   )
##D 
##D   # Add point estimate
##D   lines(theta_1_estimate, col = "black", lwd = 2)
##D 
##D   legend(
##D     "topright",
##D     legend = c(
##D       expression(theta["t,1"]),
##D       expression(hat(theta)["t,1"]),
##D       "95% Credible Interval"
##D     ),
##D     col = c("red", "black", "gray"),
##D     lty = c(2, 1, 1),
##D     lwd = c(3, 2, 8),
##D     bty = "n"
##D   )
##D 
##D   # --- 6. Latent Trend Point Estimates (theta[t,2]) ---
##D   # Plot true and estimated (median) latent trend with credible intervals
##D   theta_2_estimate <- apply(X = out$theta_2, MARGIN = 2, FUN = median)
##D   theta_2_q025 <- apply(X = out$theta_2, MARGIN = 2, FUN = quantile, probs = 0.025)
##D   theta_2_q975 <- apply(X = out$theta_2, MARGIN = 2, FUN = quantile, probs = 0.975)
##D 
##D   range_theta_2 <- range(theta_2_estimate, theta2_true, theta_2_q025, theta_2_q975)
##D   r1_theta2 <- range_theta_2[1] - 0.1 * diff(range_theta_2)
##D   r2_theta2 <- range_theta_2[2] + 0.3 * diff(range_theta_2)
##D 
##D   plot(
##D     theta2_true,
##D     col = "red",
##D     type = "l",
##D     lwd = 3,
##D     xlab = "t",
##D     ylim = c(r1_theta2, r2_theta2),
##D     lty = 2,
##D     ylab = expression(theta["t,2"]),
##D     main = "Latent trend estimation (logit scale)"
##D   )
##D 
##D   # Add 95% credible intervals
##D   polygon(
##D     c(1:length(theta_2_estimate), rev(1:length(theta_2_estimate))),
##D     c(theta_2_q025, rev(theta_2_q975)),
##D     col = rgb(0.7, 0.7, 0.7, alpha = 0.3),
##D     border = NA
##D   )
##D 
##D   # Add point estimate
##D   lines(theta_2_estimate, col = "black", lwd = 2)
##D 
##D   legend(
##D     "topright",
##D     legend = c(
##D       expression(theta["t,2"]),
##D       expression(hat(theta)["t,2"]),
##D       "95% Credible Interval"
##D     ),
##D     col = c("red", "black", "gray"),
##D     lty = c(2, 1, 1),
##D     lwd = c(3, 2, 8),
##D     bty = "n"
##D   )
##D 
##D   # --- 7. Latent Acceleration Point Estimates (theta[t,3]) ---
##D   # Plot true and estimated (median) latent acceleration with credible intervals
##D   theta_3_estimate <- apply(X = out$theta_3, MARGIN = 2, FUN = median)
##D   theta_3_q025 <- apply(X = out$theta_3, MARGIN = 2, FUN = quantile, probs = 0.025)
##D   theta_3_q975 <- apply(X = out$theta_3, MARGIN = 2, FUN = quantile, probs = 0.975)
##D 
##D   range_theta_3 <- range(theta_3_estimate, theta3_true, theta_3_q025, theta_3_q975)
##D   r1_theta3 <- range_theta_3[1] - 0.1 * diff(range_theta_3)
##D   r2_theta3 <- range_theta_3[2] + 0.3 * diff(range_theta_3)
##D 
##D   plot(
##D     theta3_true,
##D     col = "red",
##D     type = "l",
##D     lwd = 3,
##D     xlab = "t",
##D     ylim = c(r1_theta3, r2_theta3),
##D     lty = 2,
##D     ylab = expression(theta["t,3"]),
##D     main = "Latent acceleration estimation (logit scale)"
##D   )
##D 
##D   # Add 95% credible intervals
##D   polygon(
##D     c(1:length(theta_3_estimate), rev(1:length(theta_3_estimate))),
##D     c(theta_3_q025, rev(theta_3_q975)),
##D     col = rgb(0.7, 0.7, 0.7, alpha = 0.3),
##D     border = NA
##D   )
##D 
##D   # Add point estimate
##D   lines(theta_3_estimate, col = "black", lwd = 2)
##D 
##D   legend(
##D     "topright",
##D     legend = c(
##D       expression(theta["t,3"]),
##D       expression(hat(theta)["t,3"]),
##D       "95% Credible Interval"
##D     ),
##D     col = c("red", "black", "gray"),
##D     lty = c(2, 1, 1),
##D     lwd = c(3, 2, 8),
##D     bty = "n"
##D   )
##D 
##D   # --- 8. Success Probabilities (alpha[t]) ---
##D   # Plot true and estimated (median) success probabilities with uncertainty
##D   alpha_estimate <- apply(X = out$alpha, MARGIN = 2, FUN = median)
##D   alpha_q025 <- apply(X = out$alpha, MARGIN = 2, FUN = quantile, probs = 0.025)
##D   alpha_q975 <- apply(X = out$alpha, MARGIN = 2, FUN = quantile, probs = 0.975)
##D 
##D   # Calculate y-axis range for optimal legend positioning
##D   range_alpha <- range(alpha_true, alpha_estimate, alpha_q025, alpha_q975)
##D   r1_alpha <- max(0, range_alpha[1] - 0.05)  # Ensure lower bound is at least 0
##D   r2_alpha <- min(1, range_alpha[2] + 0.3 * diff(range_alpha))  # Ensure upper bound is at most 1
##D 
##D   plot(
##D     alpha_true,
##D     col = "red",
##D     type = "l",
##D     lwd = 3,
##D     xlab = "t",
##D     ylim = c(r1_alpha, r2_alpha),
##D     lty = 2,
##D     ylab = expression(alpha[t]),
##D     main = "Success probabilities"
##D   )
##D 
##D   # Add 95% credible intervals
##D   polygon(
##D     c(1:length(alpha_estimate), rev(1:length(alpha_estimate))),
##D     c(alpha_q025, rev(alpha_q975)),
##D     col = rgb(0.7, 0.7, 0.7, alpha = 0.3),
##D     border = NA
##D   )
##D 
##D   # Add point estimate
##D   lines(alpha_estimate, col = "black", lwd = 2)
##D 
##D   legend(
##D     "topright",
##D     legend = c(
##D       expression(alpha[t]),
##D       expression(hat(alpha)[t]),
##D       "95% Credible Interval"
##D     ),
##D     col = c("red", "black", "gray"),
##D     lty = c(2, 1, 1),
##D     lwd = c(3, 2, 8),
##D     bty = "n"
##D   )
##D 
##D   # --- 9. Initial Level State (theta[0,1]) Diagnostics ---
##D   # Trace plot for theta[0,1] to assess MCMC convergence
##D   range_theta_01 <- range(out$theta_01)
##D   r1_theta01 <- range_theta_01[1] - 0.1 * diff(range_theta_01)
##D   r2_theta01 <- range_theta_01[2] + 0.3 * diff(range_theta_01)
##D 
##D   plot.ts(
##D     out$theta_01,
##D     ylab = expression(theta["0,1"]),
##D     main = "Trace plot of initial level state",
##D     xlab = "Iterations",
##D     col = "gray",
##D     ylim = c(r1_theta01, r2_theta01)
##D   )
##D   abline(
##D     h = c(theta01_true, median(out$theta_01)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(theta["0,1"]), expression(hat(theta)["0,1"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
##D 
##D   # Posterior density estimate for theta[0,1]
##D   range_dens_theta01 <- range(out$theta_01)
##D   r1_dens_theta01 <- range_dens_theta01[1] - 0.1 * diff(range_dens_theta01)
##D   r2_dens_theta01 <- range_dens_theta01[2] + 0.25 * diff(range_dens_theta01)
##D 
##D   plot(
##D     density(out$theta_01),
##D     main = "Posterior density estimate of initial level state",
##D     xlab = expression(theta["0,1"]),
##D     ylab = "Density",
##D     lwd = 2,
##D     xlim = c(r1_dens_theta01, r2_dens_theta01)
##D   )
##D   abline(
##D     v = c(theta01_true, median(out$theta_01)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(theta["0,1"]), expression(hat(theta)["0,1"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
##D 
##D   # --- 10. Initial Trend State (theta[0,2]) Diagnostics ---
##D   # Trace plot for theta[0,2] to assess MCMC convergence
##D   range_theta_02 <- range(out$theta_02)
##D   r1_theta02 <- range_theta_02[1] - 0.1 * diff(range_theta_02)
##D   r2_theta02 <- range_theta_02[2] + 0.3 * diff(range_theta_02)
##D 
##D   plot.ts(
##D     out$theta_02,
##D     ylab = expression(theta["0,2"]),
##D     main = "Trace plot of initial trend state",
##D     xlab = "Iterations",
##D     col = "gray",
##D     ylim = c(r1_theta02, r2_theta02)
##D   )
##D   abline(
##D     h = c(theta02_true, median(out$theta_02)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(theta["0,2"]), expression(hat(theta)["0,2"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
##D 
##D   # Posterior density estimate for theta[0,2]
##D   range_dens_theta02 <- range(out$theta_02)
##D   r1_dens_theta02 <- range_dens_theta02[1] - 0.1 * diff(range_dens_theta02)
##D   r2_dens_theta02 <- range_dens_theta02[2] + 0.25 * diff(range_dens_theta02)
##D 
##D   plot(
##D     density(out$theta_02),
##D     main = "Posterior density estimate of initial trend state",
##D     xlab = expression(theta["0,2"]),
##D     ylab = "Density",
##D     lwd = 2,
##D     xlim = c(r1_dens_theta02, r2_dens_theta02)
##D   )
##D   abline(
##D     v = c(theta02_true, median(out$theta_02)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(theta["0,2"]), expression(hat(theta)["0,2"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
##D 
##D   # --- 11. Initial Acceleration State (theta[0,3]) Diagnostics ---
##D   # Trace plot for theta[0,3] to assess MCMC convergence
##D   range_theta_03 <- range(out$theta_03)
##D   r1_theta03 <- range_theta_03[1] - 0.1 * diff(range_theta_03)
##D   r2_theta03 <- range_theta_03[2] + 0.3 * diff(range_theta_03)
##D 
##D   plot.ts(
##D     out$theta_03,
##D     ylab = expression(theta["0,3"]),
##D     main = "Trace plot of initial acceleration state",
##D     xlab = "Iterations",
##D     col = "gray",
##D     ylim = c(r1_theta03, r2_theta03)
##D   )
##D   abline(
##D     h = c(theta03_true, median(out$theta_03)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(theta["0,3"]), expression(hat(theta)["0,3"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
##D 
##D   # Posterior density estimate for theta[0,3]
##D   range_dens_theta03 <- range(out$theta_03)
##D   r1_dens_theta03 <- range_dens_theta03[1] - 0.1 * diff(range_dens_theta03)
##D   r2_dens_theta03 <- range_dens_theta03[2] + 0.25 * diff(range_dens_theta03)
##D 
##D   plot(
##D     density(out$theta_03),
##D     main = "Posterior density estimate of initial acceleration state",
##D     xlab = expression(theta["0,3"]),
##D     ylab = "Density",
##D     lwd = 2,
##D     xlim = c(r1_dens_theta03, r2_dens_theta03)
##D   )
##D   abline(
##D     v = c(theta03_true, median(out$theta_03)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(theta["0,3"]), expression(hat(theta)["0,3"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
##D 
##D   # --- 12. Level Innovation Precision (1/W[1]) Diagnostics ---
##D   # Trace plot for 1/W[1] to assess parameter convergence
##D   range_prec_1 <- range(out$prec_1)
##D   r1_prec1 <- range_prec_1[1] - 0.1 * diff(range_prec_1)
##D   r2_prec1 <- range_prec_1[2] + 0.25 * diff(range_prec_1)
##D 
##D   plot.ts(
##D     out$prec_1,
##D     ylab = expression(1/W[1]),
##D     main = "Trace plot of level innovation precision",
##D     xlab = "Iterations",
##D     col = "gray",
##D     ylim = c(r1_prec1, r2_prec1)
##D   )
##D   abline(
##D     h = c(prec1_true, median(out$prec_1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(W[1]^-1), expression(hat(W)[1]^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
##D 
##D   # Posterior density estimate for 1/W[1]
##D   range_dens_prec1 <- range(out$prec_1)
##D   r1_dens_prec1 <- range_dens_prec1[1] - 0.1 * diff(range_dens_prec1)
##D   r2_dens_prec1 <- range_dens_prec1[2] + 0.25 * diff(range_dens_prec1)
##D 
##D   plot(
##D     density(out$prec_1),
##D     main = "Posterior density estimate of level innovation precision",
##D     xlab = expression(W[1]^-1),
##D     ylab = "Density",
##D     lwd = 2,
##D     xlim = c(r1_dens_prec1, r2_dens_prec1)
##D   )
##D   abline(
##D     v = c(prec1_true, median(out$prec_1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(W[1]^-1), expression(hat(W)[1]^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
##D 
##D   # --- 13. Trend Innovation Precision (1/W[2]) Diagnostics ---
##D   # Trace plot for 1/W[2] to assess parameter convergence
##D   range_prec_2 <- range(out$prec_2)
##D   r1_prec2 <- range_prec_2[1] - 0.1 * diff(range_prec_2)
##D   r2_prec2 <- range_prec_2[2] + 0.25 * diff(range_prec_2)
##D 
##D   plot.ts(
##D     out$prec_2,
##D     ylab = expression(1/W[2]),
##D     main = "Trace plot of trend innovation precision",
##D     xlab = "Iterations",
##D     col = "gray",
##D     ylim = c(r1_prec2, r2_prec2)
##D   )
##D   abline(
##D     h = c(prec2_true, median(out$prec_2)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(W[2]^-1), expression(hat(W)[2]^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
##D 
##D   # Posterior density estimate for 1/W[2]
##D   range_dens_prec2 <- range(out$prec_2)
##D   r1_dens_prec2 <- range_dens_prec2[1] - 0.1 * diff(range_dens_prec2)
##D   r2_dens_prec2 <- range_dens_prec2[2] + 0.25 * diff(range_dens_prec2)
##D 
##D   plot(
##D     density(out$prec_2),
##D     main = "Posterior density estimate of trend innovation precision",
##D     xlab = expression(W[2]^-1),
##D     ylab = "Density",
##D     lwd = 2,
##D     xlim = c(r1_dens_prec2, r2_dens_prec2)
##D   )
##D   abline(
##D     v = c(prec2_true, median(out$prec_2)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(W[2]^-1), expression(hat(W)[2]^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
##D 
##D   # --- 14. Acceleration Innovation Precision (1/W[3]) Diagnostics ---
##D   # Trace plot for 1/W[3] to assess parameter convergence
##D   range_prec_3 <- range(out$prec_3)
##D   r1_prec3 <- range_prec_3[1] - 0.1 * diff(range_prec_3)
##D   r2_prec3 <- range_prec_3[2] + 0.25 * diff(range_prec_3)
##D 
##D   plot.ts(
##D     out$prec_3,
##D     ylab = expression(1/W[3]),
##D     main = "Trace plot of acceleration innovation precision",
##D     xlab = "Iterations",
##D     col = "gray",
##D     ylim = c(r1_prec3, r2_prec3)
##D   )
##D   abline(
##D     h = c(prec3_true, median(out$prec_3)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(W[3]^-1), expression(hat(W)[3]^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
##D 
##D   # Posterior density estimate for 1/W[3]
##D   range_dens_prec3 <- range(out$prec_3)
##D   r1_dens_prec3 <- range_dens_prec3[1] - 0.1 * diff(range_dens_prec3)
##D   r2_dens_prec3 <- range_dens_prec3[2] + 0.25 * diff(range_dens_prec3)
##D 
##D   plot(
##D     density(out$prec_3),
##D     main = "Posterior density estimate of acceleration innovation precision",
##D     xlab = expression(W[3]^-1),
##D     ylab = "Density",
##D     lwd = 2,
##D     xlim = c(r1_dens_prec3, r2_dens_prec3)
##D   )
##D   abline(
##D     v = c(prec3_true, median(out$prec_3)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(W[3]^-1), expression(hat(W)[3]^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
## End(Not run)




cleanEx()
nameEx("mcmc_binomial_locallevel")
### * mcmc_binomial_locallevel

flush(stderr()); flush(stdout())

### Name: mcmc_binomial_locallevel
### Title: Gibbs Sampler for a Local-Level Binomial Dynamic Model
### Aliases: mcmc_binomial_locallevel

### ** Examples

## Description
# This example demonstrates how to:
# 1. Simulate data from a local-level binomial dynamic model
# 2. Use `mcmc_binomial_locallevel` to estimate parameters and latent states
# 3. Perform a detailed posterior analysis with visualizations
# 4. Set a seed for reproducibility

## Simulation of data
n <- 500        # Number of observations to simulate
n_trials <- 20  # Number of binomial trials

# True parameters for simulation:
theta0_true <- 0.5     # Initial state (theta[01]) on logit scale
prec1_true <- 100      # Innovation precision (1/W[1])

# Use a fixed seed for data simulation
set.seed(123)

# Generate noise terms:
u1 <- rnorm(n, sd = sqrt(1/prec1_true))  # Evolution noise (u1[t])

# Simulate latent states and observations:
theta1_true <- cumsum(c(theta0_true, u1))[-1]  # theta[t1] series on logit scale
alpha_true <- plogis(theta1_true)              # Success probabilities
y <- rbinom(n, size = n_trials, prob = alpha_true)  # Observed binomial counts

## Running the Gibbs sampler
# Run the Gibbs sampler with specified priors and a seed
out <- mcmc_binomial_locallevel(
  y,
  n_trials             = n_trials,
  burnin               = 1000,
  thinning             = 50,
  n_chain              = 1000,
  prior_theta01_mean   = 0,
  prior_theta01_prec   = 1,
  prior_prec1_shape    = 100,
  prior_prec1_rate     = 1,
  lag_update           = 50,
  max_step_size        = 0.1,
  base_adaptation_rate = 1,
  decay_exponent       = 0.6,
  target_acceptance    = 0.44,
  return_log_sigma     = FALSE,
  return_accrate       = TRUE,
  seed                 = 456
)

## Posterior analysis and visualization
# The following plots show how to analyze the posterior distributions.
# Point estimates are based on the median of posterior samples.
## Not run: 
##D   # --- 0. Plot the simulated data ---
##D   plot(
##D     y,
##D     main = "Simulated binomial counts",
##D     ylab = expression(y[t]),
##D     xlab = "t",
##D     type = "o",
##D     pch = 16,
##D     cex = 0.7
##D   )
##D 
##D   # --- 1. Metropolis-Hastings Acceptance Rate Diagnostics ---
##D   # Extract acceptance rate statistics for adaptive MCMC performance evaluation
##D   acc <- out$accrate
##D   min_acc <- apply(X = acc, MARGIN = 2, FUN = min)
##D   max_acc <- apply(X = acc, MARGIN = 2, FUN = max)
##D   med_acc <- apply(X = acc, MARGIN = 2, FUN = median)
##D 
##D   # Calculate y-axis range for optimal legend positioning
##D   range_acc <- range(min_acc, max_acc)
##D   r1_acc <- range_acc[1] - 0.05
##D   r2_acc <- range_acc[2] + 0.15 * diff(range_acc)
##D 
##D   # Plot acceptance rates with target reference line and confidence bands
##D   plot(
##D     med_acc,
##D     type = "l",
##D     col = "black",
##D     lwd = 2,
##D     xlab = "Time (t)",
##D     ylab = "Acceptance Rate",
##D     main = "Metropolis-Hastings Acceptance Rates",
##D     ylim = c(r1_acc, r2_acc)
##D   )
##D 
##D   # Add confidence bands showing min-max range across MCMC iterations
##D   polygon(
##D     c(1:length(med_acc), rev(1:length(med_acc))),
##D     c(min_acc, rev(max_acc)),
##D     col = rgb(0.7, 0.7, 0.7, alpha = 0.3),
##D     border = NA
##D   )
##D 
##D   # Overlay target acceptance rate
##D   abline(h = 0.44, col = "red", lty = 2, lwd = 2)
##D 
##D   # Add informative legend positioned in the expanded y-range
##D   legend(
##D     "topright",
##D     legend = c("Median acceptance rate", "Min-Max range", "Target rate (0.44)"),
##D     col = c("black", "gray", "red"),
##D     lty = c(1, 1, 2),
##D     lwd = c(2, 8, 2),
##D     bty = "n"
##D   )
##D 
##D   # --- 2. Latent State Trajectories (theta[t1]) on logit scale ---
##D   # Visualize uncertainty by plotting multiple posterior trajectory samples
##D   num_traj_to_plot <- 20
##D 
##D   # Calculate y-axis range for optimal legend positioning
##D   range_traj <- range(out$theta_1[1:num_traj_to_plot, ], theta1_true)
##D   r1_traj <- range_traj[1] - 0.1 * diff(range_traj)
##D   r2_traj <- range_traj[2] + 0.1 * diff(range_traj)
##D 
##D   matplot(
##D     t(out$theta_1[1:num_traj_to_plot, ]),
##D     type = "l",
##D     lty = 1,
##D     col = grDevices::rainbow(num_traj_to_plot, alpha = 0.3),
##D     xlab = "t",
##D     ylab = expression(theta["t,1"]),
##D     main = "Posterior trajectory samples for latent state (logit scale)",
##D     ylim = c(r1_traj, r2_traj)
##D   )
##D   # Overlay the true trajectory
##D   lines(theta1_true, col = "black", lwd = 3, lty = 2)
##D   legend(
##D     "topright",
##D     legend = expression(theta["t,1"]),
##D     col = "black",
##D     lty = 2,
##D     lwd = 3,
##D     bty = "n"
##D   )
##D 
##D   # --- 3. Latent State Point Estimates (theta[t1]) ---
##D   # Plot true and estimated (median) latent state with credible intervals
##D   theta_1_estimate <- apply(X = out$theta_1, MARGIN = 2, FUN = median)
##D   theta_1_q025 <- apply(X = out$theta_1, MARGIN = 2, FUN = quantile, probs = 0.025)
##D   theta_1_q975 <- apply(X = out$theta_1, MARGIN = 2, FUN = quantile, probs = 0.975)
##D 
##D   range_theta_1 <- range(theta_1_estimate, theta1_true, theta_1_q025, theta_1_q975)
##D   r1_theta1 <- range_theta_1[1] - 0.1 * diff(range_theta_1)
##D   r2_theta1 <- range_theta_1[2] + 0.3 * diff(range_theta_1)
##D 
##D   plot(
##D     theta1_true,
##D     col = "red",
##D     type = "l",
##D     lwd = 3,
##D     xlab = "t",
##D     ylim = c(r1_theta1, r2_theta1),
##D     lty = 2,
##D     ylab = expression(theta["t,1"]),
##D     main = "Latent state estimation (logit scale)"
##D   )
##D 
##D   # Add 95% credible intervals
##D   polygon(
##D     c(1:length(theta_1_estimate), rev(1:length(theta_1_estimate))),
##D     c(theta_1_q025, rev(theta_1_q975)),
##D     col = rgb(0.7, 0.7, 0.7, alpha = 0.3),
##D     border = NA
##D   )
##D 
##D   # Add point estimate
##D   lines(theta_1_estimate, col = "black", lwd = 2)
##D 
##D   legend(
##D     "topright",
##D     legend = c(
##D       expression(theta["t,1"]),
##D       expression(hat(theta)["t,1"]),
##D       "95% Credible Interval"
##D     ),
##D     col = c("red", "black", "gray"),
##D     lty = c(2, 1, 1),
##D     lwd = c(3, 2, 8),
##D     bty = "n"
##D   )
##D 
##D   # --- 4. Success Probabilities (alpha[t]) ---
##D   # Plot true and estimated (median) success probabilities with uncertainty
##D   alpha_estimate <- apply(X = out$alpha, MARGIN = 2, FUN = median)
##D   alpha_q025 <- apply(X = out$alpha, MARGIN = 2, FUN = quantile, probs = 0.025)
##D   alpha_q975 <- apply(X = out$alpha, MARGIN = 2, FUN = quantile, probs = 0.975)
##D 
##D   # Calculate y-axis range for optimal legend positioning
##D   range_alpha <- range(alpha_true, alpha_estimate, alpha_q025, alpha_q975)
##D   r1_alpha <- max(0, range_alpha[1] - 0.05)  # Ensure lower bound is at least 0
##D   r2_alpha <- min(1, range_alpha[2] + 0.3 * diff(range_alpha))  # Ensure upper bound is at most 1
##D 
##D   plot(
##D     alpha_true,
##D     col = "red",
##D     type = "l",
##D     lwd = 3,
##D     xlab = "t",
##D     ylim = c(r1_alpha, r2_alpha),
##D     lty = 2,
##D     ylab = expression(alpha[t]),
##D     main = "Success probabilities"
##D   )
##D 
##D   # Add 95% credible intervals
##D   polygon(
##D     c(1:length(alpha_estimate), rev(1:length(alpha_estimate))),
##D     c(alpha_q025, rev(alpha_q975)),
##D     col = rgb(0.7, 0.7, 0.7, alpha = 0.3),
##D     border = NA
##D   )
##D 
##D   # Add point estimate
##D   lines(alpha_estimate, col = "black", lwd = 2)
##D 
##D   legend(
##D     "topright",
##D     legend = c(
##D       expression(alpha[t]),
##D       expression(hat(alpha)[t]),
##D       "95% Credible Interval"
##D     ),
##D     col = c("red", "black", "gray"),
##D     lty = c(2, 1, 1),
##D     lwd = c(3, 2, 8),
##D     bty = "n"
##D   )
##D 
##D   # --- 5. Initial State (theta[01]) Diagnostics ---
##D   # Trace plot for theta[01] to assess MCMC convergence
##D   range_theta_01 <- range(out$theta_01)
##D   r1_theta01 <- range_theta_01[1] - 0.1 * diff(range_theta_01)
##D   r2_theta01 <- range_theta_01[2] + 0.3 * diff(range_theta_01)
##D 
##D   plot.ts(
##D     out$theta_01,
##D     ylab = expression(theta["0,1"]),
##D     main = "Trace plot of initial state",
##D     xlab = "Iterations",
##D     col = "gray",
##D     ylim = c(r1_theta01, r2_theta01)
##D   )
##D   abline(
##D     h = c(theta0_true, median(out$theta_01)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(theta["0,1"]), expression(hat(theta)["0,1"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
##D 
##D   # Posterior density estimate for theta[01]
##D   range_dens_theta01 <- range(out$theta_01)
##D   r1_dens_theta01 <- range_dens_theta01[1] - 0.1 * diff(range_dens_theta01)
##D   r2_dens_theta01 <- range_dens_theta01[2] + 0.25 * diff(range_dens_theta01)
##D 
##D   plot(
##D     density(out$theta_01),
##D     main = "Posterior density estimate of initial state",
##D     xlab = expression(theta["0,1"]),
##D     ylab = "Density",
##D     lwd = 2,
##D     xlim = c(r1_dens_theta01, r2_dens_theta01)
##D   )
##D   abline(
##D     v = c(theta0_true, median(out$theta_01)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(theta["0,1"]), expression(hat(theta)["0,1"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
##D 
##D   # --- 6. Innovation Precision (1/W[1]) Diagnostics ---
##D   # Trace plot for 1/W[1] to assess parameter convergence
##D   range_prec_1 <- range(out$prec_1)
##D   r1_prec1 <- range_prec_1[1] - 0.1 * diff(range_prec_1)
##D   r2_prec1 <- range_prec_1[2] + 0.25 * diff(range_prec_1)
##D 
##D   plot.ts(
##D     out$prec_1,
##D     ylab = expression(1/W[1]),
##D     main = "Trace plot of innovation precision",
##D     xlab = "Iterations",
##D     col = "gray",
##D     ylim = c(r1_prec1, r2_prec1)
##D   )
##D   abline(
##D     h = c(prec1_true, median(out$prec_1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(W[1]^-1), expression(hat(W)[1]^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
##D 
##D   # Posterior density estimate for 1/W[1]
##D   range_dens_prec1 <- range(out$prec_1)
##D   r1_dens_prec1 <- range_dens_prec1[1] - 0.1 * diff(range_dens_prec1)
##D   r2_dens_prec1 <- range_dens_prec1[2] + 0.25 * diff(range_dens_prec1)
##D 
##D   plot(
##D     density(out$prec_1),
##D     main = "Posterior density estimate of innovation precision",
##D     xlab = expression(W[1]^-1),
##D     ylab = "Density",
##D     lwd = 2,
##D     xlim = c(r1_dens_prec1, r2_dens_prec1)
##D   )
##D   abline(
##D     v = c(prec1_true, median(out$prec_1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(W[1]^-1), expression(hat(W)[1]^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
## End(Not run)




cleanEx()
nameEx("mcmc_binomial_localtrend")
### * mcmc_binomial_localtrend

flush(stderr()); flush(stdout())

### Name: mcmc_binomial_localtrend
### Title: Gibbs Sampler for a Local-Trend Binomial Dynamic Model
### Aliases: mcmc_binomial_localtrend

### ** Examples

## Description
# This example demonstrates how to:
# 1. Simulate data from a local-trend binomial dynamic model
# 2. Use `mcmc_binomial_localtrend` to estimate parameters and latent states
# 3. Perform a detailed posterior analysis with visualizations
# 4. Set a seed for reproducibility

## Simulation of data
n <- 500        # Number of observations to simulate
n_trials <- 20  # Number of binomial trials

# True parameters for simulation:
theta01_true <- 0.5     # Initial level state (theta[0,1]) on logit scale
theta02_true <- 0.01    # Initial trend state (theta[0,2]) on logit scale
prec1_true <- 100       # Level innovation precision (1/W[1])
prec2_true <- 400       # Trend innovation precision (1/W[2])

# Use a fixed seed for data simulation
set.seed(123)

# Generate noise terms:
u1 <- rnorm(n, sd = sqrt(1/prec1_true))  # Level evolution noise (u1[t])
u2 <- rnorm(n, sd = sqrt(1/prec2_true))  # Trend evolution noise (u2[t])

# Simulate latent states and observations:
theta1_true <- numeric(n)
theta2_true <- numeric(n)
theta2_true[1] <- theta02_true + u2[1]
theta1_true[1] <- theta01_true + theta02_true + u1[1]
for (t in 2:n) {
  theta2_true[t] <- theta2_true[t-1] + u2[t]
  theta1_true[t] <- theta1_true[t-1] + theta2_true[t-1] + u1[t]
}
alpha_true <- plogis(theta1_true)  # Success probabilities
y <- rbinom(n, size = n_trials, prob = alpha_true)  # Observed binomial counts

## Running the Gibbs sampler
# Run the Gibbs sampler with specified priors and a seed
out <- mcmc_binomial_localtrend(
  y,
  n_trials             = n_trials,
  burnin               = 1000,
  thinning             = 50,
  n_chain              = 1000,
  prior_theta01_mean   = 0,
  prior_theta01_prec   = 1,
  prior_theta02_mean   = 0,
  prior_theta02_prec   = 1,
  prior_prec1_shape    = 100,
  prior_prec1_rate     = 1,
  prior_prec2_shape    = 400,
  prior_prec2_rate     = 1,
  lag_update           = 50,
  max_step_size        = 0.1,
  base_adaptation_rate = 1,
  decay_exponent       = 0.6,
  target_acceptance    = 0.44,
  return_log_sigma     = FALSE,
  return_accrate       = TRUE,
  seed                 = 456
)

## Posterior analysis and visualization
# The following plots show how to analyze the posterior distributions.
# Point estimates are based on the median of posterior samples.
## Not run: 
##D   # --- 0. Plot the simulated data ---
##D   plot(
##D     y,
##D     main = "Simulated binomial counts",
##D     ylab = expression(y[t]),
##D     xlab = "t",
##D     type = "o",
##D     pch = 16,
##D     cex = 0.7
##D   )
##D 
##D   # --- 1. Metropolis-Hastings Acceptance Rate Diagnostics ---
##D   # Extract acceptance rate statistics for adaptive MCMC performance evaluation
##D   acc <- out$accrate
##D   min_acc <- apply(X = acc, MARGIN = 2, FUN = min)
##D   max_acc <- apply(X = acc, MARGIN = 2, FUN = max)
##D   med_acc <- apply(X = acc, MARGIN = 2, FUN = median)
##D 
##D   # Calculate y-axis range for optimal legend positioning
##D   range_acc <- range(min_acc, max_acc)
##D   r1_acc <- range_acc[1] - 0.05
##D   r2_acc <- range_acc[2] + 0.15 * diff(range_acc)
##D 
##D   # Plot acceptance rates with target reference line and confidence bands
##D   plot(
##D     med_acc,
##D     type = "l",
##D     col = "black",
##D     lwd = 2,
##D     xlab = "Time (t)",
##D     ylab = "Acceptance Rate",
##D     main = "Metropolis-Hastings Acceptance Rates",
##D     ylim = c(r1_acc, r2_acc)
##D   )
##D 
##D   # Add confidence bands showing min-max range across MCMC iterations
##D   polygon(
##D     c(1:length(med_acc), rev(1:length(med_acc))),
##D     c(min_acc, rev(max_acc)),
##D     col = rgb(0.7, 0.7, 0.7, alpha = 0.3),
##D     border = NA
##D   )
##D 
##D   # Overlay target acceptance rate
##D   abline(h = 0.44, col = "red", lty = 2, lwd = 2)
##D 
##D   # Add informative legend positioned in the expanded y-range
##D   legend(
##D     "topright",
##D     legend = c("Median acceptance rate", "Min-Max range", "Target rate (0.44)"),
##D     col = c("black", "gray", "red"),
##D     lty = c(1, 1, 2),
##D     lwd = c(2, 8, 2),
##D     bty = "n"
##D   )
##D 
##D   # --- 2. Latent Level Trajectories (theta[t,1]) on logit scale ---
##D   # Visualize uncertainty by plotting multiple posterior trajectory samples
##D   num_traj_to_plot <- 20
##D 
##D   # Calculate y-axis range for optimal legend positioning
##D   range_traj <- range(out$theta_1[1:num_traj_to_plot, ], theta1_true)
##D   r1_traj <- range_traj[1] - 0.1 * diff(range_traj)
##D   r2_traj <- range_traj[2] + 0.1 * diff(range_traj)
##D 
##D   matplot(
##D     t(out$theta_1[1:num_traj_to_plot, ]),
##D     type = "l",
##D     lty = 1,
##D     col = grDevices::rainbow(num_traj_to_plot, alpha = 0.3),
##D     xlab = "t",
##D     ylab = expression(theta["t,1"]),
##D     main = "Posterior trajectory samples for latent level (logit scale)",
##D     ylim = c(r1_traj, r2_traj)
##D   )
##D   # Overlay the true trajectory
##D   lines(theta1_true, col = "black", lwd = 3, lty = 2)
##D   legend(
##D     "topright",
##D     legend = expression(theta["t,1"]),
##D     col = "black",
##D     lty = 2,
##D     lwd = 3,
##D     bty = "n"
##D   )
##D 
##D   # --- 3. Latent Trend Trajectories (theta[t,2]) on logit scale ---
##D   # Visualize uncertainty by plotting multiple posterior trajectory samples
##D   range_traj2 <- range(out$theta_2[1:num_traj_to_plot, ], theta2_true)
##D   r1_traj2 <- range_traj2[1] - 0.1 * diff(range_traj2)
##D   r2_traj2 <- range_traj2[2] + 0.1 * diff(range_traj2)
##D 
##D   matplot(
##D     t(out$theta_2[1:num_traj_to_plot, ]),
##D     type = "l",
##D     lty = 1,
##D     col = grDevices::rainbow(num_traj_to_plot, alpha = 0.3),
##D     xlab = "t",
##D     ylab = expression(theta["t,2"]),
##D     main = "Posterior trajectory samples for latent trend (logit scale)",
##D     ylim = c(r1_traj2, r2_traj2)
##D   )
##D   # Overlay the true trajectory
##D   lines(theta2_true, col = "black", lwd = 3, lty = 2)
##D   legend(
##D     "topright",
##D     legend = expression(theta["t,2"]),
##D     col = "black",
##D     lty = 2,
##D     lwd = 3,
##D     bty = "n"
##D   )
##D 
##D   # --- 4. Latent Level Point Estimates (theta[t,1]) ---
##D   # Plot true and estimated (median) latent level with credible intervals
##D   theta_1_estimate <- apply(X = out$theta_1, MARGIN = 2, FUN = median)
##D   theta_1_q025 <- apply(X = out$theta_1, MARGIN = 2, FUN = quantile, probs = 0.025)
##D   theta_1_q975 <- apply(X = out$theta_1, MARGIN = 2, FUN = quantile, probs = 0.975)
##D 
##D   range_theta_1 <- range(theta_1_estimate, theta1_true, theta_1_q025, theta_1_q975)
##D   r1_theta1 <- range_theta_1[1] - 0.1 * diff(range_theta_1)
##D   r2_theta1 <- range_theta_1[2] + 0.3 * diff(range_theta_1)
##D 
##D   plot(
##D     theta1_true,
##D     col = "red",
##D     type = "l",
##D     lwd = 3,
##D     xlab = "t",
##D     ylim = c(r1_theta1, r2_theta1),
##D     lty = 2,
##D     ylab = expression(theta["t,1"]),
##D     main = "Latent level estimation (logit scale)"
##D   )
##D 
##D   # Add 95% credible intervals
##D   polygon(
##D     c(1:length(theta_1_estimate), rev(1:length(theta_1_estimate))),
##D     c(theta_1_q025, rev(theta_1_q975)),
##D     col = rgb(0.7, 0.7, 0.7, alpha = 0.3),
##D     border = NA
##D   )
##D 
##D   # Add point estimate
##D   lines(theta_1_estimate, col = "black", lwd = 2)
##D 
##D   legend(
##D     "topright",
##D     legend = c(
##D       expression(theta["t,1"]),
##D       expression(hat(theta)["t,1"]),
##D       "95% Credible Interval"
##D     ),
##D     col = c("red", "black", "gray"),
##D     lty = c(2, 1, 1),
##D     lwd = c(3, 2, 8),
##D     bty = "n"
##D   )
##D 
##D   # --- 5. Latent Trend Point Estimates (theta[t,2]) ---
##D   # Plot true and estimated (median) latent trend with credible intervals
##D   theta_2_estimate <- apply(X = out$theta_2, MARGIN = 2, FUN = median)
##D   theta_2_q025 <- apply(X = out$theta_2, MARGIN = 2, FUN = quantile, probs = 0.025)
##D   theta_2_q975 <- apply(X = out$theta_2, MARGIN = 2, FUN = quantile, probs = 0.975)
##D 
##D   range_theta_2 <- range(theta_2_estimate, theta2_true, theta_2_q025, theta_2_q975)
##D   r1_theta2 <- range_theta_2[1] - 0.1 * diff(range_theta_2)
##D   r2_theta2 <- range_theta_2[2] + 0.3 * diff(range_theta_2)
##D 
##D   plot(
##D     theta2_true,
##D     col = "red",
##D     type = "l",
##D     lwd = 3,
##D     xlab = "t",
##D     ylim = c(r1_theta2, r2_theta2),
##D     lty = 2,
##D     ylab = expression(theta["t,2"]),
##D     main = "Latent trend estimation (logit scale)"
##D   )
##D 
##D   # Add 95% credible intervals
##D   polygon(
##D     c(1:length(theta_2_estimate), rev(1:length(theta_2_estimate))),
##D     c(theta_2_q025, rev(theta_2_q975)),
##D     col = rgb(0.7, 0.7, 0.7, alpha = 0.3),
##D     border = NA
##D   )
##D 
##D   # Add point estimate
##D   lines(theta_2_estimate, col = "black", lwd = 2)
##D 
##D   legend(
##D     "topright",
##D     legend = c(
##D       expression(theta["t,2"]),
##D       expression(hat(theta)["t,2"]),
##D       "95% Credible Interval"
##D     ),
##D     col = c("red", "black", "gray"),
##D     lty = c(2, 1, 1),
##D     lwd = c(3, 2, 8),
##D     bty = "n"
##D   )
##D 
##D   # --- 6. Success Probabilities (alpha[t]) ---
##D   # Plot true and estimated (median) success probabilities with uncertainty
##D   alpha_estimate <- apply(X = out$alpha, MARGIN = 2, FUN = median)
##D   alpha_q025 <- apply(X = out$alpha, MARGIN = 2, FUN = quantile, probs = 0.025)
##D   alpha_q975 <- apply(X = out$alpha, MARGIN = 2, FUN = quantile, probs = 0.975)
##D 
##D   # Calculate y-axis range for optimal legend positioning
##D   range_alpha <- range(alpha_true, alpha_estimate, alpha_q025, alpha_q975)
##D   r1_alpha <- max(0, range_alpha[1] - 0.05)  # Ensure lower bound is at least 0
##D   r2_alpha <- min(1, range_alpha[2] + 0.3 * diff(range_alpha))  # Ensure upper bound is at most 1
##D 
##D   plot(
##D     alpha_true,
##D     col = "red",
##D     type = "l",
##D     lwd = 3,
##D     xlab = "t",
##D     ylim = c(r1_alpha, r2_alpha),
##D     lty = 2,
##D     ylab = expression(alpha[t]),
##D     main = "Success probabilities"
##D   )
##D 
##D   # Add 95% credible intervals
##D   polygon(
##D     c(1:length(alpha_estimate), rev(1:length(alpha_estimate))),
##D     c(alpha_q025, rev(alpha_q975)),
##D     col = rgb(0.7, 0.7, 0.7, alpha = 0.3),
##D     border = NA
##D   )
##D 
##D   # Add point estimate
##D   lines(alpha_estimate, col = "black", lwd = 2)
##D 
##D   legend(
##D     "topright",
##D     legend = c(
##D       expression(alpha[t]),
##D       expression(hat(alpha)[t]),
##D       "95% Credible Interval"
##D     ),
##D     col = c("red", "black", "gray"),
##D     lty = c(2, 1, 1),
##D     lwd = c(3, 2, 8),
##D     bty = "n"
##D   )
##D 
##D   # --- 7. Initial Level State (theta[0,1]) Diagnostics ---
##D   # Trace plot for theta[0,1] to assess MCMC convergence
##D   range_theta_01 <- range(out$theta_01)
##D   r1_theta01 <- range_theta_01[1] - 0.1 * diff(range_theta_01)
##D   r2_theta01 <- range_theta_01[2] + 0.3 * diff(range_theta_01)
##D 
##D   plot.ts(
##D     out$theta_01,
##D     ylab = expression(theta["0,1"]),
##D     main = "Trace plot of initial level state",
##D     xlab = "Iterations",
##D     col = "gray",
##D     ylim = c(r1_theta01, r2_theta01)
##D   )
##D   abline(
##D     h = c(theta01_true, median(out$theta_01)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(theta["0,1"]), expression(hat(theta)["0,1"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
##D 
##D   # Posterior density estimate for theta[0,1]
##D   range_dens_theta01 <- range(out$theta_01)
##D   r1_dens_theta01 <- range_dens_theta01[1] - 0.1 * diff(range_dens_theta01)
##D   r2_dens_theta01 <- range_dens_theta01[2] + 0.25 * diff(range_dens_theta01)
##D 
##D   plot(
##D     density(out$theta_01),
##D     main = "Posterior density estimate of initial level state",
##D     xlab = expression(theta["0,1"]),
##D     ylab = "Density",
##D     lwd = 2,
##D     xlim = c(r1_dens_theta01, r2_dens_theta01)
##D   )
##D   abline(
##D     v = c(theta01_true, median(out$theta_01)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(theta["0,1"]), expression(hat(theta)["0,1"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
##D 
##D   # --- 8. Initial Trend State (theta[0,2]) Diagnostics ---
##D   # Trace plot for theta[0,2] to assess MCMC convergence
##D   range_theta_02 <- range(out$theta_02)
##D   r1_theta02 <- range_theta_02[1] - 0.1 * diff(range_theta_02)
##D   r2_theta02 <- range_theta_02[2] + 0.3 * diff(range_theta_02)
##D 
##D   plot.ts(
##D     out$theta_02,
##D     ylab = expression(theta["0,2"]),
##D     main = "Trace plot of initial trend state",
##D     xlab = "Iterations",
##D     col = "gray",
##D     ylim = c(r1_theta02, r2_theta02)
##D   )
##D   abline(
##D     h = c(theta02_true, median(out$theta_02)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(theta["0,2"]), expression(hat(theta)["0,2"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
##D 
##D   # Posterior density estimate for theta[0,2]
##D   range_dens_theta02 <- range(out$theta_02)
##D   r1_dens_theta02 <- range_dens_theta02[1] - 0.1 * diff(range_dens_theta02)
##D   r2_dens_theta02 <- range_dens_theta02[2] + 0.25 * diff(range_dens_theta02)
##D 
##D   plot(
##D     density(out$theta_02),
##D     main = "Posterior density estimate of initial trend state",
##D     xlab = expression(theta["0,2"]),
##D     ylab = "Density",
##D     lwd = 2,
##D     xlim = c(r1_dens_theta02, r2_dens_theta02)
##D   )
##D   abline(
##D     v = c(theta02_true, median(out$theta_02)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(theta["0,2"]), expression(hat(theta)["0,2"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
##D 
##D   # --- 9. Level Innovation Precision (1/W[1]) Diagnostics ---
##D   # Trace plot for 1/W[1] to assess parameter convergence
##D   range_prec_1 <- range(out$prec_1)
##D   r1_prec1 <- range_prec_1[1] - 0.1 * diff(range_prec_1)
##D   r2_prec1 <- range_prec_1[2] + 0.25 * diff(range_prec_1)
##D 
##D   plot.ts(
##D     out$prec_1,
##D     ylab = expression(1/W[1]),
##D     main = "Trace plot of level innovation precision",
##D     xlab = "Iterations",
##D     col = "gray",
##D     ylim = c(r1_prec1, r2_prec1)
##D   )
##D   abline(
##D     h = c(prec1_true, median(out$prec_1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(W[1]^-1), expression(hat(W)[1]^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
##D 
##D   # Posterior density estimate for 1/W[1]
##D   range_dens_prec1 <- range(out$prec_1)
##D   r1_dens_prec1 <- range_dens_prec1[1] - 0.1 * diff(range_dens_prec1)
##D   r2_dens_prec1 <- range_dens_prec1[2] + 0.25 * diff(range_dens_prec1)
##D 
##D   plot(
##D     density(out$prec_1),
##D     main = "Posterior density estimate of level innovation precision",
##D     xlab = expression(W[1]^-1),
##D     ylab = "Density",
##D     lwd = 2,
##D     xlim = c(r1_dens_prec1, r2_dens_prec1)
##D   )
##D   abline(
##D     v = c(prec1_true, median(out$prec_1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(W[1]^-1), expression(hat(W)[1]^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
##D 
##D   # --- 10. Trend Innovation Precision (1/W[2]) Diagnostics ---
##D   # Trace plot for 1/W[2] to assess parameter convergence
##D   range_prec_2 <- range(out$prec_2)
##D   r1_prec2 <- range_prec_2[1] - 0.1 * diff(range_prec_2)
##D   r2_prec2 <- range_prec_2[2] + 0.25 * diff(range_prec_2)
##D 
##D   plot.ts(
##D     out$prec_2,
##D     ylab = expression(1/W[2]),
##D     main = "Trace plot of trend innovation precision",
##D     xlab = "Iterations",
##D     col = "gray",
##D     ylim = c(r1_prec2, r2_prec2)
##D   )
##D   abline(
##D     h = c(prec2_true, median(out$prec_2)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(W[2]^-1), expression(hat(W)[2]^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
##D 
##D   # Posterior density estimate for 1/W[2]
##D   range_dens_prec2 <- range(out$prec_2)
##D   r1_dens_prec2 <- range_dens_prec2[1] - 0.1 * diff(range_dens_prec2)
##D   r2_dens_prec2 <- range_dens_prec2[2] + 0.25 * diff(range_dens_prec2)
##D 
##D   plot(
##D     density(out$prec_2),
##D     main = "Posterior density estimate of trend innovation precision",
##D     xlab = expression(W[2]^-1),
##D     ylab = "Density",
##D     lwd = 2,
##D     xlim = c(r1_dens_prec2, r2_dens_prec2)
##D   )
##D   abline(
##D     v = c(prec2_true, median(out$prec_2)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(W[2]^-1), expression(hat(W)[2]^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
## End(Not run)




cleanEx()
nameEx("mcmc_localacceleration")
### * mcmc_localacceleration

flush(stderr()); flush(stdout())

### Name: mcmc_localacceleration
### Title: Gibbs Sampler for a Local-Acceleration Dynamic Model
### Aliases: mcmc_localacceleration

### ** Examples

## Description
# This example demonstrates how to:
# 1. Simulate data from a local acceleration dynamic model
# 2. Use `mcmc_localacceleration` to estimate parameters and latent states
# 3. Perform a detailed posterior analysis with visualizations
# 4. Set a seed for reproducibility

## Simulation of data
n <- 1000 # Number of observations to simulate

# True parameters for simulation:
theta01_true <- 10         # Initial level (theta[0,1])
theta02_true <- 0.5        # Initial trend (theta[0,2])
theta03_true <- 0.01       # Initial acceleration (theta[0,3])
prec1_true   <- 1 / 0.100  # Level innovation precision (1/W[1])
prec2_true   <- 1 / 0.010  # Trend innovation precision (1/W[2])
prec3_true   <- 1 / 0.001  # Acceleration innovation precision (1/W[3])
prec_y_true  <- 1 / 1.000  # Observation precision (1/V)

# Use a fixed seed for data simulation
set.seed(123)

# Generate noise terms:
u1      <- rnorm(n, sd = sqrt(1 / prec1_true))  # Level noise
u2      <- rnorm(n, sd = sqrt(1 / prec2_true))  # Trend noise
u3      <- rnorm(n, sd = sqrt(1 / prec3_true))  # Acceleration noise
epsilon <- rnorm(n, sd = sqrt(1 / prec_y_true)) # Observation noise

# Simulate latent states and observations:
theta1_true    <- numeric(n)
theta2_true    <- numeric(n)
theta3_true    <- numeric(n)
theta3_true[1] <- theta03_true + u3[1]
theta2_true[1] <- theta02_true + theta03_true + u2[1]
theta1_true[1] <- theta01_true + theta02_true + u1[1]
for (t in 2:n) {
  theta3_true[t] <- theta3_true[t-1] + u3[t]
  theta2_true[t] <- theta2_true[t-1] + theta3_true[t-1] + u2[t]
  theta1_true[t] <- theta1_true[t-1] + theta2_true[t-1] + u1[t]
}
y <- theta1_true + epsilon # Observed data

## Running the Gibbs sampler
# Run the Gibbs sampler with specified priors and a seed
out <- mcmc_localacceleration(
  y,
  burnin   = 2000,
  thinning = 100,
  n_chain  = 1000,
  prior_theta01_mean = y[1],
  prior_theta01_prec = 1 / var(y),
  prior_theta02_mean = y[1] / 2,
  prior_theta02_prec = 1 / var(y),
  prior_theta03_mean = 0,
  prior_theta03_prec = 1e-3,
  prior_prec1_shape  = 1e-1,
  prior_prec1_rate   = 1e-1,
  prior_prec2_shape  = 1e-2,
  prior_prec2_rate   = 1e-2,
  prior_prec3_shape  = 1e-1,
  prior_prec3_rate   = 1e-2,
  prior_prec_y_shape = 1e-1,
  prior_prec_y_rate  = 1e-1,
  seed = 456
)

## Posterior analysis and visualization
# The following plots show how to analyze the posterior distributions.
# Point estimates are based on the median of posterior samples.
## Not run: 
##D   # --- 0. Plot the simulated data ---
##D   plot.ts(
##D     y,
##D     main = "Simulated data",
##D     ylab = expression(y[t]),
##D     xlab = "t"
##D   )
##D 
##D   # --- 1. Latent Level (theta[t,1]) ---
##D 
##D   # Visualize trajectories from the first few posterior samples
##D   num_traj_to_plot <- 20
##D   matplot(
##D     t(out$theta_1[1:num_traj_to_plot, ]),
##D     type = "l",
##D     lty = 1,
##D     col = grDevices::rainbow(num_traj_to_plot, alpha = 0.5),
##D     xlab = "t",
##D     ylab = expression(theta["t,1"]),
##D     main = "Sampled trajectories for latent level"
##D   )
##D 
##D   # Plot true and estimated (median) latent level
##D   theta_1_estimate <- apply(X = out$theta_1, MARGIN = 2, FUN = median)
##D   range_theta_1 <- range(theta_1_estimate, theta1_true)
##D   r1_theta1 <- range_theta_1[1]
##D   r2_theta1 <- range_theta_1[2] + 0.25 * diff(range_theta_1)
##D 
##D   plot.ts(
##D     theta1_true,
##D     col = "red",
##D     type = "l",
##D     xlab = "t",
##D     ylim = c(r1_theta1, r2_theta1),
##D     lty = 2,
##D     ylab = expression(theta["t,1"]),
##D     main = "Latent level"
##D   )
##D   points(theta_1_estimate, type = "l")
##D   legend(
##D     "topright",
##D     legend = c(expression(theta["t,1"]), expression(hat(theta)["t,1"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n"
##D   )
##D 
##D   # --- 2. Latent Trend (theta[t,2]) ---
##D   theta_2_estimate <- apply(X = out$theta_2, MARGIN = 2, FUN = median)
##D   range_theta_2 <- range(theta_2_estimate, theta2_true)
##D   r1_theta2 <- range_theta_2[1]
##D   r2_theta2 <- range_theta_2[2] + 0.25 * diff(range_theta_2)
##D 
##D   plot.ts(
##D     theta2_true,
##D     col = "red",
##D     type = "l",
##D     xlab = "t",
##D     ylim = c(r1_theta2, r2_theta2),
##D     lty = 2,
##D     ylab = expression(theta["t,2"]),
##D     main = "Latent trend"
##D   )
##D   points(theta_2_estimate, type = "l")
##D   legend(
##D     "topright",
##D     legend = c(expression(theta["t,2"]), expression(hat(theta)["t,2"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n"
##D   )
##D 
##D   # --- 3. Latent Acceleration (theta[t,3]) ---
##D   theta_3_estimate <- apply(X = out$theta_3, MARGIN = 2, FUN = median)
##D   range_theta_3 <- range(theta_3_estimate, theta3_true)
##D   r1_theta3 <- range_theta_3[1]
##D   r2_theta3 <- range_theta_3[2] + 0.25 * diff(range_theta_3)
##D 
##D   plot.ts(
##D     theta3_true,
##D     col = "red",
##D     type = "l",
##D     xlab = "t",
##D     ylim = c(r1_theta3, r2_theta3),
##D     lty = 2,
##D     ylab = expression(theta["t,3"]),
##D     main = "Latent acceleration"
##D   )
##D   points(theta_3_estimate, type = "l")
##D   legend(
##D     "topright",
##D     legend = c(expression(theta["t,3"]), expression(hat(theta)["t,3"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n"
##D   )
##D 
##D   # --- 4. Initial Level (theta[0,1]) ---
##D   # Trace plot
##D   range_theta_01 <- range(out$theta_01, theta01_true)
##D   r1_theta01 <- range_theta_01[1]
##D   r2_theta01 <- range_theta_01[2] + 0.25 * diff(range_theta_01)
##D 
##D   plot.ts(
##D     out$theta_01,
##D     col = "gray",
##D     xlab = "Iterations",
##D     ylab = expression(theta["0,1"]),
##D     main = "Trace Plot of Initial Level",
##D     ylim = c(r1_theta01, r2_theta01)
##D   )
##D   abline(
##D     h = c(theta01_true, median(out$theta_01)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     bty = "n",
##D     legend = c(expression(theta["0,1"]), expression(hat(theta)["0,1"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D 
##D   # Density plot
##D   plot(
##D     density(out$theta_01),
##D     main = "Posterior Density of Initial Level",
##D     xlab = expression(theta["0,1"])
##D   )
##D   abline(
##D     v = c(theta01_true, median(out$theta_01)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     bty = "n",
##D     legend = c(expression(theta["0,1"]), expression(hat(theta)["0,1"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D 
##D   # --- 5. Initial Trend (theta[0,2]) ---
##D   # Trace plot
##D   range_theta_02 <- range(out$theta_02, theta02_true)
##D   r1_theta02 <- range_theta_02[1]
##D   r2_theta02 <- range_theta_02[2] + 0.25 * diff(range_theta_02)
##D 
##D   plot.ts(
##D     out$theta_02,
##D     col = "gray",
##D     xlab = "Iterations",
##D     ylab = expression(theta["0,2"]),
##D     main = "Trace Plot of Initial Trend",
##D     ylim = c(r1_theta02, r2_theta02)
##D   )
##D   abline(
##D     h = c(theta02_true, median(out$theta_02)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     bty = "n",
##D     legend = c(expression(theta["0,2"]), expression(hat(theta)["0,2"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D 
##D   # Density plot
##D   plot(
##D     density(out$theta_02),
##D     main = "Posterior Density of Initial Trend",
##D     xlab = expression(theta["0,2"])
##D   )
##D   abline(
##D     v = c(theta02_true, median(out$theta_02)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     bty = "n",
##D     legend = c(expression(theta["0,2"]), expression(hat(theta)["0,2"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D 
##D   # --- 6. Initial Acceleration (theta[0,3]) ---
##D   # Trace plot
##D   range_theta_03 <- range(out$theta_03, theta03_true)
##D   r1_theta03 <- range_theta_03[1]
##D   r2_theta03 <- range_theta_03[2] + 0.25 * diff(range_theta_03)
##D 
##D   plot.ts(
##D     out$theta_03,
##D     col = "gray",
##D     xlab = "Iterations",
##D     ylab = expression(theta["0,3"]),
##D     main = "Trace Plot of Initial Acceleration",
##D     ylim = c(r1_theta03, r2_theta03)
##D   )
##D   abline(
##D     h = c(theta03_true, median(out$theta_03)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     bty = "n",
##D     legend = c(expression(theta["0,3"]), expression(hat(theta)["0,3"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D 
##D   # Density plot
##D   plot(
##D     density(out$theta_03),
##D     main = "Posterior Density of Initial Acceleration",
##D     xlab = expression(theta["0,3"])
##D   )
##D   abline(
##D     v = c(theta03_true, median(out$theta_03)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     bty = "n",
##D     legend = c(expression(theta["0,3"]), expression(hat(theta)["0,3"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D 
##D   # --- 7. Level Precision (1/W_1) ---
##D   # Trace plot
##D   range_prec_1 <- range(out$prec_1, prec1_true)
##D   r1_prec1 <- range_prec_1[1]
##D   r2_prec1 <- range_prec_1[2] + 0.25 * diff(range_prec_1)
##D 
##D   plot.ts(
##D     out$prec_1,
##D     col = "gray",
##D     xlab = "Iterations",
##D     ylab = expression(1/W[1]),
##D     main = "Trace Plot of Level Precision",
##D     ylim = c(r1_prec1, r2_prec1)
##D   )
##D   abline(
##D     h = c(prec1_true, median(out$prec_1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     bty = "n",
##D     legend = c(expression(W[1]^-1), expression(hat(W)[1]^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D 
##D   # Density plot
##D   plot(
##D     density(out$prec_1),
##D     main = "Posterior Density of Level Precision",
##D     xlab = expression(W[1]^-1)
##D   )
##D   abline(
##D     v = c(prec1_true, median(out$prec_1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     bty = "n",
##D     legend = c(expression(W[1]^-1), expression(hat(W)[1]^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D 
##D   # --- 8. Trend Precision (1/W_2) ---
##D   # Trace plot
##D   range_prec_2 <- range(out$prec_2, prec2_true)
##D   r1_prec2 <- range_prec_2[1]
##D   r2_prec2 <- range_prec_2[2] + 0.25 * diff(range_prec_2)
##D 
##D   plot.ts(
##D     out$prec_2,
##D     col = "gray",
##D     xlab = "Iterations",
##D     ylab = expression(1/W[2]),
##D     main = "Trace Plot of Trend Precision",
##D     ylim = c(r1_prec2, r2_prec2)
##D   )
##D   abline(
##D     h = c(prec2_true, median(out$prec_2)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     bty = "n",
##D     legend = c(expression(W[2]^-1), expression(hat(W)[2]^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D 
##D   # Density plot
##D   plot(
##D     density(out$prec_2),
##D     main = "Posterior Density of Trend Precision",
##D     xlab = expression(W[2]^-1)
##D   )
##D   abline(
##D     v = c(prec2_true, median(out$prec_2)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     bty = "n",
##D     legend = c(expression(W[2]^-1), expression(hat(W)[2]^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D 
##D   # --- 9. Acceleration Precision (1/W_3) ---
##D   # Trace plot
##D   range_prec_3 <- range(out$prec_3, prec3_true)
##D   r1_prec3 <- range_prec_3[1]
##D   r2_prec3 <- range_prec_3[2] + 0.25 * diff(range_prec_3)
##D 
##D   plot.ts(
##D     out$prec_3,
##D     col = "gray",
##D     xlab = "Iterations",
##D     ylab = expression(1/W[3]),
##D     main = "Trace Plot of Acceleration Precision",
##D     ylim = c(r1_prec3, r2_prec3)
##D   )
##D   abline(
##D     h = c(prec3_true, median(out$prec_3)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     bty = "n",
##D     legend = c(expression(W[3]^-1), expression(hat(W)[3]^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D 
##D   # Density plot
##D   plot(
##D     density(out$prec_3),
##D     main = "Posterior Density of Acceleration Precision",
##D     xlab = expression(W[3]^-1)
##D   )
##D   abline(
##D     v = c(prec3_true, median(out$prec_3)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     bty = "n",
##D     legend = c(expression(W[3]^-1), expression(hat(W)[3]^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D 
##D   # --- 10. Observation Precision (1/V) ---
##D   # Trace plot
##D   range_prec_y <- range(out$prec_y, prec_y_true)
##D   r1_prec_y <- range_prec_y[1]
##D   r2_prec_y <- range_prec_y[2] + 0.25 * diff(range_prec_y)
##D 
##D   plot.ts(
##D     out$prec_y,
##D     col = "gray",
##D     xlab = "Iterations",
##D     ylab = expression(1/V),
##D     main = "Trace Plot of Observation Precision",
##D     ylim = c(r1_prec_y, r2_prec_y)
##D   )
##D   abline(
##D     h = c(prec_y_true, median(out$prec_y)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     bty = "n",
##D     legend = c(expression(V^-1), expression(hat(V)^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D 
##D   # Density plot
##D   plot(
##D     density(out$prec_y),
##D     main = "Posterior Density of Observation Precision",
##D     xlab = expression(V^-1)
##D   )
##D   abline(
##D     v = c(prec_y_true, median(out$prec_y)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     bty = "n",
##D     legend = c(expression(V^-1), expression(hat(V)^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
## End(Not run)




cleanEx()
nameEx("mcmc_locallevel")
### * mcmc_locallevel

flush(stderr()); flush(stdout())

### Name: mcmc_locallevel
### Title: Gibbs Sampler for a Local-Level Dynamic Model
### Aliases: mcmc_locallevel

### ** Examples

## Description
# This example demonstrates how to:
# 1. Simulate data from a local-level dynamic model
# 2. Use `mcmc_locallevel` to estimate parameters and latent states
# 3. Perform a detailed posterior analysis with visualizations
# 4. Set a seed for reproducibility

## Simulation of data
n <- 1000  # Number of observations to simulate

# True parameters for simulation:
theta0_true <- 10  # Initial state (theta[01])
prec1_true <- 1    # Innovation precision (1/W[1])
prec_y_true <- 5   # Observation precision (1/V)

# Use a fixed seed for data simulation
set.seed(123)

# Generate noise terms:
u1 <- rnorm(n, sd = sqrt(1/prec1_true))  # Evolution noise (u1[t])
e  <- rnorm(n, sd = sqrt(1/prec_y_true)) # Observation noise (e[t])

# Simulate latent states and observations:
theta1_true <- cumsum(c(theta0_true, u1))[-1]  # theta[t1] series
y <- theta1_true + e                           # Observed data (y[t])

## Running the Gibbs sampler
# Run the Gibbs sampler with specified priors and a seed
out <- mcmc_locallevel(
  y,
  burnin   = 1000,
  thinning = 10,
  n_chain  = 1000,
  prior_theta01_mean = y[1],
  prior_theta01_prec = 1 / var(y),
  prior_prec1_shape  = 1e-2,
  prior_prec1_rate   = 1e-2,
  prior_prec_y_shape = 1e-2,
  prior_prec_y_rate  = 1e-2,
  seed = 456
)

## Posterior analysis and visualization
# The following plots show how to analyze the posterior distributions.
# Point estimates are based on the median of posterior samples.
## Not run: 
##D   # --- 0. Plot the simulated data ---
##D   plot.ts(
##D     y,
##D     main = "Simulated data",
##D     ylab = expression(y[t]),
##D     xlab = "t"
##D   )
##D 
##D   # --- 1. Latent State (theta[t1]) ---
##D 
##D   # Visualize trajectories from the first few posterior samples
##D   num_traj_to_plot <- 20
##D   matplot(
##D     t(out$theta_1[1:num_traj_to_plot, ]),
##D     type = "l",
##D     lty = 1,
##D     col = grDevices::rainbow(num_traj_to_plot, alpha = 0.5),
##D     xlab = "t",
##D     ylab = expression(theta["t,1"]),
##D     main = "Sampled trajectories for latent state"
##D   )
##D 
##D   # Plot true and estimated (median) latent state
##D   theta_1_estimate <- apply(X = out$theta_1, MARGIN = 2, FUN = median)
##D   range_theta_1 <- range(theta_1_estimate, theta1_true)
##D   r1_theta1 <- range_theta_1[1]
##D   r2_theta1 <- range_theta_1[2] + 0.25 * diff(range_theta_1)
##D 
##D   plot.ts(
##D     theta1_true,
##D     col = "red",
##D     type = "l",
##D     xlab = "t",
##D     ylim = c(r1_theta1, r2_theta1),
##D     lty = 2,
##D     ylab = expression(theta["t,1"]),
##D     main = "Latent state"
##D   )
##D   points(theta_1_estimate, type = "l")
##D   legend(
##D     "topright",
##D     legend = c(expression(theta["t,1"]), expression(hat(theta)["t,1"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n"
##D   )
##D 
##D   # --- 2. Initial State (theta[01]) ---
##D 
##D   # Trace plot for theta[01]
##D   range_theta_01 <- range(out$theta_01)
##D   r1_theta01 <- range_theta_01[1]
##D   r2_theta01 <- range_theta_01[2] + 0.25 * diff(range_theta_01)
##D 
##D   plot.ts(
##D     out$theta_01,
##D     ylab = expression(theta["0,1"]),
##D     main = "Trace plot of initial state",
##D     xlab = "Iterations",
##D     col = "gray",
##D     ylim = c(r1_theta01, r2_theta01)
##D   )
##D   abline(
##D     h = c(theta0_true, median(out$theta_01)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(theta["0,1"]), expression(hat(theta)["0,1"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
##D 
##D   # Density estimate for theta[01]
##D   plot(
##D     density(out$theta_01),
##D     main = "Posterior density estimate of initial state",
##D     xlab = expression(theta["0,1"]),
##D     ylab = "Density",
##D     lwd = 2
##D   )
##D   abline(
##D     v = c(theta0_true, median(out$theta_01)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(theta["0,1"]), expression(hat(theta)["0,1"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
##D 
##D   # --- 3. Evolution Precision (1/W[1]) ---
##D 
##D   # Traceplot for 1/W[1]
##D   range_prec_1 <- range(out$prec_1)
##D   r1_prec1 <- range_prec_1[1]
##D   r2_prec1 <- range_prec_1[2] + 0.25 * diff(range_prec_1)
##D 
##D   plot.ts(
##D     out$prec_1,
##D     ylab = expression(1/W[1]),
##D     main = "Trace plot of evolution precision",
##D     xlab = "Iterations",
##D     col = "gray",
##D     ylim = c(r1_prec1, r2_prec1)
##D   )
##D   abline(
##D     h = c(prec1_true, median(out$prec_1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(W[1]^-1), expression(hat(W)[1]^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
##D 
##D   # Density estimate for 1/W[1]
##D   plot(
##D     density(out$prec_1),
##D     main = "Posterior density estimate of evolution precision",
##D     xlab = expression(W[1]^-1),
##D     ylab = "Density",
##D     lwd = 2
##D   )
##D   abline(
##D     v = c(prec1_true, median(out$prec_1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(W[1]^-1), expression(hat(W)[1]^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
##D 
##D   # --- 4. Observation Precision (1/V) ---
##D 
##D   # Traceplot for 1/V
##D   range_prec_y <- range(out$prec_y)
##D   r1_precy <- range_prec_y[1]
##D   r2_precy <- range_prec_y[2] + 0.25 * diff(range_prec_y)
##D 
##D   plot.ts(
##D     out$prec_y,
##D     ylab = expression(1/V),
##D     main = "Trace plot of observation precision",
##D     xlab = "Iterations",
##D     col = "gray",
##D     ylim = c(r1_precy, r2_precy)
##D   )
##D   abline(
##D     h = c(prec_y_true, median(out$prec_y)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(V^-1), expression(hat(V)^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
##D 
##D   # Density estimate for 1/V
##D   plot(
##D     density(out$prec_y),
##D     main = "Posterior density estimate of observation precision",
##D     xlab = expression(V^-1),
##D     ylab = "Density",
##D     lwd = 2
##D   )
##D   abline(
##D     v = c(prec_y_true, median(out$prec_y)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     legend = c(expression(V^-1), expression(hat(V)^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n",
##D     lwd = 2
##D   )
## End(Not run)




cleanEx()
nameEx("mcmc_localtrend")
### * mcmc_localtrend

flush(stderr()); flush(stdout())

### Name: mcmc_localtrend
### Title: Gibbs Sampler for a Local-Trend Dynamic Model
### Aliases: mcmc_localtrend

### ** Examples

## Description
# This example demonstrates how to:
# 1. Simulate data from a local trend dynamic model
# 2. Use `mcmc_localtrend` to estimate parameters and latent states
# 3. Perform a detailed posterior analysis with visualizations
# 4. Set a seed for reproducibility

## Simulation of data
n <- 1000 # Number of observations to simulate

# True parameters for simulation:
theta01_true <- 10        # Initial level (theta[0,1])
theta02_true <- 0.5       # Initial trend (theta[0,2])
prec1_true   <- 1 / 0.10  # Level innovation precision (1/W[1])
prec2_true   <- 1 / 0.01  # Trend innovation precision (1/W[2])
prec_y_true  <- 1 / 1.00  # Observation precision (1/V)

# Use a fixed seed for data simulation
set.seed(123)

# Generate noise terms:
u1      <- rnorm(n, sd = sqrt(1 / prec1_true))  # Level noise
u2      <- rnorm(n, sd = sqrt(1 / prec2_true))  # Trend noise
epsilon <- rnorm(n, sd = sqrt(1 / prec_y_true)) # Observation noise

# Simulate latent states and observations:
theta1_true    <- numeric(n)
theta2_true    <- numeric(n)
theta2_true[1] <- theta02_true + u2[1]
theta1_true[1] <- theta01_true + theta02_true + u1[1]
for (t in 2:n) {
  theta2_true[t] <- theta2_true[t-1] + u2[t]
  theta1_true[t] <- theta1_true[t-1] + theta2_true[t-1] + u1[t]
}
y <- theta1_true + epsilon # Observed data

## Running the Gibbs sampler
# Run the Gibbs sampler with specified priors and a seed
out <- mcmc_localtrend(
  y,
  burnin   = 2000,
  thinning = 100,
  n_chain  = 1000,
  prior_theta01_mean = y[1] / 2,
  prior_theta01_prec = 1 / var(y),
  prior_theta02_mean = y[1] / 2,
  prior_theta02_prec = 1 / var(y),
  prior_prec1_shape  = 1e-1,
  prior_prec1_rate   = 1e-1,
  prior_prec2_shape  = 1e-2,
  prior_prec2_rate   = 1e-2,
  prior_prec_y_shape = 1e-1,
  prior_prec_y_rate  = 1e-1,
  seed = 456
)

## Posterior analysis and visualization
# The following plots show how to analyze the posterior distributions.
# Point estimates are based on the median of posterior samples.
## Not run: 
##D   # --- 0. Plot the simulated data ---
##D   plot.ts(
##D     y,
##D     main = "Simulated data",
##D     ylab = expression(y[t]),
##D     xlab = "t"
##D   )
##D 
##D   # --- 1. Latent Level (theta[t,1]) ---
##D 
##D   # Visualize trajectories from the first few posterior samples
##D   num_traj_to_plot <- 20
##D   matplot(
##D     t(out$theta_1[1:num_traj_to_plot, ]),
##D     type = "l",
##D     lty = 1,
##D     col = grDevices::rainbow(num_traj_to_plot, alpha = 0.5),
##D     xlab = "t",
##D     ylab = expression(theta["t,1"]),
##D     main = "Sampled trajectories for latent level"
##D   )
##D 
##D   # Plot true and estimated (median) latent level
##D   theta_1_estimate <- apply(X = out$theta_1, MARGIN = 2, FUN = median)
##D   range_theta_1 <- range(theta_1_estimate, theta1_true)
##D   r1_theta1 <- range_theta_1[1]
##D   r2_theta1 <- range_theta_1[2] + 0.25 * diff(range_theta_1)
##D 
##D   plot.ts(
##D     theta1_true,
##D     col = "red",
##D     type = "l",
##D     xlab = "t",
##D     ylim = c(r1_theta1, r2_theta1),
##D     lty = 2,
##D     ylab = expression(theta["t,1"]),
##D     main = "Latent level"
##D   )
##D   points(theta_1_estimate, type = "l")
##D   legend(
##D     "topright",
##D     legend = c(expression(theta["t,1"]), expression(hat(theta)["t,1"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n"
##D   )
##D 
##D   # --- 2. Latent Trend (theta[t,2]) ---
##D   theta_2_estimate <- apply(X = out$theta_2, MARGIN = 2, FUN = median)
##D   range_theta_2 <- range(theta_2_estimate, theta2_true)
##D   r1_theta2 <- range_theta_2[1]
##D   r2_theta2 <- range_theta_2[2] + 0.25 * diff(range_theta_2)
##D 
##D   plot.ts(
##D     theta2_true,
##D     col = "red",
##D     type = "l",
##D     xlab = "t",
##D     ylim = c(r1_theta2, r2_theta2),
##D     lty = 2,
##D     ylab = expression(theta["t,2"]),
##D     main = "Latent trend"
##D   )
##D   points(theta_2_estimate, type = "l")
##D   legend(
##D     "topright",
##D     legend = c(expression(theta["t,2"]), expression(hat(theta)["t,2"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     bty = "n"
##D   )
##D 
##D   # --- 3. Initial Level (theta[0,1]) ---
##D   # Trace plot
##D   range_theta_01 <- range(out$theta_01, theta01_true)
##D   r1_theta01 <- range_theta_01[1]
##D   r2_theta01 <- range_theta_01[2] + 0.25 * diff(range_theta_01)
##D 
##D   plot.ts(
##D     out$theta_01,
##D     col = "gray",
##D     xlab = "Iterations",
##D     ylab = expression(theta["0,1"]),
##D     main = "Trace Plot of Initial Level",
##D     ylim = c(r1_theta01, r2_theta01)
##D   )
##D   abline(
##D     h = c(theta01_true, median(out$theta_01)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     bty = "n",
##D     legend = c(expression(theta["0,1"]), expression(hat(theta)["0,1"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D 
##D   # Density plot
##D   plot(
##D     density(out$theta_01),
##D     main = "Posterior Density of Initial Level",
##D     xlab = expression(theta["0,1"])
##D   )
##D   abline(
##D     v = c(theta01_true, median(out$theta_01)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     bty = "n",
##D     legend = c(expression(theta["0,1"]), expression(hat(theta)["0,1"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D 
##D   # --- 4. Initial Trend (theta[0,2]) ---
##D   # Trace plot
##D   range_theta_02 <- range(out$theta_02, theta02_true)
##D   r1_theta02 <- range_theta_02[1]
##D   r2_theta02 <- range_theta_02[2] + 0.25 * diff(range_theta_02)
##D 
##D   plot.ts(
##D     out$theta_02,
##D     col = "gray",
##D     xlab = "Iterations",
##D     ylab = expression(theta["0,2"]),
##D     main = "Trace Plot of Initial Trend",
##D     ylim = c(r1_theta02, r2_theta02)
##D   )
##D   abline(
##D     h = c(theta02_true, median(out$theta_02)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     bty = "n",
##D     legend = c(expression(theta["0,2"]), expression(hat(theta)["0,2"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D 
##D   # Density plot
##D   plot(
##D     density(out$theta_02),
##D     main = "Posterior Density of Initial Trend",
##D     xlab = expression(theta["0,2"])
##D   )
##D   abline(
##D     v = c(theta02_true, median(out$theta_02)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     bty = "n",
##D     legend = c(expression(theta["0,2"]), expression(hat(theta)["0,2"])),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D 
##D   # --- 5. Level Precision (1/W_1) ---
##D   # Trace plot
##D   range_prec_1 <- range(out$prec_1, prec1_true)
##D   r1_prec1 <- range_prec_1[1]
##D   r2_prec1 <- range_prec_1[2] + 0.25 * diff(range_prec_1)
##D 
##D   plot.ts(
##D     out$prec_1,
##D     col = "gray",
##D     xlab = "Iterations",
##D     ylab = expression(1/W[1]),
##D     main = "Trace Plot of Level Precision",
##D     ylim = c(r1_prec1, r2_prec1)
##D   )
##D   abline(
##D     h = c(prec1_true, median(out$prec_1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     bty = "n",
##D     legend = c(expression(W[1]^-1), expression(hat(W)[1]^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D 
##D   # Density plot
##D   plot(
##D     density(out$prec_1),
##D     main = "Posterior Density of Level Precision",
##D     xlab = expression(W[1]^-1)
##D   )
##D   abline(
##D     v = c(prec1_true, median(out$prec_1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     bty = "n",
##D     legend = c(expression(W[1]^-1), expression(hat(W)[1]^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D 
##D   # --- 6. Trend Precision (1/W_2) ---
##D   # Trace plot
##D   range_prec_2 <- range(out$prec_2, prec2_true)
##D   r1_prec2 <- range_prec_2[1]
##D   r2_prec2 <- range_prec_2[2] + 0.25 * diff(range_prec_2)
##D 
##D   plot.ts(
##D     out$prec_2,
##D     col = "gray",
##D     xlab = "Iterations",
##D     ylab = expression(1/W[2]),
##D     main = "Trace Plot of Trend Precision",
##D     ylim = c(r1_prec2, r2_prec2)
##D   )
##D   abline(
##D     h = c(prec2_true, median(out$prec_2)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     bty = "n",
##D     legend = c(expression(W[2]^-1), expression(hat(W)[2]^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D 
##D   # Density plot
##D   plot(
##D     density(out$prec_2),
##D     main = "Posterior Density of Trend Precision",
##D     xlab = expression(W[2]^-1)
##D   )
##D   abline(
##D     v = c(prec2_true, median(out$prec_2)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     bty = "n",
##D     legend = c(expression(W[2]^-1), expression(hat(W)[2]^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D 
##D   # --- 7. Observation Precision (1/V) ---
##D   # Trace plot
##D   range_prec_y <- range(out$prec_y, prec_y_true)
##D   r1_prec_y <- range_prec_y[1]
##D   r2_prec_y <- range_prec_y[2] + 0.25 * diff(range_prec_y)
##D 
##D   plot.ts(
##D     out$prec_y,
##D     col = "gray",
##D     xlab = "Iterations",
##D     ylab = expression(1/V),
##D     main = "Trace Plot of Observation Precision",
##D     ylim = c(r1_prec_y, r2_prec_y)
##D   )
##D   abline(
##D     h = c(prec_y_true, median(out$prec_y)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     bty = "n",
##D     legend = c(expression(V^-1), expression(hat(V)^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D 
##D   # Density plot
##D   plot(
##D     density(out$prec_y),
##D     main = "Posterior Density of Observation Precision",
##D     xlab = expression(V^-1)
##D   )
##D   abline(
##D     v = c(prec_y_true, median(out$prec_y)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
##D   legend(
##D     "topright",
##D     bty = "n",
##D     legend = c(expression(V^-1), expression(hat(V)^-1)),
##D     col = c("red", "black"),
##D     lty = c(2, 1),
##D     lwd = 2
##D   )
## End(Not run)




### * <FOOTER>
###
cleanEx()
options(digits = 7L)
base::cat("Time elapsed: ", proc.time() - base::get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')
