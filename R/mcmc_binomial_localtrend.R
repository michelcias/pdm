#' @title Gibbs Sampler for a Local-Trend Binomial Dynamic Model
#'
#' @description Runs a Gibbs sampler for the local-trend binomial dynamic model
#'   with logit link.
#'
#' @details The model is defined as:
#' \deqn{
#' \begin{aligned}
#' y_t &\sim \text{Binomial}(n_{trials}, \alpha_t), \\
#' \alpha_t &= \text{logit}^{-1}(\theta_{t,1}), \\
#' \theta_{t,1} &= \theta_{t-1,1} + \theta_{t-1,2} + u_{t,1}, & u_{t,1} \sim N(0, W_1), \\
#' \theta_{t,2} &= \theta_{t-1,2} + u_{t,2},                  & u_{t,2} \sim N(0, W_2),
#' \end{aligned}
#' }
#' where \eqn{t = 1, 2, \ldots, n} and \eqn{n} is the number of observations.
#'
#' The logit link function is defined as
#' \eqn{\text{logit}^{-1}(x) = \frac{e^x}{1 + e^x}}.
#'
#' \strong{Prior Distributions:}
#'
#' The following conjugate and semi-conjugate prior distributions are employed:
#'
#' \emph{Initial States:}
#' \deqn{
#' \begin{aligned}
#' \theta_{0,1} &\sim N(\mu_{01}, \tau_{01}^{-1}), \\
#' \theta_{0,2} &\sim N(\mu_{02}, \tau_{02}^{-1}).
#' \end{aligned}
#' }
#'
#' \emph{Precision Parameters:}
#' \deqn{
#' \begin{aligned}
#' W_1^{-1} &\sim \text{Gamma}(\nu_1, \eta_1), \\
#' W_2^{-1} &\sim \text{Gamma}(\nu_2, \eta_2).
#' \end{aligned}
#' }
#'
#' The prior hyperparameters correspond to function arguments as follows:
#' \tabular{cc}{
#'   \strong{Hyperparameter} \tab \strong{Function Argument} \cr
#'   \eqn{\mu_{01}} \tab `prior_theta01_mean` \cr
#'   \eqn{\tau_{01}} \tab `prior_theta01_prec` \cr
#'   \eqn{\mu_{02}} \tab `prior_theta02_mean` \cr
#'   \eqn{\tau_{02}} \tab `prior_theta02_prec` \cr
#'   \eqn{\nu_1} \tab `prior_prec1_shape` \cr
#'   \eqn{\eta_1} \tab `prior_prec1_rate` \cr
#'   \eqn{\nu_2} \tab `prior_prec2_shape` \cr
#'   \eqn{\eta_2} \tab `prior_prec2_rate`
#' }
#'
#' Due to the non-linear observation model with logit link, the algorithm
#' employs component-wise Metropolis-Hastings for sampling the latent states
#' \eqn{\theta_{t,1}}, with adaptive proposal tuning based on acceptance proportions.
#' Innovation precisions are sampled from conjugate Gamma posteriors.
#'
#' **Version 1.2 Enhancement:**
#' This version introduces a configurable adaptation threshold parameter
#' `min_deviation_threshold` that controls the sensitivity of proposal variance
#' adjustments. The default value of `NULL` computes a practical threshold of
#' `1.0/lag_update`, which triggers adaptation when the observed acceptance
#' proportion deviates from the target by at least the amount corresponding
#' to one additional acceptance/rejection in the sliding window.
#'
#' Burn‐in and thinning are applied so that exactly `n_chain` posterior samples
#' are returned.
#'
#' @param y Numeric vector of observed binomial counts (length \eqn{n}). Each
#'   element must satisfy \eqn{0 \leq y_t \leq n_{trials}}.
#' @param n_trials Numeric scalar > 0, number of trials for each binomial observation.
#' @param burnin Integer \eqn{\geq 0}, number of burn-in iterations.
#' @param thinning Integer \eqn{\geq 1}, thinning interval.
#' @param n_chain Integer \eqn{\geq 1}, number of posterior samples to retain.
#' @param prior_theta01_mean Numeric, prior mean for the initial state \eqn{\theta_{0,1}}.
#' @param prior_theta01_prec Numeric > 0, prior precision (inverse variance) for \eqn{\theta_{0,1}}.
#' @param prior_theta02_mean Numeric, prior mean for the initial state \eqn{\theta_{0,2}}.
#' @param prior_theta02_prec Numeric > 0, prior precision (inverse variance) for \eqn{\theta_{0,2}}.
#' @param prior_prec1_shape Numeric > 0, shape parameter of the Gamma prior for \eqn{1/W_1}.
#' @param prior_prec1_rate Numeric > 0, rate parameter of the Gamma prior for \eqn{1/W_1}.
#' @param prior_prec2_shape Numeric > 0, shape parameter of the Gamma prior for \eqn{1/W_2}.
#' @param prior_prec2_rate Numeric > 0, rate parameter of the Gamma prior for \eqn{1/W_2}.
#' @param lag_update Integer \eqn{\geq 1}, adaptation frequency for Metropolis-Hastings proposals (iterations).
#' @param max_step_size Numeric > 0, maximum proposal step size for adaptive algorithm.
#' @param base_adaptation_rate Numeric > 0, base adaptation rate for proposal scaling.
#' @param decay_exponent Numeric > 0, adaptation decay exponent for diminishing adaptation.
#' @param target_acceptance Numeric in (0,1), target acceptance proportion for Metropolis-Hastings.
#' @param min_deviation_threshold Numeric \eqn{\geq 0}, minimum absolute deviation
#'   from `target_acceptance` required to trigger log_sigma updates. If `NULL`
#'   (default), computes practical threshold as `1.0/lag_update`. Set to `0.0`
#'   for maximum sensitivity (update for any deviation). Larger values make
#'   adaptation more conservative.
#' @param return_log_sigma Logical, whether to return proposal scale diagnostics. Default is `FALSE`.
#' @param return_accept_prop Logical, whether to return acceptance proportion diagnostics. Default is `FALSE`.
#' @param seed Optional integer used to set the random number generator seed. Default is `NULL`.
#'
#' @return A list with components:
#' \describe{
#'   \item{`theta_1`}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior samples for \eqn{\theta_{t,1}}.}
#'   \item{`theta_2`}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior samples for \eqn{\theta_{t,2}}.}
#'   \item{`theta_01`}{Numeric vector of length `n_chain` of posterior samples for \eqn{\theta_{0,1}}.}
#'   \item{`theta_02`}{Numeric vector of length `n_chain` of posterior samples for \eqn{\theta_{0,2}}.}
#'   \item{`prec_1`}{Numeric vector of length `n_chain` of posterior samples for \eqn{1/W_1}.}
#'   \item{`prec_2`}{Numeric vector of length `n_chain` of posterior samples for \eqn{1/W_2}.}
#'   \item{`alpha`}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior samples for \eqn{\alpha_t}.}
#'   \item{`log_sigma`}{Numeric matrix \eqn{[n_{chain} \times n]} of proposal scale diagnostics (if requested).}
#'   \item{`accept_prop`}{Numeric matrix \eqn{[n_{chain} \times n]} of acceptance proportion diagnostics (if requested).}
#' }
#'
#' @examples
#' ## Description
#' # This example demonstrates how to:
#' # 1. Simulate data from a local-trend binomial dynamic model
#' # 2. Use `mcmc_binomial_localtrend` to estimate parameters and latent states
#' # 3. Perform a detailed posterior analysis with visualizations
#' # 4. Set a seed for reproducibility
#'
#' ## Simulation of data
#' n <- 500        # Number of observations to simulate
#' n_trials <- 20  # Number of binomial trials
#'
#' # True parameters for simulation:
#' theta01_true <- 0.5     # Initial level state (theta[0,1]) on logit scale
#' theta02_true <- 0.01    # Initial trend state (theta[0,2]) on logit scale
#' prec1_true <- 100       # Level innovation precision (1/W[1])
#' prec2_true <- 400       # Trend innovation precision (1/W[2])
#'
#' # Use a fixed seed for data simulation
#' set.seed(123)
#'
#' # Generate noise terms:
#' u1 <- rnorm(n, sd = sqrt(1/prec1_true))  # Level evolution noise (u1[t])
#' u2 <- rnorm(n, sd = sqrt(1/prec2_true))  # Trend evolution noise (u2[t])
#'
#' # Simulate latent states and observations:
#' theta1_true <- numeric(n)
#' theta2_true <- numeric(n)
#' theta2_true[1] <- theta02_true + u2[1]
#' theta1_true[1] <- theta01_true + theta02_true + u1[1]
#' for (t in 2:n) {
#'   theta2_true[t] <- theta2_true[t-1] + u2[t]
#'   theta1_true[t] <- theta1_true[t-1] + theta2_true[t-1] + u1[t]
#' }
#' alpha_true <- plogis(theta1_true)  # Success probabilities
#' y <- rbinom(n, size = n_trials, prob = alpha_true)  # Observed binomial counts
#'
#' ## Running the Gibbs sampler
#' # Run the Gibbs sampler with specified priors and a seed
#' out <- mcmc_binomial_localtrend(
#'   y,
#'   n_trials                = n_trials,
#'   burnin                  = 1000,
#'   thinning                = 50,
#'   n_chain                 = 1000,
#'   prior_theta01_mean      = 0,
#'   prior_theta01_prec      = 1,
#'   prior_theta02_mean      = 0,
#'   prior_theta02_prec      = 1,
#'   prior_prec1_shape       = 100,
#'   prior_prec1_rate        = 1,
#'   prior_prec2_shape       = 400,
#'   prior_prec2_rate        = 1,
#'   lag_update              = 50,
#'   max_step_size           = 0.1,
#'   base_adaptation_rate    = 1,
#'   decay_exponent          = 0.6,
#'   target_acceptance       = 0.44,
#'   min_deviation_threshold = NULL,  # Uses practical default: 1.0/50 = 0.02
#'   return_log_sigma        = FALSE,
#'   return_accept_prop      = TRUE,
#'   seed                    = 456
#' )
#'
#' ## Posterior analysis and visualization
#' # The following plots show how to analyze the posterior distributions.
#' # Point estimates are based on the median of posterior samples.
#' \dontrun{
#'   # --- 0. Plot the simulated data ---
#'   plot(
#'     y,
#'     main = "Simulated binomial counts",
#'     ylab = expression(y[t]),
#'     xlab = "t",
#'     type = "o",
#'     pch = 16,
#'     cex = 0.7
#'   )
#'
#'   # --- 1. Metropolis-Hastings Acceptance Proportion Diagnostics ---
#'   # Extract acceptance proportion statistics for adaptive MCMC performance evaluation
#'   acc <- out$accept_prop
#'   min_acc <- apply(X = acc, MARGIN = 2, FUN = min)
#'   max_acc <- apply(X = acc, MARGIN = 2, FUN = max)
#'   med_acc <- apply(X = acc, MARGIN = 2, FUN = median)
#'
#'   # Calculate y-axis range for optimal legend positioning
#'   range_acc <- range(min_acc, max_acc)
#'   r1_acc <- range_acc[1] - 0.05
#'   r2_acc <- range_acc[2] + 0.15 * diff(range_acc)
#'
#'   # Plot acceptance proportions with target reference line and confidence bands
#'   plot(
#'     med_acc,
#'     type = "l",
#'     col = "black",
#'     lwd = 2,
#'     xlab = "Time (t)",
#'     ylab = "Acceptance Rate",
#'     main = "Metropolis-Hastings Acceptance Rates",
#'     ylim = c(r1_acc, r2_acc)
#'   )
#'
#'   # Add confidence bands showing min-max range across MCMC iterations
#'   polygon(
#'     c(1:length(med_acc), rev(1:length(med_acc))),
#'     c(min_acc, rev(max_acc)),
#'     col = rgb(0.7, 0.7, 0.7, alpha = 0.3),
#'     border = NA
#'   )
#'
#'   # Overlay target acceptance proportion
#'   abline(h = 0.44, col = "red", lty = 2, lwd = 2)
#'
#'   # Add informative legend positioned in the expanded y-range
#'   legend(
#'     "topright",
#'     legend = c("Median acceptance proportion", "Min-Max range", "Target proportion (0.44)"),
#'     col = c("black", "gray", "red"),
#'     lty = c(1, 1, 2),
#'     lwd = c(2, 8, 2),
#'     bty = "n"
#'   )
#'
#'   # --- 2. Latent Level Trajectories (theta[t,1]) on logit scale ---
#'   # Visualize uncertainty by plotting multiple posterior trajectory samples
#'   num_traj_to_plot <- 20
#'
#'   # Calculate y-axis range for optimal legend positioning
#'   range_traj <- range(out$theta_1[1:num_traj_to_plot, ], theta1_true)
#'   r1_traj <- range_traj[1] - 0.1 * diff(range_traj)
#'   r2_traj <- range_traj[2] + 0.1 * diff(range_traj)
#'
#'   matplot(
#'     t(out$theta_1[1:num_traj_to_plot, ]),
#'     type = "l",
#'     lty = 1,
#'     col = grDevices::rainbow(num_traj_to_plot, alpha = 0.3),
#'     xlab = "t",
#'     ylab = expression(theta["t,1"]),
#'     main = "Posterior trajectory samples for latent level (logit scale)",
#'     ylim = c(r1_traj, r2_traj)
#'   )
#'   # Overlay the true trajectory
#'   lines(theta1_true, col = "black", lwd = 3, lty = 2)
#'   legend(
#'     "topright",
#'     legend = expression(theta["t,1"]),
#'     col = "black",
#'     lty = 2,
#'     lwd = 3,
#'     bty = "n"
#'   )
#'
#'   # --- 3. Latent Trend Trajectories (theta[t,2]) on logit scale ---
#'   # Visualize uncertainty by plotting multiple posterior trajectory samples
#'   range_traj2 <- range(out$theta_2[1:num_traj_to_plot, ], theta2_true)
#'   r1_traj2 <- range_traj2[1] - 0.1 * diff(range_traj2)
#'   r2_traj2 <- range_traj2[2] + 0.1 * diff(range_traj2)
#'
#'   matplot(
#'     t(out$theta_2[1:num_traj_to_plot, ]),
#'     type = "l",
#'     lty = 1,
#'     col = grDevices::rainbow(num_traj_to_plot, alpha = 0.3),
#'     xlab = "t",
#'     ylab = expression(theta["t,2"]),
#'     main = "Posterior trajectory samples for latent trend (logit scale)",
#'     ylim = c(r1_traj2, r2_traj2)
#'   )
#'   # Overlay the true trajectory
#'   lines(theta2_true, col = "black", lwd = 3, lty = 2)
#'   legend(
#'     "topright",
#'     legend = expression(theta["t,2"]),
#'     col = "black",
#'     lty = 2,
#'     lwd = 3,
#'     bty = "n"
#'   )
#'
#'   # --- 4. Latent Level Point Estimates (theta[t,1]) ---
#'   # Plot true and estimated (median) latent level with credible intervals
#'   theta_1_estimate <- apply(X = out$theta_1, MARGIN = 2, FUN = median)
#'   theta_1_q025 <- apply(X = out$theta_1, MARGIN = 2, FUN = quantile, probs = 0.025)
#'   theta_1_q975 <- apply(X = out$theta_1, MARGIN = 2, FUN = quantile, probs = 0.975)
#'
#'   range_theta_1 <- range(theta_1_estimate, theta1_true, theta_1_q025, theta_1_q975)
#'   r1_theta1 <- range_theta_1[1] - 0.1 * diff(range_theta_1)
#'   r2_theta1 <- range_theta_1[2] + 0.3 * diff(range_theta_1)
#'
#'   plot(
#'     theta1_true,
#'     col = "red",
#'     type = "l",
#'     lwd = 3,
#'     xlab = "t",
#'     ylim = c(r1_theta1, r2_theta1),
#'     lty = 2,
#'     ylab = expression(theta["t,1"]),
#'     main = "Latent level estimation (logit scale)"
#'   )
#'
#'   # Add 95% credible intervals
#'   polygon(
#'     c(1:length(theta_1_estimate), rev(1:length(theta_1_estimate))),
#'     c(theta_1_q025, rev(theta_1_q975)),
#'     col = rgb(0.7, 0.7, 0.7, alpha = 0.3),
#'     border = NA
#'   )
#'
#'   # Add point estimate
#'   lines(theta_1_estimate, col = "black", lwd = 2)
#'
#'   legend(
#'     "topright",
#'     legend = c(
#'       expression(theta["t,1"]),
#'       expression(hat(theta)["t,1"]),
#'       "95% Credible Interval"
#'     ),
#'     col = c("red", "black", "gray"),
#'     lty = c(2, 1, 1),
#'     lwd = c(3, 2, 8),
#'     bty = "n"
#'   )
#'
#'   # --- 5. Latent Trend Point Estimates (theta[t,2]) ---
#'   # Plot true and estimated (median) latent trend with credible intervals
#'   theta_2_estimate <- apply(X = out$theta_2, MARGIN = 2, FUN = median)
#'   theta_2_q025 <- apply(X = out$theta_2, MARGIN = 2, FUN = quantile, probs = 0.025)
#'   theta_2_q975 <- apply(X = out$theta_2, MARGIN = 2, FUN = quantile, probs = 0.975)
#'
#'   range_theta_2 <- range(theta_2_estimate, theta2_true, theta_2_q025, theta_2_q975)
#'   r1_theta2 <- range_theta_2[1] - 0.1 * diff(range_theta_2)
#'   r2_theta2 <- range_theta_2[2] + 0.3 * diff(range_theta_2)
#'
#'   plot(
#'     theta2_true,
#'     col = "red",
#'     type = "l",
#'     lwd = 3,
#'     xlab = "t",
#'     ylim = c(r1_theta2, r2_theta2),
#'     lty = 2,
#'     ylab = expression(theta["t,2"]),
#'     main = "Latent trend estimation (logit scale)"
#'   )
#'
#'   # Add 95% credible intervals
#'   polygon(
#'     c(1:length(theta_2_estimate), rev(1:length(theta_2_estimate))),
#'     c(theta_2_q025, rev(theta_2_q975)),
#'     col = rgb(0.7, 0.7, 0.7, alpha = 0.3),
#'     border = NA
#'   )
#'
#'   # Add point estimate
#'   lines(theta_2_estimate, col = "black", lwd = 2)
#'
#'   legend(
#'     "topright",
#'     legend = c(
#'       expression(theta["t,2"]),
#'       expression(hat(theta)["t,2"]),
#'       "95% Credible Interval"
#'     ),
#'     col = c("red", "black", "gray"),
#'     lty = c(2, 1, 1),
#'     lwd = c(3, 2, 8),
#'     bty = "n"
#'   )
#'
#'   # --- 6. Success Probabilities (alpha[t]) ---
#'   # Plot true and estimated (median) success probabilities with uncertainty
#'   alpha_estimate <- apply(X = out$alpha, MARGIN = 2, FUN = median)
#'   alpha_q025 <- apply(X = out$alpha, MARGIN = 2, FUN = quantile, probs = 0.025)
#'   alpha_q975 <- apply(X = out$alpha, MARGIN = 2, FUN = quantile, probs = 0.975)
#'
#'   # Calculate y-axis range for optimal legend positioning
#'   range_alpha <- range(alpha_true, alpha_estimate, alpha_q025, alpha_q975)
#'   r1_alpha <- max(0, range_alpha[1] - 0.05)  # Ensure lower bound is at least 0
#'   r2_alpha <- min(1, range_alpha[2] + 0.3 * diff(range_alpha))  # Ensure upper bound is at most 1
#'
#'   plot(
#'     alpha_true,
#'     col = "red",
#'     type = "l",
#'     lwd = 3,
#'     xlab = "t",
#'     ylim = c(r1_alpha, r2_alpha),
#'     lty = 2,
#'     ylab = expression(alpha[t]),
#'     main = "Success probabilities"
#'   )
#'
#'   # Add 95% credible intervals
#'   polygon(
#'     c(1:length(alpha_estimate), rev(1:length(alpha_estimate))),
#'     c(alpha_q025, rev(alpha_q975)),
#'     col = rgb(0.7, 0.7, 0.7, alpha = 0.3),
#'     border = NA
#'   )
#'
#'   # Add point estimate
#'   lines(alpha_estimate, col = "black", lwd = 2)
#'
#'   legend(
#'     "topright",
#'     legend = c(
#'       expression(alpha[t]),
#'       expression(hat(alpha)[t]),
#'       "95% Credible Interval"
#'     ),
#'     col = c("red", "black", "gray"),
#'     lty = c(2, 1, 1),
#'     lwd = c(3, 2, 8),
#'     bty = "n"
#'   )
#'
#'   # --- 7. Initial Level State (theta[0,1]) Diagnostics ---
#'   # Trace plot for theta[0,1] to assess MCMC convergence
#'   range_theta_01 <- range(out$theta_01)
#'   r1_theta01 <- range_theta_01[1] - 0.1 * diff(range_theta_01)
#'   r2_theta01 <- range_theta_01[2] + 0.3 * diff(range_theta_01)
#'
#'   plot.ts(
#'     out$theta_01,
#'     ylab = expression(theta["0,1"]),
#'     main = "Trace plot of initial level state",
#'     xlab = "Iterations",
#'     col = "gray",
#'     ylim = c(r1_theta01, r2_theta01)
#'   )
#'   abline(
#'     h = c(theta01_true, median(out$theta_01)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'   legend(
#'     "topright",
#'     legend = c(expression(theta["0,1"]), expression(hat(theta)["0,1"])),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     bty = "n",
#'     lwd = 2
#'   )
#'
#'   # Posterior density estimate for theta[0,1]
#'   range_dens_theta01 <- range(out$theta_01)
#'   r1_dens_theta01 <- range_dens_theta01[1] - 0.1 * diff(range_dens_theta01)
#'   r2_dens_theta01 <- range_dens_theta01[2] + 0.25 * diff(range_dens_theta01)
#'
#'   plot(
#'     density(out$theta_01),
#'     main = "Posterior density estimate of initial level state",
#'     xlab = expression(theta["0,1"]),
#'     ylab = "Density",
#'     lwd = 2,
#'     xlim = c(r1_dens_theta01, r2_dens_theta01)
#'   )
#'   abline(
#'     v = c(theta01_true, median(out$theta_01)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'   legend(
#'     "topright",
#'     legend = c(expression(theta["0,1"]), expression(hat(theta)["0,1"])),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     bty = "n",
#'     lwd = 2
#'   )
#'
#'   # --- 8. Initial Trend State (theta[0,2]) Diagnostics ---
#'   # Trace plot for theta[0,2] to assess MCMC convergence
#'   range_theta_02 <- range(out$theta_02)
#'   r1_theta02 <- range_theta_02[1] - 0.1 * diff(range_theta_02)
#'   r2_theta02 <- range_theta_02[2] + 0.3 * diff(range_theta_02)
#'
#'   plot.ts(
#'     out$theta_02,
#'     ylab = expression(theta["0,2"]),
#'     main = "Trace plot of initial trend state",
#'     xlab = "Iterations",
#'     col = "gray",
#'     ylim = c(r1_theta02, r2_theta02)
#'   )
#'   abline(
#'     h = c(theta02_true, median(out$theta_02)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'   legend(
#'     "topright",
#'     legend = c(expression(theta["0,2"]), expression(hat(theta)["0,2"])),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     bty = "n",
#'     lwd = 2
#'   )
#'
#'   # Posterior density estimate for theta[0,2]
#'   range_dens_theta02 <- range(out$theta_02)
#'   r1_dens_theta02 <- range_dens_theta02[1] - 0.1 * diff(range_dens_theta02)
#'   r2_dens_theta02 <- range_dens_theta02[2] + 0.25 * diff(range_dens_theta02)
#'
#'   plot(
#'     density(out$theta_02),
#'     main = "Posterior density estimate of initial trend state",
#'     xlab = expression(theta["0,2"]),
#'     ylab = "Density",
#'     lwd = 2,
#'     xlim = c(r1_dens_theta02, r2_dens_theta02)
#'   )
#'   abline(
#'     v = c(theta02_true, median(out$theta_02)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'   legend(
#'     "topright",
#'     legend = c(expression(theta["0,2"]), expression(hat(theta)["0,2"])),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     bty = "n",
#'     lwd = 2
#'   )
#'
#'   # --- 9. Level Innovation Precision (1/W[1]) Diagnostics ---
#'   # Trace plot for 1/W[1] to assess parameter convergence
#'   range_prec_1 <- range(out$prec_1)
#'   r1_prec1 <- range_prec_1[1] - 0.1 * diff(range_prec_1)
#'   r2_prec1 <- range_prec_1[2] + 0.25 * diff(range_prec_1)
#'
#'   plot.ts(
#'     out$prec_1,
#'     ylab = expression(1/W[1]),
#'     main = "Trace plot of level innovation precision",
#'     xlab = "Iterations",
#'     col = "gray",
#'     ylim = c(r1_prec1, r2_prec1)
#'   )
#'   abline(
#'     h = c(prec1_true, median(out$prec_1)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'   legend(
#'     "topright",
#'     legend = c(expression(W[1]^-1), expression(hat(W)[1]^-1)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     bty = "n",
#'     lwd = 2
#'   )
#'
#'   # Posterior density estimate for 1/W[1]
#'   range_dens_prec1 <- range(out$prec_1)
#'   r1_dens_prec1 <- range_dens_prec1[1] - 0.1 * diff(range_dens_prec1)
#'   r2_dens_prec1 <- range_dens_prec1[2] + 0.25 * diff(range_dens_prec1)
#'
#'   plot(
#'     density(out$prec_1),
#'     main = "Posterior density estimate of level innovation precision",
#'     xlab = expression(W[1]^-1),
#'     ylab = "Density",
#'     lwd = 2,
#'     xlim = c(r1_dens_prec1, r2_dens_prec1)
#'   )
#'   abline(
#'     v = c(prec1_true, median(out$prec_1)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'   legend(
#'     "topright",
#'     legend = c(expression(W[1]^-1), expression(hat(W)[1]^-1)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     bty = "n",
#'     lwd = 2
#'   )
#'
#'   # --- 10. Trend Innovation Precision (1/W[2]) Diagnostics ---
#'   # Trace plot for 1/W[2] to assess parameter convergence
#'   range_prec_2 <- range(out$prec_2)
#'   r1_prec2 <- range_prec_2[1] - 0.1 * diff(range_prec_2)
#'   r2_prec2 <- range_prec_2[2] + 0.25 * diff(range_prec_2)
#'
#'   plot.ts(
#'     out$prec_2,
#'     ylab = expression(1/W[2]),
#'     main = "Trace plot of trend innovation precision",
#'     xlab = "Iterations",
#'     col = "gray",
#'     ylim = c(r1_prec2, r2_prec2)
#'   )
#'   abline(
#'     h = c(prec2_true, median(out$prec_2)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'   legend(
#'     "topright",
#'     legend = c(expression(W[2]^-1), expression(hat(W)[2]^-1)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     bty = "n",
#'     lwd = 2
#'   )
#'
#'   # Posterior density estimate for 1/W[2]
#'   range_dens_prec2 <- range(out$prec_2)
#'   r1_dens_prec2 <- range_dens_prec2[1] - 0.1 * diff(range_dens_prec2)
#'   r2_dens_prec2 <- range_dens_prec2[2] + 0.25 * diff(range_dens_prec2)
#'
#'   plot(
#'     density(out$prec_2),
#'     main = "Posterior density estimate of trend innovation precision",
#'     xlab = expression(W[2]^-1),
#'     ylab = "Density",
#'     lwd = 2,
#'     xlim = c(r1_dens_prec2, r2_dens_prec2)
#'   )
#'   abline(
#'     v = c(prec2_true, median(out$prec_2)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'   legend(
#'     "topright",
#'     legend = c(expression(W[2]^-1), expression(hat(W)[2]^-1)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     bty = "n",
#'     lwd = 2
#'   )
#' }
#'
#' @seealso \link[pdm]{mcmc_normal_localtrend}
#' @export
mcmc_binomial_localtrend <- function(y,
                                     n_trials,
                                     burnin,
                                     thinning,
                                     n_chain,
                                     prior_theta01_mean,
                                     prior_theta01_prec,
                                     prior_theta02_mean,
                                     prior_theta02_prec,
                                     prior_prec1_shape,
                                     prior_prec1_rate,
                                     prior_prec2_shape,
                                     prior_prec2_rate,
                                     lag_update = 50,
                                     max_step_size = 2.0,
                                     base_adaptation_rate = 0.01,
                                     decay_exponent = 0.6,
                                     target_acceptance = 0.44,
                                     min_deviation_threshold = NULL,
                                     return_log_sigma = FALSE,
                                     return_accept_prop = FALSE,
                                     seed = NULL) {

  # --- Input Validation ---
  if (!is.numeric(y)) stop("`y` must be a numeric vector")
  if (!all(is.finite(y))) stop("`y` must contain only finite numeric values (no NA, NaN, Inf)")
  if (!is.numeric(n_trials) || length(n_trials) != 1 || n_trials <= 0) {
    stop("`n_trials` must be a single positive numeric value")
  }
  if (any(y < 0 | y > n_trials)) stop("`y` values must satisfy 0 <= y <= n_trials")

  if (!is.numeric(burnin) || length(burnin) != 1 || burnin < 0 || burnin != floor(burnin)) {
    stop("`burnin` must be a single non-negative integer")
  }
  if (!is.numeric(thinning) || length(thinning) != 1 || thinning < 1 || thinning != floor(thinning)) {
    stop("`thinning` must be a single positive integer")
  }
  if (!is.numeric(n_chain) || length(n_chain) != 1 || n_chain < 1 || n_chain != floor(n_chain)) {
    stop("`n_chain` must be a single positive integer")
  }

  if (!is.numeric(prior_theta01_mean) || length(prior_theta01_mean) != 1) {
    stop("`prior_theta01_mean` must be a single numeric value")
  }
  if (!is.numeric(prior_theta01_prec) || length(prior_theta01_prec) != 1 || prior_theta01_prec <= 0) {
    stop("`prior_theta01_prec` must be a single positive numeric value")
  }
  if (!is.numeric(prior_theta02_mean) || length(prior_theta02_mean) != 1) {
    stop("`prior_theta02_mean` must be a single numeric value")
  }
  if (!is.numeric(prior_theta02_prec) || length(prior_theta02_prec) != 1 || prior_theta02_prec <= 0) {
    stop("`prior_theta02_prec` must be a single positive numeric value")
  }

  if (!is.numeric(prior_prec1_shape) || length(prior_prec1_shape) != 1 || prior_prec1_shape <= 0) {
    stop("`prior_prec1_shape` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec1_rate) || length(prior_prec1_rate) != 1 || prior_prec1_rate <= 0) {
    stop("`prior_prec1_rate` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec2_shape) || length(prior_prec2_shape) != 1 || prior_prec2_shape <= 0) {
    stop("`prior_prec2_shape` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec2_rate) || length(prior_prec2_rate) != 1 || prior_prec2_rate <= 0) {
    stop("`prior_prec2_rate` must be a single positive numeric value")
  }

  if (!is.numeric(lag_update) || length(lag_update) != 1 || lag_update < 1 || lag_update != floor(lag_update)) {
    stop("`lag_update` must be a single positive integer")
  }
  if (!is.numeric(max_step_size) || length(max_step_size) != 1 || max_step_size <= 0) {
    stop("`max_step_size` must be a single positive numeric value")
  }
  if (!is.numeric(base_adaptation_rate) || length(base_adaptation_rate) != 1 || base_adaptation_rate <= 0) {
    stop("`base_adaptation_rate` must be a single positive numeric value")
  }
  if (!is.numeric(decay_exponent) || length(decay_exponent) != 1 || decay_exponent <= 0) {
    stop("`decay_exponent` must be a single positive numeric value")
  }
  if (!is.numeric(target_acceptance) || length(target_acceptance) != 1 ||
      target_acceptance <= 0 || target_acceptance >= 1) {
    stop("`target_acceptance` must be a single numeric value in (0,1)")
  }

  # Validate min_deviation_threshold parameter
  if (is.null(min_deviation_threshold)) {
    # Compute practical default threshold
    min_deviation_threshold <- 1.0 / lag_update
  } else {
    if (!is.numeric(min_deviation_threshold) || length(min_deviation_threshold) != 1 ||
        min_deviation_threshold < 0) {
      stop("`min_deviation_threshold` must be a single non-negative numeric value or NULL")
    }
  }

  if (!is.logical(return_log_sigma) || length(return_log_sigma) != 1) {
    stop("`return_log_sigma` must be a single logical value")
  }
  if (!is.logical(return_accept_prop) || length(return_accept_prop) != 1) {
    stop("`return_accept_prop` must be a single logical value")
  }

  if (!is.null(seed)) {
    if (!is.numeric(seed) || length(seed) != 1 || seed != floor(seed)) {
      stop("`seed` must be a single integer value")
    }
    set.seed(seed)
  }
  # --- End Input Validation ---

  # Call the C function with new parameter
  .Call(
    "_pdm_C_MCMC_logit_binomial_localtrend",
    as.numeric(y),
    as.numeric(n_trials),
    as.integer(burnin),
    as.integer(thinning),
    as.integer(n_chain),
    as.numeric(prior_theta01_mean),
    as.numeric(prior_theta01_prec),
    as.numeric(prior_theta02_mean),
    as.numeric(prior_theta02_prec),
    as.numeric(prior_prec1_shape),
    as.numeric(prior_prec1_rate),
    as.numeric(prior_prec2_shape),
    as.numeric(prior_prec2_rate),
    as.integer(lag_update),
    as.numeric(max_step_size),
    as.numeric(base_adaptation_rate),
    as.numeric(decay_exponent),
    as.numeric(target_acceptance),
    as.numeric(min_deviation_threshold),
    as.logical(return_log_sigma),
    as.logical(return_accept_prop)
  )
}
