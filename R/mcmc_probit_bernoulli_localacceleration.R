#' @title Gibbs Sampler for a Local-Acceleration Bernoulli Dynamic Model with Probit Link
#'
#' @description Runs a Gibbs sampler for the local-acceleration Bernoulli dynamic model
#'   with probit link using Albert-Chib data augmentation.
#'
#' @details The model is defined as:
#' \deqn{
#' \begin{aligned}
#' y_t &\sim \text{Bernoulli}(\alpha_t), \\
#' \alpha_t &= \Phi(\theta_{t,1}), \\
#' \theta_{t,1} &= \theta_{t-1,1} + \theta_{t-1,2} + u_{t,1}, & u_{t,1} \sim N(0, W_1), \\
#' \theta_{t,2} &= \theta_{t-1,2} + \theta_{t-1,3} + u_{t,2}, & u_{t,2} \sim N(0, W_2), \\
#' \theta_{t,3} &= \theta_{t-1,3} + u_{t,3},                  & u_{t,3} \sim N(0, W_3),
#' \end{aligned}
#' }
#' where \eqn{t = 1, 2, \ldots, n}, \eqn{n} is the number of observations, and
#' \eqn{\Phi(\cdot)} denotes the standard normal cumulative distribution function.
#'
#' The probit link function is defined as \eqn{\alpha_t = \Phi(\theta_{t,1})}.
#'
#' \strong{Prior Distributions:}
#'
#' The following conjugate prior distributions are employed:
#'
#' \emph{Initial States:}
#' \deqn{
#' \begin{aligned}
#' \theta_{0,1} &\sim N(\mu_{01}, \tau_{01}^{-1}), \\
#' \theta_{0,2} &\sim N(\mu_{02}, \tau_{02}^{-1}), \\
#' \theta_{0,3} &\sim N(\mu_{03}, \tau_{03}^{-1}).
#' \end{aligned}
#' }
#'
#' \emph{Precision Parameters:}
#' \deqn{
#' \begin{aligned}
#' W_1^{-1} &\sim \text{Gamma}(\nu_1, \eta_1), \\
#' W_2^{-1} &\sim \text{Gamma}(\nu_2, \eta_2), \\
#' W_3^{-1} &\sim \text{Gamma}(\nu_3, \eta_3).
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
#'   \eqn{\mu_{03}} \tab `prior_theta03_mean` \cr
#'   \eqn{\tau_{03}} \tab `prior_theta03_prec` \cr
#'   \eqn{\nu_1} \tab `prior_prec1_shape` \cr
#'   \eqn{\eta_1} \tab `prior_prec1_rate` \cr
#'   \eqn{\nu_2} \tab `prior_prec2_shape` \cr
#'   \eqn{\eta_2} \tab `prior_prec2_rate` \cr
#'   \eqn{\nu_3} \tab `prior_prec3_shape` \cr
#'   \eqn{\eta_3} \tab `prior_prec3_rate`
#' }
#'
#' The algorithm employs the Albert-Chib (1993) latent variable augmentation
#' scheme to obtain fully conjugate Gibbs updates for all parameters. This approach
#' introduces auxiliary Gaussian latent variables whose signs correspond to the
#' observed binary outcomes, yielding closed-form conditional distributions for
#' the latent states \eqn{\theta_{t,1}}, \eqn{\theta_{t,2}}, and \eqn{\theta_{t,3}},
#' the initial states \eqn{\theta_{0,1}}, \eqn{\theta_{0,2}}, and \eqn{\theta_{0,3}},
#' and the innovation precisions \eqn{1/W_1}, \eqn{1/W_2}, and \eqn{1/W_3}.
#'
#' Since this is a pure Gibbs sampler with no Metropolis-Hastings steps, the
#' acceptance rate is always 1.0, eliminating the need for proposal tuning
#' parameters.
#'
#' \strong{Progress Bar:}
#' When `verbose = TRUE`, a visual progress bar is displayed showing
#' \itemize{
#' \item{Progress bar with adaptive update frequency (based on `bar_width`)}
#' \item{Elapsed time in HH:MM:SS format}
#' \item{Estimated remaining time in HH:MM:SS format}
#' }
#'
#' The progress bar update frequency is automatically calculated as approximately
#' one update per bar segment, ensuring smooth visual feedback with minimal
#' performance overhead (~0.01\% for typical runs).
#'
#' Burn-in and thinning are applied so that exactly `n_chain` posterior
#' samples are returned.
#'
#' @param y Numeric vector of observed Bernoulli outcomes (length \eqn{n}). Each
#'   element must be either 0 or 1.
#' @param burnin Integer \eqn{\geq 0}, number of burn-in iterations.
#' @param thinning Integer \eqn{\geq 1}, thinning interval.
#' @param n_chain Integer \eqn{\geq 1}, number of posterior samples to retain.
#' @param prior_theta01_mean Numeric, prior mean for the initial state \eqn{\theta_{0,1}}.
#' @param prior_theta01_prec Numeric > 0, prior precision (inverse variance) for \eqn{\theta_{0,1}}.
#' @param prior_theta02_mean Numeric, prior mean for the initial state \eqn{\theta_{0,2}}.
#' @param prior_theta02_prec Numeric > 0, prior precision (inverse variance) for \eqn{\theta_{0,2}}.
#' @param prior_theta03_mean Numeric, prior mean for the initial state \eqn{\theta_{0,3}}.
#' @param prior_theta03_prec Numeric > 0, prior precision (inverse variance) for \eqn{\theta_{0,3}}.
#' @param prior_prec1_shape Numeric > 0, shape parameter of the Gamma prior for \eqn{1/W_1}.
#' @param prior_prec1_rate Numeric > 0, rate parameter of the Gamma prior for \eqn{1/W_1}.
#' @param prior_prec2_shape Numeric > 0, shape parameter of the Gamma prior for \eqn{1/W_2}.
#' @param prior_prec2_rate Numeric > 0, rate parameter of the Gamma prior for \eqn{1/W_2}.
#' @param prior_prec3_shape Numeric > 0, shape parameter of the Gamma prior for \eqn{1/W_3}.
#' @param prior_prec3_rate Numeric > 0, rate parameter of the Gamma prior for \eqn{1/W_3}.
#' @param verbose Logical, whether to display a progress bar during sampling. Default is `FALSE`.
#' @param bar_width Integer in [10, 120], width of the progress bar when `verbose = TRUE`. Default is `60`.
#' @param seed Optional integer used to set the random number generator seed.
#'   Default is `NULL`, which does not set the seed.
#'
#' @return A list with components:
#' \describe{
#'   \item{`theta_1`}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior
#'     samples for the latent level state \eqn{\theta_{t,1}}.}
#'   \item{`theta_2`}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior
#'     samples for the latent trend state \eqn{\theta_{t,2}}.}
#'   \item{`theta_3`}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior
#'     samples for the latent acceleration state \eqn{\theta_{t,3}}.}
#'   \item{`theta_01`}{Numeric vector of length `n_chain` of posterior samples
#'     for the initial level state \eqn{\theta_{0,1}}.}
#'   \item{`theta_02`}{Numeric vector of length `n_chain` of posterior samples
#'     for the initial trend state \eqn{\theta_{0,2}}.}
#'   \item{`theta_03`}{Numeric vector of length `n_chain` of posterior samples
#'     for the initial acceleration state \eqn{\theta_{0,3}}.}
#'   \item{`prec_theta1`}{Numeric vector of length `n_chain` of posterior samples
#'     for the level innovation precision \eqn{1/W_1}.}
#'   \item{`prec_theta2`}{Numeric vector of length `n_chain` of posterior samples
#'     for the trend innovation precision \eqn{1/W_2}.}
#'   \item{`prec_theta3`}{Numeric vector of length `n_chain` of posterior samples
#'     for the acceleration innovation precision \eqn{1/W_3}.}
#'   \item{`alpha`}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior
#'     samples for the Bernoulli probabilities \eqn{\alpha_t}.}
#' }
#'
#' @examples
#' ## Description
#' # This example demonstrates how to:
#' # 1. Simulate data from a multi-frequency sinusoidal probability pattern
#' # 2. Use `mcmc_probit_bernoulli_localacceleration` to estimate parameters and latent states
#' # 3. Perform a detailed posterior analysis with visualizations
#' # 4. Set a seed for reproducibility
#' # 5. Use progress bar for monitoring MCMC execution
#'
#' ## Simulation of data
#' n <- 500  # Number of observations to simulate
#'
#' # Use a fixed seed for data simulation
#' set.seed(123)
#'
#' # Generate true success probabilities following a multi-frequency sinusoidal pattern
#' grid_vals <- seq_len(n) / n
#' alpha_true <- (sin(4 * pi * grid_vals) + sin(8 * pi * grid_vals) + 2) / 4
#'
#' # Generate Bernoulli observations
#' y <- rbinom(n, size = 1, prob = alpha_true)
#'
#' ## Running the Gibbs sampler with progress bar
#' out <- mcmc_probit_bernoulli_localacceleration(
#'   y,
#'   burnin             = 1000,
#'   thinning           = 50,
#'   n_chain            = 1000,
#'   prior_theta01_mean = 0,
#'   prior_theta01_prec = 1,
#'   prior_theta02_mean = 0,
#'   prior_theta02_prec = 1,
#'   prior_theta03_mean = 0,
#'   prior_theta03_prec = 1,
#'   prior_prec1_shape  = 100,
#'   prior_prec1_rate   = 1,
#'   prior_prec2_shape  = 400,
#'   prior_prec2_rate   = 1,
#'   prior_prec3_shape  = 1600,
#'   prior_prec3_rate   = 1,
#'   verbose            = TRUE,
#'   bar_width          = 60,
#'   seed               = 456
#' )
#'
#' ## Posterior analysis and visualization
#' # The following plots show how to analyze the posterior distributions.
#' # Point estimates are based on the median of posterior samples.
#' \dontrun{
#'   # --- 0. Plot the simulated data ---
#'   plot(
#'     y,
#'     main = "Simulated Bernoulli outcomes",
#'     ylab = expression(y[t]),
#'     xlab = "t",
#'     type = "p",
#'     pch = 16,
#'     cex = 0.5,
#'     col = ifelse(y == 1, "blue", "red")
#'   )
#'   # Overlay the true probability curve
#'   lines(alpha_true, col = "black", lwd = 2)
#'   legend(
#'     "topright",
#'     legend = c(expression(alpha[t]), "y = 1", "y = 0"),
#'     col = c("black", "blue", "red"),
#'     lty = c(1, NA, NA),
#'     pch = c(NA, 16, 16),
#'     lwd = c(2, NA, NA),
#'     bty = "n"
#'   )
#'
#'   # --- 1. Latent Level Trajectories (theta[t,1]) on probit scale ---
#'   # Visualize uncertainty by plotting multiple posterior trajectory samples
#'   num_traj_to_plot <- 20
#'
#'   # Compute true theta_1 values (inverse probit of alpha)
#'   theta1_true <- qnorm(alpha_true)
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
#'     main = "Posterior trajectory samples for latent level (probit scale)",
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
#'   # --- 2. Latent Trend Trajectories (theta[t,2]) on probit scale ---
#'   # Visualize uncertainty by plotting multiple posterior trajectory samples
#'
#'   # Calculate y-axis range for optimal legend positioning
#'   range_traj2 <- range(out$theta_2[1:num_traj_to_plot, ])
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
#'     main = "Posterior trajectory samples for latent trend (probit scale)",
#'     ylim = c(r1_traj2, r2_traj2)
#'   )
#'   legend(
#'     "topright",
#'     legend = "Posterior samples",
#'     col = "gray",
#'     lty = 1,
#'     lwd = 2,
#'     bty = "n"
#'   )
#'
#'   # --- 3. Latent Acceleration Trajectories (theta[t,3]) on probit scale ---
#'   # Visualize uncertainty by plotting multiple posterior trajectory samples
#'
#'   # Calculate y-axis range for optimal legend positioning
#'   range_traj3 <- range(out$theta_3[1:num_traj_to_plot, ])
#'   r1_traj3 <- range_traj3[1] - 0.1 * diff(range_traj3)
#'   r2_traj3 <- range_traj3[2] + 0.1 * diff(range_traj3)
#'
#'   matplot(
#'     t(out$theta_3[1:num_traj_to_plot, ]),
#'     type = "l",
#'     lty = 1,
#'     col = grDevices::rainbow(num_traj_to_plot, alpha = 0.3),
#'     xlab = "t",
#'     ylab = expression(theta["t,3"]),
#'     main = "Posterior trajectory samples for latent acceleration (probit scale)",
#'     ylim = c(r1_traj3, r2_traj3)
#'   )
#'   legend(
#'     "topright",
#'     legend = "Posterior samples",
#'     col = "gray",
#'     lty = 1,
#'     lwd = 2,
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
#'     main = "Latent level estimation (probit scale)"
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
#'   # Plot estimated (median) latent trend with credible intervals
#'   theta_2_estimate <- apply(X = out$theta_2, MARGIN = 2, FUN = median)
#'   theta_2_q025 <- apply(X = out$theta_2, MARGIN = 2, FUN = quantile, probs = 0.025)
#'   theta_2_q975 <- apply(X = out$theta_2, MARGIN = 2, FUN = quantile, probs = 0.975)
#'
#'   range_theta_2 <- range(theta_2_estimate, theta_2_q025, theta_2_q975)
#'   r1_theta2 <- range_theta_2[1] - 0.1 * diff(range_theta_2)
#'   r2_theta2 <- range_theta_2[2] + 0.3 * diff(range_theta_2)
#'
#'   plot(
#'     theta_2_estimate,
#'     col = "black",
#'     type = "l",
#'     lwd = 2,
#'     xlab = "t",
#'     ylim = c(r1_theta2, r2_theta2),
#'     ylab = expression(theta["t,2"]),
#'     main = "Latent trend estimation (probit scale)"
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
#'   legend(
#'     "topright",
#'     legend = c(
#'       expression(hat(theta)["t,2"]),
#'       "95% Credible Interval"
#'     ),
#'     col = c("black", "gray"),
#'     lty = c(1, 1),
#'     lwd = c(2, 8),
#'     bty = "n"
#'   )
#'
#'   # --- 6. Latent Acceleration Point Estimates (theta[t,3]) ---
#'   # Plot estimated (median) latent acceleration with credible intervals
#'   theta_3_estimate <- apply(X = out$theta_3, MARGIN = 2, FUN = median)
#'   theta_3_q025 <- apply(X = out$theta_3, MARGIN = 2, FUN = quantile, probs = 0.025)
#'   theta_3_q975 <- apply(X = out$theta_3, MARGIN = 2, FUN = quantile, probs = 0.975)
#'
#'   range_theta_3 <- range(theta_3_estimate, theta_3_q025, theta_3_q975)
#'   r1_theta3 <- range_theta_3[1] - 0.1 * diff(range_theta_3)
#'   r2_theta3 <- range_theta_3[2] + 0.3 * diff(range_theta_3)
#'
#'   plot(
#'     theta_3_estimate,
#'     col = "black",
#'     type = "l",
#'     lwd = 2,
#'     xlab = "t",
#'     ylim = c(r1_theta3, r2_theta3),
#'     ylab = expression(theta["t,3"]),
#'     main = "Latent acceleration estimation (probit scale)"
#'   )
#'
#'   # Add 95% credible intervals
#'   polygon(
#'     c(1:length(theta_3_estimate), rev(1:length(theta_3_estimate))),
#'     c(theta_3_q025, rev(theta_3_q975)),
#'     col = rgb(0.7, 0.7, 0.7, alpha = 0.3),
#'     border = NA
#'   )
#'
#'   legend(
#'     "topright",
#'     legend = c(
#'       expression(hat(theta)["t,3"]),
#'       "95% Credible Interval"
#'     ),
#'     col = c("black", "gray"),
#'     lty = c(1, 1),
#'     lwd = c(2, 8),
#'     bty = "n"
#'   )
#'
#'   # --- 7. Bernoulli Probabilities (alpha[t]) ---
#'   # Plot true and estimated (median) probabilities with uncertainty
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
#'     main = "Bernoulli probabilities"
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
#'   # --- 8. Initial Level State (theta[0,1]) Diagnostics ---
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
#'   abline(h = median(out$theta_01), col = "black", lty = 1, lwd = 2)
#'   legend(
#'     "topright",
#'     legend = expression(hat(theta)["0,1"]),
#'     col = "black",
#'     lty = 1,
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
#'   abline(v = median(out$theta_01), col = "black", lty = 1, lwd = 2)
#'   legend(
#'     "topright",
#'     legend = expression(hat(theta)["0,1"]),
#'     col = "black",
#'     lty = 1,
#'     bty = "n",
#'     lwd = 2
#'   )
#'
#'   # --- 9. Initial Trend State (theta[0,2]) Diagnostics ---
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
#'   abline(h = median(out$theta_02), col = "black", lty = 1, lwd = 2)
#'   legend(
#'     "topright",
#'     legend = expression(hat(theta)["0,2"]),
#'     col = "black",
#'     lty = 1,
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
#'   abline(v = median(out$theta_02), col = "black", lty = 1, lwd = 2)
#'   legend(
#'     "topright",
#'     legend = expression(hat(theta)["0,2"]),
#'     col = "black",
#'     lty = 1,
#'     bty = "n",
#'     lwd = 2
#'   )
#'
#'   # --- 10. Initial Acceleration State (theta[0,3]) Diagnostics ---
#'   # Trace plot for theta[0,3] to assess MCMC convergence
#'   range_theta_03 <- range(out$theta_03)
#'   r1_theta03 <- range_theta_03[1] - 0.1 * diff(range_theta_03)
#'   r2_theta03 <- range_theta_03[2] + 0.3 * diff(range_theta_03)
#'
#'   plot.ts(
#'     out$theta_03,
#'     ylab = expression(theta["0,3"]),
#'     main = "Trace plot of initial acceleration state",
#'     xlab = "Iterations",
#'     col = "gray",
#'     ylim = c(r1_theta03, r2_theta03)
#'   )
#'   abline(h = median(out$theta_03), col = "black", lty = 1, lwd = 2)
#'   legend(
#'     "topright",
#'     legend = expression(hat(theta)["0,3"]),
#'     col = "black",
#'     lty = 1,
#'     bty = "n",
#'     lwd = 2
#'   )
#'
#'   # Posterior density estimate for theta[0,3]
#'   range_dens_theta03 <- range(out$theta_03)
#'   r1_dens_theta03 <- range_dens_theta03[1] - 0.1 * diff(range_dens_theta03)
#'   r2_dens_theta03 <- range_dens_theta03[2] + 0.25 * diff(range_dens_theta03)
#'
#'   plot(
#'     density(out$theta_03),
#'     main = "Posterior density estimate of initial acceleration state",
#'     xlab = expression(theta["0,3"]),
#'     ylab = "Density",
#'     lwd = 2,
#'     xlim = c(r1_dens_theta03, r2_dens_theta03)
#'   )
#'   abline(v = median(out$theta_03), col = "black", lty = 1, lwd = 2)
#'   legend(
#'     "topright",
#'     legend = expression(hat(theta)["0,3"]),
#'     col = "black",
#'     lty = 1,
#'     bty = "n",
#'     lwd = 2
#'   )
#'
#'   # --- 11. Level Innovation Precision (1/W[1]) Diagnostics ---
#'   # Trace plot for 1/W[1] to assess parameter convergence
#'   range_prec_theta1 <- range(out$prec_theta1)
#'   r1_prec1 <- range_prec_theta1[1] - 0.1 * diff(range_prec_theta1)
#'   r2_prec1 <- range_prec_theta1[2] + 0.25 * diff(range_prec_theta1)
#'
#'   plot.ts(
#'     out$prec_theta1,
#'     ylab = expression(1/W[1]),
#'     main = "Trace plot of level innovation precision",
#'     xlab = "Iterations",
#'     col = "gray",
#'     ylim = c(r1_prec1, r2_prec1)
#'   )
#'   abline(h = median(out$prec_theta1), col = "black", lty = 1, lwd = 2)
#'   legend(
#'     "topright",
#'     legend = expression(hat(W)[1]^-1),
#'     col = "black",
#'     lty = 1,
#'     bty = "n",
#'     lwd = 2
#'   )
#'
#'   # Posterior density estimate for 1/W[1]
#'   range_dens_prec1 <- range(out$prec_theta1)
#'   r1_dens_prec1 <- range_dens_prec1[1] - 0.1 * diff(range_dens_prec1)
#'   r2_dens_prec1 <- range_dens_prec1[2] + 0.25 * diff(range_dens_prec1)
#'
#'   plot(
#'     density(out$prec_theta1),
#'     main = "Posterior density estimate of level innovation precision",
#'     xlab = expression(W[1]^-1),
#'     ylab = "Density",
#'     lwd = 2,
#'     xlim = c(r1_dens_prec1, r2_dens_prec1)
#'   )
#'   abline(v = median(out$prec_theta1), col = "black", lty = 1, lwd = 2)
#'   legend(
#'     "topright",
#'     legend = expression(hat(W)[1]^-1),
#'     col = "black",
#'     lty = 1,
#'     bty = "n",
#'     lwd = 2
#'   )
#'
#'   # --- 12. Trend Innovation Precision (1/W[2]) Diagnostics ---
#'   # Trace plot for 1/W[2] to assess parameter convergence
#'   range_prec_theta2 <- range(out$prec_theta2)
#'   r1_prec2 <- range_prec_theta2[1] - 0.1 * diff(range_prec_theta2)
#'   r2_prec2 <- range_prec_theta2[2] + 0.25 * diff(range_prec_theta2)
#'
#'   plot.ts(
#'     out$prec_theta2,
#'     ylab = expression(1/W[2]),
#'     main = "Trace plot of trend innovation precision",
#'     xlab = "Iterations",
#'     col = "gray",
#'     ylim = c(r1_prec2, r2_prec2)
#'   )
#'   abline(h = median(out$prec_theta2), col = "black", lty = 1, lwd = 2)
#'   legend(
#'     "topright",
#'     legend = expression(hat(W)[2]^-1),
#'     col = "black",
#'     lty = 1,
#'     bty = "n",
#'     lwd = 2
#'   )
#'
#'   # Posterior density estimate for 1/W[2]
#'   range_dens_prec2 <- range(out$prec_theta2)
#'   r1_dens_prec2 <- range_dens_prec2[1] - 0.1 * diff(range_dens_prec2)
#'   r2_dens_prec2 <- range_dens_prec2[2] + 0.25 * diff(range_dens_prec2)
#'
#'   plot(
#'     density(out$prec_theta2),
#'     main = "Posterior density estimate of trend innovation precision",
#'     xlab = expression(W[2]^-1),
#'     ylab = "Density",
#'     lwd = 2,
#'     xlim = c(r1_dens_prec2, r2_dens_prec2)
#'   )
#'   abline(v = median(out$prec_theta2), col = "black", lty = 1, lwd = 2)
#'   legend(
#'     "topright",
#'     legend = expression(hat(W)[2]^-1),
#'     col = "black",
#'     lty = 1,
#'     bty = "n",
#'     lwd = 2
#'   )
#'
#'   # --- 13. Acceleration Innovation Precision (1/W[3]) Diagnostics ---
#'   # Trace plot for 1/W[3] to assess parameter convergence
#'   range_prec_theta3 <- range(out$prec_theta3)
#'   r1_prec3 <- range_prec_theta3[1] - 0.1 * diff(range_prec_theta3)
#'   r2_prec3 <- range_prec_theta3[2] + 0.25 * diff(range_prec_theta3)
#'
#'   plot.ts(
#'     out$prec_theta3,
#'     ylab = expression(1/W[3]),
#'     main = "Trace plot of acceleration innovation precision",
#'     xlab = "Iterations",
#'     col = "gray",
#'     ylim = c(r1_prec3, r2_prec3)
#'   )
#'   abline(h = median(out$prec_theta3), col = "black", lty = 1, lwd = 2)
#'   legend(
#'     "topright",
#'     legend = expression(hat(W)[3]^-1),
#'     col = "black",
#'     lty = 1,
#'     bty = "n",
#'     lwd = 2
#'   )
#'
#'   # Posterior density estimate for 1/W[3]
#'   range_dens_prec3 <- range(out$prec_theta3)
#'   r1_dens_prec3 <- range_dens_prec3[1] - 0.1 * diff(range_dens_prec3)
#'   r2_dens_prec3 <- range_dens_prec3[2] + 0.25 * diff(range_dens_prec3)
#'
#'   plot(
#'     density(out$prec_theta3),
#'     main = "Posterior density estimate of acceleration innovation precision",
#'     xlab = expression(W[3]^-1),
#'     ylab = "Density",
#'     lwd = 2,
#'     xlim = c(r1_dens_prec3, r2_dens_prec3)
#'   )
#'   abline(v = median(out$prec_theta3), col = "black", lty = 1, lwd = 2)
#'   legend(
#'     "topright",
#'     legend = expression(hat(W)[3]^-1),
#'     col = "black",
#'     lty = 1,
#'     bty = "n",
#'     lwd = 2
#'   )
#' }
#'
#' @references
#' Albert, J. H., & Chib, S. (1993). Bayesian Analysis of Binary and Polychotomous
#' Response Data. \emph{Journal of the American Statistical Association}, 88(422), 669-679.
#' https://doi.org/10.1080/01621459.1993.10476321
#'
#' @seealso \link[pdm]{mcmc_binomial_localacceleration}, \link[pdm]{mcmc_probit_bernoulli_localtrend}
#' @export
mcmc_probit_bernoulli_localacceleration <- function(y,
                                                    burnin,
                                                    thinning,
                                                    n_chain,
                                                    prior_theta01_mean,
                                                    prior_theta01_prec,
                                                    prior_theta02_mean,
                                                    prior_theta02_prec,
                                                    prior_theta03_mean,
                                                    prior_theta03_prec,
                                                    prior_prec1_shape,
                                                    prior_prec1_rate,
                                                    prior_prec2_shape,
                                                    prior_prec2_rate,
                                                    prior_prec3_shape,
                                                    prior_prec3_rate,
                                                    verbose = FALSE,
                                                    bar_width = 60,
                                                    seed = NULL) {
  # --- Input Validation ---
  if (!is.numeric(y)) {
    stop("`y` must be a numeric vector")
  }
  if (!all(is.finite(y))) {
    stop("`y` must contain only finite numeric values (no NA, NaN, Inf)")
  }

  # Validate Bernoulli support
  if (!all(y %in% c(0, 1))) {
    stop("`y` values must be either 0 or 1 (Bernoulli outcomes)")
  }

  if (!is.numeric(burnin) || length(burnin) != 1 || burnin < 0 ||
      burnin != floor(burnin)) {
    stop("`burnin` must be a single non-negative integer")
  }
  if (!is.numeric(thinning) || length(thinning) != 1 || thinning < 1 ||
      thinning != floor(thinning)) {
    stop("`thinning` must be a single positive integer")
  }
  if (!is.numeric(n_chain) || length(n_chain) != 1 || n_chain < 1 ||
      n_chain != floor(n_chain)) {
    stop("`n_chain` must be a single positive integer")
  }

  if (!is.numeric(prior_theta01_mean) || length(prior_theta01_mean) != 1) {
    stop("`prior_theta01_mean` must be a single numeric value")
  }
  if (!is.numeric(prior_theta01_prec) || length(prior_theta01_prec) != 1 ||
      prior_theta01_prec <= 0) {
    stop("`prior_theta01_prec` must be a single positive numeric value")
  }
  if (!is.numeric(prior_theta02_mean) || length(prior_theta02_mean) != 1) {
    stop("`prior_theta02_mean` must be a single numeric value")
  }
  if (!is.numeric(prior_theta02_prec) || length(prior_theta02_prec) != 1 ||
      prior_theta02_prec <= 0) {
    stop("`prior_theta02_prec` must be a single positive numeric value")
  }
  if (!is.numeric(prior_theta03_mean) || length(prior_theta03_mean) != 1) {
    stop("`prior_theta03_mean` must be a single numeric value")
  }
  if (!is.numeric(prior_theta03_prec) || length(prior_theta03_prec) != 1 ||
      prior_theta03_prec <= 0) {
    stop("`prior_theta03_prec` must be a single positive numeric value")
  }

  if (!is.numeric(prior_prec1_shape) || length(prior_prec1_shape) != 1 ||
      prior_prec1_shape <= 0) {
    stop("`prior_prec1_shape` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec1_rate) || length(prior_prec1_rate) != 1 ||
      prior_prec1_rate <= 0) {
    stop("`prior_prec1_rate` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec2_shape) || length(prior_prec2_shape) != 1 ||
      prior_prec2_shape <= 0) {
    stop("`prior_prec2_shape` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec2_rate) || length(prior_prec2_rate) != 1 ||
      prior_prec2_rate <= 0) {
    stop("`prior_prec2_rate` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec3_shape) || length(prior_prec3_shape) != 1 ||
      prior_prec3_shape <= 0) {
    stop("`prior_prec3_shape` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec3_rate) || length(prior_prec3_rate) != 1 ||
      prior_prec3_rate <= 0) {
    stop("`prior_prec3_rate` must be a single positive numeric value")
  }

  # Validate logical parameters
  if (!is.logical(verbose) || length(verbose) != 1) {
    stop("`verbose` must be a single logical value")
  }
  if (!is.numeric(bar_width) || length(bar_width) != 1 ||
      bar_width < 10 || bar_width > 120 || bar_width != floor(bar_width)) {
    stop("`bar_width` must be a single integer in [10, 120]")
  }

  # Validate and set seed if provided
  if (!is.null(seed)) {
    if (!is.numeric(seed) || length(seed) != 1 || seed != floor(seed)) {
      stop("`seed` must be a single integer value")
    }
    set.seed(seed)
  }
  # --- End Input Validation ---

  # Call the C function
  .Call(
    "_pdm_C_MCMC_probit_bernoulli_localacceleration",
    as.numeric(y),
    as.integer(burnin),
    as.integer(thinning),
    as.integer(n_chain),
    as.numeric(prior_theta01_mean),
    as.numeric(prior_theta01_prec),
    as.numeric(prior_theta02_mean),
    as.numeric(prior_theta02_prec),
    as.numeric(prior_theta03_mean),
    as.numeric(prior_theta03_prec),
    as.numeric(prior_prec1_shape),
    as.numeric(prior_prec1_rate),
    as.numeric(prior_prec2_shape),
    as.numeric(prior_prec2_rate),
    as.numeric(prior_prec3_shape),
    as.numeric(prior_prec3_rate),
    as.logical(verbose),
    as.integer(bar_width)
  )
}
