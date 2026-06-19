#' @title Gibbs Sampler for a Local-Acceleration Dynamic Model
#'
#' @description Runs a Gibbs sampler for the local acceleration dynamic model.
#'
#' @details The model is defined as:
#' \deqn{
#' \begin{aligned}
#' y_t &= \theta_{t,1} + \epsilon_t,                          & \epsilon_t  & \sim N(0, V), \\
#' \theta_{t,1} &= \theta_{t-1,1} + \theta_{t-1,2} + u_{t,1}, & u_{t,1}     & \sim N(0, W_1), \\
#' \theta_{t,2} &= \theta_{t-1,2} + \theta_{t-1,3} + u_{t,2}, & u_{t,2}     & \sim N(0, W_2), \\
#' \theta_{t,3} &= \theta_{t-1,3} + u_{t,3},                  & u_{t,3}     & \sim N(0, W_3),
#' \end{aligned}
#' }
#' where \eqn{t = 1, 2, \ldots, n} and \eqn{n} is the number of observations.
#'
#'
#' \strong{Prior Distributions:}
#'
#' The following conjugate prior distributions are employed to ensure computational
#' tractability and closed-form posterior updates in the Gibbs sampler:
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
#' W_3^{-1} &\sim \text{Gamma}(\nu_3, \eta_3), \\
#' V^{-1} &\sim \text{Gamma}(\nu_V, \eta_V).
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
#'   \eqn{\eta_3} \tab `prior_prec3_rate` \cr
#'   \eqn{\nu_V} \tab `prior_prec_y_shape` \cr
#'   \eqn{\eta_V} \tab `prior_prec_y_rate`
#' }
#'
#' These conjugate priors enable efficient Gibbs sampling with closed-form
#' conditional distributions. The parameterization uses precision (inverse
#' variance) to maintain natural conjugacy and ensure numerical stability.
#' The Gamma distribution uses shape-rate parameterization, where
#' \eqn{E(X) = \nu/\eta} and \eqn{\text{Var}(X) = \nu/\eta^2}.
#'
#' Burn-in and thinning are applied so that exactly `n_chain` posterior samples
#' are returned.
#'
#' @param y Numeric vector of observations (length \eqn{n}). Must contain only finite values.
#' @param burnin Integer \eqn{\geq 0}, number of burn-in iterations.
#' @param thinning Integer \eqn{\geq 1}, thinning interval.
#' @param n_chain Integer \eqn{\geq 1}, number of posterior samples to retain.
#' @param prior_theta01_mean Numeric, prior mean for the initial level \eqn{\theta_{0,1}}.
#' @param prior_theta01_prec Numeric > 0, prior precision for \eqn{\theta_{0,1}}.
#' @param prior_theta02_mean Numeric, prior mean for the initial trend \eqn{\theta_{0,2}}.
#' @param prior_theta02_prec Numeric > 0, prior precision for \eqn{\theta_{0,2}}.
#' @param prior_theta03_mean Numeric, prior mean for the initial acceleration \eqn{\theta_{0,3}}.
#' @param prior_theta03_prec Numeric > 0, prior precision for \eqn{\theta_{0,3}}.
#' @param prior_prec1_shape Numeric > 0, shape parameter of the Gamma prior for the level innovation precision \eqn{1/W_1}.
#' @param prior_prec1_rate Numeric > 0, rate parameter of the Gamma prior for \eqn{1/W_1}.
#' @param prior_prec2_shape Numeric > 0, shape parameter of the Gamma prior for the trend innovation precision \eqn{1/W_2}.
#' @param prior_prec2_rate Numeric > 0, rate parameter of the Gamma prior for \eqn{1/W_2}.
#' @param prior_prec3_shape Numeric > 0, shape parameter of the Gamma prior for the acceleration innovation precision \eqn{1/W_3}.
#' @param prior_prec3_rate Numeric > 0, rate parameter of the Gamma prior for \eqn{1/W_3}.
#' @param prior_prec_y_shape Numeric > 0, shape parameter of the Gamma prior for the data precision \eqn{1/V}.
#' @param prior_prec_y_rate Numeric > 0, rate parameter of the Gamma prior for \eqn{1/V}.
#' @param verbose Logical, whether to display a progress bar during sampling. Default is `FALSE`.
#' @param bar_width Integer in \[10, 120\], width of the progress bar when `verbose = TRUE`. Default is `60`.
#' @param seed Optional integer used to set the random number generator seed.
#'   Default is `NULL`, which does not set the seed.
#'
#' @return A list with components:
#' \describe{
#'   \item{\code{theta_1}}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior samples for the level state \eqn{\theta_{t,1}}.}
#'   \item{\code{theta_2}}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior samples for the trend state \eqn{\theta_{t,2}}.}
#'   \item{\code{theta_3}}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior samples for the acceleration state \eqn{\theta_{t,3}}.}
#'   \item{\code{theta_01}}{Numeric vector of length `n_chain` for the initial level \eqn{\theta_{0,1}}.}
#'   \item{\code{theta_02}}{Numeric vector of length `n_chain` for the initial trend \eqn{\theta_{0,2}}.}
#'   \item{\code{theta_03}}{Numeric vector of length `n_chain` for the initial acceleration \eqn{\theta_{0,3}}.}
#'   \item{\code{prec_theta1}}{Numeric vector of length `n_chain` for the level innovation precision \eqn{1/W_1}.}
#'   \item{\code{prec_theta2}}{Numeric vector of length `n_chain` for the trend innovation precision \eqn{1/W_2}.}
#'   \item{\code{prec_theta3}}{Numeric vector of length `n_chain` for the acceleration innovation precision \eqn{1/W_3}.}
#'   \item{\code{prec_y}}{Numeric vector of length `n_chain` for the data precision \eqn{1/V}.}
#' }
#'
#' @examples
#' ## Description
#' # This example demonstrates how to:
#' # 1. Simulate data from a local acceleration dynamic model
#' # 2. Use `mcmc_normal_localacceleration` to estimate parameters and latent states
#' # 3. Perform a detailed posterior analysis with visualizations
#' # 4. Set a seed for reproducibility
#'
#' ## Simulation of data
#' n <- 1000 # Number of observations to simulate
#'
#' # True parameters for simulation:
#' theta01_true <- 10         # Initial level (theta[0,1])
#' theta02_true <- 0.5        # Initial trend (theta[0,2])
#' theta03_true <- 0.01       # Initial acceleration (theta[0,3])
#' prec1_true   <- 1 / 0.100  # Level innovation precision (1/W[1])
#' prec2_true   <- 1 / 0.010  # Trend innovation precision (1/W[2])
#' prec3_true   <- 1 / 0.001  # Acceleration innovation precision (1/W[3])
#' prec_y_true  <- 1 / 1.000  # Observation precision (1/V)
#'
#' # Use a fixed seed for data simulation
#' set.seed(123)
#'
#' # Generate noise terms:
#' u1      <- rnorm(n, sd = sqrt(1 / prec1_true))  # Level noise
#' u2      <- rnorm(n, sd = sqrt(1 / prec2_true))  # Trend noise
#' u3      <- rnorm(n, sd = sqrt(1 / prec3_true))  # Acceleration noise
#' epsilon <- rnorm(n, sd = sqrt(1 / prec_y_true)) # Observation noise
#'
#' # Simulate latent states and observations:
#' theta1_true    <- numeric(n)
#' theta2_true    <- numeric(n)
#' theta3_true    <- numeric(n)
#' theta3_true[1] <- theta03_true + u3[1]
#' theta2_true[1] <- theta02_true + theta03_true + u2[1]
#' theta1_true[1] <- theta01_true + theta02_true + u1[1]
#' for (t in 2:n) {
#'   theta3_true[t] <- theta3_true[t-1] + u3[t]
#'   theta2_true[t] <- theta2_true[t-1] + theta3_true[t-1] + u2[t]
#'   theta1_true[t] <- theta1_true[t-1] + theta2_true[t-1] + u1[t]
#' }
#' y <- theta1_true + epsilon # Observed data
#'
#' ## Running the Gibbs sampler
#' # Run the Gibbs sampler with specified priors and a seed
#' out <- mcmc_normal_localacceleration(
#'   y,
#'   burnin             = 2000,
#'   thinning           = 100,
#'   n_chain            = 1000,
#'   prior_theta01_mean = y[1],
#'   prior_theta01_prec = 1 / var(y),
#'   prior_theta02_mean = y[1] / 2,
#'   prior_theta02_prec = 1 / var(y),
#'   prior_theta03_mean = 0,
#'   prior_theta03_prec = 1e-3,
#'   prior_prec1_shape  = 1e-1,
#'   prior_prec1_rate   = 1e-1,
#'   prior_prec2_shape  = 1e-2,
#'   prior_prec2_rate   = 1e-2,
#'   prior_prec3_shape  = 1e-1,
#'   prior_prec3_rate   = 1e-2,
#'   prior_prec_y_shape = 1e-1,
#'   prior_prec_y_rate  = 1e-1,
#'   verbose            = TRUE,
#'   bar_width          = 60,
#'   seed               = 456
#' )
#'
#' ## Posterior analysis and visualization
#' # The following plots show how to analyze the posterior distributions.
#' # Point estimates are based on the median of posterior samples.
#' \donttest{
#'   # --- 0. Plot the simulated data ---
#'   plot.ts(
#'     y,
#'     main = "Simulated data",
#'     ylab = expression(y[t]),
#'     xlab = "t"
#'   )
#'
#'   # --- 1. Latent Level (theta[t,1]) ---
#'
#'   # Visualize trajectories from the first few posterior samples
#'   num_traj_to_plot <- 20
#'   matplot(
#'     t(out$theta_1[1:num_traj_to_plot, ]),
#'     type = "l",
#'     lty = 1,
#'     col = grDevices::rainbow(num_traj_to_plot, alpha = 0.5),
#'     xlab = "t",
#'     ylab = expression(theta["t,1"]),
#'     main = "Sampled trajectories for latent level"
#'   )
#'
#'   # Plot true and estimated (median) latent level
#'   theta_1_estimate <- apply(X = out$theta_1, MARGIN = 2, FUN = median)
#'   range_theta_1 <- range(theta_1_estimate, theta1_true)
#'   r1_theta1 <- range_theta_1[1]
#'   r2_theta1 <- range_theta_1[2] + 0.25 * diff(range_theta_1)
#'
#'   plot.ts(
#'     theta1_true,
#'     col = "red",
#'     type = "l",
#'     xlab = "t",
#'     ylim = c(r1_theta1, r2_theta1),
#'     lty = 2,
#'     ylab = expression(theta["t,1"]),
#'     main = "Latent level"
#'   )
#'   points(theta_1_estimate, type = "l")
#'   legend(
#'     "topright",
#'     legend = c(expression(theta["t,1"]), expression(hat(theta)["t,1"])),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     bty = "n"
#'   )
#'
#'   # --- 2. Latent Trend (theta[t,2]) ---
#'   theta_2_estimate <- apply(X = out$theta_2, MARGIN = 2, FUN = median)
#'   range_theta_2 <- range(theta_2_estimate, theta2_true)
#'   r1_theta2 <- range_theta_2[1]
#'   r2_theta2 <- range_theta_2[2] + 0.25 * diff(range_theta_2)
#'
#'   plot.ts(
#'     theta2_true,
#'     col = "red",
#'     type = "l",
#'     xlab = "t",
#'     ylim = c(r1_theta2, r2_theta2),
#'     lty = 2,
#'     ylab = expression(theta["t,2"]),
#'     main = "Latent trend"
#'   )
#'   points(theta_2_estimate, type = "l")
#'   legend(
#'     "topright",
#'     legend = c(expression(theta["t,2"]), expression(hat(theta)["t,2"])),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     bty = "n"
#'   )
#'
#'   # --- 3. Latent Acceleration (theta[t,3]) ---
#'   theta_3_estimate <- apply(X = out$theta_3, MARGIN = 2, FUN = median)
#'   range_theta_3 <- range(theta_3_estimate, theta3_true)
#'   r1_theta3 <- range_theta_3[1]
#'   r2_theta3 <- range_theta_3[2] + 0.25 * diff(range_theta_3)
#'
#'   plot.ts(
#'     theta3_true,
#'     col = "red",
#'     type = "l",
#'     xlab = "t",
#'     ylim = c(r1_theta3, r2_theta3),
#'     lty = 2,
#'     ylab = expression(theta["t,3"]),
#'     main = "Latent acceleration"
#'   )
#'   points(theta_3_estimate, type = "l")
#'   legend(
#'     "topright",
#'     legend = c(expression(theta["t,3"]), expression(hat(theta)["t,3"])),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     bty = "n"
#'   )
#'
#'   # --- 4. Initial Level (theta[0,1]) ---
#'   # Trace plot
#'   range_theta_01 <- range(out$theta_01, theta01_true)
#'   r1_theta01 <- range_theta_01[1]
#'   r2_theta01 <- range_theta_01[2] + 0.25 * diff(range_theta_01)
#'
#'   plot.ts(
#'     out$theta_01,
#'     col = "gray",
#'     xlab = "Iterations",
#'     ylab = expression(theta["0,1"]),
#'     main = "Trace Plot of Initial Level",
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
#'     bty = "n",
#'     legend = c(expression(theta["0,1"]), expression(hat(theta)["0,1"])),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'
#'   # Density plot
#'   plot(
#'     density(out$theta_01),
#'     main = "Posterior Density of Initial Level",
#'     xlab = expression(theta["0,1"])
#'   )
#'   abline(
#'     v = c(theta01_true, median(out$theta_01)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'   legend(
#'     "topright",
#'     bty = "n",
#'     legend = c(expression(theta["0,1"]), expression(hat(theta)["0,1"])),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'
#'   # --- 5. Initial Trend (theta[0,2]) ---
#'   # Trace plot
#'   range_theta_02 <- range(out$theta_02, theta02_true)
#'   r1_theta02 <- range_theta_02[1]
#'   r2_theta02 <- range_theta_02[2] + 0.25 * diff(range_theta_02)
#'
#'   plot.ts(
#'     out$theta_02,
#'     col = "gray",
#'     xlab = "Iterations",
#'     ylab = expression(theta["0,2"]),
#'     main = "Trace Plot of Initial Trend",
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
#'     bty = "n",
#'     legend = c(expression(theta["0,2"]), expression(hat(theta)["0,2"])),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'
#'   # Density plot
#'   plot(
#'     density(out$theta_02),
#'     main = "Posterior Density of Initial Trend",
#'     xlab = expression(theta["0,2"])
#'   )
#'   abline(
#'     v = c(theta02_true, median(out$theta_02)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'   legend(
#'     "topright",
#'     bty = "n",
#'     legend = c(expression(theta["0,2"]), expression(hat(theta)["0,2"])),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'
#'   # --- 6. Initial Acceleration (theta[0,3]) ---
#'   # Trace plot
#'   range_theta_03 <- range(out$theta_03, theta03_true)
#'   r1_theta03 <- range_theta_03[1]
#'   r2_theta03 <- range_theta_03[2] + 0.25 * diff(range_theta_03)
#'
#'   plot.ts(
#'     out$theta_03,
#'     col = "gray",
#'     xlab = "Iterations",
#'     ylab = expression(theta["0,3"]),
#'     main = "Trace Plot of Initial Acceleration",
#'     ylim = c(r1_theta03, r2_theta03)
#'   )
#'   abline(
#'     h = c(theta03_true, median(out$theta_03)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'   legend(
#'     "topright",
#'     bty = "n",
#'     legend = c(expression(theta["0,3"]), expression(hat(theta)["0,3"])),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'
#'   # Density plot
#'   plot(
#'     density(out$theta_03),
#'     main = "Posterior Density of Initial Acceleration",
#'     xlab = expression(theta["0,3"])
#'   )
#'   abline(
#'     v = c(theta03_true, median(out$theta_03)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'   legend(
#'     "topright",
#'     bty = "n",
#'     legend = c(expression(theta["0,3"]), expression(hat(theta)["0,3"])),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'
#'   # --- 7. Level Precision (1/W_1) ---
#'   # Trace plot
#'   range_prec_theta1 <- range(out$prec_theta1, prec1_true)
#'   r1_prec1 <- range_prec_theta1[1]
#'   r2_prec1 <- range_prec_theta1[2] + 0.25 * diff(range_prec_theta1)
#'
#'   plot.ts(
#'     out$prec_theta1,
#'     col = "gray",
#'     xlab = "Iterations",
#'     ylab = expression(1/W[1]),
#'     main = "Trace Plot of Level Precision",
#'     ylim = c(r1_prec1, r2_prec1)
#'   )
#'   abline(
#'     h = c(prec1_true, median(out$prec_theta1)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'   legend(
#'     "topright",
#'     bty = "n",
#'     legend = c(expression(W[1]^-1), expression(hat(W)[1]^-1)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'
#'   # Density plot
#'   plot(
#'     density(out$prec_theta1),
#'     main = "Posterior Density of Level Precision",
#'     xlab = expression(W[1]^-1)
#'   )
#'   abline(
#'     v = c(prec1_true, median(out$prec_theta1)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'   legend(
#'     "topright",
#'     bty = "n",
#'     legend = c(expression(W[1]^-1), expression(hat(W)[1]^-1)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'
#'   # --- 8. Trend Precision (1/W_2) ---
#'   # Trace plot
#'   range_prec_theta2 <- range(out$prec_theta2, prec2_true)
#'   r1_prec2 <- range_prec_theta2[1]
#'   r2_prec2 <- range_prec_theta2[2] + 0.25 * diff(range_prec_theta2)
#'
#'   plot.ts(
#'     out$prec_theta2,
#'     col = "gray",
#'     xlab = "Iterations",
#'     ylab = expression(1/W[2]),
#'     main = "Trace Plot of Trend Precision",
#'     ylim = c(r1_prec2, r2_prec2)
#'   )
#'   abline(
#'     h = c(prec2_true, median(out$prec_theta2)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'   legend(
#'     "topright",
#'     bty = "n",
#'     legend = c(expression(W[2]^-1), expression(hat(W)[2]^-1)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'
#'   # Density plot
#'   plot(
#'     density(out$prec_theta2),
#'     main = "Posterior Density of Trend Precision",
#'     xlab = expression(W[2]^-1)
#'   )
#'   abline(
#'     v = c(prec2_true, median(out$prec_theta2)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'   legend(
#'     "topright",
#'     bty = "n",
#'     legend = c(expression(W[2]^-1), expression(hat(W)[2]^-1)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'
#'   # --- 9. Acceleration Precision (1/W_3) ---
#'   # Trace plot
#'   range_prec_theta3 <- range(out$prec_theta3, prec3_true)
#'   r1_prec3 <- range_prec_theta3[1]
#'   r2_prec3 <- range_prec_theta3[2] + 0.25 * diff(range_prec_theta3)
#'
#'   plot.ts(
#'     out$prec_theta3,
#'     col = "gray",
#'     xlab = "Iterations",
#'     ylab = expression(1/W[3]),
#'     main = "Trace Plot of Acceleration Precision",
#'     ylim = c(r1_prec3, r2_prec3)
#'   )
#'   abline(
#'     h = c(prec3_true, median(out$prec_theta3)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'   legend(
#'     "topright",
#'     bty = "n",
#'     legend = c(expression(W[3]^-1), expression(hat(W)[3]^-1)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'
#'   # Density plot
#'   plot(
#'     density(out$prec_theta3),
#'     main = "Posterior Density of Acceleration Precision",
#'     xlab = expression(W[3]^-1)
#'   )
#'   abline(
#'     v = c(prec3_true, median(out$prec_theta3)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'   legend(
#'     "topright",
#'     bty = "n",
#'     legend = c(expression(W[3]^-1), expression(hat(W)[3]^-1)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'
#'   # --- 10. Observation Precision (1/V) ---
#'   # Trace plot
#'   range_prec_y <- range(out$prec_y, prec_y_true)
#'   r1_prec_y <- range_prec_y[1]
#'   r2_prec_y <- range_prec_y[2] + 0.25 * diff(range_prec_y)
#'
#'   plot.ts(
#'     out$prec_y,
#'     col = "gray",
#'     xlab = "Iterations",
#'     ylab = expression(1/V),
#'     main = "Trace Plot of Observation Precision",
#'     ylim = c(r1_prec_y, r2_prec_y)
#'   )
#'   abline(
#'     h = c(prec_y_true, median(out$prec_y)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'   legend(
#'     "topright",
#'     bty = "n",
#'     legend = c(expression(V^-1), expression(hat(V)^-1)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'
#'   # Density plot
#'   plot(
#'     density(out$prec_y),
#'     main = "Posterior Density of Observation Precision",
#'     xlab = expression(V^-1)
#'   )
#'   abline(
#'     v = c(prec_y_true, median(out$prec_y)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#'   legend(
#'     "topright",
#'     bty = "n",
#'     legend = c(expression(V^-1), expression(hat(V)^-1)),
#'     col = c("red", "black"),
#'     lty = c(2, 1),
#'     lwd = 2
#'   )
#' }
#'
#' @seealso \link[pdm]{mcmc_normal_locallevel}, \link[pdm]{mcmc_normal_localtrend}
#' @export
#'
mcmc_normal_localacceleration <- function(y,
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
                                          prior_prec_y_shape,
                                          prior_prec_y_rate,
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
  if (!is.numeric(burnin) || length(burnin) != 1 || burnin < 0 || burnin != floor(burnin)) {
    stop("`burnin` must be a single non-negative integer")
  }
  if (!is.numeric(thinning) || length(thinning) != 1 || thinning < 1 || thinning != floor(thinning)) {
    stop("`thinning` must be a single positive integer")
  }
  if (!is.numeric(n_chain) || length(n_chain) != 1 || n_chain < 1 || n_chain != floor(n_chain)) {
    stop("`n_chain` must be a single positive integer")
  }
  # Priors for theta_01
  if (!is.numeric(prior_theta01_mean) || length(prior_theta01_mean) != 1) {
    stop("`prior_theta01_mean` must be a single numeric value")
  }
  if (!is.numeric(prior_theta01_prec) || length(prior_theta01_prec) != 1 || prior_theta01_prec <= 0) {
    stop("`prior_theta01_prec` must be a single positive numeric value")
  }
  # Priors for theta_02
  if (!is.numeric(prior_theta02_mean) || length(prior_theta02_mean) != 1) {
    stop("`prior_theta02_mean` must be a single numeric value")
  }
  if (!is.numeric(prior_theta02_prec) || length(prior_theta02_prec) != 1 || prior_theta02_prec <= 0) {
    stop("`prior_theta02_prec` must be a single positive numeric value")
  }
  # Priors for theta_03
  if (!is.numeric(prior_theta03_mean) || length(prior_theta03_mean) != 1) {
    stop("`prior_theta03_mean` must be a single numeric value")
  }
  if (!is.numeric(prior_theta03_prec) || length(prior_theta03_prec) != 1 || prior_theta03_prec <= 0) {
    stop("`prior_theta03_prec` must be a single positive numeric value")
  }
  # Priors for prec_theta1
  if (!is.numeric(prior_prec1_shape) || length(prior_prec1_shape) != 1 || prior_prec1_shape <= 0) {
    stop("`prior_prec1_shape` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec1_rate) || length(prior_prec1_rate) != 1 || prior_prec1_rate <= 0) {
    stop("`prior_prec1_rate` must be a single positive numeric value")
  }
  # Priors for prec_theta2
  if (!is.numeric(prior_prec2_shape) || length(prior_prec2_shape) != 1 || prior_prec2_shape <= 0) {
    stop("`prior_prec2_shape` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec2_rate) || length(prior_prec2_rate) != 1 || prior_prec2_rate <= 0) {
    stop("`prior_prec2_rate` must be a single positive numeric value")
  }
  # Priors for prec_theta3
  if (!is.numeric(prior_prec3_shape) || length(prior_prec3_shape) != 1 || prior_prec3_shape <= 0) {
    stop("`prior_prec3_shape` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec3_rate) || length(prior_prec3_rate) != 1 || prior_prec3_rate <= 0) {
    stop("`prior_prec3_rate` must be a single positive numeric value")
  }
  # Priors for prec_y
  if (!is.numeric(prior_prec_y_shape) || length(prior_prec_y_shape) != 1 || prior_prec_y_shape <= 0) {
    stop("`prior_prec_y_shape` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec_y_rate) || length(prior_prec_y_rate) != 1 || prior_prec_y_rate <= 0) {
    stop("`prior_prec_y_rate` must be a single positive numeric value")
  }

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
  result <- .Call(
    "_pdm_C_MCMC_normal_localacceleration",
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
    as.numeric(prior_prec_y_shape),
    as.numeric(prior_prec_y_rate),
    as.logical(verbose),
    as.integer(bar_width)
  )

  result <- new_normal_localacceleration(
    result = result,
    n_obs = length(y),
    n_chain = n_chain,
    burnin = burnin,
    thinning = thinning,
    y = y
  )

  result <- validate_normal_localacceleration(result)

  return(result)
}
