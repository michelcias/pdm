#' @title Gibbs Sampler for a Gaussian Mixture Model with Local-Acceleration Mixture Weights
#'
#' @description Runs a Gibbs sampler for a two-component Gaussian mixture model
#'   with time-varying mixture weights following a local-acceleration polynomial
#'   dynamic structure. Supports both logit and probit link functions.
#'
#' @details The model is defined as:
#' \deqn{
#' \begin{aligned}
#' y_t | z_t, \mu, \phi &\sim N\big(z_t \cdot \mu_2 + (1 - z_t) \cdot \mu_1,
#'                              [z_t \cdot \phi_2 + (1 - z_t) \cdot \phi_1]^{-1}\big), \\
#' z_t | \alpha_t &\sim \text{Bernoulli}(\alpha_t), \\
#' \alpha_t &= T^{-1}(\theta_{t,1}), \\
#' \theta_{t,1} &= \theta_{t-1,1} + \theta_{t-1,2} + u_{t,1}, & u_{t,1} &\sim N(0, W_1), \\
#' \theta_{t,2} &= \theta_{t-1,2} + \theta_{t-1,3} + u_{t,2}, & u_{t,2} &\sim N(0, W_2), \\
#' \theta_{t,3} &= \theta_{t-1,3} + u_{t,3},                  & u_{t,3} &\sim N(0, W_3),
#' \end{aligned}
#' }
#' where \eqn{t = 1, 2, \ldots, n}, \eqn{n} is the number of observations,
#' \eqn{z_t \in \{0, 1\}} are latent indicators (0 = component 1, 1 = component 2),
#' \eqn{\mu = (\mu_1, \mu_2)'} are component means, and \eqn{\phi = (\phi_1, \phi_2)'}
#' are component precisions.
#'
#' The link function \eqn{T^{-1}} determines how the latent level state \eqn{\theta_{t,1}}
#' maps to the mixture weight \eqn{\alpha_t}:
#'
#' \strong{Logit Link:}
#' \deqn{\alpha_t = \frac{\exp(\theta_{t,1})}{1 + \exp(\theta_{t,1})}}
#' Uses component-wise Metropolis-Hastings with adaptive tuning. Allows optional
#' diagnostics (log_sigma, accept_prop) to monitor proposal adaptation and
#' acceptance rates.
#'
#' \strong{Probit Link:}
#' \deqn{\alpha_t = \Phi(\theta_{t,1})}
#' where \eqn{\Phi(\cdot)} is the standard normal CDF. Uses Albert-Chib (1993)
#' data augmentation for efficient Gibbs sampling. Since this is a pure Gibbs
#' sampler, the acceptance rate is always 1.0.
#'
#' \strong{Prior Distributions:}
#'
#' The following conjugate prior distributions are employed:
#'
#' \emph{Mixture Component Parameters:}
#' \deqn{
#' \begin{aligned}
#' \mu_1 &\sim N(\mu_{01}, \tau_{01}^{-1}), \\
#' \mu_2 &\sim N(\mu_{02}, \tau_{02}^{-1}), \\
#' \phi_1 &\sim \text{Gamma}(\nu_{01}, \eta_{01}), \\
#' \phi_2 &\sim \text{Gamma}(\nu_{02}, \eta_{02}).
#' \end{aligned}
#' }
#'
#' \emph{Dynamic State Parameters:}
#' \deqn{
#' \begin{aligned}
#' \theta_{0,1} &\sim N(\mu_{\theta_{01}}, \tau_{\theta_{01}}^{-1}), \\
#' \theta_{0,2} &\sim N(\mu_{\theta_{02}}, \tau_{\theta_{02}}^{-1}), \\
#' \theta_{0,3} &\sim N(\mu_{\theta_{03}}, \tau_{\theta_{03}}^{-1}), \\
#' W_1^{-1} &\sim \text{Gamma}(\nu_1, \eta_1), \\
#' W_2^{-1} &\sim \text{Gamma}(\nu_2, \eta_2), \\
#' W_3^{-1} &\sim \text{Gamma}(\nu_3, \eta_3).
#' \end{aligned}
#' }
#'
#' The prior hyperparameters correspond to function arguments as follows:
#' \tabular{cc}{
#'   \strong{Hyperparameter} \tab \strong{Function Argument} \cr
#'   \eqn{\mu_{01}} \tab `prior_mu01_mean` \cr
#'   \eqn{\tau_{01}} \tab `prior_mu01_prec` \cr
#'   \eqn{\nu_{01}} \tab `prior_prec01_shape` \cr
#'   \eqn{\eta_{01}} \tab `prior_prec01_rate` \cr
#'   \eqn{\mu_{02}} \tab `prior_mu02_mean` \cr
#'   \eqn{\tau_{02}} \tab `prior_mu02_prec` \cr
#'   \eqn{\nu_{02}} \tab `prior_prec02_shape` \cr
#'   \eqn{\eta_{02}} \tab `prior_prec02_rate` \cr
#'   \eqn{\mu_{\theta_{01}}} \tab `prior_theta01_mean` \cr
#'   \eqn{\tau_{\theta_{01}}} \tab `prior_theta01_prec` \cr
#'   \eqn{\mu_{\theta_{02}}} \tab `prior_theta02_mean` \cr
#'   \eqn{\tau_{\theta_{02}}} \tab `prior_theta02_prec` \cr
#'   \eqn{\mu_{\theta_{03}}} \tab `prior_theta03_mean` \cr
#'   \eqn{\tau_{\theta_{03}}} \tab `prior_theta03_prec` \cr
#'   \eqn{\nu_1} \tab `prior_prec1_shape` \cr
#'   \eqn{\eta_1} \tab `prior_prec1_rate` \cr
#'   \eqn{\nu_2} \tab `prior_prec2_shape` \cr
#'   \eqn{\eta_2} \tab `prior_prec2_rate` \cr
#'   \eqn{\nu_3} \tab `prior_prec3_shape` \cr
#'   \eqn{\eta_3} \tab `prior_prec3_rate`
#' }
#'
#' \strong{Label Switching:}
#' To ensure identifiability, the constraint \eqn{\mu_1 < \mu_2} is enforced
#' by swapping components when necessary during sampling. This means component 1
#' always corresponds to the mixture component with the smaller mean.
#'
#' \strong{Adaptive Tuning (Logit Link Only):}
#' When using the logit link, the component-wise Metropolis-Hastings algorithm
#' employs adaptive proposal scaling to achieve the target acceptance rate.
#' The adaptation follows Roberts & Rosenthal (2009), with proposal variances
#' adjusted based on recent acceptance rates within a sliding window of size
#' `lag_update`. The adaptation rate decays over iterations to satisfy
#' diminishing adaptation conditions.
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
#' @param y Numeric vector of observed data (length \eqn{n}).
#' @param link Character string specifying the link function. Must be either
#'   `"logit"` or `"probit"`. Default is `"logit"`.
#' @param burnin Integer \eqn{\geq 0}, number of burn-in iterations.
#' @param thinning Integer \eqn{\geq 1}, thinning interval.
#' @param n_chain Integer \eqn{\geq 1}, number of posterior samples to retain.
#' @param prior_mu01_mean Numeric, prior mean for the mean of component 1 (\eqn{\mu_1}).
#'   If `NULL` (default), set to the 25th percentile of `y`.
#' @param prior_mu01_prec Numeric > 0, prior precision (inverse variance) for \eqn{\mu_1}.
#'   Default is 0.01 (vague prior).
#' @param prior_prec01_shape Numeric > 0, shape parameter of the Gamma prior for
#'   the precision of component 1 (\eqn{\phi_1}). Default is 0.01 (vague prior).
#' @param prior_prec01_rate Numeric > 0, rate parameter of the Gamma prior for \eqn{\phi_1}.
#'   Default is 0.01 (vague prior).
#' @param prior_mu02_mean Numeric, prior mean for the mean of component 2 (\eqn{\mu_2}).
#'   If `NULL` (default), set to the 75th percentile of `y`.
#' @param prior_mu02_prec Numeric > 0, prior precision (inverse variance) for \eqn{\mu_2}.
#'   Default is 0.01 (vague prior).
#' @param prior_prec02_shape Numeric > 0, shape parameter of the Gamma prior for
#'   the precision of component 2 (\eqn{\phi_2}). Default is 0.01 (vague prior).
#' @param prior_prec02_rate Numeric > 0, rate parameter of the Gamma prior for \eqn{\phi_2}.
#'   Default is 0.01 (vague prior).
#' @param prior_theta01_mean Numeric, prior mean for the initial level state
#'   \eqn{\theta_{0,1}}. Default is 0.
#' @param prior_theta01_prec Numeric > 0, prior precision (inverse variance) for
#'   \eqn{\theta_{0,1}}. Default is 0.01 (vague prior).
#' @param prior_theta02_mean Numeric, prior mean for the initial trend state
#'   \eqn{\theta_{0,2}}. Default is 0.
#' @param prior_theta02_prec Numeric > 0, prior precision (inverse variance) for
#'   \eqn{\theta_{0,2}}. Default is 0.01 (vague prior).
#' @param prior_theta03_mean Numeric, prior mean for the initial acceleration state
#'   \eqn{\theta_{0,3}}. Default is 0.
#' @param prior_theta03_prec Numeric > 0, prior precision (inverse variance) for
#'   \eqn{\theta_{0,3}}. Default is 0.01 (vague prior).
#' @param prior_prec1_shape Numeric > 0, shape parameter of the Gamma prior for
#'   the level innovation precision \eqn{1/W_1}. Default is 0.01 (vague prior).
#' @param prior_prec1_rate Numeric > 0, rate parameter of the Gamma prior for
#'   \eqn{1/W_1}. Default is 0.01 (vague prior).
#' @param prior_prec2_shape Numeric > 0, shape parameter of the Gamma prior for
#'   the trend innovation precision \eqn{1/W_2}. Default is 0.01 (vague prior).
#' @param prior_prec2_rate Numeric > 0, rate parameter of the Gamma prior for
#'   \eqn{1/W_2}. Default is 0.01 (vague prior).
#' @param prior_prec3_shape Numeric > 0, shape parameter of the Gamma prior for
#'   the acceleration innovation precision \eqn{1/W_3}. Default is 0.01 (vague prior).
#' @param prior_prec3_rate Numeric > 0, rate parameter of the Gamma prior for
#'   \eqn{1/W_3}. Default is 0.01 (vague prior).
#' @param lag_update Integer > 0, adaptation window size for the Metropolis-Hastings
#'   algorithm (logit link only). Acceptance rates are monitored over this window
#'   to adjust proposal scales. Default is 50. Ignored when `link = "probit"`.
#' @param max_step_size Numeric > 0, maximum step size for proposal scale adaptation
#'   (logit link only). Limits how much the log proposal scale can change in a
#'   single adaptation step. Default is 1.0. Ignored when `link = "probit"`.
#' @param base_adaptation_rate Numeric > 0, base rate for proposal adaptation
#'   (logit link only). Controls the magnitude of adaptation adjustments. Default
#'   is 0.01. Ignored when `link = "probit"`.
#' @param decay_exponent Numeric in (0.5, 1), decay exponent for adaptation rate
#'   (logit link only). The adaptation rate decays as \eqn{n^{-\text{decay\_exponent}}}
#'   to satisfy diminishing adaptation conditions. Default is 0.6. Ignored when
#'   `link = "probit"`.
#' @param target_acceptance Numeric in (0, 1), target acceptance rate for the
#'   Metropolis-Hastings algorithm (logit link only). Proposal scales are adjusted
#'   to achieve this rate. Default is 0.44 (optimal for univariate proposals).
#'   Ignored when `link = "probit"`.
#' @param min_deviation_threshold Numeric \eqn{\geq 0}, minimum deviation from
#'   target acceptance rate required to trigger adaptation (logit link only). If
#'   `NULL` (default), set to `1.0 / lag_update`. Ignored when `link = "probit"`.
#' @param return_log_sigma Logical, whether to return the log proposal scales
#'   (logit link only). Useful for diagnosing adaptation behavior. Default is
#'   `FALSE`. Ignored when `link = "probit"`.
#' @param return_accept_prop Logical, whether to return the acceptance proportions
#'   (logit link only). Useful for diagnosing MCMC mixing. Default is `FALSE`.
#'   Ignored when `link = "probit"`.
#' @param verbose Logical, whether to display a progress bar during sampling.
#'   Default is `FALSE`.
#' @param bar_width Integer in [10, 120], width of the progress bar when
#'   `verbose = TRUE`. Default is `60`.
#' @param seed Optional integer used to set the random number generator seed.
#'   Default is `NULL`, which does not set the seed.
#'
#' @return A list with components:
#' \describe{
#'   \item{`mu_1`}{Numeric vector of length `n_chain` of posterior samples for
#'     the mean of component 1 (\eqn{\mu_1}). Component 1 is defined as the
#'     component with the smaller mean due to the label switching constraint.}
#'   \item{`prec_1`}{Numeric vector of length `n_chain` of posterior samples for
#'     the precision of component 1 (\eqn{\phi_1 = 1/\sigma_1^2}).}
#'   \item{`mu_2`}{Numeric vector of length `n_chain` of posterior samples for
#'     the mean of component 2 (\eqn{\mu_2}). Component 2 is defined as the
#'     component with the larger mean.}
#'   \item{`prec_2`}{Numeric vector of length `n_chain` of posterior samples for
#'     the precision of component 2 (\eqn{\phi_2 = 1/\sigma_2^2}).}
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
#'     samples for the time-varying mixture weights \eqn{\alpha_t = P(z_t = 1)}.}
#'   \item{`z`}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior samples
#'     for the latent component indicators \eqn{z_t \in \{0, 1\}}. A value of 0
#'     indicates component 1 (smaller mean), and 1 indicates component 2 (larger mean).}
#'   \item{`log_sigma`}{(Optional, logit link only) Numeric matrix
#'     \eqn{[n_{chain} \times n]} of log proposal scales used in the
#'     Metropolis-Hastings algorithm. Only returned if `return_log_sigma = TRUE`
#'     and `link = "logit"`.}
#'   \item{`accept_prop`}{(Optional, logit link only) Numeric matrix
#'     \eqn{[n_{chain} \times n]} of acceptance proportions for each time point.
#'     Only returned if `return_accept_prop = TRUE` and `link = "logit"`.}
#' }
#'
#' @examples
#' ## Description
#' # This example demonstrates how to:
#' # 1. Simulate data from a Gaussian mixture with multi-frequency mixture weights
#' # 2. Use `mcmc_normal_mixture_localacceleration` to estimate parameters and latent states
#' # 3. Compare logit and probit link functions
#' # 4. Perform a detailed posterior analysis with visualizations
#' # 5. Set a seed for reproducibility
#' # 6. Use progress bar for monitoring MCMC execution
#'
#' ## Simulation of data
#' n <- 400  # Number of observations to simulate
#'
#' # Use a fixed seed for data simulation
#' set.seed(123)
#'
#' # Generate true mixture weights following a multi-frequency sinusoidal pattern
#' grid_vals <- seq_len(n) / n
#' alpha_true <- (sin(4 * pi * grid_vals) + sin(8 * pi * grid_vals) + 2) / 4
#'
#' # Generate latent component indicators
#' z_true <- rbinom(n, size = 1, prob = alpha_true)
#'
#' # Generate observations from mixture of N(0, 1/4) and N(2, 1/4)
#' mu_1_true <- 0
#' mu_2_true <- 2
#' sigma_1_true <- 0.5  # sqrt(1/4)
#' sigma_2_true <- 0.5  # sqrt(1/4)
#'
#' mu_y <- (1 - z_true) * mu_1_true + z_true * mu_2_true
#' sigma_y <- (1 - z_true) * sigma_1_true + z_true * sigma_2_true
#' y <- rnorm(n, mean = mu_y, sd = sigma_y)
#'
#' ## Running the Gibbs sampler with logit link
#' out_logit <- mcmc_normal_mixture_localacceleration(
#'   y,
#'   link               = "logit",
#'   burnin             = 2000,
#'   thinning           = 10,
#'   n_chain            = 1000,
#'   prior_mu01_mean    = NULL,  # Use default (25th percentile)
#'   prior_mu01_prec    = 0.01,
#'   prior_prec01_shape = 0.01,
#'   prior_prec01_rate  = 0.01,
#'   prior_mu02_mean    = NULL,  # Use default (75th percentile)
#'   prior_mu02_prec    = 0.01,
#'   prior_prec02_shape = 0.01,
#'   prior_prec02_rate  = 0.01,
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
#'   prior_prec3_shape  = 900,
#'   prior_prec3_rate   = 1,
#'   lag_update         = 50,
#'   max_step_size      = 1.0,
#'   base_adaptation_rate = 0.01,
#'   decay_exponent     = 0.6,
#'   target_acceptance  = 0.44,
#'   min_deviation_threshold = NULL,  # Use default (1/lag_update)
#'   return_log_sigma   = FALSE,
#'   return_accept_prop = FALSE,
#'   verbose            = TRUE,
#'   bar_width          = 60,
#'   seed               = 456
#' )
#'
#' ## Running the Gibbs sampler with probit link
#' out_probit <- mcmc_normal_mixture_localacceleration(
#'   y,
#'   link               = "probit",
#'   burnin             = 2000,
#'   thinning           = 10,
#'   n_chain            = 1000,
#'   prior_mu01_mean    = NULL,
#'   prior_mu01_prec    = 0.01,
#'   prior_prec01_shape = 0.01,
#'   prior_prec01_rate  = 0.01,
#'   prior_mu02_mean    = NULL,
#'   prior_mu02_prec    = 0.01,
#'   prior_prec02_shape = 0.01,
#'   prior_prec02_rate  = 0.01,
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
#'   prior_prec3_shape  = 900,
#'   prior_prec3_rate   = 1,
#'   verbose            = TRUE,
#'   bar_width          = 60,
#'   seed               = 789
#' )
#'
#' ## Posterior analysis and visualization
#' # The following plots show how to analyze the posterior distributions.
#' # Point estimates are based on the median of posterior samples.
#' \dontrun{
#'   # --- 0. Plot the simulated data with true components ---
#'
#'   range_y <- range(y)
#'   r1_y <- range_y[1] - 0.1 * diff(range_y)
#'   r2_y <- range_y[2] + 0.1 * diff(range_y)
#'
#'   plot(
#'     y,
#'     main = "Simulated Gaussian mixture data",
#'     ylab = expression(y[t]),
#'     xlab = "t",
#'     type = "p",
#'     pch = 16,
#'     cex = 0.6,
#'     ylim = c(r1_y, r2_y),
#'     col = ifelse(z_true == 1, "red", "blue")
#'   )
#'   # Overlay the true component means
#'   abline(h = mu_1_true, col = "blue", lwd = 2, lty = 2)
#'   abline(h = mu_2_true, col = "red", lwd = 2, lty = 2)
#'   legend(
#'     "topleft",
#'     legend = c(
#'       "Component 1",
#'       "Component 2"
#'     ),
#'     col = c("blue", "red"),
#'     lty = c(NA, NA),
#'     pch = c(16, 16),
#'     lwd = c(NA, NA),
#'     bty = "n"
#'   )
#'   legend(
#'     "topright",
#'     legend = c(
#'       expression(mu[1]),
#'       expression(mu[2])
#'     ),
#'     col = c("blue", "red"),
#'     lty = c(2, 2),
#'     pch = c(NA, NA),
#'     lwd = c(2, 2),
#'     bty = "n"
#'   )
#'
#'   # --- 1. Mixture Weights (alpha[t]) - Logit vs Probit (side by side) ---
#'   alpha_logit_estimate <- apply(X = out_logit$alpha, MARGIN = 2, FUN = median)
#'   alpha_probit_estimate <- apply(X = out_probit$alpha, MARGIN = 2, FUN = median)
#'   alpha_logit_q025 <- apply(X = out_logit$alpha, MARGIN = 2, FUN = quantile, probs = 0.025)
#'   alpha_logit_q975 <- apply(X = out_logit$alpha, MARGIN = 2, FUN = quantile, probs = 0.975)
#'   alpha_probit_q025 <- apply(X = out_probit$alpha, MARGIN = 2, FUN = quantile, probs = 0.025)
#'   alpha_probit_q975 <- apply(X = out_probit$alpha, MARGIN = 2, FUN = quantile, probs = 0.975)
#'
#'   par(mfrow = c(1, 2))
#'
#'   # Logit
#'   range_alpha_logit <- range(alpha_true, alpha_logit_estimate,
#'                              alpha_logit_q025, alpha_logit_q975)
#'   r1_alpha_logit <- range_alpha_logit[1] - 0.05
#'   r2_alpha_logit <- range_alpha_logit[2] + 0.30 * diff(range_alpha_logit)
#'
#'   plot(
#'     alpha_true,
#'     col = "black",
#'     type = "l",
#'     lwd = 3,
#'     xlab = "t",
#'     ylim = c(r1_alpha_logit, r2_alpha_logit),
#'     lty = 1,
#'     ylab = expression(alpha[t]),
#'     main = "Mixture weights: logit link"
#'   )
#'   polygon(
#'     c(1:length(alpha_logit_estimate), rev(1:length(alpha_logit_estimate))),
#'     c(alpha_logit_q025, rev(alpha_logit_q975)),
#'     col = rgb(0.2, 0.5, 0.8, alpha = 0.2),
#'     border = NA
#'   )
#'   lines(alpha_logit_estimate, col = "blue", lwd = 2, lty = 2)
#'   legend(
#'     "topright",
#'     legend = c(
#'       expression(alpha[t]),
#'       expression(hat(alpha)[t]),
#'       "95% Credible Interval"
#'     ),
#'     col = c("black", "blue", rgb(0.2, 0.5, 0.8, alpha = 0.2)),
#'     lty = c(1, 2, NA),
#'     lwd = c(3, 2, NA),
#'     pch = c(NA, NA, 15),
#'     pt.cex = c(NA, NA, 2),
#'     bty = "n"
#'   )
#'
#'   # Probit
#'   range_alpha_probit <- range(alpha_true, alpha_probit_estimate,
#'                               alpha_probit_q025, alpha_probit_q975)
#'   r1_alpha_probit <- range_alpha_probit[1] - 0.05
#'   r2_alpha_probit <- range_alpha_probit[2] + 0.30 * diff(range_alpha_probit)
#'
#'   plot(
#'     alpha_true,
#'     col = "black",
#'     type = "l",
#'     lwd = 3,
#'     xlab = "t",
#'     ylim = c(r1_alpha_probit, r2_alpha_probit),
#'     lty = 1,
#'     ylab = expression(alpha[t]),
#'     main = "Mixture weights: probit link"
#'   )
#'   polygon(
#'     c(1:length(alpha_probit_estimate), rev(1:length(alpha_probit_estimate))),
#'     c(alpha_probit_q025, rev(alpha_probit_q975)),
#'     col = rgb(0.8, 0.4, 0.2, alpha = 0.2),
#'     border = NA
#'   )
#'   lines(alpha_probit_estimate, col = "red", lwd = 2, lty = 2)
#'   legend(
#'     "topright",
#'     legend = c(
#'       expression(alpha[t]),
#'       expression(hat(alpha)[t]),
#'       "95% Credible Interval"
#'     ),
#'     col = c("black", "red", rgb(0.8, 0.4, 0.2, alpha = 0.2)),
#'     lty = c(1, 2, NA),
#'     lwd = c(3, 2, NA),
#'     pch = c(NA, NA, 15),
#'     pt.cex = c(NA, NA, 2),
#'     bty = "n"
#'   )
#'
#'   # --- 2. Latent Level Point Estimates (theta[t,1]) ---
#'   theta1_logit_estimate <- apply(X = out_logit$theta_1, MARGIN = 2, FUN = median)
#'   theta1_logit_q025 <- apply(X = out_logit$theta_1, MARGIN = 2, FUN = quantile, probs = 0.025)
#'   theta1_logit_q975 <- apply(X = out_logit$theta_1, MARGIN = 2, FUN = quantile, probs = 0.975)
#'   theta1_probit_estimate <- apply(X = out_probit$theta_1, MARGIN = 2, FUN = median)
#'   theta1_probit_q025 <- apply(X = out_probit$theta_1, MARGIN = 2, FUN = quantile, probs = 0.025)
#'   theta1_probit_q975 <- apply(X = out_probit$theta_1, MARGIN = 2, FUN = quantile, probs = 0.975)
#'
#'   par(mfrow = c(1, 2))
#'
#'   plot(
#'     theta1_logit_estimate,
#'     col = "blue",
#'     type = "l",
#'     lwd = 2,
#'     xlab = "t",
#'     ylab = expression(theta["t,1"]),
#'     main = "Latent level (logit link)"
#'   )
#'   polygon(
#'     c(1:length(theta1_logit_estimate), rev(1:length(theta1_logit_estimate))),
#'     c(theta1_logit_q025, rev(theta1_logit_q975)),
#'     col = rgb(0.2, 0.5, 0.8, alpha = 0.2),
#'     border = NA
#'   )
#'   lines(theta1_logit_estimate, col = "blue", lwd = 2)
#'
#'   plot(
#'     theta1_probit_estimate,
#'     col = "red",
#'     type = "l",
#'     lwd = 2,
#'     xlab = "t",
#'     ylab = expression(theta["t,1"]),
#'     main = "Latent level (probit link)"
#'   )
#'   polygon(
#'     c(1:length(theta1_probit_estimate), rev(1:length(theta1_probit_estimate))),
#'     c(theta1_probit_q025, rev(theta1_probit_q975)),
#'     col = rgb(0.8, 0.4, 0.2, alpha = 0.2),
#'     border = NA
#'   )
#'   lines(theta1_probit_estimate, col = "red", lwd = 2)
#'
#'   # --- 3. Latent Trend Point Estimates (theta[t,2]) ---
#'   theta2_logit_estimate <- apply(X = out_logit$theta_2, MARGIN = 2, FUN = median)
#'   theta2_logit_q025 <- apply(X = out_logit$theta_2, MARGIN = 2, FUN = quantile, probs = 0.025)
#'   theta2_logit_q975 <- apply(X = out_logit$theta_2, MARGIN = 2, FUN = quantile, probs = 0.975)
#'   theta2_probit_estimate <- apply(X = out_probit$theta_2, MARGIN = 2, FUN = median)
#'   theta2_probit_q025 <- apply(X = out_probit$theta_2, MARGIN = 2, FUN = quantile, probs = 0.025)
#'   theta2_probit_q975 <- apply(X = out_probit$theta_2, MARGIN = 2, FUN = quantile, probs = 0.975)
#'
#'   par(mfrow = c(1, 2))
#'
#'   plot(
#'     theta2_logit_estimate,
#'     col = "blue",
#'     type = "l",
#'     lwd = 2,
#'     xlab = "t",
#'     ylab = expression(theta["t,2"]),
#'     main = "Latent trend (logit link)"
#'   )
#'   polygon(
#'     c(1:length(theta2_logit_estimate), rev(1:length(theta2_logit_estimate))),
#'     c(theta2_logit_q025, rev(theta2_logit_q975)),
#'     col = rgb(0.2, 0.5, 0.8, alpha = 0.2),
#'     border = NA
#'   )
#'   lines(theta2_logit_estimate, col = "blue", lwd = 2)
#'
#'   plot(
#'     theta2_probit_estimate,
#'     col = "red",
#'     type = "l",
#'     lwd = 2,
#'     xlab = "t",
#'     ylab = expression(theta["t,2"]),
#'     main = "Latent trend (probit link)"
#'   )
#'   polygon(
#'     c(1:length(theta2_probit_estimate), rev(1:length(theta2_probit_estimate))),
#'     c(theta2_probit_q025, rev(theta2_probit_q975)),
#'     col = rgb(0.8, 0.4, 0.2, alpha = 0.2),
#'     border = NA
#'   )
#'   lines(theta2_probit_estimate, col = "red", lwd = 2)
#'
#'   # --- 4. Latent Acceleration Point Estimates (theta[t,3]) ---
#'   theta3_logit_estimate <- apply(X = out_logit$theta_3, MARGIN = 2, FUN = median)
#'   theta3_logit_q025 <- apply(X = out_logit$theta_3, MARGIN = 2, FUN = quantile, probs = 0.025)
#'   theta3_logit_q975 <- apply(X = out_logit$theta_3, MARGIN = 2, FUN = quantile, probs = 0.975)
#'   theta3_probit_estimate <- apply(X = out_probit$theta_3, MARGIN = 2, FUN = median)
#'   theta3_probit_q025 <- apply(X = out_probit$theta_3, MARGIN = 2, FUN = quantile, probs = 0.025)
#'   theta3_probit_q975 <- apply(X = out_probit$theta_3, MARGIN = 2, FUN = quantile, probs = 0.975)
#'
#'   par(mfrow = c(1, 2))
#'
#'   plot(
#'     theta3_logit_estimate,
#'     col = "blue",
#'     type = "l",
#'     lwd = 2,
#'     xlab = "t",
#'     ylab = expression(theta["t,3"]),
#'     main = "Latent acceleration (logit link)"
#'   )
#'   polygon(
#'     c(1:length(theta3_logit_estimate), rev(1:length(theta3_logit_estimate))),
#'     c(theta3_logit_q025, rev(theta3_logit_q975)),
#'     col = rgb(0.2, 0.5, 0.8, alpha = 0.2),
#'     border = NA
#'   )
#'   lines(theta3_logit_estimate, col = "blue", lwd = 2)
#'
#'   plot(
#'     theta3_probit_estimate,
#'     col = "red",
#'     type = "l",
#'     lwd = 2,
#'     xlab = "t",
#'     ylab = expression(theta["t,3"]),
#'     main = "Latent acceleration (probit link)"
#'   )
#'   polygon(
#'     c(1:length(theta3_probit_estimate), rev(1:length(theta3_probit_estimate))),
#'     c(theta3_probit_q025, rev(theta3_probit_q975)),
#'     col = rgb(0.8, 0.4, 0.2, alpha = 0.2),
#'     border = NA
#'   )
#'   lines(theta3_probit_estimate, col = "red", lwd = 2)
#'
#'   # --- 5. Posterior distributions of mixture parameters ---
#'   par(mfrow = c(2, 2))
#'   hist(out_logit$mu_1, breaks = 30, col = "lightblue",
#'        main = expression("Posterior of " ~ mu[1]), xlab = expression(mu[1]))
#'   hist(out_logit$mu_2, breaks = 30, col = "lightblue",
#'        main = expression("Posterior of " ~ mu[2]), xlab = expression(mu[2]))
#'   hist(out_logit$prec_1, breaks = 30, col = "lightblue",
#'        main = expression("Posterior of " ~ phi[1]), xlab = expression(phi[1]))
#'   hist(out_logit$prec_2, breaks = 30, col = "lightblue",
#'        main = expression("Posterior of " ~ phi[2]), xlab = expression(phi[2]))
#'
#'   par(mfrow = c(1, 1))
#' }
#'
#' @references
#' Albert, J. H., & Chib, S. (1993). Bayesian Analysis of Binary and Polychotomous
#' Response Data. \emph{Journal of the American Statistical Association}, 88(422), 669-679.
#' https://doi.org/10.1080/01621459.1993.10476321
#'
#' Montoril, M. H., Correia, L. T., & Migon, H. S. (2021). Bayesian estimation of
#' dynamic weights in Gaussian mixture models. arXiv:2104.03395.
#'
#' Roberts, G. O., & Rosenthal, J. S. (2009). Examples of Adaptive MCMC.
#' \emph{Journal of Computational and Graphical Statistics}, 18(2), 349-367.
#' https://doi.org/10.1198/jcgs.2009.06134
#'
#' @seealso \code{\link{mcmc_normal_mixture_locallevel}},
#'  \code{\link{mcmc_normal_mixture_localtrend}}
#'
#' @export
mcmc_normal_mixture_localacceleration <- function(y,
                                                  link = c("logit", "probit"),
                                                  burnin,
                                                  thinning,
                                                  n_chain,
                                                  prior_mu01_mean = NULL,
                                                  prior_mu01_prec = 0.01,
                                                  prior_prec01_shape = 0.01,
                                                  prior_prec01_rate = 0.01,
                                                  prior_mu02_mean = NULL,
                                                  prior_mu02_prec = 0.01,
                                                  prior_prec02_shape = 0.01,
                                                  prior_prec02_rate = 0.01,
                                                  prior_theta01_mean = 0,
                                                  prior_theta01_prec = 0.01,
                                                  prior_theta02_mean = 0,
                                                  prior_theta02_prec = 0.01,
                                                  prior_theta03_mean = 0,
                                                  prior_theta03_prec = 0.01,
                                                  prior_prec1_shape = 0.01,
                                                  prior_prec1_rate = 0.01,
                                                  prior_prec2_shape = 0.01,
                                                  prior_prec2_rate = 0.01,
                                                  prior_prec3_shape = 0.01,
                                                  prior_prec3_rate = 0.01,
                                                  lag_update = 50,
                                                  max_step_size = 1.0,
                                                  base_adaptation_rate = 0.01,
                                                  decay_exponent = 0.6,
                                                  target_acceptance = 0.44,
                                                  min_deviation_threshold = NULL,
                                                  return_log_sigma = FALSE,
                                                  return_accept_prop = FALSE,
                                                  verbose = FALSE,
                                                  bar_width = 60,
                                                  seed = NULL) {
  # --- Input Validation ---

  # Validate y
  if (!is.numeric(y)) {
    stop("`y` must be a numeric vector")
  }
  if (!all(is.finite(y))) {
    stop("`y` must contain only finite numeric values (no NA, NaN, Inf)")
  }
  if (length(y) < 3) {
    stop("`y` must have at least 3 observations")
  }

  # Validate and match link argument
  link <- match.arg(link)

  # Validate MCMC control parameters
  if (!is.numeric(burnin) || length(burnin) != 1 || burnin < 0 || burnin != floor(burnin)) {
    stop("`burnin` must be a single non-negative integer")
  }
  if (!is.numeric(thinning) || length(thinning) != 1 || thinning < 1 || thinning != floor(thinning)) {
    stop("`thinning` must be a single positive integer")
  }
  if (!is.numeric(n_chain) || length(n_chain) != 1 || n_chain < 1 || n_chain != floor(n_chain)) {
    stop("`n_chain` must be a single positive integer")
  }

  # Set intelligent defaults for mixture component priors based on data
  if (is.null(prior_mu01_mean)) {
    prior_mu01_mean <- as.numeric(quantile(y, 0.25))
  }
  if (is.null(prior_mu02_mean)) {
    prior_mu02_mean <- as.numeric(quantile(y, 0.75))
  }

  # Validate mixture component prior parameters
  if (!is.numeric(prior_mu01_mean) || length(prior_mu01_mean) != 1) {
    stop("`prior_mu01_mean` must be a single numeric value")
  }
  if (!is.numeric(prior_mu01_prec) || length(prior_mu01_prec) != 1 || prior_mu01_prec <= 0) {
    stop("`prior_mu01_prec` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec01_shape) || length(prior_prec01_shape) != 1 || prior_prec01_shape <= 0) {
    stop("`prior_prec01_shape` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec01_rate) || length(prior_prec01_rate) != 1 || prior_prec01_rate <= 0) {
    stop("`prior_prec01_rate` must be a single positive numeric value")
  }

  if (!is.numeric(prior_mu02_mean) || length(prior_mu02_mean) != 1) {
    stop("`prior_mu02_mean` must be a single numeric value")
  }
  if (!is.numeric(prior_mu02_prec) || length(prior_mu02_prec) != 1 || prior_mu02_prec <= 0) {
    stop("`prior_mu02_prec` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec02_shape) || length(prior_prec02_shape) != 1 || prior_prec02_shape <= 0) {
    stop("`prior_prec02_shape` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec02_rate) || length(prior_prec02_rate) != 1 || prior_prec02_rate <= 0) {
    stop("`prior_prec02_rate` must be a single positive numeric value")
  }

  # Validate dynamic state prior parameters
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
  if (!is.numeric(prior_theta03_mean) || length(prior_theta03_mean) != 1) {
    stop("`prior_theta03_mean` must be a single numeric value")
  }
  if (!is.numeric(prior_theta03_prec) || length(prior_theta03_prec) != 1 || prior_theta03_prec <= 0) {
    stop("`prior_theta03_prec` must be a single positive numeric value")
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
  if (!is.numeric(prior_prec3_shape) || length(prior_prec3_shape) != 1 || prior_prec3_shape <= 0) {
    stop("`prior_prec3_shape` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec3_rate) || length(prior_prec3_rate) != 1 || prior_prec3_rate <= 0) {
    stop("`prior_prec3_rate` must be a single positive numeric value")
  }

  # Validate adaptation parameters (used only for logit link)
  if (!is.numeric(lag_update) || length(lag_update) != 1 || lag_update < 1 || lag_update != floor(lag_update)) {
    stop("`lag_update` must be a single positive integer")
  }
  if (!is.numeric(max_step_size) || length(max_step_size) != 1 || max_step_size <= 0) {
    stop("`max_step_size` must be a single positive numeric value")
  }
  if (!is.numeric(base_adaptation_rate) || length(base_adaptation_rate) != 1 || base_adaptation_rate <= 0) {
    stop("`base_adaptation_rate` must be a single positive numeric value")
  }
  if (!is.numeric(decay_exponent) || length(decay_exponent) != 1 || decay_exponent <= 0.5 || decay_exponent >= 1) {
    stop("`decay_exponent` must be a single numeric value in (0.5, 1)")
  }
  if (!is.numeric(target_acceptance) || length(target_acceptance) != 1 || target_acceptance <= 0 || target_acceptance >= 1) {
    stop("`target_acceptance` must be a single numeric value in (0, 1)")
  }

  # Set default for min_deviation_threshold if NULL
  if (is.null(min_deviation_threshold)) {
    min_deviation_threshold <- 1.0 / lag_update
  }
  if (!is.numeric(min_deviation_threshold) || length(min_deviation_threshold) != 1 || min_deviation_threshold < 0) {
    stop("`min_deviation_threshold` must be a single non-negative numeric value")
  }

  # Validate diagnostic output flags
  if (!is.logical(return_log_sigma) || length(return_log_sigma) != 1) {
    stop("`return_log_sigma` must be a single logical value")
  }
  if (!is.logical(return_accept_prop) || length(return_accept_prop) != 1) {
    stop("`return_accept_prop` must be a single logical value")
  }

  # Validate progress bar parameters
  if (!is.logical(verbose) || length(verbose) != 1) {
    stop("`verbose` must be a single logical value")
  }
  if (!is.numeric(bar_width) || length(bar_width) != 1 || bar_width < 10 || bar_width > 120 || bar_width != floor(bar_width)) {
    stop("`bar_width` must be a single integer in [10, 120]")
  }

  # Validate and set seed if provided
  if (!is.null(seed)) {
    if (!is.numeric(seed) || length(seed) != 1 || seed != floor(seed)) {
      stop("`seed` must be a single integer value")
    }
    set.seed(seed)
  }

  # Issue warning if diagnostic flags are set for probit link (they will be ignored)
  if (link == "probit") {
    if (return_log_sigma) {
      warning("Argument `return_log_sigma` is ignored when link = 'probit'")
    }
    if (return_accept_prop) {
      warning("Argument `return_accept_prop` is ignored when link = 'probit'")
    }
  }

  # --- End Input Validation ---

  result <- .Call(
    "_pdm_C_MCMC_normal_mixture_localacceleration",
    as.numeric(y),
    as.character(link),
    as.integer(burnin),
    as.integer(thinning),
    as.integer(n_chain),
    as.numeric(prior_mu01_mean),
    as.numeric(prior_mu01_prec),
    as.numeric(prior_prec01_shape),
    as.numeric(prior_prec01_rate),
    as.numeric(prior_mu02_mean),
    as.numeric(prior_mu02_prec),
    as.numeric(prior_prec02_shape),
    as.numeric(prior_prec02_rate),
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
    as.integer(lag_update),
    as.numeric(max_step_size),
    as.numeric(base_adaptation_rate),
    as.numeric(decay_exponent),
    as.numeric(target_acceptance),
    as.numeric(min_deviation_threshold),
    as.logical(return_log_sigma),
    as.logical(return_accept_prop),
    as.logical(verbose),
    as.integer(bar_width)
  )

  result <- new_normal_mixture_localacceleration(
    result = result,
    link = link,
    n_obs = length(y),
    n_chain = n_chain,
    burnin = burnin,
    thinning = thinning,
    y = y
  )

  result <- validate_normal_mixture_localacceleration(result)

  return(result)
}
