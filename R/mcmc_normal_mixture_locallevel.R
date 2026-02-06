#' @title Gibbs Sampler for a Gaussian Mixture Model with Local-Level Mixture Weights
#'
#' @description Runs a Gibbs sampler for a two-component Gaussian mixture model
#'   with time-varying mixture weights governed by a local-level dynamic
#'   structure. Supports both logit and probit link functions.
#'
#' @details The model is defined as:
#' \deqn{
#' \begin{aligned}
#' y_t | z_t, \mu, \phi &\sim N(z_t \cdot \mu_2 + (1 - z_t) \cdot \mu_1,
#'                              [z_t \cdot \phi_2 + (1 - z_t) \cdot \phi_1]^{-1}), \\
#' z_t | \alpha_t &\sim \text{Bernoulli}(\alpha_t), \\
#' \alpha_t &= T^{-1}(\theta_{t,1}), \\
#' \theta_{t,1} &= \theta_{t-1,1} + u_{t,1}, & u_{t,1} \sim N(0, W_1),
#' \end{aligned}
#' }
#' where \eqn{t = 1, 2, \ldots, n}, \eqn{n} is the number of observations,
#' \eqn{z_t \in \{0, 1\}} are latent indicators (0 = component 1, 1 = component 2),
#' \eqn{\mu = (\mu_1, \mu_2)'} are component means, and \eqn{\phi = (\phi_1, \phi_2)'}
#' are component precisions.
#'
#' The link function \eqn{T^{-1}} determines how the latent state \eqn{\theta_{t,1}}
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
#' W_1^{-1} &\sim \text{Gamma}(\nu_1, \eta_1).
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
#'   \eqn{\nu_1} \tab `prior_prec1_shape` \cr
#'   \eqn{\eta_1} \tab `prior_prec1_rate`
#' }
#'
#' \strong{Label Switching:}
#' To ensure identifiability, the constraint \eqn{\mu_1 < \mu_2} is enforced
#' by swapping components when necessary during sampling. This means component 1
#' always corresponds to the mixture component with the smaller mean.
#'
#' \strong{Adaptive Metropolis-Hastings Algorithm (Logit Link Only):}
#'
#' When using the logit link, the algorithm employs component-wise Metropolis-Hastings
#' for sampling the latent states \eqn{\theta_{t,1}}, with adaptive proposal tuning
#' based on acceptance proportions.
#'
#' The adaptive algorithm uses a diminishing adaptation schedule that ensures
#' theoretical convergence guarantees (Roberts and Rosenthal, 2007). Adaptation
#' occurs every `lag_update` iterations (e.g., at MCMC iterations 50, 100, 150, ...
#' if `lag_update = 50`), and only after sufficient history has been accumulated
#' (iteration >= `lag_update`).
#'
#' At each adaptation point (MCMC iteration \eqn{m}), for each time point \eqn{t},
#' the proposal log-scale standard deviation is updated according to:
#'
#' \deqn{\log(\sigma_t) \leftarrow \log(\sigma_t) + \text{sign}(\hat{p}_t - p^*) \cdot \gamma_m \cdot \mathbb{1}_{\{|\hat{p}_t - p^*| > \tau\}},}
#'
#' where:
#' \itemize{
#'   \item \eqn{\gamma_m} is the diminishing step size at MCMC iteration \eqn{m}, computed as
#'     \deqn{\gamma_m = \min\left(\text{max\_step\_size}, \frac{\text{base\_adaptation\_rate}}{m^\xi}\right);}
#'   \item \eqn{\xi} is the decay exponent (`decay_exponent`) applied to the MCMC iteration number;
#'   \item \eqn{\hat{p}_t} is the empirical acceptance proportion at time point \eqn{t}
#'     over the last `lag_update` MCMC iterations;
#'   \item \eqn{p^*} is the target acceptance rate (`target_acceptance`);
#'   \item \eqn{\tau} is the minimum deviation threshold (`min_deviation_threshold`);
#'   \item \eqn{\mathbb{1}_{\{|\hat{p}_t - p^*| > \tau\}}} is an indicator function that equals 1 when
#'     the condition is true, 0 otherwise.
#' }
#'
#' The update only occurs if the absolute deviation exceeds the threshold:
#' \deqn{|\hat{p}_t - p^*| > \tau.}
#'
#' This threshold-based approach prevents spurious updates due to random fluctuations.
#'
#' \strong{Parameter Interactions:}
#'
#' The adaptive tuning parameters interact as follows:
#' \itemize{
#'   \item \strong{lag_update}: Controls both the sliding window size for computing acceptance
#'     proportions AND the frequency of adaptation. Adaptation occurs at MCMC iterations
#'     \eqn{m = k \cdot \text{lag\_update}} for \eqn{k = 1, 2, 3, \ldots}. Larger values
#'     provide more stable estimates but slower adaptation. Common choices: 50-200 iterations.
#'   \item \strong{target_acceptance}: Optimal acceptance rate for the Metropolis-Hastings
#'     algorithm. The value 0.44 is theoretically optimal for univariate random-walk proposals
#'     (Roberts and Rosenthal, 2001). Each time point \eqn{t} has its own acceptance rate
#'     \eqn{\hat{p}_t}.
#'   \item \strong{min_deviation_threshold}: Minimum deviation \eqn{|\hat{p}_t - p^*|}
#'     required to trigger adaptation for time point \eqn{t}. The default `NULL` uses the
#'     practical threshold \eqn{1/\text{lag\_update}}, corresponding to one additional
#'     acceptance/rejection in the sliding window.
#'   \item \strong{max_step_size}: Maximum allowed change in log-scale proposal variance
#'     per adaptation step. Prevents extreme adjustments. Common choices: 0.01-0.1.
#'   \item \strong{base_adaptation_rate}: Controls the overall speed of adaptation before
#'     decay is applied. Higher values lead to faster but potentially less stable adaptation.
#'     Common choices: 0.1-10.0.
#'   \item \strong{decay_exponent}: Controls how quickly the adaptation step size diminishes
#'     over MCMC iterations. As the algorithm runs, \eqn{\gamma_m} decreases according to
#'     \eqn{m^{-\xi}}. Must be in (0.5, 1] for theoretical convergence guarantees.
#'     Common choices: 0.5-0.8.
#' }
#'
#' \strong{Example of Adaptation Schedule:}
#'
#' With `lag_update = 50`, `base_adaptation_rate = 1.0`, `decay_exponent = 0.6`,
#' and `max_step_size = 0.1`:
#' \itemize{
#'   \item At MCMC iteration 50: \eqn{\gamma_{50} = \min(0.1, 1.0/50^{0.6}) \approx 0.0875}.
#'   \item At MCMC iteration 100: \eqn{\gamma_{100} = \min(0.1, 1.0/100^{0.6}) \approx 0.0631}.
#'   \item At MCMC iteration 1000: \eqn{\gamma_{1000} = \min(0.1, 1.0/1000^{0.6}) \approx 0.0158}.
#' }
#'
#' This ensures that adaptation becomes increasingly conservative as the chain progresses,
#' satisfying theoretical requirements for ergodicity.
#'
#' \strong{Recommended Settings:}
#'
#' For most applications:
#' \itemize{
#'   \item `lag_update = 50`: Provides good balance between stability and responsiveness;
#'   \item `max_step_size = 0.1`: Conservative adjustment rate;
#'   \item `base_adaptation_rate = 1.0`: Moderate adaptation speed;
#'   \item `decay_exponent = 0.6`: Standard diminishing adaptation;
#'   \item `target_acceptance = 0.44`: Theoretically optimal for univariate proposals;
#'   \item `min_deviation_threshold = NULL`: Uses practical default of 1/lag_update.
#' }
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
#' @param prior_prec1_shape Numeric > 0, shape parameter of the Gamma prior for
#'   the level innovation precision \eqn{1/W_1}. Default is 0.01 (vague prior).
#' @param prior_prec1_rate Numeric > 0, rate parameter of the Gamma prior for
#'   \eqn{1/W_1}. Default is 0.01 (vague prior).
#' @param lag_update Integer \eqn{\geq 1}, adaptation frequency (sliding window size) for
#'   computing acceptance proportions (logit link only). Adaptation occurs at MCMC iterations
#'   \eqn{m = k \cdot \text{lag\_update}} for \eqn{k = 1, 2, 3, \ldots}. Default is 50.
#'   Ignored when `link = "probit"`.
#' @param max_step_size Numeric > 0, maximum allowed change in log-scale proposal variance
#'   per adaptation step (logit link only). Prevents extreme adjustments. Default is 1.0.
#'   Ignored when `link = "probit"`.
#' @param base_adaptation_rate Numeric > 0, base rate controlling adaptation speed
#'   (logit link only). Higher values lead to faster but potentially less stable adaptation.
#'   Default is 0.01. Ignored when `link = "probit"`.
#' @param decay_exponent Numeric > 0, exponent controlling diminishing adaptation
#'   rate (logit link only). The adaptation rate decays as \eqn{m^{-\text{decay\_exponent}}}
#'   to satisfy diminishing adaptation conditions. Must be in (0.5, 1] for theoretical
#'   convergence guarantees. Default is 0.6. Ignored when `link = "probit"`.
#' @param target_acceptance Numeric in (0, 1), target acceptance rate for the
#'   Metropolis-Hastings algorithm (logit link only). The value 0.44 is theoretically
#'   optimal for univariate random-walk proposals. Default is 0.44. Ignored when
#'   `link = "probit"`.
#' @param min_deviation_threshold Numeric \eqn{\geq 0} or `NULL`, minimum absolute deviation
#'   from `target_acceptance` required to trigger adaptation (logit link only). If `NULL`
#'   (default), uses practical threshold of `1.0/lag_update`. Set to `0.0` for maximum
#'   sensitivity (adapt for any deviation). Larger values make adaptation more conservative.
#'   Ignored when `link = "probit"`.
#' @param return_log_sigma Logical, whether to return proposal scale diagnostics
#'   (log-scale proposal standard deviations) for logit link only. Useful for diagnosing
#'   adaptation behavior. Default is `FALSE`. Ignored when `link = "probit"`.
#' @param return_accept_prop Logical, whether to return acceptance proportion diagnostics
#'   over iterations for logit link only. Useful for diagnosing MCMC mixing. Default is
#'   `FALSE`. Ignored when `link = "probit"`.
#' @param verbose Logical, whether to display a progress bar during sampling. Default is `FALSE`.
#' @param bar_width Integer in [10, 120], width of the progress bar when `verbose = TRUE`.
#'   Default is `60`.
#' @param seed Optional integer used to set the random number generator seed for
#'   reproducibility. Default is `NULL` (no seed set).
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
#'   \item{`theta_01`}{Numeric vector of length `n_chain` of posterior samples
#'     for the initial level state \eqn{\theta_{0,1}}.}
#'   \item{`prec_theta1`}{Numeric vector of length `n_chain` of posterior samples
#'     for the level innovation precision \eqn{1/W_1}.}
#'   \item{`alpha`}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior
#'     samples for the mixture weights \eqn{\alpha_t}.}
#'   \item{`z`}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior samples
#'     for the latent component indicators \eqn{z_t}.}
#'   \item{`log_sigma`}{(Logit link only, optional) Numeric matrix of proposal
#'     log standard deviations used in the adaptive Metropolis-Hastings steps.}
#'   \item{`accept_prop`}{(Logit link only, optional) Numeric matrix of proposal
#'     acceptance proportions accumulated during adaptive tuning.}
#' }
#'
#' @examples
#' ## Description
#' # This example demonstrates how to:
#' # 1. Simulate data from a Gaussian mixture with sinusoidal mixture weights
#' # 2. Use `mcmc_normal_mixture_locallevel` to estimate parameters and latent states
#' # 3. Compare logit and probit link functions
#' # 4. Perform a detailed posterior analysis with visualizations
#' # 5. Set a seed for reproducibility
#' # 6. Use a progress bar for monitoring MCMC execution
#'
#' ## Simulation of data
#' n <- 400  # Number of observations to simulate
#'
#' # Use a fixed seed for data simulation
#' set.seed(123)
#'
#' # Generate true mixture weights following a sinusoidal pattern
#' grid_vals <- seq_len(n) / n
#' alpha_true <- (sin(2 * pi * grid_vals) + 2) / 4
#'
#' # Generate latent component indicators
#' z_true <- rbinom(n, size = 1, prob = alpha_true)
#'
#' # True component parameters for simulation
#' mu_1_true <- 0
#' mu_2_true <- 2
#' phi_1_true <- 4
#' phi_2_true <- 1
#' sigma_1_true <- 1 / sqrt(phi_1_true)
#' sigma_2_true <- 1 / sqrt(phi_2_true)
#'
#' # Generate observations from the Gaussian mixture
#' mu_y <- (1 - z_true) * mu_1_true + z_true * mu_2_true
#' sigma_y <- (1 - z_true) * sigma_1_true + z_true * sigma_2_true
#' y <- rnorm(n, mean = mu_y, sd = sigma_y)
#'
#' ## Running the Gibbs sampler with logit link
#' out_logit <- mcmc_normal_mixture_locallevel(
#'   y,
#'   link                    = "logit",
#'   burnin                  = 2000,
#'   thinning                = 10,
#'   n_chain                 = 1000,
#'   prior_mu01_mean         = NULL,
#'   prior_mu01_prec         = 0.01,
#'   prior_prec01_shape      = 0.01,
#'   prior_prec01_rate       = 0.01,
#'   prior_mu02_mean         = NULL,
#'   prior_mu02_prec         = 0.01,
#'   prior_prec02_shape      = 0.01,
#'   prior_prec02_rate       = 0.01,
#'   prior_theta01_mean      = 0,
#'   prior_theta01_prec      = 1,
#'   prior_prec1_shape       = 100,
#'   prior_prec1_rate        = 1,
#'   lag_update              = 50,
#'   max_step_size           = 1.0,
#'   base_adaptation_rate    = 0.01,
#'   decay_exponent          = 0.6,
#'   target_acceptance       = 0.44,
#'   min_deviation_threshold = NULL,
#'   return_log_sigma        = FALSE,
#'   return_accept_prop      = FALSE,
#'   verbose                 = TRUE,
#'   bar_width               = 60,
#'   seed                    = 456
#' )
#'
#' ## Running the Gibbs sampler with probit link
#' out_probit <- mcmc_normal_mixture_locallevel(
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
#'   prior_prec1_shape  = 100,
#'   prior_prec1_rate   = 1,
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
#'       "95% CI"
#'     ),
#'     col = c("black", "blue", rgb(0.2, 0.5, 0.8, alpha = 0.5)),
#'     lty = c(1, 2, 1),
#'     lwd = c(3, 2, 8),
#'     bty = "n"
#'   )
#'
#'   # Probit
#'   range_alpha_probit <- range(alpha_true, alpha_probit_estimate,
#'                               alpha_probit_q025, alpha_probit_q975)
#'   r1_alpha_probit <- range_alpha_probit[1] - 0.05
#'   r2_alpha_probit <- range_alpha_probit[2] + 0.25 * diff(range_alpha_probit)
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
#'     col = rgb(0.8, 0.2, 0.5, alpha = 0.2),
#'     border = NA
#'   )
#'   lines(alpha_probit_estimate, col = "red", lwd = 2, lty = 2)
#'   legend(
#'     "topright",
#'     legend = c(
#'       expression(alpha[t]),
#'       expression(hat(alpha)[t]),
#'       "95% CI"
#'     ),
#'     col = c("black", "red", rgb(0.8, 0.2, 0.5, alpha = 0.5)),
#'     lty = c(1, 2, 1),
#'     lwd = c(3, 2, 8),
#'     bty = "n"
#'   )
#'
#'   par(mfrow = c(1, 1))
#'
#'   # --- 2. Latent Indicators (z[t]) - Logit vs Probit (side by side) ---
#'   z_prob_logit <- apply(X = out_logit$z, MARGIN = 2, FUN = median)
#'   z_prob_probit <- apply(X = out_probit$z, MARGIN = 2, FUN = median)
#'
#'   par(mfrow = c(1, 2))
#'
#'   # Logit
#'   plot(
#'     z_true,
#'     col = "black",
#'     type = "p",
#'     pch = 16,
#'     cex = 0.8,
#'     xlab = "t",
#'     ylim = c(-0.1, 1.35),
#'     ylab = expression(P(z[t] == 1)),
#'     main = "Latent indicators: logit link"
#'   )
#'   lines(z_prob_logit, col = rgb(0.2, 0.5, 0.8, alpha = 0.2), lwd = 2)
#'   lines(alpha_true, col = "black", lwd = 2, lty = 2)
#'   legend(
#'     x = 0,
#'     y = 1.35,
#'     legend = "True z",
#'     col = 1,
#'     lty = NA,
#'     pch = 16,
#'     lwd = NA,
#'     bty = "n"
#'   )
#'   legend(
#'     "topright",
#'     legend = c(
#'       expression(hat(P)(z[t] == 1*" | "*data)),
#'       expression(alpha[t])
#'     ),
#'     col = c(rgb(0.2, 0.5, 0.8, alpha = 0.2), "black"),
#'     lty = c(1, 2),
#'     pch = c(NA, NA),
#'     lwd = c(2, 2),
#'     bty = "n"
#'   )
#'
#'   # Probit
#'   plot(
#'     z_true,
#'     col = "black",
#'     type = "p",
#'     pch = 16,
#'     cex = 0.8,
#'     xlab = "t",
#'     ylim = c(-0.1, 1.35),
#'     ylab = expression(P(z[t] == 1)),
#'     main = "Latent indicators: probit link"
#'   )
#'   lines(z_prob_probit, col = rgb(0.8, 0.2, 0.5, alpha = 0.2), lwd = 2)
#'   lines(alpha_true, col = "black", lwd = 2, lty = 2)
#'   legend(
#'     x = 0,
#'     y = 1.35,
#'     legend = "True z",
#'     col = 1,
#'     lty = NA,
#'     pch = 16,
#'     lwd = NA,
#'     bty = "n"
#'   )
#'   legend(
#'     "topright",
#'     legend = c(
#'       expression(hat(P)(z[t] == 1*" | "*data)),
#'       expression(alpha[t])
#'     ),
#'     col = c(rgb(0.8, 0.2, 0.5, alpha = 0.2), "black"),
#'     lty = c(1, 2),
#'     pch = c(NA, NA),
#'     lwd = c(2, 2),
#'     bty = "n"
#'   )
#'
#'   par(mfrow = c(1, 1))
#'
#'   # --- 3. Component Mean mu[1]: Trace, Density, ACF (Logit top, Probit bottom) ---
#'   par(mfrow = c(2, 3))
#'
#'   # Logit - Trace
#'   range_mu1_logit <- range(out_logit$mu_1)
#'   r1_mu1_logit <- range_mu1_logit[1] - 0.1 * diff(range_mu1_logit)
#'   r2_mu1_logit <- range_mu1_logit[2] + 0.1 * diff(range_mu1_logit)
#'   plot.ts(
#'     out_logit$mu_1,
#'     ylab = expression(mu[1]),
#'     main = expression(paste("Trace plot: ", mu[1], " (Logit)")),
#'     xlab = "Iteration",
#'     col = "gray",
#'     ylim = c(r1_mu1_logit, r2_mu1_logit)
#'   )
#'   abline(h = mu_1_true, col = "black", lty = 2, lwd = 2)
#'   abline(h = median(out_logit$mu_1), col = "blue", lty = 1, lwd = 2)
#'
#'   # Logit - Density
#'   plot(
#'     density(out_logit$mu_1),
#'     main = expression(paste("Density: ", mu[1], " (Logit)")),
#'     xlab = expression(mu[1]),
#'     ylab = "Density",
#'     lwd = 2,
#'     col = "blue"
#'   )
#'   abline(v = mu_1_true, col = "black", lty = 2, lwd = 2)
#'   abline(v = median(out_logit$mu_1), col = "blue", lty = 1, lwd = 2)
#'
#'   # Logit - ACF
#'   acf(
#'     out_logit$mu_1,
#'     main = expression(paste("ACF: ", mu[1], " (Logit)")),
#'     col = "blue",
#'     lwd = 2
#'   )
#'
#'   # Probit - Trace
#'   range_mu1_probit <- range(out_probit$mu_1)
#'   r1_mu1_probit <- range_mu1_probit[1] - 0.1 * diff(range_mu1_probit)
#'   r2_mu1_probit <- range_mu1_probit[2] + 0.1 * diff(range_mu1_probit)
#'   plot.ts(
#'     out_probit$mu_1,
#'     ylab = expression(mu[1]),
#'     main = expression(paste("Trace plot: ", mu[1], " (Probit)")),
#'     xlab = "Iteration",
#'     col = "gray",
#'     ylim = c(r1_mu1_probit, r2_mu1_probit)
#'   )
#'   abline(h = mu_1_true, col = "black", lty = 2, lwd = 2)
#'   abline(h = median(out_probit$mu_1), col = "red", lty = 1, lwd = 2)
#'
#'   # Probit - Density
#'   plot(
#'     density(out_probit$mu_1),
#'     main = expression(paste("Density: ", mu[1], " (Probit)")),
#'     xlab = expression(mu[1]),
#'     ylab = "Density",
#'     lwd = 2,
#'     col = "red"
#'   )
#'   abline(v = mu_1_true, col = "black", lty = 2, lwd = 2)
#'   abline(v = median(out_probit$mu_1), col = "red", lty = 1, lwd = 2)
#'
#'   # Probit - ACF
#'   acf(
#'     out_probit$mu_1,
#'     main = expression(paste("ACF: ", mu[1], " (Probit)")),
#'     col = "red",
#'     lwd = 2
#'   )
#'
#'   par(mfrow = c(1, 1))
#'
#'   # --- 4. Component Mean mu[2]: Trace, Density, ACF (Logit top, Probit bottom) ---
#'   par(mfrow = c(2, 3))
#'
#'   # Logit - Trace
#'   range_mu2_logit <- range(out_logit$mu_2)
#'   r1_mu2_logit <- range_mu2_logit[1] - 0.1 * diff(range_mu2_logit)
#'   r2_mu2_logit <- range_mu2_logit[2] + 0.1 * diff(range_mu2_logit)
#'   plot.ts(
#'     out_logit$mu_2,
#'     ylab = expression(mu[2]),
#'     main = expression(paste("Trace plot: ", mu[2], " (Logit)")),
#'     xlab = "Iteration",
#'     col = "gray",
#'     ylim = c(r1_mu2_logit, r2_mu2_logit)
#'   )
#'   abline(h = mu_2_true, col = "black", lty = 2, lwd = 2)
#'   abline(h = median(out_logit$mu_2), col = "blue", lty = 1, lwd = 2)
#'
#'   # Logit - Density
#'   plot(
#'     density(out_logit$mu_2),
#'     main = expression(paste("Density: ", mu[2], " (Logit)")),
#'     xlab = expression(mu[2]),
#'     ylab = "Density",
#'     lwd = 2,
#'     col = "blue"
#'   )
#'   abline(v = mu_2_true, col = "black", lty = 2, lwd = 2)
#'   abline(v = median(out_logit$mu_2), col = "blue", lty = 1, lwd = 2)
#'
#'   # Logit - ACF
#'   acf(
#'     out_logit$mu_2,
#'     main = expression(paste("ACF: ", mu[2], " (Logit)")),
#'     col = "blue",
#'     lwd = 2
#'   )
#'
#'   # Probit - Trace
#'   range_mu2_probit <- range(out_probit$mu_2)
#'   r1_mu2_probit <- range_mu2_probit[1] - 0.1 * diff(range_mu2_probit)
#'   r2_mu2_probit <- range_mu2_probit[2] + 0.1 * diff(range_mu2_probit)
#'   plot.ts(
#'     out_probit$mu_2,
#'     ylab = expression(mu[2]),
#'     main = expression(paste("Trace plot: ", mu[2], " (Probit)")),
#'     xlab = "Iteration",
#'     col = "gray",
#'     ylim = c(r1_mu2_probit, r2_mu2_probit)
#'   )
#'   abline(h = mu_2_true, col = "black", lty = 2, lwd = 2)
#'   abline(h = median(out_probit$mu_2), col = "red", lty = 1, lwd = 2)
#'
#'   # Probit - Density
#'   plot(
#'     density(out_probit$mu_2),
#'     main = expression(paste("Density: ", mu[2], " (Probit)")),
#'     xlab = expression(mu[2]),
#'     ylab = "Density",
#'     lwd = 2,
#'     col = "red"
#'   )
#'   abline(v = mu_2_true, col = "black", lty = 2, lwd = 2)
#'   abline(v = median(out_probit$mu_2), col = "red", lty = 1, lwd = 2)
#'
#'   # Probit - ACF
#'   acf(
#'     out_probit$mu_2,
#'     main = expression(paste("ACF: ", mu[2], " (Probit)")),
#'     col = "red",
#'     lwd = 2
#'   )
#'
#'   par(mfrow = c(1, 1))
#'
#'   # --- 5. Component Precision phi[1]: Trace, Density, ACF (Logit top, Probit bottom) ---
#'   par(mfrow = c(2, 3))
#'
#'   # Logit - Trace
#'   range_phi1_logit <- range(out_logit$prec_1)
#'   r1_phi1_logit <- range_phi1_logit[1] - 0.1 * diff(range_phi1_logit)
#'   r2_phi1_logit <- range_phi1_logit[2] + 0.1 * diff(range_phi1_logit)
#'   plot.ts(
#'     out_logit$prec_1,
#'     ylab = expression(phi[1]),
#'     main = expression(paste("Trace plot: ", phi[1], " (Logit)")),
#'     xlab = "Iteration",
#'     col = "gray",
#'     ylim = c(r1_phi1_logit, r2_phi1_logit)
#'   )
#'   abline(h = phi_1_true, col = "black", lty = 2, lwd = 2)
#'   abline(h = median(out_logit$prec_1), col = "blue", lty = 1, lwd = 2)
#'
#'   # Logit - Density
#'   plot(
#'     density(out_logit$prec_1),
#'     main = expression(paste("Density: ", phi[1], " (Logit)")),
#'     xlab = expression(phi[1]),
#'     ylab = "Density",
#'     lwd = 2,
#'     col = "blue"
#'   )
#'   abline(v = phi_1_true, col = "black", lty = 2, lwd = 2)
#'   abline(v = median(out_logit$prec_1), col = "blue", lty = 1, lwd = 2)
#'
#'   # Logit - ACF
#'   acf(
#'     out_logit$prec_1,
#'     main = expression(paste("ACF: ", phi[1], " (Logit)")),
#'     col = "blue",
#'     lwd = 2
#'   )
#'
#'   # Probit - Trace
#'   range_phi1_probit <- range(out_probit$prec_1)
#'   r1_phi1_probit <- range_phi1_probit[1] - 0.1 * diff(range_phi1_probit)
#'   r2_phi1_probit <- range_phi1_probit[2] + 0.1 * diff(range_phi1_probit)
#'   plot.ts(
#'     out_probit$prec_1,
#'     ylab = expression(phi[1]),
#'     main = expression(paste("Trace plot: ", phi[1], " (Probit)")),
#'     xlab = "Iteration",
#'     col = "gray",
#'     ylim = c(r1_phi1_probit, r2_phi1_probit)
#'   )
#'   abline(h = phi_1_true, col = "black", lty = 2, lwd = 2)
#'   abline(h = median(out_probit$prec_1), col = "red", lty = 1, lwd = 2)
#'
#'   # Probit - Density
#'   plot(
#'     density(out_probit$prec_1),
#'     main = expression(paste("Density: ", phi[1], " (Probit)")),
#'     xlab = expression(phi[1]),
#'     ylab = "Density",
#'     lwd = 2,
#'     col = "red"
#'   )
#'   abline(v = phi_1_true, col = "black", lty = 2, lwd = 2)
#'   abline(v = median(out_probit$prec_1), col = "red", lty = 1, lwd = 2)
#'
#'   # Probit - ACF
#'   acf(
#'     out_probit$prec_1,
#'     main = expression(paste("ACF: ", phi[1], " (Probit)")),
#'     col = "red",
#'     lwd = 2
#'   )
#'
#'   par(mfrow = c(1, 1))
#'
#'   # --- 6. Component Precision phi[2]: Trace, Density, ACF (Logit top, Probit bottom) ---
#'   par(mfrow = c(2, 3))
#'
#'   # Logit - Trace
#'   range_phi2_logit <- range(out_logit$prec_2)
#'   r1_phi2_logit <- range_phi2_logit[1] - 0.1 * diff(range_phi2_logit)
#'   r2_phi2_logit <- range_phi2_logit[2] + 0.1 * diff(range_phi2_logit)
#'   plot.ts(
#'     out_logit$prec_2,
#'     ylab = expression(phi[2]),
#'     main = expression(paste("Trace plot: ", phi[2], " (Logit)")),
#'     xlab = "Iteration",
#'     col = "gray",
#'     ylim = c(r1_phi2_logit, r2_phi2_logit)
#'   )
#'   abline(h = phi_2_true, col = "black", lty = 2, lwd = 2)
#'   abline(h = median(out_logit$prec_2), col = "blue", lty = 1, lwd = 2)
#'
#'   # Logit - Density
#'   plot(
#'     density(out_logit$prec_2),
#'     main = expression(paste("Density: ", phi[2], " (Logit)")),
#'     xlab = expression(phi[2]),
#'     ylab = "Density",
#'     lwd = 2,
#'     col = "blue"
#'   )
#'   abline(v = phi_2_true, col = "black", lty = 2, lwd = 2)
#'   abline(v = median(out_logit$prec_2), col = "blue", lty = 1, lwd = 2)
#'
#'   # Logit - ACF
#'   acf(
#'     out_logit$prec_2,
#'     main = expression(paste("ACF: ", phi[2], " (Logit)")),
#'     col = "blue",
#'     lwd = 2
#'   )
#'
#'   # Probit - Trace
#'   range_phi2_probit <- range(out_probit$prec_2)
#'   r1_phi2_probit <- range_phi2_probit[1] - 0.1 * diff(range_phi2_probit)
#'   r2_phi2_probit <- range_phi2_probit[2] + 0.1 * diff(range_phi2_probit)
#'   plot.ts(
#'     out_probit$prec_2,
#'     ylab = expression(phi[2]),
#'     main = expression(paste("Trace plot: ", phi[2], " (Probit)")),
#'     xlab = "Iteration",
#'     col = "gray",
#'     ylim = c(r1_phi2_probit, r2_phi2_probit)
#'   )
#'   abline(h = phi_2_true, col = "black", lty = 2, lwd = 2)
#'   abline(h = median(out_probit$prec_2), col = "red", lty = 1, lwd = 2)
#'
#'   # Probit - Density
#'   plot(
#'     density(out_probit$prec_2),
#'     main = expression(paste("Density: ", phi[2], " (Probit)")),
#'     xlab = expression(phi[2]),
#'     ylab = "Density",
#'     lwd = 2,
#'     col = "red"
#'   )
#'   abline(v = phi_2_true, col = "black", lty = 2, lwd = 2)
#'   abline(v = median(out_probit$prec_2), col = "red", lty = 1, lwd = 2)
#'
#'   # Probit - ACF
#'   acf(
#'     out_probit$prec_2,
#'     main = expression(paste("ACF: ", phi[2], " (Probit)")),
#'     col = "red",
#'     lwd = 2
#'   )
#'
#'   par(mfrow = c(1, 1))
#'
#'   # --- 7. Initial Level State (theta[0,1]): Trace, Density, ACF (Logit top, Probit bottom) ---
#'   par(mfrow = c(2, 3))
#'
#'   # Logit - Trace
#'   range_theta_01_logit <- range(out_logit$theta_01)
#'   r1_theta01_logit <- range_theta_01_logit[1] - 0.1 * diff(range_theta_01_logit)
#'   r2_theta01_logit <- range_theta_01_logit[2] + 0.3 * diff(range_theta_01_logit)
#'
#'   plot.ts(
#'     out_logit$theta_01,
#'     ylab = expression(theta["0,1"]),
#'     main = expression(paste("Trace plot: ", theta["0,1"], " (Logit)")),
#'     xlab = "Iteration",
#'     col = "gray",
#'     ylim = c(r1_theta01_logit, r2_theta01_logit)
#'   )
#'   abline(h = median(out_logit$theta_01), col = "blue", lty = 1, lwd = 2)
#'
#'   # Logit - Density
#'   plot(
#'     density(out_logit$theta_01),
#'     main = expression(paste("Density: ", theta["0,1"], " (Logit)")),
#'     xlab = expression(theta["0,1"]),
#'     ylab = "Density",
#'     lwd = 2,
#'     col = "blue"
#'   )
#'   abline(v = median(out_logit$theta_01), col = "blue", lty = 1, lwd = 2)
#'
#'   # Logit - ACF
#'   acf(
#'     out_logit$theta_01,
#'     main = expression(paste("ACF: ", theta["0,1"], " (Logit)")),
#'     col = "blue",
#'     lwd = 2
#'   )
#'
#'   # Probit - Trace
#'   range_theta_01_probit <- range(out_probit$theta_01)
#'   r1_theta01_probit <- range_theta_01_probit[1] - 0.1 * diff(range_theta_01_probit)
#'   r2_theta01_probit <- range_theta_01_probit[2] + 0.3 * diff(range_theta_01_probit)
#'
#'   plot.ts(
#'     out_probit$theta_01,
#'     ylab = expression(theta["0,1"]),
#'     main = expression(paste("Trace plot: ", theta["0,1"], " (Probit)")),
#'     xlab = "Iteration",
#'     col = "gray",
#'     ylim = c(r1_theta01_probit, r2_theta01_probit)
#'   )
#'   abline(h = median(out_probit$theta_01), col = "red", lty = 1, lwd = 2)
#'
#'   # Probit - Density
#'   plot(
#'     density(out_probit$theta_01),
#'     main = expression(paste("Density: ", theta["0,1"], " (Probit)")),
#'     xlab = expression(theta["0,1"]),
#'     ylab = "Density",
#'     lwd = 2,
#'     col = "red"
#'   )
#'   abline(v = median(out_probit$theta_01), col = "red", lty = 1, lwd = 2)
#'
#'   # Probit - ACF
#'   acf(
#'     out_probit$theta_01,
#'     main = expression(paste("ACF: ", theta["0,1"], " (Probit)")),
#'     col = "red",
#'     lwd = 2
#'   )
#'
#'   par(mfrow = c(1, 1))
#'
#'   # --- 8. Level Innovation Precision (1/W[1]): Trace, Density, ACF (Logit top, Probit bottom) ---
#'   par(mfrow = c(2, 3))
#'
#'   # Logit - Trace
#'   range_prec_1_logit <- range(out_logit$prec_theta1)
#'   r1_prec1_logit <- range_prec_1_logit[1] - 0.1 * diff(range_prec_1_logit)
#'   r2_prec1_logit <- range_prec_1_logit[2] + 0.25 * diff(range_prec_1_logit)
#'
#'   plot.ts(
#'     out_logit$prec_theta1,
#'     ylab = expression(W[1]^-1),
#'     main = expression(paste("Trace plot: ", W[1]^-1, " (Logit)")),
#'     xlab = "Iteration",
#'     col = "gray",
#'     ylim = c(r1_prec1_logit, r2_prec1_logit)
#'   )
#'   abline(h = median(out_logit$prec_theta1), col = "blue", lty = 1, lwd = 2)
#'
#'   # Logit - Density
#'   plot(
#'     density(out_logit$prec_theta1),
#'     main = expression(paste("Density: ", W[1]^-1, " (Logit)")),
#'     xlab = expression(W[1]^-1),
#'     ylab = "Density",
#'     lwd = 2,
#'     col = "blue"
#'   )
#'   abline(v = median(out_logit$prec_theta1), col = "blue", lty = 1, lwd = 2)
#'
#'   # Logit - ACF
#'   acf(
#'     out_logit$prec_theta1,
#'     main = expression(paste("ACF: ", W[1]^-1, " (Logit)")),
#'     col = "blue",
#'     lwd = 2
#'   )
#'
#'   # Probit - Trace
#'   range_prec_1_probit <- range(out_probit$prec_theta1)
#'   r1_prec1_probit <- range_prec_1_probit[1] - 0.1 * diff(range_prec_1_probit)
#'   r2_prec1_probit <- range_prec_1_probit[2] + 0.25 * diff(range_prec_1_probit)
#'
#'   plot.ts(
#'     out_probit$prec_theta1,
#'     ylab = expression(W[1]^-1),
#'     main = expression(paste("Trace plot: ", W[1]^-1, " (Probit)")),
#'     xlab = "Iteration",
#'     col = "gray",
#'     ylim = c(r1_prec1_probit, r2_prec1_probit)
#'   )
#'   abline(h = median(out_probit$prec_theta1), col = "red", lty = 1, lwd = 2)
#'
#'   # Probit - Density
#'   plot(
#'     density(out_probit$prec_theta1),
#'     main = expression(paste("Density: ", W[1]^-1, " (Probit)")),
#'     xlab = expression(W[1]^-1),
#'     ylab = "Density",
#'     lwd = 2,
#'     col = "red"
#'   )
#'   abline(v = median(out_probit$prec_theta1), col = "red", lty = 1, lwd = 2)
#'
#'   # Probit - ACF
#'   acf(
#'     out_probit$prec_theta1,
#'     main = expression(paste("ACF: ", W[1]^-1, " (Probit)")),
#'     col = "red",
#'     lwd = 2
#'   )
#'
#'   par(mfrow = c(1, 1))
#'
#'   # --- 9. Summary Statistics ---
#'   cat("\n=== Summary Statistics ===\n\n")
#'
#'   summary_df <- data.frame(
#'     Parameter = c(
#'       "mu_1   (Logit)", "mu_1  (Probit)",
#'       "mu_2   (Logit)", "mu_2  (Probit)",
#'       "phi_1  (Logit)", "phi_1 (Probit)",
#'       "phi_2  (Logit)", "phi_2 (Probit)"
#'     ),
#'     True_Value = c(
#'       mu_1_true, mu_1_true,
#'       mu_2_true, mu_2_true,
#'       phi_1_true, phi_1_true,
#'       phi_2_true, phi_2_true
#'     ),
#'     Estimate = c(
#'       median(out_logit$mu_1), median(out_probit$mu_1),
#'       median(out_logit$mu_2), median(out_probit$mu_2),
#'       median(out_logit$prec_1), median(out_probit$prec_1),
#'       median(out_logit$prec_2), median(out_probit$prec_2)
#'     ),
#'     CI_Lower = c(
#'       quantile(out_logit$mu_1, 0.025), quantile(out_probit$mu_1, 0.025),
#'       quantile(out_logit$mu_2, 0.025), quantile(out_probit$mu_2, 0.025),
#'       quantile(out_logit$prec_1, 0.025), quantile(out_probit$prec_1, 0.025),
#'       quantile(out_logit$prec_2, 0.025), quantile(out_probit$prec_2, 0.025)
#'     ),
#'     CI_Upper = c(
#'       quantile(out_logit$mu_1, 0.975), quantile(out_probit$mu_1, 0.975),
#'       quantile(out_logit$mu_2, 0.975), quantile(out_probit$mu_2, 0.975),
#'       quantile(out_logit$prec_1, 0.975), quantile(out_probit$prec_1, 0.975),
#'       quantile(out_logit$prec_2, 0.975), quantile(out_probit$prec_2, 0.975)
#'     ),
#'     stringsAsFactors = FALSE
#'   )
#'
#'   summary_df$CI_95 <- sprintf("(%.3f, %.3f)", summary_df$CI_Lower, summary_df$CI_Upper)
#'
#'   display_table <- data.frame(
#'     Parameter = summary_df$Parameter,
#'     True_Value = sprintf("%.3f", summary_df$True_Value),
#'     Estimate = sprintf("%.3f", summary_df$Estimate),
#'     CI_95 = summary_df$CI_95,
#'     stringsAsFactors = FALSE
#'   )
#'
#'   print(display_table, row.names = FALSE, right = TRUE)
#'
#'   cat("\nNote: CI_95 represents the 95% credible interval (2.5% and 97.5% quantiles)\n")
#'
#'   # --- 10. Model Comparison: Logit vs Probit ---
#'   rmse_logit <- sqrt(mean((alpha_logit_estimate - alpha_true)^2))
#'   rmse_probit <- sqrt(mean((alpha_probit_estimate - alpha_true)^2))
#'
#'   cat("\n=== Model Comparison ===\n\n")
#'   cat(sprintf("RMSE for alpha (Logit):  %.4f\n", rmse_logit))
#'   cat(sprintf("RMSE for alpha (Probit): %.4f\n", rmse_probit))
#'
#'   cor_estimates <- cor(alpha_logit_estimate, alpha_probit_estimate)
#'   cat(sprintf("\nCorrelation between Logit and Probit estimates: %.4f\n", cor_estimates))
#' }
#'
#' @references
#' Roberts, G. O., & Rosenthal, J. S. (2001). Optimal scaling for various
#' Metropolis-Hastings algorithms. \emph{Statistical Science}, 16(4), 351-367.
#'
#' Roberts, G. O., & Rosenthal, J. S. (2007). Coupling and ergodicity of adaptive MCMC.
#' \emph{Journal of Applied Probability}, 44(2), 458-475.
#'
#' Roberts, G. O., & Rosenthal, J. S. (2009). Examples of Adaptive MCMC.
#' \emph{Journal of Computational and Graphical Statistics}, 18(2), 349-367.
#'
#' Albert, J. H., & Chib, S. (1993). Bayesian Analysis of Binary and Polychotomous
#' Response Data. \emph{Journal of the American Statistical Association}, 88(422), 669-679.
#' https://doi.org/10.1080/01621459.1993.10476321
#'
#' Montoril, M. H., Correia, L. T., & Migon, H. S. (2021). Bayesian estimation of
#' dynamic weights in Gaussian mixture models. arXiv:2104.03395.
#'
#' @seealso \code{\link{mcmc_probit_bernoulli_locallevel}},
#'  \code{\link{mcmc_binomial_locallevel}},
#'  \code{\link{mcmc_normal_mixture_localtrend}},
#'  \code{\link{mcmc_normal_mixture_localacceleration}}
#'
#' @export
mcmc_normal_mixture_locallevel <- function(y,
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
                                           prior_prec1_shape = 0.01,
                                           prior_prec1_rate = 0.01,
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
  if (!is.numeric(prior_mu01_prec) || length(prior_mu01_prec) != 1 ||
      prior_mu01_prec <= 0) {
    stop("`prior_mu01_prec` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec01_shape) || length(prior_prec01_shape) != 1 ||
      prior_prec01_shape <= 0) {
    stop("`prior_prec01_shape` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec01_rate) || length(prior_prec01_rate) != 1 ||
      prior_prec01_rate <= 0) {
    stop("`prior_prec01_rate` must be a single positive numeric value")
  }

  if (!is.numeric(prior_mu02_mean) || length(prior_mu02_mean) != 1) {
    stop("`prior_mu02_mean` must be a single numeric value")
  }
  if (!is.numeric(prior_mu02_prec) || length(prior_mu02_prec) != 1 ||
      prior_mu02_prec <= 0) {
    stop("`prior_mu02_prec` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec02_shape) || length(prior_prec02_shape) != 1 ||
      prior_prec02_shape <= 0) {
    stop("`prior_prec02_shape` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec02_rate) || length(prior_prec02_rate) != 1 ||
      prior_prec02_rate <= 0) {
    stop("`prior_prec02_rate` must be a single positive numeric value")
  }

  # Validate dynamic state prior parameters
  if (!is.numeric(prior_theta01_mean) || length(prior_theta01_mean) != 1) {
    stop("`prior_theta01_mean` must be a single numeric value")
  }
  if (!is.numeric(prior_theta01_prec) || length(prior_theta01_prec) != 1 ||
      prior_theta01_prec <= 0) {
    stop("`prior_theta01_prec` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec1_shape) || length(prior_prec1_shape) != 1 ||
      prior_prec1_shape <= 0) {
    stop("`prior_prec1_shape` must be a single positive numeric value")
  }
  if (!is.numeric(prior_prec1_rate) || length(prior_prec1_rate) != 1 ||
      prior_prec1_rate <= 0) {
    stop("`prior_prec1_rate` must be a single positive numeric value")
  }

  # Validate MCMC adaptation parameters
  if (!is.numeric(lag_update) || length(lag_update) != 1 || lag_update < 1 ||
      lag_update != floor(lag_update)) {
    stop("`lag_update` must be a single positive integer")
  }
  if (!is.numeric(max_step_size) || length(max_step_size) != 1 ||
      max_step_size <= 0) {
    stop("`max_step_size` must be a single positive numeric value")
  }
  if (!is.numeric(base_adaptation_rate) || length(base_adaptation_rate) != 1 ||
      base_adaptation_rate <= 0) {
    stop("`base_adaptation_rate` must be a single positive numeric value")
  }
  if (!is.numeric(decay_exponent) || length(decay_exponent) != 1 ||
      decay_exponent <= 0) {
    stop("`decay_exponent` must be a single positive numeric value")
  }
  if (!is.numeric(target_acceptance) || length(target_acceptance) != 1 ||
      target_acceptance <= 0 || target_acceptance >= 1) {
    stop("`target_acceptance` must be a single numeric value in (0, 1)")
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

  # Validate logical parameters
  if (!is.logical(return_log_sigma) || length(return_log_sigma) != 1) {
    stop("`return_log_sigma` must be a single logical value")
  }
  if (!is.logical(return_accept_prop) || length(return_accept_prop) != 1) {
    stop("`return_accept_prop` must be a single logical value")
  }

  # Validate verbose parameter
  if (!is.logical(verbose) || length(verbose) != 1) {
    stop("`verbose` must be a single logical value")
  }

  # Validate bar_width parameter
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
    "_bdm_C_MCMC_normal_mixture_locallevel",
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
    as.numeric(prior_prec1_shape),
    as.numeric(prior_prec1_rate),
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

  result <- new_normal_mixture_locallevel(
    result = result,
    link = link,
    n_obs = length(y),
    n_chain = n_chain,
    burnin = burnin,
    thinning = thinning,
    y = y
  )

  result <- validate_normal_mixture_locallevel(result)

  return(result)
}
