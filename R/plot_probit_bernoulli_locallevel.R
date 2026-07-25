#' Plot method for probit_bernoulli_locallevel objects
#'
#' @description Produces diagnostic plots for MCMC output from Bernoulli
#'   local level models with probit link.
#'
#' @param x An object of class `probit_bernoulli_locallevel`.
#' @param type Character string specifying the type of plot. One of:
#'   \describe{
#'     \item{\code{"all"}}{Complete dashboard with all diagnostic plots (default)}
#'     \item{\code{"mcmc"}}{MCMC convergence diagnostics (trace plots, ACF, running means)}
#'     \item{\code{"states"}}{Dynamic states (theta_1 trajectory)}
#'     \item{\code{"alpha"}}{Bernoulli probabilities over time (alpha_t)}
#'   }
#' @param which Integer vector specifying which diagnostic plots to display.
#'   For `type = "mcmc"`:
#'   \describe{
#'     \item{1}{theta_01 (initial level)}
#'     \item{2}{W_1^-1 (level innovation precision)}
#'   }
#'   For `type = "states"`: indices of subplots.
#'   For `type = "alpha"`: not used.
#'   If `NULL` (default), all available plots are shown.
#' @param ask Logical; if `TRUE`, the user is asked before each plot when
#'   `type = "all"`. Default is `interactive()` when `type = "all"`,
#'   `FALSE` otherwise.
#' @param ci Logical; whether to display credible intervals in plots that
#'   support them. Default is `TRUE`.
#' @param ci_level Numeric; Bayesian confidence level for credible intervals
#'   (between 0 and 1). Default is `0.95`.
#' @param show_obs Logical; whether to display observed binary outcomes on the
#'   Bernoulli probabilities plot (`type = "alpha"`). When `TRUE`
#'   (default), observed binary values (0 or 1) are overlaid as red points
#'   on the alpha_t trajectory. Set to `FALSE` to show only the estimated
#'   trajectory without observations. This parameter only affects `type = "alpha"`
#'   and `type = "all"`.
#' @param true_values Named list containing true parameter values and/or state trajectories
#'   for comparison with MCMC estimates. If `NULL` (default), no true values are displayed.
#'
#'   \strong{Important:} All parameter names in `true_values` must match exactly
#'   the component names returned by \code{\link{mcmc_probit_bernoulli_locallevel}}.
#'
#'   Accepted elements:
#'   \describe{
#'     \item{\strong{Scalar parameters} (for `type = "mcmc"`):}{
#'       \itemize{
#'         \item `theta_01`: Initial level state
#'         \item `prec_theta1`: Level innovation precision (W_1^-1)
#'       }
#'     }
#'     \item{\strong{State trajectories} (for `type = "states"`):}{
#'       \itemize{
#'         \item `theta_1`: Numeric vector of length `n_obs` with true level state values
#'       }
#'     }
#'     \item{\strong{Bernoulli probabilities} (for `type = "alpha"`):}{
#'       \itemize{
#'         \item `alpha`: Numeric vector of length `n_obs` with true alpha_t probabilities
#'       }
#'     }
#'   }
#'
#'   \strong{Note on state trajectories:} If you only have the true `alpha`,
#'   you can obtain `theta_1` using `qnorm(alpha)` (probit link).
#'
#'   You can provide any subset of these elements. For example, to compare only
#'   the initial state and alpha:
#'   \preformatted{
#'   true_values = list(
#'     theta_01 = 0,
#'     alpha = alpha_true_vector
#'   )
#'   }
#' @param ... Additional arguments passed to plotting functions.
#'
#' @return Invisibly returns the input object `x`.
#'
#' @details
#' This function provides comprehensive visual diagnostics for Bayesian MCMC output:
#'
#' \strong{MCMC Diagnostics} (`type = "mcmc"`):
#'
#' Each parameter gets a dedicated page with 4 panels:
#' \itemize{
#'   \item \strong{Trace Plot:} Shows MCMC chain trajectory to assess mixing
#'   \item \strong{Autocorrelation:} ACF plot to detect serial correlation
#'   \item \strong{Posterior Density:} Marginal posterior distribution
#'   \item \strong{Running Mean:} Cumulative mean to assess convergence
#' }
#'
#' Available parameters: theta_01, W_1^-1
#'
#' \strong{Dynamic States} (`type = "states"`):
#' \itemize{
#'   \item Time-varying state trajectory with credible bands (on probit scale)
#'   \item Innovation sequence
#' }
#'
#' \strong{Bernoulli Probabilities} (`type = "alpha"`):
#' \itemize{
#'   \item alpha_t trajectory with credible bands
#'   \item Optional observed binary outcomes overlay (controlled by `show_obs`)
#' }
#'
#' \strong{Complete Dashboard} (`type = "all"`):
#'
#' Generates 4 pages in total:
#' \itemize{
#'   \item Pages 1-2: Individual parameter diagnostics (4 panels each)
#'   \item Page 3: Dynamic state trajectory and diagnostics
#'   \item Page 4: Bernoulli probabilities alpha_t
#' }
#'
#' @section Controlling Observed Data Display:
#'
#' The `show_obs` parameter provides control over the display of observed
#' binary outcomes in the Bernoulli probabilities plot:
#'
#' \itemize{
#'   \item When `show_obs = TRUE` (default): Observed binary values (0 or 1)
#'     are shown as red points overlaid on the estimated alpha_t trajectory. This
#'     is useful for model validation and assessing goodness-of-fit.
#'   \item When `show_obs = FALSE`: Only the estimated trajectory is shown,
#'     which can be clearer for presentations or when focusing on the temporal
#'     pattern of the Bernoulli probabilities.
#' }
#'
#' @examples
#' \donttest{
#' # =============================================================================
#' # Example 1: Practical Data Analysis (No True Parameters Known)
#' # =============================================================================
#' # This example demonstrates a typical workflow when analyzing real data where
#' # true parameter values are unknown. We generate synthetic data with a simple
#' # oscillating pattern to mimic real-world temporal variation in binary outcomes.
#'
#' set.seed(123)
#' n <- 200          # Number of time points
#'
#' # Generate simple oscillating Bernoulli probabilities (mimicking temporal patterns)
#' alpha_true <- (sin(2 * pi * seq_len(n) / n) + 2) / 4
#'
#' # Generate Bernoulli observations (e.g., binary outcomes over time)
#' y <- rbinom(n, size = 1, prob = alpha_true)
#'
#' # Fit the local-level model with weakly informative priors
#' # (appropriate when we have limited prior knowledge)
#' out <- mcmc_probit_bernoulli_locallevel(
#'   y,
#'   burnin                  = 1000,      # Discard first 1000 iterations
#'   thinning                = 20,        # Keep every 20th iteration
#'   n_draws                 = 500,       # Retain 500 posterior samples
#'   # Weakly informative prior for initial state (centered at 0 on probit scale)
#'   prior_theta01_mean      = 0,
#'   prior_theta01_prec      = 1,
#'   # Weakly informative prior for innovation precision
#'   # (shape = rate implies mean = 1, but with high variance)
#'   prior_prec1_shape       = 100,
#'   prior_prec1_rate        = 1,
#'   verbose                 = TRUE,      # Show progress bar
#'   seed                    = 456        # For reproducibility
#' )
#'
#' # --- Visualization Options ---
#'
#' # 1. Complete diagnostic dashboard (4 pages)
#' #    Includes: MCMC diagnostics, state trajectory, alpha plot
#' plot(out, type = "all")
#'
#' # 2. MCMC convergence diagnostics for all parameters
#' #    Trace plots, ACF, posterior densities, running means
#' plot(out, type = "mcmc")
#'
#' # 3. Focus on initial state parameter only
#' plot(out, type = "mcmc", which = 1)  # theta_01
#'
#' # 4. Focus on innovation precision parameter
#' plot(out, type = "mcmc", which = 2)  # W_1^-1
#'
#' # 5. Dynamic state trajectory (theta_1 on probit scale)
#' #    Shows level component over time
#' plot(out, type = "states")
#'
#' # 6. Bernoulli probabilities with observed binary outcomes overlay
#' #    Red points show observed 0/1 values for model validation
#' plot(out, type = "alpha")
#'
#' # 7. Bernoulli probabilities without observed data (cleaner for presentations)
#' plot(out, type = "alpha", show_obs = FALSE)
#'
#' # 8. Adjust credible interval level (default is 95%)
#' plot(out, type = "alpha", ci_level = 0.90)  # 90% credible intervals
#'
#' # 9. Save all diagnostics to a multi-page PDF
#' pdf(file.path(tempdir(), "model_diagnostics.pdf"), width = 10, height = 8)
#' plot(out, type = "all", ask = FALSE)  # ask = FALSE prevents pausing
#' dev.off()
#'
#'
#' # =============================================================================
#' # Example 2: Simulation Study (True Parameters Known for Validation)
#' # =============================================================================
#' # This example demonstrates how to validate model performance using simulated
#' # data where true parameter values are known. This is essential for assessing
#' # whether the model can recover known parameters and for method development.
#'
#' # --- Step 1: Set up simulation parameters ---
#' set.seed(1)
#' n <- 100           # Number of time points
#'
#' # True parameter values (these would be unknown in real applications)
#' theta01_true     <- 0        # True initial level (on probit scale)
#' prec_theta1_true <- 100      # True level innovation precision (high = smooth)
#'
#' # --- Step 2: Simulate latent states following the state-space model ---
#' # Generate innovation sequence (random shocks to state)
#' u1 <- rnorm(n, mean = 0, sd = sqrt(1 / prec_theta1_true))  # Level innovations
#'
#' # Initialize state vector
#' theta1_true <- numeric(n)  # Level state (on probit scale)
#'
#' # First time point (t=1): state = initial value + innovation
#' theta1_true[1] <- theta01_true + u1[1]
#'
#' # Subsequent time points (t=2,...,n): follow state evolution equation
#' # theta_{t,1} = theta_{t-1,1} + u_{t,1}  (random walk)
#' for (t in 2:n) {
#'   theta1_true[t] <- theta1_true[t-1] + u1[t]
#' }
#'
#' # --- Step 3: Generate observations ---
#' # Transform level state to probability scale using inverse probit (standard normal CDF)
#' alpha_true <- pnorm(theta1_true)  # Bernoulli probabilities in [0,1]
#'
#' # Generate Bernoulli observations
#' y <- rbinom(n = n, size = 1, prob = alpha_true)
#'
#' # Optional: Visualize true trajectories before fitting
#' par(mfrow = c(1, 2))
#' plot(theta1_true, type = "l", main = "True Level State (probit scale)",
#'      xlab = "Time", ylab = expression(theta["t,1"]))
#' plot(alpha_true, type = "l", main = "True Bernoulli Probabilities",
#'      xlab = "Time", ylab = expression(alpha[t]), ylim = c(0, 1))
#' points(y, col = "red", pch = 16, cex = 0.5)
#' par(mfrow = c(1, 1))
#'
#' # --- Step 4: Fit the model with informative priors ---
#' # Note: In practice, we wouldn't know true values, but here we use
#' # priors centered near truth to demonstrate parameter recovery
#' out <- mcmc_probit_bernoulli_locallevel(
#'   y,
#'   burnin                  = 1000,
#'   thinning                = 20,
#'   n_draws                 = 500,
#'   # Prior centered at true initial value
#'   prior_theta01_mean      = 0,
#'   prior_theta01_prec      = 1,
#'   # Informative prior for innovation precision
#'   # (centered near true value with moderate uncertainty)
#'   prior_prec1_shape       = 100,
#'   prior_prec1_rate        = 1,
#'   verbose                 = TRUE,
#'   seed                    = 456
#' )
#'
#' # --- Step 5: Model validation using true parameter values ---
#' # Create named list with ALL true values (matching output component names)
#' # IMPORTANT: All names must match exactly the components returned by
#' # mcmc_probit_bernoulli_locallevel() - see ?mcmc_probit_bernoulli_locallevel
#' true_vals <- list(
#'   # Scalar parameters (for MCMC diagnostics)
#'   theta_01     = theta01_true,
#'   prec_theta1  = prec_theta1_true,
#'   # State trajectory (for state plots)
#'   theta_1      = theta1_true,
#'   # Bernoulli probabilities (for alpha plot)
#'   alpha        = alpha_true
#' )
#'
#' # --- Validation Plots ---
#'
#' # 1. Complete dashboard with true values overlaid
#' #    True values appear as dashed black lines in all relevant plots
#' plot(out, type = "all", true_values = true_vals)
#'
#' # 2. MCMC diagnostics with true parameter values (scalar parameters)
#' #    Check if posterior distributions contain true values
#' plot(out, type = "mcmc", true_values = true_vals)
#'
#' # 3. Focus on initial state with true value
#' plot(out, type = "mcmc", which = 1, true_values = true_vals)
#'
#' # 4. Focus on innovation precision with true value
#' plot(out, type = "mcmc", which = 2, true_values = true_vals)
#'
#' # 5. Dynamic state with true trajectory overlaid
#' #    Assess how well the model tracks the true time-varying state
#' plot(out, type = "states", true_values = true_vals)
#'
#' # 6. Bernoulli probabilities with true alpha trajectory
#' #    Compare estimated alpha_t with true values
#' plot(out, type = "alpha", true_values = true_vals)
#'
#' # 7. Partial validation: only compare specific components
#' #    Example 1: Only initial state and precision
#' plot(out, type = "mcmc", true_values = list(
#'   theta_01    = theta01_true,
#'   prec_theta1 = prec_theta1_true
#' ))
#'
#' #    Example 2: Only level state trajectory
#' plot(out, type = "states",
#'      true_values = list(theta_1 = theta1_true))
#'
#' #    Example 3: Only Bernoulli probabilities
#' plot(out, type = "alpha",
#'      true_values = list(alpha = alpha_true))
#'
#' }
#'
#' @seealso
#'   \code{\link{mcmc_probit_bernoulli_locallevel}} (model generator),
#'   \code{\link{print.probit_bernoulli_locallevel}}, \code{\link{summary.probit_bernoulli_locallevel}}.
#'
#' @export
plot.probit_bernoulli_locallevel <- function(x,
                                             type = c("all", "mcmc", "states", "alpha"),
                                             which = NULL,
                                             ask = NULL,
                                             ci = TRUE,
                                             ci_level = 0.95,
                                             show_obs = TRUE,
                                             true_values = NULL,
                                             ...) {

  type <- match.arg(type)

  if (is.null(ask)) {
    ask <- interactive() && type == "all"
  }

  switch(type,
         all = {
           oldpar <- par(no.readonly = TRUE)
           on.exit(par(oldpar))
           if (ask) {
             oldask <- par(ask = TRUE)
             on.exit(par(oldask), add = TRUE)
           }
           plot_mcmc_diagnostics_generic(x,
                                         which = NULL,
                                         true_values = true_values,
                                         ...)
           plot_dynamic_states_generic_base(x,
                                            which = NULL,
                                            ci = ci,
                                            ci_level = ci_level,
                                            true_values = true_values,
                                            ...)
           plot_bernoulli_alpha_base(x,
                                     ci = ci,
                                     ci_level = ci_level,
                                     show_obs = show_obs,
                                     true_alpha = true_values$alpha,
                                     ...)
         },
         mcmc = plot_mcmc_diagnostics_generic(x,
                                              which = which,
                                              true_values = true_values,
                                              ...),
         states = plot_dynamic_states_generic_base(x,
                                                   which = which,
                                                   ci = ci,
                                                   ci_level = ci_level,
                                                   true_values = true_values,
                                                   ...),
         alpha = plot_bernoulli_alpha_base(x,
                                           ci = ci,
                                           ci_level = ci_level,
                                           show_obs = show_obs,
                                           true_alpha = true_values$alpha,
                                           ...)
  )

  invisible(x)
}
