#' Plot method for probit_bernoulli_localtrend objects
#'
#' @description Produces diagnostic plots for MCMC output from Bernoulli
#'   local trend models with probit link.
#'
#' @param x An object of class \code{probit_bernoulli_localtrend}.
#' @param type Character string specifying the type of plot. One of:
#'   \describe{
#'     \item{\code{"all"}}{Complete dashboard with all diagnostic plots (default)}
#'     \item{\code{"mcmc"}}{MCMC convergence diagnostics (trace plots, ACF, running means)}
#'     \item{\code{"states"}}{Dynamic states (theta_1, theta_2 trajectories)}
#'     \item{\code{"alpha"}}{Bernoulli probabilities over time (alpha_t)}
#'   }
#' @param which Integer vector specifying which diagnostic plots to display.
#'   For \code{type = "mcmc"}:
#'   \describe{
#'     \item{1}{theta_01 (initial level)}
#'     \item{2}{theta_02 (initial trend)}
#'     \item{3}{W_1^{-1} (level innovation precision)}
#'     \item{4}{W_2^{-1} (trend innovation precision)}
#'   }
#'   For \code{type = "states"}: indices of subplots.
#'   For \code{type = "alpha"}: not used.
#'   If \code{NULL} (default), all available plots are shown.
#' @param ask Logical; if \code{TRUE}, the user is asked before each plot when
#'   \code{type = "all"}. Default is \code{interactive()} when \code{type = "all"},
#'   \code{FALSE} otherwise.
#' @param ci Logical; whether to display credible intervals in plots that
#'   support them. Default is \code{TRUE}.
#' @param ci_level Numeric; Bayesian confidence level for credible intervals
#'   (between 0 and 1). Default is \code{0.95}.
#' @param show_obs Logical; whether to display observed binary outcomes on the
#'   Bernoulli probabilities plot (\code{type = "alpha"}). When \code{TRUE}
#'   (default), observed binary values (0 or 1) are overlaid as red points
#'   on the alpha_t trajectory. Set to \code{FALSE} to show only the estimated
#'   trajectory without observations. This parameter only affects \code{type = "alpha"}
#'   and \code{type = "all"}.
#' @param true_values Named list containing true parameter values and/or state trajectories
#'   for comparison with MCMC estimates. If \code{NULL} (default), no true values are displayed.
#'
#'   \strong{Important:} All parameter names in \code{true_values} must match exactly
#'   the component names returned by \code{\link{mcmc_probit_bernoulli_localtrend}}.
#'
#'   Accepted elements:
#'   \describe{
#'     \item{\strong{Scalar parameters} (for \code{type = "mcmc"}):}{
#'       \itemize{
#'         \item \code{theta_01}: Initial level state
#'         \item \code{theta_02}: Initial trend state
#'         \item \code{prec_theta1}: Level innovation precision (W_1^{-1})
#'         \item \code{prec_theta2}: Trend innovation precision (W_2^{-1})
#'       }
#'     }
#'     \item{\strong{State trajectories} (for \code{type = "states"}):}{
#'       \itemize{
#'         \item \code{theta_1}: Numeric vector of length \code{n_obs} with true level state values
#'         \item \code{theta_2}: Numeric vector of length \code{n_obs} with true trend state values
#'       }
#'     }
#'     \item{\strong{Bernoulli probabilities} (for \code{type = "alpha"}):}{
#'       \itemize{
#'         \item \code{alpha}: Numeric vector of length \code{n_obs} with true alpha_t probabilities
#'       }
#'     }
#'   }
#'
#'   \strong{Note on state trajectories:} If you only have the true \code{alpha},
#'   you can obtain \code{theta_1} using \code{qnorm(alpha)} (probit link). However,
#'   \code{theta_2} (trend) cannot be recovered from \code{alpha} alone and must
#'   come from your simulation data.
#'
#'   You can provide any subset of these elements. For example, to compare only
#'   initial states and alpha:
#'   \preformatted{
#'   true_values = list(
#'     theta_01 = 0,
#'     theta_02 = 0,
#'     alpha = alpha_true_vector
#'   )
#'   }
#' @param ... Additional arguments passed to plotting functions.
#'
#' @return Invisibly returns the input object \code{x}.
#'
#' @details
#' This function provides comprehensive visual diagnostics for Bayesian MCMC output:
#'
#' \strong{MCMC Diagnostics} (\code{type = "mcmc"}):
#'
#' Each parameter gets a dedicated page with 4 panels:
#' \itemize{
#'   \item \strong{Trace Plot:} Shows MCMC chain trajectory to assess mixing
#'   \item \strong{Autocorrelation:} ACF plot to detect serial correlation
#'   \item \strong{Posterior Density:} Marginal posterior distribution
#'   \item \strong{Running Mean:} Cumulative mean to assess convergence
#' }
#'
#' Available parameters: theta_01, theta_02, W_1^{-1}, W_2^{-1}
#'
#' \strong{Dynamic States} (\code{type = "states"}):
#' \itemize{
#'   \item Time-varying state trajectories with credible bands (on probit scale)
#'   \item Innovation sequences
#'   \item State space relationships
#' }
#'
#' \strong{Bernoulli Probabilities} (\code{type = "alpha"}):
#' \itemize{
#'   \item alpha_t trajectory with credible bands
#'   \item Optional observed binary outcomes overlay (controlled by \code{show_obs})
#' }
#'
#' \strong{Complete Dashboard} (\code{type = "all"}):
#'
#' Generates 7 pages in total:
#' \itemize{
#'   \item Pages 1-4: Individual parameter diagnostics (4 panels each)
#'   \item Pages 5-6: Dynamic state trajectories and diagnostics
#'   \item Page 7: Bernoulli probabilities alpha_t
#' }
#'
#' @section Controlling Observed Data Display:
#'
#' The \code{show_obs} parameter provides control over the display of observed
#' binary outcomes in the Bernoulli probabilities plot:
#'
#' \itemize{
#'   \item When \code{show_obs = TRUE} (default): Observed binary values (0 or 1)
#'     are shown as red points overlaid on the estimated alpha_t trajectory. This
#'     is useful for model validation and assessing goodness-of-fit.
#'   \item When \code{show_obs = FALSE}: Only the estimated trajectory is shown,
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
#' # true parameter values are unknown. We generate synthetic data with a complex
#' # oscillating pattern to mimic real-world temporal variation in binary outcomes.
#'
#' set.seed(123)
#' n <- 500          # Number of time points
#'
#' # Generate complex oscillating Bernoulli probabilities (mimicking temporal patterns)
#' grid_vals <- seq_len(n) / n
#' alpha_true <- (sin(2 * pi * grid_vals) + sin(4 * pi * grid_vals) + 2) / 4
#'
#' # Generate Bernoulli observations (e.g., binary outcomes over time)
#' y <- rbinom(n, size = 1, prob = alpha_true)
#'
#' # Fit the local-trend model with weakly informative priors
#' # (appropriate when we have limited prior knowledge)
#' out <- mcmc_probit_bernoulli_localtrend(
#'   y,
#'   burnin                  = 1000,      # Discard first 1000 iterations
#'   thinning                = 50,        # Keep every 50th iteration
#'   n_chain                 = 1000,      # Retain 1000 posterior samples
#'   # Weakly informative priors for initial states (centered at 0 on probit scale)
#'   prior_theta01_mean      = 0,
#'   prior_theta01_prec      = 1,
#'   prior_theta02_mean      = 0,
#'   prior_theta02_prec      = 1,
#'   # Weakly informative priors for innovation precisions
#'   # (shape = rate implies mean = 1, but with high variance)
#'   prior_prec1_shape       = 100,
#'   prior_prec1_rate        = 1,
#'   prior_prec2_shape       = 400,
#'   prior_prec2_rate        = 1,
#'   verbose                 = TRUE,      # Show progress bar
#'   seed                    = 456        # For reproducibility
#' )
#'
#' # --- Visualization Options ---
#'
#' # 1. Complete diagnostic dashboard (7 pages)
#' #    Includes: MCMC diagnostics, state trajectories, alpha plot
#' plot(out, type = "all")
#'
#' # 2. MCMC convergence diagnostics for all parameters
#' #    Trace plots, ACF, posterior densities, running means
#' plot(out, type = "mcmc")
#'
#' # 3. Focus on initial state parameters only
#' plot(out, type = "mcmc", which = 1:2)  # theta_01, theta_02
#'
#' # 4. Focus on innovation precision parameters
#' plot(out, type = "mcmc", which = 3:4)  # W_1^{-1}, W_2^{-1}
#'
#' # 5. Dynamic state trajectories (theta_1, theta_2 on probit scale)
#' #    Shows level and trend components over time
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
#' pdf("model_diagnostics.pdf", width = 10, height = 8)
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
#' set.seed(10)
#' n <- 100           # Number of time points
#'
#' # True parameter values (these would be unknown in real applications)
#' theta01_true     <- 0        # True initial level (on probit scale)
#' theta02_true     <- 0        # True initial trend
#' prec_theta1_true <- 1000     # True level innovation precision (high = smooth)
#' prec_theta2_true <- 10000    # True trend innovation precision (very smooth)
#'
#' # --- Step 2: Simulate latent states following the state-space model ---
#' # Generate innovation sequences (random shocks to states)
#' u1 <- rnorm(n, mean = 0, sd = sqrt(1 / prec_theta1_true))  # Level innovations
#' u2 <- rnorm(n, mean = 0, sd = sqrt(1 / prec_theta2_true))  # Trend innovations
#'
#' # Initialize state vectors
#' theta1_true <- numeric(n)  # Level state (on probit scale)
#' theta2_true <- numeric(n)  # Trend state
#'
#' # First time point (t=1): state = initial value + innovation
#' theta2_true[1] <- theta02_true + u2[1]
#' theta1_true[1] <- theta01_true + theta02_true + u1[1]
#'
#' # Subsequent time points (t=2,...,n): follow state evolution equations
#' # theta_{t,2} = theta_{t-1,2} + u_{t,2}
#' # theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1}
#' for (t in 2:n) {
#'   theta2_true[t] <- theta2_true[t-1] + u2[t]
#'   theta1_true[t] <- theta1_true[t-1] + theta2_true[t-1] + u1[t]
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
#' par(mfrow = c(2, 2))
#' plot(theta1_true, type = "l", main = "True Level State (probit scale)",
#'      xlab = "Time", ylab = expression(theta["t,1"]))
#' plot(theta2_true, type = "l", main = "True Trend State",
#'      xlab = "Time", ylab = expression(theta["t,2"]))
#' plot(alpha_true, type = "l", main = "True Bernoulli Probabilities",
#'      xlab = "Time", ylab = expression(alpha[t]), ylim = c(0, 1))
#' points(y, col = "red", pch = 16, cex = 0.5)
#' par(mfrow = c(1, 1))
#'
#' # --- Step 4: Fit the model with informative priors ---
#' # Note: In practice, we wouldn't know true values, but here we use
#' # priors centered near truth to demonstrate parameter recovery
#' out <- mcmc_probit_bernoulli_localtrend(
#'   y,
#'   burnin                  = 1000,
#'   thinning                = 50,
#'   n_chain                 = 1000,
#'   # Priors centered at true initial values
#'   prior_theta01_mean      = 0,
#'   prior_theta01_prec      = 1,
#'   prior_theta02_mean      = 0,
#'   prior_theta02_prec      = 1,
#'   # Informative priors for innovation precisions
#'   # (centered near true values with moderate uncertainty)
#'   prior_prec1_shape       = 100,
#'   prior_prec1_rate        = 1,
#'   prior_prec2_shape       = 400,
#'   prior_prec2_rate        = 1,
#'   verbose                 = TRUE,
#'   seed                    = 456
#' )
#'
#' # --- Step 5: Model validation using true parameter values ---
#' # Create named list with ALL true values (matching output component names)
#' # IMPORTANT: All names must match exactly the components returned by
#' # mcmc_probit_bernoulli_localtrend() - see ?mcmc_probit_bernoulli_localtrend
#' true_vals <- list(
#'   # Scalar parameters (for MCMC diagnostics)
#'   theta_01     = theta01_true,
#'   theta_02     = theta02_true,
#'   prec_theta1  = prec_theta1_true,
#'   prec_theta2  = prec_theta2_true,
#'   # State trajectories (for state plots)
#'   theta_1      = theta1_true,
#'   theta_2      = theta2_true,
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
#' # 3. Focus on initial states with true values
#' plot(out, type = "mcmc", which = 1:2, true_values = true_vals)
#'
#' # 4. Focus on innovation precisions with true values
#' plot(out, type = "mcmc", which = 3:4, true_values = true_vals)
#'
#' # 5. Dynamic states with true trajectories overlaid
#' #    Assess how well the model tracks the true time-varying states
#' plot(out, type = "states", true_values = true_vals)
#'
#' # 6. Bernoulli probabilities with true alpha trajectory
#' #    Compare estimated alpha_t with true values
#' plot(out, type = "alpha", true_values = true_vals)
#'
#' # 7. Partial validation: only compare specific components
#' #    Example 1: Only initial states and precisions
#' plot(out, type = "mcmc", true_values = list(
#'   theta_01    = theta01_true,
#'   theta_02    = theta02_true,
#'   prec_theta1 = prec_theta1_true,
#'   prec_theta2 = prec_theta2_true
#' ))
#'
#' #    Example 2: Only level state trajectory
#' plot(out, type = "states", which = 1,
#'      true_values = list(theta_1 = theta1_true))
#'
#' #    Example 3: Only Bernoulli probabilities
#' plot(out, type = "alpha",
#'      true_values = list(alpha = alpha_true))
#'
#' }
#'
#' @seealso \code{\link{mcmc_probit_bernoulli_localtrend}},
#'   \code{\link{summary.probit_bernoulli_localtrend}}
#'
#' @export
plot.probit_bernoulli_localtrend <- function(x,
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
