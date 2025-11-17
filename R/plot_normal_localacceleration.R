#' Plot method for normal_localacceleration objects
#'
#' @description Produces diagnostic plots for MCMC output from Gaussian
#'   local acceleration models.
#'
#' @param x An object of class \code{normal_localacceleration}.
#' @param type Character string specifying the type of plot. One of:
#'   \describe{
#'     \item{\code{"all"}}{Complete dashboard with all diagnostic plots (default)}
#'     \item{\code{"mcmc"}}{MCMC convergence diagnostics (trace plots, ACF, running means)}
#'     \item{\code{"states"}}{Dynamic states (theta_1, theta_2, theta_3 trajectories)}
#'   }
#' @param which Integer vector specifying which diagnostic plots to display.
#'   For \code{type = "mcmc"}:
#'   \describe{
#'     \item{1}{V^{-1} (observation precision)}
#'     \item{2}{theta_01 (initial level)}
#'     \item{3}{theta_02 (initial trend)}
#'     \item{4}{theta_03 (initial acceleration)}
#'     \item{5}{W_1^{-1} (level innovation precision)}
#'     \item{6}{W_2^{-1} (trend innovation precision)}
#'     \item{7}{W_3^{-1} (acceleration innovation precision)}
#'   }
#'   For \code{type = "states"}: indices of subplots.
#'   If \code{NULL} (default), all available plots are shown.
#' @param ask Logical; if \code{TRUE}, the user is asked before each plot when
#'   \code{type = "all"}. Default is \code{interactive()} when \code{type = "all"},
#'   \code{FALSE} otherwise.
#' @param ci Logical; whether to display credible intervals in plots that
#'   support them. Default is \code{TRUE}.
#' @param ci_level Numeric; Bayesian confidence level for credible intervals
#'   (between 0 and 1). Default is \code{0.95}.
#' @param true_values Named list containing true parameter values and/or state trajectories
#'   for comparison with MCMC estimates. If \code{NULL} (default), no true values are displayed.
#'
#'   \strong{Important:} All parameter names in \code{true_values} must match exactly
#'   the component names returned by \code{\link{mcmc_normal_localacceleration}}.
#'
#'   Accepted elements:
#'   \describe{
#'     \item{\strong{Scalar parameters} (for \code{type = "mcmc"}):}{
#'       \itemize{
#'         \item \code{prec_y}: Observation precision (V^{-1})
#'         \item \code{theta_01}: Initial level state
#'         \item \code{theta_02}: Initial trend state
#'         \item \code{theta_03}: Initial acceleration state
#'         \item \code{prec_theta1}: Level innovation precision (W_1^{-1})
#'         \item \code{prec_theta2}: Trend innovation precision (W_2^{-1})
#'         \item \code{prec_theta3}: Acceleration innovation precision (W_3^{-1})
#'       }
#'     }
#'     \item{\strong{State trajectories} (for \code{type = "states"}):}{
#'       \itemize{
#'         \item \code{theta_1}: Numeric vector of length \code{n_obs} with true level state values
#'         \item \code{theta_2}: Numeric vector of length \code{n_obs} with true trend state values
#'         \item \code{theta_3}: Numeric vector of length \code{n_obs} with true acceleration state values
#'       }
#'     }
#'   }
#'
#'   You can provide any subset of these elements. For example, to compare only
#'   initial states:
#'   \preformatted{
#'   true_values = list(
#'     theta_01 = 10,
#'     theta_02 = 0.5,
#'     theta_03 = 0.01
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
#' Available parameters: V^{-1}, theta_01, theta_02, theta_03, W_1^{-1},
#' W_2^{-1}, W_3^{-1}
#'
#' \strong{Dynamic States} (\code{type = "states"}):
#' \itemize{
#'   \item Time-varying state trajectories with credible bands
#'   \item Innovation sequences
#'   \item State space relationships
#' }
#'
#' \strong{Complete Dashboard} (\code{type = "all"}):
#'
#' Generates 10 pages in total:
#' \itemize{
#'   \item Pages 1-7: Individual parameter diagnostics (4 panels each)
#'   \item Pages 8-10: Dynamic state trajectories and diagnostics
#' }
#'
#' @examples
#' \dontrun{
#' # =============================================================================
#' # Example 1: Practical Data Analysis (No True Parameters Known)
#' # =============================================================================
#' # This example demonstrates a typical workflow when analyzing real data where
#' # true parameter values are unknown. We generate synthetic data with a complex
#' # time-varying pattern to mimic real-world temporal variation.
#'
#' set.seed(123)
#' n <- 500          # Number of time points
#'
#' # Generate complex time-varying pattern (mimicking real-world trends)
#' grid_vals <- seq_len(n) / n
#' mu_true <- 10 - 5 * sin(4 * pi * grid_vals) + 2 * sin(8 * pi * grid_vals)
#'
#' # Generate Gaussian observations
#' y <- rnorm(n, mean = mu_true, sd = 1)
#'
#' # Fit the local-acceleration model with weakly informative priors
#' # (appropriate when we have limited prior knowledge)
#' out <- mcmc_normal_localacceleration(
#'   y,
#'   burnin                  = 10000,     # Discard first 2000 iterations
#'   thinning                = 50,        # Keep every 50th iteration
#'   n_chain                 = 1000,      # Retain 1000 posterior samples
#'   # Weakly informative priors for initial states
#'   prior_theta01_mean      = 0,
#'   prior_theta01_prec      = 1 / 100,
#'   prior_theta02_mean      = 0,
#'   prior_theta02_prec      = 1 / 100,
#'   prior_theta03_mean      = 0,
#'   prior_theta03_prec      = 1 / 100,
#'   # Weakly informative priors for precisions
#'   prior_prec1_shape       = 100,
#'   prior_prec1_rate        = 1,
#'   prior_prec2_shape       = 1000,
#'   prior_prec2_rate        = 1,
#'   prior_prec3_shape       = 10000,
#'   prior_prec3_rate        = 1,
#'   prior_prec_y_shape      = 1,
#'   prior_prec_y_rate       = 1,
#'   verbose                 = TRUE,      # Show progress bar
#'   seed                    = 456        # For reproducibility
#' )
#'
#' # --- Visualization Options ---
#'
#' # 1. Complete diagnostic dashboard (10 pages)
#' #    Includes: MCMC diagnostics and state trajectories
#' plot(out, type = "all")
#'
#' # 2. MCMC convergence diagnostics for all parameters
#' #    Trace plots, ACF, posterior densities, running means
#' plot(out, type = "mcmc")
#'
#' # 3. Focus on observation precision only
#' plot(out, type = "mcmc", which = 1)  # V^{-1}
#'
#' # 4. Focus on initial state parameters
#' plot(out, type = "mcmc", which = 2:4)  # theta_01, theta_02, theta_03
#'
#' # 5. Focus on innovation precision parameters
#' plot(out, type = "mcmc", which = 5:7)  # W_1^{-1}, W_2^{-1}, W_3^{-1}
#'
#' # 6. Dynamic state trajectories (theta_1, theta_2, theta_3)
#' #    Shows level, trend, and acceleration components over time
#' plot(out, type = "states")
#'
#' # 7. States without credible intervals (cleaner for presentations)
#' plot(out, type = "states", ci = FALSE)
#'
#' # 8. Adjust credible interval level (default is 95%)
#' plot(out, type = "states", ci_level = 0.90)  # 90% credible intervals
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
#' set.seed(6)
#' n <- 100           # Number of time points
#'
#' # True parameter values (these would be unknown in real applications)
#' theta01_true     <- 1        # True initial level
#' theta02_true     <- 0.5      # True initial trend
#' theta03_true     <- 0.1      # True initial acceleration
#' prec_theta1_true <- 10      # True level innovation precision (1/0.10)
#' prec_theta2_true <- 100     # True trend innovation precision (1/0.01)
#' prec_theta3_true <- 1000    # True acceleration precision (1/0.001)
#' prec_y_true      <- 1        # True observation precision (1/1.00)
#'
#' # --- Step 2: Simulate latent states following the state-space model ---
#' # Generate innovation sequences (random shocks to states)
#' u1 <- rnorm(n, mean = 0, sd = sqrt(1 / prec_theta1_true))  # Level innovations
#' u2 <- rnorm(n, mean = 0, sd = sqrt(1 / prec_theta2_true))  # Trend innovations
#' u3 <- rnorm(n, mean = 0, sd = sqrt(1 / prec_theta3_true))  # Accel. innovations
#' epsilon <- rnorm(n, mean = 0, sd = sqrt(1 / prec_y_true))  # Observation noise
#'
#' # Initialize state vectors
#' theta1_true <- numeric(n)  # Level state
#' theta2_true <- numeric(n)  # Trend state
#' theta3_true <- numeric(n)  # Acceleration state
#'
#' # First time point (t=1): state = initial value + innovation
#' theta3_true[1] <- theta03_true + u3[1]
#' theta2_true[1] <- theta02_true + theta03_true + u2[1]
#' theta1_true[1] <- theta01_true + theta02_true + u1[1]
#'
#' # Subsequent time points (t=2,...,n): follow state evolution equations
#' # theta_{t,3} = theta_{t-1,3} + u_{t,3}
#' # theta_{t,2} = theta_{t-1,2} + theta_{t-1,3} + u_{t,2}
#' # theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1}
#' for (t in 2:n) {
#'   theta3_true[t] <- theta3_true[t-1] + u3[t]
#'   theta2_true[t] <- theta2_true[t-1] + theta3_true[t-1] + u2[t]
#'   theta1_true[t] <- theta1_true[t-1] + theta2_true[t-1] + u1[t]
#' }
#'
#' # --- Step 3: Generate observations ---
#' # Observations are level state plus Gaussian noise
#' y <- theta1_true + epsilon
#'
#' # Optional: Visualize true trajectories before fitting
#' par(mfrow = c(2, 2))
#' plot(theta1_true, type = "l", main = "True Level State",
#'      xlab = "Time", ylab = expression(theta["t,1"]))
#' plot(theta2_true, type = "l", main = "True Trend State",
#'      xlab = "Time", ylab = expression(theta["t,2"]))
#' plot(theta3_true, type = "l", main = "True Acceleration State",
#'      xlab = "Time", ylab = expression(theta["t,3"]))
#' plot(y, type = "p", main = "Observations vs True Level",
#'      xlab = "Time", ylab = "y", pch = 16, col = "gray50")
#' lines(theta1_true, col = "red", lwd = 2)
#' legend("topright", legend = c("True level", "Observations"),
#'        col = c("red", "gray50"), lty = c(1, NA), pch = c(NA, 16))
#' par(mfrow = c(1, 1))
#'
#' # --- Step 4: Fit the model with informative priors ---
#' # Note: In practice, we wouldn't know true values, but here we use
#' # priors centered near truth to demonstrate parameter recovery
#' out <- mcmc_normal_localacceleration(
#'   y,
#'   burnin                  = 10000,
#'   thinning                = 100,
#'   n_chain                 = 1000,
#'   # Priors centered at true initial values
#'   prior_theta01_mean      = 0,
#'   prior_theta01_prec      = 1 / 100,
#'   prior_theta02_mean      = 0,
#'   prior_theta02_prec      = 1 / 100,
#'   prior_theta03_mean      = 0,
#'   prior_theta03_prec      = 1 / 100,
#'   # Informative priors for precisions
#'   # (centered near true values with moderate uncertainty)
#'   prior_prec1_shape       = 10,
#'   prior_prec1_rate        = 1,
#'   prior_prec2_shape       = 100,
#'   prior_prec2_rate        = 1,
#'   prior_prec3_shape       = 1000,
#'   prior_prec3_rate        = 1,
#'   prior_prec_y_shape      = 1,
#'   prior_prec_y_rate       = 1,
#'   verbose                 = TRUE,
#'   seed                    = 456
#' )
#'
#' # --- Step 5: Model validation using true parameter values ---
#' # Create named list with ALL true values (matching output component names)
#' # IMPORTANT: All names must match exactly the components returned by
#' # mcmc_normal_localacceleration() - see ?mcmc_normal_localacceleration
#' true_vals <- list(
#'   # Scalar parameters (for MCMC diagnostics)
#'   prec_y       = prec_y_true,
#'   theta_01     = theta01_true,
#'   theta_02     = theta02_true,
#'   theta_03     = theta03_true,
#'   prec_theta1  = prec_theta1_true,
#'   prec_theta2  = prec_theta2_true,
#'   prec_theta3  = prec_theta3_true,
#'   # State trajectories (for state plots)
#'   theta_1      = theta1_true,
#'   theta_2      = theta2_true,
#'   theta_3      = theta3_true
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
#' # 3. Focus on observation precision with true value
#' plot(out, type = "mcmc", which = 1, true_values = true_vals)
#'
#' # 4. Focus on initial states with true values
#' plot(out, type = "mcmc", which = 2:4, true_values = true_vals)
#'
#' # 5. Focus on innovation precisions with true values
#' plot(out, type = "mcmc", which = 5:7, true_values = true_vals)
#'
#' # 6. Dynamic states with true trajectories overlaid
#' #    Assess how well the model tracks the true time-varying states
#' plot(out, type = "states", true_values = true_vals)
#'
#' # 7. Partial validation: only compare specific components
#' #    Example 1: Only observation precision and initial states
#' plot(out, type = "mcmc", true_values = list(
#'   prec_y       = prec_y_true,
#'   theta_01     = theta01_true,
#'   theta_02     = theta02_true,
#'   theta_03     = theta03_true
#' ))
#'
#' #    Example 2: Only innovation precisions
#' plot(out, type = "mcmc", which = 5:7, true_values = list(
#'   prec_theta1 = prec_theta1_true,
#'   prec_theta2 = prec_theta2_true,
#'   prec_theta3 = prec_theta3_true
#' ))
#'
#' #    Example 3: Only level state trajectory
#' plot(out, type = "states",
#'      true_values = list(theta_1 = theta1_true))
#'
#' #    Example 4: Only trend and acceleration trajectories
#' plot(out, type = "states",
#'      true_values = list(
#'        theta_2 = theta2_true,
#'        theta_3 = theta3_true
#'      ))
#'
#' }
#'
#' @seealso \code{\link{mcmc_normal_localacceleration}},
#'   \code{\link{summary.normal_localacceleration}}
#'
#' @export
plot.normal_localacceleration <- function(x,
                                          type = c("all", "mcmc", "states"),
                                          which = NULL,
                                          ask = NULL,
                                          ci = TRUE,
                                          ci_level = 0.95,
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
                                                   ...)
  )

  invisible(x)
}
