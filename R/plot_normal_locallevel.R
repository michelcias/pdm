#' Plot method for normal_locallevel objects
#'
#' @description Produces diagnostic plots for MCMC output from Gaussian
#'   local level models.
#'
#' @param x An object of class `normal_locallevel`.
#' @param type Character string specifying the type of plot. One of:
#'   \describe{
#'     \item{\code{"all"}}{Complete dashboard with all diagnostic plots (default)}
#'     \item{\code{"mcmc"}}{MCMC convergence diagnostics (trace plots, ACF, running means)}
#'     \item{\code{"states"}}{Dynamic states (theta_1 trajectory)}
#'   }
#' @param which Integer vector specifying which diagnostic plots to display.
#'   For `type = "mcmc"`:
#'   \describe{
#'     \item{1}{phi_y (observation precision)}
#'     \item{2}{theta_01 (initial level)}
#'     \item{3}{W_1^-1 (level innovation precision)}
#'   }
#'   For `type = "states"`: indices of subplots.
#'   If `NULL` (default), all available plots are shown.
#' @param ask Logical; if `TRUE`, the user is asked before each plot when
#'   `type = "all"`. Default is `interactive()` when `type = "all"`,
#'   `FALSE` otherwise.
#' @param ci Logical; whether to display credible intervals in plots that
#'   support them. Default is `TRUE`.
#' @param ci_level Numeric; Bayesian confidence level for credible intervals
#'   (between 0 and 1). Default is `0.95`.
#' @param true_values Named list containing true parameter values and/or state trajectories
#'   for comparison with MCMC estimates. If `NULL` (default), no true values are displayed.
#'
#'   \strong{Important:} All parameter names in `true_values` must match exactly
#'   the component names returned by \code{\link{mcmc_normal_locallevel}}.
#'
#'   Accepted elements:
#'   \describe{
#'     \item{\strong{Scalar parameters} (for `type = "mcmc"`):}{
#'       \itemize{
#'         \item `prec_y`: Observation precision (phi_y)
#'         \item `theta_01`: Initial level state
#'         \item `prec_theta1`: Level innovation precision (W_1^-1)
#'       }
#'     }
#'     \item{\strong{State trajectories} (for `type = "states"`):}{
#'       \itemize{
#'         \item `theta_1`: Numeric vector of length `n_obs` with true level state values
#'       }
#'     }
#'   }
#'
#'   You can provide any subset of these elements. For example, to compare only
#'   the initial state:
#'   \preformatted{
#'   true_values = list(
#'     theta_01 = 0
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
#' Available parameters: phi_y, theta_01, W_1^-1
#'
#' \strong{Dynamic States} (`type = "states"`):
#' \itemize{
#'   \item Time-varying state trajectory with credible bands
#'   \item Innovation sequence
#' }
#'
#' \strong{Complete Dashboard} (`type = "all"`):
#'
#' Generates 4 pages in total:
#' \itemize{
#'   \item Pages 1-3: Individual parameter diagnostics (4 panels each)
#'   \item Page 4: Dynamic state trajectory and diagnostics
#' }
#'
#' @examples
#' \donttest{
#' # =============================================================================
#' # Example 1: Practical Data Analysis (No True Parameters Known)
#' # =============================================================================
#' # This example demonstrates a typical workflow when analyzing real data where
#' # true parameter values are unknown. We generate synthetic data with a simple
#' # time-varying pattern to mimic real-world temporal variation.
#'
#' set.seed(123)
#' n <- 200          # Number of time points
#'
#' # Generate simple time-varying pattern (mimicking real-world smooth trends)
#' mu_true <- 10 + 3 * sin(2 * pi * seq_len(n) / n)
#'
#' # Generate Gaussian observations
#' y <- rnorm(n, mean = mu_true, sd = 0.5)
#'
#' # Fit the local-level model with weakly informative priors
#' # (appropriate when we have limited prior knowledge)
#' out <- mcmc_normal_locallevel(
#'   y,
#'   burnin                  = 1000,      # Discard first 1000 iterations
#'   thinning                = 20,        # Keep every 20th iteration
#'   n_draws                 = 500,       # Retain 500 posterior samples
#'   # Weakly informative prior for initial state
#'   prior_theta01_mean      = 0,
#'   prior_theta01_prec      = 1 / 100,
#'   # Weakly informative priors for precisions
#'   prior_prec1_shape       = 100,
#'   prior_prec1_rate        = 1,
#'   prior_prec_y_shape      = 10,
#'   prior_prec_y_rate       = 1,
#'   verbose                 = TRUE,      # Show progress bar
#'   seed                    = 456        # For reproducibility
#' )
#'
#' # --- Visualization Options ---
#'
#' # 1. Complete diagnostic dashboard (4 pages)
#' #    Includes: MCMC diagnostics and state trajectory
#' plot(out, type = "all")
#'
#' # 2. MCMC convergence diagnostics for all parameters
#' #    Trace plots, ACF, posterior densities, running means
#' plot(out, type = "mcmc")
#'
#' # 3. Focus on observation precision only
#' plot(out, type = "mcmc", which = 1)  # phi_y
#'
#' # 4. Focus on initial state parameter
#' plot(out, type = "mcmc", which = 2)  # theta_01
#'
#' # 5. Focus on innovation precision parameter
#' plot(out, type = "mcmc", which = 3)  # W_1^-1
#'
#' # 6. Dynamic state trajectory (theta_1)
#' #    Shows level component over time
#' plot(out, type = "states")
#'
#' # 7. States without credible intervals (cleaner for presentations)
#' plot(out, type = "states", ci = FALSE)
#'
#' # 8. Adjust credible interval level (default is 95%)
#' plot(out, type = "states", ci_level = 0.90)  # 90% credible intervals
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
#' set.seed(10)
#' n <- 100           # Number of time points
#'
#' # True parameter values (these would be unknown in real applications)
#' theta01_true     <- 0        # True initial level
#' prec_theta1_true <- 10       # True level innovation precision (1/0.10)
#' prec_y_true      <- 4        # True observation precision (1/0.25)
#'
#' # --- Step 2: Simulate latent states following the state-space model ---
#' # Generate innovation sequence (random shocks to state)
#' u1 <- rnorm(n, mean = 0, sd = sqrt(1 / prec_theta1_true))  # Level innovations
#' epsilon <- rnorm(n, mean = 0, sd = sqrt(1 / prec_y_true))  # Observation noise
#'
#' # Initialize state vector
#' theta1_true <- numeric(n)  # Level state
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
#' # Observations are level state plus Gaussian noise
#' y <- theta1_true + epsilon
#'
#' # Optional: Visualize true trajectories before fitting
#' par(mfrow = c(1, 2))
#' plot(theta1_true, type = "l", main = "True Level State",
#'      xlab = "Time", ylab = expression(theta["t,1"]))
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
#' out <- mcmc_normal_locallevel(
#'   y,
#'   burnin                  = 1000,
#'   thinning                = 20,
#'   n_draws                 = 500,
#'   # Prior centered at true initial value
#'   prior_theta01_mean      = 0,
#'   prior_theta01_prec      = 1 / 100,
#'   # Informative priors for precisions
#'   # (centered near true values with moderate uncertainty)
#'   prior_prec1_shape       = 10,
#'   prior_prec1_rate        = 1,
#'   prior_prec_y_shape      = 1,
#'   prior_prec_y_rate       = 1,
#'   verbose                 = TRUE,
#'   seed                    = 456
#' )
#'
#' # --- Step 5: Model validation using true parameter values ---
#' # Create named list with ALL true values (matching output component names)
#' # IMPORTANT: All names must match exactly the components returned by
#' # mcmc_normal_locallevel() - see ?mcmc_normal_locallevel
#' true_vals <- list(
#'   # Scalar parameters (for MCMC diagnostics)
#'   prec_y       = prec_y_true,
#'   theta_01     = theta01_true,
#'   prec_theta1  = prec_theta1_true,
#'   # State trajectory (for state plots)
#'   theta_1      = theta1_true
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
#' # 4. Focus on initial state with true value
#' plot(out, type = "mcmc", which = 2, true_values = true_vals)
#'
#' # 5. Focus on innovation precision with true value
#' plot(out, type = "mcmc", which = 3, true_values = true_vals)
#'
#' # 6. Dynamic state with true trajectory overlaid
#' #    Assess how well the model tracks the true time-varying state
#' plot(out, type = "states", true_values = true_vals)
#'
#' # 7. Partial validation: only compare specific components
#' #    Example 1: Only observation precision
#' plot(out, type = "mcmc", which = 1,
#'      true_values = list(prec_y = prec_y_true))
#'
#' #    Example 2: Only initial state and innovation precision
#' plot(out, type = "mcmc", which = 2:3, true_values = list(
#'   theta_01    = theta01_true,
#'   prec_theta1 = prec_theta1_true
#' ))
#'
#' #    Example 3: Only level state trajectory
#' plot(out, type = "states",
#'      true_values = list(theta_1 = theta1_true))
#'
#' }
#'
#' @seealso
#'   \code{\link{mcmc_normal_locallevel}} (model generator),
#'   \code{\link{print.normal_locallevel}}, \code{\link{summary.normal_locallevel}}.
#'
#' @export
plot.normal_locallevel <- function(x,
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
