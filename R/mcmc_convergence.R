#' MCMC convergence diagnostics for pdm models
#'
#' @description Assesses the convergence of the Markov chains produced by the
#'   `mcmc_*()` fitting functions. For every scalar parameter of a fitted
#'   `pdm_mcmc` model the function reports the Effective Sample Size (ESS)
#'   and the associated sampling efficiency and, when the \pkg{coda} package is
#'   available, the Geweke convergence diagnostic and the Heidelberger–Welch
#'   stationarity and halfwidth tests. The individual verdicts are combined into
#'   a single `Overall` classification per parameter.
#'
#'   Optionally, selected time points of the latent state chains
#'   (\eqn{\theta_{t,j}}) can also be assessed, one row per state per time
#'   point.
#'
#' @param object An object inheriting from `"pdm_mcmc"`, typically the
#'   result of one of the `mcmc_*()` fitting functions (for example
#'   \code{\link{mcmc_normal_localtrend}}).
#' @param theta_timepoints Numeric vector of fractions in \eqn{(0, 1)} that
#'   determine which time points of the latent state chains are included in the
#'   diagnostics. Each fraction is rounded to the nearest integer index. The
#'   default `c(0.25, 0.5, 0.75)` evaluates the states at the first
#'   quartile, median and third quartile of the series. Set to `NULL` to
#'   exclude all latent states (scalar parameters only).
#' @param ess_thresholds Numeric vector of length 3 giving the efficiency
#'   cut-offs (in percent) for the `ESS_status` labels `EXCELLENT`,
#'   `GOOD` and `ACCEPTABLE`. Default is `c(50, 25, 10)`,
#'   meaning efficiency > 50% is `EXCELLENT`, > 25% is `GOOD`,
#'   \eqn{>} 10% is `ACCEPTABLE`, and \eqn{\leq} 10% is `POOR`.
#'   Values must be strictly decreasing and in the range \eqn{(0, 100)}.
#' @param geweke_level Numeric value in \eqn{(0, 1)}, the significance level
#'   used for the Geweke test. A parameter passes when the absolute Geweke
#'   z-score is below the critical value for level `geweke_level`.
#'   Default is `0.05` (5%, critical value \eqn{\approx 1.96}).
#' @param show_ess_status Logical. Whether to include the `ESS_status`
#'   column in the output table. Default `TRUE`.
#' @param show_geweke Logical. Whether to include the `Geweke_z` and
#'   `Geweke_pass` columns. Default `TRUE`.
#' @param show_heidel Logical. Whether to include the `Heidel_stat` and
#'   `Heidel_hw` columns. Default `TRUE`.
#' @param show_overall Logical. Whether to include the `Overall` column.
#'   Default `TRUE`.
#' @param ... Additional arguments (currently unused).
#'
#' @return An object of class `"pdm_convergence"`, which is a list with:
#'   \describe{
#'     \item{\code{table}}{Data frame with one row per assessed parameter or
#'       state time point. Columns always present: `Parameter`, `ESS`,
#'       `Efficiency` (in percent). Optional columns, controlled by the
#'       `show_*` arguments: `ESS_status`, `Geweke_z`,
#'       `Geweke_pass`, `Heidel_stat`, `Heidel_hw`,
#'       `Overall`.}
#'     \item{\code{n_draws}}{Number of retained MCMC draws (\eqn{N}).}
#'     \item{\code{model_type}}{Character string, e.g. `"locallevel"`,
#'       `"localtrend"` or `"localacceleration"`.}
#'     \item{\code{has_coda}}{Logical, whether \pkg{coda}-based diagnostics are
#'       included.}
#'     \item{\code{ess_thresholds}}{The ESS efficiency thresholds used.}
#'     \item{\code{geweke_level}}{The significance level used for Geweke.}
#'   }
#'
#' @details
#' Only scalar parameters (initial states \eqn{\theta_{0,j}}, innovation
#' precisions \eqn{W_j^{-1}}, and for Gaussian models the observation
#' precision \eqn{V^{-1}}) are always included. When `theta_timepoints`
#' is not `NULL`, selected time points of every latent state matrix
#' (`theta_1`, `theta_2`, \ldots) are also assessed — one row per
#' state per selected time point, labelled as e.g. `theta_1[t=25]`.
#'
#' Note the distinction between the initial states and the state trajectories.
#' The initial states (`theta_01`, `theta_02`, \ldots) are scalars
#' and are reported among the scalar parameters. Only the trajectory matrices
#' (`theta_1`, `theta_2`, \ldots), which hold one column per time
#' point, are evaluated at the time points given by `theta_timepoints`.
#'
#' All diagnostics on this page are \emph{within-chain} criteria, because the
#' object being assessed holds a single chain. To obtain the Gelman–Rubin
#' \eqn{\hat{R}}, fit the model with `chains > 1` (see
#' \code{\link{mcmc_normal_locallevel}}) and pass the resulting
#' `"pdm_mcmc_list"` to the same generic; the method for that class is
#' documented in \code{\link{mcmc_convergence.pdm_mcmc_list}}.
#'
#' \subsection{Effective Sample Size (ESS)}{
#'   Autocorrelation inflates the variance of MCMC estimators relative to an
#'   i.i.d. sample of the same length. The ESS approximates the number of
#'   independent draws that would carry the same information,
#'   \deqn{\mathrm{ESS} = \frac{N}{1 + 2 \sum_{k \ge 1} \rho_k},}
#'   where \eqn{N} is the number of retained draws and \eqn{\rho_k} is the
#'   lag-\eqn{k} sample autocorrelation. Following the truncated-sum approach of
#'   Geyer (1992), the sum is taken over the positive sample autocorrelations up
#'   to a maximum lag. This estimate requires no external package. The reported
#'   efficiency is \eqn{100 \times \mathrm{ESS}/N} and is classified according
#'   to `ess_thresholds`.
#' }
#'
#' \subsection{Geweke diagnostic}{
#'   Geweke (1992) compares the posterior mean of a parameter computed from the
#'   first portion of the chain (the first 10% by default) with that computed
#'   from the last portion (the last 50%). Under convergence the two means
#'   agree and the standardised difference follows a standard normal
#'   distribution. The test passes when the absolute z-score is below the
#'   critical value at level `geweke_level`. Computed with \code{\link[coda]{geweke.diag}}.
#' }
#'
#' \subsection{Heidelberger–Welch tests}{
#'   Heidelberger and Welch (1983) propose two complementary tests. The
#'   \emph{stationarity} test uses a Cramér–von Mises statistic on the
#'   Brownian-bridge representation of the chain to decide whether the retained
#'   draws are consistent with a stationary distribution. The \emph{halfwidth}
#'   test checks whether the chain is long enough to estimate the posterior
#'   mean to a prescribed relative accuracy. Both are computed with
#'   \code{\link[coda]{heidel.diag}} via the named columns `stest` and
#'   `htest`; a parameter is considered well-behaved only when it passes
#'   both.
#' }
#'
#' \subsection{Overall classification}{
#'   When \pkg{coda} is available the Geweke and Heidelberger–Welch verdicts are
#'   pooled into the `Overall` column: `EXCELLENT` when both tests
#'   pass, `ACCEPTABLE` when exactly one passes, and `POOR` when
#'   neither does.
#' }
#'
#' \subsection{Link families: these statistics do \emph{not} stand in for
#'   \code{alpha}}{
#'   On a multi-chain fit, \code{\link{mcmc_convergence.pdm_mcmc_list}} reports
#'   rank-based statistics, and a strictly monotone link leaves ranks unchanged,
#'   so its `theta_1[t=...]` rows are the diagnostics for \eqn{\alpha_t} exactly.
#'   \strong{That argument does not carry over to this method.} The ESS here is
#'   computed from the autocorrelations of the draws themselves, and Geweke and
#'   Heidelberger–Welch compare means and spectral densities; none of the three
#'   is a function of the ranks, so none is invariant under the link. A
#'   diagnostic reported here for \eqn{\theta_{t1}} says nothing exact about
#'   \eqn{\alpha_t}, and vice versa.
#'
#'   Where the weight itself is the estimand — as it is in the mixture and
#'   probit-Bernoulli families — fit with `chains > 1` and read the multi-chain
#'   method rather than inferring from the state.
#' }
#'
#' @references
#' Geweke, J. (1992). Evaluating the accuracy of sampling-based approaches to
#'   the calculation of posterior moments. In J. M. Bernardo, J. O. Berger,
#'   A. P. Dawid, & A. F. M. Smith (Eds.), \emph{Bayesian Statistics 4}
#'   (pp. 169--193). Oxford University Press.
#'
#' Geyer, C. J. (1992). Practical Markov chain Monte Carlo.
#'   \emph{Statistical Science}, \strong{7}(4), 473--483.
#'   \doi{10.1214/ss/1177011137}
#'
#' Heidelberger, P., & Welch, P. D. (1983). Simulation run length control in the
#'   presence of an initial transient. \emph{Operations Research},
#'   \strong{31}(6), 1109--1144. \doi{10.1287/opre.31.6.1109}
#'
#' Kass, R. E., Carlin, B. P., Gelman, A., & Neal, R. M. (1998). Markov chain
#'   Monte Carlo in practice: A roundtable discussion.
#'   \emph{The American Statistician}, \strong{52}(2), 93--100.
#'   \doi{10.1080/00031305.1998.10480547}
#'
#' Plummer, M., Best, N., Cowles, K., & Vines, K. (2006). CODA: Convergence
#'   diagnosis and output analysis for MCMC. \emph{R News}, \strong{6}(1),
#'   7--11.
#'
#' @examples
#' \donttest{
#' ## 1. Simulate data from a local-trend dynamic model --------------------
#' set.seed(123)
#' n <- 200
#'
#' # True parameters
#' theta01_true <- 10        # initial level   (theta[0,1])
#' theta02_true <- 0.5       # initial trend   (theta[0,2])
#' prec1_true   <- 1 / 0.10  # level innovation precision (1/W_1)
#' prec2_true   <- 1 / 0.01  # trend innovation precision (1/W_2)
#' prec_y_true  <- 1 / 1.00  # observation precision      (1/V)
#'
#' # Innovation and observation noise
#' u1      <- rnorm(n, sd = sqrt(1 / prec1_true))
#' u2      <- rnorm(n, sd = sqrt(1 / prec2_true))
#' epsilon <- rnorm(n, sd = sqrt(1 / prec_y_true))
#'
#' # Latent states and observations
#' theta1 <- theta2 <- numeric(n)
#' theta2[1] <- theta02_true + u2[1]
#' theta1[1] <- theta01_true + theta02_true + u1[1]
#' for (t in 2:n) {
#'   theta2[t] <- theta2[t - 1] + u2[t]
#'   theta1[t] <- theta1[t - 1] + theta2[t - 1] + u1[t]
#' }
#' y <- theta1 + epsilon
#'
#' ## 2. Fit the local-trend model -----------------------------------------
#' fit <- mcmc_normal_localtrend(
#'   y,
#'   burnin             = 1000,
#'   thinning           = 10,
#'   n_draws            = 500,
#'   prior_theta01_mean = y[1] / 2,
#'   prior_theta01_prec = 1 / var(y),
#'   prior_theta02_mean = y[1] / 2,
#'   prior_theta02_prec = 1 / var(y),
#'   prior_prec1_shape  = 1e-1,
#'   prior_prec1_rate   = 1e-1,
#'   prior_prec2_shape  = 1e-2,
#'   prior_prec2_rate   = 1e-2,
#'   prior_prec_y_shape = 1e-1,
#'   prior_prec_y_rate  = 1e-1,
#'   seed               = 456
#' )
#'
#' ## 3. Default diagnostics (scalars + states at 25\%, 50\%, 75\%) ---------
#' conv <- mcmc_convergence(fit)
#' print(conv)
#'
#' ## 4. Scalar parameters only --------------------------------------------
#' conv_scalar <- mcmc_convergence(fit, theta_timepoints = NULL)
#' print(conv_scalar)
#'
#' ## 5. Custom time points and stricter Geweke level ----------------------
#' conv_custom <- mcmc_convergence(
#'   fit,
#'   theta_timepoints = c(0.1, 0.5, 0.9),
#'   geweke_level     = 0.01
#' )
#' print(conv_custom)
#'
#' ## 6. Compact table: only ESS columns -----------------------------------
#' conv_ess <- mcmc_convergence(
#'   fit,
#'   show_geweke  = FALSE,
#'   show_heidel  = FALSE,
#'   show_overall = FALSE
#' )
#' print(conv_ess)
#'
#' ## 7. Stricter ESS thresholds -------------------------------------------
#' conv_strict <- mcmc_convergence(fit, ess_thresholds = c(75, 50, 25))
#' print(conv_strict)
#'
#' # Post-process: flag parameters with ESS efficiency below 25\%:
#' conv$table[conv$table$Efficiency < 25, ]
#' }
#'
#' @seealso \code{\link{mcmc_normal_localtrend}},
#'   \code{\link{mcmc_normal_locallevel}},
#'   \code{\link{summary.normal_locallevel}};
#'   \code{\link[coda]{geweke.diag}} and \code{\link[coda]{heidel.diag}} for the
#'   underlying \pkg{coda} implementations.
#'
#' @export
mcmc_convergence <- function(object, ...) {
  UseMethod("mcmc_convergence")
}


#' @rdname mcmc_convergence
#' @export
mcmc_convergence.pdm_mcmc <- function(object,
                                      theta_timepoints = c(0.25, 0.5, 0.75),
                                      ess_thresholds   = c(50, 25, 10),
                                      geweke_level     = 0.05,
                                      show_ess_status  = TRUE,
                                      show_geweke      = TRUE,
                                      show_heidel      = TRUE,
                                      show_overall     = TRUE,
                                      ...) {

  # --- Input validation ---------------------------------------------------
  if (!inherits(object, "pdm_mcmc")) {
    stop("'object' must inherit from 'pdm_mcmc'")
  }

  if (!is.null(theta_timepoints)) {
    if (!is.numeric(theta_timepoints) ||
        any(theta_timepoints <= 0) || any(theta_timepoints >= 1)) {
      stop("'theta_timepoints' must be a numeric vector with values in (0, 1), or NULL")
    }
    theta_timepoints <- sort(unique(theta_timepoints))
  }

  if (!is.numeric(ess_thresholds) || length(ess_thresholds) != 3L ||
      any(ess_thresholds <= 0) || any(ess_thresholds >= 100) ||
      !all(diff(ess_thresholds) < 0)) {
    stop("'ess_thresholds' must be a strictly decreasing numeric vector of length 3 with values in (0, 100)")
  }

  if (!is.numeric(geweke_level) || length(geweke_level) != 1L ||
      geweke_level <= 0 || geweke_level >= 1) {
    stop("'geweke_level' must be a single numeric value in (0, 1)")
  }

  for (flag in list(show_ess_status, show_geweke, show_heidel, show_overall)) {
    if (!is.logical(flag) || length(flag) != 1L) {
      stop("'show_*' arguments must be single logical values")
    }
  }
  # ------------------------------------------------------------------------

  n_draws    <- as.integer(attr(object, "n_draws"))
  model_type <- attr(object, "model_type")
  has_coda   <- requireNamespace("coda", quietly = TRUE)

  # Critical z value for Geweke
  geweke_z_crit <- qnorm(1 - geweke_level / 2)

  # Helper: compute diagnostics row for a single draws vector
  compute_row <- function(draws, label) {

    # ESS
    acf_vals <- acf(draws, plot = FALSE,
                    lag.max = min(100L, floor(n_draws / 4L)))$acf[-1L]
    ess        <- max(1, n_draws / (1 + 2 * sum(acf_vals[acf_vals > 0])))
    efficiency <- 100 * ess / n_draws
    ess_status <- if (efficiency > ess_thresholds[1]) "EXCELLENT" else
                  if (efficiency > ess_thresholds[2]) "GOOD"      else
                  if (efficiency > ess_thresholds[3]) "ACCEPTABLE" else "POOR"

    # Stored unrounded; `print.pdm_convergence()` decides how many digits to
    # show. Rounding here degraded the object for no gain, and worse, disagreed
    # with the print method: storage kept one decimal while the display asked
    # for three, so every printed ESS ended in two zeros it had not earned.
    row <- data.frame(
      Parameter  = label,
      ESS        = ess,
      Efficiency = efficiency,
      stringsAsFactors = FALSE
    )

    if (show_ess_status) row$ESS_status <- ess_status

    # coda diagnostics
    if (has_coda) {
      mcmc_obj <- coda::mcmc(draws)

      # Geweke
      gz          <- tryCatch(coda::geweke.diag(mcmc_obj)$z,
                               error = function(e) NA_real_)
      geweke_pass <- !is.na(gz) && is.finite(gz) && abs(gz) < geweke_z_crit

      # Heidelberger-Welch
      hw <- tryCatch(coda::heidel.diag(mcmc_obj), error = function(e) NULL)
      if (!is.null(hw) && is.matrix(hw) && nrow(hw) >= 1L) {
        heidel_stat <- ifelse(hw[1L, "stest"], "PASS", "FAIL")
        heidel_hw   <- ifelse(hw[1L, "htest"], "PASS", "FAIL")
        heidel_pass <- hw[1L, "stest"] && hw[1L, "htest"]
      } else {
        heidel_stat <- "ERROR"
        heidel_hw   <- "ERROR"
        heidel_pass <- FALSE
      }

      # Overall
      tests_pass <- sum(c(geweke_pass, heidel_pass))
      overall    <- if (tests_pass == 2L) "EXCELLENT"  else
                    if (tests_pass == 1L) "ACCEPTABLE" else "POOR"

      if (show_geweke) {
        row$Geweke_z    <- gz
        row$Geweke_pass <- ifelse(geweke_pass, "PASS", "FAIL")
      }
      if (show_heidel) {
        row$Heidel_stat <- heidel_stat
        row$Heidel_hw   <- heidel_hw
      }
      if (show_overall) {
        row$Overall <- overall
      }
    }

    row
  }

  # --- Scalar parameters --------------------------------------------------
  param_config <- get_param_config(object)
  rows_scalar  <- lapply(names(param_config), function(nm) {
    compute_row(param_config[[nm]]$draws, param_config[[nm]]$name_str)
  })

  # --- Latent state time points -------------------------------------------
  rows_theta <- list()

  if (!is.null(theta_timepoints)) {

    # Detect the latent-state trajectory matrices in the object. These are
    # named theta_1, theta_2, theta_3 (no leading zero) and stored as
    # n_draws x n_obs matrices. The initial-state scalars theta_01, theta_02,
    # theta_03 must NOT be captured here: they are vectors, already reported as
    # scalar parameters, and a leading-zero name would otherwise match.
    state_names <- grep("^theta_[1-9][0-9]*$", names(object), value = TRUE)
    state_names <- state_names[vapply(state_names,
                                      function(nm) is.matrix(object[[nm]]),
                                      logical(1L))]
    # Sort by the numeric suffix (theta_1 < theta_2 < theta_3)
    state_names <- state_names[order(as.integer(sub("theta_", "", state_names)))]

    if (length(state_names) > 0L) {
      for (sname in state_names) {
        mat    <- object[[sname]]          # n_draws x n_obs matrix
        j      <- sub("theta_", "", sname) # "1", "2", ...

        # Use the actual number of time points stored in this matrix so the
        # column index can never fall outside its bounds.
        n_t          <- ncol(mat)
        time_indices <- pmax(1L, pmin(n_t, round(theta_timepoints * n_t)))
        time_indices <- sort(unique(time_indices))

        for (tidx in time_indices) {
          label   <- sprintf("theta_%s[t=%d]", j, tidx)
          draws <- mat[, tidx]
          rows_theta[[length(rows_theta) + 1L]] <- compute_row(draws, label)
        }
      }
    }
  }

  # --- Combine ------------------------------------------------------------
  all_rows <- c(rows_scalar, rows_theta)
  table    <- do.call(rbind, all_rows)
  rownames(table) <- NULL

  result <- list(
    table          = table,
    n_draws        = n_draws,
    model_type     = model_type,
    has_coda       = has_coda,
    ess_thresholds = ess_thresholds,
    geweke_level   = geweke_level
  )
  class(result) <- "pdm_convergence"
  result
}


#' Print method for pdm_convergence objects
#'
#' @param x An object of class `"pdm_convergence"`.
#' @param digits Integer, significant digits for numeric columns. Default 3.
#' @param ... Additional arguments (currently unused).
#'
#' @return Invisibly returns `x`.
#' @export
print.pdm_convergence <- function(x, digits = 3L, ...) {

  cat("\n")
  cat("MCMC Convergence Diagnostics\n")
  cat(strrep("=", 70), "\n\n", sep = "")
  cat("Model:         ", x$model_type, "\n", sep = "")
  cat("Draws:         ", x$n_draws, "\n", sep = "")

  # Show active settings only when non-default
  thr <- x$ess_thresholds
  if (!identical(thr, c(50, 25, 10))) {
    cat(sprintf("ESS thresholds: EXCELLENT >%g%%, GOOD >%g%%, ACCEPTABLE >%g%%\n",
                thr[1], thr[2], thr[3]))
  }
  if (!is.null(x$geweke_level) && !identical(x$geweke_level, 0.05)) {
    cat(sprintf("Geweke level:   alpha = %g (|z| < %.4f)\n",
                x$geweke_level, qnorm(1 - x$geweke_level / 2)))
  }
  cat("\n")

  df <- x$table

  # Format numeric columns
  # Same split as print.pdm_convergence_multi(): a count and a percentage want
  # one decimal whatever `digits` says -- three on an effective sample size is
  # noise -- while the test statistic follows `digits`, because there the fourth
  # decimal can matter.
  for (col in intersect(c("ESS", "Efficiency"), names(df))) {
    df[[col]] <- sprintf("%.1f", df[[col]])
  }
  if ("Geweke_z" %in% names(df)) {
    df$Geweke_z <- sprintf(paste0("%.", digits, "f"), df$Geweke_z)
  }
  names(df)[names(df) == "Efficiency"] <- "Efficiency(%)"

  print(df, row.names = FALSE, right = TRUE)

  cat("\n")
  if (!x$has_coda) {
    cat("Install the 'coda' package for Geweke and Heidelberger-Welch diagnostics.\n")
  }
  thr <- x$ess_thresholds
  cat(sprintf("ESS status: >%g%% EXCELLENT, >%g%% GOOD, >%g%% ACCEPTABLE, else POOR\n",
              thr[1], thr[2], thr[3]))
  cat(strrep("-", 70), "\n\n", sep = "")

  invisible(x)
}
