#' MCMC convergence diagnostics for pdm models
#'
#' @description Assesses the convergence of the Markov chains produced by the
#'   \code{mcmc_*()} fitting functions. For every scalar parameter of a fitted
#'   \code{pdm_mcmc} model the function reports the Effective Sample Size (ESS)
#'   and the associated sampling efficiency and, when the \pkg{coda} package is
#'   available, the Geweke convergence diagnostic and the Heidelberger–Welch
#'   stationarity and halfwidth tests. The individual verdicts are combined into
#'   a single \code{Overall} classification per parameter.
#'
#' @param object An object inheriting from \code{"pdm_mcmc"}, typically the
#'   result of one of the \code{mcmc_*()} fitting functions (for example
#'   \code{\link{mcmc_normal_localtrend}}).
#' @param ... Additional arguments (currently unused).
#'
#' @return An object of class \code{"pdm_convergence"}, which is a list with:
#'   \describe{
#'     \item{\code{table}}{Data frame with one row per scalar parameter. Columns
#'       always present: \code{Parameter}, \code{ESS}, \code{Efficiency} (the
#'       ratio \eqn{100 \times \mathrm{ESS} / N}, in percent), and
#'       \code{ESS_status}. When \pkg{coda} is available, the additional columns
#'       \code{Geweke_z}, \code{Geweke_pass}, \code{Heidel_stat},
#'       \code{Heidel_hw} and \code{Overall} are appended.}
#'     \item{\code{n_chain}}{Number of retained MCMC samples (\eqn{N}).}
#'     \item{\code{model_type}}{Character string, e.g. \code{"locallevel"},
#'       \code{"localtrend"} or \code{"localacceleration"}.}
#'     \item{\code{has_coda}}{Logical, whether \pkg{coda}-based diagnostics are
#'       included.}
#'   }
#'
#' @details
#' Only the scalar parameters of the model are assessed, namely the initial
#' states \eqn{\theta_{0,j}}, the innovation precisions \eqn{W_j^{-1}} and, for
#' Gaussian models, the observation precision \eqn{V^{-1}} (plus the mixture
#' components for mixture models). These are extracted automatically, so a
#' single call works for every model family in the package. The time-varying
#' latent states (\code{theta_1}, \code{theta_2}, \ldots), being matrices with
#' one column per time point, are \emph{not} summarised here; inspect their
#' trajectories with \code{plot()} instead.
#'
#' Because the \code{mcmc_*()} samplers return a single chain, the diagnostics
#' below are all \emph{within-chain} criteria. Multi-chain diagnostics such as
#' the Gelman–Rubin potential scale reduction factor \eqn{\hat{R}} require
#' several independent chains and are therefore not computed; run the sampler
#' repeatedly with different seeds if such a comparison is desired.
#'
#' \subsection{Effective Sample Size (ESS)}{
#'   Autocorrelation inflates the variance of MCMC estimators relative to an
#'   i.i.d. sample of the same length. The ESS approximates the number of
#'   independent draws that would carry the same information,
#'   \deqn{\mathrm{ESS} = \frac{N}{1 + 2 \sum_{k \ge 1} \rho_k},}
#'   where \eqn{N} is the number of retained samples and \eqn{\rho_k} is the
#'   lag-\eqn{k} sample autocorrelation. Following the truncated-sum approach of
#'   Geyer (1992), the sum is taken over the positive sample autocorrelations up
#'   to a maximum lag. This estimate requires no external package. The reported
#'   efficiency is \eqn{100 \times \mathrm{ESS}/N} and is classified as
#'   \code{EXCELLENT} (> 50\%), \code{GOOD} (> 25\%), \code{ACCEPTABLE} (> 10\%)
#'   or \code{POOR} otherwise.
#' }
#'
#' \subsection{Geweke diagnostic}{
#'   Geweke (1992) compares the posterior mean of a parameter computed from the
#'   first portion of the chain (the first 10\% by default) with that computed
#'   from the last portion (the last 50\%). Under convergence the two means
#'   agree and the standardised difference follows a standard normal
#'   distribution; the test therefore reports a z-score, and \eqn{|z| < 1.96}
#'   (the 5\% level) is taken as evidence of convergence. Computed with
#'   \code{\link[coda]{geweke.diag}}.
#' }
#'
#' \subsection{Heidelberger–Welch tests}{
#'   Heidelberger and Welch (1983) propose two complementary tests. The
#'   \emph{stationarity} test uses a Cramér–von Mises statistic on the
#'   Brownian-bridge representation of the chain to decide whether the retained
#'   draws are consistent with a stationary distribution. The \emph{halfwidth}
#'   test then checks whether the chain is long enough to estimate the posterior
#'   mean to a prescribed relative accuracy. Both are computed with
#'   \code{\link[coda]{heidel.diag}}; a parameter is considered well-behaved
#'   only when it passes both.
#' }
#'
#' \subsection{Overall classification}{
#'   When \pkg{coda} is available the Geweke and Heidelberger–Welch verdicts are
#'   pooled into the \code{Overall} column: \code{EXCELLENT} when both tests
#'   pass, \code{ACCEPTABLE} when exactly one passes, and \code{POOR} when
#'   neither does. The ESS efficiency is reported alongside as an independent,
#'   always-available indicator.
#' }
#'
#' @references
#' Geyer, C. J. (1992). Practical Markov chain Monte Carlo.
#'   \emph{Statistical Science}, \strong{7}(4), 473--483.
#'   \doi{10.1214/ss/1177011137}
#'
#' Geweke, J. (1992). Evaluating the accuracy of sampling-based approaches to
#'   the calculation of posterior moments. In J. M. Bernardo, J. O. Berger,
#'   A. P. Dawid, & A. F. M. Smith (Eds.), \emph{Bayesian Statistics 4}
#'   (pp. 169--193). Oxford University Press.
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
#' n <- 300
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
#'   n_chain            = 1000,
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
#' ## 3. Convergence diagnostics -------------------------------------------
#' conv <- mcmc_convergence(fit)
#' print(conv)
#'
#' # The diagnostics table can be inspected or post-processed directly:
#' conv$table
#'
#' # Flag any parameter whose ESS efficiency falls below 25\%:
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
mcmc_convergence.pdm_mcmc <- function(object, ...) {

  if (!inherits(object, "pdm_mcmc")) {
    stop("'object' must inherit from 'pdm_mcmc'")
  }

  n_chain    <- as.integer(attr(object, "n_chain"))
  model_type <- attr(object, "model_type")

  # Extract scalar parameter chains via the shared utility
  param_config <- get_param_config(object)

  # Build result table row by row
  has_coda <- requireNamespace("coda", quietly = TRUE)

  rows <- lapply(names(param_config), function(nm) {
    samples <- param_config[[nm]]$samples
    label   <- param_config[[nm]]$name_str

    # --- ESS (no coda needed) ---
    acf_vals <- acf(samples, plot = FALSE,
                    lag.max = min(100L, floor(n_chain / 4L)))$acf[-1L]
    ess <- max(1, n_chain / (1 + 2 * sum(acf_vals[acf_vals > 0])))
    efficiency <- 100 * ess / n_chain
    ess_status <- if (efficiency > 50) "EXCELLENT" else
                  if (efficiency > 25) "GOOD"      else
                  if (efficiency > 10) "ACCEPTABLE" else "POOR"

    row <- data.frame(
      Parameter  = label,
      ESS        = round(ess, 1),
      Efficiency = round(efficiency, 1),
      ESS_status = ess_status,
      stringsAsFactors = FALSE
    )

    # --- coda diagnostics ---
    if (has_coda) {
      mcmc_obj <- coda::mcmc(samples)

      # Geweke
      gz <- tryCatch(coda::geweke.diag(mcmc_obj)$z, error = function(e) NA_real_)
      geweke_pass <- !is.na(gz) && is.finite(gz) && abs(gz) < 1.96

      # Heidelberger-Welch
      hw <- tryCatch(coda::heidel.diag(mcmc_obj), error = function(e) NULL)
      if (!is.null(hw) && is.matrix(hw) && nrow(hw) >= 1L) {
        heidel_stat <- ifelse(hw[1L, 1L], "PASS", "FAIL")
        heidel_hw   <- ifelse(hw[1L, 3L], "PASS", "FAIL")
        heidel_pass <- hw[1L, 1L] && hw[1L, 3L]
      } else {
        heidel_stat <- "ERROR"
        heidel_hw   <- "ERROR"
        heidel_pass <- FALSE
      }

      # Overall: count passing tests
      tests_pass <- sum(c(geweke_pass, heidel_pass))
      overall <- if (tests_pass == 2L) "EXCELLENT"  else
                 if (tests_pass == 1L) "ACCEPTABLE" else "POOR"

      row$Geweke_z    <- round(gz, 4)
      row$Geweke_pass <- ifelse(geweke_pass, "PASS", "FAIL")
      row$Heidel_stat <- heidel_stat
      row$Heidel_hw   <- heidel_hw
      row$Overall     <- overall
    }

    row
  })

  table <- do.call(rbind, rows)
  rownames(table) <- NULL

  result <- list(
    table      = table,
    n_chain    = n_chain,
    model_type = model_type,
    has_coda   = has_coda
  )
  class(result) <- "pdm_convergence"
  result
}


#' Print method for pdm_convergence objects
#'
#' @param x An object of class \code{"pdm_convergence"}.
#' @param digits Integer, significant digits for numeric columns. Default 3.
#' @param ... Additional arguments (currently unused).
#'
#' @return Invisibly returns \code{x}.
#' @export
print.pdm_convergence <- function(x, digits = 3L, ...) {

  cat("\n")
  cat("MCMC Convergence Diagnostics\n")
  cat(strrep("=", 70), "\n\n", sep = "")
  cat("Model:         ", x$model_type, "\n", sep = "")
  cat("Chain samples: ", x$n_chain, "\n\n", sep = "")

  df <- x$table

  # Format numeric columns
  num_cols <- c("ESS", "Efficiency", "Geweke_z")
  for (col in intersect(num_cols, names(df))) {
    df[[col]] <- sprintf(paste0("%.", digits, "f"), df[[col]])
  }
  names(df)[names(df) == "Efficiency"] <- "Efficiency(%)"

  print(df, row.names = FALSE, right = TRUE)

  cat("\n")
  if (!x$has_coda) {
    cat("Install the 'coda' package for Geweke and Heidelberger-Welch diagnostics.\n")
  }
  cat("ESS status: >50% EXCELLENT, >25% GOOD, >10% ACCEPTABLE, else POOR\n")
  cat(strrep("-", 70), "\n\n", sep = "")

  invisible(x)
}
