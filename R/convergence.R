#' Convergence diagnostics for pdm MCMC objects
#'
#' @description Computes MCMC convergence diagnostics for all scalar parameters
#'   of a fitted \code{pdm_mcmc} model. Reports Effective Sample Size (ESS),
#'   sampling efficiency, and — when the \pkg{coda} package is available —
#'   the Geweke z-score and Heidelberger–Welch stationarity and halfwidth tests.
#'
#' @param object An object inheriting from \code{"pdm_mcmc"}, typically the
#'   result of one of the \code{mcmc_*()} fitting functions.
#' @param ... Additional arguments (currently unused).
#'
#' @return An object of class \code{"pdm_convergence"}, which is a list with:
#'   \describe{
#'     \item{\code{table}}{Data frame with one row per scalar parameter and
#'       columns: \code{Parameter}, \code{ESS}, \code{Efficiency_pct},
#'       \code{ESS_status}, and — when \pkg{coda} is available —
#'       \code{Geweke_z}, \code{Geweke_pass}, \code{Heidel_stat},
#'       \code{Heidel_hw}, \code{Overall}.}
#'     \item{\code{n_chain}}{Number of retained MCMC samples.}
#'     \item{\code{model_type}}{Character string, e.g. \code{"locallevel"}.}
#'     \item{\code{has_coda}}{Logical, whether \pkg{coda} diagnostics are
#'       included.}
#'   }
#'
#' @details
#' Effective Sample Size is estimated via the initial monotone sequence
#' estimator based on the sample autocorrelation function (no \pkg{coda}
#' required).
#'
#' When \pkg{coda} is installed, two additional tests are run:
#' \describe{
#'   \item{Geweke (1992)}{Compares means in the first 10\% and last 50\% of
#'     the chain. A z-score with \eqn{|z| < 1.96} indicates convergence.}
#'   \item{Heidelberger–Welch (1983)}{Stationarity test (does the chain come
#'     from a stationary distribution?) and halfwidth test (is the mean
#'     estimated precisely enough?). Both must pass for \code{CONVERGED}.}
#' }
#'
#' Only scalar parameters extracted by \code{get_param_config()} are assessed.
#' Time-varying state matrices (\code{theta_1}, \code{theta_2}, etc.) are
#' excluded; use \code{plot()} to inspect their trajectories.
#'
#' @examples
#' \dontrun{
#' set.seed(1)
#' y <- cumsum(rnorm(200)) + rnorm(200, sd = 0.5)
#' fit <- mcmc_normal_locallevel(y, n_chain = 1000, burnin = 500)
#'
#' conv <- convergence(fit)
#' print(conv)
#' }
#'
#' @seealso \code{\link{mcmc_normal_locallevel}}, \code{\link{summary.normal_locallevel}}
#'
#' @export
convergence <- function(object, ...) {
  UseMethod("convergence")
}


#' @rdname convergence
#' @export
convergence.pdm_mcmc <- function(object, ...) {

  if (!inherits(object, "pdm_mcmc")) {
    stop("'object' must inherit from 'pdm_mcmc'")
  }

  n_chain   <- as.integer(attr(object, "n_chain"))
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
      Parameter    = label,
      ESS          = round(ess, 1),
      Efficiency   = round(efficiency, 1),
      ESS_status   = ess_status,
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
      overall <- if (tests_pass == 2L)        "EXCELLENT"  else
                 if (tests_pass == 1L)        "ACCEPTABLE" else "POOR"

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
