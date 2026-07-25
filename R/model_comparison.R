#' Ensure the loo package is available, or fail informatively
#'
#' @return Invisibly `TRUE`; called for its side effect (an error when `loo`
#'   is not installed).
#' @keywords internal
#' @noRd
require_loo <- function() {
  if (!requireNamespace("loo", quietly = TRUE)) {
    stop("Package 'loo' is required for WAIC / PSIS-LOO model comparison. ",
         "Install it with install.packages(\"loo\").", call. = FALSE)
  }
  invisible(TRUE)
}


#' WAIC, PSIS-LOO and model comparison for pdm fits
#'
#' @description
#' Information criteria and predictive model comparison for fitted \pkg{pdm}
#' models, built on the pointwise conditional log-likelihood (see
#' \code{\link{log_lik}}) and the \pkg{loo} package.
#'
#' \itemize{
#'   \item `waic()` and `loo()` are methods for the \pkg{loo} generics of the
#'     same name; they accept a fitted `pdm_mcmc` object directly and return a
#'     `"waic"` / `"loo"` object (call them as `loo::waic(fit)` /
#'     `loo::loo(fit)`, or plain `waic(fit)` / `loo(fit)` once \pkg{loo} is
#'     attached).
#'   \item `pdm_compare()` computes the chosen criterion for several fitted
#'     models and ranks them with `loo::loo_compare()`.
#' }
#'
#' @details
#' The criteria are computed from the log-likelihood \strong{conditional on the
#' latent states}, so they are in-sample / weak cross-validation measures for
#' time series and tend to be optimistic; see \code{\link{log_lik}} for the
#' conditional-vs-predictive discussion and the leave-future-out alternative.
#'
#' For `loo()` the relative effective sample size passed to `loo::loo()` is
#' computed internally with `loo::relative_eff()` (treating the retained draws
#' as a single chain); supply your own `r_eff` through `...` to override it.
#' High Pareto \eqn{\hat{k}} diagnostics in the returned object flag
#' observations where the PSIS-LOO approximation is unreliable.
#'
#' `pdm_compare()` requires all models to have been fitted to the same
#' observations; `loo::loo_compare()` errors otherwise. The returned table is
#' ordered from best to worst, with the top model as the reference row
#' (`elpd_diff = 0`).
#'
#' @param x A fitted \pkg{pdm} model (an object inheriting class `pdm_mcmc`).
#' @param ... For `waic()` / `loo()`, further arguments passed to the
#'   corresponding \pkg{loo} function (e.g. `cores`, or `r_eff` for `loo()`).
#'   For `pdm_compare()`, two or more fitted models, or a single (optionally
#'   named) list of them.
#' @param criterion Which criterion to rank models by: `"loo"` (PSIS-LOO,
#'   the default) or `"waic"`.
#'
#' @return
#' `waic()` returns a `"waic"` object and `loo()` a `"loo"` object, exactly as
#' produced by the \pkg{loo} package. `pdm_compare()` returns the comparison
#' matrix produced by `loo::loo_compare()` (class `"compare.loo"`), with one row
#' per model labelled by the argument name.
#'
#' @references
#' Burkner, P.-C., Gabry, J., & Vehtari, A. (2020). Approximate leave-future-out
#' cross-validation for Bayesian time series models. \emph{Journal of
#' Statistical Computation and Simulation}, 90(14), 2499-2523.
#'
#' Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model
#' evaluation using leave-one-out cross-validation and WAIC. \emph{Statistics
#' and Computing}, 27(5), 1413-1432.
#'
#' @examples
#' \donttest{
#' ## Simulate a local-level series and fit two competing dynamic orders
#' n <- 200
#' set.seed(123)
#' u1 <- rnorm(n, sd = 1)
#' e  <- rnorm(n, sd = sqrt(1 / 5))
#' y  <- cumsum(c(10, u1))[-1] + e
#'
#' out_level <- mcmc_normal_locallevel(
#'   y,
#'   burnin             = 1000,
#'   thinning           = 10,
#'   n_chain            = 500,
#'   prior_theta01_mean = y[1],
#'   prior_theta01_prec = 1 / var(y),
#'   prior_prec1_shape  = 1e-2,
#'   prior_prec1_rate   = 1e-2,
#'   prior_prec_y_shape = 1e-2,
#'   prior_prec_y_rate  = 1e-2,
#'   verbose            = FALSE,
#'   seed               = 456
#' )
#'
#' out_trend <- mcmc_normal_localtrend(
#'   y,
#'   burnin             = 1000,
#'   thinning           = 20,
#'   n_chain            = 500,
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
#'   verbose            = FALSE,
#'   seed               = 456
#' )
#'
#' if (requireNamespace("loo", quietly = TRUE)) {
#'   ## Single-model criteria
#'   loo::waic(out_level)
#'   loo::loo(out_level)
#'
#'   ## Rank the two models (PSIS-LOO by default; best model on top)
#'   pdm_compare(out_level, out_trend)
#'   pdm_compare(out_level, out_trend, criterion = "waic")
#' }
#' }
#'
#' @seealso
#'   \code{\link{log_lik}} (the underlying pointwise log-likelihood);
#'   \code{\link[loo]{waic}}, \code{\link[loo]{loo}},
#'   \code{\link[loo]{loo_compare}} in the \pkg{loo} package.
#'
#' @name pdm_compare
#' @aliases pdm-model-comparison
NULL


#' @rdname pdm_compare
#' @exportS3Method loo::waic
waic.pdm_mcmc <- function(x, ...) {
  require_loo()
  loo::waic(log_lik(x), ...)
}


#' @rdname pdm_compare
#' @exportS3Method loo::loo
loo.pdm_mcmc <- function(x, ...) {
  require_loo()
  ll   <- log_lik(x)
  dots <- list(...)
  if (is.null(dots$r_eff)) {
    dots$r_eff <- loo::relative_eff(exp(ll), chain_id = rep(1L, nrow(ll)))
  }
  do.call(loo::loo, c(list(ll), dots))
}


#' @rdname pdm_compare
#' @export
pdm_compare <- function(..., criterion = c("loo", "waic")) {
  require_loo()
  criterion <- match.arg(criterion)

  models    <- list(...)
  arg_names <- vapply(substitute(list(...))[-1L], deparse, character(1L))

  # A fit is either a single chain or a multi-chain collection; `loo::loo()` and
  # `loo::waic()` dispatch on both. A `pdm_mcmc_list` is itself a list, so it
  # must be recognised here or the branch below would mistake one multi-chain
  # fit for a list of models and compare its chains against each other.
  is_fit <- function(m) inherits(m, "pdm_mcmc") || inherits(m, "pdm_mcmc_list")

  # Allow a single (optionally named) list of models.
  if (length(models) == 1L && is.list(models[[1L]]) && !is_fit(models[[1L]])) {
    models    <- models[[1L]]
    arg_names <- names(models)
    if (is.null(arg_names)) arg_names <- paste0("model", seq_along(models))
  }

  if (length(models) < 2L) {
    stop("`pdm_compare()` needs at least two fitted models to compare.",
         call. = FALSE)
  }

  is_pdm <- vapply(models, is_fit, logical(1L))
  if (!all(is_pdm)) {
    stop("All objects must be fitted pdm models (class 'pdm_mcmc' or ",
         "'pdm_mcmc_list'). Offending argument(s): ",
         paste(arg_names[!is_pdm], collapse = ", "), call. = FALSE)
  }

  # Honour explicit names (e.g. pdm_compare(level = m1, trend = m2)).
  explicit <- names(models)
  if (!is.null(explicit)) {
    keep <- nzchar(explicit)
    arg_names[keep] <- explicit[keep]
  }

  crit_fun <- if (criterion == "loo") loo::loo else loo::waic
  crits    <- lapply(models, crit_fun)   # dispatches to loo.pdm_mcmc / waic.pdm_mcmc
  names(crits) <- make.unique(arg_names)

  loo::loo_compare(crits)
}


#' Approximate leave-future-out cross-validation (LFO-CV) for pdm fits
#'
#' @description
#' One-step-ahead approximate leave-future-out cross-validation for a fitted
#' \pkg{pdm} model, following Burkner, Gabry & Vehtari (2020). It estimates the
#' expected log predictive density of each \eqn{y_t} given only its past
#' \eqn{y_{1:t-1}},
#' \deqn{\mathrm{elpd}_{\mathrm{LFO}} = \sum_{t = L+1}^{N} \log p(y_t \mid y_{1:t-1}),}
#' the \emph{honest} predictive target for time series, in contrast to the
#' in-sample / state-conditional criteria of \code{\link{pdm_compare}}.
#'
#' @details
#' The estimate is obtained purely by \strong{reweighting the conditional
#' log-likelihood matrix} \code{\link{log_lik}}, with no refitting. For each
#' forecast the full-data posterior \eqn{p(\theta \mid y_{1:N})} is importance-
#' reweighted to the past-only posterior \eqn{p(\theta \mid y_{1:t-1})} by
#' dropping the future observations,
#' \deqn{\log w_t^{(s)} = -\sum_{j = t}^{N} \ell_{s,j},}
#' the weights are stabilised with Pareto-smoothed importance sampling
#' (\code{loo::psis()}), and the one-step predictive density is
#' \eqn{\log p(y_t \mid y_{1:t-1}) \approx \mathrm{logSumExp}(\tilde{w}_t + \ell_{\cdot,t})}
#' with self-normalised log-weights \eqn{\tilde{w}_t}.
#'
#' \strong{Reliability.} Because a single full-data proposal is reused for every
#' forecast (rather than refitting at break points), the importance sampling
#' degrades as more observations are dropped -- i.e. for forecasts far from the
#' series end. The per-forecast Pareto \eqn{\hat{k}} diagnostic flags this:
#' contributions with \eqn{\hat{k} >} `threshold` are unreliable and would
#' require refitting the model on \eqn{y_{1:t-1}} to estimate accurately. The
#' returned object reports how many exceed the threshold, and a warning is
#' issued when any do.
#'
#' For the latent-state models fitted here the full-data posterior of the states
#' is sharply informed by every observation, so \strong{most forecasts are
#' typically flagged}, and only those near the end of the series are reliable
#' without refitting. Read `pdm_lfo()` as much as a diagnostic of \emph{where}
#' honest prediction is cheap as a single ELPD number; a faithful all-forecast
#' estimate would refit at the flagged prefixes and forecast the state forward,
#' which this reweighting-only routine deliberately does not do.
#'
#' The reported standard error, \eqn{\sqrt{n} \, \mathrm{sd}} of the pointwise
#' terms, is approximate: the one-step contributions overlap in the data they
#' condition on and are therefore dependent.
#'
#' @param object A fitted \pkg{pdm} model (an object inheriting class
#'   `pdm_mcmc`).
#' @param L Integer size of the initial history: the first forecast is for
#'   \eqn{y_{L+1}} given \eqn{y_{1:L}}. Must lie in \eqn{[1, N-1]}. Defaults to
#'   `ceiling(0.1 * N)`. Larger `L` yields fewer but individually more
#'   reliable forecasts.
#' @param threshold Pareto \eqn{\hat{k}} value above which a forecast is flagged
#'   as unreliable. Defaults to `0.7`.
#' @param ... Currently unused.
#'
#' @return An object of class `pdm_lfo`, a list with:
#'   \describe{
#'     \item{`estimates`}{Named numeric: `elpd_lfo` and its approximate
#'       standard error `se_elpd_lfo`.}
#'     \item{`pointwise`}{Data frame with one row per forecast: the time index
#'       `t`, the one-step predictive log density `elpd`, and its Pareto
#'       `pareto_k`.}
#'     \item{`L`, `n_pred`, `threshold`, `n_high_k`}{The history length, number
#'       of forecasts, the \eqn{\hat{k}} threshold, and how many forecasts
#'       exceeded it.}
#'   }
#'
#' @references
#' Burkner, P.-C., Gabry, J., & Vehtari, A. (2020). Approximate leave-future-out
#' cross-validation for Bayesian time series models. \emph{Journal of
#' Statistical Computation and Simulation}, 90(14), 2499-2523.
#'
#' Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model
#' evaluation using leave-one-out cross-validation and WAIC. \emph{Statistics
#' and Computing}, 27(5), 1413-1432.
#'
#' @examples
#' \donttest{
#' set.seed(123)
#' n <- 200
#' u1 <- rnorm(n, sd = 1)
#' y  <- cumsum(c(10, u1))[-1] + rnorm(n, sd = sqrt(1 / 5))
#'
#' out <- mcmc_normal_locallevel(
#'   y,
#'   burnin             = 1000,
#'   thinning           = 10,
#'   n_chain            = 500,
#'   prior_theta01_mean = y[1],
#'   prior_theta01_prec = 1 / var(y),
#'   prior_prec1_shape  = 1e-2,
#'   prior_prec1_rate   = 1e-2,
#'   prior_prec_y_shape = 1e-2,
#'   prior_prec_y_rate  = 1e-2,
#'   verbose            = FALSE,
#'   seed               = 456
#' )
#'
#' if (requireNamespace("loo", quietly = TRUE)) {
#'   lfo <- pdm_lfo(out, L = 50)
#'   lfo
#'   ## Forecasts flagged as unreliable (would need a refit on that prefix):
#'   subset(lfo$pointwise, pareto_k > 0.7)
#' }
#' }
#'
#' @seealso
#'   \code{\link{pdm_compare}} and \code{\link{log_lik}}; the in-sample
#'   \code{\link[loo]{loo}} / \code{\link[loo]{waic}} for a non-predictive
#'   alternative.
#'
#' @export
pdm_lfo <- function(object, L = NULL, threshold = 0.7, ...) {
  require_loo()

  ll <- log_lik(object)
  S  <- nrow(ll)
  N  <- ncol(ll)

  if (is.null(L)) L <- ceiling(0.1 * N)
  if (!is.numeric(L) || length(L) != 1L || !is.finite(L) ||
      L < 1 || L != floor(L) || L >= N) {
    stop("`L` must be a single integer in [1, n_obs - 1] (n_obs = ", N, ").",
         call. = FALSE)
  }
  if (!is.numeric(threshold) || length(threshold) != 1L ||
      !is.finite(threshold) || threshold <= 0) {
    stop("`threshold` must be a single positive numeric value.", call. = FALSE)
  }

  # Cumulative log-likelihood along time so that the weights for a given
  # forecast are a single column lookup instead of a fresh rowSums:
  #   sum_{j=t}^{N} ll[, j] = leftcum[, N] - leftcum[, t-1].
  leftcum <- ll
  for (j in 2:N) leftcum[, j] <- leftcum[, j - 1L] + ll[, j]
  total <- leftcum[, N]

  log_sum_exp <- function(x) {
    m <- max(x)
    if (!is.finite(m)) return(m)
    m + log(sum(exp(x - m)))
  }

  idx  <- (L + 1L):N            # forecast y_t from y_{1:t-1}
  elpd <- numeric(length(idx))
  kval <- numeric(length(idx))

  for (m in seq_along(idx)) {
    t <- idx[m]
    # Reweight full-data posterior -> p(theta | y_{1:t-1}) by dropping y_{t:N}.
    # An additive constant across draws is irrelevant (weights are normalised
    # and Pareto k is shift-invariant), so use leftcum[, t-1] - total directly.
    logw <- leftcum[, t - 1L] - total
    ps   <- suppressWarnings(loo::psis(matrix(logw, ncol = 1L), r_eff = NA))
    lw   <- as.vector(loo::weights.importance_sampling(ps, log = TRUE, normalize = TRUE))
    kval[m] <- loo::pareto_k_values(ps)
    elpd[m] <- log_sum_exp(lw + ll[, t])
  }

  elpd_lfo <- sum(elpd)
  se_elpd  <- sqrt(length(elpd)) * stats::sd(elpd)
  n_high   <- sum(kval > threshold)

  if (n_high > 0L) {
    warning(sprintf(
      paste0("%d of %d one-step-ahead forecasts have Pareto k > %.2f and are ",
             "unreliable without refitting on the corresponding prefix."),
      n_high, length(idx), threshold), call. = FALSE)
  }

  structure(
    list(
      estimates = c(elpd_lfo = elpd_lfo, se_elpd_lfo = se_elpd),
      pointwise = data.frame(t = idx, elpd = elpd, pareto_k = kval),
      L         = as.integer(L),
      n_pred    = length(idx),
      threshold = threshold,
      n_high_k  = n_high
    ),
    class = "pdm_lfo"
  )
}


#' Print method for pdm_lfo objects
#'
#' @param x An object of class `pdm_lfo`, from \code{\link{pdm_lfo}}.
#' @param digits Number of digits for the printed estimates. Default `1`.
#' @param ... Currently unused.
#'
#' @return Invisibly returns `x`.
#'
#' @seealso \code{\link{pdm_lfo}}
#' @export
print.pdm_lfo <- function(x, digits = 1, ...) {
  cat("Approximate 1-step-ahead LFO-CV (importance sampling, no refit)\n\n")
  cat(sprintf("  Forecasts:  %d  (t = %d..%d)\n",
              x$n_pred, min(x$pointwise$t), max(x$pointwise$t)))
  cat(sprintf("  History L:  %d\n\n", x$L))
  cat(sprintf("  elpd_lfo    %.*f   (se %.*f)\n",
              digits, x$estimates[["elpd_lfo"]],
              digits, x$estimates[["se_elpd_lfo"]]))
  cat(sprintf("  Pareto k:   %d / %d forecasts > %.2f (unreliable; refit needed)\n",
              x$n_high_k, x$n_pred, x$threshold))
  invisible(x)
}
