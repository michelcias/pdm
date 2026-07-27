#' Rank-normalized split-R-hat and effective sample sizes
#'
#' Internal implementations of the multi-chain diagnostics of Vehtari et al.
#' (2021), written in base R so the package gains no new dependency. Every
#' function here takes an `n_draws x n_chains` numeric matrix, one column per
#' chain, and is called by `mcmc_convergence.pdm_mcmc_list()`.
#'
#' The rank-normalized *split* form is used rather than the original
#' Gelman-Rubin statistic for two reasons: splitting each chain in half exposes
#' a chain that is still drifting (which the classic statistic cannot see, since
#' it only compares chains to each other), and rank normalization makes the
#' statistic robust to the heavy tails of the precision parameters, whose
#' variance-based statistic would otherwise be dominated by a few draws.
#'
#' @name rhat-internal
#' @keywords internal
#' @noRd
NULL


#' Biased sample autocovariance at lags 0, ..., n - 1
#'
#' Computed through the FFT, which is `O(n log n)` against the `O(n^2)` of a
#' direct sum, and normalised by `n` (not `n - lag`) as the effective sample
#' size estimator of Vehtari et al. (2021) expects.
#'
#' @param y Numeric vector.
#'
#' @return Numeric vector of length `length(y)`; element `k + 1` is the lag-`k`
#'   autocovariance.
#'
#' @keywords internal
#' @noRd
autocovariance <- function(y) {
  n  <- length(y)
  yc <- y - mean(y)

  # Zero-pad to a power of two of at least 2n so the circular correlation the
  # FFT computes coincides with the linear one.
  n_pad <- nextn(2L * n, 2L)
  yc    <- c(yc, rep.int(0, n_pad - n))

  f  <- fft(yc)
  ac <- Re(fft(f * Conj(f), inverse = TRUE)) / (n_pad * n)
  ac[seq_len(n)]
}


#' Split every chain in half
#'
#' Turns an `N x M` matrix into an `(N %/% 2) x (2M)` one. When `N` is odd the
#' middle draw is dropped, so both halves have the same length.
#'
#' @param x Numeric matrix, draws in rows and chains in columns.
#'
#' @return Numeric matrix with twice as many columns and half as many rows.
#'
#' @keywords internal
#' @noRd
split_chains <- function(x) {
  n_draw <- nrow(x)
  half   <- n_draw %/% 2L
  cbind(x[seq_len(half), , drop = FALSE],
        x[(n_draw - half + 1L):n_draw, , drop = FALSE])
}


#' Rank-normalise a set of draws
#'
#' Ranks all draws jointly (across every chain), then maps the ranks through the
#' inverse normal CDF with the Blom offset. The result is comparable across
#' parameters regardless of their marginal shape.
#'
#' @param x Numeric matrix, draws in rows and chains in columns.
#'
#' @return Numeric matrix of the same dimensions, on the z scale.
#'
#' @keywords internal
#' @noRd
z_scale <- function(x) {
  # rank() flattens the matrix in column-major order, which is exactly the
  # layout matrix() restores below, so chains stay in their own columns.
  r <- rank(x, ties.method = "average")
  z <- qnorm((r - 3 / 8) / (length(r) - 1 / 4))
  matrix(z, nrow = nrow(x), ncol = ncol(x))
}


#' Gelman-Rubin statistic on a set of chains, without transformation
#'
#' The plain between/within variance ratio. It is applied by
#' `rhat_rank_normalized()` to already split and transformed draws; it is not
#' meant to be used directly.
#'
#' @param x Numeric matrix, draws in rows and chains in columns.
#'
#' @return Single numeric, or `NA_real_` when the within-chain variance vanishes.
#'
#' @keywords internal
#' @noRd
rhat_basic <- function(x) {
  n_draw <- nrow(x)
  n_ch   <- ncol(x)

  chain_mean <- colMeans(x)
  W          <- mean(apply(x, 2L, var))
  if (!is.finite(W) || W <= 0) return(NA_real_)

  # var(chain_mean) is B/N, with B the usual between-chain sum of squares.
  var_hat <- (n_draw - 1) / n_draw * W + if (n_ch > 1L) var(chain_mean) else 0
  sqrt(var_hat / W)
}


#' Rank-normalized split-R-hat
#'
#' The maximum of the statistic computed on the rank-normalized draws (which
#' detects disagreement in location) and on the rank-normalized absolute
#' deviations from the median (the *folded* version, which detects disagreement
#' in scale). Values below roughly 1.01 are consistent with convergence.
#'
#' @param x Numeric matrix, draws in rows and chains in columns.
#'
#' @return Single numeric, or `NA_real_` for a constant, non-finite or
#'   single-chain input.
#'
#' @keywords internal
#' @noRd
rhat_rank_normalized <- function(x) {
  if (!is.matrix(x) || ncol(x) < 2L)     return(NA_real_)
  if (!all(is.finite(x)))               return(NA_real_)
  if (max(x) - min(x) < .Machine$double.eps) return(NA_real_)

  splits <- split_chains(x)
  bulk   <- rhat_basic(z_scale(splits))
  folded <- rhat_basic(z_scale(abs(splits - median(splits))))

  if (is.na(bulk) && is.na(folded)) return(NA_real_)
  max(c(bulk, folded), na.rm = TRUE)
}


#' Effective sample size of the mean over a set of chains
#'
#' Uses Geyer's initial positive and initial monotone sequences on the
#' autocorrelations pooled across chains, as in Vehtari et al. (2021).
#'
#' @param x Numeric matrix, draws in rows and chains in columns.
#'
#' @return Single numeric, or `NA_real_` when the chains are too short or the
#'   variance vanishes.
#'
#' @keywords internal
#' @noRd
ess_mean <- function(x) {
  n_draw <- nrow(x)
  n_ch   <- ncol(x)
  if (n_draw < 4L || !all(is.finite(x))) return(NA_real_)

  acov <- vapply(seq_len(n_ch), function(m) autocovariance(x[, m]),
                 numeric(n_draw))
  dim(acov) <- c(n_draw, n_ch)

  chain_mean <- colMeans(x)
  mean_var   <- mean(acov[1L, ]) * n_draw / (n_draw - 1)
  if (!is.finite(mean_var) || mean_var <= 0) return(NA_real_)

  var_plus <- mean_var * (n_draw - 1) / n_draw
  if (n_ch > 1L) var_plus <- var_plus + var(chain_mean)

  # Geyer's initial positive sequence: accumulate autocorrelations in
  # consecutive pairs and stop as soon as a pair sums to a negative value.
  rho_hat_t     <- rep.int(0, n_draw)
  t             <- 0L
  rho_hat_even  <- 1
  rho_hat_t[1L] <- rho_hat_even
  rho_hat_odd   <- 1 - (mean_var - mean(acov[2L, ])) / var_plus
  rho_hat_t[2L] <- rho_hat_odd

  while (t < n_draw - 5L && !is.nan(rho_hat_even + rho_hat_odd) &&
         (rho_hat_even + rho_hat_odd > 0)) {
    t            <- t + 2L
    rho_hat_even <- 1 - (mean_var - mean(acov[t + 1L, ])) / var_plus
    rho_hat_odd  <- 1 - (mean_var - mean(acov[t + 2L, ])) / var_plus
    if ((rho_hat_even + rho_hat_odd) >= 0) {
      rho_hat_t[t + 1L] <- rho_hat_even
      rho_hat_t[t + 2L] <- rho_hat_odd
    }
  }
  max_t <- t
  if (rho_hat_even > 0) rho_hat_t[max_t + 1L] <- rho_hat_even

  # Geyer's initial monotone sequence: enforce a non-increasing pair sequence.
  t <- 0L
  while (t <= max_t - 4L) {
    t <- t + 2L
    if (rho_hat_t[t + 1L] + rho_hat_t[t + 2L] >
        rho_hat_t[t - 1L] + rho_hat_t[t]) {
      rho_hat_t[t + 1L] <- (rho_hat_t[t - 1L] + rho_hat_t[t]) / 2
      rho_hat_t[t + 2L] <- rho_hat_t[t + 1L]
    }
  }

  n_total <- n_ch * n_draw
  tau_hat <- -1 + 2 * sum(rho_hat_t[seq_len(max_t)]) + rho_hat_t[max_t + 1L]
  # Floor on tau keeps the estimate finite for a near-independent chain.
  tau_hat <- max(tau_hat, 1 / log10(n_total))
  n_total / tau_hat
}


#' Bulk effective sample size
#'
#' Effective sample size of the rank-normalized split chains. It measures how
#' well the bulk of the distribution -- and therefore the posterior mean and
#' median -- is resolved.
#'
#' @param x Numeric matrix, draws in rows and chains in columns.
#'
#' @return Single numeric, or `NA_real_`.
#'
#' @keywords internal
#' @noRd
ess_bulk <- function(x) {
  if (!all(is.finite(x)))                    return(NA_real_)
  if (max(x) - min(x) < .Machine$double.eps) return(NA_real_)
  ess_mean(z_scale(split_chains(x)))
}


#' Tail effective sample size
#'
#' The smaller of the effective sample sizes of the indicators that a draw falls
#' below the 5% and below the 95% quantile. It measures how well the extremes
#' are resolved, which is what interval estimates depend on; a chain can have a
#' healthy bulk ESS and still be unable to place its own credible interval.
#'
#' @param x Numeric matrix, draws in rows and chains in columns.
#'
#' @return Single numeric, or `NA_real_`.
#'
#' @keywords internal
#' @noRd
ess_tail <- function(x) {
  if (!all(is.finite(x)))                    return(NA_real_)
  if (max(x) - min(x) < .Machine$double.eps) return(NA_real_)

  q <- quantile(x, probs = c(0.05, 0.95), names = FALSE)
  lower <- ess_mean(split_chains((x <= q[1L]) * 1))
  upper <- ess_mean(split_chains((x <= q[2L]) * 1))

  if (is.na(lower) && is.na(upper)) return(NA_real_)
  min(c(lower, upper), na.rm = TRUE)
}


#' Resolve a latent-state time-point specification into fractions
#'
#' Accepts either a count or an explicit vector of fractions, following the
#' convention `hist()` uses for `breaks`: a single whole number is *how many*
#' evenly spaced points to screen, anything else is the points themselves.
#' A count is expanded to `seq(0.05, 0.95, length.out = n)`, so the grid spans
#' the series without touching either endpoint, where a state is pinned by its
#' own prior rather than by the data.
#'
#' Both `mcmc_convergence.pdm_mcmc_list()` and `state_rhats()` resolve through
#' this, which is what keeps the diagnostic table and the automatic screen
#' behind `summary()` looking at the same points. They disagreed once — three
#' points against twenty — and the same fit came back at 1.037 from one and
#' 1.146 from the other.
#'
#' @param x A single whole number >= 1 (a count), a numeric vector of fractions
#'   in (0, 1), or `NULL`.
#' @param arg Character, the caller's argument name, used in error messages.
#'
#' @return Sorted unique fractions in (0, 1), or `NULL` if `x` was `NULL`.
#'
#' @keywords internal
#' @noRd
resolve_timepoints <- function(x, arg = "theta_timepoints") {
  if (is.null(x)) return(NULL)

  if (!is.numeric(x) || !length(x) || anyNA(x)) {
    stop(sprintf("'%s' must be a count, a numeric vector of fractions in (0, 1), or NULL",
                 arg), call. = FALSE)
  }

  # A count: one whole number >= 1. Fractions live strictly inside (0, 1), so
  # the two forms cannot be confused -- 1 is already outside the open interval.
  if (length(x) == 1L && x >= 1 && x == as.integer(x)) {
    n <- as.integer(x)
    return(if (n == 1L) 0.5 else seq(0.05, 0.95, length.out = n))
  }

  if (any(x <= 0) || any(x >= 1)) {
    stop(sprintf("'%s' must be a count (a single whole number >= 1) or a numeric vector with values in (0, 1), or NULL",
                 arg), call. = FALSE)
  }

  sort(unique(x))
}
