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
  # In double precision, not integer. `nextn()` returns an integer and `n` is
  # one, so `n_pad * n` is an integer product; it reaches 2^31 at n = 32768,
  # where `n_pad` is already 65536. R answers an integer overflow with NA and a
  # warning rather than an error, so the NA propagates all the way out as a
  # missing effective sample size, with nothing to say why. The callers split
  # the chains before they get here, so this reaches a user at 65536 draws per
  # chain.
  ac <- Re(fft(f * Conj(f), inverse = TRUE)) / (as.numeric(n_pad) * n)
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
  # Blom's plotting position is (r - 3/8) / (S + 1/4). The `+` matters: it is
  # what makes the map symmetric about the median, so ranks r and S + 1 - r
  # give probabilities summing to 1. A minus sign there shifts every z-score
  # upward and perturbs R-hat in the fourth decimal.
  z <- qnorm((r - 3 / 8) / (length(r) + 1 / 4))
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


#' Convergence statistics for a matrix of draws
#'
#' The three diagnostics of Vehtari et al. (2021), each taking an
#' `n_draws x n_chains` matrix and returning one number. `mcmc_convergence()`
#' computes its table from exactly these, so a value obtained here and one read
#' out of that table are the same number by construction.
#'
#' They are exported for quantities `mcmc_convergence()` does not itself
#' tabulate — a derived series, a transformed parameter, one time point of a
#' latent state at a density the table's own screen does not reach. Assembling
#' the matrix is the caller's job; the chains must be columns.
#'
#' \describe{
#'   \item{`rhat_rank_normalized()`}{The maximum of the statistic on the
#'     rank-normalized draws (which detects disagreement in location) and on the
#'     rank-normalized absolute deviations from the median (the *folded*
#'     version, which detects disagreement in scale). Values below roughly 1.01
#'     are consistent with convergence.}
#'   \item{`ess_bulk()`}{Effective sample size of the rank-normalized split
#'     chains: how well the bulk of the distribution, and therefore the
#'     posterior mean and median, is resolved.}
#'   \item{`ess_tail()`}{The smaller of the effective sample sizes of the
#'     indicators that a draw falls below the 5\% and below the 95\% quantile:
#'     how well the extremes are resolved, which is what an interval estimate
#'     depends on. A chain can have a healthy bulk ESS and still be unable to
#'     place its own credible interval.}
#' }
#'
#' @param x Numeric matrix, draws in rows and chains in columns. Chains are
#'   split in half internally, so the matrix should hold the draws as they were
#'   retained, not already halved.
#'
#' @return A single numeric, or `NA_real_`: for a constant or non-finite input
#'   in every case, additionally for a single-chain input in
#'   `rhat_rank_normalized()`, and additionally for a degenerate tail in
#'   `ess_tail()` (an indicator that is constant carries no information, and the
#'   surviving tail is not reported in its place).
#'
#' @references
#' Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021).
#' Rank-normalization, folding, and localization: An improved \eqn{\hat{R}} for
#' assessing convergence of MCMC (with discussion).
#' \emph{Bayesian Analysis}, \strong{16}(2), 667--718.
#' \doi{10.1214/20-BA1221}
#'
#' @seealso \code{\link{mcmc_convergence}}, which reports these for the
#'   parameters of a fit.
#'
#' @examples
#' set.seed(1)
#' # Four independent chains from the same distribution.
#' x <- matrix(rnorm(4000), nrow = 1000, ncol = 4)
#' rhat_rank_normalized(x)   # near 1
#' ess_bulk(x)               # near the 4000 draws
#' ess_tail(x)
#'
#' # One chain shifted: R-hat rises, the effective sample size falls.
#' x[, 1] <- x[, 1] + 1
#' rhat_rank_normalized(x)
#' ess_bulk(x)
#'
#' @name convergence-statistics
NULL


#' @rdname convergence-statistics
#' @export
rhat_rank_normalized <- function(x) {
  if (!is.matrix(x) || ncol(x) < 2L) return(NA_real_)
  rhat_rank_core(x)
}


#' The rank-normalized split-R-hat computation, without the chain-count guard
#'
#' Extracted so that `rhat_rank_normalized()` (two or more chains) and
#' `rhat_single()` (one chain, split into halves) are the *same*
#' implementation and cannot drift apart. The two differ only in what they
#' accept, which is why the chain-count guard stays with the callers.
#'
#' @param x Numeric matrix, draws in rows and chains in columns.
#'
#' @return Single numeric, or `NA_real_` for constant or non-finite input.
#'
#' @keywords internal
#' @noRd
rhat_rank_core <- function(x) {
  if (!all(is.finite(x)))                    return(NA_real_)
  if (max(x) - min(x) < .Machine$double.eps) return(NA_real_)

  # Fold around the median of *all* the draws, then split -- not the other way
  # round. For an even number of draws the two orders agree, but an odd one
  # loses its middle draw to `split_chains()`, and centring on the median of
  # what survives moves the fold. The definition centres on the full sample.
  bulk   <- rhat_basic(z_scale(split_chains(x)))
  folded <- rhat_basic(z_scale(split_chains(abs(x - median(x)))))

  if (is.na(bulk) && is.na(folded)) return(NA_real_)
  max(c(bulk, folded), na.rm = TRUE)
}


#' Rank-normalized split-R-hat for a single chain
#'
#' The statistic of Vehtari et al. (2021) applied to one chain, whose two
#' halves supply the sequences that `split_chains()` would otherwise take from
#' separate runs. This is what Stan and \pkg{posterior} report as `rhat` for a
#' one-chain run, and it reproduces `posterior::rhat()` exactly.
#'
#' Note the single split. Passing the halves in as two columns would split them
#' again into quarters and give a different, wrong number -- the core already
#' splits whatever it is handed.
#'
#' **It answers a narrower question than the multi-chain statistic.** Two
#' halves of one run share their whole history, so this asks whether the run
#' stopped drifting, not whether independent runs found the same distribution.
#' A chain that never left one mode scores near 1. See the `@details` of
#' `mcmc_convergence()`.
#'
#' @param x Numeric vector of draws from a single chain.
#'
#' @return Single numeric, or `NA_real_` when the chain is constant,
#'   non-finite, or shorter than four draws -- each half needs two for a
#'   within-half variance.
#'
#' @keywords internal
#' @noRd
rhat_single <- function(x) {
  if (!is.numeric(x) || length(x) < 4L) return(NA_real_)
  rhat_rank_core(matrix(x, ncol = 1L))
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

  # Double precision here too, for the reason spelled out in `autocovariance()`.
  # This product is the returned sample size itself, so an overflow would not
  # merely lose the answer, it would replace it with NA at the last step.
  n_total <- as.numeric(n_ch) * n_draw
  tau_hat <- -1 + 2 * sum(rho_hat_t[seq_len(max_t)]) + rho_hat_t[max_t + 1L]
  # Floor on tau keeps the estimate finite for a near-independent chain.
  tau_hat <- max(tau_hat, 1 / log10(n_total))
  n_total / tau_hat
}


#' @rdname convergence-statistics
#' @export
ess_bulk <- function(x) {
  if (!all(is.finite(x)))                    return(NA_real_)
  if (max(x) - min(x) < .Machine$double.eps) return(NA_real_)
  ess_mean(z_scale(split_chains(x)))
}


#' @rdname convergence-statistics
#' @export
ess_tail <- function(x) {
  if (!all(is.finite(x)))                    return(NA_real_)
  if (max(x) - min(x) < .Machine$double.eps) return(NA_real_)

  q <- quantile(x, probs = c(0.05, 0.95), names = FALSE)
  lower <- ess_mean(split_chains((x <= q[1L]) * 1))
  upper <- ess_mean(split_chains((x <= q[2L]) * 1))

  # A plain min(), so one degenerate tail makes the whole statistic NA. An
  # indicator goes constant when 5% or more of the draws tie at the maximum,
  # which puts the 95% quantile on the maximum itself. Falling back to the
  # surviving tail would report a healthy ESS for a quantity whose tail
  # carries no information at all.
  min(lower, upper)
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


#' The two earlier R-hat statistics, for reconciliation
#'
#' Three statistics circulate under the name "R-hat", and this package, other
#' implementations and the applied literature do not all report the same one.
#' `mcmc_convergence()` reports Vehtari et al. (2021); these are the two it
#' supersedes, computed from the same building blocks so that a number from
#' elsewhere can be matched against ours instead of guessed at:
#'
#' \describe{
#'   \item{classic}{`rhat_basic(x)` -- Gelman & Rubin (1992). No split, no
#'     rank normalization: between-chain against within-chain variance on the
#'     draws as they are.}
#'   \item{split}{`rhat_basic(split_chains(x))` -- the BDA3 (2013) form. Each
#'     chain is halved and the halves treated as separate chains, which exposes
#'     a chain still drifting.}
#' }
#'
#' Neither is invariant to a monotone transformation, which is why a diagnostic
#' computed on `alpha` by either route is a genuinely different number from the
#' one on the state, where the rank-based statistic gives the same value for
#' both.
#'
#' The guards match `rhat_rank_normalized()` exactly, so a case that yields
#' `NA` there yields `NA` here rather than `NaN` from an unguarded variance.
#'
#' @param x Numeric matrix, draws in rows and chains in columns.
#'
#' @return Named numeric vector of length two, `c(split = , classic = )`.
#'
#' @keywords internal
#' @noRd
rhat_variants <- function(x) {
  na <- c(split = NA_real_, classic = NA_real_)
  if (!is.matrix(x) || ncol(x) < 2L)         return(na)
  if (!all(is.finite(x)))                    return(na)
  if (max(x) - min(x) < .Machine$double.eps) return(na)

  c(split   = rhat_basic(split_chains(x)),
    classic = rhat_basic(x))
}


#' Split-R-hat for a single chain
#'
#' The Gelman-Rubin ratio applied to the two halves of one chain. It asks the
#' question Geweke asks -- does the start of the run agree with the end? -- and
#' needs no second chain, because `split_chains()` supplies the second
#' sequence. That is what makes an R-hat available to the single-chain method
#' at all.
#'
#' The plain statistic -- split, but neither rank-normalized nor folded -- so
#' `Rhat_split` means the same thing here as in the column of that name in
#' `mcmc_convergence.pdm_mcmc_list()`. It is a reconciliation column in both,
#' never the reported one: `rhat_single()` holds that role, because the 1.01
#' threshold in common use belongs to the rank-normalized statistic and this
#' one is the more forgiving. On a local-level precision the two read 1.12 and
#' 1.28 on the same draws.
#'
#' @param x Numeric vector of draws from a single chain.
#'
#' @return Single numeric, or `NA_real_` when the chain is constant,
#'   non-finite, or shorter than four draws -- each half needs two for a
#'   within-half variance.
#'
#' @keywords internal
#' @noRd
rhat_split_single <- function(x) {
  # The guards mirror `rhat_variants()`, minus its `ncol(x) < 2` rejection:
  # one chain is the whole point here. The length floor replaces it, since
  # `var()` on a one-draw half is NA and would surface as NaN, not NA.
  if (!is.numeric(x) || length(x) < 4L)      return(NA_real_)
  if (!all(is.finite(x)))                    return(NA_real_)
  if (max(x) - min(x) < .Machine$double.eps) return(NA_real_)

  rhat_basic(split_chains(matrix(x, ncol = 1L)))
}
