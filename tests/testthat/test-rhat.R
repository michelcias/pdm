library(testthat)

# Unit tests for the multi-chain diagnostic maths in R/rhat.R: rank-normalized
# split-R-hat and the bulk/tail effective sample sizes. These operate on plain
# `n_draws x n_chains` matrices and do not need a fitted model.

# Four independent chains from the same distribution.
iid_chains <- function(n = 1000, m = 4, seed = 1) {
  set.seed(seed)
  matrix(rnorm(n * m), nrow = n, ncol = m)
}


test_that("autocovariance() agrees with stats::acf", {
  set.seed(11)
  y <- as.numeric(arima.sim(list(ar = 0.7), 200))

  expect_equal(
    autocovariance(y),
    drop(acf(y, lag.max = length(y) - 1L, type = "covariance", plot = FALSE)$acf),
    tolerance = 1e-10
  )
})


test_that("split_chains() halves the draws and doubles the chains", {
  x <- iid_chains(n = 100, m = 3)
  s <- split_chains(x)

  expect_equal(dim(s), c(50L, 6L))
  # First half of chain 1 keeps its order.
  expect_equal(s[, 1], x[1:50, 1])
  # Second half of chain 1 lands in column 4 (m + 1).
  expect_equal(s[, 4], x[51:100, 1])
})


test_that("split_chains() drops the middle draw for an odd number of draws", {
  x <- iid_chains(n = 101, m = 2)
  s <- split_chains(x)

  expect_equal(dim(s), c(50L, 4L))
  expect_equal(s[, 1], x[1:50, 1])
  expect_equal(s[, 3], x[52:101, 1])
})


test_that("z_scale() keeps chains in their own columns", {
  x <- iid_chains(n = 50, m = 3)
  z <- z_scale(x)

  expect_equal(dim(z), dim(x))
  # Rank normalisation is monotone, so the within-column ordering is preserved.
  expect_equal(order(z[, 2]), order(x[, 2]))
  # Ranking is joint across chains, so the overall extremes map to the extremes.
  expect_equal(which.max(z), which.max(x))
})


test_that("z_scale() uses Blom's offset, (r - 3/8) / (S + 1/4)", {
  x <- matrix(c(4, 1, 3, 2), nrow = 2, ncol = 2)
  r <- rank(x)

  expect_equal(as.vector(z_scale(x)),
               qnorm((r - 3 / 8) / (length(r) + 1 / 4)))

  # The sign of that quarter is the whole point, and a wrong one is invisible
  # by inspection, so pin the property it buys: Blom's position is symmetric
  # about the median, ranks r and S + 1 - r mapping to probabilities that sum
  # to 1. With S - 1/4 the z-scores all shift upward and this fails.
  z <- sort(z_scale(iid_chains(n = 16, m = 4, seed = 5)))
  expect_equal(z, -rev(z))
})


test_that("R-hat is close to 1 for chains from the same distribution", {
  expect_lt(rhat_rank_normalized(iid_chains(seed = 21)), 1.01)
  expect_gt(rhat_rank_normalized(iid_chains(seed = 21)), 0.99)
})


test_that("R-hat detects a chain with a shifted location", {
  x <- iid_chains(seed = 22)
  x[, 1] <- x[, 1] + 1

  expect_gt(rhat_rank_normalized(x), 1.05)
})


test_that("R-hat detects a chain with a different scale", {
  # A pure scale difference leaves the chain means alone, so this is the case
  # the folded component of the statistic exists to catch.
  x <- iid_chains(seed = 23)
  x[, 1] <- x[, 1] * 3

  expect_gt(rhat_rank_normalized(x), 1.05)
})


test_that("R-hat detects a chain that is still drifting within itself", {
  # A common starting point can hide non-convergence from the classic
  # statistic; splitting each chain in half is what exposes the trend.
  set.seed(24)
  x <- iid_chains(seed = 24)
  x[, 1] <- x[, 1] + seq(0, 2, length.out = nrow(x))

  expect_gt(rhat_rank_normalized(x), 1.05)
})


test_that("R-hat returns NA for degenerate input", {
  expect_true(is.na(rhat_rank_normalized(matrix(1, nrow = 100, ncol = 4))))
  expect_true(is.na(rhat_rank_normalized(iid_chains(m = 1))))
  x <- iid_chains(n = 50, m = 2)
  x[3, 1] <- NA_real_
  expect_true(is.na(rhat_rank_normalized(x)))
  x[3, 1] <- Inf
  expect_true(is.na(rhat_rank_normalized(x)))
})


test_that("the folded statistic centres on the median of all the draws", {
  # `split_chains()` discards the middle draw when the count is odd, so folding
  # after the split centres on the median of what survived rather than on the
  # median of the sample. Both orders are computed here and shown to disagree,
  # then R-hat is checked to follow the one the definition asks for.
  x <- iid_chains(n = 301, m = 4, seed = 54)

  splits      <- split_chains(x)
  bulk        <- rhat_basic(z_scale(splits))
  fold_first  <- rhat_basic(z_scale(split_chains(abs(x - median(x)))))
  split_first <- rhat_basic(z_scale(abs(splits - median(splits))))

  expect_false(isTRUE(all.equal(fold_first, split_first)))
  expect_equal(rhat_rank_normalized(x), max(bulk, fold_first))

  # For an even count nothing is discarded and the question does not arise.
  y <- iid_chains(n = 300, m = 4, seed = 54)
  expect_equal(rhat_basic(z_scale(split_chains(abs(y - median(y))))),
               rhat_basic(z_scale(abs(split_chains(y) -
                                        median(split_chains(y))))))
})


test_that("ESS approaches the number of draws for independent chains", {
  x <- iid_chains(n = 1000, m = 4, seed = 31)

  # 4000 draws; the estimator is slightly conservative but should stay close.
  expect_gt(ess_bulk(x), 3000)
  expect_lt(ess_bulk(x), 4400)
})


test_that("ESS falls well below the number of draws for autocorrelated chains", {
  set.seed(32)
  x <- matrix(as.numeric(replicate(4, arima.sim(list(ar = 0.9), 1000))),
              nrow = 1000, ncol = 4)

  # For AR(1) the asymptotic ESS is N * (1 - rho) / (1 + rho) ~ 210 here.
  expect_lt(ess_bulk(x), 600)
  expect_gt(ess_bulk(x), 100)
})


test_that("ESS returns NA for degenerate input", {
  expect_true(is.na(ess_bulk(matrix(1, nrow = 100, ncol = 4))))
  expect_true(is.na(ess_tail(matrix(1, nrow = 100, ncol = 4))))

  x <- iid_chains(n = 50, m = 2)
  x[3, 1] <- NA_real_
  expect_true(is.na(ess_bulk(x)))
  expect_true(is.na(ess_tail(x)))
})


test_that("ess_tail() is NA when one tail indicator is degenerate", {
  # Tie a tenth of the draws at the maximum. The 95% quantile then sits on the
  # maximum itself, `x <= q95` is constant, and that tail carries no
  # information. The 5% side is still perfectly well defined -- and must not be
  # reported in its place, which is what a min(na.rm = TRUE) used to do.
  x <- iid_chains(n = 200, m = 4, seed = 41)
  x[1:20, ] <- max(x) + 1

  expect_true(is.na(ess_tail(x)))
  expect_false(is.na(ess_bulk(x)))
})


test_that("ess_tail() survives ties at the minimum", {
  # The mirror case is not degenerate and must not be swept up by the guard
  # above: with 60% of the draws at the minimum the 5% quantile is the minimum,
  # but `x <= q05` still separates that block from the rest. This is the shape a
  # saturated link parameter takes, so it has to keep returning a number.
  y <- iid_chains(n = 200, m = 4, seed = 41)
  y[1:120, ] <- min(y) - 1

  expect_false(is.na(ess_tail(y)))
  expect_gt(ess_tail(y), 0)
})

test_that("resolve_timepoints() accepts a count or an explicit grid", {
  # A count expands to evenly spaced fractions spanning (0, 1) without the ends.
  expect_equal(resolve_timepoints(20L), seq(0.05, 0.95, length.out = 20L))
  expect_length(resolve_timepoints(50), 50L)
  expect_equal(resolve_timepoints(3), c(0.05, 0.5, 0.95))

  # A count of 1 is the midpoint, not a degenerate seq() of length 1 at 0.05.
  expect_equal(resolve_timepoints(1), 0.5)

  # An explicit grid passes through, sorted and de-duplicated.
  expect_equal(resolve_timepoints(c(0.9, 0.1, 0.9)), c(0.1, 0.9))

  # NULL means "no latent states" and must survive.
  expect_null(resolve_timepoints(NULL))

  # The two forms cannot collide: fractions are strictly inside (0, 1), a count
  # is at least 1, so 1 is unambiguously a count.
  expect_equal(resolve_timepoints(0.5), 0.5)

  # Rejected inputs.
  expect_error(resolve_timepoints(0), "count")
  expect_error(resolve_timepoints(-1), "count")
  expect_error(resolve_timepoints(c(0.5, 1.5)), "count")
  expect_error(resolve_timepoints(2.5), "count")     # non-integer > 1
  expect_error(resolve_timepoints("a"), "count")
  expect_error(resolve_timepoints(NA_real_), "count")
  expect_error(resolve_timepoints(numeric(0)), "count")

  # The error names the caller's own argument.
  expect_error(resolve_timepoints(0, arg = "timepoints"), "'timepoints'")
})


# --- Differential test against the reference implementation ----------------
# The statistics above are ours, but the definitions are not: they are Vehtari
# et al. (2021), and `posterior` is the authors' own implementation. Unit tests
# that only assert shape and sign let a wrong constant through -- both defects
# this file now guards against were of exactly that kind, one a sign in a
# denominator and one an `na.rm`. Agreement to machine precision is the only
# assertion that would have caught them. `posterior` is in Suggests, so the
# block skips where it is absent.

test_that("R-hat and both ESS reproduce posterior::", {
  skip_if_not_installed("posterior")

  set.seed(20260728)
  scenarios <- list(
    iid          = iid_chains(n = 500, m = 4, seed = 51),
    autocorr     = matrix(as.numeric(replicate(4, arima.sim(list(ar = 0.9), 500))),
                          nrow = 500, ncol = 4),
    shifted      = iid_chains(n = 400, m = 4, seed = 52) +
                     rep(c(0, 0.4, -0.3, 0.1), each = 400),
    two_chains   = iid_chains(n = 600, m = 2, seed = 53),
    odd_draws    = iid_chains(n = 301, m = 4, seed = 54),
    heavy_tailed = matrix(rt(4 * 500, df = 2), nrow = 500, ncol = 4),
    # A link-transformed state: bounded in (0, 1) and skewed, which is the
    # shape the mixture models actually feed these functions.
    on_unit      = plogis(iid_chains(n = 500, m = 4, seed = 55) - 2)
  )

  for (nm in names(scenarios)) {
    x <- scenarios[[nm]]
    expect_equal(rhat_rank_normalized(x), posterior::rhat(x),
                 tolerance = 1e-10, info = nm)
    expect_equal(ess_bulk(x), posterior::ess_bulk(x),
                 tolerance = 1e-10, info = nm)
    expect_equal(ess_tail(x), posterior::ess_tail(x),
                 tolerance = 1e-10, info = nm)
  }
})


test_that("rhat_split_single() is the split statistic on one chain", {
  set.seed(60)
  x <- as.numeric(arima.sim(list(ar = 0.9), 500))

  # It is exactly what the multi-chain `Rhat_split` column computes, with the
  # two halves standing in for the two chains. Built independently here rather
  # than by calling the same helper, which would prove nothing.
  half <- 500L %/% 2L
  manual <- rhat_basic(cbind(x[seq_len(half)], x[(500L - half + 1L):500L]))
  expect_equal(rhat_split_single(x), manual)

  # A chain still drifting within itself is the case it exists to catch: the
  # first half and the second sit at different levels. Contrasted against iid
  # draws, not against the AR(1) above -- at rho = 0.9 a 500-draw run wanders
  # enough on its own to score 1.07, which is the statistic working, not a
  # baseline to assert quietness on.
  set.seed(60)
  quiet    <- rnorm(500)
  drifting <- rnorm(500) + seq(0, 5, length.out = 500)
  expect_lt(rhat_split_single(quiet), 1.02)
  expect_gt(rhat_split_single(drifting), 1.5)
})


test_that("rhat_split_single() returns NA rather than NaN for degenerate input", {
  # The guards mirror rhat_variants(), which is what keeps a degenerate case
  # from surfacing as NaN out of an unguarded variance.
  expect_true(is.na(rhat_split_single(rep(1, 100))))          # constant
  expect_true(is.na(rhat_split_single(c(1, 2, NA, 4, 5))))    # non-finite
  expect_true(is.na(rhat_split_single(c(1, 2, Inf, 4, 5))))
  expect_true(is.na(rhat_split_single(c(1, 2, 3))))           # too short
  expect_false(is.na(rhat_split_single(rnorm(4))))            # the floor itself
})


test_that("rhat_single() is the Vehtari statistic on one chain", {
  set.seed(62)
  x <- rnorm(500)

  # One split, not two. Handing the halves in as two columns would make the
  # core split them again into quarters -- a different number, and the mistake
  # is invisible because it still looks like an R-hat.
  expect_equal(rhat_single(x), rhat_rank_core(matrix(x, ncol = 1L)))
  expect_false(isTRUE(all.equal(rhat_single(x),
                                rhat_rank_normalized(cbind(x[1:250], x[251:500])))))

  # It separates from the plain split statistic where rank normalization is
  # supposed to matter: a heavy tail.
  set.seed(63)
  heavy <- rt(500, df = 2)
  expect_false(isTRUE(all.equal(rhat_single(heavy), rhat_split_single(heavy))))

  expect_lt(rhat_single(rnorm(500)), 1.05)
  expect_gt(rhat_single(rnorm(500) + seq(0, 5, length.out = 500)), 1.2)
})


test_that("rhat_single() returns NA rather than NaN for degenerate input", {
  expect_true(is.na(rhat_single(rep(1, 100))))
  expect_true(is.na(rhat_single(c(1, 2, NA, 4, 5))))
  expect_true(is.na(rhat_single(c(1, 2, Inf, 4, 5))))
  expect_true(is.na(rhat_single(c(1, 2, 3))))
  expect_false(is.na(rhat_single(rnorm(4))))
})


test_that("extracting rhat_rank_core() left the multi-chain path untouched", {
  # The guard moved to the caller so one chain could reach the same maths.
  # Multi-chain results must be bit-identical, and a single-column matrix must
  # still be NA through the multi-chain entry point -- that function's contract
  # is unchanged.
  x <- iid_chains(n = 400, m = 4, seed = 64)
  expect_identical(rhat_rank_normalized(x), rhat_rank_core(x))
  expect_true(is.na(rhat_rank_normalized(matrix(rnorm(400), ncol = 1L))))
  expect_true(is.na(rhat_rank_normalized(rnorm(400))))
})


test_that("rhat_single() reproduces posterior::rhat", {
  skip_if_not_installed("posterior")

  # The statistic this package reports for one chain must be the one Stan
  # reports for one chain, not a near neighbour of it.
  set.seed(65)
  scenarios <- list(
    iid       = rnorm(500),
    autocorr  = as.numeric(arima.sim(list(ar = 0.95), 500)),
    drifting  = rnorm(400) + seq(0, 2, length.out = 400),
    odd_draws = rnorm(301),
    heavy     = rt(500, df = 2),
    on_unit   = plogis(rnorm(500) - 2)
  )

  for (nm in names(scenarios)) {
    expect_equal(rhat_single(scenarios[[nm]]), posterior::rhat(scenarios[[nm]]),
                 tolerance = 1e-10, info = nm)
  }
})


test_that("rhat_split_single() reproduces posterior::rhat_basic", {
  skip_if_not_installed("posterior")

  # posterior::rhat_basic() is the split, non-rank-normalized statistic, and on
  # a bare vector it treats the input as one chain -- the same definition. This
  # is the reference comparison the `no new dependency` decision owes, on the
  # same footing as the block above.
  set.seed(61)
  scenarios <- list(
    iid       = rnorm(500),
    autocorr  = as.numeric(arima.sim(list(ar = 0.95), 500)),
    drifting  = rnorm(400) + seq(0, 2, length.out = 400),
    odd_draws = rnorm(301),
    heavy     = rt(500, df = 2),
    on_unit   = plogis(rnorm(500) - 2)
  )

  for (nm in names(scenarios)) {
    expect_equal(rhat_split_single(scenarios[[nm]]),
                 posterior::rhat_basic(scenarios[[nm]]),
                 tolerance = 1e-10, info = nm)
  }

  # And the split is what makes it defined at all: posterior returns NA for a
  # single chain without it.
  expect_true(is.na(posterior::rhat_basic(scenarios$iid, split = FALSE)))
})


test_that("the degenerate-tail verdict matches posterior:: too", {
  skip_if_not_installed("posterior")

  x <- iid_chains(n = 200, m = 4, seed = 41)
  x[1:20, ] <- max(x) + 1
  expect_equal(ess_tail(x), posterior::ess_tail(x))

  y <- iid_chains(n = 200, m = 4, seed = 41)
  y[1:120, ] <- min(y) - 1
  expect_equal(ess_tail(y), posterior::ess_tail(y), tolerance = 1e-10)
})
