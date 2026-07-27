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
