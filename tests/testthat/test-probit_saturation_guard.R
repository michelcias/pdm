library(testthat)

# Regression tests for the probit saturation guard in
# 'src/generate_alpha_binomial.c' (clamp_probit_state / PROBIT_THETA_CLAMP).
#
# Rationale: for |theta| >= 8 the standard normal CDF is numerically 0 or 1, so
# the Bernoulli likelihood P(y=1|theta)=Phi(theta) is flat and theta_1 becomes
# unidentified. Without a guard the Albert-Chib augmentation lets theta_1 drift
# far into that tail (values in the tens on segmented / well-separated data),
# which collapses the innovation precision 1/W_1 and stalls the chain. The guard
# clamps the latent state to [-8, 8]; this is numerically inert for alpha (Phi is
# already saturated at the bound) and for well-identified problems, where
# |theta_1| stays well below 8.

test_that("probit state update clamps a runaway latent state", {

  test_C <- function(theta_1_in, theta_01_in, prec_theta1_in, y) {
    .Call("_pdm_test_generate_alpha_probit_bernoulli_locallevel",
          as.numeric(theta_1_in), as.numeric(theta_01_in),
          as.numeric(prec_theta1_in), as.numeric(y))
  }

  # Start from an extreme, saturated state with data that "agrees" (y = 1 where
  # theta is large positive, y = 0 where it is large negative). The unclamped
  # sampler would keep theta_1 out in the tens; the guard must pull it back to
  # the [-8, 8] band.
  n <- 8
  theta_1_in <- c(40, 45, 50, 42, -40, -45, -50, -42)
  y          <- c(1,  1,  1,  1,   0,   0,   0,   0)

  set.seed(911)
  res <- test_C(theta_1_in, theta_01_in = 0.0, prec_theta1_in = 5.0, y)

  expect_true(all(is.finite(res$theta_1)))
  expect_true(all(abs(res$theta_1) <= 8 + 1e-9),
              info = "latent state must be clamped to the [-8, 8] band")
  # alpha stays strictly inside (0, 1): the downstream indicator sampler must
  # never be forced into its degenerate deterministic branch.
  expect_true(all(res$alpha > 0 & res$alpha < 1),
              info = "alpha must be strictly inside (0, 1)")
  # The transform invariant alpha == Phi(theta_1) is preserved by the guard.
  expect_equal(res$alpha, pnorm(res$theta_1), tolerance = 1e-12)
})

test_that("probit guard is inert for a well-identified (small-theta) state", {

  test_C <- function(theta_1_in, theta_01_in, prec_theta1_in, y) {
    .Call("_pdm_test_generate_alpha_probit_bernoulli_locallevel",
          as.numeric(theta_1_in), as.numeric(theta_01_in),
          as.numeric(prec_theta1_in), as.numeric(y))
  }

  # Moderate states never approach the clamp, so the update is unaffected and the
  # draw is bit-for-bit reproducible.
  n <- 6
  theta_1_in <- c(-0.2, -0.1, 0.0, 0.1, 0.05, -0.05)
  y <- c(1, 0, 1, 1, 0, 0)

  set.seed(401)
  r1 <- test_C(theta_1_in, -0.3, 8.0, y)
  set.seed(401)
  r2 <- test_C(theta_1_in, -0.3, 8.0, y)

  expect_equal(r1, r2)
  expect_true(all(abs(r1$theta_1) < 8))
  expect_equal(r1$alpha, pnorm(r1$theta_1), tolerance = 1e-12)
})

test_that("probit mixture sampler does not drift on segmented data", {

  skip_on_cran()

  # Piecewise-constant, well-separated data with pure single-component stretches
  # (the aCGH copy-number setting). Under probit this drives alpha -> {0, 1} and,
  # without the guard, theta_1 drifts to |theta_1| in the tens while 1/W_1
  # collapses. With the guard theta_1 stays within the band and alpha stays valid.
  set.seed(42)
  n <- 500
  seg <- rep(0, n)
  seg[120:200] <- 1
  seg[330:410] <- 1
  y <- rnorm(n, seg * 1.2, 0.25)

  out <- mcmc_normal_mixture_locallevel(
    y, link = "probit", burnin = 1500, thinning = 5, n_chain = 600,
    prior_theta01_prec = 1,
    prior_prec1_shape = 0.01, prior_prec1_rate = 0.01,
    seed = 7
  )

  expect_true(all(is.finite(out$theta_1)))
  expect_true(max(abs(out$theta_1)) <= 8 + 1e-9,
              info = "theta_1 must not drift beyond the saturation band")
  expect_true(all(out$alpha > 0 & out$alpha < 1),
              info = "alpha must never be exactly 0 or 1")
  # 1/W_1 must not collapse toward zero: with the guard its posterior mean stays
  # comfortably away from 0 (pre-fix runs collapsed to ~0.4; the smoothing signal
  # here supports a value of order 1 or larger).
  expect_gt(mean(out$prec_theta1), 1.0)
})
