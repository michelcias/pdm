library(testthat)

# Regression tests for the probit probability guard in
# 'src/generate_alpha_binomial.c' (clamp_link_alpha / LINK_ALPHA_MIN /
# LINK_ALPHA_MAX).
#
# Rationale: the latent state theta_1 is stored exactly as drawn -- there is no
# state clamp. Instead the *probability* alpha = Phi(theta_1) reported to the
# caller is constrained to stay strictly inside (0, 1), specifically within
# [2e-16, 1 - 2.3e-16]. This keeps alpha away from an exact 0 or 1 -- which would
# otherwise force downstream consumers (e.g. the mixture indicator sampler) into
# a degenerate deterministic branch -- while leaving the sampled latent state
# undistorted.

link_alpha_min <- 2e-16
link_alpha_max <- 1 - 2.3e-16
clamp_link_alpha <- function(p) pmin(pmax(p, link_alpha_min), link_alpha_max)

test_that("probit alpha stays inside the guarded band for a saturated state", {

  test_C <- function(theta_1_in, theta_01_in, prec_theta1_in, y) {
    .Call("_pdm_test_generate_alpha_probit_bernoulli_locallevel",
          as.numeric(theta_1_in), as.numeric(theta_01_in),
          as.numeric(prec_theta1_in), as.numeric(y))
  }

  # Start from an extreme, saturated state with data that "agrees" (y = 1 where
  # theta is large positive, y = 0 where it is large negative). The latent state
  # is no longer clamped, so theta_1 may remain large; the guard must nonetheless
  # keep alpha strictly inside (0, 1) and equal to the clamped Phi(theta_1).
  theta_1_in <- c(40, 45, 50, 42, -40, -45, -50, -42)
  y          <- c(1,  1,  1,  1,   0,   0,   0,   0)

  set.seed(911)
  res <- test_C(theta_1_in, theta_01_in = 0.0, prec_theta1_in = 5.0, y)

  expect_true(all(is.finite(res$theta_1)))
  # alpha stays strictly inside (0, 1) and within the guarded band.
  expect_true(all(res$alpha > 0 & res$alpha < 1),
              info = "alpha must be strictly inside (0, 1)")
  expect_true(all(res$alpha >= link_alpha_min & res$alpha <= link_alpha_max),
              info = "alpha must stay within [2e-16, 1 - 2.3e-16]")
  # The transform invariant alpha == clamp(Phi(theta_1)) is preserved.
  expect_equal(res$alpha, clamp_link_alpha(pnorm(res$theta_1)), tolerance = 1e-12)
})

test_that("probit guard is inert for a well-identified (small-theta) state", {

  test_C <- function(theta_1_in, theta_01_in, prec_theta1_in, y) {
    .Call("_pdm_test_generate_alpha_probit_bernoulli_locallevel",
          as.numeric(theta_1_in), as.numeric(theta_01_in),
          as.numeric(prec_theta1_in), as.numeric(y))
  }

  # Moderate states are far from saturation, so the probability guard is inert:
  # alpha == Phi(theta_1) exactly and the draw is bit-for-bit reproducible.
  theta_1_in <- c(-0.2, -0.1, 0.0, 0.1, 0.05, -0.05)
  y <- c(1, 0, 1, 1, 0, 0)

  set.seed(401)
  r1 <- test_C(theta_1_in, -0.3, 8.0, y)
  set.seed(401)
  r2 <- test_C(theta_1_in, -0.3, 8.0, y)

  expect_equal(r1, r2)
  expect_true(all(is.finite(r1$theta_1)))
  # Guard inert: alpha equals the raw, unclamped Phi(theta_1).
  expect_equal(r1$alpha, pnorm(r1$theta_1), tolerance = 1e-12)
  expect_true(all(r1$alpha > link_alpha_min & r1$alpha < link_alpha_max))
})

test_that("probit mixture sampler stays numerically valid on segmented data", {

  skip_on_cran()

  # Piecewise-constant, well-separated data with pure single-component stretches
  # (the aCGH copy-number setting). Under probit this drives alpha -> {0, 1} and
  # lets theta_1 drift into the flat tail of Phi. The latent state is intentionally
  # left unclamped; the probability guard must nonetheless keep every reported
  # alpha strictly inside (0, 1) and within the guarded band, with no NaN.
  set.seed(42)
  n <- 500
  seg <- rep(0, n)
  seg[120:200] <- 1
  seg[330:410] <- 1
  y <- rnorm(n, seg * 1.2, 0.25)

  out <- mcmc_normal_mixture_locallevel(
    y, link = "probit", burnin = 1500, thinning = 5, n_draws = 600,
    prior_theta01_prec = 1,
    prior_prec1_shape = 0.01, prior_prec1_rate = 0.01,
    seed = 7
  )

  expect_false(any(is.na(out$alpha)),
               info = "alpha must never be NaN")
  expect_true(all(out$alpha > 0 & out$alpha < 1),
              info = "alpha must never be exactly 0 or 1")
  expect_true(all(out$alpha >= link_alpha_min & out$alpha <= link_alpha_max),
              info = "alpha must stay within [2e-16, 1 - 2.3e-16]")
})
