library(testthat)

# This file contains unit tests for the probit Bernoulli state sampling functions
# defined in 'src/generate_alpha_binomial.c'. These tests focus on the Gibbs
# samplers that use Albert-Chib data augmentation.

test_that("generate_alpha_probit_bernoulli_locallevel runs and is reproducible", {

  test_C <- function(theta_1_in, theta_01_in, prec_theta1_in, y) {
    .Call("_pdm_test_generate_alpha_probit_bernoulli_locallevel",
          as.numeric(theta_1_in), as.numeric(theta_01_in),
          as.numeric(prec_theta1_in), as.numeric(y))
  }

  n <- 6
  theta_1_in <- c(-0.2, -0.1, 0.0, 0.1, 0.05, -0.05)
  theta_01_in <- -0.3
  prec_theta1_in <- 8.0
  y <- c(1, 0, 1, 1, 0, 0)

  set.seed(401)
  result <- test_C(theta_1_in, theta_01_in, prec_theta1_in, y)

  expect_true(is.list(result))
  expect_equal(names(result), c("theta_1", "alpha"))
  expect_length(result$theta_1, n)
  expect_length(result$alpha, n)
  expect_true(all(is.finite(result$theta_1)))
  expect_true(all(result$alpha >= 0 & result$alpha <= 1))
  expect_equal(result$alpha, pnorm(result$theta_1), tolerance = 1e-12)

  set.seed(401)
  result1 <- test_C(theta_1_in, theta_01_in, prec_theta1_in, y)
  set.seed(401)
  result2 <- test_C(theta_1_in, theta_01_in, prec_theta1_in, y)
  expect_equal(result1, result2)
})


test_that("generate_alpha_probit_bernoulli (local trend) runs and is reproducible", {

  test_C <- function(theta_1_in, theta_2_in, theta_01_in, theta_02_in, prec_theta1_in, y) {
    .Call("_pdm_test_generate_alpha_probit_bernoulli",
          as.numeric(theta_1_in), as.numeric(theta_2_in),
          as.numeric(theta_01_in), as.numeric(theta_02_in),
          as.numeric(prec_theta1_in), as.numeric(y))
  }

  n <- 6
  theta_1_in <- c(-0.25, -0.15, -0.05, 0.05, 0.15, 0.05)
  theta_2_in <- c(0.02, 0.015, 0.01, 0.0, -0.005, -0.01)
  theta_01_in <- -0.35
  theta_02_in <- 0.03
  prec_theta1_in <- 6.5
  y <- c(0, 1, 1, 0, 1, 0)

  set.seed(402)
  result <- test_C(theta_1_in, theta_2_in, theta_01_in, theta_02_in, prec_theta1_in, y)

  expect_true(is.list(result))
  expect_equal(names(result), c("theta_1", "alpha"))
  expect_length(result$theta_1, n)
  expect_length(result$alpha, n)
  expect_true(all(is.finite(result$theta_1)))
  expect_true(all(result$alpha >= 0 & result$alpha <= 1))
  expect_equal(result$alpha, pnorm(result$theta_1), tolerance = 1e-12)

  set.seed(402)
  result1 <- test_C(theta_1_in, theta_2_in, theta_01_in, theta_02_in, prec_theta1_in, y)
  set.seed(402)
  result2 <- test_C(theta_1_in, theta_2_in, theta_01_in, theta_02_in, prec_theta1_in, y)
  expect_equal(result1, result2)
})


test_that("probit samplers stay finite for extreme states far from the truncation bound", {

  # Regression test: when y_t = 1 and theta_{t,1} from the previous iteration is
  # ~8+ standard deviations below the truncation bound (e.g., after a long run
  # of zeros drags the level down), the old rtruncnorm computed
  # pnorm(0, theta, 1) = 1 in double precision and qnorm(1) = +Inf, poisoning
  # the whole state vector (and all subsequent MCMC iterations) with Inf/NaN.

  n <- 6
  theta_1_in <- c(-9, -12, -20, -35, 0.1, -0.05)
  y <- c(1, 1, 1, 1, 1, 0)

  set.seed(403)
  result <- .Call("_pdm_test_generate_alpha_probit_bernoulli_locallevel",
                  as.numeric(theta_1_in), as.numeric(-0.3),
                  as.numeric(8.0), as.numeric(y))
  expect_true(all(is.finite(result$theta_1)))
  expect_true(all(is.finite(result$alpha)))
  expect_true(all(result$alpha >= 0 & result$alpha <= 1))

  # Same scenario for the local trend variant
  theta_2_in <- rep(0.01, n)
  set.seed(404)
  result_trend <- .Call("_pdm_test_generate_alpha_probit_bernoulli",
                        as.numeric(theta_1_in), as.numeric(theta_2_in),
                        as.numeric(-0.35), as.numeric(0.03),
                        as.numeric(6.5), as.numeric(y))
  expect_true(all(is.finite(result_trend$theta_1)))
  expect_true(all(is.finite(result_trend$alpha)))
  expect_true(all(result_trend$alpha >= 0 & result_trend$alpha <= 1))

  # Mirrored case: y_t = 0 with the level far above the bound
  set.seed(405)
  result_pos <- .Call("_pdm_test_generate_alpha_probit_bernoulli_locallevel",
                      as.numeric(-theta_1_in), as.numeric(0.3),
                      as.numeric(8.0), as.numeric(1 - y))
  expect_true(all(is.finite(result_pos$theta_1)))
  expect_true(all(is.finite(result_pos$alpha)))
})
