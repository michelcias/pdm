library(testthat)

# This file contains unit tests for the helper functions
# defined in the 'src/utils.c' source file.

# --- Test for ilogit function ---
test_that("ilogit C function works correctly", {

  # R wrapper for the C test function
  test_ilogit_C <- function(x) {
    .Call("_pdm_test_ilogit", as.numeric(x))
  }

  # Compare against the standard R implementation (plogis)
  expect_equal(test_ilogit_C(0), plogis(0))
  expect_equal(test_ilogit_C(10), plogis(10))
  expect_equal(test_ilogit_C(-5), plogis(-5))

  # Test with a vector (the C function should only use the first element)
  expect_equal(test_ilogit_C(c(2, 3, 4)), plogis(2))
})

# --- Test for generate_normal_vector function ---
test_that("generate_normal_vector C function generates a vector of correct length and type", {

  # R wrapper for the C test function
  test_gen_norm_C <- function(y, a, b, add_a) {
    .Call("_pdm_test_generate_normal_vector", y, a, b, add_a)
  }

  # Define input parameters
  y <- c(1.0, 2.0, 3.0, 4.0, 5.0)
  a <- 1.0
  b <- 2.0
  add_a <- 1L

  # Set seed for reproducibility
  set.seed(123)
  result_c <- test_gen_norm_C(y, a, b, add_a)

  # Test 1: Check output type and length
  expect_true(is.numeric(result_c))
  expect_equal(length(result_c), length(y))

  # Test 2: Check for null or NA values
  expect_false(is.null(result_c))
  expect_false(any(is.na(result_c)))

  # Test 3 (Advanced): Verify reproducibility
  set.seed(123)
  result_c_redux <- test_gen_norm_C(y, a, b, add_a)
  expect_equal(result_c, result_c_redux)
})


test_that("rgamma_positive floors precision draws at machine epsilon", {

  test_rg_C <- function(shape, scale, n) {
    .Call("_pdm_test_rgamma_positive",
          as.numeric(shape), as.numeric(scale), as.integer(n))
  }

  # With a tiny shape (weakly informative precision prior, or an empty mixture
  # component), a large fraction of raw rgamma draws underflows below machine
  # epsilon -- many to exactly 0. The wrapper must floor every draw so that
  # 1/prec, sqrt(1/prec) and the Cholesky pivots downstream remain finite.
  set.seed(501)
  draws <- test_rg_C(0.01, 100, 1e4)
  expect_true(all(is.finite(draws)))
  expect_true(all(draws >= .Machine$double.eps))
  # sanity: the floor is actually being exercised in this regime
  expect_true(any(draws == .Machine$double.eps))

  # In a realistic posterior regime (large shape) the floor must be inert
  set.seed(502)
  draws_post <- test_rg_C(500, 1 / 500, 1e4)
  expect_true(all(draws_post > .Machine$double.eps))
  set.seed(502)
  expect_identical(rgamma(1e4, shape = 500, scale = 1 / 500), draws_post)

  # Reproducibility
  set.seed(503)
  d1 <- test_rg_C(0.01, 100, 100)
  set.seed(503)
  d2 <- test_rg_C(0.01, 100, 100)
  expect_identical(d1, d2)
})
