library(testthat)

# Reference implementation (pure R) used only to validate the C version.
hpdi_ref <- function(data, prob) {
  one <- function(col) {
    n <- length(col)
    s <- sort.int(col, method = "quick")
    m <- floor(prob * n)
    if (m < 1L) m <- 1L
    if (m > n - 1L) m <- n - 1L
    nw <- n - m
    i <- which.min(s[(1L + m):(nw + m)] - s[1L:nw])
    c(lower = s[i], upper = s[i + m])
  }
  if (is.matrix(data)) {
    t(vapply(seq_len(ncol(data)), function(j) one(data[, j]), numeric(2)))
  } else {
    one(data)
  }
}

test_that("hpdi matches the reference for vector input", {
  set.seed(11)
  v <- rgamma(3000, shape = 2, rate = 1)
  for (prob in c(0.5, 0.8, 0.9, 0.95)) {
    out <- hpdi(v, prob)
    expect_named(out, c("lower", "upper"))
    expect_equal(unname(out), unname(hpdi_ref(v, prob)), tolerance = 1e-12)
    expect_lt(out[["lower"]], out[["upper"]])
  }
})

test_that("hpdi matches the reference for matrix input (per column)", {
  set.seed(12)
  M <- matrix(rnorm(2000 * 40), nrow = 2000, ncol = 40)
  for (prob in c(0.5, 0.9, 0.95)) {
    out <- hpdi(M, prob)
    expect_equal(dim(out), c(40L, 2L))
    expect_equal(colnames(out), c("lower", "upper"))
    expect_equal(unname(out), unname(hpdi_ref(M, prob)), tolerance = 1e-12)
    expect_true(all(out[, "lower"] <= out[, "upper"]))
  }
})

test_that("hpdi contains at least the requested probability mass", {
  set.seed(13)
  v <- rnorm(5000)
  prob <- 0.9
  out <- hpdi(v, prob)
  covered <- mean(v >= out[["lower"]] & v <= out[["upper"]])
  expect_gte(covered, prob)
})

test_that("hpdi is the shortest window of its span", {
  set.seed(14)
  v <- rgamma(5000, shape = 1.5, rate = 1)  # skewed
  h <- hpdi(v, 0.9)
  # By construction the HPDI is the minimum-width window of span m over the
  # sorted sample, so it cannot exceed any particular window of that span
  # (here the central, equal-tailed-style window).
  s <- sort(v)
  n <- length(v)
  m <- floor(0.9 * n)
  i0 <- floor(0.05 * n) + 1L
  central_width <- s[i0 + m] - s[i0]
  expect_lte(h[["upper"]] - h[["lower"]], central_width + 1e-12)
})

test_that("hpdi validates its arguments", {
  expect_error(hpdi("a"), "numeric")
  expect_error(hpdi(1:10, prob = 0), "between 0 and 1")
  expect_error(hpdi(1:10, prob = 1), "between 0 and 1")
  expect_error(hpdi(1:10, prob = c(0.5, 0.9)), "single value")
  expect_error(hpdi(numeric(1)), "at least 2")
})
