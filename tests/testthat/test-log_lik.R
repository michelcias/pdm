library(testthat)

# Unit tests for the pointwise conditional log-likelihood primitive `log_lik()`
# and its per-family methods. Correctness is checked against naive, pure-R
# element-by-element references; a real fit checks dispatch and orientation.

# --- Synthetic fitted objects (exercise the log_lik computation only) --------
S <- 25L
N <- 9L

mk_obj <- function(class1, comp, y, ...) {
  o <- structure(comp, class = c(class1, "pdm_mcmc", "list"))
  attr(o, "y")       <- y
  attr(o, "n_chain") <- S
  attr(o, "n_obs")   <- N
  extra <- list(...)
  for (nm in names(extra)) attr(o, nm) <- extra[[nm]]
  o
}

grid_ref <- function(fun) outer(seq_len(S), seq_len(N), Vectorize(fun))


test_that("normal log_lik matches dnorm for every latent order", {
  set.seed(1)
  theta_1 <- matrix(rnorm(S * N), S, N)
  prec_y  <- rgamma(S, 3, 1)
  y       <- rnorm(N)
  ref <- grid_ref(function(s, t)
    dnorm(y[t], theta_1[s, t], sqrt(1 / prec_y[s]), log = TRUE))

  for (cls in c("normal_locallevel", "normal_localtrend",
                "normal_localacceleration")) {
    ll <- log_lik(mk_obj(cls, list(theta_1 = theta_1, prec_y = prec_y), y))
    expect_equal(dim(ll), c(S, N))
    expect_equal(ll, ref, tolerance = 1e-12)
  }
})

test_that("poisson log_lik matches dpois for every latent order", {
  set.seed(2)
  alpha <- matrix(rgamma(S * N, 2, 1), S, N)
  y     <- rpois(N, 3)
  ref   <- grid_ref(function(s, t) dpois(y[t], alpha[s, t], log = TRUE))

  for (cls in c("poisson_locallevel", "poisson_localtrend",
                "poisson_localacceleration")) {
    expect_equal(log_lik(mk_obj(cls, list(alpha = alpha), y)), ref,
                 tolerance = 1e-12)
  }
})

test_that("binomial log_lik matches dbinom and uses the n_trials attribute", {
  set.seed(3)
  alpha <- matrix(runif(S * N), S, N)
  nt    <- 20
  y     <- rbinom(N, nt, 0.5)
  ref   <- grid_ref(function(s, t) dbinom(y[t], nt, alpha[s, t], log = TRUE))

  o <- mk_obj("binomial_locallevel", list(alpha = alpha), y, n_trials = nt)
  expect_equal(log_lik(o), ref, tolerance = 1e-12)
})

test_that("probit_bernoulli log_lik matches the Bernoulli density", {
  set.seed(4)
  alpha <- matrix(runif(S * N), S, N)
  y     <- rbinom(N, 1, 0.5)
  ref   <- grid_ref(function(s, t) dbinom(y[t], 1, alpha[s, t], log = TRUE))

  o <- mk_obj("probit_bernoulli_localtrend", list(alpha = alpha), y)
  expect_equal(log_lik(o), ref, tolerance = 1e-12)
})

test_that("normal_mixture log_lik marginalises z and is stable at alpha 0/1", {
  set.seed(5)
  alpha <- matrix(runif(S * N), S, N)
  alpha[1, 1] <- 0        # pure component 1
  alpha[2, 2] <- 1        # pure component 2
  mu_1   <- rnorm(S)
  mu_2   <- rnorm(S, 2)
  prec_1 <- rgamma(S, 4, 1)
  prec_2 <- rgamma(S, 1, 1)
  y      <- rnorm(N)
  ref <- grid_ref(function(s, t)
    log((1 - alpha[s, t]) * dnorm(y[t], mu_1[s], sqrt(1 / prec_1[s])) +
              alpha[s, t]  * dnorm(y[t], mu_2[s], sqrt(1 / prec_2[s]))))

  o <- mk_obj("normal_mixture_locallevel",
              list(alpha = alpha, mu_1 = mu_1, mu_2 = mu_2,
                   prec_1 = prec_1, prec_2 = prec_2), y)
  ll <- log_lik(o)
  expect_equal(ll, ref, tolerance = 1e-12)
  expect_true(all(is.finite(ll)))
})

test_that("log_lik errors when the observations are not stored", {
  o <- mk_obj("normal_locallevel",
              list(theta_1 = matrix(0, S, N), prec_y = rep(1, S)), y = rnorm(N))
  attr(o, "y") <- NULL
  expect_error(log_lik(o), "not stored")
})

test_that("log_lik dispatches on a real fit with a draws-by-obs orientation", {
  set.seed(6)
  y <- cumsum(rnorm(50)) + rnorm(50, sd = 0.3)
  fit <- mcmc_normal_locallevel(
    y, burnin = 30, thinning = 1, n_chain = 80,
    prior_theta01_mean = y[1], prior_theta01_prec = 1 / var(y),
    prior_prec1_shape  = 1e-2, prior_prec1_rate  = 1e-2,
    prior_prec_y_shape = 1e-2, prior_prec_y_rate = 1e-2, seed = 42
  )
  ll <- log_lik(fit)
  expect_equal(dim(ll), c(80L, 50L))
  ref <- outer(seq_len(80), seq_len(50), Vectorize(function(s, t)
    dnorm(y[t], fit$theta_1[s, t], sqrt(1 / fit$prec_y[s]), log = TRUE)))
  expect_equal(ll, ref, tolerance = 1e-12)
})
