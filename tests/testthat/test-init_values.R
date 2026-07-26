library(testthat)

# `init` lets a caller pin any subset of the starting values; everything left
# out is still drawn from its prior. The tests below pin down three things: the
# validation (an unrecognised name must not pass for a value that had no
# effect), the draw order (an `init = NULL` run has to consume the RNG stream
# exactly as the C drivers used to, or every earlier seed changes meaning), and
# the contract with `chains > 1`.

# Small helpers matching the shapes resolve_init() expects.
gamma_prior <- function(shape, rate) {
  list(code = 0L, type = "gamma", shape = shape, rate = rate, scale = 1, df = 1)
}
halft_prior <- function(scale, df = 1) {
  list(code = 1L, type = "halft", shape = 1, rate = 1, scale = scale, df = df)
}
one_state <- list(theta_01 = list(mean = 3, prec = 0.25))


test_that("nothing supplied means everything is drawn, in the documented order", {
  states <- list(theta_01 = list(mean = 3, prec = 0.25))
  precs  <- list(prec_theta1 = halft_prior(2, df = 1),
                 prec_y      = gamma_prior(2, 1))

  set.seed(99)
  got <- resolve_init(NULL, states, precs)

  # The same sequence written out by hand: the state, then for each precision
  # its auxiliary (Half-t only) immediately before the precision itself. This
  # is the order src/mcmc_normal_*.c drew in before `init` existed, and the
  # reason a pre-0.9-0 fit of a given seed still reproduces.
  set.seed(99)
  theta_01 <- rnorm(1, 3, sqrt(1 / 0.25))
  aux_1    <- rgamma(1, shape = 0.5, scale = 2^2)
  prec_1   <- rgamma(1, shape = 0.5, scale = 1 / aux_1)
  prec_y   <- rgamma(1, shape = 2, scale = 1 / 1)

  expect_equal(got$init, list(theta_01 = theta_01, prec_theta1 = prec_1,
                              prec_y = prec_y))
  # The vector handed to .Call is states, then precisions, then one auxiliary
  # per precision in the same order; a Gamma prior contributes a 0 that C never
  # reads.
  expect_equal(got$values, c(theta_01, prec_1, prec_y, aux_1, 0))
})


test_that("a supplied value is used as given and stops its own draw", {
  precs <- list(prec_y = gamma_prior(1e-2, 1e-2))

  set.seed(7)
  got   <- resolve_init(list(theta_01 = -4, prec_y = 2.5), one_state, precs)
  after <- rnorm(1)

  expect_equal(got$init$theta_01, -4)
  expect_equal(got$init$prec_y, 2.5)

  # Neither value was drawn, so the call consumed nothing: the next draw is the
  # first one the seed produces.
  set.seed(7)
  expect_equal(after, rnorm(1))
})


test_that("a subnormal draw is floored, as in the C sampler", {
  # A tiny shape makes a draw at or below DBL_EPSILON likely. Whatever comes
  # out must never be zero: a zero precision poisons the rate of the next
  # update, and a zero auxiliary poisons the precision that follows it. This is
  # the R twin of the guard in rgamma_positive() (src/utils.c).
  set.seed(4)
  draws <- vapply(1:50, function(i) rgamma_floor(1e-3, 1), numeric(1))
  expect_true(all(draws >= .Machine$double.eps))
})


test_that("a pinned Half-t precision gets its auxiliary from the full conditional", {
  precs <- list(prec_theta1 = halft_prior(2, df = 3))

  set.seed(21)
  got <- resolve_init(list(prec_theta1 = 5), list(), precs)

  # b | W^-1 ~ Gamma((df + 1)/2, rate = df * W^-1 + 1/A^2) -- the same draw
  # generate_halft_aux() makes after every Half-t precision step, so the pair
  # (precision, auxiliary) the chain starts from is a coherent state.
  set.seed(21)
  expect_equal(got$values[[2]], rgamma(1, shape = 2, scale = 1 / (3 * 5 + 1 / 4)))
})


test_that("an unrecognised name is an error, not a silent no-op", {
  precs <- list(prec_theta1 = gamma_prior(1, 1))

  expect_error(resolve_init(list(theta0_1 = 1), one_state, precs),
               "unknown name")
  # The message names what would have been accepted.
  expect_error(resolve_init(list(theta0_1 = 1), one_state, precs),
               "theta_01")
  expect_error(resolve_init(list(theta_01 = 1, theta_01 = 2), one_state, precs),
               "duplicated")
  expect_error(resolve_init(list(1), one_state, precs), "must be named")
  expect_error(resolve_init(c(theta_01 = 1), one_state, precs),
               "must be a named list")
})


test_that("supplied values are checked for type, length and sign", {
  precs <- list(prec_theta1 = gamma_prior(1, 1))

  expect_error(resolve_init(list(theta_01 = c(1, 2)), one_state, precs),
               "single finite numeric")
  expect_error(resolve_init(list(theta_01 = NA_real_), one_state, precs),
               "single finite numeric")
  expect_error(resolve_init(list(theta_01 = Inf), one_state, precs),
               "single finite numeric")
  expect_error(resolve_init(list(theta_01 = "a"), one_state, precs),
               "single finite numeric")

  # A precision must additionally be positive.
  expect_error(resolve_init(list(prec_theta1 = 0), one_state, precs),
               "single positive finite numeric")
  expect_error(resolve_init(list(prec_theta1 = -1), one_state, precs),
               "single positive finite numeric")
})


test_that("an empty or absent init is the same as drawing everything", {
  precs <- list(prec_y = gamma_prior(1e-2, 1e-2))

  set.seed(5)
  a <- resolve_init(NULL, one_state, precs)
  set.seed(5)
  b <- resolve_init(list(), one_state, precs)

  expect_equal(a, b)
})


test_that("split_init_by_chain refuses one starting point for every chain", {
  # Sharing a start across chains removes the between-chain dispersion R-hat is
  # computed from, so the diagnostic would come out optimistic with nothing to
  # signal it. That is a wrong answer, hence an error rather than a warning.
  expect_error(split_init_by_chain(list(prec_y = 1), 3), "one per chain")
  expect_error(split_init_by_chain(list(list(prec_y = 1), list(prec_y = 2)), 3),
               "got 2 for 3 chains")

  expect_equal(split_init_by_chain(NULL, 3), vector("list", 3))
  per_chain <- list(list(prec_y = 1), NULL, list(prec_y = 3))
  expect_equal(split_init_by_chain(per_chain, 3), per_chain)
})


test_that("the per-chain form is rejected by a single-chain call", {
  expect_error(
    resolve_init(list(list(prec_theta1 = 1), list(prec_theta1 = 2)),
                 one_state, list(prec_theta1 = gamma_prior(1, 1))),
    "one list per chain"
  )
})


# --- End to end, through the samplers -----------------------------------------

test_that("init reaches the sampler and is recorded on the fit", {
  set.seed(1)
  y <- cumsum(c(10, rnorm(60)))[-1] + rnorm(60, sd = 0.4)

  f <- mcmc_normal_locallevel(y, 100, 2, 50,
                              prior_theta01_mean = y[1],
                              prior_theta01_prec = 1 / var(y),
                              init = list(prec_y = 3, theta_01 = 12),
                              seed = 456)

  rec <- attr(f, "init")
  expect_equal(rec$prec_y, 3)
  expect_equal(rec$theta_01, 12)
  # prec_theta1 was left out, so it was drawn and is reported as the number
  # actually used.
  expect_true(is.numeric(rec$prec_theta1) && rec$prec_theta1 > 0)

  # Same seed, different start, different chain: `init` is not decorative.
  g <- mcmc_normal_locallevel(y, 100, 2, 50,
                              prior_theta01_mean = y[1],
                              prior_theta01_prec = 1 / var(y),
                              seed = 456)
  expect_false(identical(f$prec_y, g$prec_y))
})


test_that("every Gaussian sampler accepts init and names its own parameters", {
  set.seed(2)
  y <- cumsum(c(10, rnorm(60)))[-1] + rnorm(60, sd = 0.4)

  lt <- mcmc_normal_localtrend(y, 100, 2, 50,
                               prior_theta01_mean = y[1],
                               prior_theta01_prec = 1 / var(y),
                               prior_theta02_mean = 0,
                               prior_theta02_prec = 1,
                               init = list(theta_02 = 0.5, prec_theta2 = 9),
                               seed = 456)
  expect_equal(attr(lt, "init")$theta_02, 0.5)
  expect_equal(attr(lt, "init")$prec_theta2, 9)

  la <- mcmc_normal_localacceleration(y, 100, 2, 50,
                                      prior_theta01_mean = y[1],
                                      prior_theta01_prec = 1 / var(y),
                                      prior_theta02_mean = 0,
                                      prior_theta02_prec = 1,
                                      prior_theta03_mean = 0,
                                      prior_theta03_prec = 1,
                                      init = list(theta_03 = 0.1,
                                                  prec_theta3 = 7),
                                      seed = 456)
  expect_equal(attr(la, "init")$theta_03, 0.1)
  expect_equal(attr(la, "init")$prec_theta3, 7)

  # theta_03 belongs to the acceleration model only.
  expect_error(
    mcmc_normal_locallevel(y, 100, 2, 50,
                           prior_theta01_mean = y[1],
                           prior_theta01_prec = 1 / var(y),
                           init = list(theta_03 = 1)),
    "unknown name"
  )
})


test_that("the names init accepts are the object's own, spelled as true_values spells them", {
  # One vocabulary across the package: a parameter is spelled the same in the
  # fit's components, in `init`, and in the plot method's `true_values`. `init`
  # covers less -- the latent trajectories are `true_values`-only -- but it must
  # never spell a shared parameter differently.
  set.seed(6)
  y <- cumsum(c(10, rnorm(40)))[-1] + rnorm(40, sd = 0.4)

  accepted_names <- function(f, extra) {
    msg <- tryCatch(
      do.call(f, c(list(y, 20, 1, 10), extra, list(init = list(nope = 1)))),
      error = function(e) conditionMessage(e))
    strsplit(sub(".*Valid names for this model are: ", "", msg), ", ")[[1]]
  }

  ll <- list(prior_theta01_mean = y[1], prior_theta01_prec = 1 / var(y))
  lt <- c(ll, list(prior_theta02_mean = 0, prior_theta02_prec = 1))
  la <- c(lt, list(prior_theta03_mean = 0, prior_theta03_prec = 1))

  # The display labels get_param_config() uses, against the component names.
  labels <- c(theta_01    = "theta_01", theta_02    = "theta_02",
              theta_03    = "theta_03", prec_theta1 = "W_1^{-1}",
              prec_theta2 = "W_2^{-1}", prec_theta3 = "W_3^{-1}",
              prec_y      = "V^{-1}")

  cases <- list(list(mcmc_normal_locallevel, ll),
                list(mcmc_normal_localtrend, lt),
                list(mcmc_normal_localacceleration, la))

  for (case in cases) {
    fit      <- do.call(case[[1]], c(list(y, 20, 1, 10), case[[2]]))
    accepted <- accepted_names(case[[1]], case[[2]])

    # Exactly the scalar components: the trajectories start flat at their
    # theta_0k in C and are deliberately not settable.
    scalars <- names(fit)[!vapply(fit, is.matrix, logical(1L))]
    expect_equal(sort(accepted), sort(scalars))

    # And true_value_for() resolves each display label to that same name, so
    # `init = list(prec_theta1 = ...)` and
    # `true_values = list(prec_theta1 = ...)` speak about one parameter.
    for (nm in accepted) {
      expect_equal(true_value_for(labels[[nm]], stats::setNames(list(42), nm)), 42)
    }
  }
})


test_that("a multi-chain call takes one starting-value list per chain", {
  set.seed(3)
  y <- cumsum(c(10, rnorm(60)))[-1] + rnorm(60, sd = 0.4)

  fits <- mcmc_normal_locallevel(y, 100, 2, 50,
                                 prior_theta01_mean = y[1],
                                 prior_theta01_prec = 1 / var(y),
                                 chains = 3,
                                 init = list(list(prec_y = 1),
                                             list(prec_y = 5),
                                             list(prec_y = 20)),
                                 seed = 456)

  expect_s3_class(fits, "pdm_mcmc_list")
  expect_equal(vapply(fits, function(ch) attr(ch, "init")$prec_y, numeric(1)),
               c(1, 5, 20))

  expect_error(
    mcmc_normal_locallevel(y, 100, 2, 50,
                           prior_theta01_mean = y[1],
                           prior_theta01_prec = 1 / var(y),
                           chains = 3, init = list(prec_y = 1)),
    "one per chain"
  )
})


# --- The Poisson and binomial families ----------------------------------------
# Same helper, same contract, one parameter fewer: a link family has no
# observation precision, so `prec_y` must not be accepted here.

test_that("init reaches the Poisson and binomial samplers and is recorded", {
  set.seed(8)
  n  <- 40
  th <- cumsum(c(1, rnorm(n, sd = 0.15)))[-1]
  yp <- rpois(n, exp(th))
  yb <- rbinom(n, 10, plogis(th - 1))

  fp <- mcmc_poisson_locallevel(yp, 60, 1, 30,
                                prior_theta01_mean = log(mean(yp)),
                                prior_theta01_prec = 1,
                                init = list(theta_01 = 1.5, prec_theta1 = 8),
                                verbose = FALSE, seed = 456)
  expect_equal(attr(fp, "init")$theta_01, 1.5)
  expect_equal(attr(fp, "init")$prec_theta1, 8)

  fb <- mcmc_binomial_locallevel(yb, 10, 60, 1, 30,
                                 prior_theta01_mean = 0,
                                 prior_theta01_prec = 1,
                                 init = list(prec_theta1 = 8),
                                 verbose = FALSE, seed = 456)
  expect_equal(attr(fb, "init")$prec_theta1, 8)
  # theta_01 was left out, so it was drawn and reported as the value used.
  expect_true(is.numeric(attr(fb, "init")$theta_01))

  # `init` is not decorative: the same seed from a different start is a
  # different chain.
  gp <- mcmc_poisson_locallevel(yp, 60, 1, 30,
                                prior_theta01_mean = log(mean(yp)),
                                prior_theta01_prec = 1,
                                verbose = FALSE, seed = 456)
  expect_false(identical(fp$prec_theta1, gp$prec_theta1))
})


test_that("a link family rejects prec_y, which it does not have", {
  set.seed(9)
  n  <- 40
  yp <- rpois(n, 5)
  yb <- rbinom(n, 10, 0.4)

  # prec_y belongs to the Gaussian observation model only. Accepting it here
  # would silently do nothing, which is the failure mode the name check exists
  # to prevent.
  expect_error(
    mcmc_poisson_locallevel(yp, 20, 1, 10, prior_theta01_mean = 1,
                            prior_theta01_prec = 1, verbose = FALSE,
                            init = list(prec_y = 1)),
    "unknown name")
  expect_error(
    mcmc_binomial_locallevel(yb, 10, 20, 1, 10, prior_theta01_mean = 0,
                             prior_theta01_prec = 1, verbose = FALSE,
                             init = list(prec_y = 1)),
    "unknown name")
  # Nor the trajectories or the derived rates, which `true_values` does accept.
  expect_error(
    mcmc_poisson_locallevel(yp, 20, 1, 10, prior_theta01_mean = 1,
                            prior_theta01_prec = 1, verbose = FALSE,
                            init = list(alpha = 1)),
    "unknown name")
})


test_that("the vocabulary invariant holds for the Poisson and binomial families", {
  # As for the Gaussian samplers: `init` accepts exactly the fit's scalar
  # components, and every one of them is a name true_value_for() resolves to.
  set.seed(10)
  n  <- 40
  th <- cumsum(c(1, rnorm(n, sd = 0.15)))[-1]
  yp <- rpois(n, exp(th))
  yb <- rbinom(n, 10, plogis(th - 1))

  labels <- c(theta_01    = "theta_01", theta_02    = "theta_02",
              theta_03    = "theta_03", prec_theta1 = "W_1^{-1}",
              prec_theta2 = "W_2^{-1}", prec_theta3 = "W_3^{-1}")

  th0 <- list(prior_theta01_mean = 0, prior_theta01_prec = 1)
  th2 <- c(th0, list(prior_theta02_mean = 0, prior_theta02_prec = 1))
  th3 <- c(th2, list(prior_theta03_mean = 0, prior_theta03_prec = 1))

  cases <- list(
    list(mcmc_poisson_locallevel,         list(yp, 20, 1, 10), th0),
    list(mcmc_poisson_localtrend,         list(yp, 20, 1, 10), th2),
    list(mcmc_poisson_localacceleration,  list(yp, 20, 1, 10), th3),
    list(mcmc_binomial_locallevel,        list(yb, 10, 20, 1, 10), th0),
    list(mcmc_binomial_localtrend,        list(yb, 10, 20, 1, 10), th2),
    list(mcmc_binomial_localacceleration, list(yb, 10, 20, 1, 10), th3))

  for (case in cases) {
    f <- case[[1]]; pos <- case[[2]]; extra <- c(case[[3]], list(verbose = FALSE))

    fit <- do.call(f, c(pos, extra))
    msg <- tryCatch(do.call(f, c(pos, extra, list(init = list(nope = 1)))),
                    error = function(e) conditionMessage(e))
    accepted <- strsplit(sub(".*Valid names for this model are: ", "", msg), ", ")[[1]]

    scalars <- names(fit)[!vapply(fit, is.matrix, logical(1L))]
    expect_equal(sort(accepted), sort(scalars))
    expect_false("prec_y" %in% accepted)

    for (nm in accepted) {
      expect_equal(true_value_for(labels[[nm]], stats::setNames(list(42), nm)), 42)
    }
  }
})


test_that("the Poisson and binomial samplers take one init per chain", {
  set.seed(11)
  n  <- 40
  yp <- rpois(n, 5)

  fits <- mcmc_poisson_locallevel(yp, 60, 1, 30,
                                  prior_theta01_mean = log(mean(yp)),
                                  prior_theta01_prec = 1,
                                  chains = 3,
                                  init = list(list(prec_theta1 = 1),
                                              list(prec_theta1 = 5),
                                              list(prec_theta1 = 20)),
                                  verbose = FALSE, seed = 456)
  expect_s3_class(fits, "pdm_mcmc_list")
  expect_equal(vapply(fits, function(ch) attr(ch, "init")$prec_theta1, numeric(1)),
               c(1, 5, 20))

  expect_error(
    mcmc_poisson_locallevel(yp, 20, 1, 10, prior_theta01_mean = 1,
                            prior_theta01_prec = 1, chains = 3,
                            verbose = FALSE, init = list(prec_theta1 = 1)),
    "one per chain")
})


# --- The probit-Bernoulli family ----------------------------------------------
# The last of the link families. Its C code lives in the same three files as the
# logit binomial sampler, one entry point below it, so these tests also stand
# guard over that split: if an edit meant for one function landed on the other,
# the accepted names or the recorded start would move.

test_that("init reaches the probit-Bernoulli samplers and is recorded", {
  set.seed(12)
  n  <- 40
  th <- cumsum(c(0, rnorm(n, sd = 0.15)))[-1]
  y  <- rbinom(n, 1, pnorm(th))

  f <- mcmc_probit_bernoulli_locallevel(y, 60, 1, 30,
                                        prior_theta01_mean = 0,
                                        prior_theta01_prec = 1,
                                        init = list(theta_01 = 0.4,
                                                    prec_theta1 = 8),
                                        verbose = FALSE, seed = 456)
  expect_equal(attr(f, "init")$theta_01, 0.4)
  expect_equal(attr(f, "init")$prec_theta1, 8)

  g <- mcmc_probit_bernoulli_locallevel(y, 60, 1, 30,
                                        prior_theta01_mean = 0,
                                        prior_theta01_prec = 1,
                                        verbose = FALSE, seed = 456)
  expect_false(identical(f$prec_theta1, g$prec_theta1))

  # Left out means drawn, and reported as the value actually used.
  h <- mcmc_probit_bernoulli_locallevel(y, 60, 1, 30,
                                        prior_theta01_mean = 0,
                                        prior_theta01_prec = 1,
                                        init = list(prec_theta1 = 8),
                                        verbose = FALSE, seed = 456)
  expect_equal(attr(h, "init")$prec_theta1, 8)
  expect_true(is.numeric(attr(h, "init")$theta_01))
})


test_that("the vocabulary invariant holds for the probit-Bernoulli family", {
  set.seed(13)
  n  <- 40
  th <- cumsum(c(0, rnorm(n, sd = 0.15)))[-1]
  y  <- rbinom(n, 1, pnorm(th))

  labels <- c(theta_01    = "theta_01", theta_02    = "theta_02",
              theta_03    = "theta_03", prec_theta1 = "W_1^{-1}",
              prec_theta2 = "W_2^{-1}", prec_theta3 = "W_3^{-1}")

  th0 <- list(prior_theta01_mean = 0, prior_theta01_prec = 1)
  th2 <- c(th0, list(prior_theta02_mean = 0, prior_theta02_prec = 1))
  th3 <- c(th2, list(prior_theta03_mean = 0, prior_theta03_prec = 1))

  cases <- list(list(mcmc_probit_bernoulli_locallevel,        th0),
                list(mcmc_probit_bernoulli_localtrend,        th2),
                list(mcmc_probit_bernoulli_localacceleration, th3))

  for (case in cases) {
    f     <- case[[1]]
    extra <- c(case[[2]], list(verbose = FALSE))

    fit <- do.call(f, c(list(y, 20, 1, 10), extra))
    msg <- tryCatch(do.call(f, c(list(y, 20, 1, 10), extra,
                                 list(init = list(nope = 1)))),
                    error = function(e) conditionMessage(e))
    accepted <- strsplit(sub(".*Valid names for this model are: ", "", msg), ", ")[[1]]

    scalars <- names(fit)[!vapply(fit, is.matrix, logical(1L))]
    expect_equal(sort(accepted), sort(scalars))
    # No observation precision here either, and alpha is derived from theta.
    expect_false("prec_y" %in% accepted)
    expect_false("alpha" %in% accepted)

    for (nm in accepted) {
      expect_equal(true_value_for(labels[[nm]], stats::setNames(list(42), nm)), 42)
    }
  }
})


test_that("the probit and logit samplers keep separate starting states", {
  # The two entry points share a C file. Pinning the probit one must not reach
  # the logit one, and vice versa: a mis-targeted edit would show up here.
  set.seed(14)
  n  <- 40
  th <- cumsum(c(0, rnorm(n, sd = 0.15)))[-1]
  yb <- rbinom(n, 1, pnorm(th))

  probit <- mcmc_probit_bernoulli_locallevel(yb, 60, 1, 30,
                                            prior_theta01_mean = 0,
                                            prior_theta01_prec = 1,
                                            init = list(prec_theta1 = 3),
                                            verbose = FALSE, seed = 456)
  logit  <- mcmc_binomial_locallevel(yb, 1, 60, 1, 30,
                                     prior_theta01_mean = 0,
                                     prior_theta01_prec = 1,
                                     init = list(prec_theta1 = 3),
                                     verbose = FALSE, seed = 456)

  expect_equal(attr(probit, "init")$prec_theta1, 3)
  expect_equal(attr(logit, "init")$prec_theta1, 3)
  # Same start, different algorithms (Albert-Chib against adaptive MH), so the
  # draws must not coincide.
  expect_false(identical(probit$prec_theta1, logit$prec_theta1))
})


test_that("every probit-Bernoulli sampler takes one init per chain", {
  set.seed(15)
  n <- 40
  y <- rbinom(n, 1, 0.4)

  fits <- mcmc_probit_bernoulli_locallevel(y, 60, 1, 30,
                                           prior_theta01_mean = 0,
                                           prior_theta01_prec = 1,
                                           chains = 3,
                                           init = list(list(prec_theta1 = 1),
                                                       list(prec_theta1 = 5),
                                                       list(prec_theta1 = 20)),
                                           verbose = FALSE, seed = 456)
  expect_s3_class(fits, "pdm_mcmc_list")
  expect_equal(vapply(fits, function(ch) attr(ch, "init")$prec_theta1, numeric(1)),
               c(1, 5, 20))

  expect_error(
    mcmc_probit_bernoulli_locallevel(y, 20, 1, 10, prior_theta01_mean = 0,
                                     prior_theta01_prec = 1, chains = 3,
                                     verbose = FALSE,
                                     init = list(prec_theta1 = 1)),
    "one per chain")
})


# --- The mixture family -------------------------------------------------------
# The last three samplers, and the only ones where `resolve_init()` needs its
# optional arguments: the component parameters interleave with the states in the
# driver's draw order, and the two components are identified only up to their
# order.

mix_data <- function(seed = 31, n = 60) {
  set.seed(seed)
  th <- cumsum(c(0, rnorm(n, sd = 0.1)))[-1]
  z  <- rbinom(n, 1, plogis(th))
  list(y = ifelse(z == 1, rnorm(n, 2, 0.5), rnorm(n, -2, 0.5)), z = z)
}


test_that("resolve_init draws in the order given, not in the layout order", {
  # The mixtures draw mu_1, prec_1, mu_2, prec_2 and only then the states, so
  # `order` has to be able to interleave the two lists. The layout of `values`
  # stays states-then-precisions-then-auxiliaries regardless, which is what lets
  # each C driver read three contiguous runs.
  states <- list(mu_1 = list(mean = -1, prec = 1),
                 mu_2 = list(mean =  1, prec = 1),
                 theta_01 = list(mean = 0, prec = 1))
  precs  <- list(prec_1 = gamma_prior(2, 1), prec_2 = gamma_prior(2, 1),
                 prec_theta1 = gamma_prior(2, 1))
  ord    <- c("mu_1", "prec_1", "mu_2", "prec_2", "theta_01", "prec_theta1")

  set.seed(77)
  got <- resolve_init(NULL, states, precs, order = ord)

  set.seed(77)
  mu_1   <- rnorm(1, -1, 1)
  prec_1 <- rgamma(1, shape = 2, scale = 1)
  mu_2   <- rnorm(1,  1, 1)
  prec_2 <- rgamma(1, shape = 2, scale = 1)
  th_01  <- rnorm(1,  0, 1)
  prec_t <- rgamma(1, shape = 2, scale = 1)

  expect_equal(got$init, list(mu_1 = mu_1, mu_2 = mu_2, theta_01 = th_01,
                              prec_1 = prec_1, prec_2 = prec_2,
                              prec_theta1 = prec_t))
  # layout: the three states, the three precisions, then a zero auxiliary each
  expect_equal(got$values,
               c(mu_1, mu_2, th_01, prec_1, prec_2, prec_t, 0, 0, 0))
})


test_that("drawn components out of order are relabelled, supplied ones are refused", {
  states <- list(mu_1 = list(mean = 10, prec = 1e6),   # forces mu_1 > mu_2
                 mu_2 = list(mean = -10, prec = 1e6))
  precs  <- list(prec_1 = gamma_prior(2, 1), prec_2 = gamma_prior(5, 1))
  oc     <- list(states = c("mu_1", "mu_2"), precs = c("prec_1", "prec_2"))
  ord    <- c("mu_1", "prec_1", "mu_2", "prec_2")

  # Both drawn: relabelling them is our own bookkeeping, as the C driver did.
  set.seed(5)
  got <- resolve_init(NULL, states, precs, order = ord, ordered_components = oc)
  expect_lte(got$init$mu_1, got$init$mu_2)
  expect_equal(got$init$mu_1, -10, tolerance = 0.01)
  # the precisions travelled with their means: prec_2 was drawn Gamma(5,1) and
  # belongs to the component that is now first
  expect_equal(got$init$prec_1, got$values[[3]])

  # Supplied: silently moving the caller's value into the other component is the
  # same failure mode as ignoring a misspelled name, so it errors.
  expect_error(
    resolve_init(list(mu_1 = 3, mu_2 = -3), states, precs,
                 order = ord, ordered_components = oc),
    "requires `mu_1` <= `mu_2`")
  # Also when only one was supplied and the draw put the pair out of order.
  expect_error(
    resolve_init(list(mu_1 = 1e6), states, precs,
                 order = ord, ordered_components = oc),
    "Supply both, in order")
})


test_that("init reaches the mixture samplers and is recorded", {
  d <- mix_data()

  f <- mcmc_normal_mixture_locallevel(d$y, 40, 2, 25,
                                      prior_theta01_mean = 0,
                                      prior_theta01_prec = 1,
                                      link = "logit",
                                      init = list(mu_1 = -2.5, mu_2 = 2.5,
                                                  prec_theta1 = 9),
                                      verbose = FALSE, seed = 456)
  rec <- attr(f, "init")
  expect_equal(rec$mu_1, -2.5)
  expect_equal(rec$mu_2, 2.5)
  expect_equal(rec$prec_theta1, 9)
  # what was left out is drawn and reported as the value used
  expect_true(all(c("prec_1", "prec_2", "theta_01") %in% names(rec)))

  g <- mcmc_normal_mixture_locallevel(d$y, 40, 2, 25,
                                      prior_theta01_mean = 0,
                                      prior_theta01_prec = 1,
                                      link = "logit",
                                      verbose = FALSE, seed = 456)
  expect_false(identical(f$mu_1, g$mu_1))
})


test_that("the mixture vocabulary invariant holds, and z and alpha stay out", {
  d <- mix_data()

  labels <- c(mu_1 = "mu_1", mu_2 = "mu_2", prec_1 = "phi_1", prec_2 = "phi_2",
              theta_01 = "theta_01", theta_02 = "theta_02", theta_03 = "theta_03",
              prec_theta1 = "W_1^{-1}", prec_theta2 = "W_2^{-1}",
              prec_theta3 = "W_3^{-1}")

  th0 <- list(prior_theta01_mean = 0, prior_theta01_prec = 1)
  th2 <- c(th0, list(prior_theta02_mean = 0, prior_theta02_prec = 1))
  th3 <- c(th2, list(prior_theta03_mean = 0, prior_theta03_prec = 1))

  cases <- list(list(mcmc_normal_mixture_locallevel,        th0),
                list(mcmc_normal_mixture_localtrend,        th2),
                list(mcmc_normal_mixture_localacceleration, th3))

  for (case in cases) {
    f     <- case[[1]]
    extra <- c(case[[2]], list(link = "logit", verbose = FALSE))

    fit <- do.call(f, c(list(d$y, 40, 2, 25), extra))
    msg <- tryCatch(do.call(f, c(list(d$y, 40, 2, 25), extra,
                                 list(init = list(nope = 1)))),
                    error = function(e) conditionMessage(e))
    accepted <- strsplit(sub(".*Valid names for this model are: ", "", msg), ", ")[[1]]

    scalars <- names(fit)[!vapply(fit, is.matrix, logical(1L))]
    expect_equal(sort(accepted), sort(scalars))

    # z is a per-observation vector and the source of between-chain dispersion;
    # alpha is derived from the trajectory. Both are matrix components, so the
    # scalar rule already excludes them -- this asserts it stays that way.
    expect_false("z" %in% accepted)
    expect_false("alpha" %in% accepted)

    for (nm in accepted) {
      expect_equal(true_value_for(labels[[nm]], stats::setNames(list(42), nm)), 42)
    }
  }
})


test_that("the mixture weight starts at link(theta_01), not at 0.5", {
  # The mixtures read alpha before writing it -- step 2 draws z | y, alpha, mu,
  # phi -- so the starting weight is live, unlike in the link families where it
  # was dead code. It is now the value the model implies for a trajectory flat at
  # theta_01 rather than a fixed 0.5. See docs/starting-values.md for the
  # measurements behind the change.
  #
  # burnin = 1 with thinning = 1 makes the first retained sample iteration 1,
  # whose z was drawn from the *initial* alpha. Step 2 does not see theta_01 at
  # all: it uses y, alpha, mu and phi, and mu/phi come from the initial z, which
  # is Bernoulli(0.5) from the same seed in every run below. So theta_01 can
  # reach that first z only through alpha, which is what isolates the change.
  d <- mix_data()
  first_z <- function(t01) {
    fit <- mcmc_normal_mixture_locallevel(d$y, 1, 1, 2,
                                          prior_theta01_mean = 0,
                                          prior_theta01_prec = 1,
                                          link = "logit",
                                          init = list(theta_01 = t01),
                                          verbose = FALSE, seed = 1)
    mean(fit$z[1, ])
  }

  # plogis(-6) = 0.0025 and plogis(6) = 0.9975, so a live starting alpha drives
  # that first z to the corresponding extreme. Under a fixed 0.5 the two would
  # be identical, and neither would be extreme.
  expect_equal(first_z(-6), 0)
  expect_equal(first_z(6), 1)
  # and the neutral point still lands strictly between them
  mid <- first_z(0)
  expect_gt(mid, 0)
  expect_lt(mid, 1)
})
