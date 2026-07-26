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
