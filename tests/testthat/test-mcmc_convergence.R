library(testthat)

# The single-chain mcmc_convergence() had no tests of its own: the multi-chain
# method is covered by test-mcmc_chains.R, and the only call here asserted a
# class. That gap is why removing the rounding from its table broke nothing --
# there was nothing to break. These pin the contract instead.

conv_fit <- function(n = 120, seed = 7) {
  set.seed(11)
  y <- rnorm(n)
  mcmc_normal_locallevel(
    y, burnin = 200, thinning = 2, n_draws = 300, seed = seed,
    prior_theta01_mean = y[1], prior_theta01_prec = 1 / var(y)
  )
}


test_that("mcmc_convergence() reports the expected table for one chain", {
  conv <- mcmc_convergence(conv_fit())

  expect_s3_class(conv, "pdm_convergence")
  expect_true(all(c("Parameter", "ESS", "Efficiency") %in% names(conv$table)))
  expect_true(all(c("V^{-1}", "theta_01", "W_1^{-1}") %in% conv$table$Parameter))

  expect_true(all(conv$table$ESS > 0))
  expect_true(all(conv$table$Efficiency > 0 & conv$table$Efficiency <= 100))

  # Scalars plus the default state screen, and NULL drops the states.
  expect_equal(sum(grepl("^theta_1\\[t=", conv$table$Parameter)), 3L)
  expect_equal(nrow(mcmc_convergence(conv_fit(), theta_timepoints = NULL)$table), 3L)
})


test_that("the stored table is unrounded, and the print method does the rounding", {
  conv <- mcmc_convergence(conv_fit())

  # The defect this pins: ESS and Efficiency used to be stored rounded to one
  # decimal while the print method formatted to three, so every printed value
  # ended in two zeros it had not earned. Storage must now carry more precision
  # than any display format asks for.
  expect_gt(max(abs(conv$table$ESS - round(conv$table$ESS, 1))), 0)
  expect_gt(max(abs(conv$table$Efficiency - round(conv$table$Efficiency, 1))), 0)

  # Efficiency is ESS as a percentage of the draws; exact, not to one decimal.
  expect_equal(conv$table$Efficiency, 100 * conv$table$ESS / conv$n_draws)

  # A count and a percentage print with one decimal whatever `digits` says.
  # Scoped to those two columns: the Geweke column on the same row carries
  # three decimals legitimately, so testing the whole line would be wrong.
  out    <- capture.output(print(conv))
  fields <- strsplit(trimws(grep("V\\^\\{-1\\}", out, value = TRUE)[1]), "\\s+")[[1]]
  expect_match(fields[2], "^[0-9]+\\.[0-9]$")   # ESS
  expect_match(fields[3], "^[0-9]+\\.[0-9]$")   # Efficiency(%)
})


test_that("digits reaches the test statistic and not the sample sizes", {
  skip_if_not_installed("coda")
  conv <- mcmc_convergence(conv_fit())
  skip_if(!"Geweke_z" %in% names(conv$table), "coda diagnostics unavailable")

  expect_gt(max(abs(conv$table$Geweke_z - round(conv$table$Geweke_z, 4))), 0)

  three <- grep("V\\^\\{-1\\}", capture.output(print(conv, digits = 3)), value = TRUE)[1]
  five  <- grep("V\\^\\{-1\\}", capture.output(print(conv, digits = 5)), value = TRUE)[1]

  # The Geweke column widens with `digits` ...
  expect_false(identical(three, five))
  expect_match(five, "-?[0-9]+\\.[0-9]{5}")

  # ... while the ESS column is unmoved by it.
  ess_of <- function(s) sub("^\\s*\\S+\\s+([0-9.]+).*$", "\\1", trimws(s))
  expect_identical(ess_of(three), ess_of(five))
})


test_that("the verdict columns are computed from unrounded values", {
  conv <- mcmc_convergence(conv_fit())
  skip_if(!"ESS_status" %in% names(conv$table), "status column disabled")

  # Guards a boundary bug that does not exist today and would be invisible if
  # introduced: classifying from a rounded Efficiency could push a value just
  # under a cut-off up into the better label.
  th <- conv$ess_thresholds
  expected <- ifelse(conv$table$Efficiency > th[1], "EXCELLENT",
              ifelse(conv$table$Efficiency > th[2], "GOOD",
              ifelse(conv$table$Efficiency > th[3], "ACCEPTABLE", "POOR")))
  expect_identical(conv$table$ESS_status, expected)

  # The check above only bites when some value lands within half a display
  # unit of a cut-off, which on an arbitrary fit it does not. So put one there
  # deliberately: pick a threshold that the exact Efficiency fails and its
  # one-decimal rounding would pass. If the classification ever starts reading
  # a rounded value, this is the assertion that notices.
  eff <- conv$table$Efficiency[1]
  skip_if(isTRUE(all.equal(eff, round(eff, 1))), "no fractional part to exploit")
  between <- mean(c(eff, round(eff, 1)))          # strictly between the two
  tuned <- mcmc_convergence(conv_fit(),
                            ess_thresholds = sort(c(between, 5, 1), decreasing = TRUE))
  expect_false(identical(tuned$table$ESS_status[1], "EXCELLENT"))
})


test_that("the show_* flags drop their columns", {
  bare <- mcmc_convergence(conv_fit(), show_ess_status = FALSE,
                           show_geweke = FALSE, show_heidel = FALSE,
                           show_overall = FALSE)
  expect_equal(names(bare$table), c("Parameter", "ESS", "Efficiency"))
})
