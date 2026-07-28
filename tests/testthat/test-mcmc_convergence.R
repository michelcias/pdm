library(testthat)

# The single-chain mcmc_convergence() had no tests of its own: the multi-chain
# method is covered by test-mcmc_chains.R, and the only call here asserted a
# class. That gap is why removing the rounding from its table broke nothing --
# there was nothing to break. These pin the contract instead.

# The printed table is fixed-width with a header row, so a field is located by
# its column *name*. Locating it by position -- `fields[2]` -- is what these
# tests did until `Rhat_split` was inserted ahead of ESS and shifted every one
# of them. The contract under test is the display precision of a named column,
# so the lookup should not depend on how many columns precede it.
printed_field <- function(out, param, column) {
  header <- strsplit(trimws(grep("Parameter", out, value = TRUE)[1]), "\\s+")[[1]]
  row    <- strsplit(trimws(grep(param, out, value = TRUE)[1]), "\\s+")[[1]]
  row[match(column, header)]
}

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
  expect_equal(sum(grepl("^theta_1\\[t=", conv$table$Parameter)), 20L)
  expect_equal(nrow(mcmc_convergence(conv_fit(), theta_timepoints = NULL)$table), 3L)

  # Both methods of the generic resolve `theta_timepoints` through the same
  # helper, so the argument means one thing across the package. They were split
  # once -- three points here, twenty there, and a count accepted only by the
  # multi-chain method, so `theta_timepoints = 20` worked on one and errored on
  # the other. Compare the resolved grids, not the declared defaults.
  expect_equal(
    pdm:::resolve_timepoints(eval(formals(pdm:::mcmc_convergence.pdm_mcmc)$theta_timepoints)),
    pdm:::resolve_timepoints(eval(formals(pdm:::mcmc_convergence.pdm_mcmc_list)$theta_timepoints))
  )
  expect_equal(sum(grepl("^theta_1\\[t=",
                         mcmc_convergence(conv_fit(), theta_timepoints = 5)$table$Parameter)), 5L)
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
  out <- capture.output(print(conv))
  expect_match(printed_field(out, "V\\^\\{-1\\}", "ESS"), "^[0-9]+\\.[0-9]$")
  expect_match(printed_field(out, "V\\^\\{-1\\}", "Efficiency(%)"), "^[0-9]+\\.[0-9]$")
})


test_that("digits reaches the test statistic and not the sample sizes", {
  skip_if_not_installed("coda")
  conv <- mcmc_convergence(conv_fit())
  skip_if(!"Geweke_z" %in% names(conv$table), "coda diagnostics unavailable")

  expect_gt(max(abs(conv$table$Geweke_z - round(conv$table$Geweke_z, 4))), 0)

  out3 <- capture.output(print(conv, digits = 3))
  out5 <- capture.output(print(conv, digits = 5))

  three <- grep("V\\^\\{-1\\}", out3, value = TRUE)[1]
  five  <- grep("V\\^\\{-1\\}", out5, value = TRUE)[1]

  # The Geweke column widens with `digits` ...
  expect_false(identical(three, five))
  expect_match(printed_field(out5, "V\\^\\{-1\\}", "Geweke_z"), "^-?[0-9]+\\.[0-9]{5}$")

  # ... while the ESS column is unmoved by it.
  expect_identical(printed_field(out3, "V\\^\\{-1\\}", "ESS"),
                   printed_field(out5, "V\\^\\{-1\\}", "ESS"))
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
  expect_equal(names(bare$table), c("Parameter", "Rhat", "ESS", "Efficiency"))
})


test_that("Rhat is the Vehtari statistic, the same one the multi method reports", {
  fit  <- conv_fit()
  conv <- mcmc_convergence(fit)

  # Always present, first after Parameter, exactly as in the multi-chain table.
  expect_equal(names(conv$table)[1:2], c("Parameter", "Rhat"))

  # It must hold the statistic itself, not a near neighbour. Compare against
  # the draws the fit carries, for a scalar and for a state point.
  row <- match("W_1^{-1}", conv$table$Parameter)
  expect_equal(conv$table$Rhat[row], pdm:::rhat_single(fit$prec_theta1))

  last  <- nrow(conv$table)
  tidx  <- as.integer(sub(".*t=([0-9]+)\\].*", "\\1", conv$table$Parameter[last]))
  expect_equal(conv$table$Rhat[last], pdm:::rhat_single(fit$theta_1[, tidx]))

  expect_true(all(is.finite(conv$table$Rhat)))

  # And it is genuinely the rank-normalized statistic, not the plain split one.
  # On a heavy-tailed precision the two separate; a table reporting the plain
  # one under this name would pass every assertion above.
  expect_false(isTRUE(all.equal(conv$table$Rhat[row],
                                pdm:::rhat_split_single(fit$prec_theta1))))
})


test_that("Rhat matches posterior::rhat on the same draws", {
  skip_if_not_installed("posterior")

  # The reference comparison, on the shipped path rather than on the internal
  # alone: what the table reports for one chain is what Stan would report.
  fit  <- conv_fit()
  conv <- mcmc_convergence(fit, theta_timepoints = NULL)

  for (nm in c("prec_y", "theta_01", "prec_theta1")) {
    lbl <- switch(nm, prec_y = "V^{-1}", theta_01 = "theta_01",
                  prec_theta1 = "W_1^{-1}")
    expect_equal(conv$table$Rhat[match(lbl, conv$table$Parameter)],
                 posterior::rhat(fit[[nm]]), tolerance = 1e-10, info = nm)
  }
})


test_that("show_rhat_split adds a column without moving any existing one", {
  # documentation-standards.md 3f: a show_* flag adds columns and never
  # redefines one. `Rhat` is the column that would be tempting to swap for the
  # friendlier statistic, and it must stay put -- as must `Overall`, which is
  # classified from neither.
  off <- mcmc_convergence(conv_fit())
  on  <- mcmc_convergence(conv_fit(), show_rhat_split = TRUE)

  expect_false("Rhat_split" %in% names(off$table))
  expect_identical(names(off$table), setdiff(names(on$table), "Rhat_split"))
  for (col in names(off$table)) {
    expect_identical(off$table[[col]], on$table[[col]], info = col)
  }

  # The two are different statistics, and the plain one is the more forgiving
  # here -- which is the reason it is not the reported column.
  row <- match("W_1^{-1}", on$table$Parameter)
  expect_lt(on$table$Rhat_split[row], on$table$Rhat[row])
})


test_that("the R-hat columns are stored unrounded and follow digits", {
  conv <- mcmc_convergence(conv_fit(), show_rhat_split = TRUE)

  # Same contract as every other numeric column: raw in the object, formatted
  # only on the way out.
  for (col in c("Rhat", "Rhat_split")) {
    expect_gt(max(abs(conv$table[[col]] - round(conv$table[[col]], 3)),
                  na.rm = TRUE), 0)
  }

  out3 <- capture.output(print(conv, digits = 3L))
  out6 <- capture.output(print(conv, digits = 6L))
  expect_match(printed_field(out3, "W_1\\^\\{-1\\}", "Rhat"), "^[0-9]+\\.[0-9]{3}$")
  expect_match(printed_field(out6, "W_1\\^\\{-1\\}", "Rhat"), "^[0-9]+\\.[0-9]{6}$")

  # The footer names the two limits that matter, so nobody reads the column as
  # a verdict or as a substitute for several chains.
  expect_match(paste(out3, collapse = "\n"), "does not enter Overall")
  expect_match(paste(out3, collapse = "\n"), "never left one mode")
})
