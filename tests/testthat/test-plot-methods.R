# Test suite for plot methods
# Tests visualization functions for normal_mixture_localtrend, plus the
# true_values overlay across the normal and binomial families
# Author: Michel Helcias (michelcias)
# Date: 2026-07-25

# Helper function (reuse from test-print-methods.R or recreate here)
create_mock_object <- function(n_draws = 100, n_obs = 50, seed = 123) {
  set.seed(seed)

  mock_obj <- list(
    mu_1 = rnorm(n_draws, mean = 0, sd = 0.5),
    mu_2 = rnorm(n_draws, mean = 2, sd = 0.5),
    prec_1 = rgamma(n_draws, shape = 2, rate = 1),
    prec_2 = rgamma(n_draws, shape = 2, rate = 1),
    theta_1 = matrix(rnorm(n_draws * n_obs), nrow = n_draws, ncol = n_obs),
    theta_2 = matrix(rnorm(n_draws * n_obs, sd = 0.5), nrow = n_draws, ncol = n_obs),
    theta_01 = rnorm(n_draws, mean = 0, sd = 1),
    theta_02 = rnorm(n_draws, mean = 0, sd = 0.5),
    prec_theta1 = rgamma(n_draws, shape = 5, rate = 1),
    prec_theta2 = rgamma(n_draws, shape = 10, rate = 1),
    alpha = matrix(runif(n_draws * n_obs, 0.2, 0.8), nrow = n_draws, ncol = n_obs),
    z = matrix(rbinom(n_draws * n_obs, 1, 0.5), nrow = n_draws, ncol = n_obs)
  )

  attr(mock_obj, "link") <- "logit"
  attr(mock_obj, "n_obs") <- n_obs
  attr(mock_obj, "n_draws") <- n_draws
  attr(mock_obj, "burnin") <- 500
  attr(mock_obj, "thinning") <- 10
  attr(mock_obj, "model_type") <- "localtrend"
  attr(mock_obj, "y") <- rnorm(n_obs)

  class(mock_obj) <- c("normal_mixture_localtrend", "pdm_mcmc", "list")

  return(mock_obj)
}


# ============================================================================
# Tests for Base R Graphics
# ============================================================================

test_that("plot() runs without error with base graphics", {
  mock_obj <- create_mock_object()

  # Suppress plotting to NULL device
  pdf(NULL)
  on.exit(dev.off())

  expect_no_error(plot(mock_obj, type = "mcmc", engine = "base"))
  expect_no_error(plot(mock_obj, type = "params", engine = "base"))
  expect_no_error(plot(mock_obj, type = "states", engine = "base"))
  expect_no_error(plot(mock_obj, type = "alpha", engine = "base"))
})

test_that("plot() type='mcmc' accepts which argument", {
  mock_obj <- create_mock_object()

  pdf(NULL)
  on.exit(dev.off())

  # Single parameter
  expect_no_error(plot(mock_obj, type = "mcmc", which = 1, engine = "base"))

  # Multiple parameters
  expect_no_error(plot(mock_obj, type = "mcmc", which = 1:3, engine = "base"))

  # All 8 parameters
  expect_no_error(plot(mock_obj, type = "mcmc", which = 1:8, engine = "base"))
})

test_that("plot() validates which argument for mcmc type", {
  mock_obj <- create_mock_object()

  pdf(NULL)
  on.exit(dev.off())

  # Invalid: which out of range
  expect_error(
    plot(mock_obj, type = "mcmc", which = 0, engine = "base"),
    "`which` must be between 1 and 8"
  )

  expect_error(
    plot(mock_obj, type = "mcmc", which = 9, engine = "base"),
    "`which` must be between 1 and 8"
  )

  expect_error(
    plot(mock_obj, type = "mcmc", which = c(1, 10), engine = "base"),
    "`which` must be between 1 and 8"
  )
})

test_that("plot() validates type argument", {
  mock_obj <- create_mock_object()

  pdf(NULL)
  on.exit(dev.off())

  expect_error(plot(mock_obj, type = "invalid"), "'arg' should be one of")
})

test_that("plot() validates engine argument", {
  skip(paste0("The 'engine' argument (base vs. ggplot2) is not yet ",
              "implemented; the ggplot2 backend in plot_utils_ggplot.R is ",
              "still a prototype. Re-enable once engine selection is wired ",
              "into the plot methods."))

  mock_obj <- create_mock_object()

  pdf(NULL)
  on.exit(dev.off())

  expect_error(plot(mock_obj, engine = "invalid"), "'arg' should be one of")
})

test_that("plot() respects which argument for different types", {
  mock_obj <- create_mock_object()

  pdf(NULL)
  on.exit(dev.off())

  # Should only plot specified subplots
  expect_no_error(plot(mock_obj, type = "mcmc", which = c(1, 3), engine = "base"))
  expect_no_error(plot(mock_obj, type = "params", which = 1:2, engine = "base"))
  expect_no_error(plot(mock_obj, type = "states", which = c(1, 2), engine = "base"))
})

test_that("plot() returns invisibly", {
  mock_obj <- create_mock_object()

  pdf(NULL)
  result <- withVisible(plot(mock_obj, type = "alpha", engine = "base"))
  dev.off()

  expect_false(result$visible)
  expect_identical(result$value, mock_obj)
})

test_that("plot() handles overlay_data argument", {
  mock_obj <- create_mock_object()

  pdf(NULL)
  on.exit(dev.off())

  # With data overlay
  expect_no_error(plot(mock_obj, type = "alpha", overlay_data = TRUE, engine = "base"))

  # Without data overlay
  expect_no_error(plot(mock_obj, type = "alpha", overlay_data = FALSE, engine = "base"))

  # Without data attribute
  mock_obj_no_data <- mock_obj
  attr(mock_obj_no_data, "y") <- NULL
  expect_no_error(plot(mock_obj_no_data, type = "alpha", overlay_data = TRUE, engine = "base"))
})

test_that("plot() type='all' generates multiple pages", {
  mock_obj <- create_mock_object()

  pdf(NULL)
  on.exit(dev.off())

  # Should generate 12 pages:
  # - 8 pages for individual MCMC diagnostics
  # - 1 page for mixture params
  # - 1 page for dynamic states
  # - 2 pages for mixture weights (alpha_t and z_t)
  expect_no_error(plot(mock_obj, type = "all", ask = FALSE, engine = "base"))
})


# ============================================================================
# Tests for the prototype ggplot2 backend (inst/prototype/)
# ============================================================================
#
# There is no `engine` argument: `529d837` removed it and `b1d1363` parked the
# ggplot2 backend in `inst/prototype/`, out of `R/` so that `R CMD check` does
# not analyse unreachable code (see inst/prototype/README.md).
#
# The five tests that used to sit here called `plot(obj, engine = "ggplot2")`.
# With no such argument the value fell into `...` and was silently ignored, so
# every one of them passed without touching ggplot2 at all -- and two of them
# asserted on a hexbin message the prototype does not emit. They announced
# themselves only through an unrelated warning, and once that was fixed in
# 0.14-0 they were silent, vacuous coverage.
#
# These exercise the prototype directly. Their job is to catch rot while the
# work is parked: the file must still parse, still expose its API, and still
# run against the current shape of a fitted object. That last point is the one
# with teeth -- `plot_dynamic_states_generic_ggplot()` calls the package
# internal `get_param_config()`, so a change there fails here rather than when
# someone resumes step 1 of the README.

# Source the prototype into an environment parented on the package namespace,
# so its calls to pdm internals resolve. Returns NULL when the file is not
# available (e.g. an install that dropped inst/), letting callers skip.
load_ggplot_prototype <- function() {
  path <- system.file("prototype", "plot_utils_ggplot.R", package = "pdm")
  if (!nzchar(path)) return(NULL)
  env <- new.env(parent = asNamespace("pdm"))
  sys.source(path, envir = env)
  env
}

test_that("the prototype ggplot2 backend still parses and exposes its API", {
  skip_if_not_installed("ggplot2")
  env <- load_ggplot_prototype()
  skip_if(is.null(env), "prototype source not available")

  # Pinned by name: the backend is unreachable from the package, so nothing
  # else would notice a helper being renamed or dropped.
  expect_setequal(ls(env), c(
    "plot_acceptance_rates_ggplot",       "plot_all_mixture_generic_ggplot",
    "plot_alpha_trajectory_ggplot",       "plot_bernoulli_alpha_ggplot",
    "plot_binomial_alpha_ggplot",         "plot_component_probabilities_ggplot",
    "plot_dynamic_states_generic_ggplot", "plot_mixture_params_ggplot",
    "plot_mixture_weights_ggplot",        "plot_param_diagnostics_ggplot"
  ))
  for (nm in ls(env)) expect_true(is.function(env[[nm]]), info = nm)
})

test_that("the prototype ggplot2 backend runs against a current fit object", {
  skip_if_not_installed("ggplot2")
  env <- load_ggplot_prototype()
  skip_if(is.null(env), "prototype source not available")

  mock_obj <- create_mock_object()

  pdf(NULL)
  on.exit(dev.off())

  # The one helper that returns its plot. Assert it is really a ggplot and
  # force the lazy build, rather than settling for "nothing errored" -- a
  # ggplot object constructs happily and only fails when it is evaluated.
  p <- env$plot_param_diagnostics_ggplot(mock_obj$mu_1, "mu_1", "mu[1]")
  expect_s3_class(p, "ggplot")
  expect_no_error(ggplot2::ggplot_build(p))

  # The rest print and return invisibly, so executing them is what can be
  # checked. It is still worth checking: each consumes components of the
  # fitted object and would break if its shape changed.
  expect_no_error(env$plot_mixture_params_ggplot(
    mock_obj$mu_1, mock_obj$mu_2, mock_obj$prec_1, mock_obj$prec_2))
  expect_no_error(env$plot_alpha_trajectory_ggplot(mock_obj$alpha))
  expect_no_error(env$plot_mixture_weights_ggplot(mock_obj$alpha, mock_obj$z))
  expect_no_error(env$plot_component_probabilities_ggplot(mock_obj$z))

  # Reaches get_param_config(); this is what ties the parked prototype to the
  # live package.
  expect_no_error(env$plot_dynamic_states_generic_ggplot(mock_obj))
})


# ============================================================================
# Integration Tests
# ============================================================================

test_that("plot() works with different object sizes", {
  # Small object
  small_obj <- create_mock_object(n_draws = 10, n_obs = 5)

  pdf(NULL)
  expect_no_error(plot(small_obj, type = "alpha", engine = "base"))
  dev.off()

  # Large object
  large_obj <- create_mock_object(n_draws = 1000, n_obs = 200)

  pdf(NULL)
  expect_no_error(plot(large_obj, type = "alpha", engine = "base"))
  dev.off()
})

test_that("plot() handles extreme values", {
  mock_obj <- create_mock_object()

  # Add extreme values
  mock_obj$mu_1[1] <- 1e10
  mock_obj$alpha[, 1] <- c(rep(0, 50), rep(1, 50))

  pdf(NULL)
  expect_no_error(plot(mock_obj, type = "params", engine = "base"))
  expect_no_error(plot(mock_obj, type = "alpha", engine = "base"))
  dev.off()
})

test_that("all plot types work together", {
  mock_obj <- create_mock_object()

  pdf(NULL)
  on.exit(dev.off())

  # Test that all types can be called sequentially
  expect_no_error({
    plot(mock_obj, type = "mcmc", which = 1, engine = "base")
    plot(mock_obj, type = "params", engine = "base")
    plot(mock_obj, type = "states", engine = "base")
    plot(mock_obj, type = "alpha", engine = "base")
  })
})

test_that("plot() mcmc diagnostics covers all 8 parameters", {
  mock_obj <- create_mock_object()

  pdf(NULL)
  on.exit(dev.off())

  # Test each parameter individually
  for (i in 1:8) {
    expect_no_error(
      plot(mock_obj, type = "mcmc", which = i, engine = "base")
    )
  }
})


# ============================================================================
# Edge Cases
# ============================================================================

test_that("plot() handles missing optional components", {
  mock_obj <- create_mock_object()

  # Remove optional components
  attr(mock_obj, "y") <- NULL

  pdf(NULL)
  expect_no_error(plot(mock_obj, type = "alpha", overlay_data = TRUE, engine = "base"))
  dev.off()
})

test_that("plot() method dispatch works correctly", {
  # Object with correct class uses our custom method
  mock_obj <- create_mock_object()
  expect_true(inherits(mock_obj, "normal_mixture_localtrend"))

  pdf(NULL)
  on.exit(dev.off())

  # Custom method should work
  expect_no_error(plot(mock_obj, type = "alpha", engine = "base"))

  # Object WITHOUT our class uses plot.default (different behavior, but no error)
  plain_list <- list(x = 1:10, y = 1:10)
  expect_false(inherits(plain_list, "normal_mixture_localtrend"))

  # plot.default handles generic lists
  expect_no_error(plot(plain_list))
})

test_that("plot() respects par() settings", {
  mock_obj <- create_mock_object()

  pdf(NULL)
  on.exit(dev.off())

  # Save original par
  oldpar <- par(no.readonly = TRUE)

  # Set custom par
  par(mfrow = c(1, 1), mar = c(5, 4, 4, 2))

  # Plot should work
  expect_no_error(plot(mock_obj, type = "alpha", engine = "base"))

  # Par should be restored after plot (our functions use on.exit)
  # Note: This is implicit in our implementation

  # Restore par
  par(oldpar)
})

# test_that("plot() handles NA values in data", {
#   mock_obj <- create_mock_object()
#
#   # Add some NA values
#   mock_obj$mu_1[1:5] <- NA
#   mock_obj$alpha[1, 1:10] <- NA
#
#   pdf(NULL)
#   on.exit(dev.off())
#
#   # Should handle NAs gracefully (density() has na.rm)
#   expect_no_error(plot(mock_obj, type = "mcmc", which = 1, engine = "base"))
#   expect_no_error(plot(mock_obj, type = "alpha", engine = "base"))
# })


# ============================================================================
# Performance Tests (optional, can be slow)
# ============================================================================

test_that("plot() is reasonably fast for typical sizes", {
  skip_on_cran()
  skip_if_not(interactive())  # Only run interactively

  mock_obj <- create_mock_object(n_draws = 1000, n_obs = 100)

  pdf(NULL)
  on.exit(dev.off())

  # Should complete in reasonable time (< 10 seconds)
  expect_lt(
    system.time({
      plot(mock_obj, type = "mcmc", which = 1, engine = "base")
    })[["elapsed"]],
    10
  )
})


# ============================================================================
# Documentation Tests
# ============================================================================

test_that("plot() examples in documentation work", {
  skip_on_cran()

  mock_obj <- create_mock_object()

  pdf(NULL)
  on.exit(dev.off())

  # Examples from documentation
  expect_no_error(plot(mock_obj, type = "all", engine = "base", ask = FALSE))
  expect_no_error(plot(mock_obj, type = "mcmc", which = 1:2))
  expect_no_error(plot(mock_obj, type = "alpha"))
})


# ============================================================================
# Tests for the true_values overlay
# ============================================================================

# Every plot method documents a `true_values` list and shows worked examples of
# it, but nothing asserted that the overlay is actually drawn -- only that the
# call did not error, which it would not even if the argument were dropped on
# the floor. These pin it per (family, type).
#
# The assertion compares the rendered output with and without the overlay. It
# has to render every page: recordPlot() keeps only the current one, so a
# display-list comparison silently measures the last page and reports "not
# drawn" for the several types whose overlay lives on page 1. postscript() is
# plain text and writes all pages, so its size responds to any added ink.
render_size <- function(fit, type, true_values, ...) {
  f <- tempfile(fileext = ".ps")
  on.exit(unlink(f), add = TRUE)

  postscript(f, width = 7, height = 7)
  on.exit(if (!is.null(dev.list())) dev.off(), add = TRUE, after = FALSE)
  suppressWarnings(suppressMessages(capture.output(
    plot(fit, type = type, true_values = true_values, ask = FALSE, ...)
  )))
  dev.off()

  length(readLines(f, warn = FALSE))
}

expect_overlay_drawn <- function(fit, type, true_values, ...) {
  without <- render_size(fit, type, NULL, ...)
  with    <- render_size(fit, type, true_values, ...)
  expect_gt(with, without)
}


test_that("each true_values element is drawn on the page that documents it", {
  mock_obj <- create_mock_object()
  n_obs    <- attr(mock_obj, "n_obs")

  # One key at a time. Passing the whole list would only prove that *something*
  # was drawn: the params page overlays four elements, so dropping support for
  # one still changes the output and the assertion would pass regardless.
  # One element at a time, so that dropping support for a single one is caught.
  # Passing the whole list would only prove that *something* was drawn.
  cases <- list(
    list(type = "mcmc",   values = list(mu_1        = 0)),
    list(type = "mcmc",   values = list(mu_2        = 2)),
    list(type = "mcmc",   values = list(prec_1      = 2)),
    list(type = "mcmc",   values = list(prec_2      = 2)),
    list(type = "mcmc",   values = list(theta_01    = 0)),
    list(type = "mcmc",   values = list(theta_02    = 0)),
    list(type = "mcmc",   values = list(prec_theta1 = 5)),
    list(type = "mcmc",   values = list(prec_theta2 = 10)),
    list(type = "states", values = list(theta_1 = rep(0, n_obs))),
    list(type = "states", values = list(theta_2 = rep(0, n_obs))),
    list(type = "alpha",  values = list(alpha = rep(0.5, n_obs)))
  )

  for (case in cases) {
    expect_overlay_drawn(mock_obj, case$type, case$values)
  }

  # The params page is four bivariate scatterplots, and each marks the true
  # value as a single point -- so it needs both coordinates of its panel, and a
  # lone element correctly draws nothing. Isolating the panel with `which` is
  # what makes a missing coordinate visible: passing all four values at once
  # would still change the output if one panel had stopped honouring them.
  panels <- list(
    list(which = 1L, values = list(mu_1   = 0, mu_2   = 2)),
    list(which = 2L, values = list(mu_1   = 0, prec_1 = 2)),
    list(which = 3L, values = list(mu_2   = 2, prec_2 = 2)),
    list(which = 4L, values = list(prec_1 = 2, prec_2 = 2))
  )

  for (panel in panels) {
    expect_overlay_drawn(mock_obj, "params", panel$values, which = panel$which)
    # Half a pair marks nothing, by design.
    expect_equal(
      render_size(mock_obj, "params", panel$values[1], which = panel$which),
      render_size(mock_obj, "params", NULL, which = panel$which)
    )
  }
})


test_that("true_values is drawn for the normal and binomial families", {
  skip_on_cran()

  set.seed(3)
  n <- 50
  y <- cumsum(c(10, rnorm(n)))[-1] + rnorm(n, sd = sqrt(1 / 5))
  p <- plogis(cumsum(rnorm(n, sd = 0.2)))

  normal <- mcmc_normal_locallevel(
    y, 100, 1, 80,
    prior_theta01_mean = y[1], prior_theta01_prec = 1 / var(y),
    verbose = FALSE, seed = 1
  )
  tv_normal <- list(prec_y = 5, theta_01 = 10, prec_theta1 = 2,
                    theta_1 = rep(10, n))
  for (ty in c("mcmc", "states")) {
    expect_overlay_drawn(normal, ty, tv_normal)
  }

  binomial <- mcmc_binomial_locallevel(
    rbinom(n, 10, p), n_trials = 10, 100, 1, 80,
    prior_theta01_mean = 0, prior_theta01_prec = 1,
    verbose = FALSE, seed = 1
  )
  tv_binomial <- list(theta_01 = 0, prec_theta1 = 2,
                      theta_1 = qlogis(p), alpha = p)
  for (ty in c("mcmc", "states", "alpha")) {
    expect_overlay_drawn(binomial, ty, tv_binomial)
  }
})


test_that("an unrelated name in true_values draws nothing extra", {
  # The overlay is looked up by name, so a list holding only names this family
  # does not use must render identically to no list at all -- never matched by
  # position, and never an error.
  mock_obj <- create_mock_object()
  irrelevant <- list(not_a_parameter = 1, prec_y = 5)

  for (ty in c("states", "params", "alpha")) {
    expect_equal(render_size(mock_obj, ty, irrelevant),
                 render_size(mock_obj, ty, NULL))
  }
})
