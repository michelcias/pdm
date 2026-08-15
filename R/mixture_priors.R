#' Data-scaled default rate for a mixture component precision
#'
#' Internal helper shared by the three `mcmc_normal_mixture_*()` wrappers.
#' Implements the Richardson & Green (1997) scaling for the Gamma prior on a
#' component precision \eqn{\phi_k}.
#'
#' Their specification is \eqn{\phi_k \sim \mathrm{Gamma}(\alpha, \beta)} with
#' \eqn{\alpha = 2} and a hyperprior \eqn{\beta \sim \mathrm{Gamma}(g, h)},
#' \eqn{h = 100g/(\alpha R^2)}, where \eqn{R} is the range of the data. `pdm`
#' has no hyperprior on the rate, so \eqn{\beta} is fixed at the mean of that
#' hyperprior, \eqn{E(\beta) = g/h = \alpha R^2 / 100}. Written in terms of the
#' resolved shape so that a user who changes `shape` keeps the same relationship.
#'
#' Note what this does *not* reproduce: Richardson & Green treat \eqn{\beta} as
#' random, and collapsing it to a point discards that layer. Implementing it
#' properly would need a sampling step for \eqn{\beta} in the C code.
#'
#' The scaling is by the **range**, not the variance, because that is what
#' Richardson & Green use and because `pdm` shares their *independent* prior on
#' \eqn{(\mu_k, \phi_k)} — the component mean's prior precision is not
#' multiplied by \eqn{\phi_k}. The variance-based figure of Nobile (2007)
#' belongs to a conditional Normal-Gamma structure that this package does not
#' have.
#'
#' @param y Numeric vector of observations.
#' @param shape The resolved Gamma shape \eqn{\alpha} for that component.
#'
#' @return Single numeric, the Gamma rate.
#'
#' @keywords internal
#' @noRd
rg_component_rate <- function(y, shape) {
  shape * diff(range(y))^2 / 100
}


#' Check the component-mean priors against the ordering constraint
#'
#' Internal helper shared by the three `mcmc_normal_mixture_*()` wrappers. The
#' mixture samplers identify the components by enforcing \eqn{\mu_1 < \mu_2},
#' relabelling after each draw, so component 1 is the lower one by construction.
#' A prior specification that places component 1 above component 2 asks for the
#' opposite.
#'
#' The sampler does not fail on such a specification: it simply relabels on
#' nearly every iteration — measured at 92.5% of iterations on a test fit, against
#' 0% when the priors agree with the constraint — and nothing in the returned
#' object reveals it. Hence a warning rather than an error: the run is still a
#' valid draw from the constrained posterior, it is just almost certainly not
#' the model the user meant to write.
#'
#' Equal means are left alone. That is the exchangeable specification, which is
#' the case the ordering constraint is designed for.
#'
#' @param mu01_mean,mu02_mean The resolved prior means for \eqn{\mu_1} and
#'   \eqn{\mu_2}, after the `NULL` defaults have been filled in from the
#'   quantiles of `y`.
#'
#' @return `NULL`, invisibly; called for the warning.
#'
#' @keywords internal
#' @noRd
check_mixture_mu_priors <- function(mu01_mean, mu02_mean) {
  if (isTRUE(mu01_mean > mu02_mean)) {
    warning(
      "`prior_mu01_mean` (", format(mu01_mean, digits = 4),
      ") is greater than `prior_mu02_mean` (", format(mu02_mean, digits = 4),
      "), but the sampler identifies the components by enforcing mu_1 < mu_2 ",
      "and relabels them to do so. Swap the two priors if component 1 is meant ",
      "to be the lower one.",
      call. = FALSE
    )
  }
  invisible(NULL)
}


#' Data-scaled default rate for a Poisson mixture component
#'
#' Internal helper shared by the three `mcmc_poisson_mixture_*()` wrappers. The
#' component rates take a conjugate \eqn{\mathrm{Gamma}(a, b)} prior, whose mean
#' is \eqn{a/b}; this returns the rate \eqn{b} that centres it on a target, so
#' the wrapper can express the default as "put component 1 near the lower
#' quartile of the counts and component 2 near the upper one" and leave the
#' shape free to control how tightly.
#'
#' This is the counterpart of `rg_component_rate()` for the Gaussian family, but
#' it is not the same construction and should not be read as one. Richardson &
#' Green's range scaling exists to keep a *precision* out of the degenerate
#' region where a component collapses onto a few observations; a Poisson rate has
#' no such region, so the default here is doing the much simpler job of putting
#' the two components on the scale of the data, the same job
#' `prior_mu01_mean = quantile(y, 0.25)` does in the Gaussian mixture.
#'
#' @param target The prior mean wanted for that component, on the count scale.
#' @param shape The resolved Gamma shape \eqn{a} for that component.
#'
#' @return Single numeric, the Gamma rate \eqn{b = a/\text{target}}.
#'
#' @keywords internal
#' @noRd
poisson_component_rate <- function(target, shape) {
  shape / target
}


#' Default component targets from the observed counts
#'
#' The quartiles of `y`, floored away from zero. The floor is not cosmetic: a
#' count series with many zeros has a lower quartile of exactly 0, and a target
#' of 0 sends the Gamma rate to infinity. Half a count is the smallest target
#' that keeps the prior proper while staying below any observable value.
#'
#' @param y Numeric vector of observed counts.
#'
#' @return Numeric vector of length 2, the targets for components 1 and 2.
#'
#' @keywords internal
#' @noRd
poisson_component_targets <- function(y) {
  targets <- as.numeric(quantile(y, c(0.25, 0.75)))
  pmax(targets, 0.5)
}


#' Check the component-rate priors against the ordering constraint
#'
#' The Poisson mixture identifies its components by enforcing
#' \eqn{\lambda_1 < \lambda_2}, relabelling after each draw. Priors that place
#' component 1 above component 2 ask for the opposite, and the run is then a
#' valid draw from the constrained posterior that is almost certainly not the
#' model the user wrote -- the same situation `check_mixture_mu_priors()`
#' handles for the Gaussian family, and warned about for the same reason.
#'
#' The comparison is between prior *means*, `shape/rate`, rather than between
#' any single hyperparameter: two components can share a shape and differ only
#' in rate, or the reverse, and neither pair by itself says which component sits
#' lower.
#'
#' @param shape_01,rate_01 The resolved Gamma hyperparameters for \eqn{\lambda_1}.
#' @param shape_02,rate_02 The resolved Gamma hyperparameters for \eqn{\lambda_2}.
#'
#' @return `NULL`, invisibly; called for the warning.
#'
#' @keywords internal
#' @noRd
check_mixture_lambda_priors <- function(shape_01, rate_01, shape_02, rate_02) {
  mean_01 <- shape_01 / rate_01
  mean_02 <- shape_02 / rate_02
  if (isTRUE(mean_01 > mean_02)) {
    warning(
      "the prior mean for `lambda_1` (", format(mean_01, digits = 4),
      ") is greater than the prior mean for `lambda_2` (",
      format(mean_02, digits = 4),
      "), but the sampler identifies the components by enforcing ",
      "lambda_1 < lambda_2 and relabels them to do so. Swap the two priors if ",
      "component 1 is meant to be the low-rate one.",
      call. = FALSE
    )
  }
  invisible(NULL)
}
