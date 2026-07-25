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
