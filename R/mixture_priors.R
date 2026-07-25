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
