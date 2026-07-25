#' Data-scaled Half-Cauchy scales for the Gaussian precision priors
#'
#' Internal helpers shared by the three `mcmc_normal_*()` wrappers, supplying
#' the default `scale` of the Half-t / Half-Cauchy prior on a standard
#' deviation when the caller leaves it `NULL`.
#'
#' @section Why the default is data-scaled:
#' A Gamma prior with fixed hyperparameters is stated in absolute units, so the
#' same series measured in metres and in centimetres gets a different prior and
#' a different fit. Measured on a constant-state series, the previous
#' `Gamma(0.01, 0.01)` default gave relative RMSE of 0.54, 0.10 and 0.14 for the
#' same data scaled by 0.01, 1 and 100; with both the innovation and observation
#' priors data-scaled the three agree to four decimals. The scales below are all
#' of the form (data-derived scale) x constant, so rescaling `y` rescales them
#' in step and the fit is equivariant.
#'
#' @section Why the k-th difference, deflated:
#' For a local-level model \eqn{\mathrm{var}(\Delta y) = W_1 + 2V}, so
#' \eqn{sd(\Delta y)} bounds the level innovation SD from above; half of it puts
#' the Half-Cauchy median at half the largest value the data could support.
#'
#' Taking the \eqn{k}-th difference for the \eqn{k}-th innovation does not
#' generalise on its own, because differencing amplifies observation noise: for
#' iid noise \eqn{\mathrm{var}(\Delta^k e) = V \sum_j \binom{k}{j}^2 =
#' V \binom{2k}{k}}, i.e. 2V, 6V and 20V for \eqn{k = 1, 2, 3}. Uncorrected, the
#' scales *grow* with order (0.73, 1.14, 2.12 on one reference series) precisely
#' where the innovations are smallest and shrinkage matters most. Deflating by
#' what pure noise alone would produce puts the orders on the same footing
#' (0.51, 0.47, 0.47).
#'
#' @param y Numeric vector of observations.
#' @param k Which innovation the scale is for: 1 for the level, 2 for the
#'   trend, 3 for the acceleration.
#'
#' @return Single positive numeric.
#'
#' @keywords internal
#' @noRd
innovation_prior_scale <- function(y, k) {
  sd(diff(y, differences = k)) / (2 * sqrt(choose(2 * k, k)))
}


#' Data-scaled Half-Cauchy scale for the observation standard deviation
#'
#' `sd(y)` is a generous upper bound: the observation noise cannot exceed the
#' total variation of the series by much. Scaling this prior as well as the
#' innovation ones is what makes the fit exactly invariant to the units of `y`
#' — with only the innovations scaled a residual dependence remains, which
#' looks fixed without being so.
#'
#' @param y Numeric vector of observations.
#'
#' @return Single positive numeric.
#'
#' @keywords internal
#' @noRd
observation_prior_scale <- function(y) {
  sd(y)
}


#' Keep an explicit Gamma specification working after the default type changed
#'
#' Until 0.4-0 the precision priors defaulted to `type = "gamma"`, so a caller
#' who passed `shape` and `rate` without naming a type got a Gamma. With
#' `"halfcauchy"` now the default, the same call would silently ignore both
#' values and fit something else. This restores `"gamma"` in exactly that case:
#' the type was not named *and* at least one Gamma hyperparameter was supplied.
#'
#' Naming the type explicitly always wins, so
#' `type = "halfcauchy", shape = 1` still errors on the unused `shape` through
#' the usual validation rather than being silently reinterpreted.
#'
#' The test is on whether the caller *supplied* the hyperparameters, not on
#' whether they are non-`NULL`. The Gaussian wrappers default them to `NULL`, so
#' the two coincide there, but the mixture wrappers default them to `0.01` — a
#' `NULL` test would read those defaults as a user's Gamma specification and
#' pin every mixture fit to a Gamma. Callers pass `!missing(...)`, which
#' `missing()` requires be evaluated in the frame that owns the argument.
#'
#' @param type The resolved type, after `match.arg()`.
#' @param type_user_set Whether the caller named the type.
#' @param hyper_user_set Whether the caller supplied `shape` or `rate`.
#'
#' @return The type to use.
#'
#' @keywords internal
#' @noRd
infer_gamma_from_hyperparams <- function(type, type_user_set, hyper_user_set) {
  if (!type_user_set && hyper_user_set) "gamma" else type
}
