#' Highest Posterior Density Interval (HPDI)
#'
#' Computes the Highest Posterior Density Interval (HPDI) of MCMC samples: the
#' shortest contiguous interval that contains a proportion `prob` of the
#' draws. Unlike an equal-tailed (quantile) interval, the HPDI is the shortest
#' such interval and may be asymmetric for skewed posteriors.
#'
#' The computation is performed in C: each chain is sorted once and, over all
#' windows of fixed span \eqn{m = \lfloor prob \cdot n \rfloor}, the window with
#' the smallest width \eqn{x_{(i + m)} - x_{(i)}} is selected.
#'
#' @param data A numeric vector (a single chain of length `n`) or a numeric
#'   matrix in which each row is one MCMC sample and each column an independent
#'   chain (for example, one column per time point of a latent state
#'   trajectory). This matches the orientation of the `theta_1` component
#'   returned by the `mcmc_*` samplers. Values must be finite (no
#'   `NA`/`NaN`).
#' @param prob Probability mass contained in the interval; a single value
#'   strictly between 0 and 1. Defaults to `0.9`.
#'
#' @return For a vector input, a named numeric vector of length 2 with elements
#'   `lower` and `upper`. For a matrix input, a numeric matrix with one
#'   row per column of `data` and columns `lower` and `upper`.
#'
#' @examples
#' set.seed(1)
#'
#' # Single chain (e.g. a scalar parameter such as theta_01)
#' chain <- rgamma(2000, shape = 2, rate = 1)  # skewed posterior
#' hpdi(chain, prob = 0.9)
#'
#' # Several chains at once (e.g. theta_1: rows = samples, columns = time)
#' draws <- matrix(rnorm(2000 * 5), nrow = 2000, ncol = 5)
#' hpdi(draws, prob = 0.95)
#'
#' @export
hpdi <- function(data, prob = 0.9) {
  if (!is.numeric(data)) {
    stop("'data' must be a numeric vector or matrix.", call. = FALSE)
  }
  if (length(prob) != 1L || !is.finite(prob) || prob <= 0 || prob >= 1) {
    stop("'prob' must be a single value strictly between 0 and 1.",
         call. = FALSE)
  }

  # Ensure REALSXP without dropping the dim attribute of a matrix.
  storage.mode(data) <- "double"

  .Call("_pdm_C_hpdi", data, as.numeric(prob))
}
