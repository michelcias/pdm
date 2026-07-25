#' @details
#' \pkg{pdm} provides Gibbs/MCMC samplers for Bayesian dynamic models built from
#' a polynomial latent structure, spanning five observation families and three
#' latent-dynamic orders. Each sampler \code{mcmc_<family>_<order>()} returns a
#' fitted object with \code{plot()}, \code{print()} and \code{summary()} methods.
#'
#' \strong{Model catalog} (rows: observation family; columns: latent dynamics):
#' \tabular{llll}{
#'   \strong{Family} \tab \strong{Local level} \tab \strong{Local trend} \tab \strong{Local acceleration} \cr
#'   Normal \tab \code{\link{mcmc_normal_locallevel}} \tab \code{\link{mcmc_normal_localtrend}} \tab \code{\link{mcmc_normal_localacceleration}} \cr
#'   Normal mixture \tab \code{\link{mcmc_normal_mixture_locallevel}} \tab \code{\link{mcmc_normal_mixture_localtrend}} \tab \code{\link{mcmc_normal_mixture_localacceleration}} \cr
#'   Poisson \tab \code{\link{mcmc_poisson_locallevel}} \tab \code{\link{mcmc_poisson_localtrend}} \tab \code{\link{mcmc_poisson_localacceleration}} \cr
#'   Binomial \tab \code{\link{mcmc_binomial_locallevel}} \tab \code{\link{mcmc_binomial_localtrend}} \tab \code{\link{mcmc_binomial_localacceleration}} \cr
#'   Probit-Bernoulli \tab \code{\link{mcmc_probit_bernoulli_locallevel}} \tab \code{\link{mcmc_probit_bernoulli_localtrend}} \tab \code{\link{mcmc_probit_bernoulli_localacceleration}} \cr
#' }
#'
#' Convergence diagnostics across fitted models are available via
#' \code{\link{mcmc_convergence}}. The Gaussian samplers additionally accept
#' `chains > 1`, which runs several independent chains and enables the
#' rank-normalized split-\eqn{\hat{R}} reported by
#' \code{\link{mcmc_convergence.pdm_mcmc_list}}.
#'
#' @keywords internal
"_PACKAGE"

## usethis namespace: start
#' @useDynLib pdm, .registration = TRUE
#' @importFrom stats acf density fft median nextn qnorm quantile sd var
#' @importFrom graphics abline axTicks axis grid legend lines mtext par points polygon segments title
#' @importFrom utils tail
## usethis namespace: end
NULL
