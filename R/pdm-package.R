#' @details
#' \pkg{pdm} provides Gibbs/MCMC samplers for Bayesian dynamic models built from
#' a polynomial latent structure, spanning six observation families and three
#' latent-dynamic orders. Each sampler \code{mcmc_<family>_<order>()} returns a
#' fitted object with \code{plot()}, \code{print()} and \code{summary()} methods.
#'
#' \strong{Model catalog} (rows: observation family; columns: latent dynamics):
#' \tabular{llll}{
#'   \strong{Family} \tab \strong{Local level} \tab \strong{Local trend} \tab \strong{Local acceleration} \cr
#'   Normal \tab \code{\link{mcmc_normal_locallevel}} \tab \code{\link{mcmc_normal_localtrend}} \tab \code{\link{mcmc_normal_localacceleration}} \cr
#'   Normal mixture \tab \code{\link{mcmc_normal_mixture_locallevel}} \tab \code{\link{mcmc_normal_mixture_localtrend}} \tab \code{\link{mcmc_normal_mixture_localacceleration}} \cr
#'   Poisson \tab \code{\link{mcmc_poisson_locallevel}} \tab \code{\link{mcmc_poisson_localtrend}} \tab \code{\link{mcmc_poisson_localacceleration}} \cr
#'   Poisson mixture \tab \code{\link{mcmc_poisson_mixture_locallevel}} \tab \code{\link{mcmc_poisson_mixture_localtrend}} \tab \code{\link{mcmc_poisson_mixture_localacceleration}} \cr
#'   Binomial \tab \code{\link{mcmc_binomial_locallevel}} \tab \code{\link{mcmc_binomial_localtrend}} \tab \code{\link{mcmc_binomial_localacceleration}} \cr
#'   Probit-Bernoulli \tab \code{\link{mcmc_probit_bernoulli_locallevel}} \tab \code{\link{mcmc_probit_bernoulli_localtrend}} \tab \code{\link{mcmc_probit_bernoulli_localacceleration}} \cr
#' }
#'
#' \strong{Two ways a latent state can meet a count.} The Poisson and
#' Poisson-mixture rows answer different questions and neither contains the
#' other. In \code{mcmc_poisson_*()} the state drives the rate through a log
#' link, so one Poisson evolves through time. In
#' \code{mcmc_poisson_mixture_*()} the state drives the mixture \emph{weight},
#' so two Poissons of fixed rate trade places -- the shape to reach for when the
#' counts switch between regimes rather than drift.
#'
#' Convergence diagnostics across fitted models are available via
#' \code{\link{mcmc_convergence}}. Every sampler additionally accepts
#' `chains > 1`, which runs several independent chains — each from its own
#' starting values drawn from the priors — and enables the rank-normalized
#' split-\eqn{\hat{R}} reported by
#' \code{\link{mcmc_convergence.pdm_mcmc_list}}.
#'
#' @keywords internal
"_PACKAGE"

## usethis namespace: start
#' @useDynLib pdm, .registration = TRUE
#' @importFrom stats acf density fft median nextn qnorm quantile rgamma rnorm sd var
#' @importFrom graphics abline axTicks axis grid legend lines mtext par points polygon segments title
#' @importFrom utils tail
## usethis namespace: end
NULL
