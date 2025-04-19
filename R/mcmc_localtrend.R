#' Gibbs Sampler for a Local‐Level Dynamic Model
#'
#' Runs a Gibbs sampler for the local‐trend (local‐level) dynamic model
#' with one state dimension (p = 1).  Burn‐in and thinning are applied
#' so that exactly \code{n_chain} posterior samples are returned.
#'
#' @param y Numeric vector of observations (length = n).
#' @param burnin Integer ≥ 0, number of burn‐in iterations.
#' @param thinning Integer ≥ 1, thinning interval.
#' @param n_chain Integer ≥ 1, number of posterior samples to retain.
#' @param prior_theta01_mean Numeric, prior mean for the initial state θ₀₁.
#' @param prior_theta01_prec Numeric, prior precision (1/variance) for θ₀₁.
#' @param prior_prec1_shape Numeric, shape parameter of the Gamma prior for the innovation precision (1/W₁).
#' @param prior_prec1_rate  Numeric, rate  parameter of the Gamma prior for the innovation precision (1/W₁).
#' @param prior_prec_y_shape Numeric, shape parameter of the Gamma prior for the data precision (1/V).
#' @param prior_prec_y_rate  Numeric, rate  parameter of the Gamma prior for the data precision (1/V).
#'
#' @return A \code{list} with components:
#' \describe{
#'   \item{\code{theta_1}}{Numeric matrix [\code{n_chain} × n] of latent‐state samples.}
#'   \item{\code{theta_01}}{Numeric vector [length = \code{n_chain}] of initial‐state samples.}
#'   \item{\code{prec_1}}{Numeric vector [length = \code{n_chain}] of innovation precisions (1/W₁).}
#'   \item{\code{prec_y}}{Numeric vector [length = \code{n_chain}] of data precisions (1/V).}
#' }
#'
#' @examples
#' ## simulate data
#' set.seed(123)
#' n <- 100
#'
#' # true parameters
#' theta01_true <- 0
#' prec1_true   <- 5   # 1/W1
#' prec_y_true  <- 10  # 1/V
#'
#' # generate evolution and observation noise
#' u <- rnorm(n, 0, sqrt(1/prec1_true))
#' e <- rnorm(n, 0, sqrt(1/prec_y_true))
#'
#' # latent state and observations
#' theta1_true <- numeric(n)
#' theta1_true[1] <- theta01_true + u[1]
#' for (i in 2:n) {
#'   theta1_true[i] <- theta1_true[i - 1] + u[i]
#' }
#' y <- theta1_true + e
#'
#' ## run the Gibbs sampler
#' out <- mcmc_localtrend(
#'   y,
#'   burnin               = 200,
#'   thinning             = 10,
#'   n_chain              = 500,
#'   prior_theta01_mean   = 0,
#'   prior_theta01_prec   = 0.01,
#'   prior_prec1_shape    = 1,
#'   prior_prec1_rate     = 1,
#'   prior_prec_y_shape   = 1,
#'   prior_prec_y_rate    = 1
#' )
#'
#' ## summarized results
#' print(summary(out$theta_01))
#' matplot(t(out$theta_1)[1:5, ], type = "l",
#'         lty = 1, col = rainbow(5),
#'         xlab = "time", ylab = "latent state")
#'
#' @seealso \link{.Call}
#' @export
mcmc_localtrend <- function(y,
                            burnin,
                            thinning,
                            n_chain,
                            prior_theta01_mean,
                            prior_theta01_prec,
                            prior_prec1_shape,
                            prior_prec1_rate,
                            prior_prec_y_shape,
                            prior_prec_y_rate) {
  if (!is.numeric(y)) {
    stop("`y` must be a numeric vector")
  }
  if (!is.numeric(burnin) || length(burnin) != 1 || burnin < 0) {
    stop("`burnin` must be a single integer ≥ 0")
  }
  if (!is.numeric(thinning) || length(thinning) != 1 || thinning < 1) {
    stop("`thinning` must be a single integer ≥ 1")
  }
  if (!is.numeric(n_chain) || length(n_chain) != 1 || n_chain < 1) {
    stop("`n_chain` must be a single integer ≥ 1")
  }

  .Call(
    "_pdm_mcmc_localtrend",
    as.numeric(y),
    as.integer(burnin),
    as.integer(thinning),
    as.integer(n_chain),
    as.numeric(prior_theta01_mean),
    as.numeric(prior_theta01_prec),
    as.numeric(prior_prec1_shape),
    as.numeric(prior_prec1_rate),
    as.numeric(prior_prec_y_shape),
    as.numeric(prior_prec_y_rate)
  )
}
