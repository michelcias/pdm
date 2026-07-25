#' Pointwise conditional log-likelihood
#'
#' @description
#' Generic for extracting the pointwise (per-observation) conditional
#' log-likelihood matrix from a fitted \pkg{pdm} model. This matrix is the
#' \emph{primitive} that model-comparison tooling is built on: feeding it to
#' `loo::waic()` or `loo::loo()` yields WAIC / PSIS-LOO, and the same matrix is
#' reweighted by leave-future-out cross-validation (LFO-CV) for honest
#' predictive comparison of time-series models.
#'
#' The observation model depends on the \emph{family} only, so all three latent
#' orders of a family (`locallevel`, `localtrend`, `localacceleration`) share a
#' method.
#'
#' @param object A fitted \pkg{pdm} model object.
#' @param ... Passed on to methods.
#'
#' @return A numeric matrix with one row per retained MCMC draw and one column
#'   per observation (an \eqn{S \times N} matrix). Entry \eqn{[s, t]} is
#'   \eqn{\log p(y_t \mid \theta^{(s)}, \phi^{(s)})}, the log density of
#'   observation \eqn{t} evaluated at the \eqn{s}-th posterior draw of the
#'   latent states \eqn{\theta} and parameters \eqn{\phi}. This orientation is
#'   the one expected by the \pkg{loo} package.
#'
#' @seealso
#'   \code{\link{log_lik.normal_locallevel}},
#'   \code{\link{log_lik.poisson_locallevel}},
#'   \code{\link{log_lik.binomial_locallevel}},
#'   \code{\link{log_lik.probit_bernoulli_locallevel}},
#'   \code{\link{log_lik.normal_mixture_locallevel}} for the per-family methods.
#'
#' @export
log_lik <- function(object, ...) {
  UseMethod("log_lik")
}


#' Retrieve the stored observations, or fail informatively
#'
#' @param object A fitted \pkg{pdm} model object.
#' @return The numeric data vector stored as attribute `y`.
#' @keywords internal
#' @noRd
loglik_stored_y <- function(object) {
  y <- attr(object, "y")
  if (is.null(y)) {
    stop("Cannot compute the log-likelihood: the original data are not stored ",
         "on this object (attribute 'y' is missing). Re-fit the model so that ",
         "`y` is retained.", call. = FALSE)
  }
  y
}


# ---------------------------------------------------------------------------
# Normal family: y_t | theta_{t,1}, V ~ Normal(theta_{t,1}, V), V^-1 = prec_y
# ---------------------------------------------------------------------------

#' @noRd
compute_loglik_normal <- function(object) {
  y       <- loglik_stored_y(object)
  n_chain <- attr(object, "n_chain")
  n_obs   <- attr(object, "n_obs")

  theta_1 <- object$theta_1   # [n_chain x n_obs]: rows = draws, columns = time
  prec_y  <- object$prec_y    # length n_chain: one observation precision per draw

  # Target: ll[s, t] = log dnorm(y_t; theta_1[s, t], sd_draw[s]). Column-major
  # recycling reuses sd_draw[s] within each row (leading dimension); y depends
  # on the column t, so it is expanded once, by row, into y_mat.
  sd_draw <- sqrt(1 / prec_y)
  y_mat   <- matrix(y, nrow = n_chain, ncol = n_obs, byrow = TRUE)

  ll <- stats::dnorm(y_mat, mean = theta_1, sd = sd_draw, log = TRUE)
  dim(ll) <- c(n_chain, n_obs)   # dnorm() drops the dim attribute; restore it
  ll
}

#' Pointwise conditional log-likelihood for normal dynamic models
#'
#' @description
#' Computes the pointwise conditional log-likelihood matrix for a fitted
#' Gaussian dynamic model
#' (\code{\link{mcmc_normal_locallevel}}, \code{\link{mcmc_normal_localtrend}},
#' \code{\link{mcmc_normal_localacceleration}}). The observation density is
#' \deqn{y_t \mid \theta_{t,1},\, V \;\sim\; \mathrm{Normal}\!\left(\theta_{t,1},\, V\right),}
#' with observation precision \eqn{V^{-1} =} \code{prec_y}, so
#' \deqn{\ell_{s,t} = \log \phi\!\left(y_t;\; \theta_{t,1}^{(s)},\; \sqrt{1 / \mathrm{prec\_y}^{(s)}}\right),}
#' where \eqn{\phi} is the normal density.
#'
#' @details
#' The returned quantity is the log-likelihood \strong{conditional on the latent
#' states}: it is evaluated at the sampled trajectories rather than integrating
#' them out. Passing it to `loo::loo()` gives a leave-one-observation-out
#' estimate \emph{conditional on the smoothed states}, which for time series is
#' an in-sample / weak cross-validation measure and tends to be optimistic. For
#' an honest one-step predictive assessment, use leave-future-out
#' cross-validation (LFO-CV), which reweights this same matrix by importance
#' sampling (Burkner, Gabry & Vehtari, 2020).
#'
#' The matrix is computed on demand from the stored draws and data; it is not
#' cached on the object, so it always reflects the current contents of
#' `object`.
#'
#' @param object An object of class `normal_locallevel`, `normal_localtrend`, or
#'   `normal_localacceleration`, the result of the corresponding `mcmc_*`
#'   sampler.
#' @param ... Currently unused.
#'
#' @return A numeric matrix of dimension `n_chain` \eqn{\times} `n_obs`
#'   (draws \eqn{\times} observations), suitable for direct use with
#'   `loo::waic()` and `loo::loo()`.
#'
#' @references
#' Burkner, P.-C., Gabry, J., & Vehtari, A. (2020). Approximate leave-future-out
#' cross-validation for Bayesian time series models. \emph{Journal of
#' Statistical Computation and Simulation}, 90(14), 2499-2523.
#'
#' Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model
#' evaluation using leave-one-out cross-validation and WAIC. \emph{Statistics
#' and Computing}, 27(5), 1413-1432.
#'
#' @examples
#' \donttest{
#' ## Simulate a Gaussian local-level series
#' n <- 200
#' set.seed(123)
#' u1 <- rnorm(n, sd = 1)
#' e  <- rnorm(n, sd = sqrt(1 / 5))
#' theta1_true <- cumsum(c(10, u1))[-1]
#' y <- theta1_true + e
#'
#' out <- mcmc_normal_locallevel(
#'   y,
#'   burnin             = 1000,
#'   thinning           = 10,
#'   n_chain            = 500,
#'   prior_theta01_mean = y[1],
#'   prior_theta01_prec = 1 / var(y),
#'   prior_prec1_shape  = 1e-2,
#'   prior_prec1_rate   = 1e-2,
#'   prior_prec_y_shape = 1e-2,
#'   prior_prec_y_rate  = 1e-2,
#'   seed               = 456
#' )
#'
#' ll <- log_lik(out)
#' dim(ll)   # c(n_chain, n_obs)
#'
#' ## The pointwise matrix feeds the loo package directly
#' ## (conditional-on-states; see Details):
#' if (requireNamespace("loo", quietly = TRUE)) {
#'   print(loo::waic(ll))
#' }
#' }
#'
#' @seealso
#'   \code{\link{mcmc_normal_locallevel}}, \code{\link{mcmc_normal_localtrend}},
#'   \code{\link{mcmc_normal_localacceleration}} (model generators);
#'   \code{\link{log_lik}} (generic).
#'
#' @rdname log_lik.normal_locallevel
#' @export
log_lik.normal_locallevel <- function(object, ...) compute_loglik_normal(object)

#' @rdname log_lik.normal_locallevel
#' @export
log_lik.normal_localtrend <- function(object, ...) compute_loglik_normal(object)

#' @rdname log_lik.normal_locallevel
#' @export
log_lik.normal_localacceleration <- function(object, ...) compute_loglik_normal(object)


# ---------------------------------------------------------------------------
# Poisson family: y_t ~ Poisson(alpha_t), alpha_t = exp(theta_{t,1})
# ---------------------------------------------------------------------------

#' @noRd
compute_loglik_poisson <- function(object) {
  y     <- loglik_stored_y(object)
  alpha <- object$alpha   # [n_chain x n_obs]: fitted rates exp(theta_{t,1})

  y_mat <- matrix(y, nrow = nrow(alpha), ncol = ncol(alpha), byrow = TRUE)
  ll    <- stats::dpois(y_mat, lambda = alpha, log = TRUE)
  dim(ll) <- dim(alpha)
  ll
}

#' Pointwise conditional log-likelihood for Poisson dynamic models
#'
#' @description
#' Computes the pointwise conditional log-likelihood matrix for a fitted Poisson
#' dynamic model
#' (\code{\link{mcmc_poisson_locallevel}}, \code{\link{mcmc_poisson_localtrend}},
#' \code{\link{mcmc_poisson_localacceleration}}). With the log link the
#' observation model is
#' \deqn{y_t \mid \theta_{t,1} \;\sim\; \mathrm{Poisson}\!\left(\alpha_t\right),
#'       \qquad \alpha_t = \exp(\theta_{t,1}),}
#' so \eqn{\ell_{s,t} = \log \mathrm{dpois}\!\left(y_t;\, \alpha_t^{(s)}\right)}.
#' The fitted rates \eqn{\alpha_t^{(s)}} are read directly from the stored
#' `alpha` component.
#'
#' @inherit log_lik.normal_locallevel details references return
#'
#' @param object An object of class `poisson_locallevel`, `poisson_localtrend`,
#'   or `poisson_localacceleration`.
#' @param ... Currently unused.
#'
#' @examples
#' \donttest{
#' set.seed(123)
#' n <- 200
#' alpha_true <- exp(sin(2 * pi * seq_len(n) / n) + 1)
#' y <- rpois(n, lambda = alpha_true)
#'
#' out <- mcmc_poisson_locallevel(
#'   y,
#'   burnin             = 1000,
#'   thinning           = 20,
#'   n_chain            = 500,
#'   prior_theta01_mean = 0,
#'   prior_theta01_prec = 1,
#'   prior_prec1_shape  = 100,
#'   prior_prec1_rate   = 1,
#'   verbose            = FALSE,
#'   seed               = 456
#' )
#'
#' ll <- log_lik(out)
#' dim(ll)   # c(n_chain, n_obs)
#'
#' ## The pointwise matrix feeds the loo package directly
#' ## (conditional-on-states; see Details):
#' if (requireNamespace("loo", quietly = TRUE)) {
#'   print(loo::waic(ll))
#' }
#' }
#'
#' @seealso
#'   \code{\link{mcmc_poisson_locallevel}}, \code{\link{mcmc_poisson_localtrend}},
#'   \code{\link{mcmc_poisson_localacceleration}} (model generators);
#'   \code{\link{log_lik}} (generic).
#'
#' @rdname log_lik.poisson_locallevel
#' @export
log_lik.poisson_locallevel <- function(object, ...) compute_loglik_poisson(object)

#' @rdname log_lik.poisson_locallevel
#' @export
log_lik.poisson_localtrend <- function(object, ...) compute_loglik_poisson(object)

#' @rdname log_lik.poisson_locallevel
#' @export
log_lik.poisson_localacceleration <- function(object, ...) compute_loglik_poisson(object)


# ---------------------------------------------------------------------------
# Binomial family: y_t ~ Binomial(n_trials, alpha_t), alpha_t = logit^-1(theta_{t,1})
# ---------------------------------------------------------------------------

#' @noRd
compute_loglik_binomial <- function(object) {
  y        <- loglik_stored_y(object)
  n_trials <- attr(object, "n_trials")
  alpha    <- object$alpha   # [n_chain x n_obs]: fitted success probabilities

  y_mat <- matrix(y, nrow = nrow(alpha), ncol = ncol(alpha), byrow = TRUE)
  ll    <- stats::dbinom(y_mat, size = n_trials, prob = alpha, log = TRUE)
  dim(ll) <- dim(alpha)
  ll
}

#' Pointwise conditional log-likelihood for binomial dynamic models
#'
#' @description
#' Computes the pointwise conditional log-likelihood matrix for a fitted
#' binomial dynamic model
#' (\code{\link{mcmc_binomial_locallevel}}, \code{\link{mcmc_binomial_localtrend}},
#' \code{\link{mcmc_binomial_localacceleration}}). With the logit link the
#' observation model is
#' \deqn{y_t \mid \theta_{t,1} \;\sim\; \mathrm{Binomial}\!\left(n_{\mathrm{trials}},\, \alpha_t\right),
#'       \qquad \alpha_t = \mathrm{logit}^{-1}(\theta_{t,1}),}
#' so \eqn{\ell_{s,t} = \log \mathrm{dbinom}\!\left(y_t;\, n_{\mathrm{trials}},\, \alpha_t^{(s)}\right)}.
#' The number of trials is read from the object's `n_trials` attribute and the
#' fitted probabilities \eqn{\alpha_t^{(s)}} from the stored `alpha` component.
#'
#' @inherit log_lik.normal_locallevel details references return
#'
#' @param object An object of class `binomial_locallevel`, `binomial_localtrend`,
#'   or `binomial_localacceleration`.
#' @param ... Currently unused.
#'
#' @examples
#' \donttest{
#' set.seed(123)
#' n <- 200
#' n_trials <- 20
#' alpha_true <- (sin(2 * pi * seq_len(n) / n) + 2) / 4
#' y <- rbinom(n, size = n_trials, prob = alpha_true)
#'
#' out <- mcmc_binomial_locallevel(
#'   y,
#'   n_trials           = n_trials,
#'   burnin             = 1000,
#'   thinning           = 20,
#'   n_chain            = 500,
#'   prior_theta01_mean = 0,
#'   prior_theta01_prec = 1,
#'   prior_prec1_shape  = 100,
#'   prior_prec1_rate   = 1,
#'   verbose            = FALSE,
#'   seed               = 456
#' )
#'
#' ll <- log_lik(out)
#' dim(ll)   # c(n_chain, n_obs)
#'
#' ## The pointwise matrix feeds the loo package directly
#' ## (conditional-on-states; see Details):
#' if (requireNamespace("loo", quietly = TRUE)) {
#'   print(loo::waic(ll))
#' }
#' }
#'
#' @seealso
#'   \code{\link{mcmc_binomial_locallevel}}, \code{\link{mcmc_binomial_localtrend}},
#'   \code{\link{mcmc_binomial_localacceleration}} (model generators);
#'   \code{\link{log_lik}} (generic).
#'
#' @rdname log_lik.binomial_locallevel
#' @export
log_lik.binomial_locallevel <- function(object, ...) compute_loglik_binomial(object)

#' @rdname log_lik.binomial_locallevel
#' @export
log_lik.binomial_localtrend <- function(object, ...) compute_loglik_binomial(object)

#' @rdname log_lik.binomial_locallevel
#' @export
log_lik.binomial_localacceleration <- function(object, ...) compute_loglik_binomial(object)


# ---------------------------------------------------------------------------
# Probit-Bernoulli family: y_t ~ Bernoulli(alpha_t), alpha_t = Phi(theta_{t,1})
# ---------------------------------------------------------------------------

#' @noRd
compute_loglik_probit_bernoulli <- function(object) {
  y     <- loglik_stored_y(object)
  alpha <- object$alpha   # [n_chain x n_obs]: fitted Bernoulli probabilities

  y_mat <- matrix(y, nrow = nrow(alpha), ncol = ncol(alpha), byrow = TRUE)
  ll    <- stats::dbinom(y_mat, size = 1, prob = alpha, log = TRUE)
  dim(ll) <- dim(alpha)
  ll
}

#' Pointwise conditional log-likelihood for probit-Bernoulli dynamic models
#'
#' @description
#' Computes the pointwise conditional log-likelihood matrix for a fitted
#' probit-Bernoulli dynamic model
#' (\code{\link{mcmc_probit_bernoulli_locallevel}},
#' \code{\link{mcmc_probit_bernoulli_localtrend}},
#' \code{\link{mcmc_probit_bernoulli_localacceleration}}). With the probit link
#' the observation model is
#' \deqn{y_t \mid \theta_{t,1} \;\sim\; \mathrm{Bernoulli}\!\left(\alpha_t\right),
#'       \qquad \alpha_t = \Phi(\theta_{t,1}),}
#' where \eqn{\Phi} is the standard normal CDF, so
#' \eqn{\ell_{s,t} = \log \mathrm{dbinom}\!\left(y_t;\, 1,\, \alpha_t^{(s)}\right)}.
#' The fitted probabilities \eqn{\alpha_t^{(s)}} are read directly from the
#' stored `alpha` component.
#'
#' @inherit log_lik.normal_locallevel details references return
#'
#' @param object An object of class `probit_bernoulli_locallevel`,
#'   `probit_bernoulli_localtrend`, or `probit_bernoulli_localacceleration`.
#' @param ... Currently unused.
#'
#' @examples
#' \donttest{
#' set.seed(123)
#' n <- 200
#' alpha_true <- (sin(2 * pi * seq_len(n) / n) + 2) / 4
#' y <- rbinom(n, size = 1, prob = alpha_true)
#'
#' out <- mcmc_probit_bernoulli_locallevel(
#'   y,
#'   burnin             = 1000,
#'   thinning           = 20,
#'   n_chain            = 500,
#'   prior_theta01_mean = 0,
#'   prior_theta01_prec = 1,
#'   prior_prec1_shape  = 100,
#'   prior_prec1_rate   = 1,
#'   verbose            = FALSE,
#'   seed               = 456
#' )
#'
#' ll <- log_lik(out)
#' dim(ll)   # c(n_chain, n_obs)
#'
#' ## The pointwise matrix feeds the loo package directly
#' ## (conditional-on-states; see Details):
#' if (requireNamespace("loo", quietly = TRUE)) {
#'   print(loo::waic(ll))
#' }
#' }
#'
#' @seealso
#'   \code{\link{mcmc_probit_bernoulli_locallevel}},
#'   \code{\link{mcmc_probit_bernoulli_localtrend}},
#'   \code{\link{mcmc_probit_bernoulli_localacceleration}} (model generators);
#'   \code{\link{log_lik}} (generic).
#'
#' @rdname log_lik.probit_bernoulli_locallevel
#' @export
log_lik.probit_bernoulli_locallevel <- function(object, ...) compute_loglik_probit_bernoulli(object)

#' @rdname log_lik.probit_bernoulli_locallevel
#' @export
log_lik.probit_bernoulli_localtrend <- function(object, ...) compute_loglik_probit_bernoulli(object)

#' @rdname log_lik.probit_bernoulli_locallevel
#' @export
log_lik.probit_bernoulli_localacceleration <- function(object, ...) compute_loglik_probit_bernoulli(object)


# ---------------------------------------------------------------------------
# Normal-mixture family (two components):
#   y_t | z_t ~ N(mu_{z_t}, 1 / prec_{z_t}), z_t ~ Bernoulli(alpha_t)
# The pointwise likelihood MARGINALISES the latent indicator z_t.
# ---------------------------------------------------------------------------

#' @noRd
compute_loglik_normal_mixture <- function(object) {
  y     <- loglik_stored_y(object)
  alpha <- object$alpha   # [n_chain x n_obs]: P(z_t = 1) = weight of component 2
  n_chain <- nrow(alpha)
  n_obs   <- ncol(alpha)

  y_mat <- matrix(y, nrow = n_chain, ncol = n_obs, byrow = TRUE)

  # Component parameters are per-draw scalars (length n_chain); they recycle
  # per row against the [n_chain x n_obs] matrices (leading dimension).
  logd1 <- stats::dnorm(y_mat, mean = object$mu_1, sd = sqrt(1 / object$prec_1), log = TRUE)
  logd2 <- stats::dnorm(y_mat, mean = object$mu_2, sd = sqrt(1 / object$prec_2), log = TRUE)
  dim(logd1) <- c(n_chain, n_obs)
  dim(logd2) <- c(n_chain, n_obs)

  # Marginalise z_t: mixture with weights (1 - alpha) on component 1 and alpha
  # on component 2. Stable two-term log-sum-exp on the log scale.
  a1 <- log1p(-alpha) + logd1
  a2 <- log(alpha)    + logd2
  m  <- pmax(a1, a2)
  ll <- m + log(exp(a1 - m) + exp(a2 - m))
  dim(ll) <- c(n_chain, n_obs)
  ll
}

#' Pointwise conditional log-likelihood for normal-mixture dynamic models
#'
#' @description
#' Computes the pointwise conditional log-likelihood matrix for a fitted
#' two-component Gaussian mixture model with dynamic weights
#' (\code{\link{mcmc_normal_mixture_locallevel}},
#' \code{\link{mcmc_normal_mixture_localtrend}},
#' \code{\link{mcmc_normal_mixture_localacceleration}}). The observation model is
#' \deqn{y_t \mid z_t \;\sim\; N\!\left(\mu_{z_t},\, \phi_{z_t}^{-1}\right),
#'       \qquad z_t \mid \alpha_t \;\sim\; \mathrm{Bernoulli}(\alpha_t),}
#' with \eqn{z_t \in \{0, 1\}} indexing components 1 and 2 and
#' \eqn{\alpha_t = T^{-1}(\theta_{t,1})} the weight of component 2.
#'
#' @details
#' The latent indicator \eqn{z_t} is \strong{marginalised out} (rather than
#' conditioned on its sampled value), giving the two-component mixture density
#' \deqn{\ell_{s,t} = \log\!\left[(1 - \alpha_t^{(s)})\,
#'       \phi\!\left(y_t; \mu_1^{(s)}, \sqrt{1/\phi_1^{(s)}}\right) +
#'       \alpha_t^{(s)}\,
#'       \phi\!\left(y_t; \mu_2^{(s)}, \sqrt{1/\phi_2^{(s)}}\right)\right],}
#' evaluated with a numerically stable log-sum-exp. Marginalising the indicator
#' is the correct choice for WAIC / LOO with mixture models; conditioning on the
#' sampled \eqn{z_t} would give a different (and inappropriate) quantity.
#'
#' Like the other families this is conditional on the latent \emph{state}
#' trajectory, so `loo::loo()` on it is a weak, in-sample cross-validation
#' measure; prefer LFO-CV (Burkner, Gabry & Vehtari, 2020) for predictive
#' comparison.
#'
#' @inherit log_lik.normal_locallevel references return
#'
#' @param object An object of class `normal_mixture_locallevel`,
#'   `normal_mixture_localtrend`, or `normal_mixture_localacceleration`.
#' @param ... Currently unused.
#'
#' @examples
#' \donttest{
#' set.seed(123)
#' n <- 200
#' alpha_true <- (sin(2 * pi * seq_len(n) / n) + 2) / 4
#' z_true <- rbinom(n, size = 1, prob = alpha_true)
#' mu_y    <- (1 - z_true) * 0 + z_true * 2
#' sigma_y <- (1 - z_true) * (1 / sqrt(4)) + z_true * (1 / sqrt(1))
#' y <- rnorm(n, mean = mu_y, sd = sigma_y)
#'
#' out <- mcmc_normal_mixture_locallevel(
#'   y,
#'   link               = "logit",
#'   burnin             = 1000,
#'   thinning           = 10,
#'   n_chain            = 500,
#'   prior_mu01_prec    = 0.01,
#'   prior_prec01_shape = 0.01,
#'   prior_prec01_rate  = 0.01,
#'   prior_mu02_prec    = 0.01,
#'   prior_prec02_shape = 0.01,
#'   prior_prec02_rate  = 0.01,
#'   prior_theta01_mean = 0,
#'   prior_theta01_prec = 1,
#'   prior_prec1_shape  = 100,
#'   prior_prec1_rate   = 1,
#'   verbose            = FALSE,
#'   seed               = 456
#' )
#'
#' ll <- log_lik(out)
#' dim(ll)   # c(n_chain, n_obs)
#'
#' ## The pointwise matrix feeds the loo package directly
#' ## (conditional-on-states; see Details):
#' if (requireNamespace("loo", quietly = TRUE)) {
#'   print(loo::waic(ll))
#' }
#' }
#'
#' @seealso
#'   \code{\link{mcmc_normal_mixture_locallevel}},
#'   \code{\link{mcmc_normal_mixture_localtrend}},
#'   \code{\link{mcmc_normal_mixture_localacceleration}} (model generators);
#'   \code{\link{log_lik}} (generic).
#'
#' @rdname log_lik.normal_mixture_locallevel
#' @export
log_lik.normal_mixture_locallevel <- function(object, ...) compute_loglik_normal_mixture(object)

#' @rdname log_lik.normal_mixture_locallevel
#' @export
log_lik.normal_mixture_localtrend <- function(object, ...) compute_loglik_normal_mixture(object)

#' @rdname log_lik.normal_mixture_locallevel
#' @export
log_lik.normal_mixture_localacceleration <- function(object, ...) compute_loglik_normal_mixture(object)
