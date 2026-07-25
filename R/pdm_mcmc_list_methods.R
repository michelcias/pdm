#' Methods for multi-chain pdm fits
#'
#' @description Methods that let an object returned with `chains > 1` be used
#'   wherever a single-chain fit would be. Except for `plot(type = "mcmc")`,
#'   they work by stacking the chains into one equivalent fit and delegating to
#'   the existing single-chain method, so results are identical to what the same
#'   number of draws from one chain would give.
#'
#' @details
#' \subsection{Why pooling is the right default}{
#'   Once the chains have converged they are draws from the same posterior, so
#'   four chains of 1000 estimate posterior quantities exactly as 4000 draws
#'   would — better, in fact, since the draws are less autocorrelated. Every
#'   method here that reports a posterior quantity (`summary`, `log_lik`,
#'   `waic`, `loo`, and every `plot` type other than `"mcmc"`) therefore pools.
#'
#'   The exception is anything that reads the draws as a *sequence*. A pooled
#'   trace plot splices chain 2 onto the end of chain 1 and shows a jump that
#'   means nothing, so `plot(type = "mcmc")` keeps the chains apart and draws
#'   one series per chain. Check convergence before trusting the pooled
#'   summaries: \code{\link{mcmc_convergence}} is the numerical counterpart of
#'   that plot.
#' }
#'
#' \subsection{`loo()` uses the chain structure}{
#'   `loo.pdm_mcmc_list()` is not a plain delegation. The single-chain method
#'   has to tell \pkg{loo} that all draws come from one chain
#'   (`chain_id = rep(1, ...)`), which makes the relative efficiency used in the
#'   PSIS smoothing a cruder estimate. With several chains the real chain
#'   identifiers are passed instead, so `r_eff` — and therefore the Pareto
#'   diagnostics — rest on a better estimate.
#' }
#'
#' @param x,object An object of class `"pdm_mcmc_list"`.
#' @param type Character, which diagnostic to draw. Accepts the same values as
#'   the plot method of the underlying model family (see
#'   \code{\link{plot.normal_locallevel}} and its siblings). Default `"mcmc"`.
#' @param which Optional integer vector selecting scalar parameters for
#'   `type = "mcmc"`, indexed as in the single-chain plot methods. `NULL` (the
#'   default) draws all of them.
#' @param true_values Optional named list of true values to overlay, as in the
#'   single-chain plot methods.
#' @param ask Logical, whether to pause between pages. Defaults to
#'   `interactive()` when more than one page will be drawn.
#' @param ... Passed to the underlying single-chain method.
#'
#' @return `summary()` returns the same object the single-chain method returns
#'   for that model class; `log_lik()` a draws-by-observations matrix pooled
#'   over chains; `waic()` and `loo()` the corresponding \pkg{loo} objects;
#'   `plot()` returns `x` invisibly.
#'
#' @examples
#' \donttest{
#' set.seed(123)
#' n <- 150
#' y <- cumsum(c(10, rnorm(n)))[-1] + rnorm(n, sd = sqrt(1 / 5))
#'
#' fits <- mcmc_normal_locallevel(
#'   y,
#'   burnin             = 1000,
#'   thinning           = 5,
#'   n_chain            = 500,
#'   prior_theta01_mean = y[1],
#'   prior_theta01_prec = 1 / var(y),
#'   prior_prec1_shape  = 1e-2,
#'   prior_prec1_rate   = 1e-2,
#'   prior_prec_y_shape = 1e-2,
#'   prior_prec_y_rate  = 1e-2,
#'   chains             = 4,
#'   seed               = 456
#' )
#'
#' ## Always check convergence before reading a pooled summary
#' mcmc_convergence(fits)
#'
#' ## Per-chain diagnostics: one series per chain, R-hat in the header
#' plot(fits, type = "mcmc", which = 1)
#'
#' ## Everything else pools the chains
#' plot(fits, type = "states")
#' summary(fits)
#'
#' ## Model comparison, with r_eff estimated from the real chains.
#' ## `loo` is a generic of the loo package, so qualify it (or attach loo first).
#' loo::loo(fits)
#' loo::waic(fits)
#' }
#'
#' @seealso \code{\link{mcmc_convergence.pdm_mcmc_list}} for the numerical
#'   diagnostics; \code{\link{print.pdm_mcmc_list}}; \code{\link{pdm_compare}}
#'   for ranking several fits.
#'
#' @name pdm_mcmc_list-methods
NULL


#' @rdname pdm_mcmc_list-methods
#' @export
summary.pdm_mcmc_list <- function(object, ...) {
  summary(pool_chains(object), ...)
}


#' @rdname pdm_mcmc_list-methods
#' @export
log_lik.pdm_mcmc_list <- function(object, ...) {
  log_lik(pool_chains(object), ...)
}


#' @rdname pdm_mcmc_list-methods
#' @exportS3Method loo::waic
waic.pdm_mcmc_list <- function(x, ...) {
  require_loo()
  loo::waic(log_lik(x), ...)
}


#' @rdname pdm_mcmc_list-methods
#' @exportS3Method loo::loo
loo.pdm_mcmc_list <- function(x, ...) {
  require_loo()
  ll   <- log_lik(x)
  dots <- list(...)
  if (is.null(dots$r_eff)) {
    # pool_chains() stacks chain 1 first, so the chain identifiers run in blocks
    # of n_chain. Unlike the single-chain method, these are the real ones.
    dots$r_eff <- loo::relative_eff(
      exp(ll),
      chain_id = rep(seq_along(x), each = attr(x[[1L]], "n_chain"))
    )
  }
  do.call(loo::loo, c(list(ll), dots))
}


#' @rdname pdm_mcmc_list-methods
#' @export
plot.pdm_mcmc_list <- function(x,
                               type        = "mcmc",
                               which       = NULL,
                               true_values = NULL,
                               ask         = NULL,
                               ...) {

  pooled      <- pool_chains(x)
  model_class <- class(pooled)[1L]

  # The admissible `type` values differ by family (only the mixture has
  # "params", only the adaptive families have "acceptance"). Read them off the
  # family's own method rather than hard-coding the table.
  family_plot <- utils::getS3method("plot", model_class)
  allowed     <- eval(formals(family_plot)$type)

  if (!is.character(type) || length(type) != 1L || !type %in% allowed) {
    stop("`type` must be one of: ", paste(dQuote(allowed), collapse = ", "))
  }

  if (is.null(ask)) ask <- interactive() && type == "all"
  if (ask) {
    oldask <- par(ask = TRUE)
    on.exit(par(oldask), add = TRUE)
  }

  if (type %in% c("mcmc", "all")) {
    plot_multichain_params(x, which = which, true_values = true_values)
  }

  if (type == "all") {
    # Delegate the remaining pages to the family's own method, one type at a
    # time, on the pooled fit.
    for (t in setdiff(allowed, c("all", "mcmc"))) {
      # "acceptance" exists only when the fit was run with
      # return_accept_prop = TRUE. Skip it on the same explicit condition the
      # single-chain methods use, rather than catching the error it would raise
      # -- a tryCatch here would also swallow a genuine failure in any page.
      if (t == "acceptance" && is.null(pooled$accept_prop)) next
      plot(pooled, type = t, true_values = true_values, ask = FALSE, ...)
    }
  } else if (type != "mcmc") {
    plot(pooled, type = type, which = which, true_values = true_values, ...)
  }

  invisible(x)
}


#' Per-chain diagnostic pages for the scalar parameters
#'
#' @param x A `"pdm_mcmc_list"`.
#' @param which Optional integer vector selecting parameters.
#' @param true_values Optional named list of true values.
#'
#' @return `NULL`, invisibly.
#'
#' @keywords internal
#' @noRd
plot_multichain_params <- function(x, which = NULL, true_values = NULL) {

  n_draw  <- attr(x[[1L]], "n_chain")
  configs <- lapply(x, get_param_config)
  params  <- names(configs[[1L]])

  if (!is.null(which)) {
    if (!is.numeric(which) || any(which < 1) || any(which > length(params))) {
      stop("`which` must index the scalar parameters, i.e. be within 1:",
           length(params))
    }
    params <- params[sort(unique(as.integer(which)))]
  }

  for (nm in params) {
    cfg <- configs[[1L]][[nm]]
    plot_param_diagnostics_multi(
      draws       = scalar_draws(configs, nm, n_draw),
      param_label = cfg$label,
      param_name  = cfg$name,
      true_value  = true_value_for(cfg$name_str, true_values)
    )
  }

  invisible(NULL)
}
