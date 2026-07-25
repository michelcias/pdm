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
#'   one series per chain.
#' }
#'
#' \subsection{`summary()` carries its own convergence evidence}{
#'   Pooling is only meaningful once the chains agree, and a pooled summary
#'   looks exactly the same whether they do or not. `summary()` therefore
#'   computes the rank-normalized split-\eqn{\hat{R}} of every scalar parameter,
#'   stores it on the returned object, and prints it underneath the usual
#'   tables. When the largest exceeds `rhat_threshold` it also raises a
#'   warning at call time, so the caveat survives being scrolled past or
#'   captured.
#'
#'   Only the scalar parameters are checked: they are cheap, and across the
#'   families they are the slowest-mixing part of these models, so they are what
#'   a convergence problem shows up in first. For the latent state trajectories
#'   as well, use \code{\link{mcmc_convergence}}.
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
#' @param x,object An object of class `"pdm_mcmc_list"` — except in
#'   `print.summary.pdm_mcmc_list()`, where `x` is what `summary()` returned.
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
#' @param rhat_threshold Numeric > 1, the \eqn{\hat{R}} above which `summary()`
#'   warns that the chains have not converged. Default `1.01`, matching
#'   \code{\link{mcmc_convergence.pdm_mcmc_list}}.
#' @param ... Passed to the underlying single-chain method.
#'
#' @return `summary()` returns what the single-chain method returns for that
#'   model class, with class `"summary.pdm_mcmc_list"` prepended and three
#'   extra elements: `chains`, `n_chain_each` and `rhat` (a named vector of
#'   rank-normalized split-\eqn{\hat{R}}, one per scalar parameter).
#'   `log_lik()` returns a draws-by-observations matrix pooled over chains;
#'   `waic()` and `loo()` the corresponding \pkg{loo} objects; `plot()` returns
#'   `x` invisibly.
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
summary.pdm_mcmc_list <- function(object, rhat_threshold = 1.01, ...) {

  if (!is.numeric(rhat_threshold) || length(rhat_threshold) != 1L ||
      rhat_threshold <= 1) {
    stop("'rhat_threshold' must be a single numeric value greater than 1")
  }

  out <- summary(pool_chains(object), ...)

  # Pooling is only meaningful once the chains agree, and the pooled summary
  # looks identical either way -- so the summary carries the evidence with it
  # rather than leaving the user to remember to check separately.
  rhat <- scalar_rhats(object)
  out$chains         <- length(object)
  out$n_chain_each   <- attr(object[[1L]], "n_chain")
  out$rhat           <- rhat
  out$rhat_threshold <- rhat_threshold

  worst <- if (all(is.na(rhat))) NA_real_ else max(rhat, na.rm = TRUE)
  if (!is.na(worst) && worst > rhat_threshold) {
    warning("The chains have not converged: max R-hat is ",
            format(worst, digits = 4), " against a threshold of ",
            rhat_threshold, ", so the pooled summary below mixes draws from ",
            "distributions that do not agree. Run mcmc_convergence() for the ",
            "full table.", call. = FALSE)
  }

  class(out) <- c("summary.pdm_mcmc_list", class(out))
  out
}


#' Rank-normalized split-R-hat for every scalar parameter of a multi-chain fit
#'
#' Restricted to the scalar parameters on purpose: they are cheap (a handful of
#' `n_draw x n_chain` matrices) and, measured across the families, they are also
#' the slowest-mixing part of these models, so they are what a convergence check
#' would flag first. The latent state trajectories are one matrix per time point
#' and are covered by \code{\link{mcmc_convergence}} instead.
#'
#' @param x An object of class `"pdm_mcmc_list"`.
#'
#' @return Named numeric vector of R-hat values, one per scalar parameter.
#'
#' @keywords internal
#' @noRd
scalar_rhats <- function(x) {
  n_draw  <- attr(x[[1L]], "n_chain")
  configs <- lapply(x, get_param_config)
  params  <- names(configs[[1L]])

  out <- vapply(params, function(nm) {
    rhat_rank_normalized(scalar_draws(configs, nm, n_draw))
  }, numeric(1L))

  names(out) <- vapply(params, function(nm) configs[[1L]][[nm]]$name_str,
                       character(1L))
  out
}


#' @rdname pdm_mcmc_list-methods
#' @param digits Integer, significant digits for the R-hat column. Default 4.
#' @export
print.summary.pdm_mcmc_list <- function(x, digits = 4L, ...) {

  # The family's own summary first, then the evidence about whether pooling
  # those chains was legitimate.
  NextMethod()

  worst <- if (all(is.na(x$rhat))) NA_real_ else max(x$rhat, na.rm = TRUE)

  cat("Multi-chain diagnostics (", x$chains, " chains x ", x$n_chain_each,
      " draws, pooled above)\n", sep = "")
  cat(strrep("-", 75), "\n", sep = "")

  # The mixture families carry six scalars, which overruns the 75-column rule
  # the surrounding summary uses. Wrapped three per line rather than by
  # strwrap(), which would break "W_1^{-1} = 2.01" at its internal spaces.
  rhat_txt <- sprintf(paste0("%s = %.", digits, "f"), names(x$rhat), x$rhat)
  rows     <- split(rhat_txt, ceiling(seq_along(rhat_txt) / 3L))
  for (i in seq_along(rows)) {
    cat(if (i == 1L) "  Scalar R-hat:  " else "                 ",
        paste(rows[[i]], collapse = "   "), "\n", sep = "")
  }

  if (is.na(worst)) {
    cat("  R-hat is undefined for every scalar parameter (constant draws?).\n")
  } else if (worst > x$rhat_threshold) {
    cat("  WARNING: max R-hat ", format(worst, digits = digits), " > ",
        x$rhat_threshold, ". The chains disagree, so the\n",
        "  summary above describes no single posterior.\n",
        "  See mcmc_convergence() for the full table.\n", sep = "")
  } else {
    cat("  All below ", x$rhat_threshold,
        "; pooling the chains is justified.\n", sep = "")
  }
  cat(strrep("-", 75), "\n\n", sep = "")

  invisible(x)
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
