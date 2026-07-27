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
#'   Both the scalar parameters and a sample of twenty time points along each
#'   latent trajectory are checked. An earlier version checked only the scalars,
#'   on the stated grounds that they are the slowest-mixing part of these
#'   models. That is true of the Gaussian family and false of the link
#'   families: measured at the settings the examples use, `binomial`, `poisson`
#'   and `probit` all had *states* with higher \eqn{\hat{R}} than any scalar,
#'   and a scalar-only screen therefore reported all-clear on fits whose
#'   trajectories had not converged. For every time point rather than a sample,
#'   use \code{\link{mcmc_convergence}}.
#' }
#'
#' \subsection{`waic()` and `loo()` warn too}{
#'   They pool the chains exactly as `summary()` does, and a `"loo"` object gives
#'   no hint whether the draws behind it agreed. Both therefore raise the same
#'   warning above `rhat_threshold`. Model comparison is where an unconverged
#'   fit does the most damage, since the number it produces looks like every
#'   other elpd.
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
#' @param rhat_threshold Numeric > 1, the \eqn{\hat{R}} above which `summary()`,
#'   `waic()` and `loo()` warn that the chains have not converged. Default
#'   `1.01`, matching \code{\link{mcmc_convergence.pdm_mcmc_list}}.
#' @param ... Passed to the underlying single-chain method.
#'
#' @return `summary()` returns what the single-chain method returns for that
#'   model class, with class `"summary.pdm_mcmc_list"` prepended and three
#'   extra elements: `chains`, `n_draws_each`, `rhat` (a named vector of
#'   rank-normalized split-\eqn{\hat{R}}, one per scalar parameter) and
#'   `rhat_states` (the same, at twenty time points along each trajectory).
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
#'   n_draws            = 500,
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
  rhat        <- scalar_rhats(object)
  rhat_states <- state_rhats(object)

  out$chains         <- length(object)
  out$n_draws_each   <- attr(object[[1L]], "n_draws")
  out$rhat           <- rhat
  out$rhat_states    <- rhat_states
  out$rhat_threshold <- rhat_threshold

  warn_if_unconverged(c(rhat, rhat_states), rhat_threshold,
                      "the pooled summary below")

  class(out) <- c("summary.pdm_mcmc_list", class(out))
  out
}


#' Warn when pooling chains that have not agreed
#'
#' Shared by `summary()`, `waic()` and `loo()` on a `pdm_mcmc_list`. All three
#' pool the chains, and all three produce output that looks the same whether the
#' chains agreed or not — so each says so rather than leaving the user to
#' remember to check separately.
#'
#' @param rhat Named numeric vector from `scalar_rhats()`.
#' @param threshold The cut-off above which to warn.
#' @param what Phrase naming what the pooling produced, spliced into the
#'   message.
#'
#' @return `NULL`, invisibly.
#'
#' @keywords internal
#' @noRd
warn_if_unconverged <- function(rhat, threshold, what) {
  worst <- if (all(is.na(rhat))) NA_real_ else max(rhat, na.rm = TRUE)
  if (!is.na(worst) && worst > threshold) {
    warning("The chains have not converged: max R-hat is ",
            format(worst, digits = 4), " against a threshold of ", threshold,
            ", so ", what, " mixes draws from distributions that do not agree. ",
            "Run mcmc_convergence() for the full table.", call. = FALSE)
  }
  invisible(NULL)
}


#' Rank-normalized split-R-hat for every scalar parameter of a multi-chain fit
#'
#' Cheap: a handful of `n_draw x n_draws` matrices. This covers only half of
#' what a convergence check needs — in the link families the latent states are
#' the slower of the two — so `summary()` pairs it with `state_rhats()`.
#'
#' @param x An object of class `"pdm_mcmc_list"`.
#'
#' @return Named numeric vector of R-hat values, one per scalar parameter.
#'
#' @keywords internal
#' @noRd
scalar_rhats <- function(x) {
  n_draw  <- attr(x[[1L]], "n_draws")
  configs <- lapply(x, get_param_config)
  params  <- names(configs[[1L]])

  out <- vapply(params, function(nm) {
    rhat_rank_normalized(scalar_draws(configs, nm, n_draw))
  }, numeric(1L))

  names(out) <- vapply(params, function(nm) configs[[1L]][[nm]]$name_str,
                       character(1L))
  out
}


#' Rank-normalized split-R-hat for the latent states, on a sample of time points
#'
#' A screen, not a census. Every time point of every trajectory would be one
#' `n_draw x n_draws` matrix each — measured at n = 400, that is 1.76s against
#' 0.04s for the scalars, which is too much to spend inside `summary()`. Twenty
#' evenly spaced points cost about 0.1s and are enough to notice a trajectory
#' that has not settled.
#'
#' Twenty rather than the three that \code{\link{mcmc_convergence}} defaults to:
#' on a measured binomial fit whose states exceeded the threshold at 47 of 199
#' time points, three evenly spaced points caught none of them and five caught
#' none; ten and twenty caught them. Three is a reasonable default for a table a
#' user reads, and a poor one for an automatic check.
#'
#' @param x An object of class `"pdm_mcmc_list"`.
#' @param timepoints How many evenly spaced time points to screen (a single
#'   whole number, the default 20), or an explicit numeric vector of fractions
#'   in \eqn{(0, 1)}. Resolved by `resolve_timepoints()`, the same helper
#'   `mcmc_convergence.pdm_mcmc_list()` uses, so the automatic screen and the
#'   diagnostic table always look at the same points.
#'
#' @return Named numeric vector of R-hat values, one per sampled time point,
#'   empty if the model carries no trajectory matrices.
#'
#' @keywords internal
#' @noRd
state_rhats <- function(x, timepoints = 20L) {

  timepoints <- resolve_timepoints(timepoints, "timepoints")

  n_draw <- attr(x[[1L]], "n_draws")

  # Trajectories are theta_1, theta_2, ... and stored as n_draw x n_obs. The
  # initial states theta_01, theta_02 are vectors, are covered by
  # scalar_rhats(), and must not match here.
  snames <- grep("^theta_[1-9][0-9]*$", names(x[[1L]]), value = TRUE)
  snames <- snames[vapply(snames, function(nm) is.matrix(x[[1L]][[nm]]),
                          logical(1L))]
  if (length(snames) == 0L) return(numeric(0L))
  snames <- snames[order(as.integer(sub("theta_", "", snames)))]

  out <- lapply(snames, function(sname) {
    n_t <- ncol(x[[1L]][[sname]])
    idx <- sort(unique(pmax(1L, pmin(n_t, round(timepoints * n_t)))))
    vapply(idx, function(tidx) {
      draws <- vapply(x, function(ch) ch[[sname]][, tidx], numeric(n_draw))
      dim(draws) <- c(n_draw, length(x))
      rhat_rank_normalized(draws)
    }, numeric(1L))
  })

  names(out) <- snames
  unlist(lapply(snames, function(sname) {
    v <- out[[sname]]
    n_t <- ncol(x[[1L]][[sname]])
    idx <- sort(unique(pmax(1L, pmin(n_t, round(timepoints * n_t)))))
    stats::setNames(v, sprintf("%s[t=%d]", sname, idx))
  }))
}


#' @rdname pdm_mcmc_list-methods
#' @param digits Integer, significant digits for the R-hat column. Default 4.
#' @export
print.summary.pdm_mcmc_list <- function(x, digits = 4L, ...) {

  # The family's own summary first, then the evidence about whether pooling
  # those chains was legitimate.
  NextMethod()

  # Reported separately, judged together: which of the two is the bottleneck
  # differs by family, so a single number would hide the one that matters.
  all_rhat <- c(x$rhat, x$rhat_states)
  worst <- if (all(is.na(all_rhat))) NA_real_ else max(all_rhat, na.rm = TRUE)

  cat("Multi-chain diagnostics (", x$chains, " chains x ", x$n_draws_each,
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

  if (length(x$rhat_states) > 0L) {
    ws <- max(x$rhat_states, na.rm = TRUE)
    cat("  State R-hat:   max ", format(ws, digits = digits), " over ",
        length(x$rhat_states), " time points (", names(which.max(x$rhat_states)),
        ")\n", sep = "")
  }

  if (is.na(worst)) {
    cat("  R-hat is undefined everywhere (constant draws?).\n")
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
waic.pdm_mcmc_list <- function(x, rhat_threshold = 1.01, ...) {
  require_loo()
  warn_if_unconverged(c(scalar_rhats(x), state_rhats(x)), rhat_threshold,
                      "the pooled log-likelihood")
  loo::waic(log_lik(x), ...)
}


#' @rdname pdm_mcmc_list-methods
#' @exportS3Method loo::loo
loo.pdm_mcmc_list <- function(x, rhat_threshold = 1.01, ...) {
  require_loo()
  warn_if_unconverged(c(scalar_rhats(x), state_rhats(x)), rhat_threshold,
                      "the pooled log-likelihood")
  ll   <- log_lik(x)
  dots <- list(...)
  if (is.null(dots$r_eff)) {
    # pool_chains() stacks chain 1 first, so the chain identifiers run in blocks
    # of n_draws. Unlike the single-chain method, these are the real ones.
    dots$r_eff <- loo::relative_eff(
      exp(ll),
      chain_id = rep(seq_along(x), each = attr(x[[1L]], "n_draws"))
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

  n_draw  <- attr(x[[1L]], "n_draws")
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
