#' Resolve a precision prior specification (Gamma or Half-t / Half-Cauchy)
#'
#' Internal helper shared by every `mcmc_*` wrapper. It validates one
#' precision's prior specification and normalises it to the integer code plus
#' the finite numeric hyperparameters expected by the C layer. All validation
#' and the `"halfcauchy"` -> Half-t(df = 1) alias normalisation happen here, in
#' R, so the C sampler only ever receives an already-decided integer code
#' (`0 = Gamma`, `1 = Half-t`) and never re-evaluates the prior choice inside the
#' MCMC loop.
#'
#' @param type One of `"gamma"`, `"halfcauchy"`, `"halft"` (already
#'   `match.arg()`-ed by the caller).
#' @param shape,rate Gamma shape/rate (required only when `type = "gamma"`).
#' @param scale Half-t scale \eqn{A > 0} (required for Half-t / Half-Cauchy).
#' @param df Half-t degrees of freedom \eqn{\nu > 0} (Half-Cauchy fixes it to 1).
#' @param df_user_set Logical; whether the caller explicitly set `df` (used to
#'   flag a contradictory `type = "halfcauchy"` combined with `df != 1`).
#' @param label Character prefix used to build informative error messages
#'   (e.g. `"prior_prec1"`).
#'
#' @return A list with elements `code` (0/1L), `type`, `shape`, `rate`,
#'   `scale`, `df`, all finite, ready to be forwarded to `.Call()`.
#'
#' @keywords internal
#' @noRd
resolve_prec_prior <- function(type, shape, rate, scale, df, df_user_set, label) {
  if (type == "gamma") {
    if (is.null(shape) || is.null(rate)) {
      stop(sprintf(
        "`%s_type = \"gamma\"` requires both `%s_shape` and `%s_rate`",
        label, label, label
      ))
    }
    if (!is.numeric(shape) || length(shape) != 1 || shape <= 0) {
      stop(sprintf("`%s_shape` must be a single positive numeric value", label))
    }
    if (!is.numeric(rate) || length(rate) != 1 || rate <= 0) {
      stop(sprintf("`%s_rate` must be a single positive numeric value", label))
    }
    return(list(code = 0L, type = "gamma",
                shape = as.numeric(shape), rate = as.numeric(rate),
                scale = 1, df = 1))
  }

  # type is "halfcauchy" (alias for Half-t with df = 1) or "halft"
  if (type == "halfcauchy") {
    if (df_user_set && !isTRUE(all.equal(df, 1))) {
      stop(sprintf(
        "`%s_type = \"halfcauchy\"` fixes df = 1; use `%s_type = \"halft\"` to set `%s_df`",
        label, label, label
      ))
    }
    df <- 1
  }
  if (is.null(scale)) {
    stop(sprintf(
      "A Half-t/Half-Cauchy prior requires `%s_scale` (the scale A > 0)", label
    ))
  }
  if (!is.numeric(scale) || length(scale) != 1 || scale <= 0) {
    stop(sprintf("`%s_scale` must be a single positive numeric value", label))
  }
  if (!is.numeric(df) || length(df) != 1 || df <= 0) {
    stop(sprintf("`%s_df` must be a single positive numeric value", label))
  }
  list(code = 1L, type = type,
       shape = 1, rate = 1,
       scale = as.numeric(scale), df = as.numeric(df))
}
