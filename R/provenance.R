#' Record what produced a fit, on the fit itself
#'
#' Internal helper called by every `mcmc_*()` wrapper just before it returns.
#' Attaches the things a saved object could not otherwise report: the package
#' version, the seed, every prior actually used, and -- for the samplers that
#' accept `init` -- the starting values the chain actually used.
#'
#' @section Why this exists:
#' Between 0.1-0 and 0.5-0 four commits changed sampler output for a given seed,
#' three of them by changing defaults. A `.rds` saved before that says nothing
#' about which version produced it, and the priors can no longer be recovered by
#' reading the calling script, because several are now derived from the data and
#' never appear in the call. A fit carrying its own provenance can be reproduced
#' on any later version by passing the recorded values back explicitly.
#'
#' @section What `priors` contains:
#' Every `prior_*` variable in the calling wrapper's frame, after resolution —
#' so a `NULL` scale that was filled in from the data appears as the number that
#' was used, and a `type` appears as the string `match.arg()` settled on.
#' `NULL` entries are dropped: they are the arguments that the chosen prior does
#' not use, such as `shape` under a Half-t.
#'
#' The names are the wrapper's own argument names, so the list can be spliced
#' straight back into a call:
#'
#' ```r
#' do.call(mcmc_normal_locallevel,
#'         c(list(y = attr(fit, "y"), burnin = attr(fit, "burnin"),
#'                thinning = attr(fit, "thinning"),
#'                n_draws = attr(fit, "n_draws")),
#'           attr(fit, "priors")))
#' ```
#'
#' @section What `init` contains:
#' The starting values after resolution, so a fit reports where its chain began
#' whether or not the user chose the point. It is recorded for reading and for
#' warm-starting a later run, **not** for the bit-identical round trip above:
#' supplying `init` changes what is drawn before the loop, so a rerun that
#' splices it back is a different (equally valid) chain. The round trip that
#' reproduces a fit exactly is the one through `priors` and `seed`, with `init`
#' left alone.
#'
#' @param object The fitted object.
#' @param seed The `seed` argument as the user supplied it, possibly `NULL`.
#' @param init The resolved starting values, from `resolve_init()`. `NULL` for
#'   the samplers that do not yet accept `init`, in which case no attribute is
#'   attached.
#' @param env The wrapper's frame, from which the `prior_*` variables are read.
#'
#' @return `object`, with `pdm_version`, `seed`, `priors` and (when supplied)
#'   `init` attached.
#'
#' @keywords internal
#' @noRd
record_provenance <- function(object, seed, init = NULL, env = parent.frame()) {
  nms  <- sort(grep("^prior_", ls(env), value = TRUE))
  vals <- mget(nms, envir = env, ifnotfound = list(NULL))

  attr(object, "pdm_version") <- as.character(utils::packageVersion("pdm"))
  attr(object, "seed")        <- seed
  attr(object, "priors")      <- vals[!vapply(vals, is.null, logical(1L))]
  if (!is.null(init)) attr(object, "init") <- init
  object
}
