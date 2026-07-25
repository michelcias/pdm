#' Run several independent chains of a pdm sampler
#'
#' Internal helper shared by every `mcmc_*` wrapper that accepts `chains`. It
#' re-issues the caller's own call once per chain, each time with `chains = 1`
#' and its own seed, and collects the fits into a `"pdm_mcmc_list"` object.
#'
#' The chains differ because every sampler draws its starting values from the
#' priors (initial state and precisions) before entering the Gibbs loop, so a
#' different seed means a genuinely different starting point -- the dispersion
#' that the Gelman-Rubin statistic needs. See `mcmc_convergence` for the
#' diagnostics computed from the result.
#'
#' Reproducibility does not depend on how the chains are executed: each chain
#' receives an explicit integer seed and calls `set.seed()` itself, so
#' sequential and parallel runs of the same `seed` give identical output.
#'
#' @param call The caller's `match.call()`.
#' @param envir The caller's `parent.frame()`, where `call` is evaluated.
#' @param chains Integer >= 2, number of chains to run.
#' @param seed Master seed (or `NULL`). Chain seeds are drawn from it, so a
#'   single `seed` reproduces the whole set.
#' @param parallel Logical; run the chains with `parallel::mclapply()`.
#'
#' @return An object of class `c("pdm_mcmc_list", "list")`.
#'
#' @keywords internal
#' @noRd
run_chains <- function(call, envir, chains, seed, parallel) {

  # Chain seeds are drawn from the master seed rather than taken as
  # `seed + 1, seed + 2, ...`: consecutive integers are poor Mersenne-Twister
  # seeds, and drawing them keeps a single `seed` reproducing the whole set.
  if (!is.null(seed)) set.seed(seed)
  seeds <- sample.int(.Machine$integer.max, chains)

  # Rebuild the caller's call as a single-chain run. `verbose` is forced off:
  # several progress bars writing to the same console interleave into noise.
  call$chains   <- 1L
  call$parallel <- NULL
  call$verbose  <- FALSE

  run_one <- function(s) {
    this_call      <- call
    this_call$seed <- s
    eval(this_call, envir)
  }

  if (parallel) {
    if (!requireNamespace("parallel", quietly = TRUE)) {
      stop("`parallel = TRUE` requires the 'parallel' package")
    }
    fits <- parallel::mclapply(seeds, run_one, mc.cores = worker_count(chains))
    # mclapply reports worker failures as condition objects instead of raising.
    failed <- vapply(fits, inherits, logical(1L), what = "try-error")
    if (any(failed)) {
      stop("chain ", which(failed)[1L], " failed: ",
           conditionMessage(attr(fits[[which(failed)[1L]]], "condition")))
    }
  } else {
    fits <- lapply(seeds, run_one)
  }

  new_pdm_mcmc_list(fits, seeds = seeds)
}


#' How many workers to fork for a parallel chain run
#'
#' Never more than there are chains, since a spare worker has nothing to do.
#'
#' `parallel::detectCores()` is not used directly, on its own documentation's
#' advice: it may return `NA`, and it counts the machine's logical CPUs rather
#' than the ones this process is *allowed* to use — a distinction that matters
#' under a container or a cluster scheduler. `getOption("mc.cores")` is the
#' documented way for a user, a job script or `R CMD check` to say how many
#' cores may be used, so it wins when set; `detectCores()` is only the fallback,
#' and 2 the fallback for that.
#'
#' @param chains Integer, number of chains about to be run.
#'
#' @return A positive integer.
#'
#' @keywords internal
#' @noRd
worker_count <- function(chains) {
  cores <- getOption("mc.cores")
  if (is.null(cores)) cores <- parallel::detectCores()
  if (length(cores) != 1L || is.na(cores) || !is.finite(cores)) cores <- 2L
  max(1L, min(as.integer(chains), as.integer(cores)))
}


#' Validate the multi-chain arguments of an mcmc_* wrapper
#'
#' Kept separate so each sampler's dispatch block stays at two statements.
#'
#' @param chains Candidate number of chains.
#' @param parallel Candidate parallel flag.
#'
#' @return `NULL`, invisibly; called for the side effect of raising an error.
#'
#' @keywords internal
#' @noRd
validate_chains_args <- function(chains, parallel) {
  if (!is.numeric(chains) || length(chains) != 1 || chains < 1 ||
      chains != floor(chains)) {
    stop("`chains` must be a single positive integer")
  }
  if (!is.logical(parallel) || length(parallel) != 1) {
    stop("`parallel` must be a single logical value")
  }
  invisible(NULL)
}


#' Constructor for pdm_mcmc_list objects
#'
#' @param fits List of fitted `pdm_mcmc` objects, one per chain.
#' @param seeds Integer vector of the seed used by each chain.
#'
#' @return An object of class `c("pdm_mcmc_list", "list")`.
#'
#' @keywords internal
#' @noRd
new_pdm_mcmc_list <- function(fits, seeds) {

  if (!is.list(fits) || length(fits) == 0L) {
    stop("Internal error: fits must be a non-empty list")
  }
  if (!all(vapply(fits, inherits, logical(1L), what = "pdm_mcmc"))) {
    stop("Internal error: every element of fits must inherit from 'pdm_mcmc'")
  }

  class(fits) <- c("pdm_mcmc_list", "list")

  # Metadata is shared by construction (the chains come from one call), so it is
  # lifted from the first fit and kept at the list level for the methods.
  first <- fits[[1L]]
  attr(fits, "chains")     <- length(fits)
  attr(fits, "seeds")      <- seeds
  attr(fits, "n_obs")      <- attr(first, "n_obs")
  attr(fits, "n_draws")    <- attr(first, "n_draws")
  attr(fits, "burnin")     <- attr(first, "burnin")
  attr(fits, "thinning")   <- attr(first, "thinning")
  attr(fits, "model_type") <- attr(first, "model_type")
  attr(fits, "y")          <- attr(first, "y")

  validate_pdm_mcmc_list(fits)
}


#' Validator for pdm_mcmc_list objects
#'
#' @param x An object to validate.
#'
#' @return The input object `x` if validation succeeds.
#'
#' @keywords internal
#' @noRd
validate_pdm_mcmc_list <- function(x) {

  if (!inherits(x, "pdm_mcmc_list")) {
    stop("Object must inherit from class 'pdm_mcmc_list'")
  }

  # Diagnostics compare chains parameter by parameter, so a mismatch in class,
  # retained length or component names would silently produce nonsense.
  classes <- lapply(x, function(ch) class(ch)[1L])
  if (length(unique(unlist(classes))) != 1L) {
    stop("Internal error: all chains must share the same model class")
  }

  n_draws <- vapply(x, function(ch) attr(ch, "n_draws"), integer(1L))
  if (length(unique(n_draws)) != 1L) {
    stop("Internal error: all chains must retain the same number of draws")
  }

  nms <- lapply(x, names)
  if (!all(vapply(nms, identical, logical(1L), y = nms[[1L]]))) {
    stop("Internal error: all chains must have the same components")
  }

  x
}


#' Collapse a multi-chain fit into a single equivalent fit
#'
#' Stacks the draws of every chain into one `pdm_mcmc` object of the original
#' model class: matrices (the latent trajectories, `alpha`, `z`) are `rbind`ed
#' and vectors (the scalar parameters) concatenated, chain 1 first. The result
#' carries the attributes of a single chain with `n_draws` corrected to the
#' pooled total, so every existing single-chain method applies to it unchanged.
#'
#' Pooling is the right operation for anything that estimates a posterior
#' quantity — summaries, credible bands, WAIC/LOO — because after convergence
#' the chains are draws from the same distribution and more draws is simply a
#' better estimate. It is the wrong operation for anything that reads the draws
#' as a *sequence*: a pooled trace plot shows spurious jumps where one chain is
#' concatenated to the next, which is why `plot.pdm_mcmc_list()` keeps the
#' chains apart for `type = "mcmc"`.
#'
#' @param x An object of class `"pdm_mcmc_list"`.
#'
#' @return A `pdm_mcmc` object of the same model class as the individual chains.
#'
#' @keywords internal
#' @noRd
pool_chains <- function(x) {
  if (!inherits(x, "pdm_mcmc_list")) {
    stop("'x' must inherit from 'pdm_mcmc_list'")
  }

  first  <- x[[1L]]
  comps  <- names(first)

  pooled <- lapply(comps, function(nm) {
    parts <- lapply(x, function(ch) ch[[nm]])
    if (is.matrix(parts[[1L]])) do.call(rbind, parts)
    else unlist(parts, use.names = FALSE)
  })
  names(pooled) <- comps

  # Carry every attribute of a single chain across (class included), then
  # correct the draw count and record how many chains produced it.
  for (a in setdiff(names(attributes(first)), "names")) {
    attr(pooled, a) <- attr(first, a)
  }
  attr(pooled, "n_draws") <- attr(first, "n_draws") * length(x)
  attr(pooled, "chains")  <- length(x)

  pooled
}


#' Assemble one scalar parameter's draws as an iterations-by-chains matrix
#'
#' Shared by `mcmc_convergence.pdm_mcmc_list()` and the multi-chain plot method,
#' which both need the same layout: the shape every function in `R/rhat.R`
#' expects.
#'
#' @param configs List of `get_param_config()` results, one per chain.
#' @param nm Name of the scalar parameter within those configs.
#' @param n_draw Number of retained draws per chain.
#'
#' @return Numeric matrix `n_draw x length(configs)`.
#'
#' @keywords internal
#' @noRd
scalar_draws <- function(configs, nm, n_draw) {
  m <- vapply(configs, function(cfg) cfg[[nm]]$samples, numeric(n_draw))
  dim(m) <- c(n_draw, length(configs))
  m
}


#' Print method for pdm_mcmc_list objects
#'
#' @description Compact summary of a multi-chain fit: the model, how the chains
#'   were run, and the seed of each chain. Use \code{\link{mcmc_convergence}}
#'   for the diagnostics and index a single chain (`x[[1]]`) to reach the
#'   ordinary `print()`, `plot()` and `summary()` methods.
#'
#' @param x An object of class `"pdm_mcmc_list"`.
#' @param ... Additional arguments (currently unused).
#'
#' @return Invisibly returns `x`.
#'
#' @examples
#' \donttest{
#' set.seed(123)
#' y <- cumsum(rnorm(100)) + rnorm(100, sd = 0.3)
#'
#' fits <- mcmc_normal_locallevel(
#'   y,
#'   burnin             = 500,
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
#' print(fits)
#' plot(fits[[1]], type = "states")   # methods apply to a single chain
#' }
#'
#' @seealso \code{\link{mcmc_convergence}} for the multi-chain diagnostics;
#'   \code{\link{pdm}} for the catalog of samplers, all of which accept
#'   `chains`.
#'
#' @export
print.pdm_mcmc_list <- function(x, ...) {

  cat("\n")
  cat("Multi-chain pdm fit\n")
  cat(strrep("=", 70), "\n\n", sep = "")
  cat("Model:         ", class(x[[1L]])[1L], "\n", sep = "")
  cat("Chains:        ", attr(x, "chains"), "\n", sep = "")
  cat("Draws/chain:   ", attr(x, "n_draws"), "\n", sep = "")
  cat("Burn-in:       ", attr(x, "burnin"), "\n", sep = "")
  cat("Thinning:      ", attr(x, "thinning"), "\n", sep = "")
  cat("Observations:  ", attr(x, "n_obs"), "\n", sep = "")
  cat("Seeds:         ", paste(attr(x, "seeds"), collapse = ", "), "\n", sep = "")
  cat("\n")
  cat("Use mcmc_convergence() for R-hat and effective sample sizes.\n")
  cat(strrep("-", 70), "\n\n", sep = "")

  invisible(x)
}
