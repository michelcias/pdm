#' Multi-chain convergence diagnostics for pdm models
#'
#' @description Assesses convergence across the independent chains produced by
#'   fitting a model with `chains > 1`. For every scalar parameter, and
#'   optionally for selected time points of the latent state trajectories, the
#'   function reports the rank-normalized split-\eqn{\hat{R}} of Vehtari et al.
#'   (2021) together with the bulk and tail effective sample sizes, and combines
#'   them into a single `Overall` verdict per parameter.
#'
#'   The single-chain counterpart, which reports ESS, Geweke and
#'   Heidelberger–Welch instead, is documented in
#'   \code{\link{mcmc_convergence}}.
#'
#' @param object An object of class `"pdm_mcmc_list"`, obtained by calling one
#'   of the `mcmc_*()` samplers with `chains > 1`.
#' @param theta_timepoints Numeric vector of fractions in \eqn{(0, 1)} that
#'   determine which time points of the latent state chains are included. Each
#'   fraction is rounded to the nearest integer index. The default
#'   `c(0.25, 0.5, 0.75)` evaluates the states at the first quartile, median and
#'   third quartile of the series. Set to `NULL` to exclude all latent states.
#' @param rhat_threshold Numeric > 1, the \eqn{\hat{R}} value above which a
#'   parameter is flagged. Default `1.01`, the cut-off recommended by Vehtari et
#'   al. (2021).
#' @param ess_threshold Numeric > 0, the effective sample size below which a
#'   parameter is flagged. Applied to both the bulk and the tail ESS. Default
#'   `400`, which corresponds to the customary 100 draws per chain for four
#'   chains.
#' @param show_ess Logical. Whether to include the `ESS_bulk` and `ESS_tail`
#'   columns in the output table. Default `TRUE`.
#' @param show_overall Logical. Whether to include the `Overall` column.
#'   Default `TRUE`.
#' @param ... Additional arguments (currently unused).
#'
#' @return An object of class `"pdm_convergence_multi"`, a list with:
#'   \describe{
#'     \item{\code{table}}{Data frame with one row per assessed parameter or
#'       state time point. Columns always present: `Parameter`, `Rhat`.
#'       Optional columns, controlled by the `show_*` arguments: `ESS_bulk`,
#'       `ESS_tail`, `Overall`.}
#'     \item{\code{chains}}{Number of chains.}
#'     \item{\code{n_chain}}{Number of retained samples per chain (\eqn{N}).}
#'     \item{\code{model_type}}{Character string, e.g. `"locallevel"`.}
#'     \item{\code{rhat_threshold}}{The \eqn{\hat{R}} cut-off used.}
#'     \item{\code{ess_threshold}}{The ESS cut-off used.}
#'   }
#'
#' @details
#' \subsection{Why several chains are needed}{
#'   \eqn{\hat{R}} compares the variance between chains with the variance within
#'   them: if the chains have converged to the same distribution the two agree
#'   and the ratio approaches 1. This only has power to detect a problem when
#'   the chains start from dispersed points, so that failing to mix leaves a
#'   visible gap between them. The `mcmc_*()` samplers draw their starting
#'   values (the initial state and every precision) from the priors before
#'   entering the Gibbs loop, so each chain begins somewhere different and the
#'   comparison is informative.
#' }
#'
#' \subsection{Rank-normalized split-\eqn{\hat{R}}}{
#'   Three refinements over the original Gelman–Rubin statistic are applied,
#'   following Vehtari et al. (2021). Each chain is first \emph{split} in half
#'   and the halves treated as separate chains, which exposes a chain that is
#'   still drifting — a trend the original statistic cannot see, since it only
#'   compares chains to one another. The draws are then \emph{rank-normalized},
#'   making the statistic robust to heavy tails such as those of the precision
#'   parameters, whose variance-based statistic would otherwise be dominated by
#'   a few extreme draws. Finally the result is the maximum of the statistic
#'   computed on the rank-normalized draws, which detects disagreement in
#'   location, and on the rank-normalized absolute deviations from the median
#'   (the \emph{folded} version), which detects disagreement in scale.
#' }
#'
#' \subsection{Bulk and tail effective sample size}{
#'   `ESS_bulk` is the effective sample size of the rank-normalized split
#'   chains and describes how well the centre of the distribution — and hence
#'   the posterior mean and median — is resolved. `ESS_tail` is the smaller of
#'   the effective sample sizes of the indicators that a draw falls below the
#'   5% and the 95% quantile, and describes how well the extremes are resolved.
#'   The distinction matters: a parameter can have a comfortable bulk ESS and
#'   still carry too few effective draws in the tails to place its own credible
#'   interval reliably.
#' }
#'
#' \subsection{Overall classification}{
#'   `GOOD` when \eqn{\hat{R}} is below `rhat_threshold` and both effective
#'   sample sizes are above `ess_threshold`; `LOW ESS` when \eqn{\hat{R}}
#'   passes but an ESS does not; `POOR` when \eqn{\hat{R}} itself exceeds the
#'   threshold. A parameter that is constant across all draws yields an
#'   undefined \eqn{\hat{R}} and is reported as `UNDEFINED`.
#' }
#'
#' @references
#' Gelman, A., & Rubin, D. B. (1992). Inference from iterative simulation using
#'   multiple sequences. \emph{Statistical Science}, \strong{7}(4), 457--472.
#'
#' Geyer, C. J. (1992). Practical Markov chain Monte Carlo.
#'   \emph{Statistical Science}, \strong{7}(4), 473--483.
#'   \doi{10.1214/ss/1177011137}
#'
#' Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021).
#'   Rank-normalization, folding, and localization: An improved \eqn{\hat{R}}
#'   for assessing convergence of MCMC (with discussion).
#'   \emph{Bayesian Analysis}, \strong{16}(2), 667--718.
#'
#' @examples
#' \donttest{
#' ## 1. Simulate data from a local-level dynamic model --------------------
#' set.seed(123)
#' n <- 200
#' theta1_true <- cumsum(c(10, rnorm(n)))[-1]
#' y <- theta1_true + rnorm(n, sd = sqrt(1 / 5))
#'
#' ## 2. Fit four chains ----------------------------------------------------
#' fits <- mcmc_normal_locallevel(
#'   y,
#'   burnin             = 1000,
#'   thinning           = 5,
#'   n_chain            = 1000,
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
#' ## 3. Multi-chain diagnostics -------------------------------------------
#' conv <- mcmc_convergence(fits)
#' print(conv)
#'
#' ## 4. Scalar parameters only --------------------------------------------
#' print(mcmc_convergence(fits, theta_timepoints = NULL))
#'
#' ## 5. Stricter cut-offs --------------------------------------------------
#' print(mcmc_convergence(fits, rhat_threshold = 1.005, ess_threshold = 1000))
#'
#' # Post-process: flag anything that did not pass
#' conv$table[conv$table$Overall != "GOOD", ]
#'
#' ## 6. Single-chain diagnostics still apply to one chain -------------------
#' print(mcmc_convergence(fits[[1]]))
#' }
#'
#' @seealso \code{\link{mcmc_convergence}} for the single-chain diagnostics;
#'   \code{\link{print.pdm_mcmc_list}} for the multi-chain fit object;
#'   \code{\link{pdm}} for the catalog of samplers, all of which accept
#'   `chains`.
#'
#' @export
mcmc_convergence.pdm_mcmc_list <- function(object,
                                           theta_timepoints = c(0.25, 0.5, 0.75),
                                           rhat_threshold   = 1.01,
                                           ess_threshold    = 400,
                                           show_ess         = TRUE,
                                           show_overall     = TRUE,
                                           ...) {

  # --- Input validation ---------------------------------------------------
  if (!inherits(object, "pdm_mcmc_list")) {
    stop("'object' must inherit from 'pdm_mcmc_list'")
  }

  if (!is.null(theta_timepoints)) {
    if (!is.numeric(theta_timepoints) ||
        any(theta_timepoints <= 0) || any(theta_timepoints >= 1)) {
      stop("'theta_timepoints' must be a numeric vector with values in (0, 1), or NULL")
    }
    theta_timepoints <- sort(unique(theta_timepoints))
  }

  if (!is.numeric(rhat_threshold) || length(rhat_threshold) != 1L ||
      rhat_threshold <= 1) {
    stop("'rhat_threshold' must be a single numeric value greater than 1")
  }

  if (!is.numeric(ess_threshold) || length(ess_threshold) != 1L ||
      ess_threshold <= 0) {
    stop("'ess_threshold' must be a single positive numeric value")
  }

  for (flag in list(show_ess, show_overall)) {
    if (!is.logical(flag) || length(flag) != 1L) {
      stop("'show_*' arguments must be single logical values")
    }
  }
  # ------------------------------------------------------------------------

  n_chain    <- as.integer(attr(object, "n_chain"))
  n_chains   <- as.integer(attr(object, "chains"))
  model_type <- attr(object, "model_type")

  # Helper: diagnostics row for one n_chain x n_chains matrix of draws
  compute_row <- function(draws, label) {

    rhat <- rhat_rank_normalized(draws)
    bulk <- ess_bulk(draws)
    tail <- ess_tail(draws)

    row <- data.frame(
      Parameter = label,
      Rhat      = round(rhat, 4),
      stringsAsFactors = FALSE
    )

    if (show_ess) {
      row$ESS_bulk <- round(bulk, 1)
      row$ESS_tail <- round(tail, 1)
    }

    if (show_overall) {
      row$Overall <-
        if (is.na(rhat))                       "UNDEFINED" else
        if (rhat > rhat_threshold)             "POOR"      else
        if (is.na(bulk) || is.na(tail) ||
            min(bulk, tail) < ess_threshold)   "LOW ESS"   else "GOOD"
    }

    row
  }

  # --- Scalar parameters --------------------------------------------------
  # The configuration is resolved once per chain; the chains are validated to
  # share their components, so the names of the first one drive the loop.
  configs     <- lapply(object, get_param_config)
  param_names <- names(configs[[1L]])

  rows_scalar <- lapply(param_names, function(nm) {
    draws <- vapply(configs, function(cfg) cfg[[nm]]$samples, numeric(n_chain))
    dim(draws) <- c(n_chain, n_chains)
    compute_row(draws, configs[[1L]][[nm]]$name_str)
  })

  # --- Latent state time points -------------------------------------------
  rows_theta <- list()

  if (!is.null(theta_timepoints)) {

    # Trajectory matrices are named theta_1, theta_2, ... (no leading zero) and
    # stored as n_chain x n_obs. The initial-state scalars theta_01, theta_02
    # are vectors already covered above and must not match here.
    state_names <- grep("^theta_[1-9][0-9]*$", names(object[[1L]]), value = TRUE)
    state_names <- state_names[vapply(state_names,
                                      function(nm) is.matrix(object[[1L]][[nm]]),
                                      logical(1L))]
    state_names <- state_names[order(as.integer(sub("theta_", "", state_names)))]

    for (sname in state_names) {
      n_t          <- ncol(object[[1L]][[sname]])
      time_indices <- pmax(1L, pmin(n_t, round(theta_timepoints * n_t)))
      time_indices <- sort(unique(time_indices))
      j            <- sub("theta_", "", sname)

      for (tidx in time_indices) {
        draws <- vapply(object, function(ch) ch[[sname]][, tidx],
                        numeric(n_chain))
        dim(draws) <- c(n_chain, n_chains)
        rows_theta[[length(rows_theta) + 1L]] <-
          compute_row(draws, sprintf("theta_%s[t=%d]", j, tidx))
      }
    }
  }

  # --- Combine ------------------------------------------------------------
  table <- do.call(rbind, c(rows_scalar, rows_theta))
  rownames(table) <- NULL

  result <- list(
    table          = table,
    chains         = n_chains,
    n_chain        = n_chain,
    model_type     = model_type,
    rhat_threshold = rhat_threshold,
    ess_threshold  = ess_threshold
  )
  class(result) <- "pdm_convergence_multi"
  result
}


#' Print method for pdm_convergence_multi objects
#'
#' @param x An object of class `"pdm_convergence_multi"`.
#' @param digits Integer, significant digits for the \eqn{\hat{R}} column.
#'   Default 4.
#' @param ... Additional arguments (currently unused).
#'
#' @return Invisibly returns `x`.
#'
#' @seealso \code{\link{mcmc_convergence.pdm_mcmc_list}}.
#'
#' @export
print.pdm_convergence_multi <- function(x, digits = 4L, ...) {

  cat("\n")
  cat("MCMC Convergence Diagnostics (multi-chain)\n")
  cat(strrep("=", 70), "\n\n", sep = "")
  cat("Model:         ", x$model_type, "\n", sep = "")
  cat("Chains:        ", x$chains, "\n", sep = "")
  cat("Samples/chain: ", x$n_chain, "\n", sep = "")
  cat("\n")

  df <- x$table
  if ("Rhat" %in% names(df)) {
    df$Rhat <- sprintf(paste0("%.", digits, "f"), df$Rhat)
  }
  for (col in intersect(c("ESS_bulk", "ESS_tail"), names(df))) {
    df[[col]] <- sprintf("%.1f", df[[col]])
  }

  print(df, row.names = FALSE, right = TRUE)

  cat("\n")
  cat(sprintf("Overall: GOOD if Rhat < %g and both ESS > %g; LOW ESS if only the\n",
              x$rhat_threshold, x$ess_threshold))
  cat("ESS criterion fails; POOR if Rhat exceeds the threshold.\n")
  cat(strrep("-", 70), "\n\n", sep = "")

  invisible(x)
}
