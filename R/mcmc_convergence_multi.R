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
#' @param theta_timepoints Which time points of the latent state chains to
#'   include. Either **a count** — a single whole number, meaning that many
#'   evenly spaced points, the default `20` — or **an explicit numeric vector**
#'   of fractions in \eqn{(0, 1)} for an irregular grid. This is the convention
#'   `hist()` uses for `breaks`, and the two forms cannot be confused because
#'   fractions lie strictly inside \eqn{(0, 1)} while a count is at least 1.
#'   Each fraction is rounded to the nearest integer index; `NULL` excludes the
#'   latent states entirely.
#'
#'   A count expands to `seq(0.05, 0.95, length.out = n)`, which spans the
#'   series without touching either endpoint, where a state is pinned by its
#'   own prior rather than by the data. The same default and the same expansion
#'   are used by the screen behind \code{\link{summary.pdm_mcmc_list}}, so the
#'   two cannot disagree about one fit. See "How many time points" below for
#'   why twenty.
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
#' @param show_rhat_variants Logical. Whether to add the `Rhat_split` and
#'   `Rhat_classic` columns, the two earlier statistics that also travel under
#'   the name "R-hat". Default `FALSE`. These are for **reconciliation** — for
#'   matching a number reported by another implementation or in a paper — and
#'   not for deciding convergence: `Rhat` always holds the Vehtari et al.
#'   (2021) statistic whatever this is set to, and `Overall` is classified from
#'   that column alone. See "The three R-hat statistics" below.
#' @param warmup Integer >= 0, how many draws to discard from the start of every
#'   chain before diagnosing. Default `0`, which diagnoses everything stored and
#'   is what earlier versions did.
#'
#'   **This method only.** The single-chain method
#'   (\code{\link{mcmc_convergence}}) does not take it and warns if it is
#'   passed, rather than letting the generic's `...` swallow it in silence.
#'
#'   Prefer this to truncating the fit yourself. The draws are reached one
#'   parameter at a time, so discarding here costs nothing, whereas building a
#'   truncated copy of a `pdm_mcmc_list` duplicates every retained draw — for a
#'   long unthinned run, gigabytes. The diagnostics are computed on exactly the
#'   draws such a copy would have carried.
#'
#'   The returned object reports `n_draws` after the discard, and `warmup`
#'   alongside it.
#' @param ... Additional arguments (currently unused).
#'
#' @return An object of class `"pdm_convergence_multi"`, a list with:
#'   \describe{
#'     \item{\code{table}}{Data frame with one row per assessed parameter or
#'       state time point. Columns always present: `Parameter`, `Rhat`.
#'       Numeric columns are stored **unrounded** — the print method formats to
#'       four decimals for display, so a value read from here agrees exactly
#'       with the one \code{\link{summary.pdm_mcmc_list}} reports, rather than
#'       to the fourth decimal. Optional columns, controlled by the `show_*` arguments: `ESS_bulk`,
#'       `ESS_tail`, `Overall`.}
#'     \item{\code{chains}}{Number of chains.}
#'     \item{\code{n_draws}}{Number of samples per chain the diagnostics saw
#'       (\eqn{N}), that is the stored count less `warmup`.}
#'     \item{\code{warmup}}{Draws discarded from the start of each chain.}
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
#' \subsection{The three R-hat statistics}{
#'   Three statistics circulate under one name, and this package, other
#'   implementations and the applied literature do not all report the same one.
#'   `show_rhat_variants = TRUE` puts them side by side:
#'
#'   \tabular{lll}{
#'     \strong{Column} \tab \strong{Statistic} \tab \strong{Source} \cr
#'     `Rhat_classic` \tab no split, no ranks \tab Gelman & Rubin (1992) \cr
#'     `Rhat_split` \tab split, no ranks \tab BDA3 (2013) \cr
#'     `Rhat` \tab split, rank-normalized, folded \tab Vehtari et al. (2021)
#'   }
#'
#'   They are not interchangeable, and the gap between them is not a rounding
#'   detail. Measured on a two-component mixture with a logit weight
#'   (\eqn{n = 800}, four chains), taking the largest value over all 800 points
#'   of the weight trajectory:
#'
#'   \tabular{lrr}{
#'     \strong{Statistic} \tab \strong{Largest} \tab \strong{Points above 1.1} \cr
#'     classic \tab 1.073 \tab 0% \cr
#'     split \tab 1.100 \tab 0% \cr
#'     rank-normalized, folded \tab 1.153 \tab 3.7%
#'   }
#'
#'   Splitting contributes 0.027 and rank-normalization with folding a further
#'   0.053, and the verdict changes only at the third: the first two clear the
#'   conventional 1.1 at every one of the 800 points. That is why `Rhat` is the
#'   third and stays the third — the earlier statistics are, on this evidence,
#'   the more forgiving of the three, and a table in which the reported value
#'   could quietly be one of them would invite reporting whichever passes.
#'
#'   Use the extra columns to explain a discrepancy, not to resolve one.
#' }
#'
#' \subsection{How many time points}{
#'   A latent trajectory is a whole vector of parameters, and a run that has
#'   failed to mix rarely fails everywhere: the disagreement is usually confined
#'   to a stretch of the series, so a screen that samples too few points can
#'   step over it entirely. Measured on a two-component mixture with a logit
#'   weight (\eqn{n = 800}, four chains), on one fit, taking the largest
#'   \eqn{\hat{R}} over the screened points:
#'
#'   \tabular{lr}{
#'     \strong{Points screened} \tab \strong{Largest \eqn{\hat{R}} found} \cr
#'     3   \tab 1.037 \cr
#'     5   \tab 1.078 \cr
#'     10  \tab 1.063 \cr
#'     20  \tab 1.148 \cr
#'     50  \tab 1.149 \cr
#'     100 \tab 1.152 \cr
#'     all 800 \tab 1.153
#'   }
#'
#'   Three points report a value that clears every threshold in common use,
#'   on a fit whose true maximum is 1.153; twenty recover 97% of it. The grids
#'   are not nested, which is why the sequence is not monotone. Twenty is
#'   therefore the default, matching the screen behind
#'   \code{\link{summary.pdm_mcmc_list}} so that the two never disagree. A
#'   denser grid costs little — the whole screen
#'   is a fraction of a second on a fit that takes seconds to minutes — but
#'   twenty is where the return flattens.
#' }
#'
#' \subsection{Link families: the states already cover \code{alpha}}{
#'   For the families that carry a link (the mixture weight, the binomial and
#'   probit probabilities, the Poisson rate), \code{alpha} has no row of its own
#'   and needs none. \eqn{\hat{R}} and both effective sample sizes here are
#'   computed from the \emph{ranks} of the draws, and a link is a strictly
#'   monotone transformation, so \eqn{\alpha_t = g(\theta_{t1})} has exactly the
#'   same ranks as \eqn{\theta_{t1}}: the `theta_1[t=...]` rows \emph{are} the
#'   diagnostics for `alpha`, to the last bit.
#'
#'   This holds only for the rank-based statistics reported here. The original
#'   Gelman–Rubin statistic is built from means and variances and is not
#'   invariant, so a diagnostic computed on `alpha` by that route is a genuinely
#'   different number and will not match this table.
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
#'   \doi{10.1214/ss/1177011136}
#'
#' Geyer, C. J. (1992). Practical Markov chain Monte Carlo.
#'   \emph{Statistical Science}, \strong{7}(4), 473--483.
#'   \doi{10.1214/ss/1177011137}
#'
#' Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021).
#'   Rank-normalization, folding, and localization: An improved \eqn{\hat{R}}
#'   for assessing convergence of MCMC (with discussion).
#'   \emph{Bayesian Analysis}, \strong{16}(2), 667--718.
#'   \doi{10.1214/20-BA1221}
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
                                           theta_timepoints = 20L,
                                           rhat_threshold   = 1.01,
                                           ess_threshold    = 400,
                                           show_ess         = TRUE,
                                           show_overall     = TRUE,
                                           show_rhat_variants = FALSE,
                                           warmup           = 0L,
                                           ...) {

  # --- Input validation ---------------------------------------------------
  if (!inherits(object, "pdm_mcmc_list")) {
    stop("'object' must inherit from 'pdm_mcmc_list'")
  }

  theta_timepoints <- resolve_timepoints(theta_timepoints, "theta_timepoints")

  if (!is.numeric(rhat_threshold) || length(rhat_threshold) != 1L ||
      rhat_threshold <= 1) {
    stop("'rhat_threshold' must be a single numeric value greater than 1")
  }

  if (!is.numeric(ess_threshold) || length(ess_threshold) != 1L ||
      ess_threshold <= 0) {
    stop("'ess_threshold' must be a single positive numeric value")
  }

  for (flag in list(show_ess, show_overall, show_rhat_variants)) {
    if (!is.logical(flag) || length(flag) != 1L) {
      stop("'show_*' arguments must be single logical values")
    }
  }
  # ------------------------------------------------------------------------

  n_stored   <- as.integer(attr(object, "n_draws"))
  n_chains   <- as.integer(attr(object, "chains"))
  model_type <- attr(object, "model_type")

  if (!is.numeric(warmup) || length(warmup) != 1L || is.na(warmup) ||
      warmup < 0 || warmup != as.integer(warmup)) {
    stop("'warmup' must be a single non-negative whole number")
  }
  warmup <- as.integer(warmup)
  if (warmup >= n_stored) {
    stop("'warmup' (", warmup, ") leaves no draws to diagnose; the chains hold ",
         n_stored, ".")
  }

  # Discarding here rather than in the caller is the whole point of the
  # argument. Every draw is reached one parameter at a time, as an
  # `n_draws x n_chains` matrix, so subsetting inside those extractions costs
  # nothing; a caller that instead truncates the fit first pays for a second
  # copy of every retained draw, which on a long run is gigabytes. The
  # diagnostics are computed on exactly the draws a truncated copy would have
  # carried, so the two routes agree to the bit.
  keep    <- if (warmup > 0L) seq.int(warmup + 1L, n_stored) else NULL
  n_draws <- n_stored - warmup

  # Helper: diagnostics row for one n_draws x n_chains matrix of draws
  compute_row <- function(draws, label) {

    rhat <- rhat_rank_normalized(draws)
    bulk <- ess_bulk(draws)
    tail <- ess_tail(draws)

    # Stored unrounded. `print.pdm_convergence_multi()` formats to four decimals
    # for display, so nothing about the printed table changes -- but the object
    # is what a script reads, and it used to disagree with `summary()$rhat` in
    # the fifth decimal for no reason other than storage. Rounding is a
    # presentation decision and belongs in the print method alone.
    row <- data.frame(
      Parameter = label,
      Rhat      = rhat,
      stringsAsFactors = FALSE
    )

    # Reconciliation columns only. `Rhat` keeps its meaning -- Vehtari et al.
    # (2021) -- whatever these are set to, and `Overall` below is classified
    # from it alone. That is deliberate: the earlier statistics are almost
    # always the milder of the three, and a table where the reported value
    # could silently be one of them invites picking whichever clears the
    # threshold.
    if (show_rhat_variants) {
      variants         <- rhat_variants(draws)
      row$Rhat_split   <- variants[["split"]]
      row$Rhat_classic <- variants[["classic"]]
    }

    if (show_ess) {
      row$ESS_bulk <- bulk
      row$ESS_tail <- tail
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
    compute_row(scalar_draws(configs, nm, n_draws, keep),
                configs[[1L]][[nm]]$name_str)
  })

  # --- Latent state time points -------------------------------------------
  rows_theta <- list()

  if (!is.null(theta_timepoints)) {

    # Trajectory matrices are named theta_1, theta_2, ... (no leading zero) and
    # stored as n_draws x n_obs. The initial-state scalars theta_01, theta_02
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
        draws <- if (is.null(keep)) {
          vapply(object, function(ch) ch[[sname]][, tidx], numeric(n_draws))
        } else {
          vapply(object, function(ch) ch[[sname]][keep, tidx], numeric(n_draws))
        }
        dim(draws) <- c(n_draws, n_chains)
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
    # The draws the diagnostics actually saw, not the number stored. `warmup`
    # carries the difference, so a table can say which it is instead of leaving
    # the reader to infer it.
    n_draws        = n_draws,
    warmup         = warmup,
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
#' @param n_worst Integer, how many latent-state rows to display. The state
#'   screen covers twenty time points per state block by default, which is too
#'   many to read as a table, so only the `n_worst` with the largest
#'   \eqn{\hat{R}} are shown, followed by a count of how many exceeded the
#'   threshold. Default 3; use `Inf` to print every row. The scalar parameters
#'   are always shown in full, and the complete table is always available
#'   unabridged in `x$table`.
#' @param ... Additional arguments (currently unused).
#'
#' @return Invisibly returns `x`.
#'
#' @seealso \code{\link{mcmc_convergence.pdm_mcmc_list}}.
#'
#' @export
print.pdm_convergence_multi <- function(x, digits = 4L, n_worst = 3L, ...) {

  cat("\n")
  cat("MCMC Convergence Diagnostics (multi-chain)\n")
  cat(strrep("=", 70), "\n\n", sep = "")
  cat("Model:         ", x$model_type, "\n", sep = "")
  cat("Chains:        ", x$chains, "\n", sep = "")
  cat("Draws/chain:   ", x$n_draws, "\n", sep = "")
  cat("\n")

  # The state screen is dense on purpose (see the "How many time points"
  # section of ?mcmc_convergence.pdm_mcmc_list), which makes it unreadable as a
  # flat table: three state blocks at twenty points each is sixty rows. Scalars
  # print in full; the states collapse to the worst few plus a count.
  is_state <- grepl("^theta_[0-9]+\\[", x$table$Parameter)
  scalars  <- x$table[!is_state, , drop = FALSE]
  states   <- x$table[ is_state, , drop = FALSE]

  fmt <- function(df) {
    for (col in intersect(c("Rhat", "Rhat_split", "Rhat_classic"), names(df))) {
      df[[col]] <- sprintf(paste0("%.", digits, "f"), df[[col]])
    }
    for (col in intersect(c("ESS_bulk", "ESS_tail"), names(df))) {
      df[[col]] <- sprintf("%.1f", df[[col]])
    }
    df
  }

  if (nrow(scalars)) print(fmt(scalars), row.names = FALSE, right = TRUE)

  if (nrow(states)) {
    # One block is "theta_1[t=...]"; count the distinct blocks to report the
    # per-block density rather than the raw row count, which is what the user
    # chose through `theta_timepoints`.
    block    <- sub("\\[.*$", "", states$Parameter)
    n_block  <- length(unique(block))
    per_blk  <- nrow(states) / n_block
    n_flag   <- sum(states$Rhat > x$rhat_threshold, na.rm = TRUE)
    n_show   <- min(nrow(states), n_worst)

    cat("\n")
    cat(sprintf("Latent states - %g time point%s screened per block, %d block%s (%d rows)\n",
                per_blk, if (per_blk == 1) "" else "s",
                n_block, if (n_block == 1) "" else "s", nrow(states)))

    if (n_show < nrow(states)) {
      ord <- order(states$Rhat, decreasing = TRUE, na.last = TRUE)
      cat(sprintf("Worst %d of %d:\n", n_show, nrow(states)))
      print(fmt(states[ord[seq_len(n_show)], , drop = FALSE]),
            row.names = FALSE, right = TRUE)
    } else {
      print(fmt(states), row.names = FALSE, right = TRUE)
    }

    cat(sprintf("  %d of %d above Rhat %g.", n_flag, nrow(states), x$rhat_threshold))
    if (n_show < nrow(states)) cat("  Full table in $table.")
    cat("\n")
  }

  cat("\n")
  cat(sprintf("Overall: GOOD if Rhat < %g and both ESS > %g; LOW ESS if only the\n",
              x$rhat_threshold, x$ess_threshold))
  cat("ESS criterion fails; POOR if Rhat exceeds the threshold.\n")
  cat(strrep("-", 70), "\n\n", sep = "")

  invisible(x)
}
